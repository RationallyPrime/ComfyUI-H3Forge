"""Disagreement between actual window writes, and local flow-Euler retries."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import blake2b
from itertools import combinations

import torch
import torch.nn.functional as F

SEAM_STATE = "h3forge_seam_repair"


@dataclass
class SeamState:
    """One sampling run. Comfy copies the options dictionaries between calls."""
    video_shape: torch.Size | None = None
    scores: dict[tuple[int, int], float] = field(default_factory=dict)
    row_ratio: torch.Tensor | None = None
    collecting: bool = False

    def video(self, packed):
        # H3 packs the whole video in channel-major order, followed by audio.
        return packed[..., :self.video_shape.numel()].view(self.video_shape)


class OverlapScores:
    """Keep only the first prediction on each shared write until its peer runs."""

    def __init__(self, writes):
        self.pairs = []
        for (i, left), (j, right) in combinations(enumerate(writes), 2):
            for a, b in left:
                for c, d in right:
                    lo, hi = max(a, c), min(b, d)
                    if lo < hi:
                        self.pairs.append((i, j, lo, hi))
        self.values = {}

    def observe(self, index, start, prediction):
        for pair in self.pairs:
            left, right, lo, hi = pair
            if index == left or index == right:
                rows = prediction[:, :, lo - start:hi - start].float()
                if index == left:
                    self.values[pair] = rows.clone()
                else:
                    self.values[pair] = torch.linalg.vector_norm(self.values[pair] - rows)

    def publish(self, fused, state):
        scores = []
        for pair, difference in self.values.items():
            lo, hi = pair[2:]
            norm = torch.linalg.vector_norm(fused[:, :, lo:hi].float())
            # Exact cancellation with differing predictions is maximal disagreement.
            scores.append(torch.where(difference == 0, 0., difference / norm))
        if scores:
            for pair, score in zip(self.values, torch.stack(scores).tolist()):
                span = pair[2:]
                state.scores[span] = max(score, state.scores.get(span, 0.0))


def _fresh_noise(video, seed, step, retry):
    # Domain-separate from Comfy's initial video noise, including step 0 / retry 0.
    key = f"{seed}/{step}/{retry}".encode("ascii")
    digest = blake2b(key, person=b"H3Forge repair", digest_size=8).digest()
    generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest, "little"))
    return torch.randn(video.shape, generator=generator, dtype=torch.float32, device="cpu").to(video)


def _renoise_rows(advanced, x0, noise, sigma, following, weight, fresh):
    noisy = (1 - sigma) * x0 + sigma * noise
    if not fresh:
        # The original noise is correlated with the residual already in the row.
        return torch.lerp(advanced, noisy, weight)

    # Blend independent residuals in quadrature. A plain lerp halves their
    # variance near the middle of a ramp when adjacent sigmas are close.
    old = (1 - weight) * following
    new = weight * sigma
    row_sigma = following + weight * (sigma - following)
    residual = (1 - weight) * (advanced - (1 - following) * x0) + new * noise
    norm = torch.hypot(old, new).clamp_min(torch.finfo(advanced.dtype).tiny)
    return (1 - row_sigma) * x0 + residual * (row_sigma / norm)


@torch.no_grad()
def sample_seam_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, *,
                      repairs=0, repair_threshold=0.15, repair_until=0.4, repair_ramp=4,
                      repair_budget=0.3, repair_noise="fresh"):
    from comfy.k_diffusion.sampling import sample_euler, to_d, trange

    from .renoise import temporal_weights

    args = extra_args or {}
    state = args.get("model_options", {}).get("transformer_options", {}).get(SEAM_STATE)
    steps = len(sigmas) - 1
    budget = int(steps * repair_budget)
    print(f"[H3Forge] Seam Euler repairs={repairs} threshold={repair_threshold} until={repair_until} "
          f"ramp={repair_ramp} noise={repair_noise} context={'on' if state else 'off'} "
          f"extra_call_limit={budget}/{steps}", flush=True)
    if not repairs or not budget or state is None:
        return sample_euler(model, x, sigmas, extra_args, callback, disable)

    def evaluate(latent, sigma, ratio=None):
        state.scores.clear()
        state.collecting, state.row_ratio = True, ratio
        try:
            return model(latent, sigma * s_in, **args)
        finally:
            state.collecting, state.row_ratio = False, None

    s_in = x.new_ones([x.shape[0]])
    extra_calls = 0
    protected = None
    for i in trange(steps, disable=disable):
        sigma, following = sigmas[i:i + 2]
        denoised = evaluate(x, sigma)
        x_next = x + to_d(x, sigma, denoised) * (following - sigma)
        for retry in range(repairs):
            if extra_calls >= budget or sigma <= repair_until:
                break
            before = {span: d for span, d in state.scores.items() if d > repair_threshold}
            if not before:
                break

            next_video = state.video(x_next)
            weights = torch.stack([temporal_weights(next_video.shape[2], a, b - 1, repair_ramp)
                                   for a, b in before]).amax(dim=0).to(next_video).view(1, 1, -1, 1, 1)
            active_t = weights.flatten().nonzero().flatten()
            if args.get("denoise_mask") is not None and protected is None:
                # Preserve the ordinary inpaint trajectory on any H3 2x2 token
                # containing a partially or fully pinned pixel, including its label.
                mask = state.video(args["denoise_mask"])
                pooled = F.max_pool2d((mask.amin(1) < 1).float(), 2, ceil_mode=True)
                protected = pooled.repeat_interleave(2, -2).repeat_interleave(2, -1)
                protected = protected[..., :mask.shape[-2], :mask.shape[-1]].unsqueeze(1).bool()
            if protected is not None:
                weights = weights.masked_fill(protected, 0)
                if not weights.any():
                    break
            row_sigma = following + weights * (sigma - following)
            x0 = state.video(denoised)
            local_x0 = x0.index_select(2, active_t)
            noise = (state.video(model.noise).index_select(2, active_t) if repair_noise == "original" else
                     _fresh_noise(local_x0, int(args.get("seed") or 0), i, retry))
            # Audio stays at the base call's sigma in the retry input. Only video is committed.
            mixed = x.clone()
            trial = state.video(mixed)
            trial.copy_(next_video)
            renoised = _renoise_rows(next_video.index_select(2, active_t), local_x0, noise, sigma,
                                     following, weights.index_select(2, active_t), repair_noise == "fresh")
            trial.index_copy_(2, active_t, renoised)
            if protected is not None:
                trial.copy_(torch.where(protected, state.video(x), trial))
                row_sigma = torch.where(protected, sigma, row_sigma)
            retry_x0 = state.video(evaluate(mixed, sigma, row_sigma / sigma))
            # The diffusion wrapper scales the velocity by row_sigma / sigma before
            # Comfy converts it to x0. Advance each row only its own remaining distance.
            fraction = (weights * (sigma - following)) / row_sigma.clamp_min(torch.finfo(x.dtype).tiny)
            active = weights > 0
            next_video.copy_(torch.where(active, torch.lerp(trial, retry_x0, fraction), next_video))
            x0.copy_(torch.where(active, retry_x0, x0))
            extra_calls += 1
            changes = " ".join(f"[{a},{b})={d:.4g}->{state.scores[a, b]:.4g}" for (a, b), d in before.items())
            print(f"[H3Forge] step={i} repair={retry + 1} extra_calls={extra_calls}/{budget} d {changes}", flush=True)
        if callback is not None:
            callback(dict(x=x, i=i, sigma=sigma, sigma_hat=sigma, denoised=denoised))
        x = x_next
    return x

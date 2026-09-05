from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import torch

from .control import control_window
from .layout import audio_range_for_video_window, clone_window_layout, padded_spatial_shape
from .prompt import segment_ranges
from .state import resolve_step

LOG = "[H3Forge]"
_LATENT_CADENCE = 5


@dataclass(frozen=True)
class ContextPolicy:
    window_frames: int = 25
    overlap_frames: int = 8
    stagger: bool = True
    blend: str = "pyramid"
    strict: bool = False


def ordered_halving(value: int) -> float:
    """Return the bit-reversed base-2 fraction used for context staggering."""
    binary = f"{value:064b}"
    return int(binary[::-1], 2) / (1 << 64)


def max_stagger_phase(window: int, overlap: int) -> int:
    """Largest interior-boundary shift that still keeps every adjacent overlap.

    Staggering exists to move seams between steps, not to let a window jump
    across its neighbour: a shift larger than the requested overlap lets two
    windows abut with no blend and puts every latent under a different
    prompt/neighbour pair on alternate steps, which reads as constant morphing.
    """
    stride = window - overlap
    return max(0, min(overlap, stride - 1))


def stagger_phase(step: int, window: int, overlap: int) -> int:
    """Ordered-halving phase for ``step`` bounded to ``[0, max_stagger_phase]``."""
    return int(ordered_halving(step) * (max_stagger_phase(window, overlap) + 1))


def _shifted_starts(anchor: list[int], phase: int, stride: int, *, snap: bool) -> list[int]:
    """Greedy plan ``anchor`` shifted by ``phase``, keeping anchors, count and overlap.

    Every adjacent pair keeps at least ``overlap`` latents in common: the next
    start is never more than one stride away from the last one, and never so
    far back that the remaining windows could not reach the final anchor within
    a stride each. Because ``anchor`` itself satisfies those bounds, every
    interior start lands in ``[anchor[i], anchor[i] + phase]``: a phase never
    moves a seam further than itself, and a feasible ``anchor`` is a fixed point
    at phase 0. With ``snap`` each interior start moves to the nearest cadence
    point inside its feasible interval when one exists.
    """
    final_start = anchor[-1]
    count = len(anchor)
    starts = [0]
    for i in range(1, count - 1):
        remaining = count - 1 - i
        lower = max(starts[-1] + 1, final_start - remaining * stride)
        upper = min(starts[-1] + stride, final_start - remaining)
        candidate = min(max(anchor[i] + phase, lower), upper)
        if snap:
            first = ((lower + _LATENT_CADENCE - 1) // _LATENT_CADENCE) * _LATENT_CADENCE
            last = (upper // _LATENT_CADENCE) * _LATENT_CADENCE
            if first <= last:
                candidate = min(max(round(candidate / _LATENT_CADENCE) * _LATENT_CADENCE, first), last)
        starts.append(candidate)
    starts.append(final_start)
    return starts


def _stagger_layouts(anchor: list[int], stride: int, max_phase: int) -> int:
    """Number of distinct plans the phases ``0..max_phase`` reach from ``anchor``."""
    return len({tuple(_shifted_starts(anchor, phase, stride, snap=False)) for phase in range(max_phase + 1)})


def window_starts(total: int, window: int, overlap: int, phase: int = 0, max_phase: int = 0) -> list[int]:
    """Window starts for ``phase`` of a run whose stagger visits phases ``0..max_phase``.

    The phase-0 plan snaps interior starts to the latent cadence unless that
    would leave the stagger fewer distinct layouts than the even spread does;
    a static run (``max_phase == 0``) therefore always snaps, and a staggering
    run keeps its off-cadence spread only where cadence and seam movement
    collide. Every active phase is then derived from that same phase-0 plan,
    so no seam ever moves further than the phase.
    """
    if window >= total:
        return [0]
    if window < 2:
        raise ValueError("window must be >=2")
    stride = window - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than window")

    if not isinstance(phase, int) or not 0 <= phase < stride:
        raise ValueError(f"phase must be an integer in [0, {stride})")
    if not isinstance(max_phase, int) or not 0 <= max_phase < stride:
        raise ValueError(f"max_phase must be an integer in [0, {stride})")

    final_start = total - window
    count = 1 + (final_start + stride - 1) // stride
    base = [round(i * final_start / (count - 1)) for i in range(count)]
    if count <= 2:
        return base

    spread = _shifted_starts(base, 0, stride, snap=False)
    snapped = _shifted_starts(base, 0, stride, snap=True)
    anchor = snapped
    if _stagger_layouts(snapped, stride, max_phase) < _stagger_layouts(spread, stride, max_phase):
        anchor = spread
    return _shifted_starts(anchor, phase, stride, snap=False)


def blend_weights(length: int, overlap: int, *, device, dtype, mode="pyramid",
                  ramp_start: bool = True, ramp_end: bool = True):
    """Return full-window pyramid, overlap-linear, or flat fusion weights."""
    if mode not in {"pyramid", "overlap-linear", "flat"}:
        raise ValueError(f"unknown blend mode: {mode}")
    if length <= 0:
        return torch.empty(0, device=device, dtype=dtype)
    if mode == "flat":
        return torch.ones(length, device=device, dtype=dtype)

    if mode == "pyramid":
        peak = (length + 1) // 2
        ascending = torch.arange(1, peak + 1, device=device, dtype=torch.float32)
        descending_start = peak if length % 2 == 0 else peak - 1
        descending = torch.arange(descending_start, 0, -1, device=device, dtype=torch.float32)
        return torch.cat((ascending, descending)).to(dtype)

    if overlap <= 0 or not (ramp_start or ramp_end):
        return torch.ones(length, device=device, dtype=dtype)
    ramp = min(overlap, length // 2)
    if ramp == 0:
        return torch.ones(length, device=device, dtype=dtype)
    w = torch.ones(length, device=device, dtype=torch.float32)
    edge = torch.linspace(1.0 / (ramp + 1), 1.0, ramp, device=device)
    if ramp_start:
        w[:ramp] = edge
    if ramp_end:
        w[-ramp:] = edge.flip(0)
    return w.to(dtype)


def audio_overlap_frames(overlap_frames: int, video_len: int, audio_len: int) -> int:
    """Audio-latent overlap induced by the physical-time mapping of a video overlap.

    No video overlap means no audio overlap: the audio ramp must not be forced
    to one sample when the video windows do not overlap at all.
    """
    if overlap_frames <= 0:
        return 0
    return max(1, round(overlap_frames * max(audio_len / max(video_len, 1), 1.0)))


def assert_full_coverage(video_den: torch.Tensor, audio_den: torch.Tensor) -> None:
    """Strict-mode gate: every target element must carry positive blend weight."""
    if not bool((video_den > 0).all()):
        raise RuntimeError("context windows left video latents with zero accumulated blend weight")
    if not bool((audio_den > 0).all()):
        raise RuntimeError("context windows left audio latents with zero accumulated blend weight")


def context_plan_summary(
    total: int,
    starts: list[int],
    window: int,
    overlap: int,
    *,
    phase: int,
    blend: str = "pyramid",
    stagger: bool = False,
    assignments: list[int] | None = None,
    frame_cuts: list[int] | None = None,
    audio_t: int | None = None,
    max_phase: int | None = None,
) -> str:
    """Return one compact, truthful account of a context-window pass."""
    ranges = [(start, min(start + window, total)) for start in starts]
    latent_visits = sum(end - start for start, end in ranges) / max(total, 1)
    bits = [
        f"video_latents={total}",
        f"windows={len(ranges)}",
        f"window/overlap={window}/{overlap}",
        f"phase={phase}",
        f"video_latent_visits={latent_visits:.2f}x",
        f"stride={window - overlap}",
        f"min_overlap={min((window - (right - left) for left, right in pairwise(starts)), default=0)}",
        f"blend={blend}",
        f"stagger={'on' if stagger else 'off'}",
        f"cadence={_LATENT_CADENCE}",
        f"off_cadence_starts={sum(start % _LATENT_CADENCE != 0 for start in starts)}",
    ]
    if max_phase is not None:
        bits.append(f"max_phase={max_phase}")
    if assignments:
        runs = []
        for index in assignments:
            if runs and runs[-1][0] == index + 1:
                runs[-1][1] += 1
            else:
                runs.append([index + 1, 1])
        bits.append("prompt_windows=" + ",".join(f"{index}x{count}" for index, count in runs))
    if frame_cuts is not None:
        bits.append("prompt_frame_cuts=" + ",".join(map(str, frame_cuts)))
    if audio_t is not None:
        bits.append(f"audio_context=full:{audio_t}")
    return " ".join(bits)


def _slice_optional_video(mask, v0, v1):
    if mask is None:
        return None
    return mask[:, :, v0:v1]


def _segment_windows(total, window, overlap, ranges):
    """Give every beat forwards and exclusive output ownership on the native grid.

    Windows see neighboring video for continuity, but a beat only contributes
    predictions inside its own interval. There is no cross-prompt interpolation.
    """
    plan = []
    for index, (lo, hi) in enumerate(ranges):
        start, stop = max(0, lo - overlap), min(total, hi + overlap)
        if stop - start < window:
            start = max(0, min((lo + hi - window) // 2, total - window))
            stop = start + window
        for offset in window_starts(stop - start, window, overlap):
            v0, v1 = start + offset, start + offset + window
            if max(v0, lo) < min(v1, hi):
                plan.append((index, v0, v1, max(v0, lo), min(v1, hi)))
    return plan


def make_context_wrapper(policy: ContextPolicy):
    """Window video while every forward sees the complete shared audio timeline."""
    def wrapper(executor, x, timestep, context, transformer_options, **kwargs):
        video_x, audio_x = x
        total_t, audio_t = int(video_x.shape[2]), int(audio_x.shape[-1])
        payload = kwargs.get("minimax_payload") or {}
        prompts = payload.get("h3forge_prompt_segments") or (context,)
        tags = payload.get("h3forge_prompt_segment_tags")
        segmented = len(prompts) > 1
        step, _ = resolve_step(transformer_options)
        window = min(total_t, policy.window_frames)
        overlap = min(policy.overlap_frames, window - 1) if total_t > window else 0
        if total_t <= window and not segmented:
            if step == 0:
                print(f"{LOG} context plan " + context_plan_summary(total_t, [0], window, 0,
                      phase=0, blend=policy.blend, audio_t=audio_t), flush=True)
            return executor(x, timestep, context, transformer_options, **kwargs)

        model = executor.class_obj
        padded_h, padded_w = padded_spatial_shape(video_x.shape[3], video_x.shape[4], model.patch_size)
        full_layout = payload.get("layout")
        if full_layout is None:
            from comfy.ldm.minimax.model import PackedLayout
            full_layout = PackedLayout(max(c.shape[1] for c in prompts), total_t, padded_h, padded_w,
                                       audio_t, keyframes=payload.get("keyframes"), refs=payload.get("refs"))
        frame_cuts = None
        audio_cuts = [0, audio_t]
        phase = max_phase = 0
        stagger = policy.stagger and not segmented
        if segmented:
            ranges, frame_cuts = segment_ranges(total_t, len(prompts), payload.get("h3forge_prompt_segment_durations"))
            # H3 audio has 40 ticks per 24 decoded frames. Shared cuts give each
            # audio tick exactly one prompt owner, including a seam-crossing line.
            audio_cuts = [round(frame * 5 / 3) for frame in frame_cuts]
            audio_cuts[-1] = audio_t
            plan = _segment_windows(total_t, window, overlap, ranges)
        else:
            if stagger:
                max_phase = max_stagger_phase(window, overlap)
                phase = stagger_phase(step, window, overlap) if step is not None else 0
            starts = window_starts(total_t, window, overlap, phase, max_phase)
            plan = [(0, v0, v0 + window, v0, v0 + window) for v0 in starts]
        if step == 0:
            print(f"{LOG} context plan " + context_plan_summary(total_t, [p[1] for p in plan], window, overlap,
                  phase=phase, blend=policy.blend, stagger=stagger, max_phase=max_phase if stagger else None,
                  assignments=[p[0] for p in plan] if segmented else None, frame_cuts=frame_cuts, audio_t=audio_t),
                  flush=True)

        video_acc = torch.zeros_like(video_x, dtype=torch.float32)
        audio_acc = torch.zeros_like(audio_x, dtype=torch.float32)
        video_den = torch.zeros((1, 1, total_t, 1, 1), device=video_x.device, dtype=torch.float32)
        audio_den = torch.zeros((1, 1, 1, audio_t), device=audio_x.device, dtype=torch.float32)
        for index, v0, v1, write_v0, write_v1 in plan:
            local_context = prompts[index]
            local_layout = clone_window_layout(full_layout=full_layout, text_len=local_context.shape[1],
                video_shape=(v1 - v0, padded_h, padded_w), audio_t=audio_t,
                video_range=(v0, v1), audio_range=(0, audio_t),
                keyframes=payload.get("keyframes"), refs=payload.get("refs"))
            local_payload = {**payload, "layout": local_layout}
            if tags is not None:
                local_payload["text_token_tags"] = tags[index]
            local_kwargs = {**kwargs, "minimax_payload": local_payload,
                            "denoise_mask": _slice_optional_video(kwargs.get("denoise_mask"), v0, v1)}
            # The global audio input and denoise mask travel intact. Only the
            # output projection is local, so all windows can see prior utterances.
            local_x = [video_x[:, :, v0:v1], audio_x]
            sentinel = object()
            previous_layout = transformer_options.get("h3forge_active_layout", sentinel)
            transformer_options["h3forge_active_layout"] = local_layout
            try:
                with control_window(executor, video_x.shape, (v0, v1), timestep, transformer_options):
                    v_out, a_out = executor(local_x, timestep, local_context, transformer_options, **local_kwargs)
            except Exception as exc:
                raise RuntimeError(f"{LOG} window [{v0},{v1}) prompt {index + 1} failed: {exc}") from exc
            finally:
                if previous_layout is sentinel:
                    transformer_options.pop("h3forge_active_layout", None)
                else:
                    transformer_options["h3forge_active_layout"] = previous_layout

            vw = blend_weights(v1 - v0, overlap, device=v_out.device, dtype=torch.float32, mode=policy.blend,
                               ramp_start=v0 > 0, ramp_end=v1 < total_t).view(1, 1, -1, 1, 1)
            keep = slice(write_v0 - v0, write_v1 - v0)
            video_acc[:, :, write_v0:write_v1].add_(v_out[:, :, keep].float() * vw[:, :, keep])
            video_den[:, :, write_v0:write_v1].add_(vw[:, :, keep])

            a0, a1 = audio_range_for_video_window(full_layout, v0, v1)
            write_a0, write_a1 = max(a0, audio_cuts[index]), min(a1, audio_cuts[index + 1])
            aw = blend_weights(a1 - a0, audio_overlap_frames(overlap, v1 - v0, a1 - a0),
                               device=a_out.device, dtype=torch.float32, mode=policy.blend,
                               ramp_start=a0 > 0, ramp_end=a1 < audio_t).view(1, 1, 1, -1)
            aw = aw[..., write_a0 - a0:write_a1 - a0]
            audio_acc[..., write_a0:write_a1].add_(a_out[..., write_a0:write_a1].float() * aw)
            audio_den[..., write_a0:write_a1].add_(aw)

        if policy.strict:
            assert_full_coverage(video_den, audio_den)
        return [(video_acc / video_den.clamp_min(1e-6)).to(video_x.dtype),
                (audio_acc / audio_den.clamp_min(1e-6)).to(audio_x.dtype)]

    return wrapper

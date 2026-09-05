from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import torch

from .layout import audio_range_for_video_window, clone_window_layout, expand_audio_range, padded_spatial_shape
from .prompt import select_segment_index, unreachable_segments
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
    prompt_count: int = 0,
    prompt_durations=None,
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
    if prompt_count:
        assigned = [
            select_segment_index(start, end, total, prompt_count, prompt_durations) + 1
            for start, end in ranges
        ]
        runs = []
        for index in assigned:
            if runs and runs[-1][0] == index:
                runs[-1][1] += 1
            else:
                runs.append([index, 1])
        bits.append(
            "prompt_windows="
            + ",".join(f"{index}x{count}" for index, count in runs)
        )
    return " ".join(bits)


def _slice_optional_video(mask, v0, v1):
    if mask is None:
        return None
    return mask[:, :, v0:v1]


def _slice_optional_audio(mask, a0, a1):
    if mask is None:
        return None
    return mask[..., a0:a1]


def make_context_wrapper(policy: ContextPolicy):
    """Build a ComfyUI DIFFUSION_MODEL wrapper for synchronized overlap-add."""
    def wrapper(executor, x, timestep, context, transformer_options, **kwargs):
        video_x, audio_x = x[0], x[1]
        total_t = int(video_x.shape[2])
        payload = dict(kwargs.get("minimax_payload") or {})
        prompt_segments = payload.get("h3forge_prompt_segments")
        prompt_durations = payload.get("h3forge_prompt_segment_durations")
        step, _ = resolve_step(transformer_options)

        if total_t <= policy.window_frames:
            if prompt_segments and len(prompt_segments) > 1:
                message = (
                    f"{LOG} pipe prompt has {len(prompt_segments)} segments but the whole video fits "
                    f"one context window ({total_t} <= {policy.window_frames} latents); only segment 1 "
                    "is used — lengthen the video or shrink window_frames"
                )
                if policy.strict:
                    raise RuntimeError(message)
                if step in (None, 0):
                    print(message, flush=True)
            if step == 0:
                summary = context_plan_summary(
                    total_t,
                    [0],
                    total_t,
                    0,
                    phase=0,
                    blend=policy.blend,
                    stagger=False,
                )
                if prompt_segments:
                    # The fast path runs the primary conditioning context
                    # directly; it does not use midpoint selection.
                    summary += " prompt_windows=1x1"
                print(f"{LOG} context plan " + summary, flush=True)
            return executor(x, timestep, context, transformer_options, **kwargs)

        full_layout = payload.get("layout")
        try:
            if full_layout is None:
                from comfy.ldm.minimax.model import PackedLayout
                model = executor.class_obj
                full_layout = PackedLayout(context.shape[1], video_x.shape[2], video_x.shape[3], video_x.shape[4],
                                           audio_x.shape[-1], keyframes=payload.get("keyframes"),
                                           refs=payload.get("refs"))

            # Each window is denoised under one hard-selected prompt, so with a
            # segmented prompt every seam is a prompt boundary: moving it moves
            # which latents blend segments N and N+1 from step to step, and that
            # per-latent prompt drift is the morphing this wrapper exists to
            # prevent. Staggering therefore runs only when every window carries
            # the same prompt; segmented prompts keep fixed window coverage.
            segmented = bool(prompt_segments) and len(prompt_segments) > 1
            stagger = policy.stagger and not segmented
            if policy.stagger and segmented and step in (None, 0):
                print(
                    f"{LOG} stagger pinned off: {len(prompt_segments)} pipe prompt segments need "
                    "fixed window coverage so no latent changes prompt between steps",
                    flush=True,
                )
            phase = 0
            max_phase = 0
            if stagger:
                max_phase = max_stagger_phase(policy.window_frames, policy.overlap_frames)
                if step is not None:
                    phase = stagger_phase(step, policy.window_frames, policy.overlap_frames)
            starts = window_starts(total_t, policy.window_frames, policy.overlap_frames, phase, max_phase)
            segment_indices: list[int] = []
            if prompt_segments:
                segment_indices = [
                    select_segment_index(v0, min(v0 + policy.window_frames, total_t), total_t,
                                         len(prompt_segments), prompt_durations)
                    for v0 in starts
                ]
                missing = unreachable_segments(
                    starts, policy.window_frames, total_t, len(prompt_segments), prompt_durations)
                if missing:
                    message = (
                        f"{LOG} pipe prompt segments {[m + 1 for m in missing]} are never selected by any "
                        f"context window ({len(prompt_segments)} segments across {len(starts)} windows); "
                        "use fewer segments or a smaller window_frames"
                    )
                    if policy.strict:
                        raise RuntimeError(message)
                    if step in (None, 0):
                        print(message, flush=True)
            raw_audio_ranges = [audio_range_for_video_window(full_layout, v0, min(v0 + policy.window_frames, total_t))
                                for v0 in starts]
            # H3's cadence can make otherwise equal video windows carry 141--143
            # audio latents. Expand all of them to the longest interval so the
            # compiled attention shape stays constant within the forward.
            target_audio_t = max(a1 - a0 for a0, a1 in raw_audio_ranges)
            audio_ranges = [expand_audio_range(r, audio_x.shape[-1], target_audio_t)
                            for r in raw_audio_ranges]
            model = executor.class_obj
            padded_h, padded_w = padded_spatial_shape(video_x.shape[3], video_x.shape[4], model.patch_size)

            if step == 0:
                print(
                    f"{LOG} context plan " + context_plan_summary(
                        total_t,
                        starts,
                        policy.window_frames,
                        policy.overlap_frames,
                        phase=phase,
                        blend=policy.blend,
                        stagger=stagger,
                        prompt_count=len(prompt_segments) if prompt_segments else 0,
                        prompt_durations=prompt_durations,
                        max_phase=max_phase,
                    ),
                    flush=True,
                )

            video_acc = torch.zeros_like(video_x)
            video_den = torch.zeros((1, 1, total_t, 1, 1), device=video_x.device, dtype=torch.float32)
            audio_acc = torch.zeros_like(audio_x)
            audio_den = torch.zeros((1, 1, 1, audio_x.shape[-1]), device=audio_x.device, dtype=torch.float32)

            for index, (v0, (a0, a1)) in enumerate(zip(starts, audio_ranges)):
                v1 = min(v0 + policy.window_frames, total_t)
                local_context = context
                if prompt_segments:
                    # Hard per-window selection: contextualized token slots from
                    # different prompts do not correspond, so windows never mix
                    # hidden states — boundary crossfade happens in output space
                    # through the overlap-add, at seams that never move.
                    local_context = prompt_segments[segment_indices[index]]
                local_layout = clone_window_layout(
                    full_layout=full_layout,
                    text_len=local_context.shape[1],
                    # MiniMax pads before validating payload["layout"]. Build the
                    # transplanted layout from those post-pad H/W dimensions so
                    # upstream cannot silently discard it on odd latent shapes.
                    video_shape=(v1 - v0, padded_h, padded_w),
                    audio_t=audio_x.shape[-1],
                    video_range=(v0, v1),
                    audio_range=(a0, a1),
                    keyframes=payload.get("keyframes"), refs=payload.get("refs"),
                )
                local_payload = dict(payload)
                local_payload["layout"] = local_layout
                local_kwargs = dict(kwargs)
                local_kwargs["minimax_payload"] = local_payload
                local_kwargs["denoise_mask"] = _slice_optional_video(kwargs.get("denoise_mask"), v0, v1)
                local_kwargs["audio_denoise_mask"] = _slice_optional_audio(kwargs.get("audio_denoise_mask"), a0, a1)

                local_x = [video_x[:, :, v0:v1], audio_x[..., a0:a1]]
                sentinel = object()
                previous_layout = transformer_options.get("h3forge_active_layout", sentinel)
                transformer_options["h3forge_active_layout"] = local_layout
                try:
                    v_out, a_out = executor(local_x, timestep, local_context, transformer_options, **local_kwargs)
                finally:
                    if previous_layout is sentinel:
                        transformer_options.pop("h3forge_active_layout", None)
                    else:
                        transformer_options["h3forge_active_layout"] = previous_layout

                vw = blend_weights(v1 - v0, policy.overlap_frames, device=v_out.device,
                                   dtype=torch.float32, mode=policy.blend,
                                   ramp_start=v0 > 0, ramp_end=v1 < total_t).view(1, 1, -1, 1, 1)
                # Audio overlap is induced by the physical-time mapping, so use a
                # proportionate ramp rather than blindly copying video indices.
                audio_overlap = audio_overlap_frames(policy.overlap_frames, v1 - v0, a1 - a0)
                aw = blend_weights(a1 - a0, audio_overlap, device=a_out.device,
                                   dtype=torch.float32, mode=policy.blend,
                                   ramp_start=a0 > 0, ramp_end=a1 < audio_x.shape[-1]).view(1, 1, 1, -1)

                video_acc[:, :, v0:v1].add_(v_out * vw.to(v_out.dtype))
                video_den[:, :, v0:v1].add_(vw)
                audio_acc[..., a0:a1].add_(a_out * aw.to(a_out.dtype))
                audio_den[..., a0:a1].add_(aw)

            if policy.strict:
                assert_full_coverage(video_den, audio_den)
            video_out = video_acc / video_den.clamp_min(1e-6).to(video_acc.dtype)
            audio_out = audio_acc / audio_den.clamp_min(1e-6).to(audio_acc.dtype)
            return [video_out, audio_out]
        except Exception as exc:
            if policy.strict:
                raise RuntimeError(f"{LOG} context windowing failed: {type(exc).__name__}: {exc}") from exc
            print(
                f"{LOG} context windowing declined ({type(exc).__name__}: {exc}); dense full-context forward",
                flush=True,
            )
            return executor(x, timestep, context, transformer_options, **kwargs)

    return wrapper

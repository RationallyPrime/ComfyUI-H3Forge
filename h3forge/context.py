from __future__ import annotations

from dataclasses import dataclass

import torch

from .layout import audio_range_for_video_window, clone_window_layout, expand_audio_range, padded_spatial_shape
from .prompt import select_segment_index, unreachable_segments
from .state import resolve_step

LOG = "[H3Forge]"


@dataclass(frozen=True)
class ContextPolicy:
    window_frames: int = 25
    overlap_frames: int = 5
    stagger: bool = True
    blend: str = "pyramid"
    strict: bool = False


def window_starts(total: int, window: int, overlap: int, phase: int = 0) -> list[int]:
    if window >= total:
        return [0]
    if window < 2:
        raise ValueError("window must be >=2")
    stride = window - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than window")

    starts = [0]
    phase = max(0, min(int(phase), overlap))
    s = stride + phase
    while s < total - window:
        starts.append(max(1, s))
        s += stride
    starts.append(total - window)
    return sorted(set(starts))


def blend_weights(length: int, overlap: int, *, device, dtype, mode="pyramid",
                  ramp_start: bool = True, ramp_end: bool = True):
    """Overlap-add blend weights with explicit first/last edge correction.

    The outermost edges of the full timeline have no neighbouring window to
    blend with, so the first window keeps full weight on its left edge and the
    last window on its right edge (ramp_start/ramp_end False). Interior edges
    ramp as usual.
    """
    if length <= 0:
        return torch.empty(0, device=device, dtype=dtype)
    if mode == "flat" or overlap <= 0 or not (ramp_start or ramp_end):
        return torch.ones(length, device=device, dtype=dtype)
    ramp = min(overlap, max(length // 2, 1))
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
            return executor(x, timestep, context, transformer_options, **kwargs)

        full_layout = payload.get("layout")
        try:
            if full_layout is None:
                from comfy.ldm.minimax.model import PackedLayout
                model = executor.class_obj
                full_layout = PackedLayout(context.shape[1], video_x.shape[2], video_x.shape[3], video_x.shape[4],
                                           audio_x.shape[-1], keyframes=payload.get("keyframes"),
                                           refs=payload.get("refs"))

            stride = policy.window_frames - policy.overlap_frames
            phase = 0
            if policy.stagger and stride > 2 and step is not None:
                phase = (step * max(stride // 3, 1)) % stride
            starts = window_starts(total_t, policy.window_frames, policy.overlap_frames, phase)
            if prompt_segments:
                # Validate reachability over every stagger phase the run can
                # visit, not just this step's starts: a config that survives
                # step 0 must not strict-abort or silently drop a segment once
                # the window boundaries shift on later steps.
                phases = {0}
                if policy.stagger and stride > 2:
                    increment = max(stride // 3, 1)
                    phases = {min((s * increment) % stride, policy.overlap_frames) for s in range(stride)}
                missing_union: set[int] = set()
                for p in sorted(phases):
                    missing_union.update(unreachable_segments(
                        window_starts(total_t, policy.window_frames, policy.overlap_frames, p),
                        policy.window_frames, total_t, len(prompt_segments)))
                missing = sorted(missing_union)
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

            if prompt_segments and step == 0:
                print(
                    f"{LOG} pipe prompt segments={len(prompt_segments)} mapped across {total_t} video latents",
                    flush=True,
                )

            video_acc = torch.zeros_like(video_x)
            video_den = torch.zeros((1, 1, total_t, 1, 1), device=video_x.device, dtype=torch.float32)
            audio_acc = torch.zeros_like(audio_x)
            audio_den = torch.zeros((1, 1, 1, audio_x.shape[-1]), device=audio_x.device, dtype=torch.float32)

            for v0, (a0, a1) in zip(starts, audio_ranges):
                v1 = min(v0 + policy.window_frames, total_t)
                local_context = context
                if prompt_segments:
                    # Hard per-window selection: contextualized token slots from
                    # different prompts do not correspond, so windows never mix
                    # hidden states — boundary crossfade happens in output space
                    # through the overlap-add below.
                    segment_index = select_segment_index(v0, v1, total_t, len(prompt_segments))
                    local_context = prompt_segments[segment_index]
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

from __future__ import annotations

from dataclasses import dataclass
import torch

from .layout import audio_range_for_video_window, clone_window_layout, expand_audio_range, padded_spatial_shape
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


def blend_weights(length: int, overlap: int, *, device, dtype, mode="pyramid"):
    if length <= 0:
        return torch.empty(0, device=device, dtype=dtype)
    if mode == "flat" or overlap <= 0:
        return torch.ones(length, device=device, dtype=dtype)
    ramp = min(overlap, max(length // 2, 1))
    w = torch.ones(length, device=device, dtype=torch.float32)
    edge = torch.linspace(1.0 / (ramp + 1), 1.0, ramp, device=device)
    w[:ramp] = edge
    w[-ramp:] = edge.flip(0)
    return w.to(dtype)


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
        if total_t <= policy.window_frames:
            return executor(x, timestep, context, transformer_options, **kwargs)

        payload = dict(kwargs.get("minimax_payload") or {})
        full_layout = payload.get("layout")
        try:
            if full_layout is None:
                from comfy.ldm.minimax.model import PackedLayout
                model = executor.class_obj
                full_layout = PackedLayout(context.shape[1], video_x.shape[2], video_x.shape[3], video_x.shape[4],
                                           audio_x.shape[-1], keyframes=payload.get("keyframes"), refs=payload.get("refs"))

            step, _ = resolve_step(transformer_options)
            stride = policy.window_frames - policy.overlap_frames
            phase = 0
            if policy.stagger and stride > 2 and step is not None:
                phase = (step * max(stride // 3, 1)) % stride
            starts = window_starts(total_t, policy.window_frames, policy.overlap_frames, phase)
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

            video_acc = torch.zeros_like(video_x)
            video_den = torch.zeros((1, 1, total_t, 1, 1), device=video_x.device, dtype=torch.float32)
            audio_acc = torch.zeros_like(audio_x)
            audio_den = torch.zeros((1, 1, 1, audio_x.shape[-1]), device=audio_x.device, dtype=torch.float32)

            for v0, (a0, a1) in zip(starts, audio_ranges):
                v1 = min(v0 + policy.window_frames, total_t)
                local_layout = clone_window_layout(
                    full_layout=full_layout,
                    text_len=context.shape[1],
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
                    v_out, a_out = executor(local_x, timestep, context, transformer_options, **local_kwargs)
                finally:
                    if previous_layout is sentinel:
                        transformer_options.pop("h3forge_active_layout", None)
                    else:
                        transformer_options["h3forge_active_layout"] = previous_layout

                vw = blend_weights(v1 - v0, policy.overlap_frames, device=v_out.device,
                                   dtype=torch.float32, mode=policy.blend).view(1, 1, -1, 1, 1)
                # Audio overlap is induced by the physical-time mapping, so use a
                # proportionate ramp rather than blindly copying video indices.
                audio_overlap = max(1, round(policy.overlap_frames * max((a1 - a0) / max(v1 - v0, 1), 1.0)))
                aw = blend_weights(a1 - a0, audio_overlap, device=a_out.device,
                                   dtype=torch.float32, mode=policy.blend).view(1, 1, 1, -1)

                video_acc[:, :, v0:v1].add_(v_out * vw.to(v_out.dtype))
                video_den[:, :, v0:v1].add_(vw)
                audio_acc[..., a0:a1].add_(a_out * aw.to(a_out.dtype))
                audio_den[..., a0:a1].add_(aw)

            video_out = video_acc / video_den.clamp_min(1e-6).to(video_acc.dtype)
            audio_out = audio_acc / audio_den.clamp_min(1e-6).to(audio_acc.dtype)
            return [video_out, audio_out]
        except Exception as exc:
            if policy.strict:
                raise RuntimeError(f"{LOG} context windowing failed: {type(exc).__name__}: {exc}") from exc
            print(f"{LOG} context windowing declined ({type(exc).__name__}: {exc}); dense full-context forward", flush=True)
            return executor(x, timestep, context, transformer_options, **kwargs)

    return wrapper

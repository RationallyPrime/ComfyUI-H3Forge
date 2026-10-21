"""Masked frame-interval renoise: re-denoise one region over one stretch of an existing H3 clip.

A glitch confined to part of the picture for part of the clip does not need a
full regeneration. Native H3 already runs every packed video row at its own
strength: a latent noise mask value ``m`` puts that row at ``sigma = m * sigma``
and pins ``m = 0`` rows to the clean source at the condition timestep. This
module only builds that mask and the pixel-space paste that goes with it:

* a temporal weight, 1 inside the chosen frames and a smoothstep ramp outside
  them, so the renoise fades in and out rather than starting at a hard seam;
* a spatial mask (static or per frame), grown and feathered in pixels;
* a crop of the clip to the shortest window on H3's 17-frame encoder cycle that
  holds the ramped interval plus clean context, so the sampler pays for a few
  seconds instead of the whole clip. The H3 video VAE encodes each 17-frame
  cycle independently, so the cropped window's latents equal the full clip's;
* the stitch, which pastes the decoded window back through the same pixel
  mask, leaving every frame and pixel outside it bit-identical to the source.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .timeline import FPS, FRAMES_PER_CYCLE, LATENTS_PER_CYCLE, align_frame_count

AUDIO_LATENT_FPS = 40
SPATIAL_RATIO = 16


def latent_frame_spans(frame_count: int) -> list[tuple[int, int]]:
    """Pixel frames ``[a, b)`` each video latent of a grid-aligned clip decodes to.

    Every 17-frame encoder cycle yields one latent for its first frame and four
    for four frames each; the trailing cycle is the 5-frame ``1 + 4`` remainder.
    """
    if frame_count < 5 or frame_count % FRAMES_PER_CYCLE != LATENTS_PER_CYCLE:
        raise ValueError(f"{frame_count} frames is not on H3's 17k+5 grid")
    spans = []
    for cycle in range(frame_count // FRAMES_PER_CYCLE + 1):
        base = cycle * FRAMES_PER_CYCLE
        spans.append((base, base + 1))
        last = 4 if base + FRAMES_PER_CYCLE <= frame_count else 1
        spans.extend((base + 1 + 4 * j, base + 5 + 4 * j) for j in range(last))
    return spans


def temporal_weights(frame_count: int, start: int, end: int, ramp: int) -> torch.Tensor:
    """Per-frame renoise weight: 1 on ``[start, end]``, a smoothstep down to 0 over ``ramp`` frames each side."""
    f = torch.arange(frame_count, dtype=torch.float32)
    distance = torch.clamp(torch.maximum(start - f, f - end), min=0.0)
    if ramp <= 0:
        return (distance == 0).float()
    x = torch.clamp(1.0 - distance / (ramp + 1), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


@dataclass(frozen=True)
class Window:
    """The cropped stretch of the clip that is actually sampled."""

    start: int          # first source frame of the window, a multiple of 17
    length: int         # window frame count, on the 17k+5 grid
    valid: int          # frames of the window that exist in the source (the rest repeat the last frame)
    interval: tuple[int, int]  # renoised source frames, inclusive
    ramp: int


def plan_window(source_frames: int, start: int, end: int, ramp: int, context: int, crop: bool) -> Window:
    """Smallest cycle-aligned window covering the ramped interval plus ``context`` clean frames each side."""
    if source_frames < 1:
        raise ValueError("the source clip has no frames")
    start = start if start >= 0 else source_frames + start
    end = end if end >= 0 else source_frames + end
    if not 0 <= start <= end < source_frames:
        raise ValueError(f"frame range {start}-{end} is outside the clip's {source_frames} frames")
    ramp, context = max(0, int(ramp)), max(0, int(context))
    if not crop:
        length = align_frame_count(source_frames)
        return Window(0, length, source_frames, (start, end), ramp)
    lo = max(0, start - ramp - context)
    hi = min(source_frames - 1, end + ramp + context)
    first = lo // FRAMES_PER_CYCLE * FRAMES_PER_CYCLE
    length = align_frame_count(hi - first + 1)
    # Prefer a window made of real frames: slide it back a cycle at a time
    # when it runs past the end, as long as it still covers the interval.
    while first + length > source_frames and first >= FRAMES_PER_CYCLE \
            and first - FRAMES_PER_CYCLE + length > end + ramp:
        first -= FRAMES_PER_CYCLE
    return Window(first, length, min(length, source_frames - first), (start, end), ramp)


def _resize_masks(masks: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if masks.shape[-2:] == (height, width):
        return masks
    return F.interpolate(masks.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False).squeeze(1)


def _grow(masks: torch.Tensor, pixels: int) -> torch.Tensor:
    if pixels <= 0:
        return masks
    return F.max_pool2d(masks.unsqueeze(1), 2 * pixels + 1, stride=1, padding=pixels).squeeze(1)


def _feather(masks: torch.Tensor, pixels: int) -> torch.Tensor:
    if pixels <= 0:
        return masks
    sigma = pixels / 2.0
    taps = torch.arange(-pixels, pixels + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (taps / sigma) ** 2)
    kernel = (kernel / kernel.sum()).to(masks)
    x = masks.unsqueeze(1)
    x = F.conv2d(F.pad(x, (pixels, pixels, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, pixels, pixels), mode="replicate"), kernel.view(1, 1, -1, 1))
    return x.squeeze(1).clamp(0.0, 1.0)


def spatial_masks(mask, frames: torch.Tensor, source_frames: int, interval: tuple[int, int],
                  height: int, width: int, grow: int, feather: int) -> torch.Tensor:
    """Spatial masks ``[len(frames), H, W]`` in ``[0, 1]`` for the given source frames.

    ``mask`` may be None (whole frame), one static mask, one mask per source
    frame, or one per interval frame (frames in the ramp reuse the nearest end).
    """
    frames = frames.clamp(0, source_frames - 1)
    if mask is None:
        return torch.ones(frames.numel(), height, width)
    mask = mask.float().cpu()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    count = mask.shape[0]
    a, b = interval
    if count == 1:
        index = torch.zeros_like(frames)
    elif count == source_frames:
        index = frames
    elif count == b - a + 1:
        index = (frames - a).clamp(0, count - 1)
    else:
        raise ValueError(f"mask has {count} frames; give 1, one per clip frame ({source_frames}) "
                         f"or one per renoised frame ({b - a + 1})")
    picked = _resize_masks(mask[index], height, width).clamp(0.0, 1.0)
    return _feather(_grow(picked, grow), feather)


def window_indices(window: Window) -> torch.Tensor:
    return torch.arange(window.start, window.start + window.length)


def blend_masks(spatial: torch.Tensor, window: Window) -> torch.Tensor:
    """Pixel paste weights for the window: spatial mask times the ramped interval weight."""
    a, b = window.interval
    weights = temporal_weights(window.start + window.length, a, b, window.ramp)[window.start:]
    weights[window.valid:] = 0.0  # padded tail frames never paste
    return spatial * weights.view(-1, 1, 1)


def video_noise_mask(blend: torch.Tensor, window: Window, denoise: float) -> torch.Tensor:
    """Latent noise mask ``[1, 1, T, H/16, W/16]``: each latent takes the max of the pixels it decodes to."""
    rows = []
    for a, b in latent_frame_spans(window.length):
        frame = blend[a:b].amax(dim=0, keepdim=True).unsqueeze(0)
        rows.append(F.max_pool2d(frame, SPATIAL_RATIO, stride=SPATIAL_RATIO, ceil_mode=True))
    return (torch.cat(rows, dim=1).unsqueeze(0) * float(denoise)).clamp(0.0, 1.0)


def audio_noise_mask(audio_t: int, window: Window, denoise: float, renoise: bool) -> torch.Tensor:
    """Audio noise mask ``[1, 1, 2, T40]``: zero (kept) unless ``renoise``, then the interval ramp in time."""
    if not renoise:
        return torch.zeros(1, 1, 2, audio_t)
    a, b = window.interval
    weights = temporal_weights(window.start + window.length, a, b, window.ramp)
    # Audio latent i sits at video frame start + i * 24 / 40.
    frame = torch.clamp((window.start + torch.arange(audio_t) * FPS / AUDIO_LATENT_FPS).round().long(),
                        max=weights.numel() - 1)
    return (weights[frame] * float(denoise)).view(1, 1, 1, -1).expand(1, 1, 2, audio_t).contiguous()


def window_frames(images: torch.Tensor, window: Window) -> torch.Tensor:
    """Source frames of the window, the tail padded by repeating the last real frame."""
    clip = images[window.start:window.start + window.valid]
    if window.valid < window.length:
        clip = torch.cat([clip, clip[-1:].expand(window.length - window.valid, *clip.shape[1:])])
    return clip


def splice(images: torch.Tensor, inpainted: torch.Tensor, frame: int, spatial) -> torch.Tensor:
    """Paste inpainted frames over the source starting at ``frame``, through the spatial mask when given."""
    out = images.clone()
    frame = frame if frame >= 0 else images.shape[0] + frame
    count = min(inpainted.shape[0], images.shape[0] - frame)
    if frame < 0 or count < 1:
        raise ValueError(f"inpainted_frame_idx {frame} is outside the clip's {images.shape[0]} frames")
    patch = inpainted[:count, ..., :images.shape[-1]].to(out)
    if patch.shape[1:3] != images.shape[1:3]:
        patch = F.interpolate(patch.movedim(-1, 1), size=images.shape[1:3], mode="bilinear",
                              align_corners=False).movedim(1, -1)
    if spatial is None:
        out[frame:frame + count] = patch
    else:
        weight = spatial[:count].unsqueeze(-1).to(out)
        out[frame:frame + count] = out[frame:frame + count] * (1 - weight) + patch * weight
    return out


def stitch(source: torch.Tensor, window: Window, blend: torch.Tensor, renoised: torch.Tensor) -> torch.Tensor:
    """Paste the decoded window back over the source through the pixel blend mask."""
    if renoised.shape[0] < window.valid:
        raise ValueError(f"renoised clip has {renoised.shape[0]} frames; the window needs {window.valid}")
    if renoised.shape[1:3] != source.shape[1:3]:
        renoised = F.interpolate(renoised.movedim(-1, 1), size=source.shape[1:3], mode="bilinear",
                                 align_corners=False).movedim(1, -1)
    out = source.clone()
    a, n = window.start, window.valid
    weight = blend[:n].unsqueeze(-1).to(out)
    out[a:a + n] = out[a:a + n] * (1 - weight) + renoised[:n, ..., :out.shape[-1]].to(out) * weight
    return out


def fit_audio_latent(latent: torch.Tensor, audio_t: int) -> torch.Tensor:
    """Crop or edge-pad an encoded audio latent ``[1, 32, 2, T]`` to the window's audio length."""
    if latent.shape[-1] >= audio_t:
        return latent[..., :audio_t].contiguous()
    pad = latent[..., -1:].expand(*latent.shape[:-1], audio_t - latent.shape[-1])
    return torch.cat([latent, pad], dim=-1)


def audio_slice(audio: dict, window: Window) -> dict:
    """The window's stretch of a ComfyUI AUDIO dict."""
    rate = int(audio["sample_rate"])
    a = int(round(window.start / FPS * rate))
    b = a + int(math.ceil(window.length / FPS * rate))
    waveform = audio["waveform"][..., a:b]
    if waveform.shape[-1] < b - a:
        waveform = F.pad(waveform, (0, b - a - waveform.shape[-1]))
    return {"waveform": waveform, "sample_rate": rate}

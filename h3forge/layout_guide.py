"""One sparse, joint A/V forward supplies the shared coarse video prediction."""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F

from .beat_refs import filter_payload
from .control import control_window
from .layout import audio_window_length, clone_window_layout, video_frame_positions


def sparse_video_indices(total, window):
    """Prefer H3's single-frame latents, keep both endpoints, fill the window budget."""
    from .timeline import LATENTS_PER_CYCLE

    if total <= window:
        return list(range(total))
    if window < 2:
        raise ValueError("a whole-take layout guide needs at least two video latents")
    anchors = list(range(0, total, LATENTS_PER_CYCLE))
    if anchors[-1] != total - 1:
        anchors.append(total - 1)
    if len(anchors) >= window:
        return [anchors[round(k * (len(anchors) - 1) / (window - 1))] for k in range(window)]
    selected = set(anchors)
    pool = [i for i in range(total) if i not in selected]
    extra = window - len(selected)
    selected.update(pool[int((k + 0.5) * len(pool) / extra)] for k in range(extra))
    return sorted(selected)


def spatial_lowpass(x):
    """Separable 7×7 Gaussian on each latent plane, with replicated borders."""
    b, c, t, h, w = x.shape
    if h == w == 1:
        return x.float()
    coord = torch.arange(-3, 4, device=x.device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (coord / 1.5).square())
    kernel = kernel / kernel.sum()
    planes = x.float().reshape(b * c * t, 1, h, w)
    if h > 1:
        planes = F.conv2d(F.pad(planes, (0, 0, 3, 3), mode="replicate"), kernel.view(1, 1, 7, 1))
    if w > 1:
        planes = F.conv2d(F.pad(planes, (3, 3, 0, 0), mode="replicate"), kernel.view(1, 1, 1, 7))
    return planes.reshape(b, c, t, h, w)


def interpolate_sparse_video(sparse, anchors, times):
    """Reconstruct only the requested window on H3's nonuniform physical time grid."""
    right = torch.searchsorted(anchors, times).clamp(max=len(anchors) - 1)
    left = torch.where(anchors[right] == times, right, (right - 1).clamp(min=0))
    distance = anchors[right] - anchors[left]
    alpha = (times - anchors[left]) / distance.clamp_min(1e-6)
    return torch.lerp(sparse.index_select(2, left.to(sparse.device)), sparse.index_select(2, right.to(sparse.device)),
                      alpha.to(device=sparse.device, dtype=torch.float32).view(1, 1, -1, 1, 1))


class SharedLayout(NamedTuple):
    low_video: torch.Tensor
    anchor_times: torch.Tensor
    full_times: torch.Tensor
    audio_ticks: int

    def low_for_window(self, start, stop):
        return interpolate_sparse_video(self.low_video, self.anchor_times, self.full_times[start:stop])


def shared_layout_prediction(executor, x, timestep, text, tags, options, kwargs, full_layout,
                             window, padded_h, padded_w, ref_keeps):
    """Subsample both streams and masks in global coordinates; consume only video velocity."""
    video, audio = x
    payload = kwargs.get("minimax_payload") or {}
    vid = sparse_video_indices(video.shape[2], window)
    audio_t = audio.shape[-1]
    audio_win = audio_window_length(full_layout, window)
    aud = torch.linspace(0, audio_t - 1, audio_win, dtype=torch.float64).round().long().tolist()
    keep = None
    if ref_keeps is not None:
        keep = sorted(set.intersection(*(set(indices) for indices in ref_keeps if indices is not None)))
    layout = clone_window_layout(
        full_layout=full_layout, text_len=text.shape[1], video_shape=(len(vid), padded_h, padded_w),
        audio_t=audio_t, sparse_indices=(vid, aud), keyframes=payload.get("keyframes"),
        refs=payload.get("refs"), ref_keep=keep)
    wide_payload = {**(payload if keep is None else filter_payload(payload, keep)), "layout": layout}
    if tags is not None:
        wide_payload["text_token_tags"] = tags
    wide_kwargs = {**kwargs, "minimax_payload": wide_payload}
    for key, axis, indices in (("denoise_mask", 2, vid), ("audio_denoise_mask", -1, aud)):
        mask = kwargs.get(key)
        if mask is not None:
            wide_kwargs[key] = mask.index_select(axis, torch.tensor(indices, device=mask.device))
    wide_options = {**options, "h3forge_active_layout": layout}
    with control_window(executor, video.shape, None, timestep, wide_options, video_indices=vid):
        wide_video, _ = executor([video[:, :, vid], audio[..., aud]], timestep, text, wide_options, **wide_kwargs)
    times = video_frame_positions(full_layout).contiguous()
    return SharedLayout(spatial_lowpass(wide_video), times[vid], times, len(aud))

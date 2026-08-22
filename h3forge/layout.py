from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class TargetSegments:
    audio_start: int
    audio_stop: int
    video_start: int
    video_stop: int


def padded_spatial_shape(height: int, width: int, patch_size: tuple[int, int, int]) -> tuple[int, int]:
    """Return H3's post-pad spatial latent shape without allocating a tensor."""
    if len(patch_size) != 3 or int(patch_size[0]) != 1:
        raise ValueError(f"expected H3 patch_size=(1, ph, pw), got {patch_size}")
    ph, pw = int(patch_size[1]), int(patch_size[2])
    if ph <= 0 or pw <= 0:
        raise ValueError(f"invalid H3 patch_size={patch_size}")
    return ((int(height) + ph - 1) // ph * ph,
            (int(width) + pw - 1) // pw * pw)


def expand_audio_range(audio_range: tuple[int, int], total: int, target_length: int) -> tuple[int, int]:
    """Expand a valid audio interval to a constant length while preserving it."""
    a0, a1 = map(int, audio_range)
    total, target_length = int(total), int(target_length)
    if not (0 <= a0 < a1 <= total):
        raise ValueError(f"invalid audio range {audio_range} for total={total}")
    if not (a1 - a0 <= target_length <= total):
        raise ValueError(f"cannot expand {audio_range} to {target_length} within total={total}")
    spare = target_length - (a1 - a0)
    left = min(a0, spare // 2)
    right = min(total - a1, spare - left)
    left += spare - left - right
    return a0 - left, a1 + right


def target_segments(layout) -> TargetSegments:
    audio = [s for s in layout.segments if s[2] == "audio"]
    video = [s for s in layout.segments if s[2] == "video"]
    if len(audio) != 1 or len(video) != 1:
        raise ValueError(f"expected one target audio and video segment, got audio={audio}, video={video}")
    return TargetSegments(int(audio[0][0]), int(audio[0][1]), int(video[0][0]), int(video[0][1]))


def video_shape_from_layout(layout) -> tuple[int, int]:
    seg = target_segments(layout)
    times = layout.position_ids[seg.video_start:seg.video_stop, 0]
    if times.numel() == 0:
        raise ValueError("empty target video segment")
    changes = torch.ones_like(times, dtype=torch.bool)
    changes[1:] = times[1:] != times[:-1]
    latent_t = int(changes.sum())
    if latent_t <= 0 or times.numel() % latent_t:
        raise ValueError("video rows are not frame-major with a constant rows-per-frame")
    return latent_t, int(times.numel() // latent_t)


def clone_window_layout(
    *,
    full_layout,
    text_len: int,
    video_shape: tuple[int, int, int],
    audio_t: int,
    video_range: tuple[int, int],
    audio_range: tuple[int, int],
    keyframes=None,
    refs=None,
):
    """Build a local H3 PackedLayout, then transplant global target positions.

    This preserves absolute multimodal RoPE coordinates while allowing the model
    to process a smaller synchronized A/V window.
    """
    from comfy.ldm.minimax.model import PackedLayout

    v0, v1 = video_range
    a0, a1 = audio_range
    local_t, latent_h, latent_w = map(int, video_shape)
    if local_t != v1 - v0:
        raise ValueError(f"local video shape T={local_t} does not match window {video_range}")
    if not (0 <= a0 < a1 <= int(audio_t)):
        raise ValueError(f"invalid audio range {audio_range} for T={audio_t}")
    local = PackedLayout(text_len, local_t, latent_h, latent_w, a1 - a0,
                         keyframes=keyframes, refs=refs)
    expected_signature = (int(text_len), local_t, latent_h, latent_w, a1 - a0)
    if local.signature != expected_signature:
        raise ValueError(f"local PackedLayout signature {local.signature} != expected {expected_signature}")
    src = target_segments(full_layout)
    dst = target_segments(local)

    full_t, frame_rows = video_shape_from_layout(full_layout)
    if v1 > full_t:
        raise ValueError(f"video window {video_range} exceeds full latent length {full_t}")

    # Video rows are frame-major and therefore a contiguous slice.
    src_v0 = src.video_start + v0 * frame_rows
    src_v1 = src.video_start + v1 * frame_rows
    local.position_ids[dst.video_start:dst.video_stop] = full_layout.position_ids[src_v0:src_v1]

    # Audio rows are channel-major: [L0..Lt-1, R0..Rt-1].
    full_audio_t = (src.audio_stop - src.audio_start) // 2
    left = full_layout.position_ids[src.audio_start + a0:src.audio_start + a1]
    right = full_layout.position_ids[src.audio_start + full_audio_t + a0:
                                     src.audio_start + full_audio_t + a1]
    local.position_ids[dst.audio_start:dst.audio_stop] = torch.cat([left, right], dim=0)
    # Runtime-only metadata for compiler-friendly sparse mask arithmetic.
    local._h3forge_video_offset = int(v0)
    local._h3forge_audio_offset = int(a0)
    return local


def audio_range_for_video_window(full_layout, v0: int, v1: int) -> tuple[int, int]:
    """Map a video-latent interval onto H3 audio latent indices by timeline overlap."""
    seg = target_segments(full_layout)
    latent_t, frame_rows = video_shape_from_layout(full_layout)
    if not (0 <= v0 < v1 <= latent_t):
        raise ValueError(f"invalid video range {(v0, v1)} for T={latent_t}")

    vpos = full_layout.position_ids[seg.video_start:seg.video_stop, 0].reshape(latent_t, frame_rows)[:, 0]
    apos = full_layout.position_ids[seg.audio_start:seg.audio_stop, 0]
    audio_t = apos.numel() // 2
    apos = apos[:audio_t]

    start_t = float(vpos[v0])
    if v1 < latent_t:
        stop_t = float(vpos[v1])
    else:
        # Infer the final video-token span from the established H3 cadence.
        from comfy.ldm.minimax.model import _video_t_spans
        stop_t = float(vpos[-1]) + float(_video_t_spans(latent_t)[-1])

    # Include every audio latent whose unit interval overlaps [start_t, stop_t).
    origin = float(apos[0]) if audio_t else start_t
    a0 = max(0, int(math.floor(start_t - origin)))
    a1 = min(audio_t, int(math.ceil(stop_t - origin)))
    if a1 <= a0:
        # Defensive minimum; a synchronized window must carry some audio.
        a0 = min(max(a0, 0), max(audio_t - 1, 0))
        a1 = min(a0 + 1, audio_t)
    return a0, a1


def modality_vectors(layout, device):
    """Per-token time/space coordinates and target-modality tags for sparse routing."""
    seg = target_segments(layout)
    pos = layout.position_ids.to(device=device, dtype=torch.float32)
    idx = torch.arange(layout.seq_len, device=device)
    is_global = idx < seg.audio_start
    is_audio = (idx >= seg.audio_start) & (idx < seg.audio_stop)
    is_video = (idx >= seg.video_start) & (idx < seg.video_stop)
    return pos, is_global, is_audio, is_video

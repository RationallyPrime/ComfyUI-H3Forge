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
    extra_keyframes=None,
):
    """Build a local H3 PackedLayout, then transplant global target positions.

    This preserves absolute multimodal RoPE coordinates while allowing the model
    to process a smaller synchronized A/V window. Optional ``extra_keyframes``
    are ``(descriptor, anchor_video_latent)`` pairs appended after any shared
    ``keyframes`` for this window only; each receives the global RoPE position
    of ``anchor_video_latent`` in ``full_layout``.
    """
    from comfy.ldm.minimax.model import PackedLayout

    v0, v1 = video_range
    a0, a1 = audio_range
    local_t, latent_h, latent_w = map(int, video_shape)
    if local_t != v1 - v0:
        raise ValueError(f"local video shape T={local_t} does not match window {video_range}")
    if not (0 <= a0 < a1 <= int(audio_t)):
        raise ValueError(f"invalid audio range {audio_range} for T={audio_t}")
    extra_list = list(extra_keyframes or ())
    all_keyframes = list(keyframes or ()) + [kf for kf, _ in extra_list] if extra_list else keyframes
    local = PackedLayout(text_len, local_t, latent_h, latent_w, a1 - a0,
                         keyframes=all_keyframes, refs=refs)
    expected_signature = (int(text_len), local_t, latent_h, latent_w, a1 - a0)
    if local.signature != expected_signature:
        raise ValueError(f"local PackedLayout signature {local.signature} != expected {expected_signature}")
    src = target_segments(full_layout)
    dst = target_segments(local)
    full_t, frame_rows = video_shape_from_layout(full_layout)
    if v1 > full_t:
        raise ValueError(f"video window {video_range} exceeds full latent length {full_t}")

    if extra_list:
        insert_at = 1 if full_layout.segments and full_layout.segments[0][2] == "text" else 0
        while insert_at < len(full_layout.segments) and full_layout.segments[insert_at][2] in ("cond", "cond_audio"):
            insert_at += 1
        extra_segs = local.segments[insert_at:insert_at + len(extra_list)]
        matched_local = local.segments[:insert_at] + local.segments[insert_at + len(extra_list):]
    else:
        extra_segs = ()
        matched_local = local.segments

    if len(full_layout.segments) != len(matched_local):
        raise ValueError("window reference structure differs from the full layout")
    # Independently refined text has native per-segment lengths. Preserve the
    # shared refs/keyframes at the template's origin without inserting fake
    # text rows in the local stream.
    for (fa, fb, fk), (la, lb, lk) in zip(full_layout.segments, matched_local):
        if fk != lk:
            raise ValueError("window reference structure differs from the full layout")
        if fk not in ("text", "audio", "video"):
            if fb - fa != lb - la:
                raise ValueError("window reference row count differs from the full layout")
            local.position_ids[la:lb] = full_layout.position_ids[fa:fb]

    if extra_list:
        if len(extra_segs) != len(extra_list):
            raise ValueError("window extra keyframe segment count differs from request")
        for (la, lb, lk), (_, anchor_latent) in zip(extra_segs, extra_list):
            anchor = int(anchor_latent)
            if lk != "cond" or lb - la != frame_rows:
                raise ValueError("extra keyframe must occupy one single-frame video cond segment")
            if not (0 <= anchor < full_t):
                raise ValueError(f"extra keyframe anchor {anchor} out of range [0, {full_t})")
            src_a0 = src.video_start + anchor * frame_rows
            local.position_ids[la:lb] = full_layout.position_ids[src_a0:src_a0 + frame_rows]

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


def audio_window_length(full_layout, video_window: int) -> int:
    """Fixed audio-tick span covering every ``video_window``-latent slice of ``full_layout``.

    A fixed video window overlaps ``N`` or ``N + 1`` 40 Hz audio ticks depending
    on where its start lands on H3's ``1,4,4,4,4`` cadence. Taking the maximum
    across all starts keeps ``PackedLayout.signature`` constant across windows
    and stagger phases so sparse ``flex_attention`` compiles one shape per prompt.
    """
    seg = target_segments(full_layout)
    latent_t, frame_rows = video_shape_from_layout(full_layout)
    video_window = int(video_window)
    if not (1 <= video_window <= latent_t):
        raise ValueError(f"invalid video_window={video_window} for T={latent_t}")
    audio_t = (seg.audio_stop - seg.audio_start) // 2
    if video_window == latent_t or audio_t <= 1:
        return audio_t

    vpos = full_layout.position_ids[seg.video_start:seg.video_stop, 0].reshape(
        latent_t, frame_rows)[:, 0].to(torch.float64)
    apos = full_layout.position_ids[seg.audio_start:seg.audio_stop, 0]
    origin = float(apos[0])
    from comfy.ldm.minimax.model import _video_t_spans

    final_stop = vpos[-1:] + float(_video_t_spans(latent_t)[-1])
    starts = vpos[:latent_t - video_window + 1]
    stops = torch.cat([vpos[video_window:], final_stop])
    a0 = torch.floor(starts - origin).to(torch.int64).clamp(min=0)
    a1 = torch.ceil(stops - origin).to(torch.int64).clamp(max=audio_t)
    return max(1, min(audio_t, int((a1 - a0).max().item())))


def modality_vectors(layout, device):
    """Per-token time/space coordinates and target-modality tags for sparse routing."""
    seg = target_segments(layout)
    pos = layout.position_ids.to(device=device, dtype=torch.float32)
    idx = torch.arange(layout.seq_len, device=device)
    is_global = idx < seg.audio_start
    is_audio = (idx >= seg.audio_start) & (idx < seg.audio_stop)
    is_video = (idx >= seg.video_start) & (idx < seg.video_stop)
    return pos, is_global, is_audio, is_video

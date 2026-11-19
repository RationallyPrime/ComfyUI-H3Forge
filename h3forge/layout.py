from __future__ import annotations

from dataclasses import dataclass
import math

import torch

POSITION_MODES = ("global", "rebased", "wrapped")
LATENTS_PER_CYCLE = 5
FRAMES_PER_CYCLE = 17
FRAME_RESCALE = 5.0 / 3.0
CYCLE_TIME_SPAN = FRAMES_PER_CYCLE * FRAME_RESCALE
# 107 video latents (362 pixel frames, 15.08 s) is the top of H3's trained range.
TRAINED_MAX_LATENTS = 107
TRAINED_MAX_FRAMES = 362
TRAINED_MAX_TIME = TRAINED_MAX_FRAMES * FRAME_RESCALE


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


def window_position_shift(full_layout, v0: int, v1: int, position_mode: str = "global") -> float:
    """Time-coordinate subtraction (in 40 Hz audio ticks) for one window's RoPE positions.

    ``PackedLayout`` places text on ``[0, text_len)``, optional refs on
    ``[text_len, origin)``, and target audio/video starting at ``origin`` in
    40 Hz ticks (1.0 per audio latent, ``5/3`` per 24 fps pixel frame, and
    ``85/3`` per 5-latent / 17-frame VAE cycle).

    - ``global``: shift is ``0.0`` (absolute clip coordinates).
    - ``rebased``: subtract ``(v0 // 5)`` full cycles so target video starts at
      ``origin`` plus its offset inside the 5-latent cycle, and target audio
      shifts by the same amount to preserve A/V alignment.
    - ``wrapped``: keep global positions while the window's centre lies inside
      H3's trained range (``[0, TRAINED_MAX_TIME]`` relative to ``origin``);
      otherwise subtract the smallest integer multiple of ``CYCLE_TIME_SPAN``
      (capped at ``v0 // 5`` cycles) that brings the centre into that range.
    """
    if position_mode not in POSITION_MODES:
        raise ValueError(f"position_mode must be one of {POSITION_MODES}, got {position_mode!r}")
    if position_mode == "global":
        return 0.0

    seg = target_segments(full_layout)
    full_t, frame_rows = video_shape_from_layout(full_layout)
    if not (0 <= v0 < v1 <= full_t):
        raise ValueError(f"invalid video range {(v0, v1)} for T={full_t}")

    max_cycles = v0 // LATENTS_PER_CYCLE
    if position_mode == "rebased":
        return max_cycles * CYCLE_TIME_SPAN

    vpos = full_layout.position_ids[seg.video_start:seg.video_stop, 0].reshape(full_t, frame_rows)[:, 0]
    origin = float(vpos[0])
    start_t = float(vpos[v0]) - origin
    if v1 < full_t:
        stop_t = float(vpos[v1]) - origin
    else:
        from comfy.ldm.minimax.model import _video_t_spans
        stop_t = float(vpos[-1]) + float(_video_t_spans(full_t)[-1]) - origin
    centre_t = 0.5 * (start_t + stop_t)
    if centre_t <= TRAINED_MAX_TIME:
        return 0.0
    cycles = min(max_cycles, max(0, math.ceil((centre_t - TRAINED_MAX_TIME) / CYCLE_TIME_SPAN - 1e-9)))
    return cycles * CYCLE_TIME_SPAN


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
    position_mode: str = "global",
):
    """Build a local H3 PackedLayout, then transplant global or cycle-shifted target positions.

    This preserves multimodal A/V RoPE alignment while allowing the model
    to process a smaller synchronized A/V window.
    """
    from comfy.ldm.minimax.model import PackedLayout

    if position_mode not in POSITION_MODES:
        raise ValueError(f"position_mode must be one of {POSITION_MODES}, got {position_mode!r}")

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
    # Independently refined text has native per-segment lengths. Preserve the
    # shared refs/keyframes at the template's origin without inserting fake
    # text rows in the local stream.
    for (fa, fb, fk), (la, lb, lk) in zip(full_layout.segments, local.segments):
        if fk != lk:
            raise ValueError("window reference structure differs from the full layout")
        if fk not in ("text", "audio", "video"):
            if fb - fa != lb - la:
                raise ValueError("window reference row count differs from the full layout")
            local.position_ids[la:lb] = full_layout.position_ids[fa:fb]

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

    if position_mode != "global":
        shift = window_position_shift(full_layout, v0, v1, position_mode)
        # A tail audio slice can start a few ticks before its video window. Drop
        # whole cycles from the shift until no target row lands before the
        # origin, where the text/reference prefix lives.
        origin = float(full_layout.position_ids[src.video_start, 0])
        earliest = float(min(local.position_ids[dst.video_start, 0], local.position_ids[dst.audio_start, 0]))
        while shift > 0.0 and earliest - shift < origin - 1e-4:
            shift -= CYCLE_TIME_SPAN
        shift = max(shift, 0.0)
        if shift != 0.0:
            local.position_ids[dst.video_start:dst.video_stop, 0] -= shift
            local.position_ids[dst.audio_start:dst.audio_stop, 0] -= shift

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

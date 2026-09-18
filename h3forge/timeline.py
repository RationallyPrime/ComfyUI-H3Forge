"""Derive a context-window plan from a clip length instead of asking for one.

The rule prefers the fewest windows that fit under a cap, then spreads the clip
evenly across them: every window is the same size and none is larger than the
count requires, which is what bounds peak VRAM for a given window count. The
cap defaults to the top of H3's trained range, so each window is a clip the
model has seen the length of.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .context import window_starts

FPS = 24
FRAMES_PER_CYCLE = 17
LATENTS_PER_CYCLE = 5
TRAINED_MAX_SECONDS = 15.0  # 362 frames, 107 latents: the top of H3's trained range
OVERLAP_FRACTION = 0.12
MIN_OVERLAP = 8
MAX_OVERLAP = 16
MIN_STRIDE = 3  # the context node's stagger floor
# Smallest cap with a cadence-aligned window that still leaves MIN_STRIDE past the overlap.
MIN_CAP = math.ceil((MIN_OVERLAP + MIN_STRIDE) / LATENTS_PER_CYCLE) * LATENTS_PER_CYCLE


def align_frame_count(n: int) -> int:
    """Snap a frame count up to H3's ``17k + 5`` grid, as native core does."""
    n = max(5, int(n))
    while n % FRAMES_PER_CYCLE != LATENTS_PER_CYCLE:
        n += 1
    return n


def video_latent_t(frame_count: int) -> int:
    """Native core's frame-count to video-latent mapping."""
    return 2 if frame_count <= 5 else ((frame_count - 5) // FRAMES_PER_CYCLE) * LATENTS_PER_CYCLE + 2


def frames_for_seconds(seconds: float) -> int:
    """Frame count for a duration, snapped up to the grid."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"duration must be a positive number of seconds, got {seconds!r}")
    return align_frame_count(round(seconds * FPS))


def latents_for_seconds(seconds: float) -> int:
    return video_latent_t(frames_for_seconds(seconds))


@dataclass(frozen=True)
class WindowPlan:
    latent_t: int
    window: int
    overlap: int
    count: int

    @property
    def windowed(self) -> bool:
        return self.count > 1


def plan_windows(latent_t: int, max_window: int) -> WindowPlan:
    """Fewest windows under ``max_window``, spread evenly and snapped to the cadence.

    ``overlap`` is a fixed fraction of the cap clamped to ``[MIN_OVERLAP,
    MAX_OVERLAP]``, so the blend region does not shrink when the clip is short
    of the cap. ``window`` is the clip divided across ``count`` windows with
    that overlap, rounded up to the 5-latent cadence so window starts land on
    cycle boundaries, then clamped to the clip. The cap is a ceiling for the
    count decision; the even spread may land a little below it.
    """
    latent_t, max_window = int(latent_t), int(max_window)
    if latent_t < 2:
        raise ValueError(f"latent length must be at least 2, got {latent_t}")
    if max_window < MIN_CAP:
        raise ValueError(f"max_window must be at least {MIN_CAP} latents, got {max_window}")
    if latent_t <= max_window:
        return WindowPlan(latent_t, latent_t, 0, 1)

    overlap = min(MAX_OVERLAP, max(MIN_OVERLAP, round(max_window * OVERLAP_FRACTION)))
    count = math.ceil((latent_t - overlap) / (max_window - overlap))
    while True:
        window = math.ceil((latent_t + (count - 1) * overlap) / count)
        window = min(latent_t, math.ceil(window / LATENTS_PER_CYCLE) * LATENTS_PER_CYCLE)
        if window <= max_window:
            break
        count += 1  # the cadence round-up crossed the cap; one more window brings it back under
    if window - overlap < MIN_STRIDE:
        raise ValueError(f"cannot plan {latent_t} latents under a {max_window}-latent cap with overlap {overlap}")
    # The even spread can leave the last stride short enough that the real
    # planner needs one window fewer; report what it will actually run.
    count = len(window_starts(latent_t, window, overlap))
    return WindowPlan(latent_t, window, overlap, count)


def plan_for_seconds(seconds: float, max_window_seconds: float = TRAINED_MAX_SECONDS) -> WindowPlan:
    """Plan from a clip duration and a window cap, both in seconds."""
    if not math.isfinite(max_window_seconds) or max_window_seconds <= 0:
        raise ValueError(f"max_window_seconds must be a positive number, got {max_window_seconds!r}")
    return plan_windows(latents_for_seconds(seconds), latents_for_seconds(max_window_seconds))

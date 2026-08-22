"""Context-window layout math against Ref2VA/I2VA/FL2VA-style packed layouts.

These use the fake ``comfy.ldm.minimax.model`` in ``fake_minimax``: the payload
prefixes (reference rows, keyframe rows) shift every target segment boundary,
which is exactly the structure the window transplant has to survive.
"""
import pytest
import torch

import fake_minimax
from fake_minimax import PackedLayout


@pytest.mark.parametrize(
    "keyframes,refs",
    [
        pytest.param(None, None, id="t2va"),
        pytest.param(1, None, id="i2va-first-frame"),
        pytest.param(2, None, id="fl2va-first-last"),
        pytest.param(None, 3, id="ref2va-reference-rows"),
        pytest.param(1, 2, id="keyframes-plus-refs"),
    ],
)
def test_clone_window_layout_transplants_global_positions(monkeypatch, keyframes, refs):
    fake_minimax.install(monkeypatch)
    from h3forge.layout import audio_range_for_video_window, clone_window_layout, target_segments

    full = PackedLayout(7, 10, 4, 4, 57, keyframes=keyframes, refs=refs)
    v0, v1 = 2, 7
    a0, a1 = audio_range_for_video_window(full, v0, v1)
    # Frame 2 starts at H3 tick 8.33 and frame 7 at 36.67, so the physically
    # overlapping audio-latent interval is [8, 37) regardless of the prefix.
    assert (a0, a1) == (8, 37)

    local = clone_window_layout(
        full_layout=full, text_len=7, video_shape=(v1 - v0, 4, 4), audio_t=57,
        video_range=(v0, v1), audio_range=(a0, a1), keyframes=keyframes, refs=refs,
    )
    src, dst = target_segments(full), target_segments(local)
    frame_rows = full.frame_rows

    assert torch.equal(
        local.position_ids[dst.video_start:dst.video_stop],
        full.position_ids[src.video_start + v0 * frame_rows:src.video_start + v1 * frame_rows],
    )
    audio_len = a1 - a0
    left = local.position_ids[dst.audio_start:dst.audio_start + audio_len]
    right = local.position_ids[dst.audio_start + audio_len:dst.audio_stop]
    assert torch.equal(left, full.position_ids[src.audio_start + a0:src.audio_start + a1])
    assert torch.equal(
        right,
        full.position_ids[src.audio_start + 57 + a0:src.audio_start + 57 + a1],
    )
    assert local._h3forge_video_offset == v0
    assert local._h3forge_audio_offset == a0
    # The local prefix mirrors the full prefix structure.
    assert [s[2] for s in local.segments] == [s[2] for s in full.segments]


@pytest.mark.parametrize("keyframes,refs", [(None, None), (2, None), (None, 3)])
def test_tail_window_audio_range_reaches_the_final_audio_latent(monkeypatch, keyframes, refs):
    fake_minimax.install(monkeypatch)
    from h3forge.layout import audio_range_for_video_window

    full = PackedLayout(7, 10, 4, 4, 57, keyframes=keyframes, refs=refs)
    a0, a1 = audio_range_for_video_window(full, 5, 10)
    # Frame 5 starts at tick 28.33; the last frame spans through tick 56.67.
    assert (a0, a1) == (28, 57)


def test_leading_window_audio_range_starts_at_zero(monkeypatch):
    fake_minimax.install(monkeypatch)
    from h3forge.layout import audio_range_for_video_window

    full = PackedLayout(7, 10, 4, 4, 57, keyframes=1, refs=2)
    a0, a1 = audio_range_for_video_window(full, 0, 5)
    assert a0 == 0
    # Frame 5 starts at tick 28.33, so the interval must cover through latent 28.
    assert a1 == 29


def test_windowed_audio_ranges_cover_all_audio_latents(monkeypatch):
    fake_minimax.install(monkeypatch)
    from h3forge.context import window_starts
    from h3forge.layout import audio_range_for_video_window

    full = PackedLayout(7, 20, 4, 4, 114, keyframes=1, refs=2)
    starts = window_starts(20, 8, 2, 0)
    covered = set()
    for v0 in starts:
        a0, a1 = audio_range_for_video_window(full, v0, min(v0 + 8, 20))
        covered.update(range(a0, a1))
    assert covered == set(range(114))

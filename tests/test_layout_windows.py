"""Context-window layout math against Ref2VA/I2VA/FL2VA-style packed layouts.

These use the fake ``comfy.ldm.minimax.model`` in ``fake_minimax``: the payload
prefixes (reference rows, keyframe rows) shift every target segment boundary,
which is exactly the structure the window transplant has to survive.
"""
import pytest
import torch

import fake_minimax
from fake_minimax import PackedLayout


def keyframe_descriptors(count):
    return [{"resolved_frame_index": i * 27, "latent": torch.zeros(1, 24, 1, 4, 4)}
            for i in range(count)]


def reference_descriptors(count):
    return [{"kind": "image", "latent_h": 2, "latent_w": 2} for _ in range(count)]


@pytest.mark.parametrize(
    "keyframes,refs",
    [
        pytest.param(None, None, id="t2va"),
        pytest.param(keyframe_descriptors(1), None, id="i2va-first-frame"),
        pytest.param(keyframe_descriptors(2), None, id="fl2va-first-last"),
        pytest.param(None, reference_descriptors(3), id="ref2va-reference-rows"),
        pytest.param(keyframe_descriptors(1), reference_descriptors(2), id="keyframes-plus-refs"),
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
    # The prefix keeps exactly the global coordinates it already had. Its
    # distance to any retained target is identical to the full-run distance,
    # including off-cadence windows and far-future keyframe anchors.
    assert torch.equal(local.position_ids[:dst.audio_start], full.position_ids[:src.audio_start])
    assert torch.equal(
        local.position_ids[dst.audio_start] - local.position_ids[:dst.audio_start],
        full.position_ids[src.audio_start + a0] - full.position_ids[:src.audio_start],
    )
    # The local prefix mirrors the full prefix structure.
    assert [s[2] for s in local.segments] == [s[2] for s in full.segments]


@pytest.mark.parametrize("keyframes,refs", [(None, None), (keyframe_descriptors(2), None), (None, reference_descriptors(3))])
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

    full = PackedLayout(7, 10, 4, 4, 57, keyframes=keyframe_descriptors(1), refs=reference_descriptors(2))
    a0, a1 = audio_range_for_video_window(full, 0, 5)
    assert a0 == 0
    # Frame 5 starts at tick 28.33, so the interval must cover through latent 28.
    assert a1 == 29


def test_windowed_audio_ranges_cover_all_audio_latents(monkeypatch):
    fake_minimax.install(monkeypatch)
    from h3forge.context import window_starts
    from h3forge.layout import audio_range_for_video_window

    full = PackedLayout(7, 20, 4, 4, 114, keyframes=keyframe_descriptors(1), refs=reference_descriptors(2))
    starts = window_starts(20, 8, 2, 0)
    covered = set()
    for v0 in starts:
        a0, a1 = audio_range_for_video_window(full, v0, min(v0 + 8, 20))
        covered.update(range(a0, a1))
    assert covered == set(range(114))


def test_audio_and_video_share_40hz_time_after_the_prefix(monkeypatch):
    fake_minimax.install(monkeypatch)
    from h3forge.attention import _video_cumulative_time
    from h3forge.layout import audio_range_for_video_window, target_segments

    full = PackedLayout(12, 427, 4, 4, 2417,
                        refs=[{"kind": "audio", "ref_audio_t": 40}])
    seg = target_segments(full)
    origin = 52.0
    audio = full.position_ids[seg.audio_start:seg.audio_start + 2417, 0]
    video = full.position_ids[seg.video_start:seg.video_stop:full.frame_rows, 0]
    assert torch.equal(audio[:8], origin + torch.arange(8, dtype=torch.float64))
    assert torch.allclose(video - origin, _video_cumulative_time(torch.arange(427)).double())
    # Eighty video latents span 272 output frames = 11.333s = 453.333
    # audio ticks. Overlap at the boundaries needs 454 or 455 audio latents.
    for start in (0, 69, 139, 208, 278, 347):
        a0, a1 = audio_range_for_video_window(full, start, start + 80)
        assert a1 - a0 in (454, 455)
    assert audio_range_for_video_window(full, 347, 427)[1] == 2417


def test_unequal_pipe_encodings_keep_one_global_prefix_origin(monkeypatch):
    fake_minimax.install(monkeypatch)
    from h3forge.layout import audio_range_for_video_window, clone_window_layout, target_segments
    from h3forge.prompt import combine_conditioning_segments

    combined = combine_conditioning_segments([[[torch.zeros(1, 7, 4), {}]],
                                              [[torch.zeros(1, 12, 4), {}]]])
    primary, metadata = combined[0]
    refs = [{"kind": "audio", "ref_audio_t": 40}]
    keyframes = keyframe_descriptors(2)
    full = PackedLayout(primary.shape[1], 20, 4, 4, 114, refs=refs, keyframes=keyframes)
    src = target_segments(full)
    for start, context in zip((0, 10), metadata["h3forge_prompt_segments"]):
        assert context.shape[1] == primary.shape[1] == 12
        ar = audio_range_for_video_window(full, start, start + 10)
        local = clone_window_layout(full_layout=full, text_len=context.shape[1],
                                    video_shape=(10, 4, 4), audio_t=114,
                                    video_range=(start, start + 10), audio_range=ar,
                                    refs=refs, keyframes=keyframes)
        dst = target_segments(local)
        assert torch.equal(local.position_ids[:dst.audio_start], full.position_ids[:src.audio_start])

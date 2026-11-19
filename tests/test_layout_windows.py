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
    full = PackedLayout(12, 20, 4, 4, 114, refs=refs, keyframes=keyframes)
    src = target_segments(full)
    for start, context in zip((0, 10), metadata["h3forge_prompt_segments"]):
        assert context.shape[1] in (7, 12)
        ar = audio_range_for_video_window(full, start, start + 10)
        local = clone_window_layout(full_layout=full, text_len=context.shape[1],
                                    video_shape=(10, 4, 4), audio_t=114,
                                    video_range=(start, start + 10), audio_range=ar,
                                    refs=refs, keyframes=keyframes)
        dst = target_segments(local)
        assert torch.equal(local.position_ids[context.shape[1]:dst.audio_start], full.position_ids[12:src.audio_start])


@pytest.mark.parametrize("total,window,expected", [
    (427, 80, 455),
    (67, 25, 143),
    (20, 15, 86),
    (20, 6, 36),
    (10, 10, 57),
])
def test_audio_window_length_is_constant_across_all_starts(monkeypatch, total, window, expected):
    fake_minimax.install(monkeypatch)
    from h3forge.layout import audio_range_for_video_window, audio_window_length, clone_window_layout

    frames = sum((1, 4, 4, 4, 4)[i % 5] for i in range(total))
    audio_t = round(frames * 5 / 3)
    full = PackedLayout(7, total, 4, 4, audio_t, keyframes=keyframe_descriptors(1), refs=reference_descriptors(1))
    audio_win = audio_window_length(full, window)
    assert audio_win == expected
    signatures = set()
    for v0 in range(total - window + 1):
        a0, a1 = audio_range_for_video_window(full, v0, v0 + window)
        in_a0 = min(a0, audio_t - audio_win)
        in_a1 = in_a0 + audio_win
        assert 0 <= in_a0 <= a0 < a1 <= in_a1 <= audio_t
        assert in_a1 - in_a0 == audio_win
        local = clone_window_layout(
            full_layout=full, text_len=7, video_shape=(window, 4, 4), audio_t=audio_t,
            video_range=(v0, v0 + window), audio_range=(in_a0, in_a1),
            keyframes=keyframe_descriptors(1), refs=reference_descriptors(1),
        )
        signatures.add(local.signature)
    assert signatures == {(7, window, 4, 4, audio_win)}
    for bad in (0, -1, total + 1):
        with pytest.raises(ValueError, match="video_window"):
            audio_window_length(full, bad)


@pytest.mark.parametrize("refs", [None, [{"kind": "image", "latent_h": 2, "latent_w": 2}, {"kind": "audio", "ref_audio_t": 40}]])
@pytest.mark.parametrize("v0", [0, 1, 2, 3, 4, 58, 60, 117])
def test_rebased_positions_start_at_cycle_offset_and_keep_av_alignment(monkeypatch, refs, v0):
    fake_minimax.install(monkeypatch)
    from fake_minimax import video_frame_times
    from h3forge.layout import (
        CYCLE_TIME_SPAN,
        audio_range_for_video_window,
        audio_window_length,
        clone_window_layout,
        target_segments,
        window_position_shift,
    )

    total, window, audio_t = 202, 85, 1142
    full = PackedLayout(12, total, 4, 4, audio_t, refs=refs)
    src = target_segments(full)
    origin = float(full.position_ids[src.video_start, 0])
    audio_win = audio_window_length(full, window)

    v1 = v0 + window
    a0, _ = audio_range_for_video_window(full, v0, v1)
    in_a0 = min(a0, audio_t - audio_win)
    in_a1 = in_a0 + audio_win

    local_global = clone_window_layout(
        full_layout=full, text_len=12, video_shape=(window, 4, 4), audio_t=audio_t,
        video_range=(v0, v1), audio_range=(in_a0, in_a1), refs=refs, position_mode="global",
    )
    local_rebased = clone_window_layout(
        full_layout=full, text_len=12, video_shape=(window, 4, 4), audio_t=audio_t,
        video_range=(v0, v1), audio_range=(in_a0, in_a1), refs=refs, position_mode="rebased",
    )
    dst = target_segments(local_rebased)
    shift = window_position_shift(full, v0, v1, "rebased")
    assert shift == pytest.approx((v0 // 5) * CYCLE_TIME_SPAN)

    # Video starts at target origin plus its offset inside the 5-latent cycle.
    expected_cycle_offset = video_frame_times(5)[v0 % 5]
    rebased_vpos = local_rebased.position_ids[dst.video_start:dst.video_stop:full.frame_rows, 0]
    global_vpos = local_global.position_ids[dst.video_start:dst.video_stop:full.frame_rows, 0]
    assert float(rebased_vpos[0]) - origin == pytest.approx(expected_cycle_offset, abs=1e-11)
    assert torch.allclose(rebased_vpos, global_vpos - shift, atol=1e-11, rtol=0.0)

    # Audio shifts by the exact same amount on both stereo channels, preserving A/V alignment.
    rebased_apos = local_rebased.position_ids[dst.audio_start:dst.audio_stop, 0]
    global_apos = local_global.position_ids[dst.audio_start:dst.audio_stop, 0]
    assert torch.allclose(rebased_apos, global_apos - shift, atol=1e-11, rtol=0.0)
    assert torch.allclose(
        rebased_vpos[:, None] - rebased_apos[None, :audio_win],
        global_vpos[:, None] - global_apos[None, :audio_win],
        atol=1e-11,
        rtol=0.0,
    )
    assert torch.equal(rebased_apos[:audio_win], rebased_apos[audio_win:])

    # Prefix rows and spatial coordinates stay untouched, and sparse offsets stay global.
    assert torch.equal(local_rebased.position_ids[:dst.audio_start], local_global.position_ids[:dst.audio_start])
    assert torch.equal(local_rebased.position_ids[:, 1:], local_global.position_ids[:, 1:])
    assert local_rebased._h3forge_video_offset == v0
    assert local_rebased._h3forge_audio_offset == in_a0


def test_wrapped_positions_keep_global_inside_trained_range_and_wrap_centre_beyond(monkeypatch):
    fake_minimax.install(monkeypatch)
    from h3forge.layout import (
        CYCLE_TIME_SPAN,
        TRAINED_MAX_TIME,
        audio_range_for_video_window,
        audio_window_length,
        clone_window_layout,
        target_segments,
        window_position_shift,
    )

    # 202-latent clip (Battle Brief): windows 0 and 1 have centres inside trained range; window 2 wraps.
    full = PackedLayout(10, 202, 4, 4, 1142, refs=[{"kind": "image", "latent_h": 2, "latent_w": 2}])
    src = target_segments(full)
    origin = float(full.position_ids[src.video_start, 0])
    audio_win = audio_window_length(full, 85)

    for v0, expect_shifted in ((0, False), (60, False), (117, True)):
        v1 = v0 + 85
        a0, _ = audio_range_for_video_window(full, v0, v1)
        in_a0 = min(a0, 1142 - audio_win)
        in_a1 = in_a0 + audio_win
        shift = window_position_shift(full, v0, v1, "wrapped")
        local_g = clone_window_layout(
            full_layout=full, text_len=10, video_shape=(85, 4, 4), audio_t=1142,
            video_range=(v0, v1), audio_range=(in_a0, in_a1),
            refs=[{"kind": "image", "latent_h": 2, "latent_w": 2}], position_mode="global",
        )
        local_w = clone_window_layout(
            full_layout=full, text_len=10, video_shape=(85, 4, 4), audio_t=1142,
            video_range=(v0, v1), audio_range=(in_a0, in_a1),
            refs=[{"kind": "image", "latent_h": 2, "latent_w": 2}], position_mode="wrapped",
        )
        dst = target_segments(local_w)
        if not expect_shifted:
            assert shift == 0.0
            assert torch.equal(local_w.position_ids, local_g.position_ids)
        else:
            cycles = round(shift / CYCLE_TIME_SPAN)
            assert cycles > 0 and shift == pytest.approx(cycles * CYCLE_TIME_SPAN)
            vpos = local_w.position_ids[dst.video_start:dst.video_stop:full.frame_rows, 0] - origin
            apos = local_w.position_ids[dst.audio_start:dst.audio_stop, 0] - origin
            centre = 0.5 * (float(vpos[0]) + float(vpos[-1]))
            assert 0.0 <= float(vpos[0]) and centre <= TRAINED_MAX_TIME
            assert torch.allclose(
                local_w.position_ids[dst.video_start:dst.video_stop, 0],
                local_g.position_ids[dst.video_start:dst.video_stop, 0] - shift,
                atol=1e-11, rtol=0.0,
            )
            assert torch.allclose(
                local_w.position_ids[dst.audio_start:dst.audio_stop, 0],
                local_g.position_ids[dst.audio_start:dst.audio_stop, 0] - shift,
                atol=1e-11, rtol=0.0,
            )
            assert float(apos[0]) == pytest.approx(
                float(local_g.position_ids[dst.audio_start, 0]) - origin - shift, abs=1e-11
            )

    # 60s clip (427 latents, window 100): every window's wrapped centre lies inside trained range.
    long_full = PackedLayout(8, 427, 4, 4, 2417)
    long_src = target_segments(long_full)
    long_origin = float(long_full.position_ids[long_src.video_start, 0])
    for v0 in (0, 80, 165, 245, 327):
        shift = window_position_shift(long_full, v0, v0 + 100, "wrapped")
        vpos = long_full.position_ids[long_src.video_start + v0 * long_full.frame_rows:
                                      long_src.video_start + (v0 + 100) * long_full.frame_rows:long_full.frame_rows, 0]
        shifted = vpos - long_origin - shift
        assert float(shifted[0]) >= -1e-11
        assert 0.5 * (float(shifted[0]) + float(shifted[-1])) <= TRAINED_MAX_TIME + 1e-9


def test_sparse_block_mask_keeps_working_with_rebased_and_wrapped_layouts(monkeypatch):
    fake_minimax.install(monkeypatch)
    from h3forge.attention import _make_block_mask
    from h3forge.layout import audio_range_for_video_window, audio_window_length, clone_window_layout
    from h3forge.state import AttentionPolicy, RuntimeState

    calls = []

    def fake_create_block_mask(mask_mod, **kwargs):
        calls.append({"mask_mod": mask_mod, **kwargs})
        return object()

    import torch.nn.attention.flex_attention as flex_module
    monkeypatch.setattr(flex_module, "create_block_mask", fake_create_block_mask)
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)

    full = PackedLayout(8, 202, 4, 4, 1142)
    audio_win = audio_window_length(full, 85)
    v0, v1 = 117, 202
    a0, _ = audio_range_for_video_window(full, v0, v1)
    in_a0 = min(a0, 1142 - audio_win)
    in_a1 = in_a0 + audio_win

    state = RuntimeState(AttentionPolicy())
    masks = {}
    for mode in ("global", "rebased", "wrapped"):
        layout = clone_window_layout(
            full_layout=full, text_len=8, video_shape=(85, 4, 4), audio_t=1142,
            video_range=(v0, v1), audio_range=(in_a0, in_a1), position_mode=mode,
        )
        assert layout._h3forge_video_offset == v0
        assert layout._h3forge_audio_offset == in_a0
        state.layout = layout
        q = torch.zeros(1, 2, layout.seq_len, 8)
        masks[mode] = _make_block_mask(state, q, device=q.device)

    # Preserved global offsets mean all three modes share the exact same compiled sparse mask.
    assert masks["rebased"] is masks["global"]
    assert masks["wrapped"] is masks["global"]
    assert len(calls) == 1
    assert (state.mask_misses, state.mask_hits) == (1, 2)

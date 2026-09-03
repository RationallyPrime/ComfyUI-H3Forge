import pytest
import torch

from h3forge.context import (
    assert_full_coverage,
    audio_overlap_frames,
    blend_weights,
    context_plan_summary,
    window_starts,
)
from h3forge.layout import expand_audio_range, padded_spatial_shape
from h3forge.prompt import (
    combine_conditioning_segments,
    encode_pipe_prompt,
    make_segmented_extra_conds,
    pad_segment_contexts,
    parse_segment_durations,
    select_segment_index,
    split_pipe_prompt,
    unreachable_segments,
)


def test_windows_cover_full_range():
    starts = window_starts(61, 25, 5, 0)
    covered = set()
    for s in starts:
        covered.update(range(s, min(s + 25, 61)))
    assert covered == set(range(61))
    assert starts[0] == 0
    assert starts[-1] == 36


def test_phase_still_covers_full_range():
    starts = window_starts(61, 25, 5, 7)
    covered = set()
    for s in starts:
        covered.update(range(s, min(s + 25, 61)))
    assert covered == set(range(61))


def test_blend_positive():
    w = blend_weights(25, 5, device="cpu", dtype=torch.float32)
    assert torch.all(w > 0)
    assert w[0] < w[5]
    assert w[-1] < w[-6]


def test_blend_edges_keep_full_weight_at_timeline_boundaries():
    first = blend_weights(25, 5, device="cpu", dtype=torch.float32, ramp_start=False)
    assert torch.all(first[:5] == 1.0)
    assert first[-1] < first[-6]
    last = blend_weights(25, 5, device="cpu", dtype=torch.float32, ramp_end=False)
    assert torch.all(last[-5:] == 1.0)
    assert last[0] < last[5]
    only = blend_weights(25, 5, device="cpu", dtype=torch.float32, ramp_start=False, ramp_end=False)
    assert torch.all(only == 1.0)


def test_audio_overlap_is_zero_when_video_windows_do_not_overlap():
    assert audio_overlap_frames(0, 25, 142) == 0
    assert audio_overlap_frames(5, 25, 142) == round(5 * 142 / 25)
    assert audio_overlap_frames(1, 25, 12) == 1


def test_strict_coverage_assertion():
    video_den = torch.ones(1, 1, 3, 1, 1)
    audio_den = torch.ones(1, 1, 1, 4)
    assert_full_coverage(video_den, audio_den)
    holed = video_den.clone()
    holed[0, 0, 1] = 0.0
    with pytest.raises(RuntimeError, match="video"):
        assert_full_coverage(holed, audio_den)
    holed_audio = audio_den.clone()
    holed_audio[..., 2] = 0.0
    with pytest.raises(RuntimeError, match="audio"):
        assert_full_coverage(video_den, holed_audio)


def test_odd_latent_spatial_shape_uses_post_pad_dimensions():
    assert padded_spatial_shape(95, 167, (1, 2, 2)) == (96, 168)


def test_audio_ranges_can_be_pinned_to_one_compiled_length():
    ranges = [(0, 141), (139, 282), (284, 426)]
    expanded = [expand_audio_range(r, total=426, target_length=143) for r in ranges]
    assert {a1 - a0 for a0, a1 in expanded} == {143}
    for old, new in zip(ranges, expanded):
        assert new[0] <= old[0] < old[1] <= new[1]


def test_pipe_prompt_split_and_escape():
    assert split_pipe_prompt(r"first | second with a \| literal | third") == [
        "first", "second with a | literal", "third",
    ]


def test_pipe_prompt_rejects_empty_segments():
    with pytest.raises(ValueError, match="empty segment"):
        split_pipe_prompt("first || third")


def test_segment_durations_are_exact_and_fail_closed():
    assert parse_segment_durations("2, 18\n40", 3) == (2.0, 18.0, 40.0)
    assert parse_segment_durations("", 3) == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="exactly one"):
        parse_segment_durations("2,18", 3)
    with pytest.raises(ValueError, match="greater than zero"):
        parse_segment_durations("2,0,40", 3)
    with pytest.raises(ValueError, match="greater than zero"):
        parse_segment_durations("2,nan,40", 3)


def test_segment_contexts_pad_to_one_compiled_shape():
    contexts = [torch.ones(1, 2, 3), torch.full((1, 4, 3), 2.0)]
    padded = pad_segment_contexts(contexts)
    assert [tuple(context.shape) for context in padded] == [(1, 4, 3), (1, 4, 3)]
    assert torch.equal(padded[0][:, :2], contexts[0])
    assert torch.count_nonzero(padded[0][:, 2:]) == 0


def test_segment_selection_is_hard_and_midpoint_based():
    # Interior window fully inside segment 1.
    assert select_segment_index(20, 40, total=60, count=3) == 1
    # Boundary-crossing windows use the segment containing their midpoint;
    # hidden states are never mixed.
    assert select_segment_index(0, 25, total=60, count=3) == 0
    assert select_segment_index(10, 28, total=60, count=3) == 0
    assert select_segment_index(16, 30, total=60, count=3) == 1
    assert select_segment_index(36, 60, total=60, count=3) == 2
    assert select_segment_index(0, 60, total=60, count=1) == 0
    with pytest.raises(ValueError, match="invalid window"):
        select_segment_index(5, 5, 10, 2)
    with pytest.raises(ValueError, match="positive"):
        select_segment_index(0, 5, 10, 0)


def test_segment_selection_respects_unequal_durations():
    durations = (2.0, 18.0, 40.0)
    # On a 60-latent timeline, the requested boundaries are exactly 2 and 20.
    assert select_segment_index(0, 2, 60, 3, durations) == 0
    assert select_segment_index(1, 3, 60, 3, durations) == 1
    assert select_segment_index(10, 20, 60, 3, durations) == 1
    assert select_segment_index(20, 40, 60, 3, durations) == 2
    with pytest.raises(ValueError, match="expected 3"):
        select_segment_index(0, 2, 60, 3, (1.0, 2.0))


def test_segment_selection_is_scale_safe_for_huge_durations():
    durations = (1e307, 1e307)
    assert select_segment_index(0, 50, 100, 2, durations) == 0
    assert select_segment_index(50, 100, 100, 2, durations) == 1


def test_segment_selection_is_scale_invariant():
    # Durations are unitless ratios, so a huge or tiny finite scale must route
    # exactly like its reduced form, including when sum(weights) itself would
    # overflow in float.
    windows = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60)]
    for scaled in ((1e307, 1e307), (1e308, 1e308), (1e-307, 1e-307)):
        assert [select_segment_index(v0, v1, 60, 2, scaled) for v0, v1 in windows] == [
            select_segment_index(v0, v1, 60, 2, (1.0, 1.0)) for v0, v1 in windows
        ]


def test_segment_selection_ties_are_exact():
    # A midpoint sitting exactly on a duration boundary belongs to the later
    # segment (strict less-than), independent of float rounding: with (1,1,10)
    # on 60 latents the boundaries are exactly 5 and 10.
    durations = (1.0, 1.0, 10.0)
    assert select_segment_index(0, 10, 60, 3, durations) == 1
    assert select_segment_index(1, 9, 60, 3, durations) == 1
    assert select_segment_index(0, 20, 60, 3, durations) == 2
    assert select_segment_index(4, 6, 60, 3, durations) == 1
    assert select_segment_index(3, 6, 60, 3, durations) == 0


def test_unreachable_segments_detected_when_segments_outnumber_windows():
    # 30 latents with the default 25/5 policy give windows [0,25) and [5,30);
    # both midpoints land in segment 2 of 3, so segments 1 and 3 would vanish.
    starts = window_starts(30, 25, 5, 0)
    assert starts == [0, 5]
    assert unreachable_segments(starts, 25, 30, 3) == [0, 2]
    # With enough windows every segment is selected somewhere.
    assert unreachable_segments(window_starts(60, 25, 5, 0), 25, 60, 3) == []
    assert unreachable_segments(window_starts(30, 25, 5, 0), 25, 30, 1) == []


def test_context_plan_reports_work_and_prompt_assignment():
    summary = context_plan_summary(
        total=60,
        starts=[0, 20, 35],
        window=25,
        overlap=5,
        phase=0,
        prompt_count=3,
        prompt_durations=(2.0, 18.0, 40.0),
    )
    assert "windows=3" in summary
    assert "video_latent_visits=1.25x" in summary
    assert "prompt_windows=2x1,3x2" in summary


class _Clip:
    @staticmethod
    def tokenize(text):
        return text

    @staticmethod
    def encode_from_tokens_scheduled(text):
        tokens = len(text.split())
        tags = torch.ones(tokens, dtype=torch.long)
        if text.startswith("img"):
            tags[0] = 2
        meta = {"minimax_token_tags": tags}
        if "extrameta" in text:
            meta["extra_key"] = True
        return [[torch.full((1, tokens, 2), float(tokens)), meta]]


def test_pipe_segments_are_encoded_independently_and_annotated():
    conditioning = encode_pipe_prompt(_Clip(), "short text | a deliberately longer segment")
    context, metadata = conditioning[0]
    assert tuple(context.shape) == (1, 4, 2)
    assert metadata["h3forge_prompt_segment_count"] == 2
    assert [tuple(x.shape) for x in metadata["h3forge_prompt_segments"]] == [(1, 4, 2), (1, 4, 2)]
    assert torch.equal(metadata["minimax_token_tags"], torch.ones(4, dtype=torch.long))


def test_pipe_prompt_repeats_global_anchor_and_carries_durations():
    seen = []

    class RecordingClip(_Clip):
        @staticmethod
        def tokenize(text):
            seen.append(text)
            return text

    conditioning = encode_pipe_prompt(
        RecordingClip(),
        "first action | second action",
        global_prompt="same red coat, 35mm film",
        segment_durations="2, 8",
    )
    assert seen == [
        "same red coat, 35mm film\n\nfirst action",
        "same red coat, 35mm film\n\nsecond action",
    ]
    assert conditioning[0][1]["h3forge_prompt_segment_durations"] == (2.0, 8.0)


def test_reference_conditioning_segments_retain_shared_native_payload():
    reference = {"kind": "image", "latent": torch.ones(1, 2, 3)}
    first = [[
        torch.ones(1, 2, 4),
        {"minimax_token_tags": torch.tensor([2, 1]), "minimax_refs": [reference]},
    ]]
    second = [[
        torch.full((1, 4, 4), 2.0),
        {"minimax_token_tags": torch.tensor([2, 1, 1, 1]), "minimax_refs": [reference]},
    ]]

    combined = combine_conditioning_segments([first, second])
    context, metadata = combined[0]
    assert tuple(context.shape) == (1, 4, 4)
    assert metadata["minimax_refs"] is first[0][1]["minimax_refs"]
    assert metadata["h3forge_prompt_segment_count"] == 2
    assert torch.equal(metadata["h3forge_prompt_segments"][1], second[0][0])


def test_pipe_prompt_rejects_divergent_token_tags():
    with pytest.raises(ValueError, match="token tags"):
        encode_pipe_prompt(_Clip(), "plain words here | img insert segment")


def test_pipe_prompt_rejects_divergent_metadata_keys():
    with pytest.raises(ValueError, match="metadata keys"):
        encode_pipe_prompt(_Clip(), "plain words here | words with extrameta")


def test_segment_contexts_are_carried_through_minimax_payload():
    class Cond:
        def __init__(self, cond):
            self.cond = cond

        def _copy_with(self, cond):
            return Cond(cond)

    class BaseModel:
        @staticmethod
        def get_dtype_inference():
            return torch.float32

    class Diffusion:
        @staticmethod
        def preprocess_text_embeds(context):
            return context + 1

    def base_extra_conds(**kwargs):
        return {"minimax_payload": Cond({"seed": kwargs.get("seed", 0)})}

    wrapper = make_segmented_extra_conds(base_extra_conds, BaseModel(), Diffusion())
    raw = (torch.zeros(1, 3, 2), torch.ones(1, 3, 2))
    result = wrapper(
        device="cpu",
        seed=7,
        h3forge_prompt_segments=raw,
        h3forge_prompt_segment_durations=(2.0, 8.0),
    )
    payload = result["minimax_payload"].cond
    assert payload["seed"] == 7
    assert torch.equal(payload["h3forge_prompt_segments"][0], torch.ones_like(raw[0]))
    assert torch.equal(payload["h3forge_prompt_segments"][1], torch.full_like(raw[1], 2))
    assert payload["h3forge_prompt_segment_durations"] == (2.0, 8.0)

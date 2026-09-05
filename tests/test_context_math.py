from itertools import pairwise

import pytest
import torch

from h3forge.context import (
    ContextPolicy,
    assert_full_coverage,
    audio_overlap_frames,
    blend_weights,
    context_plan_summary,
    make_context_wrapper,
    max_stagger_phase,
    ordered_halving,
    stagger_phase,
    window_starts,
)
from h3forge.layout import padded_spatial_shape
from h3forge.prompt import (
    combine_conditioning_segments,
    encode_pipe_prompt,
    make_segmented_extra_conds,
    parse_segment_durations,
    split_pipe_prompt,
    segment_ranges,
    validate_segment_delimiter,
)


def test_window_starts_use_even_spacing_without_a_near_duplicate_tail():
    starts = window_starts(427, 112, 6)
    overlaps = [112 - (right - left) for left, right in pairwise(starts)]
    assert starts == [0, 105, 210, 315]
    assert len(starts) == 4
    assert starts[0] == 0
    assert starts[-1] == 315
    assert all(value >= 6 for value in overlaps)

    starts = window_starts(427, 107, 6)
    overlaps = [107 - (right - left) for left, right in pairwise(starts)]
    assert starts == [0, 80, 160, 240, 320]
    assert len(starts) == 5
    assert max(overlaps) - min(overlaps) <= 1
    assert all(value < 50 for value in overlaps)


def test_phase_moves_only_interior_starts_and_keeps_fixed_anchors():
    base = window_starts(427, 112, 6, 0)
    shifted = window_starts(427, 112, 6, 53)
    assert shifted[0] == base[0] == 0
    assert shifted[-1] == base[-1] == 315
    assert all(after > before for before, after in zip(base[1:-1], shifted[1:-1]))
    assert window_starts(427, 107, 6, 20) == [0, 100, 180, 260, 320]


@pytest.mark.parametrize("phase", [-1, 101, 1.5])
def test_phase_must_be_an_integer_inside_the_stride(phase):
    with pytest.raises(ValueError, match="phase must be an integer"):
        window_starts(427, 107, 6, phase)


def test_window_at_least_total_uses_one_start():
    assert window_starts(25, 25, 8, 0) == [0]
    assert window_starts(24, 25, 8, 0) == [0]


@pytest.mark.parametrize("window,overlap", [(80, 10), (80, 11), (107, 6)])
def test_shipped_context_policies_snap_interiors_but_preserve_the_tail(window, overlap):
    starts = window_starts(427, window, overlap, 0)
    assert all(start % 5 == 0 for start in starts[:-1])
    # The tail is exempt: moving it would lose coverage or change length.
    assert starts[-1] == 427 - window
    receipt = context_plan_summary(427, starts, window, overlap, phase=0)
    assert "cadence=5" in receipt
    assert f"off_cadence_starts={int((427 - window) % 5 != 0)}" in receipt


def _stagger_plans(total, window, overlap):
    max_phase = max_stagger_phase(window, overlap)
    return [window_starts(total, window, overlap, phase, max_phase) for phase in range(max_phase + 1)]


@pytest.mark.parametrize("window,overlap", [(80, 10), (80, 11), (107, 6), (25, 8)])
def test_cadence_alignment_does_not_disable_staggering(window, overlap):
    plans = _stagger_plans(427, window, overlap)
    assert len({tuple(starts) for starts in plans}) > 1
    schedule = {tuple(window_starts(427, window, overlap, stagger_phase(step, window, overlap),
                                    max_stagger_phase(window, overlap))) for step in range(32)}
    assert len(schedule) > 1
    for starts in plans:
        count = sum(start % 5 != 0 for start in starts)
        receipt = context_plan_summary(427, starts, window, overlap, phase=1)
        assert f"off_cadence_starts={count}" in receipt


def test_rigid_stagger_keeps_the_off_cadence_spread_instead_of_collapsing():
    # 80/10 at 427 latents has three latents of slack across five seams: the
    # only fully aligned plan is the upper feasible wall, and every active phase
    # shifted from it clamps back onto it. A static run takes the aligned plan;
    # a staggering run keeps the even spread and its three distinct layouts.
    assert window_starts(427, 80, 10, 0) == [0, 70, 140, 210, 280, 347]
    assert window_starts(427, 80, 10, 0, max_stagger_phase(80, 10)) == [0, 69, 139, 208, 278, 347]
    plans = {tuple(starts) for starts in _stagger_plans(427, 80, 10)}
    assert plans == {
        (0, 69, 139, 208, 278, 347),
        (0, 70, 140, 209, 279, 347),
        (0, 70, 140, 210, 280, 347),
    }
    # 80/11 has room for both: the aligned phase-0 plan survives and every
    # phase still reaches its own layout.
    assert window_starts(427, 80, 11, 0, max_stagger_phase(80, 11)) == [0, 60, 115, 175, 230, 290, 347]
    assert len({tuple(starts) for starts in _stagger_plans(427, 80, 11)}) == 12


@pytest.mark.parametrize("total,window,overlap", [(427, 80, 10), (427, 80, 11), (427, 107, 6), (427, 25, 8), (427, 112, 6)])
def test_no_seam_moves_further_than_its_phase_from_the_phase_zero_plan(total, window, overlap):
    max_phase = max_stagger_phase(window, overlap)
    nominal = window_starts(total, window, overlap, 0, max_phase)
    for phase in range(max_phase + 1):
        starts = window_starts(total, window, overlap, phase, max_phase)
        moved = [after - before for before, after in zip(nominal, starts)]
        assert all(0 <= delta <= phase for delta in moved), (phase, nominal, starts)


@pytest.mark.parametrize("max_phase", [-1, 101, 1.5])
def test_max_phase_must_be_an_integer_inside_the_stride(max_phase):
    with pytest.raises(ValueError, match="max_phase must be an integer"):
        window_starts(427, 107, 6, 0, max_phase)


def test_tight_stride_keeps_unsnappable_starts_and_reports_them():
    starts = window_starts(13, 4, 1)
    assert starts == [0, 3, 6, 9]
    assert "off_cadence_starts=3" in context_plan_summary(13, starts, 4, 1, phase=0)

    # For 25/8, 24 gaps must cover 402 latents. All interior starts on the
    # cadence would allow gaps of at most 15, which cannot reach the tail.
    # Adding windows would change the prompt-segment contract, so some
    # interior starts must remain off cadence as well as the exact tail.
    starts = window_starts(427, 25, 8)
    assert len(starts) == 25
    assert any(start % 5 for start in starts[1:-1])
    count = sum(start % 5 != 0 for start in starts)
    assert f"off_cadence_starts={count}" in context_plan_summary(427, starts, 25, 8, phase=0)


def test_pyramid_blend_uses_the_whole_window_for_odd_and_even_lengths():
    odd = blend_weights(7, 2, device="cpu", dtype=torch.float32, mode="pyramid")
    even = blend_weights(6, 2, device="cpu", dtype=torch.float32, mode="pyramid")
    assert odd.tolist() == [1, 2, 3, 4, 3, 2, 1]
    assert even.tolist() == [1, 2, 3, 3, 2, 1]
    assert torch.equal(
        blend_weights(7, 0, device="cpu", dtype=torch.float32, mode="pyramid",
                      ramp_start=False, ramp_end=False),
        odd,
    )


def test_overlap_linear_preserves_the_old_edge_ramp_and_boundary_handling():
    expected_edge = torch.linspace(1 / 6, 1, 5)
    interior = blend_weights(25, 5, device="cpu", dtype=torch.float32, mode="overlap-linear")
    assert torch.allclose(interior[:5], expected_edge)
    assert torch.allclose(interior[-5:], expected_edge.flip(0))

    first = blend_weights(25, 5, device="cpu", dtype=torch.float32,
                          mode="overlap-linear", ramp_start=False)
    assert torch.all(first[:5] == 1.0)
    assert first[-1] < first[-6]
    last = blend_weights(25, 5, device="cpu", dtype=torch.float32,
                         mode="overlap-linear", ramp_end=False)
    assert torch.all(last[-5:] == 1.0)
    assert last[0] < last[5]
    only = blend_weights(25, 5, device="cpu", dtype=torch.float32,
                         mode="overlap-linear", ramp_start=False, ramp_end=False)
    assert torch.all(only == 1.0)
    clamped = blend_weights(7, 20, device="cpu", dtype=torch.float32, mode="overlap-linear")
    assert torch.allclose(clamped, torch.tensor([0.25, 0.625, 1, 1, 1, 0.625, 0.25]))


def test_blend_rejects_unknown_modes():
    with pytest.raises(ValueError, match="unknown blend mode"):
        blend_weights(7, 2, device="cpu", dtype=torch.float32, mode="mystery")
    assert torch.equal(
        blend_weights(7, 2, device="cpu", dtype=torch.float32, mode="flat"),
        torch.ones(7),
    )


def test_ordered_halving_first_eight_values():
    assert [ordered_halving(i) for i in range(8)] == [
        0, 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875,
    ]


@pytest.mark.parametrize("total,window,overlap", [(427, 80, 10), (427, 80, 11), (427, 107, 6), (427, 25, 8)])
def test_every_phase_keeps_the_requested_overlap_and_window_count(total, window, overlap):
    nominal = window_starts(total, window, overlap, 0)
    for phase in range(window - overlap):
        starts = window_starts(total, window, overlap, phase)
        assert len(starts) == len(nominal)
        assert starts[0] == 0 and starts[-1] == total - window
        overlaps = [window - (right - left) for left, right in pairwise(starts)]
        assert min(overlaps) >= overlap, (phase, starts)


def test_stagger_phase_is_bounded_by_the_overlap_not_the_stride():
    assert max_stagger_phase(80, 10) == 10
    assert max_stagger_phase(80, 0) == 0
    assert max_stagger_phase(25, 20) == 4  # stride 5 -> phases stay inside it
    phases = {stagger_phase(step, 80, 10) for step in range(64)}
    assert phases == set(range(11))
    assert stagger_phase(1, 80, 10) == 5
    assert stagger_phase(0, 80, 10) == 0


def test_segmented_windows_cover_every_beat_without_midpoint_selection(monkeypatch):
    import fake_minimax
    from h3forge.context import _segment_windows
    fake_minimax.install(monkeypatch)
    ranges, _ = segment_ranges(427, 6)
    plan = _segment_windows(427, 80, 10, ranges)
    assert {p[0] for p in plan} == set(range(6))
    for index, (start, stop) in enumerate(ranges):
        covered = set()
        for segment, _, _, lo, hi in plan:
            if segment == index:
                covered.update(range(lo, hi))
        assert covered == set(range(start, stop))


@pytest.mark.parametrize("total,window,overlap", [(427, 112, 6), (427, 80, 28)])
def test_pyramid_blend_covers_every_latent_at_static_and_half_stride_phases(total, window, overlap):
    stride = window - overlap
    for phase in (0, stride // 2):
        accumulated = torch.zeros(total, dtype=torch.float32)
        for start in window_starts(total, window, overlap, phase):
            weights = blend_weights(window, overlap, device="cpu", dtype=torch.float32, mode="pyramid")
            accumulated[start:start + window] += weights
        assert torch.all(accumulated > 0)


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


def test_pipe_prompt_split_and_escape():
    assert split_pipe_prompt(r"first | second with a \| literal | third") == [
        "first", "second with a | literal", "third",
    ]


def test_minimax_special_tokens_are_never_split():
    """<|cutoff|> and friends contain the historical delimiter but are single tokens."""
    assert split_pipe_prompt("say <|cutoff|> now | and rest") == ["say <|cutoff|> now", "and rest"]
    assert split_pipe_prompt("a <|lyrics_start|>la<|lyrics_end|> b") == ["a <|lyrics_start|>la<|lyrics_end|> b"]
    # An unterminated "<|" is not a token, so its pipe stays an ordinary delimiter. That
    # fails visibly on a malformed prompt instead of swallowing the rest into one segment.
    assert split_pipe_prompt("broken <| open | tail") == ["broken <", "open", "tail"]
    # A malformed opener may not borrow a later token's closer: the delimiters between
    # them are still delimiters, and the later token still survives intact.
    assert split_pipe_prompt("first <| broken | second <|cutoff|> | third") == [
        "first <", "broken", "second <|cutoff|>", "third",
    ]
    # Only a bare identifier is a token body; anything else is ordinary text.
    assert split_pipe_prompt("a <|not a token|> b") == ["a <", "not a token", "> b"]


def test_custom_delimiters_split_and_escape():
    assert split_pipe_prompt("one ||| two <|cutoff|> ||| three", "|||") == [
        "one", "two <|cutoff|>", "three",
    ]
    assert split_pipe_prompt(r"a %%% b with \%%% inside %%% c", "%%%") == [
        "a", "b with %%% inside", "c",
    ]
    # A single pipe is ordinary text once the delimiter is longer.
    assert split_pipe_prompt("a | b ||| c", "|||") == ["a | b", "c"]


@pytest.mark.parametrize("delimiter", ["", "<|", "|>", "a>b", "back\\slash"])
def test_unscannable_delimiters_are_rejected(delimiter):
    with pytest.raises(ValueError, match="segment delimiter"):
        validate_segment_delimiter(delimiter)
    with pytest.raises(ValueError, match="segment delimiter"):
        split_pipe_prompt("a b c", delimiter)


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


def test_context_plan_reports_work_and_prompt_assignment():
    summary = context_plan_summary(
        total=60,
        starts=[0, 20, 35],
        window=25,
        overlap=5,
        phase=0,
        blend="overlap-linear",
        stagger=True,
        assignments=[1, 2, 2],
        frame_cuts=[0, 7, 68, 204],
    )
    assert "windows=3" in summary
    assert "video_latent_visits=1.25x" in summary
    assert "stride=20" in summary
    assert "min_overlap=5" in summary
    assert "blend=overlap-linear" in summary
    assert "stagger=on" in summary
    assert "prompt_windows=2x1,3x2" in summary
    assert "max_phase" not in summary
    assert "max_phase=5" in context_plan_summary(
        total=60, starts=[0, 20, 35], window=25, overlap=5, phase=0, max_phase=5)


def test_single_window_path_emits_step_zero_context_plan(capsys):
    class Executor:
        def __call__(self, x, timestep, context, transformer_options, **kwargs):
            return x

    wrapper = make_context_wrapper(ContextPolicy(window_frames=25, overlap_frames=5))
    video = torch.zeros(1, 1, 10, 1, 1)
    audio = torch.zeros(1, 1, 1, 20)
    transformer_options = {
        "sample_sigmas": torch.tensor([1.0, 0.0]),
        "sigmas": torch.tensor([1.0]),
    }
    result = wrapper(
        Executor(),
        [video, audio],
        timestep=None,
        context=torch.zeros(1, 1, 1),
        transformer_options=transformer_options,
    )
    assert result[0] is video
    assert result[1] is audio
    receipt = capsys.readouterr().out
    assert "context plan video_latents=10 windows=1 window/overlap=10/0" in receipt
    assert "phase=0 video_latent_visits=1.00x" in receipt
    assert "stride=10 min_overlap=0 blend=pyramid stagger=off" in receipt


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
    assert tuple(context.shape) == (1, 2, 2)
    assert metadata["h3forge_prompt_segment_count"] == 2
    assert [tuple(x.shape) for x in metadata["h3forge_prompt_segments"]] == [(1, 2, 2), (1, 4, 2)]
    assert torch.equal(metadata["minimax_token_tags"], torch.ones(2, dtype=torch.long))


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
    assert tuple(context.shape) == (1, 2, 4)
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

    refinements = []

    class Diffusion:
        @staticmethod
        def preprocess_text_embeds(context):
            refinements.append(context)
            return context + 1

    def base_extra_conds(**kwargs):
        return {"minimax_payload": Cond({"seed": kwargs.get("seed", 0)}),
                "c_crossattn": Cond(torch.ones(1, 3, 2))}

    wrapper = make_segmented_extra_conds(base_extra_conds, BaseModel(), Diffusion())
    raw = (torch.zeros(1, 3, 2), torch.ones(1, 3, 2))
    result = wrapper(
        device="cpu",
        seed=7,
        h3forge_prompt_segments=raw,
        h3forge_prompt_segment_tags=(torch.ones(3), torch.ones(3)),
        h3forge_prompt_segment_durations=(2.0, 8.0),
    )
    payload = result["minimax_payload"].cond
    assert payload["seed"] == 7
    assert len(refinements) == 1 and refinements[0] is raw[1]
    assert payload["h3forge_prompt_segments"][0] is result["c_crossattn"].cond
    assert torch.equal(payload["h3forge_prompt_segments"][0], torch.ones_like(raw[0]))
    assert torch.equal(payload["h3forge_prompt_segments"][1], torch.full_like(raw[1], 2))
    assert payload["h3forge_prompt_segment_durations"] == (2.0, 8.0)


def test_duration_projection_is_scale_invariant(monkeypatch):
    import fake_minimax
    fake_minimax.install(monkeypatch)
    expected = segment_ranges(427, 2, (1, 3))
    for scaled in ((1e307, 3e307), (1e-307, 3e-307)):
        assert segment_ranges(427, 2, scaled) == expected
    with pytest.raises(ValueError, match="shorter than"):
        segment_ranges(10, 3, (0.00001, 1, 1))

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
        for segment, _, _, lo, hi, _ in plan:
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
    assert "audio_context=full:2417" in context_plan_summary(
        total=427, starts=[0, 80], window=80, overlap=10, phase=0, audio_t=2417)
    assert "audio_context=full:2417" in context_plan_summary(
        total=427, starts=[0, 80], window=80, overlap=10, phase=0, audio_t=2417, audio_window=2417)
    assert "audio_context=window:455/2417" in context_plan_summary(
        total=427, starts=[0, 80], window=80, overlap=10, phase=0, audio_t=2417, audio_window=455)


def test_context_policy_validates_audio_context_modes():
    for mode in ("auto", "window", "full"):
        assert ContextPolicy(audio_context=mode).audio_context == mode
    with pytest.raises(ValueError, match="audio_context"):
        ContextPolicy(audio_context="clip")


def test_single_window_path_emits_step_zero_context_plan(capsys):
    class Executor:
        class_obj = object()

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


def test_blended_segment_windows_assign_by_majority_and_rescue_short_beats(monkeypatch):
    import fake_minimax
    from h3forge.context import _blended_segment_windows, max_stagger_phase, window_starts
    fake_minimax.install(monkeypatch)
    total, window, overlap = 427, 80, 10
    ranges, _ = segment_ranges(total, 3, (2, 18, 40))
    assert ranges[0][1] - ranges[0][0] < window - overlap  # beat 1 is shorter than the stride
    plan = _blended_segment_windows(total, window, overlap, ranges)
    regular = [p for p in plan if (p[3], p[4]) == (p[1], p[2])]
    rescue = [p for p in plan if (p[3], p[4]) != (p[1], p[2])]
    assert [p[1] for p in regular] == window_starts(total, window, overlap)
    assert {p[0] for p in plan} == {0, 1, 2}
    # The short first beat is mostly covered by no window, so it gets exactly one
    # window of its own that writes only inside the beat.
    assert [(p[0], p[3], p[4]) for p in rescue] == [(0, ranges[0][0], ranges[0][1])]
    for index, v0, v1, _, _, excluded in regular:
        shared = [min(v1, hi) - max(v0, lo) for lo, hi in ranges]
        assert shared[index] == max(shared)
        assert excluded == (0,)  # every regular window carves out the rescued first beat
    assert all(p[5] == () for p in rescue)
    assert plan == sorted(plan, key=lambda p: (p[1], p[0]))
    max_phase = max_stagger_phase(window, overlap)
    shifted = _blended_segment_windows(total, window, overlap, ranges, max_phase, max_phase)
    regular = [p for p in shifted if (p[3], p[4]) == (p[1], p[2])]
    assert [p[1] for p in regular] == window_starts(total, window, overlap, max_phase, max_phase)
    assert {p[0] for p in shifted} == {0, 1, 2}


def test_blended_assignments_are_frozen_across_stagger_phases():
    from h3forge.context import _blended_segment_windows, max_stagger_phase
    total, window, overlap = 427, 80, 10
    # A beat boundary near a window midpoint: at phase 0 the window at 69 covers
    # beat 2 by a hair; a phase would tip it to beat 3 if assignment followed it.
    ranges = [(0, 40), (40, 109), (109, 427)]
    max_phase = max_stagger_phase(window, overlap)
    base = _blended_segment_windows(total, window, overlap, ranges, 0, max_phase)
    base_regular = [p for p in base if (p[3], p[4]) == (p[1], p[2])]
    base_rescue = [p for p in base if (p[3], p[4]) != (p[1], p[2])]
    assert [p[0] for p in base_regular][:2] == [0, 1]
    assert all(p[5] == () for p in base)  # every beat is mostly covered: nothing rescued
    for phase in range(max_phase + 1):
        plan = _blended_segment_windows(total, window, overlap, ranges, phase, max_phase)
        regular = [p for p in plan if (p[3], p[4]) == (p[1], p[2])]
        rescue = [p for p in plan if (p[3], p[4]) != (p[1], p[2])]
        assert [p[0] for p in regular] == [p[0] for p in base_regular]
        assert rescue == base_rescue
        assert len(plan) == len(base)


def test_freenoise_follows_the_actual_window_starts():
    from h3forge.context import apply_freenoise
    original = torch.randn(1, 4, 30, 2, 2)
    noise = original.clone()

    def rows(t):
        return {tuple(t[:, :, i].flatten().tolist()) for i in range(t.shape[2])}

    # 30 latents, window 12, overlap 2: the real plan is 0, 10, 18, not a nominal 0, 10, 20.
    starts = window_starts(30, 12, 2)
    assert starts == [0, 10, 18]
    assert apply_freenoise(noise, 2, starts, 12, seed=7) is noise
    assert torch.equal(noise[:, :, :12], original[:, :, :12])
    # window 2 adds frames 12..22 beyond window 1: a permutation of window 1's lead 0..10;
    # window 3 adds 22..30: a permutation of window 2's (already shuffled) lead 10..18.
    assert rows(noise[:, :, 12:22]) == rows(original[:, :, 0:10])
    assert rows(noise[:, :, 22:30]) == rows(noise[:, :, 10:18])
    assert torch.equal(apply_freenoise(original.clone(), 2, starts, 12, seed=7), noise)
    assert not torch.equal(apply_freenoise(original.clone(), 2, starts, 12, seed=8), noise)
    for bad in ([0, 0, 18], [0, 19], [-1, 9, 18]):
        with pytest.raises(ValueError):
            apply_freenoise(original.clone(), 2, bad, 12, seed=0)


def test_freenoise_wrapper_shuffles_video_only_and_only_when_windowed():
    from types import SimpleNamespace
    from h3forge.context import ContextPolicy, make_freenoise_wrapper
    shapes = [torch.Size([1, 2, 30, 2, 2]), torch.Size([1, 3, 2, 50])]
    video, audio = torch.randn(shapes[0]), torch.randn(shapes[1])
    packed = torch.cat([video.reshape(1, 1, -1), audio.reshape(1, 1, -1)], dim=-1)
    before = packed.clone()
    guider = SimpleNamespace(inner_model=SimpleNamespace(latent_shapes=shapes))
    seen = {}

    def executor(g, sigmas, extra_args, callback, noise, *args):
        seen["noise"] = noise
        return "samples"

    wrapper = make_freenoise_wrapper(ContextPolicy(window_frames=12, overlap_frames=2))
    assert wrapper(executor, guider, None, {"seed": 3}, None, packed, "latent") == "samples"
    out_video = seen["noise"][:, :, :video.numel()].reshape(shapes[0])
    out_audio = seen["noise"][:, :, video.numel():].reshape(shapes[1])
    assert torch.equal(out_audio, audio)
    assert torch.equal(out_video[:, :, :12], video[:, :, :12])
    assert not torch.equal(out_video, video)
    assert torch.equal(packed, before)  # the sampler's own noise tensor is never mutated

    for policy in (ContextPolicy(window_frames=30, overlap_frames=2),
                   ContextPolicy(window_frames=12, overlap_frames=2, freenoise=False)):
        make_freenoise_wrapper(policy)(executor, guider, None, {"seed": 3}, None, packed)
        assert seen["noise"] is packed


@pytest.mark.parametrize("seconds,latents,count,window,overlap", [
    (5, 37, 1, 37, 0), (15, 107, 1, 107, 0), (16, 117, 2, 65, 13), (20, 142, 2, 80, 13),
    (30, 217, 3, 85, 13), (60, 427, 5, 100, 13), (120, 852, 10, 100, 13),
])
def test_timeline_plan_prefers_few_even_cadence_aligned_windows(seconds, latents, count, window, overlap):
    from h3forge.timeline import plan_for_seconds
    plan = plan_for_seconds(seconds)
    assert (plan.latent_t, plan.count, plan.window, plan.overlap) == (latents, count, window, overlap)
    assert plan.windowed == (count > 1)
    if plan.windowed:
        assert plan.window % 5 == 0 and plan.window <= 107
        assert len(window_starts(plan.latent_t, plan.window, plan.overlap)) == plan.count


@pytest.mark.parametrize("cap", [15, 23, 40, 61, 107, 129])
@pytest.mark.parametrize("latent_t", [2, 7, 37, 112, 427, 1062])
def test_timeline_plan_matches_the_real_planner_for_any_cap(cap, latent_t):
    from h3forge.timeline import plan_windows
    plan = plan_windows(latent_t, cap)
    if latent_t <= cap:
        assert plan == plan.__class__(latent_t, latent_t, 0, 1)
        return
    assert plan.window <= cap and plan.window - plan.overlap >= 3 and 8 <= plan.overlap <= 16
    assert len(window_starts(latent_t, plan.window, plan.overlap)) == plan.count


def test_every_offered_duration_plans_and_reaches_the_length_asked_for():
    from h3forge.timeline import (
        DEFAULT_DURATION_SECONDS,
        DURATION_CHOICES,
        FPS,
        TRAINED_MAX_SECONDS,
        frames_for_seconds,
        plan_for_seconds,
    )
    assert DURATION_CHOICES == tuple(range(5, 61, 5)) and DEFAULT_DURATION_SECONDS in DURATION_CHOICES
    for seconds in DURATION_CHOICES:
        frames = frames_for_seconds(seconds)
        assert frames / FPS >= seconds  # snapping up the grid never returns a clip shorter than asked
        plan = plan_for_seconds(seconds, TRAINED_MAX_SECONDS)
        assert plan.count == len(window_starts(plan.latent_t, plan.window, plan.overlap))
        if plan.windowed:
            # Every window is a clip length H3 was trained on, and the stagger floor is clear.
            assert plan.window % 5 == 0 and plan.window <= 107 and plan.window - plan.overlap >= 3
        else:
            assert seconds <= TRAINED_MAX_SECONDS


def test_timeline_rejects_unplannable_inputs():
    from h3forge.timeline import frames_for_seconds, plan_for_seconds, plan_windows
    assert frames_for_seconds(5.0) == 124 and frames_for_seconds(0.01) == 5
    assert frames_for_seconds(5.17) == 141  # 124.08 frames snaps up, never down to a shorter clip
    assert frames_for_seconds(124 / 24) == 124  # an exact grid duration stays put
    with pytest.raises(ValueError):
        plan_windows(100, 14)
    with pytest.raises(ValueError):
        plan_for_seconds(0)
    with pytest.raises(ValueError):
        plan_for_seconds(10, float("inf"))


def test_cap_seconds_never_admits_a_longer_window_than_requested():
    from h3forge.timeline import MIN_CAP, MIN_CAP_SECONDS, cap_latents_for_seconds, plan_for_seconds
    assert cap_latents_for_seconds(15.5) == 107
    assert cap_latents_for_seconds(15.07) == 102  # 361.68 frames floors below the 362 grid point
    assert cap_latents_for_seconds(MIN_CAP_SECONDS) >= MIN_CAP
    plan = plan_for_seconds(8.5, 4.5)
    assert (plan.window, plan.count) == (30, 3) and plan.window * 17 / 5 / 24 <= 4.5
    assert plan_for_seconds(1.0, MIN_CAP_SECONDS).windowed is False


def test_subtract_intervals_carves_holes_in_order():
    from h3forge.context import _subtract_intervals
    assert _subtract_intervals(0, 10, []) == [(0, 10)]
    assert _subtract_intervals(0, 10, [(3, 5)]) == [(0, 3), (5, 10)]
    assert _subtract_intervals(0, 10, [(7, 12), (0, 2)]) == [(2, 7)]
    assert _subtract_intervals(4, 6, [(0, 10)]) == []
    assert _subtract_intervals(0, 10, [(2, 4), (3, 6)]) == [(0, 2), (6, 10)]


def test_frames_for_latents_inverts_the_grid_and_rejects_off_grid_lengths():
    from h3forge.timeline import av_latent_length, frames_for_latents, latents_for_seconds, video_latent_t
    for seconds in (1, 5, 20, 60):
        latents = latents_for_seconds(seconds)
        assert video_latent_t(frames_for_latents(latents)) == latents
    assert frames_for_latents(2) == 5 and frames_for_latents(427) == 1450
    with pytest.raises(ValueError):
        frames_for_latents(30)
    from types import SimpleNamespace as NS
    assert av_latent_length({"samples": NS(tensors=[torch.zeros(1, 24, 67, 2, 2), torch.zeros(1, 32, 2, 10)])}) == 67
    with pytest.raises(ValueError):
        av_latent_length({"samples": NS(tensors=[torch.zeros(1, 24, 67, 2, 2)])})


def test_context_policy_validates_neighbour_keyframe_options():
    for mode in ("off", "first", "both"):
        policy = ContextPolicy(neighbour_keyframe=mode, neighbour_keyframe_until=0.3)
        assert policy.neighbour_keyframe == mode
        assert policy.neighbour_keyframe_until == 0.3
    assert ContextPolicy().neighbour_keyframe == "off"
    with pytest.raises(ValueError, match="neighbour_keyframe"):
        ContextPolicy(neighbour_keyframe="last")
    for bad in (-0.1, 1.1, float("nan"), True):
        with pytest.raises(ValueError, match="neighbour_keyframe_until"):
            ContextPolicy(neighbour_keyframe_until=bad)


def test_neighbour_keyframe_off_is_bit_identical_to_main(monkeypatch, capsys):
    import fake_minimax
    from types import SimpleNamespace

    fake_minimax.install(monkeypatch)
    total_t, window, overlap, audio_t = 67, 25, 8, 377
    gen = torch.Generator().manual_seed(123)
    video = torch.randn(1, 4, total_t, 4, 4, generator=gen)
    audio = torch.randn(1, 2, 2, audio_t, generator=gen)
    context = torch.randn(1, 6, 8, generator=gen)
    sigmas = torch.tensor([1.0, 0.7, 0.2, 0.0])

    def run_policy(policy):
        calls = []

        class Executor:
            class_obj = SimpleNamespace(patch_size=(1, 2, 2))

            def __call__(self, local_x, timestep, local_ctx, opts, **kwargs):
                pl = kwargs["minimax_payload"]
                calls.append((
                    pl["layout"]._h3forge_video_offset,
                    pl.get("keyframes"),
                    pl.get("cond_video_latents"),
                    pl.get("visual_cond_noise_aug"),
                ))
                # Deterministic window-dependent velocity prediction.
                offset = pl["layout"]._h3forge_video_offset
                v_pred = local_x[0] * 0.5 + float(offset) * 0.01
                a_pred = local_x[1] * 0.25
                return [v_pred, a_pred]

        wrapper = make_context_wrapper(policy)
        outs = []
        for s in (1.0, 0.7, 0.2):
            opts = {"sample_sigmas": sigmas, "sigmas": torch.tensor([s])}
            outs.append(wrapper(Executor(), [video, audio], torch.tensor([s * 1000.0]), context, opts))
        return outs, calls

    default_outs, default_calls = run_policy(ContextPolicy(window, overlap, True, "pyramid"))
    off_outs, off_calls = run_policy(
        ContextPolicy(window, overlap, True, "pyramid", neighbour_keyframe="off", neighbour_keyframe_until=0.3)
    )
    for (dv, da), (ov, oa) in zip(default_outs, off_outs):
        assert torch.equal(dv, ov)
        assert torch.equal(da, oa)
    assert default_calls == off_calls
    assert all(kf is None and cl is None and aug is None for _, kf, cl, aug in off_calls)

    # Compare step 0 directly against manual main-style overlap-add reference.
    ref_starts = window_starts(total_t, window, overlap, 0, max_stagger_phase(window, overlap))
    ref_acc = torch.zeros_like(video)
    ref_den = torch.zeros(1, 1, total_t, 1, 1)
    for v0 in ref_starts:
        v1 = v0 + window
        v_pred = video[:, :, v0:v1] * 0.5 + float(v0) * 0.01
        w = blend_weights(window, overlap, device=video.device, dtype=torch.float32, mode="pyramid",
                          ramp_start=v0 > 0, ramp_end=v1 < total_t).view(1, 1, -1, 1, 1)
        ref_acc[:, :, v0:v1] += v_pred * w
        ref_den[:, :, v0:v1] += w
    assert torch.equal(off_outs[0][0], ref_acc / ref_den.clamp_min(1e-6))
    assert "[H3Forge] neighbour_keyframe=off" in capsys.readouterr().out


def test_neighbour_keyframe_first_passes_previous_window_x0_and_cadence_stagger(monkeypatch, capsys):
    import fake_minimax
    from types import SimpleNamespace
    from h3forge.context import neighbour_keyframe_noise_aug
    from h3forge.layout import target_segments

    fake_minimax.install(monkeypatch)
    # Real baseline geometry: 202 latents (5*40+2), window 85, overlap 21 -> starts [0, 60, 117].
    total_t, window, overlap, audio_t = 202, 85, 21, 1140
    gen = torch.Generator().manual_seed(7)
    video = torch.randn(1, 4, total_t, 4, 4, generator=gen)
    audio = torch.randn(1, 2, 2, audio_t, generator=gen)
    context = torch.randn(1, 6, 8, generator=gen)
    sigmas = torch.tensor([0.9, 0.5, 0.2, 0.0])

    existing_ref = {"kind": "image", "latent_h": 4, "latent_w": 4, "latent": torch.full((1, 4, 1, 4, 4), 9.0)}
    base_payload = {"refs": [existing_ref], "cond_video_latents": [existing_ref["latent"]]}

    seen = []

    class Executor:
        class_obj = SimpleNamespace(patch_size=(1, 2, 2))

        def __call__(self, local_x, timestep, local_ctx, opts, **kwargs):
            pl = kwargs["minimax_payload"]
            layout = pl["layout"]
            v0 = layout._h3forge_video_offset
            # Window-specific velocity so each window produces a distinct x0 estimate.
            v_pred = torch.full_like(local_x[0], float(len(seen) + 1))
            seen.append((v0, pl, v_pred.clone()))
            return [v_pred, local_x[1].clone()]

    policy = ContextPolicy(window, overlap, True, "pyramid", neighbour_keyframe="first", neighbour_keyframe_until=0.3)
    wrapper = make_context_wrapper(policy)

    # Step 0 (sigma=0.9 >= 0.3): active.
    opts0 = {"sample_sigmas": sigmas, "sigmas": torch.tensor([0.9])}
    out_v, out_a = wrapper(Executor(), [video, audio], torch.tensor([900.0]), context, opts0,
                           minimax_payload=base_payload)

    # Audio fusion is untouched (executor returned local_x[1], weighted mean equals audio).
    assert torch.allclose(out_a, audio)
    # Original payload was not mutated.
    assert "keyframes" not in base_payload
    assert len(base_payload["cond_video_latents"]) == 1
    assert "visual_cond_noise_aug" not in base_payload

    assert [v0 for v0, _, _ in seen] == [0, 60, 117]
    # Window 0: no neighbour keyframe added.
    assert seen[0][1].get("keyframes") is None
    assert len(seen[0][1]["cond_video_latents"]) == 1

    # Window 1 ([60, 145)): gets window 0's x0 at global latent 60 (on cadence, v0 itself).
    pl1 = seen[1][1]
    expected_x0_60 = video[:, :, 60:61].float() - 0.9 * seen[0][2][:, :, 60:61].float()
    assert len(pl1["keyframes"]) == 1
    assert pl1["keyframes"][0]["resolved_frame_index"] == (60 // 5) * 17
    assert torch.allclose(pl1["keyframes"][0]["latent"], expected_x0_60)
    # cond_video_latents order: [neighbour_kf, existing_ref] matching PackedLayout!
    assert len(pl1["cond_video_latents"]) == 2
    assert torch.equal(pl1["cond_video_latents"][0], pl1["keyframes"][0]["latent"])
    assert torch.equal(pl1["cond_video_latents"][1], existing_ref["latent"])
    assert pl1["visual_cond_noise_aug"] == pytest.approx(neighbour_keyframe_noise_aug(0.9))
    # Layout cond segment sits at global latent 60's exact RoPE coordinates.
    layout1 = pl1["layout"]
    cond_seg1 = next(s for s in layout1.segments if s[2] == "cond")
    v_seg1 = target_segments(layout1)
    assert torch.equal(
        layout1.position_ids[cond_seg1[0]:cond_seg1[1]],
        layout1.position_ids[v_seg1.video_start:v_seg1.video_start + layout1.frame_rows],
    )

    # Window 2 ([117, 202)): final start 117 is off-cadence, so anchor picks first cadence-0
    # latent in overlap [117, 145) -> latent 120 (rel index 60 in window 1, rel index 3 in window 2).
    pl2 = seen[2][1]
    expected_x0_120 = video[:, :, 120:121].float() - 0.9 * seen[1][2][:, :, 60:61].float()
    assert len(pl2["keyframes"]) == 1
    assert pl2["keyframes"][0]["resolved_frame_index"] == (120 // 5) * 17
    assert torch.allclose(pl2["keyframes"][0]["latent"], expected_x0_120)
    layout2 = pl2["layout"]
    cond_seg2 = next(s for s in layout2.segments if s[2] == "cond")
    v_seg2 = target_segments(layout2)
    fr = layout2.frame_rows
    assert torch.equal(
        layout2.position_ids[cond_seg2[0]:cond_seg2[1]],
        layout2.position_ids[v_seg2.video_start + 3 * fr:v_seg2.video_start + 4 * fr],
    )

    # Step 0 log line confirms option, cadence-restricted stagger, aug, and anchors.
    log = capsys.readouterr().out
    assert "[H3Forge] neighbour_keyframe=first" in log
    assert "until=0.30" in log and "active=yes" in log
    assert "stagger=cadence_multiples(step=5,max_phase=20)" in log
    assert "first_anchors=[60, 120]" in log

    # Step 2 (sigma=0.2 < 0.3): below until threshold -> no neighbour keyframes injected.
    seen.clear()
    opts_low = {"sample_sigmas": sigmas, "sigmas": torch.tensor([0.2])}
    wrapper(Executor(), [video, audio], torch.tensor([200.0]), context, opts_low, minimax_payload=base_payload)
    assert all(pl.get("keyframes") is None for _, pl, _ in seen)
    assert all(len(pl["cond_video_latents"]) == 1 for _, pl, _ in seen)


def test_neighbour_keyframe_both_uses_previous_step_cache_and_resets_between_runs(monkeypatch):
    import fake_minimax
    from types import SimpleNamespace

    fake_minimax.install(monkeypatch)
    total_t, window, overlap, audio_t = 202, 85, 21, 1140
    video = torch.zeros(1, 4, total_t, 4, 4)
    audio = torch.zeros(1, 2, 2, audio_t)
    context = torch.zeros(1, 6, 8)
    sigmas = torch.tensor([1.0, 0.6, 0.2, 0.0])

    step_tag = [1]
    seen = []

    class Executor:
        class_obj = SimpleNamespace(patch_size=(1, 2, 2))

        def __call__(self, local_x, timestep, local_ctx, opts, **kwargs):
            pl = kwargs["minimax_payload"]
            v0 = pl["layout"]._h3forge_video_offset
            # Encode (step_tag, window_start) directly into velocity so x0 = -sigma * v is unique.
            val = float(step_tag[0] * 1000 + v0)
            v_pred = torch.full_like(local_x[0], val)
            seen.append((v0, pl.get("keyframes") or []))
            return [v_pred, local_x[1]]

    policy = ContextPolicy(window, overlap, True, "pyramid", neighbour_keyframe="both", neighbour_keyframe_until=0.3)
    wrapper = make_context_wrapper(policy)

    # Step 0 (sigma=1.0): no prior step, so only forward (first) keyframes on windows 1 and 2.
    wrapper(Executor(), [video, audio], torch.tensor([1000.0]), context,
            {"sample_sigmas": sigmas, "sigmas": torch.tensor([1.0])})
    assert [len(kfs) for _, kfs in seen] == [0, 1, 1]

    # Step 1 (sigma=0.6): window 0 gets trailing keyframe from step 0 window 1 (anchor 80);
    # window 1 gets BOTH leading (step 1 window 0, anchor 60) and trailing (step 0 window 2, anchor 140);
    # window 2 gets leading (step 1 window 1, anchor 120).
    seen.clear()
    step_tag[0] = 2
    wrapper(Executor(), [video, audio], torch.tensor([600.0]), context,
            {"sample_sigmas": sigmas, "sigmas": torch.tensor([0.6])})
    assert [len(kfs) for _, kfs in seen] == [1, 2, 1]
    # Window 0 trailing keyframe: anchor 80 from step 0 window 1 (val=1060, sigma=1.0 -> x0=-1060).
    assert seen[0][1][0]["resolved_frame_index"] == (80 // 5) * 17
    assert torch.allclose(seen[0][1][0]["latent"], torch.full((1, 4, 1, 4, 4), -1060.0))
    # Window 1 leading: anchor 60 from step 1 window 0 (val=2000, sigma=0.6 -> x0=-1200).
    # Window 1 trailing: anchor 140 from step 0 window 2 (val=1117, sigma=1.0 -> x0=-1117).
    assert [kf["resolved_frame_index"] for kf in seen[1][1]] == [(60 // 5) * 17, (140 // 5) * 17]
    assert torch.allclose(seen[1][1][0]["latent"], torch.full((1, 4, 1, 4, 4), -1200.0))
    assert torch.allclose(seen[1][1][1]["latent"], torch.full((1, 4, 1, 4, 4), -1117.0))
    # Window 2 leading: anchor 120 from step 1 window 1 (val=2060, sigma=0.6 -> x0=-1236).
    assert seen[2][1][0]["resolved_frame_index"] == (120 // 5) * 17
    assert torch.allclose(seen[2][1][0]["latent"], torch.full((1, 4, 1, 4, 4), -1236.0))

    # Starting a new run (step 0 again) clears the previous-step cache.
    seen.clear()
    step_tag[0] = 3
    wrapper(Executor(), [video, audio], torch.tensor([1000.0]), context,
            {"sample_sigmas": sigmas, "sigmas": torch.tensor([1.0])})
    assert [len(kfs) for _, kfs in seen] == [0, 1, 1]


def test_cadence_only_stagger_keeps_interior_starts_on_5_latent_grid():
    for total, window, overlap in [(202, 85, 21), (252, 85, 25), (322, 95, 15)]:
        mp = max_stagger_phase(window, overlap, cadence_only=True)
        assert mp % 5 == 0
        for step in range(30):
            phase = stagger_phase(step, window, overlap, cadence_only=True)
            assert phase % 5 == 0 and 0 <= phase <= mp
            starts = window_starts(total, window, overlap, phase, mp, cadence_only=True)
            # Every non-final start is strictly on H3's 5-latent cadence.
            assert all(s % 5 == 0 for s in starts[:-1])


def test_neighbour_keyframe_freenoise_uses_cadence_starts_and_both_survives_stagger_shift(monkeypatch, capsys):
    import fake_minimax
    from types import SimpleNamespace
    from h3forge.context import make_freenoise_wrapper

    fake_minimax.install(monkeypatch)
    # 202 / 85 / 21: default stagger picks off-cadence anchor [0, 58, 117];
    # neighbour_keyframe restricts FreeNoise to cadence-aligned [0, 60, 117].
    shapes = [(1, 2, 202, 2, 2), (1, 2, 2, 20)]
    guider = SimpleNamespace(inner_model=SimpleNamespace(latent_shapes=shapes))
    packed = torch.randn(1, 1, 2 * 202 * 4 + 2 * 2 * 20)
    make_freenoise_wrapper(ContextPolicy(85, 21, True, "pyramid", neighbour_keyframe="first"))(
        lambda *args, **kwargs: args[-1], guider, None, {"seed": 42}, None, packed
    )
    assert "starts=[0, 60, 117]" in capsys.readouterr().out

    # 252 / 85 / 25: cadence-restricted stagger moves interior starts across steps
    # (step 0 -> [0, 55, 110, 167], step 1 -> [0, 60, 120, 167]); mode `both` must
    # still supply valid trailing keyframes from the previous-step cache.
    total_t, window, overlap, audio_t = 252, 85, 25, 1422
    video = torch.zeros(1, 4, total_t, 4, 4)
    audio = torch.zeros(1, 2, 2, audio_t)
    context = torch.zeros(1, 6, 8)
    sigmas = torch.tensor([1.0, 0.7, 0.4, 0.0])
    seen = []

    class Executor:
        class_obj = SimpleNamespace(patch_size=(1, 2, 2))

        def __call__(self, local_x, timestep, local_ctx, opts, **kwargs):
            pl = kwargs["minimax_payload"]
            seen.append((pl["layout"]._h3forge_video_offset, [kf["resolved_frame_index"] for kf in pl.get("keyframes") or []]))
            return [torch.ones_like(local_x[0]), local_x[1]]

    wrapper = make_context_wrapper(
        ContextPolicy(window, overlap, True, "pyramid", neighbour_keyframe="both", neighbour_keyframe_until=0.3)
    )
    wrapper(Executor(), [video, audio], torch.tensor([1000.0]), context,
            {"sample_sigmas": sigmas, "sigmas": torch.tensor([1.0])})
    seen.clear()
    wrapper(Executor(), [video, audio], torch.tensor([700.0]), context,
            {"sample_sigmas": sigmas, "sigmas": torch.tensor([0.7])})
    assert [v0 for v0, _ in seen] == [0, 60, 120, 167]
    # Window 0 has 1 trailing, windows 1 & 2 have leading + trailing, window 3 has 1 leading.
    assert [len(frames) for _, frames in seen] == [1, 2, 2, 1]
    assert all(frames[0] < frames[1] for _, frames in seen if len(frames) == 2)


def test_neighbour_keyframe_packs_against_real_comfy_core():
    import math
    import torch.nn as nn
    from core_source import definitions

    env = definitions(
        "comfy/ldm/minimax/model.py",
        [
            "_axis_from_sqrt_area",
            "_frame_grid",
            "_video_t_spans",
            "_video_t_grid",
            "_ref_t_span",
            "_audio_grid",
            "_video_grid",
            "PackedLayout",
            "patchify_video",
        ],
        {"torch": torch, "nn": nn, "math": math, "FRAME_PER_TOKEN": (1, 4, 4, 4, 4), "FRAME_RESCALE": 5.0 / 3.0},
    )
    PackedLayout = env["PackedLayout"]
    patchify_video = env["patchify_video"]

    # Build a full layout and clone a window with one neighbour keyframe using real PackedLayout.
    import sys
    import types
    from h3forge.layout import clone_window_layout
    from h3forge.context import _inject_neighbour_keyframes, _latent_pixel_frame

    mod = types.ModuleType("comfy.ldm.minimax.model")
    mod.PackedLayout = PackedLayout
    mod._video_t_spans = env["_video_t_spans"]
    old = {k: sys.modules.get(k) for k in ("comfy", "comfy.ldm", "comfy.ldm.minimax", "comfy.ldm.minimax.model")}
    try:
        sys.modules["comfy.ldm.minimax.model"] = mod
        full = PackedLayout(8, 202, 4, 4, 1140)
        kf_latent = torch.randn(1, 24, 1, 4, 4)
        extra = [({"resolved_frame_index": _latent_pixel_frame(60), "latent": kf_latent}, 60)]
        local = clone_window_layout(
            full_layout=full, text_len=8, video_shape=(85, 4, 4), audio_t=1140,
            video_range=(60, 145), audio_range=(0, 1140), extra_keyframes=extra,
        )
        local_payload = _inject_neighbour_keyframes({}, extra, sigma=0.8)
    finally:
        for k, v in old.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    # Verify real PackedLayout's img_update / cond_video_latents row counts match _embed_and_pack.
    cond_rows = patchify_video(local_payload["cond_video_latents"][0].to(torch.float32), (1, 2, 2))
    video_rows = patchify_video(torch.zeros(1, 24, 85, 4, 4), (1, 2, 2))
    assert int((~local.img_update).sum()) == cond_rows.shape[0]
    assert int(local.img_update.sum()) == video_rows.shape[0]
    # And the cond row's RoPE coordinates equal target video frame 0's (global latent 60) coordinates!
    cond_seg = next(s for s in local.segments if s[2] == "cond")
    vid_seg = next(s for s in local.segments if s[2] == "video")
    frame_rows = cond_seg[1] - cond_seg[0]
    assert torch.equal(
        local.position_ids[cond_seg[0]:cond_seg[1]],
        local.position_ids[vid_seg[0]:vid_seg[0] + frame_rows],
    )
    assert 0.5 <= local_payload["visual_cond_noise_aug"] <= 0.999

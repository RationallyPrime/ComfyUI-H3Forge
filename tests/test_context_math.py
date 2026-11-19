from itertools import pairwise

import pytest
import torch

from h3forge.context import (
    ContextPolicy,
    assert_full_coverage,
    audio_overlap_frames,
    band_overlap_weights,
    blend_weights,
    context_plan_summary,
    laplacian_pyramid,
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


def _band_test_inputs(total, h, w, *, channels=4, dtype=torch.float32):
    from fake_minimax import PackedLayout, _video_t_spans

    audio_t = max(1, round(sum(_video_t_spans(total))))
    video = torch.zeros(1, channels, total, h, w, dtype=dtype)
    audio = torch.zeros(1, 2, 2, audio_t, dtype=dtype)
    context = torch.ones(1, 3, 8, dtype=dtype)
    payload = {"layout": PackedLayout(3, total, h + (h % 2), w + (w % 2), audio_t)}
    return [video, audio], context, payload


def _band_executor(fn):
    from types import SimpleNamespace

    class Executor:
        class_obj = SimpleNamespace(patch_size=(1, 2, 2))

        def __call__(self, *args, **kwargs):
            return fn(*args, **kwargs)

    return Executor()


def _checkerboard(h, w, *, dtype=torch.float32):
    ys = torch.arange(h, dtype=torch.long).view(h, 1)
    xs = torch.arange(w, dtype=torch.long).view(1, w)
    return torch.where(((ys + xs) % 2) == 0, 1.0, -1.0).to(dtype)


def test_context_policy_validates_band_fusion_options():
    default = ContextPolicy()
    assert default.band_fusion == 0
    assert default.band_crossfade == 2
    assert default.band_adaptive is False
    assert default.band_stagger_hold is True
    for levels in (0, 2, 3, 4):
        assert ContextPolicy(band_fusion=levels).band_fusion == levels
    for bad in (-1, 1, True, 2.5):
        with pytest.raises(ValueError, match="band_fusion"):
            ContextPolicy(band_fusion=bad)
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError, match="band_crossfade"):
            ContextPolicy(band_crossfade=bad)


@pytest.mark.parametrize("levels", [1, 2, 3, 4])
@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (5, 7), (16, 20), (34, 60)])
def test_laplacian_pyramid_reconstructs_exactly_and_separates_frequencies(levels, shape):
    h, w = shape
    gen = torch.Generator().manual_seed(17 + levels * 31 + h * 7 + w)
    x = torch.randn(1, 3, 4, h, w, generator=gen)
    bands = laplacian_pyramid(x, levels)
    assert len(bands) == levels
    assert all(band.shape == x.shape and band.dtype == torch.float32 for band in bands)
    reconstructed = sum(bands)
    assert torch.allclose(reconstructed, x, atol=1e-5, rtol=1e-5)

    # Spatially constant input stays entirely in the lowest band.
    const = torch.full_like(x, 3.25)
    const_bands = laplacian_pyramid(const, levels)
    assert torch.allclose(const_bands[0], const, atol=1e-5)
    for high in const_bands[1:]:
        assert torch.allclose(high, torch.zeros_like(high), atol=1e-5)


def test_laplacian_pyramid_puts_nyquist_checkerboard_in_highest_band():
    cb = _checkerboard(20, 20).view(1, 1, 1, 20, 20)
    bands = laplacian_pyramid(cb, 3)
    assert torch.allclose(sum(bands), cb, atol=1e-5)
    # Away from the replicate-padded border (radius 2 at level 1 + radius 4 at level 2 = 6),
    # [1, 4, 6, 4, 1] / 16 annihilates Nyquist exactly.
    assert torch.allclose(bands[0][..., 6:-6, 6:-6], torch.zeros(1, 1, 1, 8, 8), atol=1e-5)
    assert torch.allclose(bands[1][..., 6:-6, 6:-6], torch.zeros(1, 1, 1, 8, 8), atol=1e-5)
    assert torch.allclose(bands[2][..., 6:-6, 6:-6], cb[..., 6:-6, 6:-6], atol=1e-5)
    for bad in (0, -2, True):
        with pytest.raises(ValueError, match="levels"):
            laplacian_pyramid(cb, bad)


@pytest.mark.parametrize("levels", [2, 3, 4])
@pytest.mark.parametrize("crossfade", [0, 1, 2, 5, 40])
@pytest.mark.parametrize("ov_len", [8, 21, 24, 25])
@pytest.mark.parametrize("blend_mode", ["pyramid", "overlap-linear", "flat"])
def test_band_overlap_weights_sum_to_one_and_narrow_with_band(levels, crossfade, ov_len, blend_mode):
    seg_start = 60
    seg_end = seg_start + ov_len
    w_a = blend_weights(85, ov_len, device="cpu", dtype=torch.float32, mode=blend_mode)[-ov_len:]
    w_b = blend_weights(85, ov_len, device="cpu", dtype=torch.float32, mode=blend_mode)[:ov_len]
    pair_bounds = [(float(seg_start), float(seg_end), 0.5 * (seg_start + seg_end), float(ov_len))]
    weights = band_overlap_weights(seg_start, seg_end, [w_a, w_b], levels, crossfade, pair_bounds)
    assert len(weights) == levels
    for band_w in weights:
        assert len(band_w) == 2
        total = band_w[0] + band_w[1]
        assert torch.allclose(total, torch.ones(ov_len), atol=1e-6)
        assert torch.all(band_w[0] >= 0.0) and torch.all(band_w[1] >= 0.0)
        assert torch.all(band_w[0] <= 1.0) and torch.all(band_w[1] <= 1.0)

    # Band 0 matches the normalized existing blend weights exactly.
    expected_0_a = w_a / (w_a + w_b)
    assert torch.allclose(weights[0][0], expected_0_a, atol=1e-6)

    # Higher bands transition at least as narrowly as lower bands.
    if crossfade < ov_len:
        mixed_counts = [int(((w[1] > 1e-6) & (w[1] < 1.0 - 1e-6)).sum()) for w in weights[1:]]
        assert mixed_counts == sorted(mixed_counts, reverse=True)
        # Highest band equals pure window A on the far left and pure window B on the far right.
        if crossfade <= ov_len - 4:
            assert float(weights[-1][0][0]) == 1.0 and float(weights[-1][1][0]) == 0.0
            assert float(weights[-1][0][-1]) == 0.0 and float(weights[-1][1][-1]) == 1.0


def test_band_overlap_weights_sum_to_one_on_three_way_overlap():
    seg_start, seg_end = 14, 21
    base = [torch.full((7,), 1.0), torch.full((7,), 2.0), torch.full((7,), 1.0)]
    pair_bounds = [
        (7.0, 25.0, 16.0, 18.0),
        (14.0, 32.0, 23.0, 18.0),
    ]
    weights = band_overlap_weights(seg_start, seg_end, base, 3, 2, pair_bounds)
    for band_w in weights:
        assert torch.allclose(sum(band_w), torch.ones(7), atol=1e-6)
        for w in band_w:
            assert torch.all((w >= 0.0) & (w <= 1.0))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("blend", ["pyramid", "overlap-linear", "flat"])
def test_band_fusion_level_zero_is_bit_identical_to_plain_overlap_add(monkeypatch, dtype, blend):
    import fake_minimax

    fake_minimax.install(monkeypatch)
    total, window, overlap = 40, 25, 10
    x, context, payload = _band_test_inputs(total, 12, 12, channels=3, dtype=dtype)
    gen = torch.Generator().manual_seed(123)
    win_videos = [torch.randn(1, 3, window, 12, 12, generator=gen, dtype=dtype) for _ in range(2)]
    win_audios = [torch.randn(x[1].shape, generator=gen, dtype=dtype) for _ in range(2)]
    options = {"sigmas": torch.tensor([0.5]), "sample_sigmas": torch.tensor([1.0, 0.5, 0.0])}

    def make_runner():
        call = 0

        def run(local_x, timestep, text, opts, **kwargs):
            nonlocal call
            idx = call
            call += 1
            return [win_videos[idx], win_audios[idx]]

        return run

    plain = make_context_wrapper(ContextPolicy(window, overlap, True, blend, band_fusion=0))(
        _band_executor(make_runner()), x, torch.tensor([500.0]), context, options, minimax_payload=payload
    )
    # Non-default auxiliary band options must have zero effect when band_fusion=0.
    also_off = make_context_wrapper(
        ContextPolicy(window, overlap, True, blend, band_fusion=0, band_crossfade=0,
                      band_adaptive=True, band_stagger_hold=False)
    )(_band_executor(make_runner()), x, torch.tensor([500.0]), context, options, minimax_payload=payload)
    assert torch.equal(plain[0], also_off[0])
    assert torch.equal(plain[1], also_off[1])

    # Verify directly against manual plain overlap-add.
    acc = torch.zeros_like(x[0], dtype=torch.float32)
    den = torch.zeros((1, 1, total, 1, 1), dtype=torch.float32)
    starts = window_starts(total, window, overlap, stagger_phase(1, window, overlap), max_stagger_phase(window, overlap))
    for v0, v_out in zip(starts, win_videos):
        vw = blend_weights(window, overlap, device="cpu", dtype=torch.float32, mode=blend,
                           ramp_start=v0 > 0, ramp_end=v0 + window < total).view(1, 1, -1, 1, 1)
        acc[:, :, v0:v0 + window].add_(v_out.float() * vw)
        den[:, :, v0:v0 + window].add_(vw)
    expected_video = (acc / den.clamp_min(1e-6)).to(dtype)
    assert torch.equal(plain[0], expected_video)


@pytest.mark.parametrize("levels", [2, 3])
def test_band_fusion_splits_high_band_away_from_crossfade(monkeypatch, levels):
    import fake_minimax

    fake_minimax.install(monkeypatch)
    # Windows [0, 25) and [15, 40) -> overlap [15, 25), midpoint = 20.0.
    total, window, overlap = 40, 25, 10
    x, context, payload = _band_test_inputs(total, 20, 20, channels=2)
    low = torch.full((1, 2, window, 20, 20), 2.5)
    cb = _checkerboard(20, 20).view(1, 1, 1, 20, 20).expand(1, 2, window, 20, 20)
    win_a = low + 1.5 * cb
    win_b = low - 1.5 * cb
    audio_a = torch.full_like(x[1], 1.0)
    audio_b = torch.full_like(x[1], 3.0)
    options = {"sigmas": torch.tensor([1.0]), "sample_sigmas": torch.tensor([1.0, 0.0])}

    def make_runner():
        outputs = [(win_a, audio_a), (win_b, audio_b)]
        idx = 0

        def run(local_x, timestep, text, opts, **kwargs):
            nonlocal idx
            out = outputs[idx]
            idx += 1
            return [out[0], out[1]]

        return run

    plain = make_context_wrapper(ContextPolicy(window, overlap, False, "pyramid", band_fusion=0))(
        _band_executor(make_runner()), x, torch.tensor([1000.0]), context, options, minimax_payload=payload
    )
    banded = make_context_wrapper(
        ContextPolicy(window, overlap, False, "pyramid", band_fusion=levels, band_crossfade=2)
    )(_band_executor(make_runner()), x, torch.tensor([1000.0]), context, options, minimax_payload=payload)

    # Audio and solo video spans are strictly untouched.
    assert torch.equal(banded[1], plain[1])
    assert torch.equal(banded[0][:, :, :15], plain[0][:, :, :15])
    assert torch.equal(banded[0][:, :, 25:], plain[0][:, :, 25:])

    # With crossfade=2 centred at 20.0, transition spans [19.0, 21.0] (latents 19 and 20).
    # Away from the crossfade (and replicate border), latents 15..18 equal window A and
    # latents 21..24 equal window B in both the high band and the reconstructed frame.
    interior = (slice(None), slice(None), slice(None), slice(6, -6), slice(6, -6))
    fused_high = laplacian_pyramid(banded[0][:, :, 15:25], levels)[-1][interior]
    high_a = laplacian_pyramid(win_a[:, :, 15:25], levels)[-1][interior]
    high_b = laplacian_pyramid(win_b[:, :, :10], levels)[-1][interior]
    assert torch.allclose(fused_high[:, :, :4], high_a[:, :, :4], atol=1e-5)
    assert torch.allclose(fused_high[:, :, 6:], high_b[:, :, 6:], atol=1e-5)
    assert torch.allclose(banded[0][:, :, 15:19, 6:-6, 6:-6], win_a[:, :, 15:19, 6:-6, 6:-6], atol=1e-5)
    assert torch.allclose(banded[0][:, :, 21:25, 6:-6, 6:-6], win_b[:, :, 6:10, 6:-6, 6:-6], atol=1e-5)
    # Inside the 2-latent crossfade (19 and 20), weights are 0.75/0.25 and 0.25/0.75.
    assert torch.allclose(
        banded[0][:, :, 19, 6:-6, 6:-6],
        0.75 * win_a[:, :, 19, 6:-6, 6:-6] + 0.25 * win_b[:, :, 4, 6:-6, 6:-6],
        atol=1e-5,
    )
    assert torch.allclose(
        banded[0][:, :, 20, 6:-6, 6:-6],
        0.25 * win_a[:, :, 20, 6:-6, 6:-6] + 0.75 * win_b[:, :, 5, 6:-6, 6:-6],
        atol=1e-5,
    )


def test_band_fusion_keeps_low_band_blended_while_switching_high_band(monkeypatch):
    import fake_minimax

    fake_minimax.install(monkeypatch)
    total, window, overlap = 40, 25, 10
    x, context, payload = _band_test_inputs(total, 16, 16, channels=2)
    cb = _checkerboard(16, 16).view(1, 1, 1, 16, 16).expand(1, 2, window, 16, 16)
    win_a = torch.full((1, 2, window, 16, 16), 1.0) + 1.0 * cb
    win_b = torch.full((1, 2, window, 16, 16), 5.0) - 1.0 * cb
    options = {"sigmas": torch.tensor([1.0]), "sample_sigmas": torch.tensor([1.0, 0.0])}

    def make_runner():
        outputs = [win_a, win_b]
        idx = 0

        def run(local_x, timestep, text, opts, **kwargs):
            nonlocal idx
            v = outputs[idx]
            idx += 1
            return [v, local_x[1]]

        return run

    plain = make_context_wrapper(ContextPolicy(window, overlap, False, "pyramid", band_fusion=0))(
        _band_executor(make_runner()), x, torch.tensor([1000.0]), context, options, minimax_payload=payload
    )
    banded = make_context_wrapper(
        ContextPolicy(window, overlap, False, "pyramid", band_fusion=2, band_crossfade=2)
    )(_band_executor(make_runner()), x, torch.tensor([1000.0]), context, options, minimax_payload=payload)

    # Interior low band (mean over 2x2 blocks cancels checkerboard) matches plain pyramid blend.
    plain_low = laplacian_pyramid(plain[0][:, :, 15:25], 2)[0][..., 4:-4, 4:-4]
    banded_low = laplacian_pyramid(banded[0][:, :, 15:25], 2)[0][..., 4:-4, 4:-4]
    assert torch.allclose(banded_low, plain_low, atol=1e-5)

    # Interior high band switches cleanly to window A on 15..18 and window B on 21..24.
    banded_high = laplacian_pyramid(banded[0][:, :, 15:25], 2)[1][..., 4:-4, 4:-4]
    plain_high = laplacian_pyramid(plain[0][:, :, 15:25], 2)[1][..., 4:-4, 4:-4]
    assert torch.allclose(banded_high[:, :, :4], cb[:, :, :4, 4:-4, 4:-4], atol=1e-5)
    assert torch.allclose(banded_high[:, :, 6:], -cb[:, :, :4, 4:-4, 4:-4], atol=1e-5)
    # Plain averaging damps the high band in the overlap; band fusion keeps full amplitude.
    assert banded_high[:, :, 3].abs().mean() > plain_high[:, :, 3].abs().mean() * 1.5


def test_band_fusion_adaptive_gates_narrow_transition_by_local_disagreement(monkeypatch):
    import fake_minimax

    fake_minimax.install(monkeypatch)
    total, window, overlap = 40, 25, 10
    x, context, payload = _band_test_inputs(total, 16, 32, channels=2)
    cb = _checkerboard(16, 32).view(1, 1, 1, 16, 32).expand(1, 2, window, 16, 32)
    win_a = torch.ones(1, 2, window, 16, 32) + cb
    win_b = win_a.clone()
    # Left half (x < 16): windows nearly agree (small 5% amplitude difference -> ratio ~ 0.025 < 0.15).
    win_b[..., :16] = 1.0 + 0.95 * cb[..., :16]
    # Right half (x >= 16): windows maximally disagree (opposite sign -> ratio = 1.0 > 0.50).
    win_b[..., 16:] = 1.0 - 1.0 * cb[..., 16:]
    options = {"sigmas": torch.tensor([1.0]), "sample_sigmas": torch.tensor([1.0, 0.0])}

    def run_policy(policy, scale=1.0):
        outputs = [win_a * scale, win_b * scale]
        idx = 0

        def run(local_x, timestep, text, opts, **kwargs):
            nonlocal idx
            v = outputs[idx]
            idx += 1
            return [v, local_x[1]]

        return make_context_wrapper(policy)(
            _band_executor(run), x, torch.tensor([1000.0]), context, options, minimax_payload=payload
        )

    plain = run_policy(ContextPolicy(window, overlap, False, "pyramid", band_fusion=0))
    adaptive = run_policy(
        ContextPolicy(window, overlap, False, "pyramid", band_fusion=2, band_crossfade=2, band_adaptive=True)
    )
    non_adaptive = run_policy(
        ContextPolicy(window, overlap, False, "pyramid", band_fusion=2, band_crossfade=2, band_adaptive=False)
    )

    # Agreeing region (deep inside left half): adaptive keeps the plain blend.
    assert torch.allclose(adaptive[0][:, :, 15:25, 4:-4, 4:12], plain[0][:, :, 15:25, 4:-4, 4:12], atol=1e-5)
    assert not torch.allclose(non_adaptive[0][:, :, 18, 4:-4, 4:12], plain[0][:, :, 18, 4:-4, 4:12], atol=1e-4)

    # Disagreeing region (deep inside right half), away from the crossfade: adaptive snaps like non-adaptive.
    assert torch.allclose(adaptive[0][:, :, 15:19, 4:-4, 20:28], win_a[:, :, 15:19, 4:-4, 20:28], atol=1e-5)
    assert torch.allclose(adaptive[0][:, :, 21:25, 4:-4, 20:28], win_b[:, :, 6:10, 4:-4, 20:28], atol=1e-5)

    # Scale invariance: scaling both windows by 100 scales the adaptive output by 100.
    scaled = run_policy(
        ContextPolicy(window, overlap, False, "pyramid", band_fusion=2, band_crossfade=2, band_adaptive=True),
        scale=100.0,
    )
    assert torch.allclose(scaled[0], adaptive[0] * 100.0, atol=1e-3, rtol=1e-5)


def test_band_fusion_stagger_hold_locks_high_band_owner_at_or_below_sigma_0_8(monkeypatch):
    import fake_minimax

    fake_minimax.install(monkeypatch)
    # Baseline geometry: 202 video latents, window 85, overlap 21 (3 windows).
    # Phase-0 starts: [0, 58, 117]; max-phase starts: [0, 64, 117].
    # Held intersection for overlap 0-1 is [64, 85), midpoint = 74.5 (latents <=73 win 0, >=75 win 1).
    total, window, overlap = 202, 85, 21
    x, context, payload = _band_test_inputs(total, 12, 12, channels=1)
    cb = _checkerboard(12, 12).view(1, 1, 1, 12, 12)
    # Schedule with 5 steps so steps 0..3 hit distinct stagger phases and sigmas above & below 0.8.
    sample_sigmas = torch.tensor([1.0, 0.9, 0.8, 0.5, 0.2, 0.0])

    def run_step(policy, step_idx):
        sigma = float(sample_sigmas[step_idx])
        opts = {"sigmas": torch.tensor([sigma]), "sample_sigmas": sample_sigmas}
        call = 0

        def run(local_x, timestep, text, options, **kwargs):
            nonlocal call
            win_idx = call
            call += 1
            t_len = local_x[0].shape[2]
            sign = 1.0 if win_idx % 2 == 0 else -1.0
            # Distinct low band (win_idx + 1) AND opposite high-band checkerboard per window.
            v = torch.full((1, 1, t_len, 12, 12), float(win_idx + 1)) + sign * cb
            a = torch.full_like(local_x[1], float(win_idx + 1))
            return [v, a]

        return make_context_wrapper(policy)(
            _band_executor(run), x, torch.tensor([sigma * 1000.0]), context, opts, minimax_payload=payload
        )

    held = ContextPolicy(window, overlap, True, "pyramid", band_fusion=2, band_crossfade=2, band_stagger_hold=True)
    unheld = ContextPolicy(window, overlap, True, "pyramid", band_fusion=2, band_crossfade=2, band_stagger_hold=False)
    plain = ContextPolicy(window, overlap, True, "pyramid", band_fusion=0)

    # At sigma = 0.9 (> 0.8, step 1): hold is NOT active yet, so held == unheld.
    assert torch.equal(run_step(held, 1)[0], run_step(unheld, 1)[0])

    # At sigma <= 0.8 (steps 2, 3, 4 -> distinct stagger phases):
    # Held keeps the high-band owner identical at every non-crossfade latent around overlap 0-1 (latents 60..84).
    held_highs = []
    held_lows = []
    for step_idx in (2, 3, 4):
        out_held = run_step(held, step_idx)
        out_plain = run_step(plain, step_idx)
        # Audio is always bit-identical to plain (staggered) main.
        assert torch.equal(out_held[1], out_plain[1])
        pyr = laplacian_pyramid(out_held[0], 2)
        held_lows.append(pyr[0][:, :, 60:85, 4:-4, 4:-4])
        held_highs.append(pyr[1][:, :, 60:85, 4:-4, 4:-4])

    # Every step at sigma <= 0.8 produces the EXACT same high band across the overlap!
    assert torch.allclose(held_highs[0], held_highs[1], atol=1e-5)
    assert torch.allclose(held_highs[0], held_highs[2], atol=1e-5)
    # Meanwhile, the low band STILL staggers across steps 2 and 3 (different stagger phases).
    assert not torch.allclose(held_lows[0], held_lows[1], atol=1e-4)

    # Without hold, the high-band midpoint moves between step 2 and step 3.
    unheld_2 = laplacian_pyramid(run_step(unheld, 2)[0], 2)[1][:, :, 60:85, 4:-4, 4:-4]
    unheld_3 = laplacian_pyramid(run_step(unheld, 3)[0], 2)[1][:, :, 60:85, 4:-4, 4:-4]
    assert not torch.allclose(unheld_2, unheld_3, atol=1e-4)


def test_band_fusion_three_way_overlap_and_step_zero_logging(monkeypatch, capsys):
    import fake_minimax

    fake_minimax.install(monkeypatch)
    # 35 latents, window 25, overlap 18 -> starts [0, 5, 10], 3-way overlap on [10, 25).
    total, window, overlap = 35, 25, 18
    assert window_starts(total, window, overlap) == [0, 5, 10]
    x, context, payload = _band_test_inputs(total, 8, 8, channels=2)
    options = {"sigmas": torch.tensor([1.0]), "sample_sigmas": torch.tensor([1.0, 0.0])}

    def run(local_x, timestep, text, opts, **kwargs):
        return [torch.randn_like(local_x[0]), torch.ones_like(local_x[1])]

    torch.manual_seed(7)
    plain = make_context_wrapper(ContextPolicy(window, overlap, False, "pyramid", band_fusion=0))(
        _band_executor(run), x, torch.tensor([1000.0]), context, options, minimax_payload=payload
    )
    off_log = capsys.readouterr().out
    assert "[H3Forge] context plan" in off_log and "band_fusion=off" in off_log

    torch.manual_seed(7)
    banded = make_context_wrapper(
        ContextPolicy(window, overlap, False, "pyramid", band_fusion=3, band_crossfade=2, band_adaptive=True)
    )(_band_executor(run), x, torch.tensor([1000.0]), context, options, minimax_payload=payload)
    on_log = capsys.readouterr().out
    assert "[H3Forge] context plan" in on_log
    assert "band_fusion=3 band_crossfade=2 band_adaptive=on band_stagger_hold=off band_overlap_latents=25" in on_log
    assert torch.isfinite(banded[0]).all()
    assert torch.equal(banded[1], plain[1])
    # Solo latents [0, 5) and [30, 35) are bit-identical.
    assert torch.equal(banded[0][:, :, :5], plain[0][:, :, :5])
    assert torch.equal(banded[0][:, :, 30:], plain[0][:, :, 30:])

    # Single-window clip logs inactive status clearly.
    small_x, small_ctx, small_payload = _band_test_inputs(10, 8, 8, channels=2)
    make_context_wrapper(ContextPolicy(25, 8, True, "pyramid", band_fusion=2))(
        _band_executor(lambda lx, *a, **kw: lx), small_x, None, small_ctx, options, minimax_payload=small_payload
    )
    single_log = capsys.readouterr().out
    assert "band_fusion=2(inactive:single-window)" in single_log

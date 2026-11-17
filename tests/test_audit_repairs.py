from types import SimpleNamespace
from typing import Callable

import pytest
import torch

from core_source import definitions, native_h3
from h3forge.attention import make_attention_override
from h3forge.context import ContextPolicy, make_context_wrapper
from h3forge.prompt import combine_conditioning_segments, segment_ranges
from h3forge.state import AttentionPolicy, RuntimeState, require_eager_allocations
from test_nodes import FakePatcher, _import_nodes


def test_native_allocation_recorder_is_rejected_before_forward():
    require_eager_allocations(SimpleNamespace())
    with pytest.raises(RuntimeError, match="--disable-comfy-compiler"):
        require_eager_allocations(SimpleNamespace(_comfy_malloc_graph=object()))


@pytest.mark.parametrize("order", [("attention", "nag"), ("nag", "attention")])
def test_node_composition_with_actual_comfy_setter(monkeypatch, order):
    setter = definitions("comfy/model_patcher.py", ["set_model_options_patch_replace"])["set_model_options_patch_replace"]
    copy_options = definitions("comfy/utils.py", ["deepcopy_list_dict"])["deepcopy_list_dict"]
    nodes = _import_nodes(monkeypatch)

    def replace(self, fn, name, block, index):
        self.model_options = setter(self.model_options, fn, name, block, index)

    monkeypatch.setattr(FakePatcher, "set_model_patch_replace", replace)
    model = FakePatcher()
    model.model_options = {"transformer_options": {}}
    model.model = SimpleNamespace(diffusion_model=type("MiniMaxH3Model", (), {"blocks": [object()] * 2})())
    def attention(m):
        return nodes.H3ForgeAttention().patch(m, "flex_sliding", 40, 8, 40, 2, .15,
                                              True, 2, 1.15, 6, 42, True)[0]

    def nag(m):
        return nodes.H3ForgeNAG().patch(m, [[torch.ones(1, 3, 8), {}]], "lite", 3, 2.5, .15,
                                       .7, 8, 28, 1, .5, True)[0]
    patches = {"attention": attention, "nag": nag}
    first = patches[order[0]](model)
    second = patches[order[1]](first)
    options = second.model_options["transformer_options"]
    state = options[nodes.STATE_GETTER]()
    copied = copy_options(options)
    assert copied["patches_replace"]["dit"][("double_block", 0)].state is state
    wrapper = next(fn for kind, key, fn in second.wrappers if kind == "diffusion" and key == nodes.ATTN_KEY)
    wrapper(lambda *args, **kwargs: None, [None, None], None, None, {}, minimax_payload={"layout": object()})
    assert state.policy.mode == "flex_sliding" and state.policy.feta_enabled
    assert state.nag.scale == 3
    first_options = first.model_options["transformer_options"]
    assert (nodes.NAG_KEY in first_options) == (order[0] == "nag")
    assert (nodes.POLICY_KEY in first_options) == (order[0] == "attention")


def test_stamp_preserves_previous_hook_dependencies_and_isolates_control_attention(monkeypatch):
    nodes = _import_nodes(monkeypatch)
    state = RuntimeState(AttentionPolicy())
    calls, dependency = [], object()

    class Previous:
        def __call__(self, args, extra):
            calls.append(("before", state.block_index))
            result = extra["original_block"](args)
            calls.append(("after", state.block_index))
            return result

        def models(self):
            return [dependency]

        def to(self, value):
            calls.append(("to", value))
            return self

        def cleanup(self):
            calls.append(("cleanup", None))

    stamp = nodes._BlockStamp(state, 7, Previous())
    def native(args):
        calls.append(("native", state.block_index))
        assert state.block_args is args
        return {"img": "result"}
    assert stamp({"transformer_options": {}}, {"original_block": native}) == {"img": "result"}
    assert calls == [("before", None), ("native", 7), ("after", None)]
    assert stamp.models() == [dependency]
    assert stamp.to("cpu") is stamp
    stamp.cleanup()
    assert calls[-2:] == [("to", "cpu"), ("cleanup", None)]
    assert state.block_args is None


def _context_inputs(total=67, dtype=torch.float32, text_lengths=(3,)):
    core = native_h3()
    frames = sum(core.FRAME_PER_TOKEN[i % 5] for i in range(total))
    audio_t = round(frames * 5 / 3)
    x = [torch.zeros(1, 1, total, 2, 2, dtype=dtype), torch.zeros(1, 1, 2, audio_t, dtype=dtype)]
    prompts = tuple(torch.full((1, n, 8), float(i + 1), dtype=dtype) for i, n in enumerate(text_lengths))
    payload = {"layout": core.PackedLayout(max(text_lengths), total, 2, 2, audio_t),
               "h3forge_prompt_segments": prompts,
               "h3forge_prompt_segment_tags": tuple(torch.ones(n, dtype=torch.long) for n in text_lengths)}
    return x, prompts[0], payload


def _executor(fn):
    class Executor:
        class_obj = SimpleNamespace(patch_size=(1, 2, 2))

        def __call__(self, *args, **kwargs):
            return fn(*args, **kwargs)
    return Executor()


@pytest.mark.parametrize("audio_context", ["auto", "window", "full"])
@pytest.mark.parametrize(
    "total,window,durations,single_window_beats",
    [
        (427, 80, (2, 18, 40), False),
        (217, 95, (1, 1, 1), True),
        (17, 25, (1, 1, 1), False),
    ],
)
def test_every_beat_owns_its_video_and_audio_output(
    total, window, durations, single_window_beats, audio_context, capsys
):
    x, context, payload = _context_inputs(total, text_lengths=(3, 19, 5))
    payload["h3forge_prompt_segment_durations"] = durations
    calls = []
    audio_lengths = set()
    expect_full = (
        total <= window
        or audio_context == "full"
        or (audio_context == "auto" and not single_window_beats)
    )

    def run(local_x, timestep, text, options, **kwargs):
        layout = kwargs["minimax_payload"]["layout"]
        assert layout.signature[0] == text.shape[1]
        assert layout.signature[4] == local_x[1].shape[-1]
        assert kwargs["minimax_payload"]["text_token_tags"].numel() == text.shape[1]
        if expect_full:
            assert local_x[1] is x[1]  # every utterance remains directly visible
        else:
            assert local_x[1].shape[-1] < x[1].shape[-1]
        audio_lengths.add(local_x[1].shape[-1])
        calls.append((int(text[0, 0, 0]), text.shape[1]))
        return [torch.full_like(t, float(text[0, 0, 0])) for t in local_x]

    policy = ContextPolicy(window, 10, True, "pyramid", segment_seams="exclusive", audio_context=audio_context)
    result = make_context_wrapper(policy)(
        _executor(run), x, torch.tensor([1000]), context,
        {"sigmas": torch.tensor([1.]), "sample_sigmas": torch.tensor([1., 0.])}, minimax_payload=payload)
    ranges, cuts = segment_ranges(total, 3, durations)
    audio_cuts = [round(f * 5 / 3) for f in cuts]
    audio_cuts[-1] = x[1].shape[-1]
    for index, (lo, hi) in enumerate(ranges):
        assert torch.all(result[0][:, :, lo:hi] == index + 1)
        assert torch.all(result[1][..., audio_cuts[index]:audio_cuts[index + 1]] == index + 1)
    assert set(calls) == {(1, 3), (2, 19), (3, 5)}
    assert len(audio_lengths) == 1
    log = capsys.readouterr().out
    assert "stagger=off" in log and "seams=exclusive" in log
    assert "prompt_frame_cuts=" in log
    if expect_full:
        assert f"audio_context=full:{x[1].shape[-1]}" in log
    else:
        (audio_win,) = audio_lengths
        assert f"audio_context=window:{audio_win}/{x[1].shape[-1]}" in log


def _prompt_valued_run(x, calls, *, expect_full_audio=True):
    def run(local_x, timestep, text, options, **kwargs):
        layout = kwargs["minimax_payload"]["layout"]
        assert layout.signature[4] == local_x[1].shape[-1]
        if expect_full_audio:
            assert local_x[1] is x[1]
        else:
            assert local_x[1].shape[-1] < x[1].shape[-1]
        calls.append(int(text[0, 0, 0]))
        return [torch.full_like(t, float(text[0, 0, 0])) for t in local_x]
    return run


def test_blended_seams_ramp_between_prompts_and_stagger(capsys):
    from h3forge.layout import audio_range_for_video_window
    total, window, overlap = 67, 25, 8
    x, context, payload = _context_inputs(total, text_lengths=(3, 19, 5))
    calls = []
    options = {"sigmas": torch.tensor([1.]), "sample_sigmas": torch.tensor([1., 0.])}
    result = make_context_wrapper(ContextPolicy(window, overlap, True, "pyramid"))(
        _executor(_prompt_valued_run(x, calls)), x, torch.tensor([1000]), context, options, minimax_payload=payload)
    # Phase-0 windows start at 0, 14, 28, 42 and mostly cover beats 1, 2, 2, 3 of [0,23) [23,44) [44,67).
    assert calls == [1, 2, 2, 3]
    video = result[0][0, 0, :, 0, 0]
    assert torch.all(video[:14] == 1) and torch.all(video[25:42] == 2) and torch.all(video[53:] == 3)
    assert torch.all((video[14:25] > 1) & (video[14:25] < 2))  # prompt 1 -> 2 is a ramp, not a cut
    assert torch.all((video[42:53] > 2) & (video[42:53] < 3))
    layout = payload["layout"]
    audio = result[1][0, 0, 0]
    assert torch.all(audio[:audio_range_for_video_window(layout, 14, 39)[0]] == 1)
    assert torch.all(audio[audio_range_for_video_window(layout, 28, 53)[1]:] == 3)
    assert not torch.all(audio == audio.round())  # the audio seam is blended too
    log = capsys.readouterr().out
    assert "stagger=on" in log and "seams=blend" in log and "prompt_windows=1x1,2x2,3x1" in log
    # Beat 2 spans two windows (2x2), so auto keeps full audio to preserve F11 inside Beat 2.
    assert "audio_context=full:377" in log

    # A later step moves the seams: the same graph, one sampler step further on.
    calls.clear()
    options = {"sigmas": torch.tensor([0.5]), "sample_sigmas": torch.tensor([1., 0.5, 0.])}
    shifted = make_context_wrapper(ContextPolicy(window, overlap, True, "pyramid"))(
        _executor(_prompt_valued_run(x, calls)), x, torch.tensor([500]), context, options, minimax_payload=payload)
    assert not torch.equal(shifted[0], result[0])


def test_blended_seams_rescue_beats_inside_a_single_window():
    total = 17
    x, context, payload = _context_inputs(total, text_lengths=(3, 19, 5))
    calls = []
    result = make_context_wrapper(ContextPolicy(25, 10, True, "pyramid"))(
        _executor(_prompt_valued_run(x, calls)), x, torch.tensor([1000]), context,
        {"sigmas": torch.tensor([1.]), "sample_sigmas": torch.tensor([1., 0.])}, minimax_payload=payload)
    ranges, cuts = segment_ranges(total, 3)
    # One clip-wide window carries the majority beat (beat 1 on a tie); beats 2 and 3
    # each get a rescue window that owns its interval outright, in video and audio,
    # so a short clip with a pipe prompt is not a mixture with the majority prompt.
    assert calls == [1, 2, 3]
    video = result[0][0, 0, :, 0, 0]
    audio_cuts = [round(f * 5 / 3) for f in cuts]
    audio_cuts[-1] = x[1].shape[-1]
    for index, (lo, hi) in enumerate(ranges):
        assert torch.all(video[lo:hi] == index + 1)
        assert torch.all(result[1][..., audio_cuts[index]:audio_cuts[index + 1]] == index + 1)


def test_blended_rescue_beat_is_owned_inside_a_multi_window_plan():
    total, window, overlap = 427, 80, 10
    x, context, payload = _context_inputs(total, text_lengths=(3, 19, 5))
    payload["h3forge_prompt_segment_durations"] = (2, 18, 40)
    calls = []
    result = make_context_wrapper(ContextPolicy(window, overlap, True, "pyramid"))(
        _executor(_prompt_valued_run(x, calls)), x, torch.tensor([1000]), context,
        {"sigmas": torch.tensor([1.]), "sample_sigmas": torch.tensor([1., 0.])}, minimax_payload=payload)
    ranges, cuts = segment_ranges(total, 3, (2, 18, 40))
    assert ranges[0][1] - ranges[0][0] < window - overlap  # beat 1 is shorter than the stride: rescued
    video = result[0][0, 0, :, 0, 0]
    lo, hi = ranges[0]
    assert torch.all(video[lo:hi] == 1)  # owned by its rescue window, no prompt-2 bleed
    assert torch.all(video[hi:hi + 5] > 1)  # the regular windows resume right after the beat
    audio_cut = round(cuts[1] * 5 / 3)
    assert torch.all(result[1][..., :audio_cut] == 1)
    assert calls.count(1) == 1


@pytest.mark.parametrize(
    "total,window,overlap,text_lengths,durations,mode,expect_full",
    [
        # Single prompt: auto keeps full audio; explicit window slices.
        (67, 25, 8, (5,), None, "auto", True),
        (67, 25, 8, (5,), None, "window", False),
        # Multi-window beats on pipe timelines: auto keeps full audio so a beat
        # never loses its own earlier speech (F11 inside a multi-window beat).
        (427, 100, 13, (3, 5, 4), (20, 20, 20), "auto", True),
        (322, 95, 13, (3, 5, 4), (20, 5, 20), "auto", True),
        (217, 85, 13, (3, 5, 4), (13, 4, 13), "auto", True),
        # Beats shorter than window (22, 23, 22 <= 25), yet 4 windows assign Beat 2 twice (1x1,2x2,3x1).
        (67, 25, 8, (3, 5, 4), None, "auto", True),
        # Distinct window assignments ([0, 1, 2, 3]), yet Beat 1 exceeds window (98 > 95).
        (322, 95, 13, (3, 5, 4, 6), (14, 10, 11, 11), "auto", True),
        # One beat per window and every beat fits within window: auto slices audio.
        (322, 95, 13, (3, 5, 4, 6), (12, 11, 12, 11), "auto", False),
        (217, 85, 13, (3, 5, 4), (10, 10, 10), "auto", False),
        (142, 80, 13, (3, 7), (10, 10), "auto", False),
        # Single-window clip with pipe prompt stays on full audio.
        (98, 105, 13, (3, 5, 4), (5, 5, 4), "auto", True),
        # Explicit overrides win in either direction.
        (322, 95, 13, (3, 5, 4, 6), (12, 11, 12, 11), "full", True),
        (427, 100, 13, (3, 5, 4), (20, 20, 20), "window", False),
    ],
)
def test_audio_context_auto_windows_only_single_window_beats(
    total, window, overlap, text_lengths, durations, mode, expect_full, capsys
):
    from h3forge.layout import audio_window_length

    options = {"sigmas": torch.tensor([1.]), "sample_sigmas": torch.tensor([1., 0.])}
    x, context, payload = _context_inputs(total, text_lengths=text_lengths)
    if durations is not None:
        payload["h3forge_prompt_segment_durations"] = durations
    audio_t = x[1].shape[-1]
    expected_win = audio_t if expect_full else audio_window_length(payload["layout"], min(total, window))
    audio_mask = torch.linspace(0.0, 1.0, audio_t).view(1, 1, 1, audio_t).expand(1, 1, 2, audio_t).contiguous()
    seen_lengths = set()

    def run(local_x, timestep, text, opts, **kwargs):
        layout = kwargs["minimax_payload"]["layout"]
        mask = kwargs["audio_denoise_mask"]
        seen_lengths.add(local_x[1].shape[-1])
        if expect_full:
            assert local_x[1] is x[1]
            assert mask is audio_mask
            assert layout._h3forge_audio_offset == 0
        else:
            offset = layout._h3forge_audio_offset
            assert local_x[1].shape[-1] == expected_win
            assert torch.equal(mask, audio_mask[..., offset:offset + expected_win])
        return [torch.ones_like(local_x[0]), torch.ones_like(local_x[1])]

    make_context_wrapper(ContextPolicy(window, overlap, True, "pyramid", audio_context=mode))(
        _executor(run), x, torch.tensor([1000]), context, options,
        minimax_payload=payload, audio_denoise_mask=audio_mask)
    log = capsys.readouterr().out
    assert seen_lengths == {expected_win}
    if expect_full:
        assert f"audio_context=full:{audio_t}" in log
    else:
        assert f"audio_context=window:{expected_win}/{audio_t}" in log


@pytest.mark.parametrize("text_lengths,mode", [((3, 5, 4, 6, 4, 5), "auto"), ((3, 5, 4), "window")])
def test_windowed_audio_preserves_exact_global_ticks_across_stagger_and_tail_clamp(text_lengths, mode):
    from h3forge.layout import target_segments

    total, window, overlap = 427, 80, 10
    x, context, payload = _context_inputs(total, text_lengths=text_lengths)
    full_layout = payload["layout"]
    full_seg = target_segments(full_layout)
    audio_t = x[1].shape[-1]
    ticks = torch.arange(audio_t, dtype=torch.float32)
    x[1] = torch.stack([ticks, ticks + 10000.0], dim=0).view(1, 1, 2, audio_t)
    audio_mask = torch.linspace(0.0, 1.0, audio_t).view(1, 1, 1, audio_t).expand(1, 1, 2, audio_t).contiguous()

    for sigma, schedule in ((1.0, [1.0, 0.5, 0.0]), (0.5, [1.0, 0.5, 0.0])):
        offsets = []

        def echo(local_x, timestep, text, opts, **kwargs):
            local_layout = kwargs["minimax_payload"]["layout"]
            local_seg = target_segments(local_layout)
            assert local_x[1].shape[-1] == 455
            assert local_layout.signature[4] == 455
            offset = local_layout._h3forge_audio_offset
            offsets.append(offset)
            assert torch.equal(local_x[1], x[1][..., offset:offset + 455])
            assert torch.equal(kwargs["audio_denoise_mask"], audio_mask[..., offset:offset + 455])
            assert torch.equal(
                local_layout.position_ids[local_seg.audio_start:local_seg.audio_start + 455],
                full_layout.position_ids[full_seg.audio_start + offset:full_seg.audio_start + offset + 455],
            )
            assert torch.equal(
                local_layout.position_ids[local_seg.audio_start + 455:local_seg.audio_stop],
                full_layout.position_ids[
                    full_seg.audio_start + audio_t + offset:full_seg.audio_start + audio_t + offset + 455
                ],
            )
            return [local_x[0].clone(), local_x[1].clone()]

        options = {"sigmas": torch.tensor([sigma]), "sample_sigmas": torch.tensor(schedule)}
        out = make_context_wrapper(ContextPolicy(window, overlap, True, "pyramid", audio_context=mode))(
            _executor(echo), x, torch.tensor([sigma * 1000]), context, options,
            minimax_payload=payload, audio_denoise_mask=audio_mask)
        # Tail window [347, 427) spans [1963, 2417) = 454 ticks, so its input clamps to [1962, 2417).
        assert offsets[0] == 0 and offsets[-1] == audio_t - 455
        assert torch.allclose(out[1], x[1], atol=1e-5, rtol=0.0)


def test_window_failure_never_retries_full_clip():
    x, context, payload = _context_inputs()
    calls = []
    def fail(local_x, *args, **kwargs):
        calls.append(local_x[0].shape[2])
        raise RuntimeError("model failure")
    with pytest.raises(RuntimeError, match="window .*model failure"):
        make_context_wrapper(ContextPolicy(25, 8, False, "pyramid"))(
            _executor(fail), x, None, context, {}, minimax_payload=payload)
    assert calls == [25]


@pytest.mark.parametrize("dtype,total,window,overlap,value", [
    (torch.bfloat16, 67, 25, 8, 1.234375), (torch.float16, 427, 257, 256, 8.),
])
def test_fusion_preserves_identical_predictions(dtype, total, window, overlap, value):
    x, context, payload = _context_inputs(total, dtype)
    def predict(local_x, *args, **kwargs):
        return [torch.full_like(t, value) for t in local_x]
    out = make_context_wrapper(ContextPolicy(window, overlap, False, "pyramid"))(
        _executor(predict), x, None, context, {}, minimax_payload=payload)
    for stream in out:
        assert stream.dtype == dtype and torch.all(stream == value)


def test_strict_sparse_contracts_raise_but_declared_dense_schedule_runs():
    state = RuntimeState(AttentionPolicy(strict=True, first_dense_layers=2))
    state.block_index = 3
    state.layout = SimpleNamespace(seq_len=99)
    q = torch.zeros(1, 2, 5, 4)
    override = make_attention_override(state)
    with pytest.raises(RuntimeError, match="layout-length-mismatch"):
        override(lambda *args, **kwargs: q, q, q, q, 2, skip_reshape=True)
    state.layout.seq_len = 5
    state.block_index = 0
    assert override(lambda *args, **kwargs: q, q, q, q, 2, skip_reshape=True) is q


def test_different_shared_reference_values_are_rejected():
    inputs = [[[torch.ones(1, 3, 8), {"minimax_refs": [{"latent": torch.tensor([value])}]}]] for value in (1, 2)]
    with pytest.raises(ValueError, match="different shared minimax_refs"):
        combine_conditioning_segments(inputs)


@pytest.mark.parametrize("control_first", [True, False])
def test_native_control_wrappers_receive_global_control_slices(control_first):
    cls = definitions("comfy_extras/nodes_minimax_h3.py", ["MiniMaxH3FunControlPatch"],
                      {"torch": torch})["MiniMaxH3FunControlPatch"]
    executor = definitions("comfy/patcher_extension.py", ["WrapperExecutor"],
                           {"Callable": Callable})["WrapperExecutor"]
    patch = cls(None, None, None, None, None, 1., 1., 0.)
    prepared, seen = [], []

    def prepare(shape):
        shape = tuple(shape)
        if patch.control_latent is not None and patch.control_latent_shape == shape:
            return
        prepared.append(shape[2])
        patch.control_latent = torch.arange(shape[2]).view(1, 1, -1, 1, 1).expand(shape)
        patch.control_latent_shape = shape
    patch.prepare_control_latent = prepare
    x, context, payload = _context_inputs()

    def forward(local_x, timestep, text, options, **kwargs):
        layout = kwargs["minimax_payload"]["layout"]
        start = layout._h3forge_video_offset
        assert patch.active
        assert patch.control_latent_shape == tuple(local_x[0].shape)
        assert torch.equal(patch.control_latent[0, 0, :, 0, 0], torch.arange(start, start + local_x[0].shape[2]))
        seen.append(start)
        return local_x

    window = make_context_wrapper(ContextPolicy(25, 8, False, "pyramid"))
    wrappers = [patch.diffusion_model_wrapper, window] if control_first else [window, patch.diffusion_model_wrapper]
    executor.new_class_executor(forward, SimpleNamespace(patch_size=(1, 2, 2)), wrappers).execute(
        x, torch.tensor([700.]), context, {"sigmas": torch.tensor([.7])}, minimax_payload=payload)
    assert prepared == [67] and seen[0] == 0 and seen[-1] > 0
    assert patch.control_latent_shape == tuple(x[0].shape)
    assert patch.control_stream is None

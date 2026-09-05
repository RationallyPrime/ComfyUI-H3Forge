from types import SimpleNamespace
from typing import Callable

import pytest
import torch

from core_source import definitions, native_h3
from h3forge.attention import make_attention_override
from h3forge.context import ContextPolicy, make_context_wrapper
from h3forge.prompt import combine_conditioning_segments, segment_ranges
from h3forge.state import AttentionPolicy, RuntimeState
from test_nodes import FakePatcher, _import_nodes


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


@pytest.mark.parametrize("total,window,durations", [(427, 80, (2, 18, 40)), (17, 25, (1, 1, 1))])
def test_every_beat_owns_its_video_and_audio_output(total, window, durations, capsys):
    x, context, payload = _context_inputs(total, text_lengths=(3, 19, 5))
    payload["h3forge_prompt_segment_durations"] = durations
    calls = []

    def run(local_x, timestep, text, options, **kwargs):
        layout = kwargs["minimax_payload"]["layout"]
        assert layout.signature[0] == text.shape[1]
        assert kwargs["minimax_payload"]["text_token_tags"].numel() == text.shape[1]
        assert local_x[1] is x[1]  # every utterance remains directly visible
        calls.append((int(text[0, 0, 0]), text.shape[1]))
        return [torch.full_like(t, float(text[0, 0, 0])) for t in local_x]

    result = make_context_wrapper(ContextPolicy(window, 10, True, "pyramid", True))(
        _executor(run), x, torch.tensor([1000]), context,
        {"sigmas": torch.tensor([1.]), "sample_sigmas": torch.tensor([1., 0.])}, minimax_payload=payload)
    ranges, cuts = segment_ranges(total, 3, durations)
    audio_cuts = [round(f * 5 / 3) for f in cuts]
    audio_cuts[-1] = x[1].shape[-1]
    for index, (lo, hi) in enumerate(ranges):
        assert torch.all(result[0][:, :, lo:hi] == index + 1)
        assert torch.all(result[1][..., audio_cuts[index]:audio_cuts[index + 1]] == index + 1)
    assert set(calls) == {(1, 3), (2, 19), (3, 5)}
    log = capsys.readouterr().out
    assert "stagger=off" in log and "prompt_frame_cuts=" in log and "audio_context=full:" in log


def test_window_failure_never_retries_full_clip():
    x, context, payload = _context_inputs()
    calls = []
    def fail(local_x, *args, **kwargs):
        calls.append(local_x[0].shape[2])
        raise RuntimeError("model failure")
    with pytest.raises(RuntimeError, match="window .*model failure"):
        make_context_wrapper(ContextPolicy(25, 8, False, "pyramid", False))(
            _executor(fail), x, None, context, {}, minimax_payload=payload)
    assert calls == [25]


@pytest.mark.parametrize("dtype,total,window,overlap,value", [
    (torch.bfloat16, 67, 25, 8, 1.234375), (torch.float16, 427, 257, 256, 8.),
])
def test_fusion_preserves_identical_predictions(dtype, total, window, overlap, value):
    x, context, payload = _context_inputs(total, dtype)
    def predict(local_x, *args, **kwargs):
        return [torch.full_like(t, value) for t in local_x]
    out = make_context_wrapper(ContextPolicy(window, overlap, False, "pyramid", True))(
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

    window = make_context_wrapper(ContextPolicy(25, 8, False, "pyramid", True))
    wrappers = [patch.diffusion_model_wrapper, window] if control_first else [window, patch.diffusion_model_wrapper]
    executor.new_class_executor(forward, SimpleNamespace(patch_size=(1, 2, 2)), wrappers).execute(
        x, torch.tensor([700.]), context, {"sigmas": torch.tensor([.7])}, minimax_payload=payload)
    assert prepared == [67] and seen[0] == 0 and seen[-1] > 0
    assert patch.control_latent_shape == tuple(x[0].shape)
    assert patch.control_stream is None

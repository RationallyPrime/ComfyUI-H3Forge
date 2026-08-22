import torch
from h3forge.state import resolve_step


def test_resolve_step():
    opts = {"sample_sigmas": torch.tensor([1.0, 0.7, 0.3, 0.0]), "sigmas": torch.tensor([0.69])}
    assert resolve_step(opts) == (1, 3)


def test_forward_wrapper_clears_stale_block_index(monkeypatch):
    import importlib
    import sys
    import types

    comfy = types.ModuleType("comfy")
    patcher_extension = types.ModuleType("comfy.patcher_extension")

    class WrappersMP:
        OUTER_SAMPLE = "outer"
        DIFFUSION_MODEL = "diffusion"

    patcher_extension.WrappersMP = WrappersMP
    comfy.patcher_extension = patcher_extension
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", patcher_extension)
    nodes = importlib.import_module("h3forge.nodes")

    from h3forge.state import AttentionPolicy, RuntimeState

    state = RuntimeState(AttentionPolicy(feta_enabled=True, feta_last_layer=49))
    state.block_index = 49
    layout = object()

    def executor(x, timestep, context, transformer_options, **kwargs):
        assert state.block_index is None
        state.block_index = 49  # emulate the final stamped DiT block
        return "ok"

    wrapped = nodes._forward_wrapper(state)
    options = {}
    assert wrapped(executor, [None, None], None, torch.zeros(1, 1, 5376), options,
                   minimax_payload={"layout": layout}) == "ok"
    assert state.block_index is None
    assert "h3forge_active_layout" not in options

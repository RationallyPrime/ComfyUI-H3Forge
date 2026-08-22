import copy
import importlib
import sys
import types
from types import SimpleNamespace

import torch


def _import_nodes(monkeypatch):
    comfy = types.ModuleType("comfy")
    patcher_extension = types.ModuleType("comfy.patcher_extension")

    class WrappersMP:
        OUTER_SAMPLE = "outer"
        DIFFUSION_MODEL = "diffusion"

    patcher_extension.WrappersMP = WrappersMP
    comfy.patcher_extension = patcher_extension
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", patcher_extension)
    return importlib.import_module("h3forge.nodes")


class FakePatcher:
    def __init__(self):
        self.model_options = {}
        self.wrappers = []
        self.replacements = {}
        self.model = None

    def clone(self):
        clone = FakePatcher()
        # Mirrors ComfyUI's ModelPatcher.clone: model_options are deep-copied
        # (functions survive as themselves), patches are carried over.
        clone.model_options = copy.deepcopy(self.model_options)
        clone.wrappers = list(self.wrappers)
        clone.replacements = dict(self.replacements)
        clone.model = self.model
        return clone

    def add_wrapper_with_key(self, kind, key, wrapper):
        self.wrappers.append((kind, key, wrapper))

    def set_model_patch_replace(self, fn, *keys):
        self.replacements[keys] = fn


def test_h3forge_nodes_share_one_runtime_regardless_of_order(monkeypatch):
    nodes = _import_nodes(monkeypatch)
    diffusion = SimpleNamespace(blocks=[object(), object(), object()])
    base = FakePatcher()

    first, first_opts = nodes._acquire_runtime(base, diffusion)
    second, second_opts = nodes._acquire_runtime(first, diffusion)
    state_a = first_opts[nodes.STATE_GETTER]()
    state_b = second_opts[nodes.STATE_GETTER]()
    assert state_a is state_b
    assert state_a.blocks is diffusion.blocks
    assert state_a.default_policy is not None
    # Machinery is installed exactly once; the second clone inherits it.
    assert len(first.replacements) == 3
    assert len(second.replacements) == 3
    assert len(second.wrappers) == 2
    assert (second.model_options["transformer_options"]["optimized_attention_override"]
            is first.model_options["transformer_options"]["optimized_attention_override"])
    # The upstream model remains unpatched.
    assert base.model_options == {}


def test_nag_and_attention_configs_stay_per_branch(monkeypatch):
    nodes = _import_nodes(monkeypatch)

    class MiniMaxH3Model:
        def __init__(self):
            self.blocks = [object()] * 4

    model = FakePatcher()
    model.model = SimpleNamespace(diffusion_model=MiniMaxH3Model())

    negative = [[torch.zeros(1, 5, 8), {}]]
    (nag_only,) = nodes.H3ForgeNAG().patch(
        model, negative, "lite", 3.0, 2.5, 0.15, 0.70, 8, 28, 1.0, 0.5, False)
    (with_both,) = nodes.H3ForgeAttention().patch(
        nag_only, "flex_sliding", 40.0, 8.0, 40, 2, 0.15, False, 2.0, 1.15, 6, 42, False)

    nag_opts = nag_only.model_options["transformer_options"]
    both_opts = with_both.model_options["transformer_options"]
    state = nag_opts[nodes.STATE_GETTER]()
    assert state is both_opts[nodes.STATE_GETTER]()
    # Configuration is per model clone: applying the attention node downstream
    # must not leak a policy back into the NAG-only branch's options.
    assert nodes.POLICY_KEY not in nag_opts
    assert both_opts[nodes.POLICY_KEY].mode == "flex_sliding"
    assert nag_opts[nodes.NAG_KEY].scale == 3.0
    assert both_opts[nodes.NAG_KEY].scale == 3.0


def test_forward_wrapper_resolves_config_from_the_sampled_clone(monkeypatch):
    nodes = _import_nodes(monkeypatch)

    class MiniMaxH3Model:
        def __init__(self):
            self.blocks = [object()] * 4

    model = FakePatcher()
    model.model = SimpleNamespace(diffusion_model=MiniMaxH3Model())

    negative = [[torch.zeros(1, 5, 8), {}]]
    (nag_only,) = nodes.H3ForgeNAG().patch(
        model, negative, "lite", 3.0, 2.5, 0.15, 0.70, 8, 28, 1.0, 0.5, False)
    (with_both,) = nodes.H3ForgeAttention().patch(
        nag_only, "flex_sliding", 40.0, 8.0, 40, 2, 0.15, False, 2.0, 1.15, 6, 42, False)
    nag_opts = nag_only.model_options["transformer_options"]
    both_opts = with_both.model_options["transformer_options"]
    state = nag_opts[nodes.STATE_GETTER]()

    wrapper = nodes._forward_wrapper(state)
    layout = object()

    def executor(x, timestep, context, transformer_options, **kwargs):
        return (state.policy.mode, state.nag is not None)

    def run(options):
        return wrapper(executor, [None, None], None, None, dict(options),
                       minimax_payload={"layout": layout})

    assert run(both_opts) == ("flex_sliding", True)
    # The NAG-only branch runs with the dense default policy, not with a policy
    # a sibling node applied elsewhere.
    assert run(nag_opts) == ("dense", True)
    # A clone whose options carry no NAG key (e.g. the NAG node was bypassed and
    # ComfyUI reused a cached upstream output) must not inherit stale NAG state.
    plain = {key: value for key, value in both_opts.items() if key != nodes.NAG_KEY}
    assert run(plain) == ("flex_sliding", False)

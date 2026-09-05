import importlib
import sys
import types
from types import SimpleNamespace

import pytest
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
        def copy_options(value):
            if isinstance(value, dict):
                return {k: copy_options(v) for k, v in value.items()}
            if isinstance(value, list):
                return [copy_options(v) for v in value]
            return value
        clone.model_options = copy_options(self.model_options)
        clone.wrappers = list(self.wrappers)
        clone.replacements = dict(self.replacements)
        clone.model = self.model
        return clone

    def add_wrapper_with_key(self, kind, key, wrapper):
        self.wrappers.append((kind, key, wrapper))

    def remove_wrappers_with_key(self, kind, key):
        self.wrappers = [entry for entry in self.wrappers if entry[:2] != (kind, key)]

    def set_model_patch_replace(self, fn, *keys):
        self.replacements[keys] = fn
        name, block, index = keys
        opts = self.model_options.setdefault("transformer_options", {}).copy()
        opts["patches_replace"] = opts.get("patches_replace", {}).copy()
        opts["patches_replace"][name] = opts["patches_replace"].get(name, {}).copy()
        opts["patches_replace"][name][(block, index)] = fn
        self.model_options["transformer_options"] = opts


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


def test_context_node_exposes_core_parity_defaults(monkeypatch):
    nodes = _import_nodes(monkeypatch)
    required = nodes.H3ForgeContextWindows.INPUT_TYPES()["required"]
    assert required["overlap_frames"][1]["default"] == 8
    assert required["blend"][0] == ["pyramid", "overlap-linear", "flat"]
    assert required["blend"][1]["default"] == "pyramid"


@pytest.mark.parametrize("stride", [1, 2])
def test_context_node_rejects_too_small_a_stride_for_stagger(monkeypatch, stride):
    nodes = _import_nodes(monkeypatch)
    monkeypatch.setattr(nodes, "_require_h3", lambda model: object())
    with pytest.raises(ValueError, match="stride of at least 3"):
        nodes.H3ForgeContextWindows().patch(
            object(), 25, 25 - stride, True, "pyramid", False,
        )


def test_context_node_allows_small_stride_when_stagger_is_off(monkeypatch):
    nodes = _import_nodes(monkeypatch)

    class MinimalPatcher:
        model = object()

        def clone(self):
            return self

        @staticmethod
        def get_model_object(name):
            assert name == "extra_conds"
            return lambda **kwargs: kwargs

        @staticmethod
        def add_object_patch(name, value):
            assert name == "extra_conds"
            assert callable(value)

        @staticmethod
        def add_wrapper_with_key(kind, key, wrapper):
            assert kind == "diffusion"
            assert key == nodes.CTX_KEY
            assert callable(wrapper)

    monkeypatch.setattr(nodes, "_require_h3", lambda model: object())
    model = MinimalPatcher()
    assert nodes.H3ForgeContextWindows().patch(
        model, 25, 23, False, "pyramid", False,
    ) == (model,)
    assert nodes.H3ForgeContextWindows().patch(
        model, 25, 22, True, "pyramid", False,
    ) == (model,)


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


def test_bound_forward_wrapper_survives_runtime_option_reconstruction(monkeypatch):
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
        nag_only, "flex_sliding", 40.0, 8.0, 40, 2, 0.15,
        False, 2.0, 1.15, 6, 42, False)

    state = with_both.model_options["transformer_options"][nodes.STATE_GETTER]()
    layout = object()

    def run(patcher):
        wrappers = [
            wrapper for kind, key, wrapper in patcher.wrappers
            if kind == "diffusion" and key == nodes.ATTN_KEY
        ]
        assert len(wrappers) == 1

        def executor(x, timestep, context, transformer_options, **kwargs):
            return state.policy.mode, state.nag is not None

        # Mirrors ComfyUI's conditioned runtime path: machinery such as the
        # wrapper survives, while arbitrary H3Forge config keys may be absent.
        return wrappers[0](
            executor,
            [None, None],
            None,
            None,
            {},
            minimax_payload={"layout": layout},
        )

    assert run(with_both) == ("flex_sliding", True)
    assert run(nag_only) == ("dense", True)


def test_reference_pipe_node_prepares_native_refs_once(monkeypatch):
    nodes = _import_nodes(monkeypatch)
    core = types.ModuleType("comfy_extras.nodes_minimax_h3")
    calls, texts = [], []
    shared = [{"kind": "image", "latent": torch.ones(1)}]

    class Clip:
        def tokenize(self, text, **kwargs):
            texts.append((text, kwargs))
            return text

        def encode_from_tokens_scheduled(self, text):
            count = len(text.split())
            return [[torch.ones(1, count, 3), {"minimax_token_tags": torch.ones(count, dtype=torch.long)}]]

    class MiniMaxH3ReferenceToVideo:
        @classmethod
        def execute(cls, **kwargs):
            calls.append(kwargs)
            tokens = kwargs["clip"].tokenize(kwargs["prompt"], minimax_ref_items=["prepared image and voice"])
            conditioning = kwargs["clip"].encode_from_tokens_scheduled(tokens)
            conditioning[0][1]["minimax_refs"] = shared
            return conditioning, {"samples": "native-latent"}

    core.MiniMaxH3ReferenceToVideo = MiniMaxH3ReferenceToVideo
    monkeypatch.setitem(sys.modules, "comfy_extras", types.ModuleType("comfy_extras"))
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_minimax_h3", core)
    image, voice, video = object(), object(), object()
    conditioning, latent = nodes.H3ForgeReferencePipePrompt().encode(
        clip=Clip(), vae=object(), audio_vae=object(), ref_image_1=image,
        ref_audio_1=voice, ref_video_2=video, ref_video_audio_2=voice,
        prompt="short segment | a deliberately longer segment", width=864, height=480, length=124,
        global_prompt="same woman and red coat", segment_durations="2,8")
    assert len(calls) == 1
    assert calls[0]["ref_images"] == {"ref_image_1": image}
    assert calls[0]["ref_audios"] == {"ref_audio_1": voice}
    assert calls[0]["ref_videos"] == {"ref_video_2": video}
    assert calls[0]["ref_video_audios"] == {"ref_video_audio_2": voice}
    assert [text for text, _ in texts] == ["same woman and red coat\n\nshort segment",
                                           "same woman and red coat\n\na deliberately longer segment"]
    assert texts[0][1] == texts[1][1]
    assert conditioning[0][1]["minimax_refs"] is shared
    assert conditioning[0][1]["h3forge_prompt_segment_durations"] == (2, 8)
    assert latent == {"samples": "native-latent"}

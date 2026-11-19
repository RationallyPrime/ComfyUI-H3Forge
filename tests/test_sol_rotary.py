from contextlib import contextmanager, nullcontext
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch

from core_source import definitions
from test_nodes import FakePatcher, _import_nodes


@pytest.fixture
def native(monkeypatch, tmp_path):
    source = os.environ.get("H3FORGE_SOL_CORE_SOURCE")
    if not source:
        pytest.skip("requires the audited ComfyUI 0.45.0 source")
    files = ("comfy/ldm/minimax/model.py", "comfy_extras/nodes_sparse_attention.py")
    for name in files:
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(source) / name, dest)
    patch = Path(__file__).resolve().parents[1] / "patches/comfyui-0.45.0-h3-inference.patch"
    subprocess.run(["git", "apply", str(patch)], cwd=tmp_path, check=True)
    monkeypatch.setenv("H3FORGE_COMFY_SOURCE", str(tmp_path))
    nodes = _import_nodes(monkeypatch)
    prefetch = SimpleNamespace(pause_malloc_graph=nullcontext, make_prefetch_queue=lambda *a: None,
                               prefetch_queue_pop=lambda *a, **kw: None)
    comfy = SimpleNamespace(model_prefetch=prefetch, model_management=SimpleNamespace(
        cast_to=lambda tensor, device: tensor.to(device)))
    monkeypatch.setitem(sys.modules, "comfy.model_prefetch", prefetch)
    env = definitions(files[0], ["MiniMaxH3Model", "rope_rotation_table"],
                      {"torch": torch, "nn": torch.nn, "comfy": comfy})
    sparse = definitions(files[1], ["SparseAttnPatch"])["SparseAttnPatch"]
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_sparse_attention",
                        SimpleNamespace(SparseAttnPatch=sparse))
    model = env["MiniMaxH3Model"].__new__(env["MiniMaxH3Model"])
    torch.nn.Module.__init__(model)
    model.rope = torch.nn.Module()
    model.rope.register_buffer("inv_freq", torch.linspace(0.01, 0.1, 16))
    patcher = FakePatcher()
    patcher.model = SimpleNamespace(diffusion_model=model)
    monkeypatch.setattr(FakePatcher, "get_model_object", lambda self, key: getattr(self.model, key), raising=False)
    return nodes.H3ForgeSolRotary(), patcher, model, sparse


@pytest.mark.parametrize("sigma_on,rope_on", [(False, False), (True, False), (False, True), (True, True)])
def test_native_decisions_and_tables(native, monkeypatch, sigma_on, rope_on):
    node, patcher, model, sparse_type = native
    sparse = sparse_type(1.3, 0, False, 0.8, 0.2, 100, {2}, "exact_kv_and_rows", 256, False)
    patched, = node.patch(patcher, sigma_on, rope_on)
    wrappers = {kind: fn for kind, _, fn in patched.wrappers}
    layout = SimpleNamespace(position_ids=torch.arange(24, dtype=torch.float64).reshape(8, 3))
    h = torch.zeros(1, dtype=torch.bfloat16)
    options = {"sigmas": torch.tensor([0.5])}
    seen_tables = []

    def block(x, *args, transformer_options):
        seen_tables.append(args[-1])
        return x

    model.blocks = [block]
    original = sparse_type.sigma_window_reason
    reads = []
    def count_read(self, sigmas):
        reads.append(sigmas)
        return original(self, sigmas)
    monkeypatch.setattr(sparse_type, "sigma_window_reason", count_read)
    build = model._rotation_table
    builds = []
    def count_table(*args):
        builds.append(None)
        return build(*args)
    monkeypatch.setattr(model, "_rotation_table", count_table)

    def eval_step(x, timestep, context, transformer_options):
        expected = original(sparse, options["sigmas"])
        for index in range(50):
            reason = sparse.dense_reason(transformer_options, 120, index)
            assert reason == (expected or ("block 2 in dense_blocks" if index == 2 else None))
        assert sparse.dense_reason(transformer_options, 99, 0) == (expected or "99 tokens < min_tokens 100")
        model._run_blocks(h, None, [], layout, transformer_options)
        assert torch.equal(seen_tables[-1], build(layout, h.device, h.dtype))

    def sample():
        for value in [0.9, 0.8, 0.5, 0.2, 0.1]:
            options["sigmas"].fill_(value)
            wrappers.get("diffusion", lambda fn, *a: fn(*a))(eval_step, None, None, None, options)
            assert "h3forge_sol_window" not in options
            assert "h3forge_rotary_cache" not in options

    for _ in range(2):
        wrappers.get("outer", lambda fn: fn())(sample)
    assert len(reads) == (10 if sigma_on else 510)
    assert len(builds) == (2 if rope_on else 10)


@pytest.mark.parametrize("inference", [False, True])
def test_rotary_invalidation_and_interrupted_run(native, inference):
    node, patcher, model, _ = native
    wrappers = {kind: fn for kind, _, fn in node.patch(patcher, True, True)[0].wrappers}
    sol = importlib.import_module("h3forge.sol_rotary")
    with torch.inference_mode(inference):
        layout = SimpleNamespace(position_ids=torch.ones(3, 3))
        model.rope.inv_freq = model.rope.inv_freq.clone()
        cache = sol.RotaryCache()
        cpu = torch.device("cpu")
        first = cache.table(model, layout, cpu, torch.bfloat16)
        assert first is cache.table(model, layout, cpu, torch.bfloat16)
        layout.position_ids.add_(3)
        moved = cache.table(model, layout, cpu, torch.bfloat16)
        assert not torch.equal(first, moved)
        model.rope.inv_freq.mul_(2)
        new = cache.table(model, layout, cpu, torch.bfloat16)
        assert torch.equal(new, model._rotation_table(layout, cpu, torch.bfloat16))
        assert not torch.equal(new, moved)
        assert cache.table(model, layout, cpu, torch.float32).dtype == torch.float32
        other = SimpleNamespace(position_ids=layout.position_ids.clone())
        assert cache.table(model, other, cpu, torch.float32) is not cache.table(model, layout, cpu, torch.float32)

    options = {}
    def fail(x, timestep, context, opts):
        assert "h3forge_rotary_cache" in opts
        raise ValueError("aborted")
    with pytest.raises(ValueError, match="aborted"):
        wrappers["outer"](lambda: wrappers["diffusion"](fail, None, None, None, options))
    assert options == {}
    def outside_run(x, t, c, opts):
        assert "h3forge_rotary_cache" not in opts

    wrappers["diffusion"](outside_run, None, None, None, options)


def test_rotary_replacement_releases_persistent_tensors_inside_pause(native, monkeypatch):
    _, _, model, _ = native
    from h3forge.sol_rotary import RotaryCache

    paused = False
    @contextmanager
    def pause():
        nonlocal paused
        paused = True
        try:
            yield
        finally:
            paused = False

    monkeypatch.setattr(sys.modules["comfy.model_prefetch"], "pause_malloc_graph", pause)
    cache = RotaryCache()
    drops = []
    with torch.inference_mode():
        a, b = (SimpleNamespace(position_ids=torch.full((4, 3), i)) for i in (1, 2))
        model.rope.inv_freq = model.rope.inv_freq.clone()
        def collect(label):
            drops.append((label, paused))
        for i, layout in enumerate([a, b, a]):
            result = cache.table(model, layout, torch.device("cpu"), torch.float32)
            assert torch.equal(result, model._rotation_table(layout, torch.device("cpu"), torch.float32))
            for label, tensor in [(f"table{i}", result), (f"pos{i}", cache.key[-2].snapshot),
                                  (f"freq{i}", cache.key[-1].snapshot)]:
                weakref.finalize(tensor, collect, label)
            del result, tensor
    assert sorted(drops) == sorted((f"{name}{i}", True) for name in ("table", "pos", "freq") for i in (0, 1))

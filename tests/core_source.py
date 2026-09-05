"""Load the small native contracts under test without initializing CUDA/models."""
import ast
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


def definitions(file, names, env=None):
    root = os.environ.get("H3FORGE_COMFY_SOURCE")
    if not root:
        pytest.skip("set H3FORGE_COMFY_SOURCE to the ComfyUI checkout; CI always runs these contracts")
    path = Path(root) / file
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    env = dict(env or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
    return env


def native_h3():
    def sdpa(q, k, v, heads, **kwargs):
        out = F.scaled_dot_product_attention(q, k, v)
        return out if kwargs.get("skip_output_reshape") else out.transpose(1, 2).flatten(2)

    def attention(q, k, v, heads, **kwargs):
        q, k, v = (x.take() for x in (q, k, v))
        override = kwargs.get("transformer_options", {}).get("optimized_attention_override")
        return override(sdpa, q, k, v, heads, **kwargs) if override else sdpa(q, k, v, heads, **kwargs)

    def rms_rope(q, k, rope, qw, kw, epsilon, rot_dim):
        # CPU reference for Comfy Kitchen's fused kernel. The native Attention
        # class still decides weights, epsilon, rotary extent, shape and order.
        for x, weight in ((q, qw), (k, kw)):
            normalized = F.rms_norm(x, (x.shape[-1],), weight, epsilon)
            half = rot_dim // 2
            pairs = torch.stack((normalized[..., :half], normalized[..., half:rot_dim]), dim=-1)
            rotated = torch.matmul(rope, pairs.unsqueeze(-1)).squeeze(-1)
            x.copy_(torch.cat((rotated[..., 0], rotated[..., 1], normalized[..., rot_dim:]), dim=-1))

    def linear_act(layer, x, act):
        assert act == "swiglu"
        a, b = x.chunk(2, dim=-1)
        return layer(F.silu(a) * b)

    comfy = SimpleNamespace(
        model_management=SimpleNamespace(cast_to=lambda x, device: x.to(device), in_training=False),
        quant_ops=SimpleNamespace(ck=SimpleNamespace(rms_rope_split_half_=rms_rope)),
        ops=SimpleNamespace(linear_input_act=linear_act),
    )
    container = definitions("comfy/ldm/modules/attention.py", ["AttentionTensorContainer"],
                            {"torch": torch})["AttentionTensorContainer"]
    env = definitions("comfy/ldm/minimax/model.py", [
        "Attention", "MLP", "AdalnProj", "RefinerBlock", "TokenRefiner", "DiTBlock", "MiniMaxH3Model",
        "_mod_row", "_mod_scale_shift", "_mod_gate", "rope_rotation_table",
        "_axis_from_sqrt_area", "_frame_grid", "_video_t_spans", "_video_t_grid", "_ref_t_span",
        "_audio_grid", "_video_grid", "PackedLayout", "patchify_video",
    ], dict(torch=torch, nn=nn, math=math, comfy=comfy, optimized_attention=attention,
            AttentionTensorContainer=container, FRAME_PER_TOKEN=(1, 4, 4, 4, 4), FRAME_RESCALE=5 / 3))
    module = ModuleType("comfy.ldm.minimax.model")
    module.__dict__.update(env)
    sys.modules[module.__name__] = module
    return SimpleNamespace(**env)

from types import MethodType, SimpleNamespace

import pytest
import torch

from core_source import native_h3

import h3forge.nag as nag
from fake_minimax import PackedLayout
from h3forge.nag import NAGConfig, apply_nag, nag_combine, negative_text_kv
from h3forge.state import AttentionPolicy, RuntimeState

CHANNELS = 16
HEADS = 2
HEAD_DIM = 8


class _CountingLinear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return super().forward(x)


def _make_state(**config_overrides):
    torch.manual_seed(0)
    # text rows [0,3), stereo audio rows [3,7), video rows [7,11).
    layout = PackedLayout(3, 2, 4, 2, 2)
    state = RuntimeState(AttentionPolicy())
    state.layout = layout
    state.block_index = 5
    state.current_sigma = 1.0
    config = dict(negative_context=torch.randn(1, 4, CHANNELS), mode="lite", scale=3.0,
                  tau=2.5, alpha=0.15, sigma_end=0.70, first_block=0, last_block=10,
                  video_strength=1.0, audio_strength=0.5, strict=True)
    config.update(config_overrides)
    state.nag = NAGConfig(**config)
    core = native_h3()
    state.diffusion = SimpleNamespace(rope=SimpleNamespace(inv_freq=torch.tensor([0.1])), hidden_size=CHANNELS)
    state.diffusion.preprocess_text_embeds = MethodType(core.MiniMaxH3Model.preprocess_text_embeds, state.diffusion)
    state.diffusion.rope_freqs = MethodType(core.MiniMaxH3Model.rope_freqs, state.diffusion)
    state.blocks = [core.DiTBlock(CHANNELS, HEADS, HEAD_DIM, 32, 4, 1e-5, 1e-5,
                    operations=SimpleNamespace(Linear=_CountingLinear, RMSNorm=torch.nn.RMSNorm)) for _ in range(12)]
    state.block_args = {"t_emb": torch.ones(1, 4), "mod_segments": [(0, layout.seq_len, 1)]}
    return state


def _qkv(layout):
    torch.manual_seed(1)
    q = torch.randn(1, HEADS, layout.seq_len, HEAD_DIM)
    k = torch.randn(1, HEADS, layout.seq_len, HEAD_DIM)
    v = torch.randn(1, HEADS, layout.seq_len, HEAD_DIM)
    return q, k, v


def test_nag_combine_is_identity_at_scale_one():
    z_pos = torch.randn(1, 2, 4, 8)
    z_neg = torch.randn(1, 2, 4, 8)
    assert torch.allclose(nag_combine(z_pos, z_neg, 1.0, 2.5, 0.5), z_pos, atol=1e-6)


def test_nag_combine_is_identity_when_negative_matches_positive():
    z_pos = torch.randn(1, 2, 4, 8)
    assert torch.allclose(nag_combine(z_pos, z_pos.clone(), 5.0, 2.5, 1.0), z_pos, atol=1e-5)


def test_nag_combine_caps_extrapolation_at_tau_in_l1():
    z_pos = torch.full((1, 1, 1, 8), 1.0)
    z_neg = torch.full((1, 1, 1, 8), -1.0)
    tau = 2.0
    # guided = 1*9 - (-1)*8 = 17 per element => ratio 17 >> tau, so the guided
    # feature is renormalized to tau * |pos|_1.
    guided = nag_combine(z_pos, z_neg, 9.0, tau, 1.0)
    assert torch.allclose(guided.abs().sum(dim=-1), z_pos.abs().sum(dim=-1) * tau, atol=1e-4)


def test_nag_combine_alpha_blends_back_toward_positive():
    z_pos = torch.randn(1, 2, 4, 8)
    z_neg = torch.randn(1, 2, 4, 8)
    full = nag_combine(z_pos, z_neg, 3.0, 100.0, 1.0)
    half = nag_combine(z_pos, z_neg, 3.0, 100.0, 0.5)
    assert torch.allclose(half, 0.5 * full + 0.5 * z_pos, atol=1e-5)


def test_nag_combine_preserves_dtype():
    z_pos = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
    z_neg = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
    assert nag_combine(z_pos, z_neg, 3.0, 2.5, 0.5).dtype == torch.bfloat16


@pytest.mark.parametrize("mode", ["lite", "faithful_selective"])
def test_apply_nag_touches_only_target_rows(mode):
    state = _make_state(mode=mode)
    q, k, v = _qkv(state.layout)
    out = torch.zeros(1, HEADS, state.layout.seq_len, HEAD_DIM)
    result = apply_nag(state, q, k, v, out, skip_output_reshape=True)
    assert torch.count_nonzero(result[:, :, :3]) == 0
    assert torch.count_nonzero(result[:, :, 3:7]) > 0
    assert torch.count_nonzero(result[:, :, 7:]) > 0
    assert state.nag_calls == 1


def test_apply_nag_respects_zero_audio_strength():
    state = _make_state(audio_strength=0.0)
    q, k, v = _qkv(state.layout)
    out = torch.zeros(1, HEADS, state.layout.seq_len, HEAD_DIM)
    result = apply_nag(state, q, k, v, out, skip_output_reshape=True)
    assert torch.count_nonzero(result[:, :, 3:7]) == 0
    assert torch.count_nonzero(result[:, :, 7:]) > 0


def test_apply_nag_handles_reshaped_attention_output():
    state = _make_state()
    q, k, v = _qkv(state.layout)
    out = torch.zeros(1, state.layout.seq_len, HEADS * HEAD_DIM)
    result = apply_nag(state, q, k, v, out, skip_output_reshape=False)
    assert torch.count_nonzero(result[:, :3]) == 0
    assert torch.count_nonzero(result[:, 7:]) > 0


def test_apply_nag_gates_on_block_sigma_and_layout():
    state = _make_state()
    q, k, v = _qkv(state.layout)
    out = torch.zeros(1, HEADS, state.layout.seq_len, HEAD_DIM)

    state.block_index = 20
    assert apply_nag(state, q, k, v, out, skip_output_reshape=True) is out
    state.block_index = 5

    state.current_sigma = 0.5
    assert apply_nag(state, q, k, v, out, skip_output_reshape=True) is out
    state.current_sigma = 1.0

    layout = state.layout
    state.layout = None
    assert apply_nag(state, q, k, v, out, skip_output_reshape=True) is out
    state.layout = layout

    state.nag.alpha = 0.0
    assert apply_nag(state, q, k, v, out, skip_output_reshape=True) is out


def test_delta_is_zero_when_sidecar_equals_positive_text(monkeypatch):
    for mode in ("lite", "faithful_selective"):
        state = _make_state(mode=mode)
        q, k, v = _qkv(state.layout)
        monkeypatch.setattr(nag, "negative_text_kv",
                            lambda *args, **kwargs: (k[:, :, 0:3], v[:, :, 0:3]))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        result = apply_nag(state, q, k, v, out, skip_output_reshape=True)
        assert torch.allclose(result, out, atol=1e-5)


def test_sidecar_keys_cache_per_sigma_and_refiner_per_run():
    state = _make_state()
    preprocess_calls = []

    def preprocess(context):
        preprocess_calls.append(context.shape)
        return context * 2.0

    state.diffusion.preprocess_text_embeds = preprocess
    q, k, v = _qkv(state.layout)
    out = torch.zeros(1, HEADS, state.layout.seq_len, HEAD_DIM)
    apply_nag(state, q, k, v, out, skip_output_reshape=True)
    apply_nag(state, q, k, v, out, skip_output_reshape=True)
    assert len(preprocess_calls) == 1
    assert state.blocks[5].attn.qkv_proj.calls == 1

    state.current_sigma = 0.9
    state.block_args["t_emb"] = torch.full((1, 4), 0.5)
    apply_nag(state, q, k, v, out, skip_output_reshape=True)
    assert state.blocks[5].attn.qkv_proj.calls == 2
    assert len(preprocess_calls) == 1

    # A new run clears the sidecar caches.
    state.begin_run()
    assert state.nag_runtime is None


def test_native_attention_shape_mismatch_is_named():
    state = _make_state()
    with pytest.raises(RuntimeError, match="shape differs"):
        negative_text_kv(state, state.nag, 5, heads=HEADS + 1, head_dim=HEAD_DIM,
                         device="cpu", dtype=torch.float32)


@torch.no_grad()
def test_sidecar_matches_native_block_text_keys_with_modulation_and_rope():
    core = native_h3()
    state = _make_state()
    negative = state.nag.negative_context
    state.layout = core.PackedLayout(negative.shape[1], 2, 4, 2, 2)
    block = state.blocks[state.block_index]
    h = torch.randn(state.layout.seq_len, CHANNELS)
    h[:negative.shape[1]] = negative[0]
    rope = core.rope_rotation_table(state.diffusion.rope_freqs(state.layout.position_ids, "cpu"), h.dtype)
    captured = []

    def capture(func, q, k, v, heads, **kwargs):
        captured.append((k[:, :, :negative.shape[1]].clone(), v[:, :, :negative.shape[1]].clone()))
        return q.new_zeros((1, q.shape[2], heads * q.shape[-1]))

    args = state.block_args
    block(h, args["t_emb"], [(0, h.shape[0], 1)], rope,
          transformer_options={"optimized_attention_override": capture})
    actual = negative_text_kv(state, state.nag, state.block_index, heads=HEADS, head_dim=HEAD_DIM,
                              device="cpu", dtype=torch.float32)
    for expected, value in zip(captured[0], actual):
        torch.testing.assert_close(value, expected)


def test_nag_config_validation():
    context = torch.zeros(1, 4, CHANNELS)
    with pytest.raises(ValueError, match="mode"):
        NAGConfig(negative_context=context, mode="wan")
    with pytest.raises(ValueError, match="first_block"):
        NAGConfig(negative_context=context, first_block=30, last_block=8)
    with pytest.raises(ValueError, match="nag_alpha"):
        NAGConfig(negative_context=context, alpha=1.5)
    with pytest.raises(ValueError, match="nag_scale"):
        NAGConfig(negative_context=context, scale=0.5)
    with pytest.raises(ValueError, match="nag_tau"):
        NAGConfig(negative_context=context, tau=0.5)
    with pytest.raises(ValueError, match="negative conditioning"):
        NAGConfig(negative_context=torch.zeros(2, 4, CHANNELS))

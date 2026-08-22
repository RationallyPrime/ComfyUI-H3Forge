import torch

from h3forge.attention import _make_block_mask, _run_flex, _runtime_int, _video_cumulative_time
from h3forge.state import AttentionPolicy, RuntimeState


def test_h3_video_cadence():
    frames = torch.arange(11)
    got = _video_cumulative_time(frames)
    base = torch.tensor([0, 1, 5, 9, 13, 17, 18, 22, 26, 30, 34], dtype=torch.float32)
    assert torch.allclose(got, base * (5.0 / 3.0))


def test_block_mask_is_compiled_and_head_broadcast(monkeypatch):
    class Layout:
        signature = (8, 2, 2, 2, 2)
        segments = [(0, 8, "text"), (8, 12, "audio"), (12, 14, "video")]
        seq_len = 14

    calls = []
    marker = object()

    def fake_create_block_mask(mask_mod, **kwargs):
        calls.append(kwargs)
        return marker

    import torch.nn.attention.flex_attention as flex_module
    monkeypatch.setattr(flex_module, "create_block_mask", fake_create_block_mask)

    state = RuntimeState(AttentionPolicy())
    state.layout = Layout()
    q = torch.zeros(1, 56, Layout.seq_len, 128)
    assert _make_block_mask(state, q, device=q.device) is marker
    assert calls[0]["H"] is None
    assert calls[0]["_compile"] is True

    # A repeated layout reuses the compiled block mask and records the hit.
    assert _make_block_mask(state, q, device=q.device) is marker
    assert len(calls) == 1
    assert (state.mask_hits, state.mask_misses) == (1, 1)


def test_runtime_offsets_are_tensor_data():
    value = _runtime_int(17, device="cpu")
    assert value.ndim == 0
    assert value.dtype == torch.int64
    assert value.item() == 17


def test_flex_attention_uses_fixed_equalized_window_shape(monkeypatch):
    import h3forge.attention as attention
    import torch.nn.attention.flex_attention as flex_module

    compile_calls = []

    def fake_flex(q, k, v, *, block_mask):
        assert block_mask is marker
        return q + k + v

    def fake_compile(func, **kwargs):
        compile_calls.append((func, kwargs))
        return func

    marker = object()
    monkeypatch.setattr(flex_module, "flex_attention", fake_flex)
    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(attention, "_make_block_mask", lambda state, q, device: marker)
    monkeypatch.setattr(attention, "_COMPILED_FLEX", None)

    state = RuntimeState(AttentionPolicy())
    q = torch.ones(1, 2, 3, 4)
    assert torch.equal(_run_flex(state, q, q, q), q * 3)
    assert len(compile_calls) == 1
    assert compile_calls[0][1]["dynamic"] is False

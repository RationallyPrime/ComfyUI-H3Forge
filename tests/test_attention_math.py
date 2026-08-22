import torch

from h3forge.attention import _make_block_mask, _video_cumulative_time
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

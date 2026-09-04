import sys
import types

import pytest
import torch

from fake_minimax import PackedLayout
from h3forge.attention import (
    _make_block_mask,
    _run_flex,
    _runtime_int,
    _scale_video_output,
    _video_cumulative_time,
    feta_gain,
)
from h3forge.state import AttentionPolicy, RuntimeState


def test_h3_video_cadence():
    frames = torch.arange(11)
    got = _video_cumulative_time(frames)
    base = torch.tensor([0, 1, 5, 9, 13, 17, 18, 22, 26, 30, 34], dtype=torch.float32)
    assert torch.allclose(got, base * (5.0 / 3.0))


class _Layout:
    def __init__(self, signature, segments):
        self.signature = signature
        self.segments = segments
        self.seq_len = segments[-1][1]


def _fake_mask_factory(monkeypatch, calls):
    def fake_create_block_mask(mask_mod, **kwargs):
        calls.append({"mask_mod": mask_mod, **kwargs})
        return object()

    import torch.nn.attention.flex_attention as flex_module
    monkeypatch.setattr(flex_module, "create_block_mask", fake_create_block_mask)


def test_block_mask_is_compiled_and_head_broadcast(monkeypatch):
    calls = []
    _fake_mask_factory(monkeypatch, calls)

    state = RuntimeState(AttentionPolicy())
    state.layout = _Layout((8, 2, 2, 2, 2), [(0, 8, "text"), (8, 12, "audio"), (12, 14, "video")])
    q = torch.zeros(1, 56, state.layout.seq_len, 128)
    first = _make_block_mask(state, q, device=q.device)
    assert calls[0]["H"] is None
    assert calls[0]["_compile"] is True

    # A repeated layout reuses the compiled block mask and records the hit.
    assert _make_block_mask(state, q, device=q.device) is first
    assert len(calls) == 1
    assert (state.mask_hits, state.mask_misses) == (1, 1)


def test_block_mask_cache_distinguishes_layouts_with_equal_seq_len(monkeypatch):
    calls = []
    _fake_mask_factory(monkeypatch, calls)

    # Same total sequence length (14), different text/audio/video boundaries.
    layout_a = _Layout((8, 2, 2, 2, 2), [(0, 8, "text"), (8, 12, "audio"), (12, 14, "video")])
    layout_b = _Layout((6, 2, 2, 2, 3), [(0, 6, "text"), (6, 12, "audio"), (12, 14, "video")])
    assert layout_a.seq_len == layout_b.seq_len

    state = RuntimeState(AttentionPolicy())
    q = torch.zeros(1, 56, 14, 128)

    state.layout = layout_a
    mask_a = _make_block_mask(state, q, device=q.device)
    state.layout = layout_b
    mask_b = _make_block_mask(state, q, device=q.device)
    assert mask_a is not mask_b
    assert state.mask_misses == 2

    # Identical segmentation but a different context-window offset is a
    # different mask too.
    layout_a_shifted = _Layout(layout_a.signature, layout_a.segments)
    layout_a_shifted._h3forge_video_offset = 5
    state.layout = layout_a_shifted
    mask_c = _make_block_mask(state, q, device=q.device)
    assert mask_c is not mask_a
    assert state.mask_misses == 3

    # The original segmentation still hits its own cached mask.
    state.layout = layout_a
    assert _make_block_mask(state, q, device=q.device) is mask_a
    assert state.mask_hits == 1


def test_runtime_offsets_are_tensor_data():
    value = _runtime_int(17, device="cpu")
    assert value.ndim == 0
    assert value.dtype == torch.int64
    assert value.item() == 17


def _bridge_mask_mod(monkeypatch, policy):
    calls = []
    _fake_mask_factory(monkeypatch, calls)
    # text 4 rows; stereo audio 2x30 rows; video 6 frames x 4 rows (4x4 latent,
    # 2x2 patches => grid_w=2). Frame times: [0,1,5,9,13,17] * 5/3.
    layout = _Layout((4, 6, 4, 4, 30), [(0, 4, "text"), (4, 64, "audio"), (64, 88, "video")])
    state = RuntimeState(policy)
    state.layout = layout
    q = torch.zeros(1, 2, layout.seq_len, 8)
    _make_block_mask(state, q, device=q.device)
    mask_mod = calls[0]["mask_mod"]

    def allow(q_idx, kv_idx):
        return bool(mask_mod(0, 0, torch.tensor(q_idx), torch.tensor(kv_idx)))

    return allow


def video_row(frame, row):
    return 64 + frame * 4 + row


def audio_row(time):
    return 4 + time


def test_bridge_keys_reopen_stride_aligned_times_for_every_query(monkeypatch):
    """Documents bridge semantics: the bridge clause ORs stride-aligned non-global
    keys back into the whole mask, reopening them past both the temporal band and
    the video radial restriction."""
    policy = AttentionPolicy(temporal_window=2.0, spatial_radius=0.25, bridge_stride=5,
                             first_dense_layers=0, first_dense_fraction=0.0)
    allow = _bridge_mask_mod(monkeypatch, policy)

    # Frame times (40 Hz ticks): f0=0, f2=8.33, f3=15, f4=21.67, f5=28.33.
    # Bridge buckets round to 0, 8, 15, 22, 28; stride 5 selects times 0 and 15.
    # Distant video frame at a bridge time is reachable despite the temporal band.
    assert allow(video_row(4, 0), video_row(0, 0))
    # ... including across the spatial radial restriction (site (1,1) vs (0,0)).
    assert allow(video_row(5, 3), video_row(3, 0))
    # A distant non-bridge video frame stays blocked, same spatial site.
    assert not allow(video_row(5, 0), video_row(2, 0))
    # Audio keys at stride-aligned times are bridge keys too; others are not.
    assert allow(video_row(5, 0), audio_row(10))
    assert not allow(video_row(5, 0), audio_row(7))
    # Global prefix rows stay reachable in both directions regardless of time.
    assert allow(video_row(5, 0), 2)
    assert allow(2, video_row(2, 0))
    # Same-frame video attention stays spatially dense.
    assert allow(video_row(5, 0), video_row(5, 3))
    # Local audio band still works without a bridge.
    assert allow(audio_row(9), audio_row(8))


def test_bridge_stride_zero_disables_bridges(monkeypatch):
    policy = AttentionPolicy(temporal_window=2.0, spatial_radius=0.25, bridge_stride=0,
                             first_dense_layers=0, first_dense_fraction=0.0)
    allow = _bridge_mask_mod(monkeypatch, policy)
    assert not allow(video_row(4, 0), video_row(0, 0))
    assert not allow(video_row(5, 3), video_row(3, 0))


def test_flex_attention_compiles_one_runner_per_shape(monkeypatch):
    import h3forge.attention as attention
    import torch.nn.attention.flex_attention as flex_module

    compile_calls = []
    marker = object()

    def fake_flex(q, k, v, *, block_mask):
        assert block_mask is marker
        return q + k + v

    def fake_compile(func, **kwargs):
        compile_calls.append((func, kwargs))
        return func

    monkeypatch.setattr(flex_module, "flex_attention", fake_flex)
    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(attention, "_make_block_mask", lambda state, q, device: marker)
    monkeypatch.setattr(attention, "_COMPILED_FLEX_CACHE", {})

    state = RuntimeState(AttentionPolicy())
    q = torch.ones(1, 2, 3, 4)
    assert torch.equal(_run_flex(state, q, q, q), q * 3)
    assert len(compile_calls) == 1
    assert compile_calls[0][1]["dynamic"] is False

    # The same shape reuses its runner; a new sequence length compiles a new one.
    _run_flex(state, q, q, q)
    assert len(compile_calls) == 1
    q2 = torch.ones(1, 2, 5, 4)
    _run_flex(state, q2, q2, q2)
    assert len(compile_calls) == 2


def _feta_state():
    policy = AttentionPolicy(feta_enabled=True, feta_first_layer=0, feta_last_layer=49)
    state = RuntimeState(policy)
    state.layout = PackedLayout(2, 2, 4, 2, 2)
    state.block_index = 5
    return state


def test_feta_gain_stays_a_device_tensor():
    state = _feta_state()
    torch.manual_seed(0)
    q = torch.randn(1, 2, state.layout.seq_len, 8)
    k = torch.randn(1, 2, state.layout.seq_len, 8)
    gain = feta_gain(state, q, k)
    assert isinstance(gain, torch.Tensor)
    assert gain.ndim == 0
    assert 1.0 <= float(gain) <= state.policy.feta_max_gain


def test_feta_gain_returns_none_outside_its_gates():
    state = _feta_state()
    q = torch.randn(1, 2, state.layout.seq_len, 8)
    state.block_index = None
    assert feta_gain(state, q, q) is None
    state.block_index = 5
    state.policy.feta_first_layer = 10
    state.policy.feta_last_layer = 20
    assert feta_gain(state, q, q) is None


def test_feta_scales_only_target_video_rows():
    state = _feta_state()
    gain = torch.tensor(1.1)
    video_start, video_stop = state.layout.segments[-1][0], state.layout.segments[-1][1]

    out = torch.ones(1, 2, state.layout.seq_len, 8)
    scaled = _scale_video_output(state, out, gain, skip_output_reshape=True)
    assert torch.allclose(scaled[:, :, :video_start], torch.ones(1, 2, video_start, 8))
    assert torch.allclose(scaled[:, :, video_start:video_stop],
                          torch.full((1, 2, video_stop - video_start, 8), 1.1))

    flat = torch.ones(1, state.layout.seq_len, 16)
    scaled_flat = _scale_video_output(state, flat, gain, skip_output_reshape=False)
    assert torch.allclose(scaled_flat[:, :video_start], torch.ones(1, video_start, 16))
    assert torch.allclose(scaled_flat[:, video_start:video_stop],
                          torch.full((1, video_stop - video_start, 16), 1.1))


def test_reversed_feta_layer_range_is_rejected():
    with pytest.raises(ValueError, match="feta_first_layer"):
        AttentionPolicy(feta_first_layer=10, feta_last_layer=3)


def test_dynamo_headroom_exceeds_the_flex_runner_cache(monkeypatch):
    """Dynamo must be able to hold every shape the flex cache keeps.

    The budget is shared across all torch.compile wrappers of one function, so
    a limit equal to the cache size makes the next shape silently fall back to
    eager flex_attention, which is O(S^2) and out-of-memories at H3 lengths.
    """
    from h3forge import attention as attention_module

    class FakeConfig:
        def __init__(self, limit):
            self.recompile_limit = limit

    config = FakeConfig(8)
    fake = types.SimpleNamespace(config=config)
    monkeypatch.setattr(attention_module, "_DYNAMO_HEADROOM_APPLIED", False)
    # "import torch._dynamo as x" reads the attribute off the torch module when
    # it is already imported, so patching sys.modules alone would not be seen.
    monkeypatch.setattr(torch, "_dynamo", fake, raising=False)
    monkeypatch.setitem(sys.modules, "torch._dynamo", fake)
    attention_module._ensure_dynamo_headroom()
    assert config.recompile_limit > attention_module._COMPILED_FLEX_CACHE_LIMIT

    # A limit already larger is never reduced.
    config.recompile_limit = 4096
    monkeypatch.setattr(attention_module, "_DYNAMO_HEADROOM_APPLIED", False)
    attention_module._ensure_dynamo_headroom()
    assert config.recompile_limit == 4096

    # And the whole thing is done once per process, not per attention call.
    config.recompile_limit = 8
    attention_module._ensure_dynamo_headroom()
    assert config.recompile_limit == 8

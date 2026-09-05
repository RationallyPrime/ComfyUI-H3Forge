import inspect

import pytest
import torch

from fake_minimax import PackedLayout
from h3forge.attention import (
    _FlexRunner,
    _make_block_mask,
    _release_runner,
    _run_flex,
    _runtime_int,
    _scale_video_output,
    _video_cumulative_time,
    _with_private_code,
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
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)


def test_block_mask_is_compiled_and_head_broadcast(monkeypatch):
    calls = []
    _fake_mask_factory(monkeypatch, calls)

    state = RuntimeState(AttentionPolicy())
    state.layout = _Layout((8, 2, 2, 2, 2), [(0, 8, "text"), (8, 12, "audio"), (12, 14, "video")])
    q = torch.zeros(1, 56, state.layout.seq_len, 128)
    first = _make_block_mask(state, q, device=q.device)
    assert calls[0]["H"] is None
    assert "_compile" not in calls[0]  # compiled explicitly through a private code object

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


@pytest.mark.parametrize("bridge_stride", [0, 5])
def test_audio_self_attention_keeps_the_whole_stereo_utterance(monkeypatch, bridge_stride):
    policy = AttentionPolicy(temporal_window=2.0, spatial_radius=0.25, bridge_stride=bridge_stride,
                             first_dense_layers=0, first_dense_fraction=0.0)
    allow = _bridge_mask_mod(monkeypatch, policy)
    # Far-apart, non-bridge times in both channel-major stereo streams must
    # remain mutually visible, including the last audio token in the sequence.
    for query in (audio_row(1), audio_row(29), audio_row(1) + 30, audio_row(29) + 30):
        for key in (audio_row(1), audio_row(29), audio_row(1) + 30, audio_row(29) + 30):
            assert allow(query, key)
    # Exempting audio pairs must leave cross-modal and video sparsity intact.
    assert not allow(video_row(5, 0), audio_row(7))
    assert not allow(audio_row(7), video_row(5, 0))
    assert not allow(video_row(5, 0), video_row(2, 0))


@pytest.mark.parametrize("negative_length", [2, 8])
def test_swapped_negative_text_preserves_sparse_target_visibility(monkeypatch, negative_length):
    calls = []
    _fake_mask_factory(monkeypatch, calls)
    state = RuntimeState(AttentionPolicy(temporal_window=2, spatial_radius=.25, bridge_stride=0))
    state.layout = _Layout((4, 6, 4, 4, 30), [(0, 4, "text"), (4, 64, "audio"), (64, 88, "video")])
    q = torch.zeros(1, 2, 88, 8)
    _make_block_mask(state, q, device=q.device)
    _make_block_mask(state, q, device=q.device, negative_text_len=negative_length)
    positive, negative = calls
    queries = torch.arange(88)[:, None]
    base = positive["mask_mod"](0, 0, queries, torch.arange(88)[None, :])
    swapped = negative["mask_mod"](0, 0, queries, torch.arange(88 - 4 + negative_length)[None, :])
    assert torch.all(swapped[:, :negative_length])
    assert torch.equal(swapped[:, negative_length:], base[:, 4:])
    assert not torch.all(swapped[:, negative_length:])  # it did not become dense


def _fake_runner_factory(monkeypatch, compile_calls, marker):
    import h3forge.attention as attention
    import torch.nn.attention.flex_attention as flex_module

    def fake_flex(q, k, v, *, block_mask):
        assert block_mask is marker
        return q + k + v

    def fake_compile(func, **kwargs):
        compile_calls.append((func, kwargs))
        return func

    monkeypatch.setattr(flex_module, "flex_attention", fake_flex)
    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(attention, "_make_block_mask", lambda state, q, device, **kwargs: marker)
    monkeypatch.setattr(attention, "_COMPILED_FLEX_CACHE", {})
    monkeypatch.setattr(attention, "_RELEASED_KERNELS", [])
    return fake_flex


def _runner_state():
    state = RuntimeState(AttentionPolicy())
    state.layout = _Layout((8, 2, 2, 2, 2), [(0, 8, "text"), (8, 12, "audio"), (12, 14, "video")])
    return state


def test_flex_attention_compiles_one_runner_per_shape_and_mask(monkeypatch):
    compile_calls = []
    fake_flex = _fake_runner_factory(monkeypatch, compile_calls, object())

    state = _runner_state()
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
    # Each runner wraps its own copy of flex_attention: a private code object
    # per runner is what keeps Dynamo's recompile budget per runner.
    first, second = (call[0] for call in compile_calls)
    assert first.__code__ is not fake_flex.__code__
    assert first.__code__ is not second.__code__

    # mask_mod closes over policy and segment values, and Dynamo guards a
    # closure on those values: a changed window at an unchanged shape would be
    # a second specialization of the same runner, spending its budget. It gets
    # its own runner instead.
    state.policy.temporal_window += 1.0
    _run_flex(state, q, q, q)
    assert len(compile_calls) == 3
    state.layout.segments = [(0, 6, "text"), (6, 12, "audio"), (12, 14, "video")]
    _run_flex(state, q, q, q)
    assert len(compile_calls) == 4
    # Stagger offsets are tensor captures, never guard values: not a new runner.
    state.layout._h3forge_video_offset = 3
    _run_flex(state, q, q, q)
    assert len(compile_calls) == 4


def test_evicted_runner_resets_dynamo_state_and_recycles_its_kernel(monkeypatch):
    import torch._dynamo as dynamo

    import h3forge.attention as attention

    compile_calls = []
    _fake_runner_factory(monkeypatch, compile_calls, object())
    monkeypatch.setattr(attention, "_COMPILED_FLEX_CACHE_LIMIT", 2)
    resets = []
    monkeypatch.setattr(dynamo, "reset_code", resets.append)

    state = _runner_state()
    shapes = [torch.ones(1, 2, s, 4) for s in (3, 5, 7, 9)]
    for q in shapes[:2]:
        _run_flex(state, q, q, q)
    assert resets == []

    # The third shape evicts the first runner: its private code object has its
    # Dynamo entries dropped and waits in the pool.
    _run_flex(state, shapes[2], shapes[2], shapes[2])
    evicted = compile_calls[0][0]
    assert resets == [evicted.__code__]
    assert attention._RELEASED_KERNELS == [evicted]
    assert len(attention._COMPILED_FLEX_CACHE) == 2

    # The fourth shape reuses that code object instead of minting another, and
    # the runner it evicts refills the pool: the module never holds more than
    # limit + 1 code objects.
    _run_flex(state, shapes[3], shapes[3], shapes[3])
    assert compile_calls[3][0] is evicted
    assert attention._RELEASED_KERNELS == [compile_calls[1][0]]
    assert resets == [evicted.__code__, compile_calls[1][0].__code__]
    # code.replace() copies compare equal, so count identities, not values.
    assert len({id(call[0].__code__) for call in compile_calls}) == 3


def test_active_prompt_shapes_survive_steps_then_release_at_run_end(monkeypatch):
    import h3forge.attention as attention

    calls = []
    _fake_runner_factory(monkeypatch, calls, object())
    state = _runner_state()
    state.begin_run()
    state.layout = _runner_state().layout
    shapes = [torch.ones(1, 2, length, 4) for length in range(3, 15)]
    for _ in range(2):
        for q in shapes:
            _run_flex(state, q, q, q)
    assert len(calls) == len(shapes)
    assert len(attention._COMPILED_FLEX_CACHE) == len(shapes)
    state.in_run = False
    attention.prune_run_caches(state)
    assert len(attention._COMPILED_FLEX_CACHE) == attention._COMPILED_FLEX_CACHE_LIMIT
    assert len(attention._RELEASED_KERNELS) == 1


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


def test_private_code_copy_is_a_faithful_distinct_function():
    from torch.nn.attention.flex_attention import flex_attention

    first = _with_private_code(flex_attention)
    second = _with_private_code(flex_attention)
    assert first.__code__ is not flex_attention.__code__
    assert first.__code__ is not second.__code__
    assert first.__code__.co_code == flex_attention.__code__.co_code
    assert inspect.signature(first) == inspect.signature(flex_attention)
    assert first.__name__ == flex_attention.__name__
    assert first.__module__ == flex_attention.__module__
    assert first.__globals__ is flex_attention.__globals__


def test_each_runner_owns_its_dynamo_recompile_budget():
    """Shapes beyond Dynamo's per-code-object budget must not fall back to eager.

    Dynamo counts recompiles on the wrapped function's code object, so N
    runners of one function share one budget and a session that meets more
    shapes than the limit silently degrades to eager flex_attention (O(S^2),
    OOM at H3 lengths). A private code object per runner has no such ceiling.
    """
    import torch._dynamo as dynamo

    def kernel(x):
        return x * 2 + 1

    inputs = [torch.ones(n) for n in range(3, 8)]
    dynamo.reset()
    try:
        with dynamo.config.patch(recompile_limit=1, fail_on_recompile_limit_hit=True):
            # One shared code object: the second distinct shape exhausts the budget.
            with pytest.raises(Exception, match="recompile_limit"):
                for x in inputs:
                    torch.compile(kernel, dynamic=False, backend="eager")(x)
            # A private code object per runner: every shape compiles.
            runners = [torch.compile(_with_private_code(kernel), dynamic=False, backend="eager")
                       for _ in inputs]
            for runner, x in zip(runners, inputs):
                assert torch.equal(runner(x), x * 3)
    finally:
        dynamo.reset()


def test_released_runner_refunds_its_dynamo_budget(monkeypatch):
    """Eviction must give Dynamo state back, not just drop our reference.

    Dynamo pins every traced code object, so an evicted runner's compiled
    graphs would otherwise outlive the LRU. Releasing resets the code object:
    its installed ``__compiled_fn_*`` globals disappear, and its budget is
    refunded exactly (a fresh compile succeeds, the next shape trips the limit
    again), so the pooled kernel can back a new runner.
    """
    import torch._dynamo as dynamo

    import h3forge.attention as attention

    monkeypatch.setattr(attention, "_RELEASED_KERNELS", [])

    def kernel(x):
        return x * 2 + 1

    def installed(fn):
        return [name for name in fn.__globals__ if name.startswith("__compiled_fn")]

    dynamo.reset()
    try:
        with dynamo.config.patch(recompile_limit=1, fail_on_recompile_limit_hit=True):
            private = _with_private_code(kernel)
            runner = _FlexRunner(private, torch.compile(private, dynamic=False, backend="eager"))
            assert torch.equal(runner.run(torch.ones(3)), torch.full((3,), 3.0))
            assert len(installed(private)) == 1
            with pytest.raises(Exception, match="recompile_limit"):
                runner.run(torch.ones(4))

            _release_runner(runner)
            assert attention._RELEASED_KERNELS == [private]
            assert installed(private) == []

            reused = torch.compile(private, dynamic=False, backend="eager")
            assert torch.equal(reused(torch.ones(4)), torch.full((4,), 3.0))
            with pytest.raises(Exception, match="recompile_limit"):
                reused(torch.ones(5))
    finally:
        dynamo.reset()

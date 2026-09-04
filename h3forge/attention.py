from __future__ import annotations

import sys
import types
from typing import Callable

import torch

from .layout import target_segments, video_shape_from_layout
from .nag import apply_nag

LOG = "[H3Forge]"
_COMPILED_FLEX_CACHE: dict = {}
_COMPILED_FLEX_CACHE_LIMIT = 8
# Private flex_attention copies whose runners were evicted, Dynamo state already
# reset. A cache miss takes one of these before minting another, so this module
# never holds more than _COMPILED_FLEX_CACHE_LIMIT + 1 code objects.
_RELEASED_KERNELS: list = []


class _FlexRunner:
    """One compiled flex_attention runner and everything the process holds for it."""

    __slots__ = ("kernel", "run", "artifacts")

    def __init__(self, kernel: types.FunctionType, run: Callable):
        self.kernel = kernel  # private copy of flex_attention; owns one Dynamo budget
        self.run = run  # torch.compile(kernel)
        self.artifacts: list[types.ModuleType] = []  # Inductor modules loaded on this runner's behalf


def _with_private_code(fn):
    """Return a copy of ``fn`` whose code object belongs to one runner alone.

    Dynamo keys its recompile budget (``torch._dynamo.config.recompile_limit``,
    default 8) on the *code object* of the function it wraps, not on the
    ``torch.compile`` wrapper around it. Every ``torch.compile(flex_attention)``
    therefore spends a slot of one process-wide budget hanging off
    ``flex_attention.__code__``, and evicting our own runner never refunds one.
    Once a session has met more distinct attention shapes than that budget --
    text tokens are part of the shape, so a prompt edit is a new shape -- every
    further shape falls back to eager ``flex_attention``, which materializes
    the full S x S score matrix and out-of-memories at H3 sequence lengths.
    Raising the budget only moves the session length at which that happens.

    ``code.replace()`` mints a distinct code object with identical bytecode, so
    a runner compiled from this copy owns a full budget of its own that no other
    runner can spend. The number of shapes a process may meet is then bounded
    by nothing but the runner LRU below.
    """
    copy = types.FunctionType(fn.__code__.replace(), fn.__globals__, fn.__name__,
                              fn.__defaults__, fn.__closure__)
    copy.__kwdefaults__ = fn.__kwdefaults__
    copy.__annotations__ = fn.__annotations__
    copy.__qualname__ = fn.__qualname__
    copy.__module__ = fn.__module__
    copy.__doc__ = fn.__doc__
    copy.__dict__.update(fn.__dict__)
    return copy


def _acquire_kernel(flex_attention):
    if _RELEASED_KERNELS:
        return _RELEASED_KERNELS.pop()
    return _with_private_code(flex_attention)


def _call_runner(runner, *args, **kwargs):
    """Run the runner and attribute every Inductor module it loads to it.

    Dynamo compiles on the first call (and on any later recompile), and
    Inductor registers each module it loads for the graph -- the wrapper and,
    on CUDA, every Triton kernel -- in process-wide registries that nothing
    ever shrinks. The attention path is single-threaded, so whatever joins
    those registries during this call was loaded for this runner. A failed
    compile is attributed too, so eviction still releases what it loaded.
    """
    from torch._inductor.codecache import PyCodeCache

    before = len(PyCodeCache.modules)
    try:
        return runner.run(*args, **kwargs)
    finally:
        runner.artifacts.extend(PyCodeCache.modules[before:])


def _forget_inductor_modules(modules):
    """Drop evicted Inductor modules from the registries that pin them.

    ``PyCodeCache.modules`` / ``modules_no_attr`` / ``linemaps`` and
    ``sys.modules`` each hold a strong reference to every module Inductor has
    loaded, for the life of the process. Removing the index entries is all it
    takes: a live runner still reaches its own module through its compiled
    graph, so only a module nothing runs anymore becomes collectable, and a
    later compile of identical source simply reloads it from Inductor's disk
    cache. Membership is checked first because these are registries, not
    ledgers: a module already gone is the state this function exists to reach.
    """
    from torch._inductor.codecache import PyCodeCache

    for mod in modules:
        path, name = mod.__file__, mod.__name__
        PyCodeCache.modules[:] = [m for m in PyCodeCache.modules if m is not mod]
        if PyCodeCache.modules_no_attr.get(path) is mod:
            del PyCodeCache.modules_no_attr[path]
        if not any(m.__file__ == path for m in PyCodeCache.modules):
            PyCodeCache.linemaps.pop(path, None)
        if sys.modules.get(name) is mod:
            del sys.modules[name]


def _release_runner(runner):
    """Evict a runner: drop its Dynamo and Inductor state, pool its kernel.

    Deleting the cache entry alone frees nothing. Dynamo hangs a runner's cache
    entries -- guard managers, traced graphs, the compiled graph -- off the
    private code object and keeps that code object alive in an lru_cache of
    cleaned bytecode; Inductor keeps every module it loaded for the graph in
    process-wide registries. ``reset_code`` drops the Dynamo entries (their
    installed ``__compiled_fn_*`` globals go with them) and refunds the
    budget; forgetting the runner's modules lets the compiled graph, its
    wrapper and its kernel objects be collected. What no eviction returns is
    the device-side binary Triton loaded for each kernel: Triton never unloads
    a module, so those stay resident per distinct shape at kernel-code size,
    as for any ``torch.compile`` user.
    """
    from torch._dynamo import reset_code

    reset_code(runner.kernel.__code__)
    _forget_inductor_modules(runner.artifacts)
    runner.artifacts = []
    _RELEASED_KERNELS.append(runner.kernel)


def _unwrap(x):
    # ComfyUI's wrap_attn consumes AttentionTensorContainer inputs with .take()
    # before invoking overrides that do not provide a container_function. The
    # override therefore receives ordinary tensors; this no-op documents that seam.
    return x


def _dense(func, q, k, v, heads, args, kwargs):
    return func(q, k, v, heads, *args, **kwargs)


def _can_sparse(state, q, kwargs):
    p = state.policy
    if p.mode != "flex_sliding":
        return "mode=dense"
    if state.layout is None:
        return "no-layout"
    if q.ndim != 4 or q.shape[0] != 1:
        return "unexpected-q-shape"
    if q.shape[2] != state.layout.seq_len:
        return "layout-length-mismatch"
    if kwargs.get("mask") is not None:
        return "preexisting-mask"
    if not kwargs.get("skip_reshape", False):
        return "not-bhsd"
    if state.block_index is None:
        return "not-dit-block"
    if state.block_index < p.first_dense_layers:
        return "first-dense-layers"
    if state.step_index is not None and state.total_steps:
        frac = state.step_index / max(state.total_steps, 1)
        if frac < p.first_dense_fraction:
            return "first-dense-steps"
    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask  # noqa:F401
    except Exception:
        return "flex-unavailable"
    return None


def _video_cumulative_time(frame_idx):
    """H3 target-video time before global latent frame `frame_idx` (1/4/4/4/4 cadence)."""
    cycle = torch.div(frame_idx, 5, rounding_mode="floor")
    rem = torch.remainder(frame_idx, 5)
    prefix = torch.where(
        rem == 0, torch.zeros_like(rem),
        torch.where(rem == 1, torch.ones_like(rem),
                    torch.where(rem == 2, torch.full_like(rem, 5),
                                torch.where(rem == 3, torch.full_like(rem, 9),
                                            torch.full_like(rem, 13)))))
    return (cycle * 17 + prefix).to(torch.float32) * (5.0 / 3.0)


def _runtime_int(value: int, *, device):
    """Capture a changing mask scalar as data, not a Dynamo guard value."""
    return torch.tensor(int(value), dtype=torch.int64, device=device)


def _mask_specialization(state):
    """Every Python value ``mask_mod`` closes over, as one hashable key.

    Dynamo guards a traced closure on the values it captured, so two masks
    built from different policy or segment values are two specializations of
    whichever runner consumes them, even at one tensor shape. Two packed
    layouts can share one total sequence length while placing text,
    references, audio, and video at different boundaries, and the signature
    alone does not describe reference/keyframe prefix structure, so the full
    segment table is part of the key. The stagger offsets are not: they are
    captured as tensors and never specialize anything.
    """
    layout = state.layout
    p = state.policy
    return (
        tuple(layout.signature),
        tuple(tuple(segment) for segment in layout.segments),
        p.temporal_window,
        p.spatial_radius,
        p.bridge_stride,
    )


def _make_block_mask(state, q, *, device):
    from torch.nn.attention.flex_attention import create_block_mask

    layout = state.layout
    p = state.policy
    seg = target_segments(layout)
    _, latent_t, latent_h, latent_w, audio_t = layout.signature
    frame_rows = (seg.video_stop - seg.video_start) // int(latent_t)
    grid_w = max(int(latent_w) // 2, 1)
    video_offset = int(getattr(layout, "_h3forge_video_offset", 0))
    audio_offset = int(getattr(layout, "_h3forge_audio_offset", 0))
    # Each concrete BlockMask is one specialization plus one pair of offsets.
    key = (str(device), _mask_specialization(state), video_offset, audio_offset)
    cached = state.mask_cache.get(key)
    if cached is not None:
        state.mask_cache.move_to_end(key)
        state.mask_hits += 1
        return cached
    state.mask_misses += 1

    temporal_window = float(p.temporal_window)
    spatial_radius = float(p.spatial_radius)
    bridge_stride = int(p.bridge_stride)
    audio_start, audio_stop = seg.audio_start, seg.audio_stop
    video_start, video_stop = seg.video_start, seg.video_stop
    audio_t = int(audio_t)
    frame_rows = int(frame_rows)
    # Context staggering changes absolute offsets while preserving the compiled
    # window shape. Python integer closures make Dynamo specialize flex_attention
    # once per offset and eventually fall back to its eager S^2 implementation.
    # Tensor captures are runtime data, so offset values can change without a
    # kernel recompile while each concrete BlockMask remains separately cached.
    video_offset_runtime = _runtime_int(video_offset, device=device)
    audio_offset_runtime = _runtime_int(audio_offset, device=device)

    def token_time(idx):
        in_audio = (idx >= audio_start) & (idx < audio_stop)
        in_video = (idx >= video_start) & (idx < video_stop)
        audio_rel = torch.remainder(idx - audio_start, audio_t) + audio_offset_runtime
        video_frame = torch.div(idx - video_start, frame_rows, rounding_mode="floor") + video_offset_runtime
        video_t = _video_cumulative_time(video_frame)
        return torch.where(in_audio, audio_rel, torch.where(in_video, video_t, torch.zeros_like(idx)))

    def mask_mod(b, h, q_idx, kv_idx):
        gq = q_idx < audio_start
        gk = kv_idx < audio_start
        vq = (q_idx >= video_start) & (q_idx < video_stop)
        vk = (kv_idx >= video_start) & (kv_idx < video_stop)

        tq = token_time(q_idx)
        tk = token_time(kv_idx)
        dt = torch.abs(tq - tk)
        allow = gq | gk | (dt <= temporal_window)

        # Video↔video links between different temporal positions use a local
        # spatial radius in patch-grid coordinates; same-frame attention is dense.
        qrow = torch.remainder(q_idx - video_start, frame_rows)
        krow = torch.remainder(kv_idx - video_start, frame_rows)
        qy, qx = torch.div(qrow, grid_w, rounding_mode="floor"), torch.remainder(qrow, grid_w)
        ky, kx = torch.div(krow, grid_w, rounding_mode="floor"), torch.remainder(krow, grid_w)
        local_spatial = (torch.abs(qy - ky) <= spatial_radius) & (torch.abs(qx - kx) <= spatial_radius)
        both_video = vq & vk
        allow = allow & (~both_video | (dt == 0) | local_spatial)

        if bridge_stride > 0:
            bridge_bucket = torch.round(tk).to(torch.int64)
            bridge = (~gk) & (torch.remainder(bridge_bucket, bridge_stride) == 0)
            allow = allow | bridge
        return allow

    # mask_mod is head-independent. H=None builds one broadcastable block mask
    # rather than duplicating eager work across all 56 H3 heads. Compiling mask
    # construction avoids materializing the eager [B,H,Q,K] boolean grid.
    block_mask = create_block_mask(mask_mod, B=1, H=None,
                                   Q_LEN=layout.seq_len, KV_LEN=layout.seq_len,
                                   device=device, _compile=True)
    state.mask_cache[key] = block_mask
    state.mask_cache.move_to_end(key)
    while len(state.mask_cache) > state.mask_cache_limit:
        state.mask_cache.popitem(last=False)
        state.mask_evictions += 1
    return block_mask

def _run_flex(state, q, k, v):
    from torch.nn.attention.flex_attention import flex_attention

    # H3 arrives BHSD; flex_attention consumes BHSD directly. Fixed-shape
    # compilation avoids PyTorch Inductor's symbolic BlockMask lowering path,
    # while tensor-captured offsets can still vary at runtime without causing
    # value-specialized recompiles. One runner is kept per concrete shape and
    # mask specialization, so a duration/resolution/window change triggers an
    # observable compile instead of relying on hidden Dynamo guard-cache
    # semantics, and each runner wraps a private copy of flex_attention so
    # Dynamo's per-code-object recompile budget is per runner too (see
    # _with_private_code). A runner therefore never meets a second mask
    # specialization and never approaches that budget.
    shape = (str(q.device), q.dtype, int(q.shape[0]), int(q.shape[1]), int(q.shape[2]), int(q.shape[3]))
    spec = _mask_specialization(state)
    key = (shape, spec)
    runner = _COMPILED_FLEX_CACHE.pop(key, None)
    if runner is None:
        print(f"{LOG} compiling flex_attention runner for (device, dtype, B, H, S, D)={shape} "
              f"with mask (temporal_window, spatial_radius, bridge_stride)={spec[-3:]}", flush=True)
        kernel = _acquire_kernel(flex_attention)
        runner = _FlexRunner(kernel, torch.compile(kernel, dynamic=False))
    # Re-insert as most recent and bound the cache like mask_cache: a session
    # sweeping durations/resolutions must not accumulate runners forever, and
    # an evicted runner gives its Dynamo and Inductor state back (see
    # _release_runner).
    _COMPILED_FLEX_CACHE[key] = runner
    while len(_COMPILED_FLEX_CACHE) > _COMPILED_FLEX_CACHE_LIMIT:
        _release_runner(_COMPILED_FLEX_CACHE.pop(next(iter(_COMPILED_FLEX_CACHE))))
    mask = _make_block_mask(state, q, device=q.device)
    return _call_runner(runner, q, k, v, block_mask=mask)


@torch.no_grad()
def feta_gain(state, q, k) -> torch.Tensor | None:
    """H3-native FETA-style gain from temporal off-diagonal video attention.

    Only target-video rows participate. The diagnostic samples spatial sites and
    heads, preserving same-site temporal correspondence and avoiding an S² map.
    Returns a zero-dim device tensor (never converted to a Python float on the
    hot path — that would force a GPU→CPU synchronization per FETA-enabled
    block), or None when FETA does not apply to this call.
    """
    p = state.policy
    if not p.feta_enabled or state.layout is None or state.block_index is None:
        return None
    if not (p.feta_first_layer <= state.block_index <= p.feta_last_layer):
        return None

    seg = target_segments(state.layout)
    latent_t, frame_rows = video_shape_from_layout(state.layout)
    if latent_t < 2:
        return None

    qv = q[:, :, seg.video_start:seg.video_stop, :].reshape(q.shape[0], q.shape[1], latent_t, frame_rows, q.shape[-1])
    kv = k[:, :, seg.video_start:seg.video_stop, :].reshape(k.shape[0], k.shape[1], latent_t, frame_rows, k.shape[-1])

    h_take = min(qv.shape[1], max(1, p.feta_max_heads))
    r_take = min(frame_rows, max(1, p.feta_max_spatial_samples))
    h_idx = torch.linspace(0, qv.shape[1] - 1, h_take, device=q.device).round().long()
    r_idx = torch.linspace(0, frame_rows - 1, r_take, device=q.device).round().long()
    qv = qv[:, h_idx][:, :, :, r_idx].float()
    kv = kv[:, h_idx][:, :, :, r_idx].float()

    # [B,H,T,R,D] x same -> [B,H,R,T,T], same spatial site only.
    scores = torch.einsum("bhtrd,bhurd->bhrtu", qv, kv) * (q.shape[-1] ** -0.5)
    probs = scores.softmax(dim=-1)
    # Match Enhance-A-Video/FETA's scalar form: mean off-diagonal attention
    # multiplied by (T + weight), clamped to at least one. We sample heads and
    # spatial sites, but do not alter the estimator itself.
    diag_sum = probs.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    mean_offdiag = (latent_t - diag_sum).mean() / max(latent_t * (latent_t - 1), 1)
    gain = mean_offdiag * (latent_t + p.feta_strength)
    return gain.clamp(min=1.0, max=float(p.feta_max_gain))


def _scale_video_output(state, out, gain, *, skip_output_reshape, owned=False):
    if state.layout is None:
        return out
    seg = target_segments(state.layout)
    gain = gain.to(out.dtype)
    if not owned:
        out = out.clone()
    if skip_output_reshape:
        # BHSD output
        out[:, :, seg.video_start:seg.video_stop, :].mul_(gain)
    else:
        # B,S,H*D output expected by H3's Attention module.
        out[:, seg.video_start:seg.video_stop, :].mul_(gain)
    state.feta_calls += 1
    return out


def make_attention_override(state):
    def override(func, q, k, v, heads, *args, **kwargs):
        q, k, v = _unwrap(q), _unwrap(k), _unwrap(v)
        options = kwargs.get("transformer_options") or {}
        active_layout = options.get("h3forge_active_layout")
        if active_layout is not None:
            state.layout = active_layout
        reason = _can_sparse(state, q, kwargs)
        skip_output_reshape = bool(kwargs.get("skip_output_reshape", False))

        if reason is None:
            try:
                raw = _run_flex(state, q, k, v)
                state.sparse_calls += 1
                if skip_output_reshape:
                    out = raw
                else:
                    out = raw.transpose(1, 2).reshape(raw.shape[0], raw.shape[2], heads * raw.shape[3])
            except Exception as exc:
                if state.policy.strict:
                    raise RuntimeError(f"{LOG} sparse attention failed: {type(exc).__name__}: {exc}") from exc
                state.note_decline(f"flex-error:{type(exc).__name__}")
                state.dense_calls += 1
                out = _dense(func, q, k, v, heads, args, kwargs)
        else:
            state.note_decline(reason)
            state.dense_calls += 1
            out = _dense(func, q, k, v, heads, args, kwargs)

        nag_input = out
        if state.nag is not None:
            try:
                out = apply_nag(state, q, k, v, out, skip_output_reshape=skip_output_reshape,
                                transformer_options=options, attn_mask=kwargs.get("mask"))
            except Exception as exc:
                if state.nag.strict or state.policy.strict:
                    raise RuntimeError(f"{LOG} NAG failed: {type(exc).__name__}: {exc}") from exc
                state.note_decline(f"nag-error:{type(exc).__name__}")

        if state.policy.feta_enabled:
            try:
                gain = feta_gain(state, q, k)
                if gain is not None:
                    # A NAG-cloned output is exclusively ours: scale it in place
                    # instead of cloning the packed tensor a second time.
                    out = _scale_video_output(state, out, gain, skip_output_reshape=skip_output_reshape,
                                              owned=out is not nag_input)
            except Exception as exc:
                if state.policy.strict:
                    raise RuntimeError(f"{LOG} FETA failed: {type(exc).__name__}: {exc}") from exc
                state.note_decline(f"feta-error:{type(exc).__name__}")
        return out

    return override

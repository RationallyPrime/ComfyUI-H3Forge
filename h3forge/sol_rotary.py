"""Run-scoped optimizations for native ComfyUI H3 and Sol attention."""
from contextvars import ContextVar
from functools import lru_cache

import torch


class _TensorStamp:
    """Inference tensors lack version counters. Check their contents on reuse."""

    def __init__(self, tensor):
        self.tensor = tensor
        self.version = None if torch.is_inference(tensor) else tensor._version
        self.snapshot = tensor.detach().clone() if self.version is None else None

    def matches(self, tensor):
        if self.tensor is not tensor:
            return False
        if self.version is None:
            return torch.equal(self.snapshot, tensor)
        return self.version == tensor._version


class RotaryCache:
    def __init__(self):
        self.key = None
        self.value = None

    def _matches(self, model, layout, device, dtype):
        if self.key is None:
            return False
        old_model, old_layout, old_device, old_dtype, pos, freq = self.key
        return (old_model is model and old_layout is layout and old_device == device and old_dtype == dtype
                and pos.matches(layout.position_ids) and freq.matches(model.rope.inv_freq))

    def table(self, model, layout, device, dtype):
        from comfy.model_prefetch import pause_malloc_graph

        if self._matches(model, layout, device, dtype):
            return self.value

        # Persistent allocations must also be *released* outside the recorder.
        # Drop the old entry first so two large tables never overlap on a miss.
        with pause_malloc_graph():
            self.key = self.value = None
            value = model._rotation_table(layout, device, dtype)
            self.key = (model, layout, device, dtype, _TensorStamp(layout.position_ids),
                        _TensorStamp(model.rope.inv_freq))
            self.value = value
        return value


def install(model, cache_sigma, cache_rotary):
    from comfy.patcher_extension import WrappersMP

    key = "h3forge_sol_rotary"
    patched = model.clone()
    patched.remove_wrappers_with_key(WrappersMP.OUTER_SAMPLE, key)
    patched.remove_wrappers_with_key(WrappersMP.DIFFUSION_MODEL, key)
    if not (cache_sigma or cache_rotary):
        return patched

    from comfy_extras.nodes_sparse_attention import SparseAttnPatch

    diffusion = patched.get_model_object("diffusion_model")
    if ((cache_rotary and not hasattr(diffusion, "_rotation_table"))
            or (cache_sigma and not hasattr(SparseAttnPatch, "sigma_window_reason"))):
        raise RuntimeError("Apply H3Forge's patches/comfyui-0.45.0-h3-inference.patch to ComfyUI and restart.")

    active_cache = ContextVar(key, default=None)

    def outer(executor, *args, **kwargs):
        token = active_cache.set(RotaryCache())
        try:
            return executor(*args, **kwargs)
        finally:
            active_cache.reset(token)

    def forward(executor, x, timestep, context, transformer_options, **kwargs):
        # Keep other wrappers' option mutations visible; remove only our bindings.
        bindings = {}
        if cache_sigma:
            bindings["h3forge_sol_window"] = lru_cache(maxsize=1)(SparseAttnPatch.sigma_window_reason)
        if cache_rotary and active_cache.get() is not None:
            bindings["h3forge_rotary_cache"] = active_cache.get()
        missing = object()
        previous = {name: transformer_options.get(name, missing) for name in bindings}
        transformer_options.update(bindings)
        try:
            return executor(x, timestep, context, transformer_options, **kwargs)
        finally:
            for name, value in previous.items():
                if value is missing:
                    transformer_options.pop(name, None)
                else:
                    transformer_options[name] = value

    if cache_rotary:
        patched.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, key, outer)
    patched.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, key, forward)
    return patched

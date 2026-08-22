from __future__ import annotations

from comfy.patcher_extension import WrappersMP

from .attention import LOG, make_attention_override
from .context import ContextPolicy, make_context_wrapper
from .layout import padded_spatial_shape
from .prompt import encode_pipe_prompt, make_segmented_extra_conds
from .state import AttentionPolicy, RuntimeState, resolve_step

ATTN_KEY = "h3forge_attention"
CTX_KEY = "h3forge_context"
STAMP = "h3forge_block"


def _require_h3(model):
    diffusion = getattr(getattr(model, "model", None), "diffusion_model", None)
    if diffusion is None or type(diffusion).__name__ != "MiniMaxH3Model":
        raise ValueError(f"{LOG} MiniMax-H3 required; got {type(diffusion).__name__}")
    return diffusion


class H3ForgeAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "mode": (["flex_sliding", "dense"], {"default": "flex_sliding"}),
            "temporal_window": ("FLOAT", {"default": 40.0, "min": 1.0, "max": 256.0, "step": 1.0}),
            "spatial_radius": ("FLOAT", {"default": 8.0, "min": 0.25, "max": 64.0, "step": 0.25}),
            "bridge_stride": ("INT", {"default": 40, "min": 0, "max": 256}),
            "first_dense_layers": ("INT", {"default": 2, "min": 0, "max": 50}),
            "first_dense_fraction": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
            "feta_enabled": ("BOOLEAN", {"default": False}),
            "feta_strength": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 16.0, "step": 0.05}),
            "feta_max_gain": ("FLOAT", {"default": 1.15, "min": 1.0, "max": 2.0, "step": 0.01}),
            "feta_first_layer": ("INT", {"default": 6, "min": 0, "max": 49}),
            "feta_last_layer": ("INT", {"default": 42, "min": 0, "max": 49}),
            "strict": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/attention"
    DESCRIPTION = "H3-native sliding/radial sparse attention plus optional video-only FETA-style enrichment."

    def patch(self, model, mode, temporal_window, spatial_radius, bridge_stride,
              first_dense_layers, first_dense_fraction, feta_enabled, feta_strength,
              feta_max_gain, feta_first_layer, feta_last_layer, strict):
        diffusion = _require_h3(model)
        policy = AttentionPolicy(
            mode=mode, temporal_window=temporal_window, spatial_radius=spatial_radius,
            bridge_stride=bridge_stride, first_dense_layers=first_dense_layers,
            first_dense_fraction=first_dense_fraction, strict=strict,
            feta_enabled=feta_enabled, feta_strength=feta_strength,
            feta_max_gain=feta_max_gain, feta_first_layer=feta_first_layer,
            feta_last_layer=feta_last_layer,
        )
        state = RuntimeState(policy)
        patched = model.clone()
        opts = patched.model_options.setdefault("transformer_options", {})
        if "optimized_attention_override" in opts:
            print(
                f"{LOG} replacing an existing optimized_attention_override; "
                "do not stack SolAttnH3 with H3ForgeAttention",
                flush=True,
            )
        opts["optimized_attention_override"] = make_attention_override(state)
        patched.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, ATTN_KEY, _run_wrapper(state))
        patched.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, ATTN_KEY, _forward_wrapper(state))
        for i in range(len(diffusion.blocks)):
            patched.set_model_patch_replace(_stamp_block(state, i), "dit", "double_block", i)
        return (patched,)


def _stamp_block(state, index):
    def stamp(args, extra):
        state.block_index = index
        args["transformer_options"][STAMP] = index
        return extra["original_block"](args)
    return stamp


def _run_wrapper(state):
    def wrapper(executor, *args, **kwargs):
        state.begin_run()
        try:
            return executor(*args, **kwargs)
        finally:
            if state.sparse_calls or state.dense_calls:
                print(f"{LOG} {state.stats()}", flush=True)
    return wrapper


def _forward_wrapper(state):
    def wrapper(executor, x, timestep, context, transformer_options, **kwargs):
        # The token refiner can call optimized_attention before the first stamped
        # DiT block. Never let the previous forward's final block index make that
        # call look like block 49 (which can incorrectly activate FETA).
        state.block_index = None
        payload = kwargs.get("minimax_payload") or {}
        layout = payload.get("layout")
        if layout is None:
            try:
                from comfy.ldm.minimax.model import PackedLayout
                model = executor.class_obj
                padded_h, padded_w = padded_spatial_shape(x[0].shape[3], x[0].shape[4], model.patch_size)
                layout = PackedLayout(context.shape[1], x[0].shape[2], padded_h, padded_w, x[1].shape[-1],
                                      keyframes=payload.get("keyframes"), refs=payload.get("refs"))
            except Exception as exc:
                if state.policy.strict:
                    raise
                state.note_decline(f"layout-error:{type(exc).__name__}")
        state.layout = layout
        state.step_index, state.total_steps = resolve_step(transformer_options)
        sentinel = object()
        previous_layout = transformer_options.get("h3forge_active_layout", sentinel)
        if layout is not None:
            transformer_options["h3forge_active_layout"] = layout
        try:
            return executor(x, timestep, context, transformer_options, **kwargs)
        finally:
            state.block_index = None
            if previous_layout is sentinel:
                transformer_options.pop("h3forge_active_layout", None)
            else:
                transformer_options["h3forge_active_layout"] = previous_layout
    return wrapper


class H3ForgeContextWindows:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "window_frames": ("INT", {"default": 25, "min": 2, "max": 512}),
            "overlap_frames": ("INT", {"default": 5, "min": 0, "max": 256}),
            "stagger": ("BOOLEAN", {"default": True}),
            "blend": (["pyramid", "flat"], {"default": "pyramid"}),
            "strict": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/context"
    DESCRIPTION = "Synchronized MiniMax-H3 audio/video overlap-add context windows with absolute RoPE preservation."

    def patch(self, model, window_frames, overlap_frames, stagger, blend, strict):
        diffusion = _require_h3(model)
        if overlap_frames >= window_frames:
            raise ValueError("overlap_frames must be smaller than window_frames")
        policy = ContextPolicy(window_frames=window_frames, overlap_frames=overlap_frames,
                               stagger=stagger, blend=blend, strict=strict)
        patched = model.clone()
        base_model = patched.model
        base_extra_conds = patched.get_model_object("extra_conds")
        patched.add_object_patch(
            "extra_conds",
            make_segmented_extra_conds(base_extra_conds, base_model, diffusion),
        )
        patched.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, CTX_KEY, make_context_wrapper(policy))
        return (patched,)


class H3ForgePipePrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Encode | separated MiniMax-H3 prompts independently and map them in equal time spans "
        "through H3Forge context windows. Escape a literal pipe as \\|."
    )

    def encode(self, clip, prompt):
        try:
            return (encode_pipe_prompt(clip, prompt),)
        except ValueError as exc:
            raise ValueError(f"{LOG} {exc}") from exc


NODE_CLASS_MAPPINGS = {
    "H3ForgeAttention": H3ForgeAttention,
    "H3ForgeContextWindows": H3ForgeContextWindows,
    "H3ForgePipePrompt": H3ForgePipePrompt,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ForgeAttention": "H3 Forge — Sliding Attention + FETA",
    "H3ForgeContextWindows": "H3 Forge — Chained A/V Context Windows",
    "H3ForgePipePrompt": "H3 Forge — Pipe Timeline Prompt",
}

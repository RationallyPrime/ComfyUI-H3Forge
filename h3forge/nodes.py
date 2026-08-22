from __future__ import annotations

from comfy.patcher_extension import WrappersMP

from .attention import LOG, make_attention_override
from .context import ContextPolicy, make_context_wrapper
from .layout import padded_spatial_shape
from .nag import NAG_MODES, NAGConfig
from .prompt import (
    combine_conditioning_segments,
    encode_pipe_prompt,
    make_segmented_extra_conds,
    split_pipe_prompt,
)
from .state import AttentionPolicy, RuntimeState, resolve_sigma, resolve_step

ATTN_KEY = "h3forge_attention"
CTX_KEY = "h3forge_context"
STAMP = "h3forge_block"
STATE_GETTER = "h3forge_state_getter"
POLICY_KEY = "h3forge_attention_policy"
NAG_KEY = "h3forge_nag"


def _require_h3(model):
    diffusion = getattr(getattr(model, "model", None), "diffusion_model", None)
    if diffusion is None or type(diffusion).__name__ != "MiniMaxH3Model":
        raise ValueError(f"{LOG} MiniMax-H3 required; got {type(diffusion).__name__}")
    return diffusion


def _acquire_runtime(model, diffusion):
    """Clone the model and return (clone, transformer_options of the clone).

    The override, wrappers, and block stamps are installed once, whichever
    H3Forge node runs first; the shared RuntimeState behind them carries only
    caches and per-forward transients. Node configuration (attention policy,
    NAG config) is stored in the clone's transformer_options — deep-copied per
    ModelPatcher.clone(), so sibling branches and cached upstream outputs keep
    their own configuration — and resolved into the state at each forward.
    """
    patched = model.clone()
    opts = patched.model_options.setdefault("transformer_options", {})
    if opts.get(STATE_GETTER) is not None:
        state = opts[STATE_GETTER]()
        _bind_forward_config(patched, state, opts)
        return patched, opts

    state = RuntimeState(AttentionPolicy(mode="dense", feta_enabled=False))
    state.default_policy = state.policy
    state.diffusion = diffusion
    state.blocks = diffusion.blocks
    if "optimized_attention_override" in opts:
        print(
            f"{LOG} replacing an existing optimized_attention_override; "
            "do not stack SolAttnH3 with H3Forge attention nodes",
            flush=True,
        )
    opts["optimized_attention_override"] = make_attention_override(state)
    opts[STATE_GETTER] = lambda: state
    patched.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, ATTN_KEY, _run_wrapper(state))
    _bind_forward_config(patched, state, opts)
    for i in range(len(diffusion.blocks)):
        patched.set_model_patch_replace(_stamp_block(state, i), "dit", "double_block", i)
    return patched, opts


def _bind_forward_config(patched, state, configured_options):
    """Bind one model clone's H3Forge configuration to its forward wrapper.

    ComfyUI reconstructs the transformer-options dictionary for conditioned
    model calls. Wrappers and attention overrides are explicitly propagated,
    but arbitrary custom keys are not guaranteed to survive that path. Capture
    this clone's options in the wrapper that ComfyUI does preserve, and replace
    the inherited binding whenever another H3Forge config node creates a new
    branch. This keeps sibling branches isolated without depending on runtime
    option passthrough.
    """
    patched.remove_wrappers_with_key(WrappersMP.DIFFUSION_MODEL, ATTN_KEY)
    patched.add_wrapper_with_key(
        WrappersMP.DIFFUSION_MODEL,
        ATTN_KEY,
        _forward_wrapper(state, configured_options),
    )


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
        try:
            policy = AttentionPolicy(
                mode=mode, temporal_window=temporal_window, spatial_radius=spatial_radius,
                bridge_stride=bridge_stride, first_dense_layers=first_dense_layers,
                first_dense_fraction=first_dense_fraction, strict=strict,
                feta_enabled=feta_enabled, feta_strength=feta_strength,
                feta_max_gain=feta_max_gain, feta_first_layer=feta_first_layer,
                feta_last_layer=feta_last_layer,
            )
        except ValueError as exc:
            raise ValueError(f"{LOG} {exc}") from exc
        patched, opts = _acquire_runtime(model, diffusion)
        opts[POLICY_KEY] = policy
        return (patched,)


class H3ForgeNAG:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "negative": ("CONDITIONING",),
            "mode": (list(NAG_MODES), {"default": "lite"}),
            "nag_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.1}),
            "nag_tau": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1}),
            "nag_alpha": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
            "nag_sigma_end": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 1.0, "step": 0.01}),
            "first_block": ("INT", {"default": 8, "min": 0, "max": 49}),
            "last_block": ("INT", {"default": 28, "min": 0, "max": 49}),
            "video_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "audio_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
            "strict": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/guidance"
    EXPERIMENTAL = True
    DESCRIPTION = (
        "Experimental H3 NAG-Lite: restores negative-prompt control on the guidance-distilled H3 "
        "checkpoints at CFG 1 by guiding only the negative-text sidecar contribution to target "
        "audio/video attention rows. Not a faithful dual-branch NAG; see the README for the "
        "documented approximations."
    )

    def patch(self, model, negative, mode, nag_scale, nag_tau, nag_alpha, nag_sigma_end,
              first_block, last_block, video_strength, audio_strength, strict):
        diffusion = _require_h3(model)
        if len(negative) != 1:
            raise ValueError(f"{LOG} NAG expects exactly one negative conditioning entry, got {len(negative)}")
        try:
            config = NAGConfig(
                negative_context=negative[0][0], mode=mode, scale=nag_scale, tau=nag_tau,
                alpha=nag_alpha, sigma_end=nag_sigma_end, first_block=first_block,
                last_block=last_block, video_strength=video_strength,
                audio_strength=audio_strength, strict=strict,
            )
        except ValueError as exc:
            raise ValueError(f"{LOG} {exc}") from exc
        patched, opts = _acquire_runtime(model, diffusion)
        opts[NAG_KEY] = config
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
            # Free the NAG sidecar K/V cache when sampling ends; holding it
            # until the next H3Forge run would pin its VRAM through decoding.
            state.nag_runtime = None
            if state.sparse_calls or state.dense_calls:
                print(f"{LOG} {state.stats()}", flush=True)
    return wrapper


def _forward_wrapper(state, configured_options=None):
    def wrapper(executor, x, timestep, context, transformer_options, **kwargs):
        # Resolve configuration from the model clone whose wrapper ComfyUI
        # preserved. Fall back to runtime options for direct/unit-test callers.
        # A sibling branch therefore cannot inherit whatever another branch
        # most recently wrote into the shared runtime state.
        config = configured_options if configured_options is not None else transformer_options
        state.policy = config.get(POLICY_KEY, state.default_policy or state.policy)
        state.nag = config.get(NAG_KEY)
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
        state.current_sigma = resolve_sigma(transformer_options)
        state.step_index, state.total_steps = resolve_step(transformer_options, sigma=state.current_sigma)
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
        "Encode | separated MiniMax-H3 prompts independently; each H3Forge context window uses the "
        "segment covering its midpoint, and windows generated under different prompts crossfade in "
        "output space through the overlap-add blend. Escape a literal pipe as \\|."
    )

    def encode(self, clip, prompt):
        try:
            return (encode_pipe_prompt(clip, prompt),)
        except ValueError as exc:
            raise ValueError(f"{LOG} {exc}") from exc


class H3ForgeReferencePipePrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "ref_image_1": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "width": ("INT", {"default": 1344, "min": 32, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": 16384, "step": 32}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17}),
                "ref_image_size": (["match", "max"], {"default": "match"}),
            },
            "optional": {
                "ref_image_2": ("IMAGE",),
                "ref_image_3": ("IMAGE",),
                "ref_image_4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "LATENT")
    FUNCTION = "encode"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Encode | separated MiniMax-H3 Ref2VA prompts independently with the same one-to-four "
        "reference images. Returns native reference conditioning plus the H3 AV latent for use "
        "with H3Forge context windows. Reference video/audio inputs are not supported by this node."
    )

    def encode(self, clip, vae, audio_vae, ref_image_1, prompt, width, height, length,
               ref_image_size, ref_image_2=None, ref_image_3=None, ref_image_4=None):
        texts = split_pipe_prompt(prompt)
        if len(texts) < 2:
            raise ValueError(f"{LOG} reference pipe prompt requires at least two | separated segments")
        if len(texts) > 8:
            raise ValueError(f"{LOG} reference pipe prompt supports at most eight segments")

        try:
            from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
        except Exception as exc:
            raise RuntimeError(f"{LOG} native MiniMaxH3ReferenceToVideo is required") from exc

        images = [ref_image_1, ref_image_2, ref_image_3, ref_image_4]
        ref_images = {
            f"ref_image_{index}": image
            for index, image in enumerate(image for image in images if image is not None)
        }
        conditionings = []
        latent = None
        for text in texts:
            result = MiniMaxH3ReferenceToVideo.execute(
                clip=clip,
                vae=vae,
                audio_vae=audio_vae,
                prompt=text,
                width=width,
                height=height,
                length=length,
                ref_image_size=ref_image_size,
                ref_images=ref_images,
            )
            conditionings.append(result[0])
            if latent is None:
                latent = result[1]

        try:
            return combine_conditioning_segments(conditionings), latent
        except ValueError as exc:
            raise ValueError(f"{LOG} {exc}") from exc


NODE_CLASS_MAPPINGS = {
    "H3ForgeAttention": H3ForgeAttention,
    "H3ForgeContextWindows": H3ForgeContextWindows,
    "H3ForgePipePrompt": H3ForgePipePrompt,
    "H3ForgeReferencePipePrompt": H3ForgeReferencePipePrompt,
    "H3ForgeNAG": H3ForgeNAG,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ForgeAttention": "H3 Forge — Sliding Attention + FETA",
    "H3ForgeContextWindows": "H3 Forge — Chained A/V Context Windows",
    "H3ForgePipePrompt": "H3 Forge — Pipe Timeline Prompt",
    "H3ForgeReferencePipePrompt": "H3 Forge — Reference Pipe Timeline Prompt",
    "H3ForgeNAG": "H3 Forge — Normalized Attention Guidance",
}

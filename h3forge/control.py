"""Synchronize native H3 Fun ControlNet with a Forge video window."""
from contextlib import contextmanager

import torch


@contextmanager
def control_window(executor, full_shape, video_range, timestep, options, *, video_indices=None):
    """Keep the native cache in full-clip coordinates between local forwards.

    Core exposes its controls as bound DIFFUSION_MODEL wrappers. Looking through
    the executor's complete wrapper list works with controls before or after
    Forge; a matching local cache prevents core from re-encoding a VAE-invalid
    fragment or replaying the first control clip at every window.
    """
    patches = {id(p): p for w in getattr(executor, "wrappers", ())
               if type(p := getattr(w, "__self__", None)).__name__ == "MiniMaxH3FunControlPatch"}
    if not patches:
        yield
        return
    sigmas = options.get("sigmas")
    sigma = float(sigmas[0]) if sigmas is not None else float(timestep.flatten()[0]) / 1000
    saved = []
    try:
        for patch in patches.values():
            if not patch.sigma_end <= sigma <= patch.sigma_start:
                continue
            patch.prepare_control_latent(full_shape)
            saved.append((patch, patch.control_latent, patch.control_latent_shape))
            if video_indices is not None:
                idx = torch.tensor(video_indices, dtype=torch.long, device=patch.control_latent.device)
                patch.control_latent = patch.control_latent.index_select(2, idx)
                patch.control_latent_shape = (*full_shape[:2], len(video_indices), *full_shape[3:])
            else:
                v0, v1 = video_range
                patch.control_latent = patch.control_latent[:, :, v0:v1]
                patch.control_latent_shape = (*full_shape[:2], v1 - v0, *full_shape[3:])
            patch.control_stream = patch.pristine_stream = None
        yield
    finally:
        for patch, latent, shape in saved:
            patch.control_latent, patch.control_latent_shape = latent, shape
            patch.control_stream = patch.pristine_stream = None

"""Synchronize native H3 Fun ControlNet with a Forge video window."""
from contextlib import contextmanager

LOG = "[H3Forge]"
BEAT_CONTROL_KEY = "h3forge_beat_control"
_WORDS = {"on": 1.0, "off": 0.0}


def parse_beat_control(text):
    """Parse ``"1 | 1 | 0"``: how strongly the control network steers each beat.

    One value per beat, separated by ``|`` or by line breaks. A value multiplies
    the strength set on the control node: ``1`` (or ``on``) leaves it as set,
    ``0`` (or ``off``) leaves that beat to the prompt, ``0.5`` halves it. An
    empty spec returns ``None``: every beat is steered as the control node says.
    """
    if not str(text or "").strip():
        return None
    scales = []
    for number, chunk in enumerate(str(text).replace("\n", "|").split("|"), 1):
        token = chunk.strip().lower()
        try:
            scale = _WORDS[token] if token in _WORDS else float(token)
        except ValueError:
            raise ValueError(f"{LOG} control per beat: beat {number} is '{chunk.strip()}'; write a number such as "
                             "1, 0.5 or 0, or on / off") from None
        if not 0.0 <= scale < float("inf"):
            raise ValueError(f"{LOG} control per beat: beat {number} is {scale:g}; a strength cannot be negative")
        scales.append(scale)
    return tuple(scales)


def describe_beat_control(scales):
    return ", ".join(f"beat {i + 1} = " + ("off" if scale == 0 else f"{scale:g}") for i, scale in enumerate(scales))


def beat_control_scales(scales, beat_count):
    """The per-beat multipliers checked against the take, or ``None`` when there is nothing to change."""
    if scales is None:
        return None
    if len(scales) != int(beat_count):
        raise ValueError(
            f"{LOG} control per beat names {len(scales)} beat(s) but the take has {int(beat_count)}. "
            "Write one value per beat, separated by '|'.")
    return tuple(scales) if any(scale != 1.0 for scale in scales) else None


@contextmanager
def control_window(executor, full_shape, video_range, timestep, options, scale=1.0):
    """Keep the native cache in full-clip coordinates between local forwards.

    Core exposes its controls as bound DIFFUSION_MODEL wrappers. Looking through
    the executor's complete wrapper list works with controls before or after
    Forge; a matching local cache prevents core from re-encoding a VAE-invalid
    fragment or replaying the first control clip at every window.

    ``scale`` multiplies each control's strength for this window only. The
    native patch reads its strength when it adds each control block's output,
    so 0 runs the window without the control.
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
            saved.append((patch, patch.control_latent, patch.control_latent_shape, patch.strength))
            v0, v1 = video_range
            patch.control_latent = patch.control_latent[:, :, v0:v1]
            patch.control_latent_shape = (*full_shape[:2], v1 - v0, *full_shape[3:])
            patch.control_stream = patch.pristine_stream = None
            patch.strength = patch.strength * scale
        yield
    finally:
        for patch, latent, shape, strength in saved:
            patch.control_latent, patch.control_latent_shape, patch.strength = latent, shape, strength
            patch.control_stream = patch.pristine_stream = None

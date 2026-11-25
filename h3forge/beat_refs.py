"""Per-beat RefMod selection for pipe-prompt takes.

A RefMod is conditioning: the Continuum RefMod Bridge appends one reference
block per loaded mod to ``minimax_refs`` and the model packs each block's rows
between the text and the targets. Because every context window is its own
forward pass, a window can be handed a subset of those blocks. This module
turns a small text spec ("which mods does each beat see") into per-beat keep
lists and filters the native payload to match.

Only blocks the bridge marked ``refmod`` are ever dropped. Native Ref2VA
references and keyframes stay in every window.
"""
from __future__ import annotations

from dataclasses import dataclass
import os

LOG = "[H3Forge]"
BEAT_REFS_KEY = "h3forge_beat_refmods"
_ALL = ("*", "all")
_NONE = ("none", "-")


@dataclass(frozen=True)
class BeatRefs:
    """Which RefMod blocks each beat keeps, by position in the bridge's order."""
    names: tuple[str, ...]
    shapes: tuple[tuple, ...]
    beats: tuple[tuple[int, ...], ...]

    def describe(self) -> str:
        return "; ".join(
            f"beat {i + 1} = " + (", ".join(dict.fromkeys(self.names[j] for j in beat)) or "none")
            for i, beat in enumerate(self.beats))


def mod_labels(mod) -> tuple[str, ...]:
    """Names a mod answers to: its own name and the file it was loaded from."""
    labels = [str(getattr(mod, "name", "") or "")]
    path = str(getattr(mod, "path", "") or "")
    if path:
        labels.append(os.path.splitext(os.path.basename(path))[0])
    return tuple(dict.fromkeys(label for label in labels if label))


def mod_shape(mod) -> tuple:
    kind = getattr(mod, "kind", None)
    if kind == "audio":
        return ("audio",)
    return (kind, getattr(mod, "latent_h", None), getattr(mod, "latent_w", None))


def _block_shape(block) -> tuple:
    if block.get("kind") == "audio":
        return ("audio",)
    return (block.get("kind"), block.get("latent_h"), block.get("latent_w"))


def parse_beat_refs(text: str, mods) -> BeatRefs | None:
    """Parse ``"tairn, sgaeyl, lake | tairn, sgaeyl, hall"`` against a loaded mod list.

    Beats are separated by ``|`` or by line breaks, names by commas. A name
    matches every loaded mod whose name or file name equals it, or failing
    that contains it, ignoring case. ``*`` keeps every mod and ``none`` keeps
    no mod. An empty spec returns ``None``: every beat sees every mod.
    """
    if not str(text or "").strip():
        return None
    mods = [entry[0] if isinstance(entry, (tuple, list)) else entry for entry in mods or ()]
    if not mods:
        raise ValueError(f"{LOG} per-beat RefMods need the loaded mods; connect the RefMod loader's 'mods' output")
    labels = [mod_labels(mod) for mod in mods]
    names = tuple(label[0] if label else f"mod {i + 1}" for i, label in enumerate(labels))
    lowered = [tuple(label.lower() for label in entry) for entry in labels]
    known = ", ".join(dict.fromkeys(label for entry in labels for label in entry))

    beats = []
    for number, chunk in enumerate(str(text).replace("\n", "|").split("|"), 1):
        tokens = [token.strip() for token in chunk.split(",") if token.strip()]
        if not tokens:
            raise ValueError(f"{LOG} beat {number} names no RefMod; write 'none' to give a beat no RefMods")
        keep: set[int] = set()
        for token in tokens:
            needle = token.lower()
            if needle in _ALL:
                keep.update(range(len(mods)))
                continue
            if needle in _NONE:
                continue
            hits = [i for i, entry in enumerate(lowered) if needle in entry]
            if not hits:
                hits = [i for i, entry in enumerate(lowered) if any(needle in label for label in entry)]
            if not hits:
                raise ValueError(f"{LOG} beat {number}: no loaded RefMod matches '{token}'. Loaded: {known}")
            keep.update(hits)
        beats.append(tuple(sorted(keep)))
    return BeatRefs(names=names, shapes=tuple(mod_shape(mod) for mod in mods), beats=tuple(beats))


def beat_ref_keeps(refs, spec: BeatRefs | None, beat_count: int, *, positive: bool = True):
    """Per-beat keep lists over ``refs``, or ``None`` when nothing is filtered.

    With a spec set, a positive pass must receive the bridge's RefMod blocks:
    none arriving raises instead of sampling as a reference-free baseline. A
    negative (unconditional) pass carries no references and passes through.

    Each entry is ``None`` (the beat sees every reference) or the ascending
    positions in ``refs`` that beat keeps.
    """
    if spec is None:
        return None
    marked = [i for i, block in enumerate(refs or ()) if isinstance(block, dict) and block.get("refmod")]
    if not marked:
        if not positive:
            return None
        raise ValueError(
            f"{LOG} per-beat RefMods were set up for {len(spec.names)} reference(s) but the sampler received 0. "
            "Without them this take would run as a reference-free baseline. Connect the same 'mods' the RefMod "
            "bridge uses, keep the bridge enabled with its retention above 0 and leave its scramble_seed at -1.")
    if len(marked) != len(spec.names):
        raise ValueError(
            f"{LOG} per-beat RefMods were set up for {len(spec.names)} reference(s) but the sampler received "
            f"{len(marked)}. Connect the same 'mods' the RefMod bridge uses, keep its retention above 0 and "
            "leave its scramble_seed at -1.")
    for position, (index, shape) in enumerate(zip(marked, spec.shapes)):
        got = _block_shape(refs[index])
        if None not in shape and None not in got and tuple(shape) != tuple(got):
            raise ValueError(
                f"{LOG} RefMod reference {position + 1} ('{spec.names[position]}') does not match the loaded mod "
                f"order ({got} vs {shape}). Leave the bridge's scramble_seed at -1 for per-beat RefMods.")
    if len(spec.beats) != int(beat_count):
        raise ValueError(
            f"{LOG} per-beat RefMods name {len(spec.beats)} beat(s) but the take has {int(beat_count)}. "
            "Write one group per beat, separated by '|'.")
    keeps = []
    for beat in spec.beats:
        if len(beat) == len(marked):
            keeps.append(None)
            continue
        dropped = {marked[j] for j in range(len(marked)) if j not in beat}
        keeps.append(tuple(i for i in range(len(refs)) if i not in dropped))
    return keeps if any(keep is not None for keep in keeps) else None


def filter_payload(payload: dict, keep) -> dict:
    """The native payload with only ``keep`` references, minus its prebuilt layout.

    ``cond_video_latents`` / ``cond_audio_latents`` are the keyframe latents
    followed by one entry per reference that carries that stream, in reference
    order (ComfyUI ``MiniMaxH3._cond_latents``). The rows are consumed
    positionally against the packed layout, so they are filtered in step with
    ``refs`` and anything that does not add up fails instead of misaligning.
    """
    refs = list(payload.get("refs") or ())
    kept = set(keep)
    out = {key: value for key, value in payload.items() if key != "layout"}
    out["refs"] = [refs[i] for i in keep]
    keyframes = payload.get("keyframes") or ()
    streams = (
        ("cond_video_latents", [i for i, block in enumerate(refs) if "latent" in block],
         sum(1 for kf in keyframes if kf.get("latent") is not None)),
        ("cond_audio_latents", [i for i, block in enumerate(refs) if block.get("audio_latent") is not None],
         sum(1 for kf in keyframes if kf.get("audio_latent") is not None)),
    )
    for key, owners, lead in streams:
        if key not in payload:
            continue
        latents = list(payload[key])
        if len(latents) != lead + len(owners):
            raise ValueError(
                f"{LOG} {key} holds {len(latents)} entries where {lead} keyframe + {len(owners)} reference "
                "entries were expected; cannot select references per beat on this ComfyUI version")
        out[key] = latents[:lead] + [z for i, z in zip(owners, latents[lead:]) if i in kept]
    return out

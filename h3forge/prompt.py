from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from decimal import Decimal, InvalidOperation
from fractions import Fraction

import torch


DEFAULT_SEGMENT_DELIMITER = "|"
_TOKEN_OPEN = "<|"
_TOKEN_CLOSE = "|>"
# A MiniMax special token is ``<|name|>`` with a bare identifier inside. Anchoring the
# match here means an unterminated ``<|`` cannot borrow a later token's closer and
# swallow every delimiter in between; it stays ordinary text and splits normally.
_TOKEN_PATTERN = re.compile(re.escape(_TOKEN_OPEN) + r"\w+" + re.escape(_TOKEN_CLOSE))


def validate_segment_delimiter(delimiter: str) -> str:
    """Reject delimiters that cannot be scanned unambiguously."""
    if not delimiter:
        raise ValueError("segment delimiter must not be empty")
    if "\\" in delimiter:
        raise ValueError("segment delimiter must not contain a backslash; it introduces an escape")
    if "<" in delimiter or ">" in delimiter:
        raise ValueError(
            "segment delimiter must not contain < or >; those bracket MiniMax's "
            f"{_TOKEN_OPEN}...{_TOKEN_CLOSE} tokens, which are never split"
        )
    return delimiter


def split_pipe_prompt(prompt: str, delimiter: str = DEFAULT_SEGMENT_DELIMITER) -> list[str]:
    r"""Split a timeline prompt on ``delimiter``; ``\<delimiter>`` is a literal.

    MiniMax spells its own special tokens ``<|cutoff|>``, ``<|lyrics_start|>`` and so on,
    so the historical ``|`` delimiter cuts one in half and silently turns a single prompt
    into extra empty-ish segments. Text between ``<|`` and ``|>`` is therefore never
    split, whatever the delimiter, and the delimiter itself may not contain the brackets
    that would make that rule ambiguous.
    """
    validate_segment_delimiter(delimiter)
    segments: list[str] = []
    current: list[str] = []
    index, length = 0, len(prompt)
    while index < length:
        char = prompt[index]
        if char == "\\":
            if prompt.startswith(delimiter, index + 1):
                current.append(delimiter)
                index += 1 + len(delimiter)
                continue
            if index + 1 < length and prompt[index + 1] == "\\":
                current.append("\\")
                index += 2
                continue
            current.append("\\")
            index += 1
            continue
        token = _TOKEN_PATTERN.match(prompt, index)
        if token is not None:
            current.append(token.group(0))
            index = token.end()
            continue
        if prompt.startswith(delimiter, index):
            segments.append("".join(current).strip())
            current = []
            index += len(delimiter)
            continue
        current.append(char)
        index += 1
    segments.append("".join(current).strip())

    empty = [str(i + 1) for i, segment in enumerate(segments) if not segment]
    if empty:
        raise ValueError(f"pipe prompt contains empty segment(s): {', '.join(empty)}")
    return segments


Duration = int | float | str | Fraction


def _duration_fraction(value: Duration) -> Fraction:
    """Resolve one duration without inheriting avoidable binary-float drift."""
    if isinstance(value, Fraction):
        resolved = value
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("segment durations must be finite and greater than zero")
        # ``str(float)`` retains the user's decimal-scale intent. Feeding the
        # float directly to Fraction would instead preserve its binary storage
        # error, so equivalent ratios such as (1,3) and (1e307,3e307) could
        # disagree exactly at a cut boundary.
        resolved = Fraction(str(value))
    else:
        try:
            resolved = Fraction(str(value).strip())
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError("segment durations must contain only numbers") from exc
    if resolved <= 0:
        raise ValueError("segment durations must be finite and greater than zero")
    return resolved


def parse_segment_durations(raw: str | None, count: int) -> tuple[Fraction, ...]:
    """Parse positive comma/newline-delimited prompt durations.

    Values are deliberately unitless: ``2,18,40`` can mean seconds, frames,
    beats, or any other durations because only their relative proportions are
    needed to map them onto the target latent timeline. An empty input retains
    the original equal-duration behaviour.
    """
    if count < 1:
        raise ValueError("segment count must be positive")
    if raw is None or not str(raw).strip():
        return (Fraction(1),) * count

    parts = [part.strip() for line in str(raw).splitlines() for part in line.split(",")]
    if any(not part for part in parts):
        raise ValueError("segment_durations contains an empty value")
    if len(parts) != count:
        raise ValueError(
            f"segment_durations needs exactly one value per prompt segment "
            f"({count} segments, {len(parts)} values)"
        )
    try:
        decimals = tuple(Decimal(part) for part in parts)
    except InvalidOperation as exc:
        raise ValueError("segment_durations must contain only numbers") from exc
    if any(not value.is_finite() or value <= 0 for value in decimals):
        raise ValueError("segment_durations values must be finite and greater than zero")
    return tuple(Fraction(part) for part in parts)


def compose_segment_prompts(segments: Sequence[str], global_prompt: str = "") -> list[str]:
    """Repeat an optional global anchor inside every independent encoding."""
    anchor = str(global_prompt).strip()
    if not anchor:
        return list(segments)
    return [f"{anchor}\n\n{segment}" for segment in segments]


def segment_ranges(total: int, count: int, durations=None) -> tuple[list[tuple[int, int]], list[int]]:
    """Project relative durations to the nearest native video-token boundaries.

    Return half-open latent ranges and their decoded-frame cuts. A beat shorter
    than the native grid is rejected before any denoiser forward, never dropped.
    """
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN

    if count < 1:
        raise ValueError("segment count must be positive")
    weights = tuple(_duration_fraction(v) for v in (durations or (1,) * count))
    if len(weights) != count:
        raise ValueError(f"expected {count} segment durations, got {len(weights)}")
    frames = [0]
    for i in range(total):
        frames.append(frames[-1] + FRAME_PER_TOKEN[i % len(FRAME_PER_TOKEN)])
    cuts, cumulative = [0], Fraction(0)
    for weight in weights[:-1]:
        cumulative += weight
        target = cumulative * frames[-1] / sum(weights)
        cuts.append(min(range(total + 1), key=lambda i: abs(frames[i] - target)))
    cuts.append(total)
    if any(a >= b for a, b in zip(cuts, cuts[1:])):
        raise ValueError("a prompt segment is shorter than the native video-token grid; lengthen that duration")
    return list(zip(cuts, cuts[1:])), [frames[i] for i in cuts]


def _same_payload(a, b):
    """Compare the tensor/list/dict values in native H3 reference payloads."""
    if a is b:
        return True
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.shape == b.shape and torch.equal(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_payload(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same_payload(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def make_segmented_extra_conds(
    base_extra_conds: Callable,
    base_model,
    diffusion,
):
    """Carry raw pipe contexts through MiniMax's existing constant payload."""
    def extra_conds(**kwargs):
        out = base_extra_conds(**kwargs)
        segments = kwargs.get("h3forge_prompt_segments")
        if not segments:
            return out

        payload_cond = out.get("minimax_payload")
        if payload_cond is None or not isinstance(getattr(payload_cond, "cond", None), dict):
            raise ValueError("H3Forge pipe prompts require MiniMax-H3 conditioning")

        device = kwargs["device"]
        dtype = base_model.get_dtype_inference()
        # Native extra_conds already refined the primary, at its real length.
        processed = [out["c_crossattn"].cond]
        processed.extend(diffusion.preprocess_text_embeds(segment.to(device=device, dtype=dtype))
                         for segment in segments[1:])

        payload = dict(payload_cond.cond)
        payload["h3forge_prompt_segments"] = tuple(processed)
        payload["h3forge_prompt_segment_tags"] = kwargs["h3forge_prompt_segment_tags"]
        # This full layout is a coordinate template only. Local calls contain
        # each segment's real tokens, with refs/targets at one common origin.
        layout = payload.get("layout")
        if layout is not None and max(c.shape[1] for c in processed) != layout.signature[0]:
            from comfy.ldm.minimax.model import PackedLayout
            payload["layout"] = PackedLayout(max(c.shape[1] for c in processed), *layout.signature[1:],
                                              keyframes=payload.get("keyframes"), refs=payload.get("refs"))
        durations = kwargs.get("h3forge_prompt_segment_durations")
        if durations is not None:
            payload["h3forge_prompt_segment_durations"] = tuple(
                _duration_fraction(value) for value in durations
            )
        out["minimax_payload"] = payload_cond._copy_with(payload)
        return out

    return extra_conds


def combine_conditioning_segments(
    conditionings: Sequence,
    durations: Sequence[Duration] | None = None,
):
    """Combine independently encoded MiniMax conditionings into one timeline.

    Each input may already contain native reference-aware Qwen context and DiT
    payload metadata. The first entry supplies the shared payload while every
    native-length context remains available for per-window selection.
    """
    encoded = []
    metadata = []
    for conditioning in conditionings:
        if len(conditioning) != 1:
            raise ValueError("pipe prompt segments must each encode to one conditioning entry")
        encoded.append(conditioning[0][0])
        metadata.append(conditioning[0][1])

    if not encoded:
        raise ValueError("at least one prompt segment is required")
    for context in encoded:
        if context.ndim != 3 or context.shape[0] != 1 or context.shape[1] < 1:
            raise ValueError(f"expected segment context [1,T,C], got {tuple(context.shape)}")
        if context.shape[-1] != encoded[0].shape[-1]:
            raise ValueError("all segment contexts must have the same channel width")
    resolved_durations = tuple(_duration_fraction(v) for v in (durations or (1,) * len(encoded)))
    if len(resolved_durations) != len(encoded):
        raise ValueError(f"expected one duration per prompt segment ({len(encoded)} segments)")
    tags = [meta.get("minimax_token_tags", torch.ones(c.shape[1], dtype=torch.long)).reshape(-1)
            for c, meta in zip(encoded, metadata)]
    for context, tag in zip(encoded, tags):
        if tag.numel() != context.shape[1]:
            raise ValueError("MiniMax token tags must match the real encoded context length")
    for index, meta in enumerate(metadata[1:], start=2):
        if set(meta) != set(metadata[0]):
            raise ValueError(f"pipe prompt segments 1 and {index} produce different conditioning metadata keys")
        for key in ("minimax_refs", "minimax_keyframes", "minimax_visual_cond_noise_aug", "minimax_audio_cond_noise_aug"):
            if not _same_payload(metadata[0].get(key), meta.get(key)):
                raise ValueError(f"pipe prompt segments 1 and {index} have different shared {key} payloads")
        # Text lengths may differ; the shared vision presentation may not.
        for value in (0, 2):
            if not torch.equal(torch.where(tags[0] == value)[0], torch.where(tags[index - 1] == value)[0]):
                raise ValueError(f"pipe prompt segments 1 and {index} produce different MiniMax token tags")

    primary_meta = metadata[0].copy()
    primary_meta["minimax_token_tags"] = tags[0]
    primary_meta["h3forge_prompt_segments"] = tuple(encoded)
    primary_meta["h3forge_prompt_segment_tags"] = tuple(tags)
    primary_meta["h3forge_prompt_segment_count"] = len(encoded)
    primary_meta["h3forge_prompt_segment_durations"] = resolved_durations
    return [[encoded[0], primary_meta]]


def encode_pipe_prompt(
    clip,
    prompt: str,
    global_prompt: str = "",
    segment_durations: str = "",
    delimiter: str = DEFAULT_SEGMENT_DELIMITER,
):
    """Encode each text-only timeline segment and return one annotated conditioning."""
    texts = split_pipe_prompt(prompt, delimiter)
    durations = parse_segment_durations(segment_durations, len(texts))
    texts = compose_segment_prompts(texts, global_prompt)
    conditionings = [
        clip.encode_from_tokens_scheduled(clip.tokenize(text))
        for text in texts
    ]
    return combine_conditioning_segments(conditionings, durations)

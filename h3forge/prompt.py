from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from decimal import Decimal, InvalidOperation
from fractions import Fraction

import torch
import torch.nn.functional as F


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


def pad_segment_contexts(contexts: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """Right-pad independently encoded [1, tokens, channels] contexts."""
    if not contexts:
        raise ValueError("at least one prompt segment is required")
    hidden = contexts[0].shape[-1]
    max_tokens = max(int(context.shape[1]) for context in contexts)
    padded = []
    for context in contexts:
        if context.ndim != 3 or context.shape[0] != 1:
            raise ValueError(f"expected segment context [1,T,C], got {tuple(context.shape)}")
        if context.shape[-1] != hidden:
            raise ValueError("all segment contexts must have the same channel width")
        padded.append(F.pad(context, (0, 0, 0, max_tokens - context.shape[1])))
    return padded


def pad_text_tags(tags: torch.Tensor | None, tokens: int) -> torch.Tensor:
    """Pad MiniMax text-modality tags to the common segment token length."""
    if tags is None:
        return torch.ones(tokens, dtype=torch.long)
    tags = tags.reshape(-1)
    if tags.shape[0] > tokens:
        raise ValueError("MiniMax text token tags exceed the encoded context length")
    return F.pad(tags, (0, tokens - tags.shape[0]), value=1)


def select_segment_index(
    v0: int,
    v1: int,
    total: int,
    count: int,
    durations: Sequence[Duration] | None = None,
) -> int:
    """Pick the prompt segment whose duration span contains the window midpoint.

    Contextualized hidden states from independently encoded prompts do not
    share a common token basis, so a boundary window uses the one segment that
    dominates it; adjacent windows generated under different prompts crossfade
    in output space through the context-window overlap-add blend instead.
    """
    if count < 1:
        raise ValueError("segment count must be positive")
    if not (0 <= v0 < v1 <= total):
        raise ValueError(f"invalid window [{v0}, {v1}) for total {total}")
    weights = tuple(_duration_fraction(value) for value in (durations or (Fraction(1),) * count))
    if len(weights) != count:
        raise ValueError(f"expected {count} segment durations, got {len(weights)}")

    # Only the durations' ratios carry meaning, so the boundary predicate is
    # evaluated in exact rational arithmetic from their decimal representation.
    # The float form
    # ``midpoint * sum(weights) / total`` overflows to ``inf`` for large finite
    # durations such as ``1e307,1e307`` and routes every window to the final
    # segment; float normalization avoids the overflow but makes a midpoint
    # sitting exactly on a boundary land on whichever side the rounding fell.
    total_weight = sum(weights)
    target = Fraction(v0 + v1, 2 * total) * total_weight
    boundary = Fraction(0)
    for index, weight in enumerate(weights[:-1]):
        boundary += weight
        if target < boundary:
            return index
    return count - 1


def unreachable_segments(
    starts: Sequence[int],
    window: int,
    total: int,
    count: int,
    durations: Sequence[Duration] | None = None,
) -> list[int]:
    """Return zero-based segment indices no context window would ever select.

    Window midpoints are quantized to the scheduler's stride and confined to
    [window/2, total - window/2], so when segments outnumber windows some
    prompts are silently unreachable; callers surface that instead of letting
    a prompt vanish without a trace.
    """
    selected = {
        select_segment_index(v0, min(v0 + window, total), total, count, durations)
        for v0 in starts
    }
    return sorted(set(range(count)) - selected)


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
        processed = []
        for segment in segments:
            segment = segment.to(device=device, dtype=dtype)
            processed.append(diffusion.preprocess_text_embeds(segment))

        payload = dict(payload_cond.cond)
        payload["h3forge_prompt_segments"] = tuple(processed)
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
    padded context remains available for per-window selection.
    """
    encoded = []
    metadata = []
    for conditioning in conditionings:
        if len(conditioning) != 1:
            raise ValueError("pipe prompt segments must each encode to one conditioning entry")
        encoded.append(conditioning[0][0])
        metadata.append(conditioning[0][1])

    padded = pad_segment_contexts(encoded)
    resolved_durations = tuple(
        _duration_fraction(value) for value in (durations or (Fraction(1),) * len(padded))
    )
    if len(resolved_durations) != len(padded):
        raise ValueError(
            f"expected one duration per prompt segment ({len(padded)} segments, "
            f"{len(resolved_durations)} durations)"
        )
    tokens = padded[0].shape[1]
    padded_tags = [pad_text_tags(meta.get("minimax_token_tags"), tokens) for meta in metadata]
    # The run carries one set of conditioning metadata, so it is only correct
    # when every segment shares it. Refuse divergent multimodal structure
    # instead of silently stamping segment 1's tags onto every window.
    for index, tags in enumerate(padded_tags[1:], start=2):
        if not torch.equal(tags, padded_tags[0]):
            raise ValueError(
                f"pipe prompt segments 1 and {index} produce different MiniMax token tags; "
                "multimodal inserts and presentation tags must be identical across segments"
            )
    for index, meta in enumerate(metadata[1:], start=2):
        if set(meta.keys()) != set(metadata[0].keys()):
            raise ValueError(
                f"pipe prompt segments 1 and {index} produce different conditioning metadata keys; "
                "segments must use the same conditioning structure"
            )

    primary_meta = metadata[0].copy()
    primary_meta["minimax_token_tags"] = padded_tags[0]
    primary_meta["h3forge_prompt_segments"] = tuple(padded)
    primary_meta["h3forge_prompt_segment_count"] = len(padded)
    primary_meta["h3forge_prompt_segment_durations"] = resolved_durations
    return [[padded[0], primary_meta]]


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

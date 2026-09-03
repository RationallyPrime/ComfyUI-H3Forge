from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F


def split_pipe_prompt(prompt: str) -> list[str]:
    r"""Split a pipe timeline, supporting ``\|`` as a literal pipe."""
    segments: list[str] = []
    current: list[str] = []
    escaped = False
    for char in prompt:
        if escaped:
            if char not in ("|", "\\"):
                current.append("\\")
            current.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "|":
            segments.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if escaped:
        current.append("\\")
    segments.append("".join(current).strip())

    empty = [str(i + 1) for i, segment in enumerate(segments) if not segment]
    if empty:
        raise ValueError(f"pipe prompt contains empty segment(s): {', '.join(empty)}")
    return segments


def parse_segment_durations(raw: str | None, count: int) -> tuple[float, ...]:
    """Parse positive comma/newline-delimited prompt durations.

    Values are deliberately unitless: ``2,18,40`` can mean seconds, frames,
    beats, or any other durations because only their relative proportions are
    needed to map them onto the target latent timeline. An empty input retains
    the original equal-duration behaviour.
    """
    if count < 1:
        raise ValueError("segment count must be positive")
    if raw is None or not str(raw).strip():
        return (1.0,) * count

    parts = [part.strip() for line in str(raw).splitlines() for part in line.split(",")]
    if any(not part for part in parts):
        raise ValueError("segment_durations contains an empty value")
    if len(parts) != count:
        raise ValueError(
            f"segment_durations needs exactly one value per prompt segment "
            f"({count} segments, {len(parts)} values)"
        )
    try:
        durations = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError("segment_durations must contain only numbers") from exc
    if any(not math.isfinite(value) or value <= 0 for value in durations):
        raise ValueError("segment_durations values must be finite and greater than zero")
    return durations


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
    durations: Sequence[float] | None = None,
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
    weights = tuple(float(value) for value in (durations or (1.0,) * count))
    if len(weights) != count:
        raise ValueError(f"expected {count} segment durations, got {len(weights)}")
    if any(not math.isfinite(value) or value <= 0 for value in weights):
        raise ValueError("segment durations must be finite and greater than zero")

    target = ((v0 + v1) / 2.0) * sum(weights) / total
    boundary = 0.0
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
    durations: Sequence[float] | None = None,
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
            payload["h3forge_prompt_segment_durations"] = tuple(float(value) for value in durations)
        out["minimax_payload"] = payload_cond._copy_with(payload)
        return out

    return extra_conds


def combine_conditioning_segments(
    conditionings: Sequence,
    durations: Sequence[float] | None = None,
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
    resolved_durations = tuple(float(value) for value in (durations or (1.0,) * len(padded)))
    if len(resolved_durations) != len(padded):
        raise ValueError(
            f"expected one duration per prompt segment ({len(padded)} segments, "
            f"{len(resolved_durations)} durations)"
        )
    if any(not math.isfinite(value) or value <= 0 for value in resolved_durations):
        raise ValueError("segment durations must be finite and greater than zero")
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
):
    """Encode each text-only pipe segment and return one annotated conditioning."""
    texts = split_pipe_prompt(prompt)
    durations = parse_segment_durations(segment_durations, len(texts))
    texts = compose_segment_prompts(texts, global_prompt)
    conditionings = [
        clip.encode_from_tokens_scheduled(clip.tokenize(text))
        for text in texts
    ]
    return combine_conditioning_segments(conditionings, durations)

from __future__ import annotations

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


def segment_overlap_weights(v0: int, v1: int, total: int, count: int) -> list[float]:
    """Return normalized equal-timeline overlap weights for one context window."""
    if count < 1:
        raise ValueError("segment count must be positive")
    if not (0 <= v0 < v1 <= total):
        raise ValueError(f"invalid window [{v0}, {v1}) for total {total}")

    weights = []
    for index in range(count):
        start = total * index / count
        stop = total * (index + 1) / count
        weights.append(max(0.0, min(float(v1), stop) - max(float(v0), start)))
    denominator = sum(weights)
    if denominator <= 0.0:
        midpoint = (v0 + v1) * 0.5
        selected = min(int(midpoint * count / total), count - 1)
        return [1.0 if index == selected else 0.0 for index in range(count)]
    return [weight / denominator for weight in weights]


def blend_segment_contexts(contexts: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
    if len(contexts) != len(weights) or not contexts:
        raise ValueError("segment contexts and weights must have the same non-zero length")
    active = [(context, weight) for context, weight in zip(contexts, weights) if weight > 0.0]
    if len(active) == 1:
        return active[0][0]
    output = torch.zeros_like(active[0][0])
    for context, weight in active:
        output.add_(context, alpha=float(weight))
    return output


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
        out["minimax_payload"] = payload_cond._copy_with(payload)
        return out

    return extra_conds


def encode_pipe_prompt(clip, prompt: str):
    """Encode each pipe segment independently and return one annotated conditioning."""
    texts = split_pipe_prompt(prompt)
    encoded = []
    metadata = []
    for text in texts:
        conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(text))
        if len(conditioning) != 1:
            raise ValueError("pipe prompt segments must each encode to one conditioning entry")
        encoded.append(conditioning[0][0])
        metadata.append(conditioning[0][1])

    padded = pad_segment_contexts(encoded)
    primary_meta = metadata[0].copy()
    primary_meta["minimax_token_tags"] = pad_text_tags(
        primary_meta.get("minimax_token_tags"), padded[0].shape[1]
    )
    primary_meta["h3forge_prompt_segments"] = tuple(padded)
    primary_meta["h3forge_prompt_segment_count"] = len(padded)
    return [[padded[0], primary_meta]]

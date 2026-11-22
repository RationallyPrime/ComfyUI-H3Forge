"""Prepare early text conditioning while the original H3 Qwen contexts are available."""
from __future__ import annotations

from bisect import bisect_left
import re
import unicodedata

import torch


TEXT_HANDOVER_MODES = ("off", "reencode", "token_reuse")
DEFAULT_HANDOVER_STEPS = 16
HANDOVER_KEY = "h3forge_text_handover"
_VISUAL = re.compile(r"(?mi)^[ \t]*(?:integrated_multimodal_description|detailed_description)[ \t]*:[ \t]*")
_AUDIO = re.compile(r"(?mi)^[ \t]*(?:overall_soundscape|non_diegetic_music)[ \t]*:")
_SHOT = re.compile(r"\[Shot\s+\d+\]\s*", re.IGNORECASE)
_SENTENCE_END = re.compile(r"""(?:(?P<speech><d>.*?</d>)|(?P<stop>[.!?。！？]+))["'”’»)\]]*""", re.DOTALL)
_SPEECH_END = re.compile(r"""([.!?。！？])["'”’»)\]]*\s*</d>$""")


def _visual_span(beat):
    field = _VISUAL.search(beat)
    start = field.end() if field else 0
    sound = _AUDIO.search(beat, start)
    end = sound.start() if sound else len(beat)
    while start < end and beat[start].isspace():
        start += 1
    label = _SHOT.match(beat, start, end)
    if label:
        start = label.end()
    while end > start and beat[end - 1].isspace():
        end -= 1
    if end <= start:
        raise ValueError("text hand-over needs a visual description in every beat")
    return start, end


def _ending_span(beat, start, end):
    # Treat native dialogue as one unit. Interior punctuation cannot start a fragment
    # without its opening <d>; a completed utterance can precede a final visual action.
    for boundary in _SENTENCE_END.finditer(beat, start, end):
        stop = boundary["stop"]
        if stop is None:
            last_spoken = _SPEECH_END.search(boundary["speech"])
            if last_spoken is None:
                continue
            stop = last_spoken[1]
        next_word = boundary.end()
        if next_word < end and (beat[next_word].isspace() or stop[-1] in "。！？"):
            while next_word < end and beat[next_word].isspace():
                next_word += 1
            if next_word < end:
                start = next_word
    label = _SHOT.match(beat, start, end)
    start = label.end() if label else start
    if end <= start:
        raise ValueError("text hand-over needs an ending sentence after the shot label")
    return start, end


def _sentence_rows(clip, tokenize, text, start, end, raw, tags):
    """Locate an authored caption span in its original, possibly vision-expanded Qwen output."""
    qwen = clip.tokenizer.qwen3vl_32b
    caption = qwen.tokenize_with_weights(text, disable_weights=True)[0]
    ids = [entry[0] for entry in caption]
    presented = tokenize(text)["qwen3vl_32b"][0]
    if (not ids or not all(isinstance(t, int) for t in ids)
            or len(presented) < len(ids) or [entry[0] for entry in presented[-len(ids):]] != ids):
        raise ValueError("text hand-over cannot align the caption IDs with the native H3 presentation")

    def decode(tokens):
        return qwen.tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False,
                                     spaces_between_special_tokens=False)

    def normalized(s):
        # Native SDTokenizer unescapes parentheses and Qwen can normalize Unicode.
        return unicodedata.normalize("NFC", s.replace(r"\(", "(").replace(r"\)", ")"))

    def decoded(tokens):
        return unicodedata.normalize("NFC", decode(tokens))

    if decoded(ids) != normalized(text):
        raise ValueError("text hand-over cannot map the decoded Qwen caption back to the authored text")
    # Qwen's slow tokenizer has no offset mapping. Find the shortest token prefix/suffix
    # covering each boundary. Decode whole slices so split UTF-8 codepoints stay intact.
    counts = range(len(ids) + 1)
    prefix, suffix = normalized(text[:end]), normalized(text[start:])
    stop = bisect_left(counts, True, key=lambda n: decoded(ids[:n]).startswith(prefix))
    tail = bisect_left(counts, True, key=lambda n: decoded(ids[len(ids) - n:]).endswith(suffix))
    first = len(ids) - tail
    if decoded(ids[first:stop]).strip() != normalized(text[start:end]):
        raise ValueError("text hand-over sentence shares a Qwen token with adjacent words; separate it with whitespace")
    # Native H3 appends the caption after all reference labels / expanded vision blocks.
    origin = raw.shape[1] - len(ids)
    rows = slice(origin + first, origin + stop)
    if origin < 0 or stop <= first or not torch.all(tags[rows] == 1):
        raise ValueError("text hand-over must select only original caption rows tagged as text")
    return raw[:, rows], tags[rows]


def prepare_text_handover(clip, beats, texts, conditioning, tokenize, mode, steps, sigmas):
    """Attach raw per-incoming-beat alternatives. No work or metadata when disabled."""
    if mode not in TEXT_HANDOVER_MODES:
        raise ValueError(f"text_handover must be one of {TEXT_HANDOVER_MODES}")
    if not isinstance(steps, int) or steps < 0:
        raise ValueError("handover_steps must be a nonnegative integer")
    if mode == "off" or steps == 0 or len(beats) == 1:
        return conditioning
    cutoff = None
    if sigmas is not None:
        if (sigmas.ndim != 1 or sigmas.numel() < 2 or not torch.all(torch.isfinite(sigmas))
                or torch.any(sigmas[1:] > sigmas[:-1])):
            raise ValueError("handover_sigmas must be the full descending sampler schedule")
        cutoff = float(sigmas[min(steps, len(sigmas) - 1)])

    meta = conditioning[0][1]
    native = meta["h3forge_prompt_segments"]
    native_tags = meta["h3forge_prompt_segment_tags"]
    spans = [_visual_span(beat) for beat in beats]
    early, tags = [], []
    for i in range(1, len(beats)):
        start, end = _ending_span(beats[i - 1], *spans[i - 1])
        if mode == "reencode":
            pos = len(texts[i]) - len(beats[i]) + spans[i][0]
            edited = texts[i][:pos] + beats[i - 1][start:end] + " " + texts[i][pos:]
            encoded = clip.encode_from_tokens_scheduled(tokenize(edited))
            if len(encoded) != 1:
                raise ValueError("text hand-over needs one conditioning entry per edited beat")
            raw, entry = encoded[0]
            tag = entry.get("minimax_token_tags", torch.ones(raw.shape[1], dtype=torch.long)).reshape(-1)
            if (raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[2] != native[i].shape[2]
                    or len(tag) != raw.shape[1] or any(not torch.equal(
                        torch.where(tag == kind)[0], torch.where(native_tags[i] == kind)[0]) for kind in (0, 2))):
                raise ValueError("text hand-over changed the native reference presentation or token layout")
        else:
            prefix = len(texts[i - 1]) - len(beats[i - 1])
            rows, row_tags = _sentence_rows(clip, tokenize, texts[i - 1], prefix + start, prefix + end,
                                            native[i - 1], native_tags[i - 1])
            raw = torch.cat((native[i], rows), dim=1)
            tag = torch.cat((native_tags[i], row_tags))
        early.append(raw)
        tags.append(tag)
    meta[HANDOVER_KEY] = {"mode": mode, "steps": steps, "cutoff": cutoff,
                          "segments": tuple(early), "tags": tuple(tags)}
    return conditioning


def handover_cutoff(handover, transformer_options):
    """Sigma at the first native step. A split sampler needs the original schedule."""
    if handover["cutoff"] is not None:
        return handover["cutoff"]
    schedule = transformer_options.get("sample_sigmas")
    if schedule is None or len(schedule) < 2 or float(schedule[-1]) != 0:
        raise ValueError("text hand-over needs the complete sampler schedule; connect handover_sigmas "
                         "before splitting it for frozen or partial sampling")
    return float(schedule[min(handover["steps"], len(schedule) - 1)])

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import torch

from .control import control_window
from .layout import audio_range_for_video_window, clone_window_layout, padded_spatial_shape
from .prompt import segment_ranges
from .state import require_eager_allocations, resolve_step

LOG = "[H3Forge]"
_LATENT_CADENCE = 5
SEGMENT_SEAMS = ("blend", "exclusive")


@dataclass(frozen=True)
class ContextPolicy:
    window_frames: int = 25
    overlap_frames: int = 8
    stagger: bool = True
    blend: str = "pyramid"
    segment_seams: str = "blend"
    freenoise: bool = True

    def __post_init__(self):
        if self.segment_seams not in SEGMENT_SEAMS:
            raise ValueError(f"segment_seams must be one of {SEGMENT_SEAMS}, got {self.segment_seams!r}")


def ordered_halving(value: int) -> float:
    """Return the bit-reversed base-2 fraction used for context staggering."""
    binary = f"{value:064b}"
    return int(binary[::-1], 2) / (1 << 64)


def max_stagger_phase(window: int, overlap: int) -> int:
    """Largest interior-boundary shift that still keeps every adjacent overlap.

    Staggering exists to move seams between steps, not to let a window jump
    across its neighbour: a shift larger than the requested overlap lets two
    windows abut with no blend and puts every latent under a different
    prompt/neighbour pair on alternate steps, which reads as constant morphing.
    """
    stride = window - overlap
    return max(0, min(overlap, stride - 1))


def stagger_phase(step: int, window: int, overlap: int) -> int:
    """Ordered-halving phase for ``step`` bounded to ``[0, max_stagger_phase]``."""
    return int(ordered_halving(step) * (max_stagger_phase(window, overlap) + 1))


def _shifted_starts(anchor: list[int], phase: int, stride: int, *, snap: bool) -> list[int]:
    """Greedy plan ``anchor`` shifted by ``phase``, keeping anchors, count and overlap.

    Every adjacent pair keeps at least ``overlap`` latents in common: the next
    start is never more than one stride away from the last one, and never so
    far back that the remaining windows could not reach the final anchor within
    a stride each. Because ``anchor`` itself satisfies those bounds, every
    interior start lands in ``[anchor[i], anchor[i] + phase]``: a phase never
    moves a seam further than itself, and a feasible ``anchor`` is a fixed point
    at phase 0. With ``snap`` each interior start moves to the nearest cadence
    point inside its feasible interval when one exists.
    """
    final_start = anchor[-1]
    count = len(anchor)
    starts = [0]
    for i in range(1, count - 1):
        remaining = count - 1 - i
        lower = max(starts[-1] + 1, final_start - remaining * stride)
        upper = min(starts[-1] + stride, final_start - remaining)
        candidate = min(max(anchor[i] + phase, lower), upper)
        if snap:
            first = ((lower + _LATENT_CADENCE - 1) // _LATENT_CADENCE) * _LATENT_CADENCE
            last = (upper // _LATENT_CADENCE) * _LATENT_CADENCE
            if first <= last:
                candidate = min(max(round(candidate / _LATENT_CADENCE) * _LATENT_CADENCE, first), last)
        starts.append(candidate)
    starts.append(final_start)
    return starts


def _stagger_layouts(anchor: list[int], stride: int, max_phase: int) -> int:
    """Number of distinct plans the phases ``0..max_phase`` reach from ``anchor``."""
    return len({tuple(_shifted_starts(anchor, phase, stride, snap=False)) for phase in range(max_phase + 1)})


def window_starts(total: int, window: int, overlap: int, phase: int = 0, max_phase: int = 0) -> list[int]:
    """Window starts for ``phase`` of a run whose stagger visits phases ``0..max_phase``.

    The phase-0 plan snaps interior starts to the latent cadence unless that
    would leave the stagger fewer distinct layouts than the even spread does;
    a static run (``max_phase == 0``) therefore always snaps, and a staggering
    run keeps its off-cadence spread only where cadence and seam movement
    collide. Every active phase is then derived from that same phase-0 plan,
    so no seam ever moves further than the phase.
    """
    if window >= total:
        return [0]
    if window < 2:
        raise ValueError("window must be >=2")
    stride = window - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than window")

    if not isinstance(phase, int) or not 0 <= phase < stride:
        raise ValueError(f"phase must be an integer in [0, {stride})")
    if not isinstance(max_phase, int) or not 0 <= max_phase < stride:
        raise ValueError(f"max_phase must be an integer in [0, {stride})")

    final_start = total - window
    count = 1 + (final_start + stride - 1) // stride
    base = [round(i * final_start / (count - 1)) for i in range(count)]
    if count <= 2:
        return base

    spread = _shifted_starts(base, 0, stride, snap=False)
    snapped = _shifted_starts(base, 0, stride, snap=True)
    anchor = snapped
    if _stagger_layouts(snapped, stride, max_phase) < _stagger_layouts(spread, stride, max_phase):
        anchor = spread
    return _shifted_starts(anchor, phase, stride, snap=False)


def blend_weights(length: int, overlap: int, *, device, dtype, mode="pyramid",
                  ramp_start: bool = True, ramp_end: bool = True):
    """Return full-window pyramid, overlap-linear, or flat fusion weights."""
    if mode not in {"pyramid", "overlap-linear", "flat"}:
        raise ValueError(f"unknown blend mode: {mode}")
    if length <= 0:
        return torch.empty(0, device=device, dtype=dtype)
    if mode == "flat":
        return torch.ones(length, device=device, dtype=dtype)

    if mode == "pyramid":
        peak = (length + 1) // 2
        ascending = torch.arange(1, peak + 1, device=device, dtype=torch.float32)
        descending_start = peak if length % 2 == 0 else peak - 1
        descending = torch.arange(descending_start, 0, -1, device=device, dtype=torch.float32)
        return torch.cat((ascending, descending)).to(dtype)

    if overlap <= 0 or not (ramp_start or ramp_end):
        return torch.ones(length, device=device, dtype=dtype)
    ramp = min(overlap, length // 2)
    if ramp == 0:
        return torch.ones(length, device=device, dtype=dtype)
    w = torch.ones(length, device=device, dtype=torch.float32)
    edge = torch.linspace(1.0 / (ramp + 1), 1.0, ramp, device=device)
    if ramp_start:
        w[:ramp] = edge
    if ramp_end:
        w[-ramp:] = edge.flip(0)
    return w.to(dtype)


def audio_overlap_frames(overlap_frames: int, video_len: int, audio_len: int) -> int:
    """Audio-latent overlap induced by the physical-time mapping of a video overlap.

    No video overlap means no audio overlap: the audio ramp must not be forced
    to one sample when the video windows do not overlap at all.
    """
    if overlap_frames <= 0:
        return 0
    return max(1, round(overlap_frames * max(audio_len / max(video_len, 1), 1.0)))


def assert_full_coverage(video_den: torch.Tensor, audio_den: torch.Tensor) -> None:
    """Every target element must carry positive blend weight after the overlap-add."""
    if not bool((video_den > 0).all()):
        raise RuntimeError("context windows left video latents with zero accumulated blend weight")
    if not bool((audio_den > 0).all()):
        raise RuntimeError("context windows left audio latents with zero accumulated blend weight")


def context_plan_summary(
    total: int,
    starts: list[int],
    window: int,
    overlap: int,
    *,
    phase: int,
    blend: str = "pyramid",
    stagger: bool = False,
    assignments: list[int] | None = None,
    frame_cuts: list[int] | None = None,
    audio_t: int | None = None,
    max_phase: int | None = None,
    seams: str | None = None,
) -> str:
    """Return one compact, truthful account of a context-window pass."""
    ranges = [(start, min(start + window, total)) for start in starts]
    latent_visits = sum(end - start for start, end in ranges) / max(total, 1)
    bits = [
        f"video_latents={total}",
        f"windows={len(ranges)}",
        f"window/overlap={window}/{overlap}",
        f"phase={phase}",
        f"video_latent_visits={latent_visits:.2f}x",
        f"stride={window - overlap}",
        f"min_overlap={min((window - (right - left) for left, right in pairwise(starts)), default=0)}",
        f"blend={blend}",
        f"stagger={'on' if stagger else 'off'}",
        f"cadence={_LATENT_CADENCE}",
        f"off_cadence_starts={sum(start % _LATENT_CADENCE != 0 for start in starts)}",
    ]
    if max_phase is not None:
        bits.append(f"max_phase={max_phase}")
    if assignments:
        runs = []
        for index in assignments:
            if runs and runs[-1][0] == index + 1:
                runs[-1][1] += 1
            else:
                runs.append([index + 1, 1])
        bits.append("prompt_windows=" + ",".join(f"{index}x{count}" for index, count in runs))
    if seams is not None:
        bits.append(f"seams={seams}")
    if frame_cuts is not None:
        bits.append("prompt_frame_cuts=" + ",".join(map(str, frame_cuts)))
    if audio_t is not None:
        bits.append(f"audio_context=full:{audio_t}")
    return " ".join(bits)


def _slice_optional_video(mask, v0, v1):
    if mask is None:
        return None
    return mask[:, :, v0:v1]


def _beat_for_window(v0, v1, ranges):
    """Index of the beat covering most of ``[v0, v1)``; ties go to the earlier beat."""
    best, best_shared = 0, -1
    for index, (lo, hi) in enumerate(ranges):
        shared = min(v1, hi) - max(v0, lo)
        if shared > best_shared:
            best, best_shared = index, shared
    return best


def _blended_segment_windows(total, window, overlap, ranges, phase=0, max_phase=0):
    """Uniform, staggerable windows that each write their whole extent under one beat's prompt.

    This is the WanVideoWrapper / ComfyUI-core arrangement: a window carries the
    prompt of the beat it mostly covers, and neighbouring windows blend across
    the beat boundary exactly like any other overlap, so a prompt change is a
    ramp about one overlap wide that staggering moves from step to step rather
    than a cut fixed at one latent. A beat that no window mostly covers (one
    shorter than the stride) still gets one window centred on it that owns the
    beat outright: it writes only inside the beat, and the regular windows do
    not write there, so the beat is that prompt's alone rather than a mixture
    with whichever prompt the covering window carries. Every prompt therefore
    reaches the model on every step. With a single regular window this reduces
    to exclusive ownership for every beat, since there is no overlap to blend.

    Returns ``(index, v0, v1, w0, w1, excluded)`` entries: ``excluded`` names
    the rescued beats carved out of a regular window's write range.
    """
    # Prompt assignment and rescue membership come from the phase-0 plan and are
    # reused by window ordinal: a phase moves a seam, it must not flip a window
    # to another prompt or make a rescue window appear on some steps only.
    anchors = window_starts(total, window, overlap, 0, max_phase)
    assignments = [_beat_for_window(v0, v0 + window, ranges) for v0 in anchors]
    rescued = tuple(index for index in range(len(ranges)) if index not in assignments)
    plan = [(index, v0, v0 + window, v0, v0 + window, rescued)
            for index, v0 in zip(assignments, window_starts(total, window, overlap, phase, max_phase))]
    for index in rescued:
        lo, hi = ranges[index]
        v0 = min(max((lo + hi - window) // 2, 0), total - window)
        plan.append((index, v0, v0 + window, max(v0, lo), min(v0 + window, hi), ()))
    plan.sort(key=lambda entry: (entry[1], entry[0]))
    return plan


def _subtract_intervals(start, stop, holes):
    """Half-open ``[start, stop)`` minus every ``(a, b)`` in ``holes``, in order."""
    pieces = []
    cursor = start
    for a, b in sorted(holes):
        if b <= cursor or a >= stop:
            continue
        if a > cursor:
            pieces.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < stop:
        pieces.append((cursor, stop))
    return pieces


def _segment_windows(total, window, overlap, ranges):
    """Give every beat forwards and exclusive output ownership on the native grid.

    Windows see neighboring video for continuity, but a beat only contributes
    predictions inside its own interval. There is no cross-prompt interpolation,
    which also means the prompt change is a hard cut at the same latent on every
    step; ``_blended_segment_windows`` is the soft alternative.
    """
    plan = []
    for index, (lo, hi) in enumerate(ranges):
        start, stop = max(0, lo - overlap), min(total, hi + overlap)
        if stop - start < window:
            start = max(0, min((lo + hi - window) // 2, total - window))
            stop = start + window
        for offset in window_starts(stop - start, window, overlap):
            v0, v1 = start + offset, start + offset + window
            if max(v0, lo) < min(v1, hi):
                plan.append((index, v0, v1, max(v0, lo), min(v1, hi), ()))
    return plan


def apply_freenoise(noise: torch.Tensor, dim: int, starts: list[int], window: int, seed: int) -> torch.Tensor:
    """FreeNoise shuffle in place on the actual window plan.

    For each adjacent pair of windows, the frames the later window adds beyond
    the earlier one are a seeded permutation of the frames the earlier window
    holds ahead of the later one (FreeNoise, as carried by AnimateDiff-Evolved,
    WanVideoWrapper and ComfyUI core, which all walk a nominal stride). Walking
    the real starts instead keeps the shared noise pool on the seams the
    denoiser uses when the clip is not a multiple of the stride. Windows that
    start from correlated noise agree more in their overlaps, and a seam is
    exactly where two windows disagree.
    """
    length = int(noise.shape[dim])
    if window < 1 or any(b <= a for a, b in pairwise(starts)) or starts[0] < 0 or starts[-1] + window > length:
        raise ValueError(f"freenoise needs increasing starts inside [0, {length - window}], got {starts}")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for start, following in pairwise(starts):
        place = start + window
        count = min(following - start, length - place)
        if count <= 0:
            continue
        order = torch.randperm(count, generator=generator) + start
        source = noise.index_select(dim, order.to(noise.device))
        noise.narrow(dim, place, count).copy_(source)
    return noise


def _split_packed_latent(packed: torch.Tensor, shapes) -> list[torch.Tensor]:
    """Views of a ``[B, 1, N]`` sampler pack as the per-modality tensors it flattens."""
    parts, offset = [], 0
    for shape in shapes:
        count = math.prod(int(n) for n in shape[1:])
        parts.append(packed[:, :, offset:offset + count].reshape(int(packed.shape[0]), *(int(n) for n in shape[1:])))
        offset += count
    return parts


def make_freenoise_wrapper(policy: ContextPolicy):
    """SAMPLER_SAMPLE wrapper: FreeNoise-shuffle the video noise on the window grid.

    Only the video stream is shuffled. Audio is generated with its complete
    timeline visible to every window, so it gains nothing from a shared noise
    pool, and periodic audio noise is an invitation to periodic audio.
    """
    def wrapper(executor, guider, sigmas, extra_args, callback, noise, *args, **kwargs):
        shapes = getattr(getattr(guider, "inner_model", None), "latent_shapes", None)
        if policy.freenoise and shapes is not None and len(shapes) >= 2 and noise.ndim == 3:
            video_t = int(shapes[0][2])
            window = min(policy.window_frames, video_t)
            if video_t > window:
                overlap = min(policy.overlap_frames, window - 1)
                max_phase = max_stagger_phase(window, overlap) if policy.stagger else 0
                # The phase-0 anchor plan the context wrapper derives every phase from.
                starts = window_starts(video_t, window, overlap, 0, max_phase)
                seed = int(extra_args.get("seed", 0)) if isinstance(extra_args, dict) else 0
                parts = _split_packed_latent(noise, shapes)
                parts[0] = apply_freenoise(parts[0].clone(), 2, starts, window, seed)
                noise = torch.cat([part.reshape(part.shape[0], 1, -1) for part in parts], dim=-1)
                print(f"{LOG} freenoise video_latents={video_t} window={window} starts={starts} seed={seed}",
                      flush=True)
        return executor(guider, sigmas, extra_args, callback, noise, *args, **kwargs)

    return wrapper


def make_context_wrapper(policy: ContextPolicy):
    """Window video while every forward sees the complete shared audio timeline."""
    def wrapper(executor, x, timestep, context, transformer_options, **kwargs):
        require_eager_allocations(executor.class_obj)
        video_x, audio_x = x
        total_t, audio_t = int(video_x.shape[2]), int(audio_x.shape[-1])
        payload = kwargs.get("minimax_payload") or {}
        prompts = payload.get("h3forge_prompt_segments") or (context,)
        tags = payload.get("h3forge_prompt_segment_tags")
        segmented = len(prompts) > 1
        step, _ = resolve_step(transformer_options)
        window = min(total_t, policy.window_frames)
        overlap = min(policy.overlap_frames, window - 1) if total_t > window else 0
        if total_t <= window and not segmented:
            if step == 0:
                print(f"{LOG} context plan " + context_plan_summary(total_t, [0], window, 0,
                      phase=0, blend=policy.blend, audio_t=audio_t), flush=True)
            return executor(x, timestep, context, transformer_options, **kwargs)

        model = executor.class_obj
        padded_h, padded_w = padded_spatial_shape(video_x.shape[3], video_x.shape[4], model.patch_size)
        full_layout = payload.get("layout")
        if full_layout is None:
            from comfy.ldm.minimax.model import PackedLayout
            full_layout = PackedLayout(max(c.shape[1] for c in prompts), total_t, padded_h, padded_w,
                                       audio_t, keyframes=payload.get("keyframes"), refs=payload.get("refs"))
        frame_cuts = None
        ranges = []
        audio_cuts = [0, audio_t]
        phase = max_phase = 0
        blended = policy.segment_seams == "blend"
        stagger = policy.stagger and (not segmented or blended)
        if stagger:
            max_phase = max_stagger_phase(window, overlap)
            phase = stagger_phase(step, window, overlap) if step is not None else 0
        if segmented:
            ranges, frame_cuts = segment_ranges(total_t, len(prompts), payload.get("h3forge_prompt_segment_durations"))
            # H3 audio has 40 ticks per 24 decoded frames. Shared cuts give each
            # owned audio interval exactly one prompt, including a seam-crossing line.
            audio_cuts = [round(frame * 5 / 3) for frame in frame_cuts]
            audio_cuts[-1] = audio_t
            if blended:
                plan = _blended_segment_windows(total_t, window, overlap, ranges, phase, max_phase)
            else:
                plan = _segment_windows(total_t, window, overlap, ranges)
        else:
            starts = window_starts(total_t, window, overlap, phase, max_phase)
            plan = [(0, v0, v0 + window, v0, v0 + window, ()) for v0 in starts]
        # A window whose write interval is narrower than its extent owns that
        # interval: its audio is clipped to the beat's ticks as well. Windows that
        # write their whole extent blend audio over the window's own ticks, minus
        # any rescued beat, which its own window owns in both streams.
        plan = [(index, v0, v1, w0, w1, (w0, w1) != (v0, v1) or not blended, excluded)
                for index, v0, v1, w0, w1, excluded in plan]
        if step == 0:
            print(f"{LOG} context plan " + context_plan_summary(total_t, [p[1] for p in plan], window, overlap,
                  phase=phase, blend=policy.blend, stagger=stagger, max_phase=max_phase if stagger else None,
                  assignments=[p[0] for p in plan] if segmented else None, frame_cuts=frame_cuts, audio_t=audio_t,
                  seams=policy.segment_seams if segmented else None),
                  flush=True)

        video_acc = torch.zeros_like(video_x, dtype=torch.float32)
        audio_acc = torch.zeros_like(audio_x, dtype=torch.float32)
        video_den = torch.zeros((1, 1, total_t, 1, 1), device=video_x.device, dtype=torch.float32)
        audio_den = torch.zeros((1, 1, 1, audio_t), device=audio_x.device, dtype=torch.float32)
        for index, v0, v1, write_v0, write_v1, owned, excluded in plan:
            local_context = prompts[index]
            local_layout = clone_window_layout(full_layout=full_layout, text_len=local_context.shape[1],
                video_shape=(v1 - v0, padded_h, padded_w), audio_t=audio_t,
                video_range=(v0, v1), audio_range=(0, audio_t),
                keyframes=payload.get("keyframes"), refs=payload.get("refs"))
            local_payload = {**payload, "layout": local_layout}
            if tags is not None:
                local_payload["text_token_tags"] = tags[index]
            local_kwargs = {**kwargs, "minimax_payload": local_payload,
                            "denoise_mask": _slice_optional_video(kwargs.get("denoise_mask"), v0, v1)}
            # The global audio input and denoise mask travel intact. Only the
            # output projection is local, so all windows can see prior utterances.
            local_x = [video_x[:, :, v0:v1], audio_x]
            sentinel = object()
            previous_layout = transformer_options.get("h3forge_active_layout", sentinel)
            transformer_options["h3forge_active_layout"] = local_layout
            try:
                with control_window(executor, video_x.shape, (v0, v1), timestep, transformer_options):
                    v_out, a_out = executor(local_x, timestep, local_context, transformer_options, **local_kwargs)
            except Exception as exc:
                raise RuntimeError(f"{LOG} window [{v0},{v1}) prompt {index + 1} failed: {exc}") from exc
            finally:
                if previous_layout is sentinel:
                    transformer_options.pop("h3forge_active_layout", None)
                else:
                    transformer_options["h3forge_active_layout"] = previous_layout

            vw = blend_weights(v1 - v0, overlap, device=v_out.device, dtype=torch.float32, mode=policy.blend,
                               ramp_start=v0 > 0, ramp_end=v1 < total_t).view(1, 1, -1, 1, 1)
            for w0, w1 in _subtract_intervals(write_v0, write_v1, [ranges[b] for b in excluded]):
                keep = slice(w0 - v0, w1 - v0)
                video_acc[:, :, w0:w1].add_(v_out[:, :, keep].float() * vw[:, :, keep])
                video_den[:, :, w0:w1].add_(vw[:, :, keep])

            a0, a1 = audio_range_for_video_window(full_layout, v0, v1)
            write_a0, write_a1 = a0, a1
            if owned:
                write_a0, write_a1 = max(a0, audio_cuts[index]), min(a1, audio_cuts[index + 1])
            aw = blend_weights(a1 - a0, audio_overlap_frames(overlap, v1 - v0, a1 - a0),
                               device=a_out.device, dtype=torch.float32, mode=policy.blend,
                               ramp_start=a0 > 0, ramp_end=a1 < audio_t).view(1, 1, 1, -1)
            holes = [(audio_cuts[b], audio_cuts[b + 1]) for b in excluded]
            for w0, w1 in _subtract_intervals(write_a0, write_a1, holes):
                audio_acc[..., w0:w1].add_(a_out[..., w0:w1].float() * aw[..., w0 - a0:w1 - a0])
                audio_den[..., w0:w1].add_(aw[..., w0 - a0:w1 - a0])

        # Unconditional: a zero-weight latent would otherwise divide by the clamp
        # floor and decode as a silent black smear. The check is one comparison.
        assert_full_coverage(video_den, audio_den)
        return [(video_acc / video_den.clamp_min(1e-6)).to(video_x.dtype),
                (audio_acc / audio_den.clamp_min(1e-6)).to(audio_x.dtype)]

    return wrapper

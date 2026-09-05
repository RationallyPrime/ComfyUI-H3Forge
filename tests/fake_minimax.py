"""A minimal stand-in for ``comfy.ldm.minimax.model`` used by the CPU tests.

It mirrors the packing rules H3Forge depends on: prefix rows (text, then
optional reference rows, then optional keyframe rows), channel-major stereo
audio rows, frame-major video rows, H3's ``1,4,4,4,4`` latent-frame cadence on
the shared 40 Hz timeline (scaled by 5/3), and audio's one-step grid.

Text and prefix origins follow ComfyUI 12d5279438bf's PackedLayout. Reference
and keyframe descriptors use the real fields, so prefix-coordinate assertions
exercise nonzero positions instead of an all-zero placeholder. Spatial rows
remain synthetic; these tests concern packing and time coordinates.
"""
from __future__ import annotations

import sys
import types

import torch

_CADENCE = (1.0, 4.0, 4.0, 4.0, 4.0)
_TIME_SCALE = 5.0 / 3.0


def _video_t_spans(latent_t):
    return [_CADENCE[i % 5] * _TIME_SCALE for i in range(int(latent_t))]


def video_frame_times(latent_t):
    times = [0.0]
    for span in _video_t_spans(latent_t)[:-1]:
        times.append(times[-1] + span)
    return times[: int(latent_t)]


class PackedLayout:
    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None):
        text_len, latent_t = int(text_len), int(latent_t)
        latent_h, latent_w, audio_t = int(latent_h), int(latent_w), int(audio_t)
        self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
        frame_rows = max(latent_h // 2, 1) * max(latent_w // 2, 1)
        self.frame_rows = frame_rows

        segments = []
        cursor = 0

        positions = []

        def push(times, kind):
            nonlocal cursor
            times = torch.as_tensor(times, dtype=torch.float64)
            count = times.numel()
            if count > 0:
                segments.append((cursor, cursor + count, kind))
                cursor += count
                pos = torch.zeros(count, 3, dtype=torch.float64)
                pos[:, 0] = times
                positions.append(pos)

        def video_times(t, rows, origin):
            return (torch.tensor(video_frame_times(t), dtype=torch.float64) + origin).repeat_interleave(rows)

        def ref_span(blk):
            if blk["kind"] == "image":
                return 1.0
            if blk["kind"] == "audio":
                return float(blk["ref_audio_t"])
            return max(float(blk["ref_audio_t"]), sum(_video_t_spans(blk["latent_t"])))

        push(torch.arange(text_len), "text")
        origin = text_len + sum(ref_span(blk) for blk in refs or ())
        for kf in keyframes or ():
            anchor = origin + _TIME_SCALE * kf["resolved_frame_index"]
            if kf.get("latent") is not None:
                push(video_times(kf["latent"].shape[2], frame_rows, anchor), "cond")
            if kf.get("audio_latent") is not None:
                push((torch.arange(kf["audio_latent"].shape[-1]) + anchor).repeat(2), "cond_audio")
        ref_origin = float(text_len)
        for blk in refs or ():
            kind = blk["kind"]
            if kind == "image":
                rows = (blk["latent_h"] // 2) * (blk["latent_w"] // 2)
                push(torch.full((rows,), ref_origin), "ref_img")
            else:
                push((torch.arange(blk["ref_audio_t"]) + ref_origin).repeat(2), "ref_audio")
                if kind in ("video", "video_audio"):
                    rows = (blk["latent_h"] // 2) * (blk["latent_w"] // 2)
                    push(video_times(blk["latent_t"], rows, ref_origin), "ref_img")
            ref_origin += ref_span(blk)
        push((torch.arange(audio_t) + origin).repeat(2), "audio")
        push(video_times(latent_t, frame_rows, origin), "video")
        self.segments = segments
        self.seq_len = cursor

        self.position_ids = torch.cat(positions)


def install(monkeypatch):
    """Register the fake module tree in sys.modules for one test."""
    model_module = types.ModuleType("comfy.ldm.minimax.model")
    model_module.PackedLayout = PackedLayout
    model_module._video_t_spans = _video_t_spans

    minimax_module = types.ModuleType("comfy.ldm.minimax")
    minimax_module.model = model_module
    ldm_module = types.ModuleType("comfy.ldm")
    ldm_module.minimax = minimax_module
    comfy_module = types.ModuleType("comfy")
    comfy_module.ldm = ldm_module

    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.ldm", ldm_module)
    monkeypatch.setitem(sys.modules, "comfy.ldm.minimax", minimax_module)
    monkeypatch.setitem(sys.modules, "comfy.ldm.minimax.model", model_module)
    return model_module

"""A minimal stand-in for ``comfy.ldm.minimax.model`` used by the CPU tests.

It mirrors the packing rules H3Forge depends on: prefix rows (text, then
optional reference rows, then optional keyframe rows), channel-major stereo
audio rows, frame-major video rows, H3's ``1,4,4,4,4`` latent-frame cadence on
the shared 40 Hz timeline (scaled by 5/3), and audio's one-step grid.

Keyframes are modelled as a keyframe-latent count and references as a raw row
count; the real model passes richer descriptors, but H3Forge treats both as
opaque prefix structure.
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

        def push(count, kind):
            nonlocal cursor
            if count > 0:
                segments.append((cursor, cursor + count, kind))
                cursor += count

        push(text_len, "text")
        push(int(refs or 0), "ref")
        push(int(keyframes or 0) * frame_rows, "keyframe")
        audio_start = cursor
        push(2 * audio_t, "audio")
        video_start = cursor
        push(latent_t * frame_rows, "video")
        self.segments = segments
        self.seq_len = cursor

        position_ids = torch.zeros(self.seq_len, 3)
        for frame, time in enumerate(video_frame_times(latent_t)):
            row0 = video_start + frame * frame_rows
            position_ids[row0:row0 + frame_rows, 0] = time
        for channel in range(2):
            row0 = audio_start + channel * audio_t
            position_ids[row0:row0 + audio_t, 0] = torch.arange(audio_t, dtype=torch.float32)
        self.position_ids = position_ids


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

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from h3forge import renoise
from h3forge.timeline import video_latent_t


def test_latent_spans_tile_the_clip_on_the_encoder_cycle():
    for frames in (5, 22, 124, 311):
        spans = renoise.latent_frame_spans(frames)
        assert len(spans) == video_latent_t(frames)
        assert spans[0] == (0, 1) and spans[-1][1] == frames
        assert all(a == prev_b for (a, _), (_, prev_b) in zip(spans[1:], spans))
        # every cycle opens on a one-frame latent
        assert all(spans[k] == (17 * (k // 5), 17 * (k // 5) + 1) for k in range(0, len(spans), 5))
    with pytest.raises(ValueError):
        renoise.latent_frame_spans(100)


def test_temporal_weights_ramp_smoothly_and_are_full_inside():
    w = renoise.temporal_weights(60, 20, 30, 5)
    assert torch.all(w[20:31] == 1)
    assert torch.all(w[:15] == 0) and torch.all(w[36:] == 0)
    assert torch.all(torch.diff(w[14:21]) > 0) and torch.all(torch.diff(w[30:37]) < 0)
    assert torch.allclose(w[15:20], w[31:36].flip(0))
    assert torch.equal(renoise.temporal_weights(10, 3, 4, 0), torch.tensor([0, 0, 0, 1, 1, 0, 0, 0, 0, 0.]))


def test_window_is_cycle_aligned_on_grid_and_covers_ramp_and_context():
    w = renoise.plan_window(311, 150, 180, 12, 17, crop=True)
    assert w.start % 17 == 0 and w.length % 17 == 5
    assert w.start <= 150 - 12 - 17 and w.start + w.valid - 1 >= 180 + 12 + 17
    assert w.valid == w.length and w.length < 311
    # near the end the window slides back onto real frames
    tail = renoise.plan_window(311, 290, -1, 8, 17, crop=True)
    assert tail.start + tail.length <= 311 and tail.valid == tail.length
    # no crop: the whole clip, padded to the grid when it is off it
    whole = renoise.plan_window(300, 10, 20, 4, 4, crop=False)
    assert (whole.start, whole.length, whole.valid) == (0, 311, 300)
    with pytest.raises(ValueError):
        renoise.plan_window(311, 200, 100, 4, 4, crop=True)


def test_masks_track_the_region_and_pin_everything_else():
    window = renoise.plan_window(124, 40, 60, 6, 17, crop=True)
    mask = torch.zeros(1, 64, 96)
    mask[:, 16:48, 48:80] = 1
    spatial = renoise.spatial_masks(mask, renoise.window_indices(window), 124, window.interval, 64, 96, 0, 0)
    blend = renoise.blend_masks(spatial, window)
    assert blend.shape == (window.length, 64, 96)
    inside = 50 - window.start
    assert blend[inside, 20, 60] == 1 and blend[inside, 5, 5] == 0
    assert blend[:40 - 6 - window.start].abs().sum() == 0  # before the ramp: untouched
    latent = renoise.video_noise_mask(blend, window, 0.6)
    assert latent.shape == (1, 1, video_latent_t(window.length), 4, 6)
    assert torch.isclose(latent.max(), torch.tensor(0.6))
    assert latent[0, 0, :, 0, 0].abs().sum() == 0  # outside the region in every latent
    spans = renoise.latent_frame_spans(window.length)
    for k, (a, b) in enumerate(spans):
        assert (latent[0, 0, k].amax() > 0) == bool(blend[a:b].amax() > 0)


def test_per_interval_masks_reuse_the_nearest_end_in_the_ramp():
    window = renoise.plan_window(124, 40, 41, 3, 0, crop=False)
    masks = torch.zeros(2, 8, 8)
    masks[0, 0, 0] = 1
    masks[1, 7, 7] = 1
    spatial = renoise.spatial_masks(masks, renoise.window_indices(window), 124, window.interval, 8, 8, 0, 0)
    assert spatial[38, 0, 0] == 1 and spatial[43, 7, 7] == 1
    with pytest.raises(ValueError):
        renoise.spatial_masks(torch.zeros(5, 8, 8), renoise.window_indices(window), 124, window.interval,
                              8, 8, 0, 0)


def test_grow_and_feather_soften_the_edge():
    mask = torch.zeros(1, 32, 32)
    mask[:, 12:20, 12:20] = 1
    frames = torch.tensor([0])
    grown = renoise.spatial_masks(mask, frames, 1, (0, 0), 32, 32, 4, 0)
    assert grown[0, 8, 8] == 1 and grown[0, 7, 7] == 0
    soft = renoise.spatial_masks(mask, frames, 1, (0, 0), 32, 32, 0, 4)
    edge = soft[0, 16, 8:16]
    assert torch.all(torch.diff(edge) >= 0) and 0 < edge[2] < 1


def test_audio_mask_is_kept_unless_renoised():
    window = renoise.plan_window(124, 40, 60, 6, 17, crop=False)
    assert renoise.audio_noise_mask(207, window, 1.0, False).abs().sum() == 0
    mask = renoise.audio_noise_mask(207, window, 0.5, True)
    assert mask.shape == (1, 1, 2, 207)
    assert mask[0, 0, 0, 80] == 0.5 and mask[0, 0, 0, 0] == 0  # frame 48 is inside, frame 0 is not


def test_stitch_leaves_every_pixel_outside_the_blend_identical():
    torch.manual_seed(0)
    source = torch.rand(124, 32, 48, 3)
    window = renoise.plan_window(124, 40, 60, 6, 17, crop=True)
    mask = torch.zeros(1, 32, 48)
    mask[:, 8:24, 8:24] = 1
    spatial = renoise.spatial_masks(mask, renoise.window_indices(window), 124, window.interval, 32, 48, 2, 2)
    blend = renoise.blend_masks(spatial, window)
    out = renoise.stitch(source, window, blend, torch.zeros(window.length, 32, 48, 3))
    full = torch.zeros(124, 32, 48)
    full[window.start:window.start + window.valid] = blend[:window.valid]
    untouched = (full == 0).unsqueeze(-1).expand_as(source)
    assert torch.equal(out[untouched], source[untouched])
    assert torch.all(out[50, 16, 16] == 0)


def test_splice_pastes_through_the_mask():
    source = torch.ones(10, 4, 4, 3)
    region = torch.zeros(1, 4, 4)
    region[:, :2] = 1
    out = renoise.splice(source, torch.zeros(1, 4, 4, 3), 5, region)
    assert torch.all(out[5, :2] == 0) and torch.all(out[5, 2:] == 1) and torch.all(out[4] == 1)


def test_audio_slice_and_fit_cover_the_window():
    window = renoise.Window(17, 22, 22, (20, 30), 0)
    audio = {"waveform": torch.arange(48000.).view(1, 1, -1), "sample_rate": 24000}
    part = renoise.audio_slice(audio, window)
    assert part["waveform"][0, 0, 0] == 17000 and part["waveform"].shape[-1] == 22000
    z = torch.rand(1, 32, 2, 30)
    assert renoise.fit_audio_latent(z, 37).shape[-1] == 37
    assert torch.equal(renoise.fit_audio_latent(z, 20), z[..., :20])


def _import_nodes(monkeypatch):
    comfy = types.ModuleType("comfy")
    patcher_extension = types.ModuleType("comfy.patcher_extension")
    patcher_extension.WrappersMP = SimpleNamespace(OUTER_SAMPLE="o", DIFFUSION_MODEL="d", SAMPLER_SAMPLE="s")
    nested = types.ModuleType("comfy.nested_tensor")

    class NestedTensor:
        def __init__(self, tensors):
            self.tensors = list(tensors)
            self.is_nested = True

    nested.NestedTensor = NestedTensor
    comfy.patcher_extension, comfy.nested_tensor = patcher_extension, nested
    extras = types.ModuleType("comfy_extras")
    h3 = types.ModuleType("comfy_extras.nodes_minimax_h3")

    def temporal_shape(length):
        frames = renoise.align_frame_count(length)
        return frames, video_latent_t(frames), round(frames / 24 * 40)

    h3.temporal_shape = temporal_shape
    h3._encode_ref_audio = lambda vae, audio: (torch.zeros(1, 32, 2, 10), 10)
    for name, module in (("comfy", comfy), ("comfy.patcher_extension", patcher_extension),
                         ("comfy.nested_tensor", nested), ("comfy_extras", extras),
                         ("comfy_extras.nodes_minimax_h3", h3)):
        monkeypatch.setitem(sys.modules, name, module)
    return importlib.import_module("h3forge.nodes")


class FakeVAE:
    def encode(self, frames):
        n = frames.shape[0]
        return torch.zeros(1, 24, video_latent_t(n), frames.shape[1] // 16, frames.shape[2] // 16)


def test_window_node_builds_a_masked_av_latent_and_the_stitch_restores_the_clip(monkeypatch, capsys):
    nodes = _import_nodes(monkeypatch)
    images = torch.rand(311, 64, 96, 3)
    mask = torch.zeros(1, 64, 96)
    mask[:, :32, :48] = 1
    latent, plan, width, height, length, guide = nodes.H3ForgeRenoiseWindow().plan(
        images, FakeVAE(), 150, 180, 1.0, 12, 17, True, 0, 0, mask=mask,
        audio={"waveform": torch.zeros(1, 2, 24000 * 13), "sample_rate": 24000}, audio_vae=object(),
        inpainted_frames=torch.zeros(1, 64, 96, 3), inpainted_frame_idx=150)
    window = plan["window"]
    video, audio = latent["samples"].tensors
    video_mask, audio_mask = latent["noise_mask"].tensors
    assert (width, height, length) == (96, 64, window.length)
    assert guide == 150 - window.start
    assert video.shape[2] == video_mask.shape[2] and video_mask.shape[-2:] == (4, 6)
    assert audio.shape[-1] == audio_mask.shape[-1] == round(length / 24 * 40)
    assert audio_mask.abs().sum() == 0
    assert torch.all(plan["source"][150, :32, :48] == 0) and torch.equal(plan["source"][150, 40:], images[150, 40:])
    assert "renoise frames 150-180" in capsys.readouterr().out

    out, blend = nodes.H3ForgeRenoiseStitch().stitch(plan, torch.ones(length, 64, 96, 3))
    assert blend.shape == (311, 64, 96)
    untouched = (blend == 0).unsqueeze(-1).expand_as(images)
    assert torch.equal(out[untouched], images[untouched])
    assert torch.all(out[160, :32, :48] == 1)


def test_example_graph_wires_every_renoise_input(monkeypatch):
    import json
    from pathlib import Path

    nodes = _import_nodes(monkeypatch)
    graph = json.loads((Path(__file__).parent.parent / "docs/renoise-workflows/masked_interval_renoise.api.json")
                       .read_text())
    for node in graph.values():
        cls = nodes.NODE_CLASS_MAPPINGS.get(node["class_type"])
        if cls is None:
            continue
        spec = cls.INPUT_TYPES()
        assert set(spec["required"]) <= set(node["inputs"])
        assert set(node["inputs"]) <= set(spec["required"]) | set(spec.get("optional", {}))

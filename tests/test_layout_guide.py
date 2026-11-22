from types import SimpleNamespace

import pytest
import torch

import fake_minimax
from core_source import definitions, native_h3
from h3forge.beat_refs import BEAT_REFS_KEY, parse_beat_refs
from h3forge.context import ContextPolicy, make_context_wrapper
from h3forge.layout import audio_window_length, target_segments, video_frame_positions
from h3forge.layout_guide import interpolate_sparse_video, sparse_video_indices, spatial_lowpass
from h3forge.prompt import combine_conditioning_segments, encode_pipe_prompt, make_segmented_extra_conds
from test_audit_repairs import _executor
from test_beat_refs import _image_ref, _mod


@pytest.mark.parametrize("total,window", [(217, 90), (217, 85), (1200, 85), (1201, 2), (17, 25)])
def test_sparse_anchors_span_the_take_within_one_window(total, window):
    points = sparse_video_indices(total, window)
    assert len(points) == min(total, window)
    assert points == sorted(set(points))
    assert points[0] == 0 and points[-1] == total - 1
    if total == 217:
        assert set(range(0, total, 5)) <= set(points)


def test_time_interpolation_uses_physical_cadence_and_preserves_anchors():
    times = torch.tensor(fake_minimax.video_frame_times(7), dtype=torch.float64)
    indices = [0, 5, 6]
    sparse = torch.tensor([0., 17., -4.]).reshape(1, 1, -1, 1, 1)
    v = interpolate_sparse_video(sparse, times[indices], times).flatten()
    torch.testing.assert_close(v, torch.tensor([0., 1., 5., 9., 13., 17., -4.]))
    assert torch.equal(v[indices], sparse.flatten())


def test_gaussian_details_are_independent_for_each_channel_and_time():
    torch.manual_seed(17)
    local = torch.randn(1, 3, 4, 9, 7)
    shared = torch.randn_like(local)
    low = spatial_lowpass(local)
    other = spatial_lowpass(shared)
    mixed = local + .5 * (other - low)
    torch.testing.assert_close(mixed - torch.lerp(low, other, .5), local - low)
    assert torch.equal(low[:, :, 2:3], spatial_lowpass(local[:, :, 2:3]))
    torch.testing.assert_close(spatial_lowpass(torch.full_like(local, 3.)), torch.full_like(local, 3.))


class _Clip:
    def __init__(self):
        self.seen = []

    def tokenize(self, text):
        self.seen.append(text)
        return text

    def encode_from_tokens_scheduled(self, text):
        marker = {"Hall": 9, "Hall\n\nTurn.": 1, "Hall\n\nStep.": 2, "Hall\n\nStop here.": 3}[text]
        n = len(text.split())
        return [[torch.full((1, n, 2), float(marker)), {"minimax_token_tags": torch.ones(n, dtype=torch.long)}]]


class _Cond:
    def __init__(self, cond):
        self.cond = cond

    def _copy_with(self, cond):
        return _Cond(cond)


def _pipe_payload(prepare, strength, extra=None):
    clip = _Clip()
    ctx, meta = encode_pipe_prompt(clip, "Turn. | Step. | Stop here.", global_prompt="Hall",
                                   prepare_layout_guide=prepare)[0]
    expected = ["Hall\n\nTurn.", "Hall\n\nStep.", "Hall\n\nStop here."]
    assert clip.seen == expected + (["Hall"] if prepare else [])
    assert len(meta["h3forge_prompt_segments"]) == 3
    calls = []

    def preprocess(raw):
        calls.append(raw)
        return raw + 10

    build = make_segmented_extra_conds(
        lambda **_: {"minimax_payload": _Cond(extra or {}), "c_crossattn": _Cond(ctx + 10)},
        SimpleNamespace(get_dtype_inference=lambda: torch.float32),
        SimpleNamespace(preprocess_text_embeds=preprocess), layout_guide=bool(strength))
    result = build(device="cpu", **meta)
    assert len(calls) == 2 + int(prepare and strength > 0)
    return result["c_crossattn"].cond, result["minimax_payload"].cond


def test_missing_setting_fails_before_encoding_and_single_beat_uses_its_own_text():
    clip = _Clip()
    with pytest.raises(ValueError, match="shared setting in global_prompt"):
        encode_pipe_prompt(clip, "Turn. | Step.", prepare_layout_guide=True)
    assert clip.seen == []
    _, meta = encode_pipe_prompt(clip, "Turn.", global_prompt="Hall", prepare_layout_guide=True)[0]
    assert clip.seen == ["Hall\n\nTurn."] and "h3forge_layout_conditioning" not in meta


@pytest.mark.parametrize("frozen", [False, True])
def test_cockpit_217_latent_take_guides_owner_before_lead_and_uses_shared_refs(monkeypatch, capsys, frozen):
    fake_minimax.install(monkeypatch)
    video = torch.arange(217.).reshape(1, 1, -1, 1, 1).expand(-1, -1, -1, 2, 2).clone()
    audio = torch.arange(2454.).reshape(1, 1, 2, 1227)
    vmask, amask = torch.ones_like(video), torch.ones_like(audio)
    if frozen:
        vmask[:, :, :85] = 0
        amask[..., :481] = 0
    refs = [_image_ref(20, refmod=False), *(_image_ref(i) for i in (21, 22, 23))]
    source_layout = fake_minimax.PackedLayout(3, 217, 2, 2, 1227, refs=refs)
    text, payload = _pipe_payload(True, .5, {"layout": source_layout, "refs": refs,
                                             "cond_video_latents": [ref["latent"] for ref in refs]})
    spec = parse_beat_refs("shared, left | shared, right | *", [_mod(n) for n in ("shared", "left", "right")])
    opts = {"sigmas": torch.tensor([1.]), "sample_sigmas": torch.tensor([1., .9]), BEAT_REFS_KEY: spec}
    calls = []
    starts = (0, 65, 127)
    expected = torch.empty_like(video)
    detail = torch.tensor([[-1., 1.], [1., -1.]]).view(1, 1, 1, 2, 2)

    def predict(x, timestep, context, options, minimax_payload, denoise_mask, audio_denoise_mask):
        layout = minimax_payload["layout"]
        assert options["h3forge_active_layout"] is layout
        code = int(context[0, 0, 0]) - 10
        calls.append(code)
        assert layout.signature == (context.shape[1], x[0].shape[2], 2, 2, x[1].shape[-1])
        assert len(minimax_payload["text_token_tags"]) == context.shape[1]
        kept = [int(ref["latent"].flatten()[0]) for ref in minimax_payload["refs"]]
        assert kept == {9: [20, 21], 1: [20, 21, 22], 2: [20, 21, 23], 3: [20, 21, 22, 23]}[code]
        assert all(lat is ref["latent"] for lat, ref in zip(minimax_payload["cond_video_latents"], minimax_payload["refs"]))
        if code == 9:
            assert context.shape[1] == 1
            vid = list(layout._h3forge_video_indices)
            assert vid == sparse_video_indices(217, 90)
            assert torch.equal(video_frame_positions(layout), video_frame_positions(source_layout)[vid])
            assert torch.equal(x[0], video[:, :, vid]) and torch.equal(denoise_mask, vmask[:, :, vid])
            seg, src = target_segments(layout), target_segments(source_layout)
            aud = x[1][0, 0, 0].long()
            assert (aud[0], aud[-1], len(aud), len(aud.unique())) == (0, 1226, 511, 511)
            assert torch.equal(x[1], audio[..., aud]) and torch.equal(audio_denoise_mask, amask[..., aud])
            sup = source_layout.position_ids[src.audio_start:src.audio_stop].reshape(2, 1227, 3)
            assert torch.equal(layout.position_ids[seg.audio_start:seg.audio_stop], sup[:, aud].flatten(0, 1))
            return torch.full_like(x[0], 9), torch.full_like(x[1], torch.nan)

        start = starts[code - 1]
        overlap = starts[code - 2] + 90 - start if code > 1 else 0
        baseline = video[:, :, start:start + 90]
        if overlap:
            owned = expected[:, :, start:start + overlap]
            original_mask = vmask[:, :, start:start + overlap]
            torch.testing.assert_close(x[0][:, :, :overlap], baseline[:, :, :overlap] - .5 * original_mask * owned)
            assert torch.equal(denoise_mask[:, :, :overlap], original_mask * .5)
        assert torch.equal(x[0][:, :, overlap:], baseline[:, :, overlap:])
        row_mask = vmask[:, :, start + overlap:start + 90]
        native = code + detail
        expected[:, :, start + overlap:start + 90] = torch.where(row_mask == 0, native,
                                                                 native + .5 * (9 - spatial_lowpass(native)))
        return native.expand_as(x[0]), x[1]

    guided_video, guided_audio = make_context_wrapper(ContextPolicy(90, 21, False, handoff="lead", layout_guide=.5))(
        _executor(predict), [video, audio], torch.tensor([1000.]), text, opts, minimax_payload=payload,
        denoise_mask=vmask, audio_denoise_mask=amask)
    assert calls == [9, 1, 2, 3]
    torch.testing.assert_close(guided_video, expected)
    torch.testing.assert_close(guided_audio, audio)
    assert "h3forge_active_layout" not in opts
    log = capsys.readouterr().out
    assert "active=True sparse_video=90/217 sparse_audio=511/1227" in log
    assert float(log.split("delta_rms=")[1]) > 0


def test_default_off_exact_and_missing_shared_setting_is_actionable(monkeypatch):
    fake_minimax.install(monkeypatch)
    x = [torch.ones(1, 1, 17, 2, 2), torch.ones(1, 1, 2, 94)]
    opts = {"sigmas": torch.tensor([1.]), "sample_sigmas": torch.tensor([1., 0.])}

    def run(prepare, policy):
        text, payload = _pipe_payload(prepare, policy.layout_guide)
        calls = []

        def predict(x, t, context, *args, **kwargs):
            calls.append(int(context[0, 0, 0]))
            return [v + calls[-1] for v in x]

        out = make_context_wrapper(policy)(_executor(predict), x, torch.tensor([1000.]), text, opts,
                                           minimax_payload=payload)
        return calls, out

    control_calls, control = run(False, ContextPolicy(8, 2, False))
    for prepare, policy in ((True, ContextPolicy(8, 2, False, layout_guide=0)),
                            (True, ContextPolicy(8, 2, False, layout_guide=.5, layout_guide_until=1))):
        calls, out = run(prepare, policy)
        assert calls == control_calls == [11, 12, 13]
        assert all(torch.equal(a, b) for a, b in zip(control, out))
    with pytest.raises(ValueError, match="requires prepare_layout_guide"):
        run(False, ContextPolicy(8, 2, False, layout_guide=.5))


@pytest.mark.parametrize("prebuilt", [True, False])
def test_long_explicit_setting_uses_only_the_wide_text_prefix(monkeypatch, prebuilt):
    fake_minimax.install(monkeypatch)
    refs = [_image_ref(1, refmod=False)]
    keyframes = [{"resolved_frame_index": 2, "latent": torch.ones(1, 1, 1, 2, 2),
                  "audio_latent": torch.ones(1, 1, 2, 1)}]
    meta = {"minimax_refs": refs, "minimax_keyframes": keyframes}
    beats = [[[torch.ones(1, n, 2), meta]] for n in (2, 3)]
    setting = [[torch.ones(1, 12, 2), meta]]
    text, metadata = combine_conditioning_segments(beats, layout_conditioning=setting)[0]
    video = torch.zeros(1, 1, 12, 2, 2)
    audio = torch.arange(130.).reshape(1, 1, 2, 65)
    mask = torch.ones_like(video)
    mask[:, :, :4] = 0
    shared = {"refs": refs, "keyframes": keyframes}
    full = fake_minimax.PackedLayout(3, 12, 2, 2, 65, **shared)

    def sample(strength, sigma):
        native = dict(shared)
        if prebuilt:
            native["layout"] = fake_minimax.PackedLayout(2, 12, 2, 2, 65, **shared)
        build = make_segmented_extra_conds(
            lambda **_: {"minimax_payload": _Cond(native), "c_crossattn": _Cond(text)},
            SimpleNamespace(get_dtype_inference=lambda: torch.float32),
            SimpleNamespace(preprocess_text_embeds=lambda c: c), layout_guide=bool(strength))
        payload = build(device="cpu", **metadata)["minimax_payload"].cond
        local, wide = [], []

        def predict(x, t, ctx, opts, **kwargs):
            layout = kwargs["minimax_payload"]["layout"]
            if hasattr(layout, "_h3forge_video_indices"):
                wide.append((layout, x[1][0, 0, 0].long()))
            else:
                local.append(layout.position_ids.clone())
            v = video_frame_positions(layout).float().view(1, 1, -1, 1, 1)
            seg = target_segments(layout)
            a = layout.position_ids[seg.audio_start:seg.audio_stop, 0].reshape_as(x[1]).float()
            return x[0] + v, a

        out = make_context_wrapper(ContextPolicy(8, 3, False, handoff="lead", layout_guide=strength))(
            _executor(predict), [video, audio], torch.tensor([1000 * sigma]), text,
            {"sigmas": torch.tensor([sigma]), "sample_sigmas": torch.tensor([sigma, 0.])},
            minimax_payload=payload, denoise_mask=mask)
        return local, wide, out

    original, _, control = sample(0, .8)
    late, no_guide, tail = sample(.5, .8)
    assert len(late) == len(original) == 2 and not no_guide
    assert all(torch.equal(a, b) for a, b in zip(late, original))
    assert all(torch.equal(a, b) for a, b in zip(tail, control))
    early, wide, _ = sample(.5, 1.)
    assert all(torch.equal(a, b) for a, b in zip(early, original))
    (layout, audio_indices), = wide
    assert torch.equal(layout.position_ids[:12, 0], torch.arange(-9, 3))
    for (a, b, kind), (c, d, _) in zip(layout.segments, full.segments):
        if kind in ("cond", "cond_audio", "ref_img"):
            assert torch.equal(layout.position_ids[a:b], full.position_ids[c:d])
    src, dst = target_segments(full), target_segments(layout)
    native_audio = full.position_ids[src.audio_start:src.audio_stop].reshape(2, 65, 3)
    assert torch.equal(layout.position_ids[dst.audio_start:dst.audio_stop], native_audio[:, audio_indices].flatten(0, 1))
    assert torch.equal(video_frame_positions(layout), video_frame_positions(full)[list(layout._h3forge_video_indices)])


def test_cutoff_uses_pinned_simple_schedule_across_frozen_replays(monkeypatch):
    fake_minimax.install(monkeypatch)
    flow = definitions("comfy/model_sampling.py", ["time_snr_shift", "ModelSamplingDiscreteFlow"],
                       {"torch": torch})["ModelSamplingDiscreteFlow"]()
    flow.set_parameters(shift=12)
    schedule = definitions("comfy/samplers.py", ["simple_scheduler"], {"torch": torch})["simple_scheduler"](flow, 30)
    x = [torch.zeros(1, 1, 7, 1, 1), torch.zeros(1, 1, 2, 37)]
    text = torch.ones(1, 2, 8)
    run = make_context_wrapper(ContextPolicy(5, 2, False, layout_guide=.5))
    counts = []
    for phase in (schedule, schedule[:17], schedule[:17], schedule[16:]):
        active = []
        for sigma in phase[:-1]:
            calls = []

            def predict(x, t, c, opts, **kwargs):
                calls.append(hasattr(opts["h3forge_active_layout"], "_h3forge_video_indices"))
                return x

            run(_executor(predict), x, sigma.reshape(1) * 1000, text,
                {"sigmas": sigma.reshape(1), "sample_sigmas": phase})
            active.append(sum(calls))
        counts.append(active)
    assert counts == [[1] * 16 + [0] * 14, [1] * 16, [1] * 16, [0] * 14]


def test_sparse_budget_and_positions_against_native_packed_layout():
    core = native_h3()
    from h3forge.layout import clone_window_layout

    full = core.PackedLayout(5, 217, 4, 4, 1227)
    vid = sparse_video_indices(217, 90)
    audio_win = audio_window_length(full, 90)
    aud = torch.linspace(0, 1226, audio_win).round().long().tolist()
    guide = clone_window_layout(full_layout=full, text_len=2, video_shape=(90, 4, 4), audio_t=1227,
                                sparse_indices=(vid, aud))
    assert guide.signature == (2, 90, 4, 4, audio_win)
    assert audio_win == 511
    seg, src = target_segments(guide), target_segments(full)
    physical_video = full.position_ids[src.video_start:src.video_stop].reshape(217, 4, 3)
    physical_audio = full.position_ids[src.audio_start:src.audio_stop].reshape(2, 1227, 3)
    assert torch.equal(guide.position_ids[seg.video_start:seg.video_stop], physical_video[vid].flatten(0, 1))
    assert torch.equal(guide.position_ids[seg.audio_start:seg.audio_stop], physical_audio[:, aud].flatten(0, 1))

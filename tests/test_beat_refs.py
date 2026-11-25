"""Per-beat RefMod selection: spec parsing, payload filtering, window layouts."""
from types import SimpleNamespace

import pytest
import torch

import fake_minimax
from core_source import definitions, native_h3
from fake_minimax import PackedLayout
from h3forge.beat_refs import BEAT_REFS_KEY, BeatRefs, beat_ref_keeps, filter_payload, parse_beat_refs
from h3forge.context import ContextPolicy, make_context_wrapper
from h3forge.prompt import make_segmented_extra_conds
from test_nodes import FakePatcher, _import_nodes


def _mod(name, path="", kind="image", h=4, w=4):
    return SimpleNamespace(name=name, path=path, kind=kind, latent_h=h, latent_w=w)


def _image_ref(value, refmod=True, h=4, w=4):
    block = {"kind": "image", "latent_h": h, "latent_w": w, "latent": torch.full((1, 2, 1, h, w), float(value))}
    if refmod:
        block["refmod"] = True
    return block


def _video_ref(value, latent_t=2, audio_t=3, h=4, w=4):
    return {"kind": "video", "latent_h": h, "latent_w": w, "latent_t": latent_t, "ref_audio_t": audio_t,
            "latent": torch.full((1, 2, latent_t, h, w), float(value)),
            "audio_latent": torch.full((1, 2, 2, audio_t), float(value)) if audio_t else None, "refmod": True}


def _audio_ref(value, audio_t=5):
    return {"kind": "audio", "ref_audio_t": audio_t, "audio_latent": torch.full((1, 2, 2, audio_t), float(value)),
            "refmod": True}


LOADED = [(_mod("tairn_v2"), 1.0), (_mod("sgaeyl_v2"), 1.0),
          (_mod("plate 1", "/mods/fw_moon_lake_video_v1"), 1.0), (_mod("plate 2", "/mods/fw_moon_lake_video_v1"), 1.0),
          (_mod("aretia_hall"), 0.8)]


def test_spec_matches_names_files_and_parts_per_beat():
    spec = parse_beat_refs("tairn, SGAEYL_v2, lake | tairn, sgaeyl\naretia_hall, *|none", LOADED)
    assert spec.names == ("tairn_v2", "sgaeyl_v2", "plate 1", "plate 2", "aretia_hall")
    # A bundle's members share its file name, so one word selects every plate.
    assert spec.beats == ((0, 1, 2, 3), (0, 1), (0, 1, 2, 3, 4), ())
    assert spec.describe().startswith("beat 1 = tairn_v2, sgaeyl_v2, plate 1, plate 2; beat 2 = tairn_v2, sgaeyl_v2;")
    assert spec.describe().endswith("beat 4 = none")


def test_an_exact_name_wins_over_a_partial_match():
    mods = [(_mod("lake"), 1.0), (_mod("lake_night"), 1.0)]
    assert parse_beat_refs("lake | lake_night | lak", mods).beats == ((0,), (1,), (0, 1))


def test_empty_spec_means_every_beat_sees_every_mod():
    assert parse_beat_refs("", LOADED) is None
    assert parse_beat_refs("  \n ", LOADED) is None


@pytest.mark.parametrize("text,message", [
    ("tairn | dragon_cave", "no loaded RefMod matches 'dragon_cave'"),
    ("tairn | ", "beat 2 names no RefMod"),
    ("tairn || lake", "beat 2 names no RefMod"),
])
def test_spec_errors_name_the_beat_and_the_loaded_mods(text, message):
    with pytest.raises(ValueError, match=message):
        parse_beat_refs(text, LOADED)
    with pytest.raises(ValueError, match="connect the RefMod loader"):
        parse_beat_refs("tairn", [])


def test_keeps_drop_only_marked_refmods_and_leave_native_references():
    native = _image_ref(9, refmod=False)
    refs = [native, _image_ref(1), _image_ref(2), _image_ref(3)]
    spec = parse_beat_refs("a, b | a, c | *", [(_mod("a"), 1), (_mod("b"), 1), (_mod("c"), 1)])
    assert beat_ref_keeps(refs, spec, 3) == [(0, 1, 2), (0, 1, 3), None]
    # Nothing to select from on a negative pass: no refs at all, or none the bridge injected.
    assert beat_ref_keeps(None, spec, 3, positive=False) is None
    assert beat_ref_keeps([native], spec, 3, positive=False) is None
    assert beat_ref_keeps(refs, None, 3) is None
    assert beat_ref_keeps(refs, parse_beat_refs("* | * | a, b, c", [(_mod(n), 1) for n in "abc"]), 3) is None


def test_keeps_fail_closed_when_the_sampler_and_the_spec_disagree():
    refs = [_image_ref(1), _image_ref(2)]
    spec = parse_beat_refs("a | b", [(_mod("a"), 1), (_mod("b"), 1)])
    with pytest.raises(ValueError, match="name 2 beat\\(s\\) but the take has 3"):
        beat_ref_keeps(refs, spec, 3)
    with pytest.raises(ValueError, match="set up for 2 reference\\(s\\) but the sampler received 1"):
        beat_ref_keeps(refs[:1], spec, 2)
    # A scrambled bridge reorders blocks; differently shaped mods expose it.
    shuffled = parse_beat_refs("a | b", [(_mod("a", h=8), 1), (_mod("b"), 1)])
    with pytest.raises(ValueError, match="scramble_seed at -1"):
        beat_ref_keeps(refs, shuffled, 2)


def test_filter_payload_keeps_condition_latents_in_step_with_references():
    keyframe = {"resolved_frame_index": 0, "latent": torch.zeros(1, 2, 1, 4, 4),
                "audio_latent": torch.zeros(1, 2, 2, 1)}
    refs = [_image_ref(1), _video_ref(2), _audio_ref(3), _image_ref(4)]
    payload = {
        "seed": 5, "layout": object(), "keyframes": [keyframe], "refs": refs,
        "cond_video_latents": [keyframe["latent"], refs[0]["latent"], refs[1]["latent"], refs[3]["latent"]],
        "cond_audio_latents": [keyframe["audio_latent"], refs[1]["audio_latent"], refs[2]["audio_latent"]],
    }
    out = filter_payload(payload, (0, 2, 3))
    assert "layout" not in out and out["seed"] == 5 and out["keyframes"] is payload["keyframes"]
    assert [r is refs[i] for r, i in zip(out["refs"], (0, 2, 3))] == [True] * 3
    assert [float(z.flatten()[0]) for z in out["cond_video_latents"]] == [0.0, 1.0, 4.0]
    assert [float(z.flatten()[0]) for z in out["cond_audio_latents"]] == [0.0, 3.0]
    assert len(payload["cond_video_latents"]) == 4 and len(payload["refs"]) == 4

    payload["cond_video_latents"] = payload["cond_video_latents"][1:]
    with pytest.raises(ValueError, match="cond_video_latents holds 3 entries"):
        filter_payload(payload, (0, 2, 3))


def _assert_window_matches_full(full, local, refs, keep, v0, v1, frame_rows):
    from h3forge.layout import _ref_segment_count, target_segments

    counts = [_ref_segment_count(block) for block in refs]
    full_ref = [s for s in full.segments if s[2] in ("ref_img", "ref_audio")]
    local_ref = [s for s in local.segments if s[2] in ("ref_img", "ref_audio")]
    expected, cursor = [], 0
    for index, count in enumerate(counts):
        if index in keep:
            expected.extend(full_ref[cursor:cursor + count])
        cursor += count
    assert [s[2] for s in local_ref] == [s[2] for s in expected]
    for (la, lb, _), (fa, fb, _) in zip(local_ref, expected):
        assert torch.equal(local.position_ids[la:lb], full.position_ids[fa:fb])
    src, dst = target_segments(full), target_segments(local)
    assert torch.equal(local.position_ids[dst.video_start:dst.video_stop],
                       full.position_ids[src.video_start + v0 * frame_rows:src.video_start + v1 * frame_rows])
    dropped_rows = sum(b - a for a, b, _ in full_ref) - sum(b - a for a, b, _ in local_ref)
    assert dropped_rows > 0


@pytest.mark.parametrize("keep", [(0, 2, 3), (1,), (3,), ()])
def test_window_layout_packs_a_reference_subset_at_full_layout_positions(monkeypatch, keep):
    fake_minimax.install(monkeypatch)
    from h3forge.layout import audio_range_for_video_window, clone_window_layout

    refs = [_image_ref(1), _video_ref(2), _audio_ref(3), _image_ref(4, h=2, w=2)]
    keyframes = [{"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 1, 4, 4)}]
    full = PackedLayout(7, 10, 4, 4, 57, keyframes=keyframes, refs=refs)
    v0, v1 = 2, 7
    a0, a1 = audio_range_for_video_window(full, v0, v1)
    local = clone_window_layout(full_layout=full, text_len=5, video_shape=(v1 - v0, 4, 4), audio_t=57,
                                video_range=(v0, v1), audio_range=(a0, a1), keyframes=keyframes, refs=refs,
                                ref_keep=keep)
    _assert_window_matches_full(full, local, refs, keep, v0, v1, full.frame_rows)
    # The keyframe rows are not references and stay where the full layout has them.
    cond_full = next(s for s in full.segments if s[2] == "cond")
    cond_local = next(s for s in local.segments if s[2] == "cond")
    assert torch.equal(local.position_ids[cond_local[0]:cond_local[1]], full.position_ids[cond_full[0]:cond_full[1]])


def test_window_layout_rejects_a_keep_list_it_cannot_honour(monkeypatch):
    fake_minimax.install(monkeypatch)
    from h3forge.layout import clone_window_layout

    refs = [_image_ref(1), _image_ref(2)]
    full = PackedLayout(7, 10, 4, 4, 57, refs=refs)
    for keep in [(1, 0), (0, 0), (2,)]:
        with pytest.raises(ValueError, match="ref_keep must be ascending"):
            clone_window_layout(full_layout=full, text_len=7, video_shape=(5, 4, 4), audio_t=57,
                                video_range=(0, 5), audio_range=(0, 30), refs=refs, ref_keep=keep)


def test_native_layout_rows_match_the_filtered_condition_latents():
    """Against ComfyUI's own PackedLayout: the rows a window reserves for
    conditions are exactly the rows the filtered latents fill."""
    core = native_h3()
    from h3forge.layout import audio_range_for_video_window, clone_window_layout

    refs = [_image_ref(1), _video_ref(2), _audio_ref(3), _image_ref(4, h=2, w=6)]
    keyframe = {"resolved_frame_index": 0, "latent": torch.zeros(1, 2, 1, 4, 4)}
    payload = {
        "keyframes": [keyframe], "refs": refs,
        "cond_video_latents": [keyframe["latent"], refs[0]["latent"], refs[1]["latent"], refs[3]["latent"]],
        "cond_audio_latents": [refs[1]["audio_latent"], refs[2]["audio_latent"]],
    }
    total, audio_t = 12, 68
    full = core.PackedLayout(7, total, 4, 4, audio_t, keyframes=[keyframe], refs=refs)
    v0, v1 = 3, 9
    a0, a1 = audio_range_for_video_window(full, v0, v1)
    for keep in [(0, 2, 3), (1,), (2,), ()]:
        local = clone_window_layout(full_layout=full, text_len=4, video_shape=(v1 - v0, 4, 4), audio_t=audio_t,
                                    video_range=(v0, v1), audio_range=(a0, a1), keyframes=[keyframe], refs=refs,
                                    ref_keep=keep)
        _assert_window_matches_full(full, local, refs, keep, v0, v1, 4)
        filtered = filter_payload(payload, keep)
        video_rows = sum(z.shape[2] * (z.shape[3] // 2) * (z.shape[4] // 2) for z in filtered["cond_video_latents"])
        audio_rows = sum(z.shape[-1] * 2 for z in filtered["cond_audio_latents"])
        assert int((~local.img_update).sum()) == video_rows
        assert int((~local.audio_update).sum()) == audio_rows
        # Built from scratch, the same subset packs the same number of rows.
        fresh = core.PackedLayout(4, v1 - v0, 4, 4, a1 - a0, keyframes=[keyframe], refs=filtered["refs"])
        assert fresh.seq_len == local.seq_len and [s[2] for s in fresh.segments] == [s[2] for s in local.segments]


def _executor(fn):
    class Executor:
        class_obj = SimpleNamespace(patch_size=(1, 2, 2))

        def __call__(self, *args, **kwargs):
            return fn(*args, **kwargs)
    return Executor()


def _take(monkeypatch, beats=2, total=40, with_layout=True):
    fake_minimax.install(monkeypatch)
    frames = sum((1, 4, 4, 4, 4)[i % 5] for i in range(total))
    audio_t = round(frames * 5 / 3)
    x = [torch.zeros(1, 1, total, 2, 2), torch.zeros(1, 1, 2, audio_t)]
    prompts = tuple(torch.full((1, 3 + i, 8), float(i + 1)) for i in range(beats))
    native = _image_ref(9, refmod=False, h=2, w=2)
    refs = [native, _image_ref(1, h=2, w=2), _image_ref(2, h=2, w=2), _image_ref(3, h=2, w=2)]
    payload = {"refs": refs, "cond_video_latents": [r["latent"] for r in refs], "cond_audio_latents": []}
    if with_layout:
        payload["layout"] = PackedLayout(max(p.shape[1] for p in prompts), total, 2, 2, audio_t, refs=refs)
    if beats > 1:
        payload["h3forge_prompt_segments"] = prompts
    return x, prompts[0], payload


def _options(spec=None):
    options = {"sigmas": torch.tensor([1.0]), "sample_sigmas": torch.tensor([1.0, 0.0])}
    if spec is not None:
        options[BEAT_REFS_KEY] = spec
    return options


MODS = [(_mod("dragons", h=2, w=2), 1.0), (_mod("lake", h=2, w=2), 1.0), (_mod("hall", h=2, w=2), 1.0)]


def test_each_window_gets_its_beats_references(monkeypatch, capsys):
    x, context, payload = _take(monkeypatch)
    spec = parse_beat_refs("dragons, lake | dragons, hall", MODS)
    seen = []

    def run(local_x, timestep, text, opts, **kwargs):
        local = kwargs["minimax_payload"]
        beat = int(text[0, 0, 0]) - 1
        values = [float(r["latent"].flatten()[0]) for r in local["refs"]]
        assert [float(z.flatten()[0]) for z in local["cond_video_latents"]] == values
        ref_segments = [s for s in local["layout"].segments if s[2] == "ref_img"]
        assert len(ref_segments) == len(values)
        seen.append((beat, values))
        return [torch.ones_like(local_x[0]), torch.ones_like(local_x[1])]

    video, audio = make_context_wrapper(ContextPolicy(25, 8, False, handoff="lead"))(
        _executor(run), x, torch.tensor([1000]), context, _options(spec), minimax_payload=payload)
    assert {beat for beat, _ in seen} == {0, 1}
    for beat, values in seen:
        assert values == ([9.0, 1.0, 2.0] if beat == 0 else [9.0, 1.0, 3.0])
    assert torch.equal(video, torch.ones_like(video)) and torch.equal(audio, torch.ones_like(audio))
    assert len(payload["refs"]) == 4 and len(payload["cond_video_latents"]) == 4
    assert "per-beat RefMods: beat 1 = dragons, lake; beat 2 = dragons, hall" in capsys.readouterr().out


def test_without_a_spec_every_window_keeps_the_payload_it_always_had(monkeypatch):
    x, context, payload = _take(monkeypatch)

    def run(local_x, timestep, text, opts, **kwargs):
        local = kwargs["minimax_payload"]
        assert local["refs"] is payload["refs"]
        assert local["cond_video_latents"] is payload["cond_video_latents"]
        assert len([s for s in local["layout"].segments if s[2] == "ref_img"]) == 4
        return [torch.ones_like(local_x[0]), torch.ones_like(local_x[1])]

    for options in (_options(), _options(parse_beat_refs("* | dragons, lake, hall", MODS))):
        make_context_wrapper(ContextPolicy(25, 8, False))(
            _executor(run), x, torch.tensor([1000]), context, options, minimax_payload=payload)


def test_a_single_window_take_filters_its_one_beat(monkeypatch):
    x, context, payload = _take(monkeypatch, beats=1, total=10)
    seen = []

    def run(local_x, timestep, text, opts, **kwargs):
        local = kwargs["minimax_payload"]
        assert "layout" not in local  # the forward rebuilds it from the filtered references
        seen.append([float(r["latent"].flatten()[0]) for r in local["refs"]])
        return local_x

    make_context_wrapper(ContextPolicy(25, 8))(
        _executor(run), x, torch.tensor([1000]), context, _options(parse_beat_refs("hall", MODS)),
        minimax_payload=payload)
    assert seen == [[9.0, 3.0]]


def test_a_spec_with_the_wrong_number_of_beats_stops_before_any_window_runs(monkeypatch):
    x, context, payload = _take(monkeypatch)

    def run(*args, **kwargs):
        raise AssertionError("no window may run")

    with pytest.raises(ValueError, match="name 3 beat\\(s\\) but the take has 2"):
        make_context_wrapper(ContextPolicy(25, 8, False))(
            _executor(run), x, torch.tensor([1000]), context,
            _options(parse_beat_refs("dragons | lake | hall", MODS)), minimax_payload=payload)


def test_node_stores_the_spec_on_a_clone_and_clears_it_when_empty(monkeypatch):
    nodes = _import_nodes(monkeypatch)
    base = FakePatcher()
    patched, = nodes.H3ForgeBeatRefMods().patch(base, MODS, "dragons, lake | dragons, hall")
    spec = patched.model_options["transformer_options"][BEAT_REFS_KEY]
    assert isinstance(spec, BeatRefs) and spec.beats == ((0, 1), (0, 2))
    assert base.model_options == {}
    # Later clones (the context node's, the sampler's) carry it along.
    assert patched.clone().model_options["transformer_options"][BEAT_REFS_KEY] is spec
    cleared, = nodes.H3ForgeBeatRefMods().patch(patched, MODS, "")
    assert BEAT_REFS_KEY not in cleared.model_options["transformer_options"]
    assert "H3ForgeBeatRefMods" in nodes.NODE_CLASS_MAPPINGS and "H3ForgeBeatRefMods" in nodes.NODE_DISPLAY_NAME_MAPPINGS


@pytest.mark.parametrize("arrived", [2, 1, 0])
@pytest.mark.parametrize("positive", [True, False])
def test_reference_arrivals_on_positive_and_negative_passes(arrived, positive):
    spec = parse_beat_refs("a | b", [(_mod("a"), 1), (_mod("b"), 1)])
    refs = [_image_ref(1), _image_ref(2)][:arrived] or None
    if arrived == 2:
        assert beat_ref_keeps(refs, spec, 2, positive=positive) == [(0,), (1,)]
    elif arrived == 1:
        with pytest.raises(ValueError, match="set up for 2 reference\\(s\\) but the sampler received 1"):
            beat_ref_keeps(refs, spec, 2, positive=positive)
    elif positive:
        with pytest.raises(ValueError, match="received 0.*reference-free baseline"):
            beat_ref_keeps(refs, spec, 2, positive=positive)
    else:
        assert beat_ref_keeps(refs, spec, 2, positive=positive) is None


def test_a_positive_window_pass_without_references_stops_before_any_window_runs(monkeypatch):
    x, context, payload = _take(monkeypatch)
    payload = {**payload, "refs": None}
    spec = parse_beat_refs("dragons | lake", MODS)

    def run(*args, **kwargs):
        raise AssertionError("no window may run")

    with pytest.raises(ValueError, match="received 0"):
        make_context_wrapper(ContextPolicy(25, 8, False))(
            _executor(run), x, torch.tensor([1000]), context, _options(spec), minimax_payload=payload)


@pytest.mark.parametrize("beats", [1, 2])
@pytest.mark.parametrize("group,conditioning,slots,negative", [
    ("negative", {}, [1], True),                         # ordinary CFG
    ("negative", {}, [0], True),                         # negative-only forward, DualCFG
    ("middle", {"prompt_type": "negative"}, [1], True),  # DualCFG tags its middle conditioning
    ("positive", {}, [2], False),                        # DualCFG positive
    ("positive", {}, [0, 1], False),                     # slot lists cannot establish polarity
])
def test_missing_refs_follow_the_encoded_prompt_role(monkeypatch, beats, group, conditioning, slots, negative):
    x, context, payload = _take(monkeypatch, beats=beats, with_layout=False)
    prompts = payload.get("h3forge_prompt_segments")
    constants = definitions("comfy/conds.py", ["CONDRegular", "CONDConstant"])["CONDConstant"]
    encode = definitions("comfy/samplers.py", ["encode_model_conds"])["encode_model_conds"]
    def native(**kw):
        return {"minimax_payload": constants({}), "c_crossattn": constants(context)}

    prepare = make_segmented_extra_conds(native, SimpleNamespace(get_dtype_inference=lambda: context.dtype),
                                         SimpleNamespace(preprocess_text_embeds=lambda text: text))
    row = {"model_conds": {}, **conditioning}
    if prompts:
        row.update(h3forge_prompt_segments=prompts, h3forge_prompt_segment_tags=(None,) * beats)
    prepared = encode(prepare, [row], x[0], "cpu", group)[0]["model_conds"]["minimax_payload"].cond
    spec = parse_beat_refs(" | ".join(["dragons"] * beats), MODS)
    options = {**_options(spec), "cond_or_uncond": slots}
    calls = []

    def run(local_x, *args, **kw):
        calls.append(True)
        return local_x

    forward = make_context_wrapper(ContextPolicy(25, 8, False))
    if negative:
        forward(_executor(run), x, torch.tensor([1000]), context, options, minimax_payload=prepared)
        assert calls
    else:
        with pytest.raises(ValueError, match="received 0"):
            forward(_executor(run), x, torch.tensor([1000]), context, options, minimax_payload=prepared)
        assert not calls

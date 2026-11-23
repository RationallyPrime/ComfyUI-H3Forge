"""Control strength per beat: spec parsing, one window's strength, the context wrapper, the node."""
from types import SimpleNamespace

import pytest
import torch

import fake_minimax
from fake_minimax import PackedLayout
from h3forge.context import ContextPolicy, make_context_wrapper
from h3forge.control import BEAT_CONTROL_KEY, beat_control_scales, control_window, parse_beat_control
from test_nodes import FakePatcher, _import_nodes


class MiniMaxH3FunControlPatch:
    """Stands in for ComfyUI's patch: the fields ``control_window`` touches and the wrapper it is found by."""

    def __init__(self, strength=0.8, sigma_start=1.0, sigma_end=0.0):
        self.strength, self.sigma_start, self.sigma_end = strength, sigma_start, sigma_end
        self.control_latent = self.control_latent_shape = self.control_stream = self.pristine_stream = None

    def prepare_control_latent(self, shape):
        shape = tuple(shape)
        if self.control_latent is None or self.control_latent_shape != shape:
            self.control_latent = torch.arange(shape[2]).view(1, 1, -1, 1, 1).expand(shape)
            self.control_latent_shape = shape

    def diffusion_model_wrapper(self, executor, *args, **kwargs):
        return executor(*args, **kwargs)


SHAPE = (1, 1, 10, 2, 2)
AT = (torch.tensor([700.]), {"sigmas": torch.tensor([.7])})


def _holder(*patches):
    return SimpleNamespace(wrappers=[patch.diffusion_model_wrapper for patch in patches])


def test_spec_reads_numbers_and_words_per_beat():
    assert parse_beat_control("1 | 1 | 0") == (1.0, 1.0, 0.0)
    assert parse_beat_control("on\n0.5\nOFF") == (1.0, 0.5, 0.0)
    assert parse_beat_control("") is None and parse_beat_control("  \n ") is None and parse_beat_control(None) is None


@pytest.mark.parametrize("text,message", [
    ("1 | pose | 0", "beat 2 is 'pose'"),
    ("1 | | 0", "beat 2 is ''"),
    ("1 | -0.5", "beat 2 is -0.5"),
    ("nan", "beat 1 is nan"),
])
def test_spec_errors_name_the_beat(text, message):
    with pytest.raises(ValueError, match=message):
        parse_beat_control(text)


def test_scales_must_count_the_takes_beats_and_all_ones_change_nothing():
    assert beat_control_scales(None, 3) is None
    assert beat_control_scales((1.0, 1.0, 1.0), 3) is None
    assert beat_control_scales((1.0, 1.0, 0.0), 3) == (1.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="names 2 beat\\(s\\) but the take has 3"):
        beat_control_scales((1.0, 0.0), 3)


def test_a_window_multiplies_the_strength_and_gives_it_back():
    patch = MiniMaxH3FunControlPatch(strength=0.8)
    with control_window(_holder(patch), SHAPE, (4, 10), *AT, scale=0.0):
        assert patch.strength == 0.0
        assert patch.control_latent_shape == (1, 1, 6, 2, 2)
        assert torch.equal(patch.control_latent[0, 0, :, 0, 0], torch.arange(4, 10))
    assert patch.strength == 0.8 and patch.control_latent_shape == SHAPE
    with control_window(_holder(patch), SHAPE, (0, 6), *AT):
        assert patch.strength == 0.8
    with pytest.raises(RuntimeError, match="forward failed"):
        with control_window(_holder(patch), SHAPE, (0, 6), *AT, scale=0.5):
            assert patch.strength == pytest.approx(0.4)
            raise RuntimeError("forward failed")
    assert patch.strength == 0.8 and patch.control_latent_shape == SHAPE


def test_a_control_outside_its_sigma_range_is_left_alone():
    patch = MiniMaxH3FunControlPatch(strength=0.8, sigma_start=0.5)
    with control_window(_holder(patch), SHAPE, (4, 10), *AT, scale=0.0):
        assert patch.strength == 0.8 and patch.control_latent is None
    assert patch.strength == 0.8


class _Executor:
    class_obj = SimpleNamespace(patch_size=(1, 2, 2))

    def __init__(self, fn, wrappers):
        self.fn, self.wrappers = fn, wrappers

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def _take(monkeypatch, beats=3, total=58):
    fake_minimax.install(monkeypatch)
    frames = sum((1, 4, 4, 4, 4)[i % 5] for i in range(total))
    audio_t = round(frames * 5 / 3)
    x = [torch.zeros(1, 1, total, 2, 2), torch.zeros(1, 1, 2, audio_t)]
    prompts = tuple(torch.full((1, 3 + i, 8), float(i + 1)) for i in range(beats))
    payload = {"layout": PackedLayout(max(p.shape[1] for p in prompts), total, 2, 2, audio_t)}
    if beats > 1:
        payload["h3forge_prompt_segments"] = prompts
    return x, prompts[0], payload


def _options(spec=None):
    options = {"sigmas": torch.tensor([1.0]), "sample_sigmas": torch.tensor([1.0, 0.0])}
    if spec is not None:
        options[BEAT_CONTROL_KEY] = spec
    return options


def _run_take(monkeypatch, spec, beats=3, total=58, policy=None):
    x, context, payload = _take(monkeypatch, beats, total)
    patch = MiniMaxH3FunControlPatch(strength=0.8)
    seen = []

    def run(local_x, timestep, text, opts, **kwargs):
        start = getattr(kwargs["minimax_payload"].get("layout"), "_h3forge_video_offset", 0)
        assert patch.control_latent_shape == tuple(local_x[0].shape)
        assert int(patch.control_latent[0, 0, 0, 0, 0]) == start
        seen.append((int(text[0, 0, 0]) - 1, patch.strength))
        return [torch.ones_like(local_x[0]), torch.ones_like(local_x[1])]

    make_context_wrapper(policy or ContextPolicy(25, 8, False, handoff="lead"))(
        _Executor(run, [patch.diffusion_model_wrapper]), x, torch.tensor([1000]), context, _options(spec),
        minimax_payload=payload)
    assert patch.strength == 0.8
    return seen


def test_each_window_runs_at_its_beats_strength(monkeypatch, capsys):
    seen = _run_take(monkeypatch, parse_beat_control("1 | 0.5 | off"))
    assert {beat for beat, _ in seen} == {0, 1, 2}
    for beat, strength in seen:
        assert strength == pytest.approx((0.8, 0.4, 0.0)[beat])
    assert "control per beat: beat 1 = 1, beat 2 = 0.5, beat 3 = off" in capsys.readouterr().out


def test_without_a_spec_every_window_keeps_the_strength_of_the_control_node(monkeypatch, capsys):
    for spec in (None, parse_beat_control("1 | 1 | 1")):
        assert {strength for _, strength in _run_take(monkeypatch, spec)} == {0.8}
    assert "control per beat" not in capsys.readouterr().out


def test_a_single_window_take_runs_at_its_one_beats_strength(monkeypatch):
    assert _run_take(monkeypatch, parse_beat_control("0.5"), beats=1, total=10, policy=ContextPolicy(25, 8)) == [
        (0, pytest.approx(0.4))]


def test_a_spec_with_the_wrong_number_of_beats_stops_before_any_window_runs(monkeypatch):
    x, context, payload = _take(monkeypatch)

    def run(*args, **kwargs):
        raise AssertionError("no window may run")

    with pytest.raises(ValueError, match="names 2 beat\\(s\\) but the take has 3"):
        make_context_wrapper(ContextPolicy(25, 8, False))(
            _Executor(run, []), x, torch.tensor([1000]), context, _options(parse_beat_control("1 | 0")),
            minimax_payload=payload)


def test_node_stores_the_spec_on_a_clone_and_clears_it_when_empty(monkeypatch):
    nodes = _import_nodes(monkeypatch)
    base = FakePatcher()
    patched, = nodes.H3ForgeBeatControl().patch(base, "1 | 1 | 0")
    assert patched.model_options["transformer_options"][BEAT_CONTROL_KEY] == (1.0, 1.0, 0.0)
    assert base.model_options == {}
    assert patched.clone().model_options["transformer_options"][BEAT_CONTROL_KEY] == (1.0, 1.0, 0.0)
    cleared, = nodes.H3ForgeBeatControl().patch(patched, "")
    assert BEAT_CONTROL_KEY not in cleared.model_options["transformer_options"]
    assert "H3ForgeBeatControl" in nodes.NODE_CLASS_MAPPINGS and "H3ForgeBeatControl" in nodes.NODE_DISPLAY_NAME_MAPPINGS

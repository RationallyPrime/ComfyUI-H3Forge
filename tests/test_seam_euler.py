"""Exercise the sampler and window wrapper across Comfy's packed A/V boundary."""
import hashlib
import sys
from types import SimpleNamespace

import pytest
import torch

from core_source import definitions
from fake_minimax import install
from h3forge.context import ContextPolicy, make_context_wrapper, make_freenoise_wrapper
from h3forge.seam_euler import SEAM_STATE, OverlapScores, SeamState, _fresh_noise, _renoise_rows, sample_seam_euler
from test_audit_repairs import _executor


@pytest.fixture
def core_euler(monkeypatch):
    # Use the checkout's actual sampler and derivative without initializing devices.
    utils = definitions("comfy/k_diffusion/utils.py", ["append_dims"])
    env = definitions("comfy/k_diffusion/sampling.py", ["to_d", "sample_euler"], {
        "torch": torch, "utils": SimpleNamespace(**utils), "trange": lambda n, **kw: range(n),
    })
    sampling = SimpleNamespace(**env)
    monkeypatch.setitem(sys.modules, "comfy.k_diffusion.sampling", sampling)
    return sampling.sample_euler


def test_zero_repairs_matches_core_euler_including_callbacks(core_euler):
    x = torch.arange(48, dtype=torch.float32).view(2, 1, 24) / 17
    sigmas = torch.tensor([.93, .71, .43, .06, 0.])
    def run(sample):
        calls, previews = [], []
        def model(x, sigma, seed):
            calls.append((x.clone(), sigma.clone()))
            return x.sin() / 3 + sigma[:, None, None] * seed
        out = sample(model, x, sigmas, {"seed": 12}, previews.append, disable=True)
        return out, calls, previews

    expected, calls, previews = run(core_euler)
    actual, seen, callbacks = run(sample_seam_euler)
    assert torch.equal(actual, expected)
    assert len(seen) == len(calls) == len(callbacks) == len(previews) == 4
    for (x, s), (y, t), a, b in zip(seen, calls, previews, callbacks):
        assert torch.equal(x, y) and torch.equal(s, t)
        assert all(torch.equal(a[k], b[k]) for k in ("x", "sigma", "sigma_hat", "denoised"))
        assert a["i"] == b["i"]


def _run(monkeypatch, *, context=True, sigmas=(.8, .6, .4, .2, 0.), seed=42, settle=False,
         pinned=False, freenoise=False, **options):
    install(monkeypatch)
    policy = ContextPolicy(4, 2, stagger=False, blend="flat", freenoise=freenoise, seam_repair=context)
    window = make_context_wrapper(policy)
    video = torch.arange(48, dtype=torch.float32).view(1, 2, 6, 2, 2) / 48
    audio = torch.arange(60, dtype=torch.float32).view(1, 1, 2, 30) / 60
    def pack(v, a):
        return torch.cat([v.reshape(1, 1, -1), a.reshape(1, 1, -1)], -1)
    x = pack(video, audio)
    source_noise = x + 2
    input_mask = torch.ones_like(video[:, :1])
    if pinned:
        input_mask[:, :, 2, 0, 0] = 0.5
    mask = pack(input_mask.expand_as(video), torch.ones_like(audio)) if pinned else None
    sigmas = torch.tensor(sigmas)
    args = {"seed": seed, "model_options": {"transformer_options": {"sample_sigmas": sigmas}}, "denoise_mask": mask}
    guider = SimpleNamespace(inner_model=SimpleNamespace(latent_shapes=[video.shape, audio.shape]))
    calls, previews = [], []

    def sample(guider, sigmas, args, callback, noise):
        def model(latent, sigma, model_options, **kwargs):
            opts = dict(model_options["transformer_options"], sigmas=sigma)
            state = opts.get(SEAM_STATE)
            retry = state is not None and state.row_ratio is not None
            views = [latent[..., :48].view_as(video), latent[..., 48:].view_as(audio)]
            masks = []
            def predict(local, timestep, prompt, options, **kw):
                start = kw["minimax_payload"]["layout"]._h3forge_video_offset
                m = kw["denoise_mask"]
                masks.append(None if m is None else m.clone())
                # At the shared rows the two predictions are 1 and 3, fused to 2.
                v = 1 + (0 if settle and retry else start)
                return torch.full_like(local[0], v), torch.full_like(local[1], 5 if retry else 2)
            v, a = window(_executor(predict), views, sigma * 1000, torch.zeros(1, 2, 8), opts,
                          denoise_mask=input_mask if pinned else None)
            if pinned:  # native H3 forward applies the original mask outside the wrapper
                v = v * input_mask
            calls.append((retry, latent.clone(), masks, dict(state.scores) if state else {}))
            return latent - sigma * pack(v, a)
        model.noise = noise  # KSAMPLER keeps the post-FreeNoise pack here
        return sample_seam_euler(model, x, sigmas, args, callback, True, **options)

    out = make_freenoise_wrapper(policy)(sample, guider, sigmas, args, previews.append, source_noise)
    assert SEAM_STATE not in args["model_options"]["transformer_options"]
    return out, calls, previews


@pytest.mark.parametrize(("changes", "retries"), [
    ({"repair_threshold": 1.}, 0), ({"repair_threshold": .999}, 2),
    ({"repair_until": .8}, 0), ({"repair_until": .6}, 1),
    ({"repairs": 4, "repair_budget": .49}, 1),
    ({"repairs": 4, "repair_budget": .24}, 0),
    ({"context": False}, 0),
])
def test_threshold_cutoff_and_global_budget(monkeypatch, core_euler, changes, retries):
    opts = {"repairs": 1, "repair_budget": 2., **changes}
    _, calls, callbacks = _run(monkeypatch, **opts)
    assert len(calls) == 4 + retries
    assert sum(retry for retry, *_ in calls) == retries
    assert len(callbacks) == 4
    if opts.get("context", True):
        assert all(scores == {(2, 4): 1.} for _, _, _, scores in calls)


def test_repeat_stops_on_measured_agreement(monkeypatch, core_euler):
    _, calls, _ = _run(monkeypatch, sigmas=(.8, .2), repairs=4, repair_budget=4, settle=True)
    assert [scores for *_, scores in calls] == [{(2, 4): 1.}, {(2, 4): 0.}]


def test_original_noise_uses_freenoise_and_mixed_row_sigmas(monkeypatch, core_euler):
    opts = dict(sigmas=(.8, .2), freenoise=True, repair_ramp=1, repair_noise="original", repair_budget=1)
    base, control, base_preview = _run(monkeypatch, context=False, repairs=0, **opts)
    out, calls, callbacks = _run(monkeypatch, repairs=1, **opts)
    assert len(control) == 1 and len(calls) == 2
    initial = calls[0][1]
    def video(p):
        return p[..., :48].reshape(1, 2, 6, 2, 2)
    base_x0 = video(base_preview[0]["denoised"])
    weights = torch.tensor([0, .5, 1, 1, .5, 0]).view(1, 1, 6, 1, 1)
    noise = video(initial) + 2
    # FreeNoise replaces the added tail with a seeded permutation of frames 0,1.
    order = torch.randperm(2, generator=torch.Generator().manual_seed(42))
    noise[:, :, 4:] = noise.index_select(2, order)
    mixed = torch.lerp(video(base), .2 * base_x0 + .8 * noise, weights)
    retry, trial, masks, scores = calls[1]
    assert retry and scores == {(2, 4): 1.}
    torch.testing.assert_close(video(trial), mixed)
    assert torch.equal(trial[..., 48:], initial[..., 48:])
    row_sigma = .2 + .6 * weights
    for mask, start in zip(masks, (0, 2)):
        torch.testing.assert_close(mask, (row_sigma[:, :, start:start + 4] / .8).expand_as(mask))
    v = torch.tensor([1., 1., 2., 2., 3., 3.]).view(1, 1, 6, 1, 1)
    torch.testing.assert_close(video(out), mixed - (row_sigma - .2) * v)
    expected_x0 = torch.where(weights > 0, mixed - row_sigma * v, base_x0)
    torch.testing.assert_close(video(callbacks[0]["denoised"]), expected_x0)
    assert torch.equal(out[..., 48:], base[..., 48:])
    assert torch.equal(callbacks[0]["denoised"][..., 48:], base_preview[0]["denoised"][..., 48:])
    assert torch.equal(video(out)[:, :, [0, 5]], video(base)[:, :, [0, 5]])


def test_fresh_noise_reproducible_and_private(monkeypatch, core_euler):
    opts = dict(repairs=2, repair_budget=2, sigmas=(.8, .2))
    before = torch.random.get_rng_state()
    a, calls, _ = _run(monkeypatch, **opts)
    b, _, _ = _run(monkeypatch, **opts)
    c, _, _ = _run(monkeypatch, seed=43, **opts)
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert len(calls) == 3 and not torch.equal(calls[1][1], calls[2][1])
    draws = [_fresh_noise(a, 2**64 - 1, i, j) for i, j in [(0, 0), (0, 1), (1, 0)]]
    assert len({bytes(v.contiguous().flatten().view(torch.uint8).tolist()) for v in draws}) == 3
    assert torch.equal(before, torch.random.get_rng_state())


def test_first_fresh_repair_does_not_replay_initial_video_noise():
    # Comfy prepare_noise starts a CPU generator at the run seed. These dimensions
    # exercise the vectorized normal generator used for a video and its overlap.
    seed = 42
    base = torch.randn((1, 24, 202, 4, 8), generator=torch.Generator().manual_seed(seed))
    fresh = _fresh_noise(base[:, :, :21], seed, 0, 0)
    assert not torch.equal(fresh.flatten(), base.flatten()[:fresh.numel()])


def test_existing_inpaint_token_keeps_base_state_label_and_output(monkeypatch, core_euler):
    opts = dict(pinned=True, sigmas=(.8, .2), repair_ramp=0, repair_budget=1)
    base, _, _ = _run(monkeypatch, repairs=0, **opts)
    out, calls, _ = _run(monkeypatch, repairs=1, **opts)
    def view(x):
        return x[..., :48].reshape(1, 2, 6, 2, 2)
    assert torch.equal(view(out)[:, :, 2], view(base)[:, :, 2])
    assert torch.equal(view(calls[1][1])[:, :, 2], view(calls[0][1])[:, :, 2])
    assert calls[1][2][0][0, 0, 2].tolist() == [[.5, 1.], [1., 1.]]
    assert not torch.equal(view(out)[:, :, 3], view(base)[:, :, 3])

@pytest.mark.parametrize("enabled", [False, True])
def test_context_fusion_matches_main_bytes(monkeypatch, enabled):
    install(monkeypatch)
    video = torch.arange(48, dtype=torch.float32).reshape(1, 2, 6, 2, 2) / 37
    audio = torch.arange(60, dtype=torch.float32).reshape(1, 1, 2, 30) / 21
    def predict(local, timestep, text, options, **kw):
        start = kw["minimax_payload"]["layout"]._h3forge_video_offset
        return [local[0] * .3 + start, local[1] * .7 - start]
    state = SeamState()
    policy = ContextPolicy(window_frames=4, overlap_frames=2, seam_repair=enabled)
    outputs = make_context_wrapper(policy)(_executor(predict), [video, audio], None, torch.zeros(1, 1, 8),
                                           {SEAM_STATE: state})
    # Recorded from the same inputs on untouched 05792b7, video then audio.
    assert [hashlib.sha256(bytes(t.contiguous().flatten().view(torch.uint8).tolist())).hexdigest() for t in outputs] == [
        "059239e9a5cebc30544ff1056f47653919e0a808589dab3ba45a86a9945a0cd0",
        "11ac54d7474f5f66e9b6551c077074a587d2f5ebbe006106be6d0d1d98e9bd7f",
    ]
    assert bool(state.scores) is enabled


def test_scores_use_pairwise_shared_writes_and_final_fused_denominator():
    # Three contributing windows, with a hole reserved for an exclusive fourth.
    metric = OverlapScores([[(0, 2), (3, 5)], [(1, 2), (3, 6)], [(3, 7)], [(2, 3)]])
    rows = torch.ones(1, 1, 7, 1, 1)
    for i, (start, end, v) in enumerate([(0, 5, 1), (1, 6, -1), (3, 7, 3), (2, 3, 9)]):
        metric.observe(i, start, rows[:, :, start:end] * v)
    fused = torch.tensor([1., 0., 9., 1., 1., 1., 3.]).view_as(rows)
    state = SeamState()
    metric.publish(fused, state)
    assert state.scores == pytest.approx({(1, 2): float("inf"), (3, 5): 2., (3, 6): 4.})


def test_fresh_ramp_residual_matches_the_sigma_given_to_h3():
    rng = torch.Generator().manual_seed(42)
    x0 = torch.full((65536,), 3.)
    original = torch.randn(x0.shape, generator=rng)
    fresh = torch.randn(x0.shape, generator=rng)
    sigma, following = torch.tensor(1.), torch.tensor(.98)
    advanced = (1 - following) * x0 + following * original
    for w in (.1, .5, 1.):
        row_sigma = following + w * (sigma - following)
        trial = _renoise_rows(advanced, x0, fresh, sigma, following, torch.tensor(w), fresh=True)
        residual = trial - (1 - row_sigma) * x0
        assert float(residual.std()) == pytest.approx(float(row_sigma), abs=.01)
        assert float(residual.mean()) == pytest.approx(0, abs=.01)
    torch.testing.assert_close(trial, fresh)

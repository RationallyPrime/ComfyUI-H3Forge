# Temporary test hook (20261120): lets /tmp/h3forge_seam.json also set blend, stagger, freenoise and
# renoise_until (FlowLong-style stochastic early phase) for the published-method baselines.
# Originals: /workspace/backups/seam-sweep-20261120/{takes.py,__init__.py}
import pathlib, shutil
d = pathlib.Path("/workspace/runpod-slim/ComfyUI/custom_nodes/ComfyUI-MMH3-ClipLoops")
b = pathlib.Path("/workspace/backups/seam-sweep-20261120"); b.mkdir(parents=True, exist_ok=True)

t = d / "takes.py"; s = t.read_text()
assert "seam-sweep-hook" in s and (b / "takes.py").exists()
if "seam-baseline-hook" not in s:
    old = ("    return forge.patch(model, window_frames=plan['window'], overlap_frames=plan['overlap'], stagger=plan['stagger'],\n"
           "                       blend='pyramid', segment_seams='blend', freenoise=plan['freenoise'], **handoff(forge))[0]\n")
    assert s.count(old) == 1
    new = '''    # seam-baseline-hook: temporary test override (20261120)
    import json as _json, os as _os
    _o = {}
    if _os.path.exists('/tmp/h3forge_seam.json'):
        with open('/tmp/h3forge_seam.json') as _f:
            _o = _json.load(_f)
    _kw = dict(stagger=bool(_o.get('stagger', plan['stagger'])), blend=_o.get('blend', 'pyramid'),
               freenoise=bool(_o.get('freenoise', plan['freenoise'])))
    print('[seam-baseline-hook] windows', _kw, flush=True)
    return forge.patch(model, window_frames=plan['window'], overlap_frames=plan['overlap'],
                       segment_seams='blend', **_kw, **handoff(forge))[0]
'''
    t.write_text(s.replace(old, new)); print("takes hooked")

p = d / "__init__.py"; s = p.read_text()
if "seam-baseline-hook" not in s:
    shutil.copy2(p, b / "__init__.py")
    old = "    guider = BasicGuider.execute(model, positive)[0]\n    noise = RandomNoise.execute(seed)[0]\n"
    assert s.count(old) == 1
    new = '''    # seam-baseline-hook: temporary test override (20261120)
    if not refine:
        import json as _json, os as _os
        _o = {}
        if _os.path.exists('/tmp/h3forge_seam.json'):
            with open('/tmp/h3forge_seam.json') as _f:
                _o = _json.load(_f)
        if _o.get('renoise_until') is not None:
            sampler = comfy.samplers.KSAMPLER(_seam_renoise_sampler, extra_options={'t_star': float(_o['renoise_until'])})
            print('[seam-baseline-hook] renoise sampler t_star', _o['renoise_until'], flush=True)
''' + old
    s = s.replace(old, new)
    fn = '''
def _seam_renoise_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, t_star=0.6):
    """seam-baseline-hook: full renoise of the clean estimate while sigma >= t_star, Euler below (FlowLong 4.2)."""
    import comfy.k_diffusion.sampling as _kd
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = _kd.default_noise_sampler(x, seed=extra_args.get('seed', None))
    model_sampling = model.inner_model.model_patcher.get_model_object('model_sampling')
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        nxt = sigmas[i + 1]
        if nxt <= 0:
            x = denoised
        elif float(sigmas[i]) >= t_star:
            x = model_sampling.noise_scaling(nxt, noise_sampler(sigmas[i], nxt), denoised)
        else:
            x = denoised + (x - denoised) * (nxt / sigmas[i])
    return x


def sample(model, positive, latent, seed, options, refine=False):
'''
    old2 = "\ndef sample(model, positive, latent, seed, options, refine=False):\n"
    assert s.count(old2) == 1
    p.write_text(s.replace(old2, fn)); print("init hooked")
import py_compile
py_compile.compile(str(t), doraise=True); py_compile.compile(str(p), doraise=True); print("compiles")

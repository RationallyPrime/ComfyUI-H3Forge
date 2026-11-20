# Temporary test hook (20261120): "sequential" hand-over. /tmp/h3forge_seam.json {"sequential": [85, 149]} makes sample()
# run the take once as usual, then once more per boundary with every latent before the boundary frozen (denoise mask 0),
# so the later window denoises against the finished overlap instead of the earlier window's current guess.
# Originals: /workspace/backups/seam-sweep-20261120/{takes.py,__init__.py}
import pathlib, py_compile
p = pathlib.Path("/workspace/runpod-slim/ComfyUI/custom_nodes/ComfyUI-MMH3-ClipLoops/__init__.py"); s = p.read_text()
assert "seam-baseline-hook" in s and pathlib.Path("/workspace/backups/seam-sweep-20261120/__init__.py").exists()
if "seam-sequential-hook" not in s:
    old = "    noise = RandomNoise.execute(seed)[0]\n    return SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent)[1]\n"
    assert s.count(old) == 1
    new = '''    noise = RandomNoise.execute(seed)[0]
    out = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent)[1]
    # seam-sequential-hook: temporary test override (20261120)
    if not refine and _o.get('sequential'):
        for _b in _o['sequential']:
            _v, _a = out['samples'].unbind()
            _vm = torch.ones_like(_v); _vm[:, :, :int(_b)] = 0
            _ab = int(round(_a.shape[-1] * int(_b) / _v.shape[2]))
            _am = torch.ones_like(_a); _am[..., :_ab] = 0
            _lat = dict(latent); _lat['samples'] = NestedTensor((_v, _a)); _lat['noise_mask'] = NestedTensor((_vm, _am))
            print('[seam-sequential-hook] freeze video latents <', int(_b), 'of', _v.shape[2], 'audio <', _ab, 'of', _a.shape[-1], flush=True)
            _keep = _v[:, :, :int(_b)].clone()
            out = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, _lat)[1]
            print('[seam-sequential-hook] frozen part changed by', float((out['samples'].unbind()[0][:, :, :int(_b)] - _keep).abs().max()), flush=True)
    return out
'''
    p.write_text(s.replace(old, new)); print("sequential hooked")
py_compile.compile(str(p), doraise=True); print("compiles")

# hook6 (20261120): "early hand-over" = sequential idea applied to the first k steps only.
# opts {"early": k, "sequential": [85, 149]}: run steps 0..k all together; for each bound b freeze the
# step-k picture (x0 estimate) of latents < b and redo steps 0..k for the rest; stitch at step k; run steps k..end once.
# Cost 30 + 2k steps instead of 90. Patches ClipLoops __init__.py sample(); needs hook5 already applied.
import re
P = '/workspace/runpod-slim/ComfyUI/custom_nodes/ComfyUI-MMH3-ClipLoops/__init__.py'
s = open(P).read()
assert '# seam-early-hook' not in s, 'already applied'
old = "    out = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent)[1]\n    # seam-sequential-hook: temporary test override (20261120)\n    if not refine and _o.get('sequential'):"
assert s.count(old) == 1
new = '''    # seam-early-hook: temporary test override (20261120)
    if not refine and _o.get('early') and _o.get('sequential'):
        from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise
        _k = int(_o['early']); _sa = sigmas[:_k + 1]; _sb = sigmas[_k:]
        _r = SamplerCustomAdvanced.execute(noise, guider, sampler, _sa, latent)
        _nv, _na = _r[0]['samples'].unbind(); _xv, _xa = _r[1]['samples'].unbind()
        print('[seam-early-hook] k', _k, 'sigma_k', float(sigmas[_k]), 'steps', len(sigmas) - 1, flush=True)
        for _b in _o['sequential']:
            _b = int(_b); _ab = int(round(_xa.shape[-1] * _b / _xv.shape[2]))
            _vm = torch.ones_like(_xv); _vm[:, :, :_b] = 0
            _am = torch.ones_like(_xa); _am[..., :_ab] = 0
            _lat = dict(latent); _lat['samples'] = NestedTensor((_xv, _xa)); _lat['noise_mask'] = NestedTensor((_vm, _am))
            _r2 = SamplerCustomAdvanced.execute(noise, guider, sampler, _sa, _lat)
            _nv2, _na2 = _r2[0]['samples'].unbind(); _xv2, _xa2 = _r2[1]['samples'].unbind()
            print('[seam-early-hook] bound', _b, 'later part moved by', float((_xv2[:, :, _b:] - _xv[:, :, _b:]).abs().mean()), flush=True)
            _nv = torch.cat([_nv[:, :, :_b], _nv2[:, :, _b:]], dim=2); _na = torch.cat([_na[..., :_ab], _na2[..., _ab:]], dim=-1)
            _xv = torch.cat([_xv[:, :, :_b], _xv2[:, :, _b:]], dim=2); _xa = torch.cat([_xa[..., :_ab], _xa2[..., _ab:]], dim=-1)
        _lat = dict(latent); _lat['samples'] = NestedTensor((_nv, _na)); _lat.pop('noise_mask', None)
        out = SamplerCustomAdvanced.execute(Noise_EmptyNoise(), guider, sampler, _sb, _lat)[1]
        print('[seam-early-hook] done', flush=True)
        return out
''' + old
open(P, 'w').write(s.replace(old, new))
import py_compile; py_compile.compile(P, doraise=True); print('hook6 applied')

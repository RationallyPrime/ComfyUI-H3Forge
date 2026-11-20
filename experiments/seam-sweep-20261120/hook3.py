import pathlib, shutil
t = pathlib.Path("/workspace/runpod-slim/ComfyUI/custom_nodes/ComfyUI-MMH3-ClipLoops/takes.py")
b = pathlib.Path("/workspace/backups/seam-sweep-20261120"); b.mkdir(parents=True, exist_ok=True)
s = t.read_text()
if "seam-sweep-hook" not in s:
    shutil.copy2(t, b / "takes.py")
    old = "    return {'handoff': 'lead', 'lead': 0.5}\n"
    assert s.count(old) == 1
    new = '''    # seam-sweep-hook: temporary test override (20261120); original in /workspace/backups/seam-sweep-20261120/takes.py
    import json as _json, os as _os
    if _os.path.exists('/tmp/h3forge_seam.json'):
        with open('/tmp/h3forge_seam.json') as _f:
            _o = _json.load(_f)
        print('[seam-sweep-hook]', _o, flush=True)
        return {'handoff': _o.get('handoff', 'lead'), 'lead': float(_o.get('lead', 0.5))}
''' + old
    t.write_text(s.replace(old, new)); print("hooked")
else: print("already hooked")

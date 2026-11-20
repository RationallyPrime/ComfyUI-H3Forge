# Keeps the temporary seam override file honest (20261120): while a render that is NOT one of the
# seam test jobs is running, the file holds the kit default; while a seam job runs it holds that job's opts.
# Usage: nohup python3 guard.py <queuefile> >> guard.log 2>&1 &
import json, sys, time, urllib.request
F = "/tmp/h3forge_seam.json"; DEF = {"handoff": "lead", "lead": 0.5}; last = None
while True:
    try:
        q = json.load(urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=10))["queue_running"]
        if q:
            rn = [n["inputs"]["run_name"] for n in q[0][2].values() if isinstance(n, dict) and "run_name" in n.get("inputs", {})]
            rn = rn[0] if rn else ""
            want = DEF
            if isinstance(rn, str) and rn.startswith("seam-"):
                want = next((e["opts"] for e in json.load(open(sys.argv[1])) if e["name"] == rn[5:]), None)
            if want is not None:
                try: cur = json.load(open(F))
                except Exception: cur = None
                if cur != want:
                    json.dump(want, open(F, "w")); print(time.strftime("%H:%M:%S"), "set", want, "for", rn, flush=True)
                elif rn != last: print(time.strftime("%H:%M:%S"), "ok", rn, flush=True)
                last = rn
    except Exception as ex:
        print(time.strftime("%H:%M:%S"), "err", ex, flush=True)
    time.sleep(1)

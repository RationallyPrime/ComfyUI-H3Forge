"""Seam sweep driver (20261120). Queue file entries: {name, api, opts, seed}. Log: /workspace/seam-gpu/driver.log"""
import json, sys, time, urllib.request, urllib.error, pathlib
BASE = "http://127.0.0.1:8188"
HERE = pathlib.Path("/workspace/seam-gpu")
OPTS = pathlib.Path("/tmp/h3forge_seam.json")
def log(*a):
    line = time.strftime("%H:%M:%S ", time.gmtime()) + " ".join(str(x) for x in a)
    with open(HERE / "driver.log", "a") as f: f.write(line + "\n")
    print(line, flush=True)
def get(path): return json.load(urllib.request.urlopen(BASE + path, timeout=20))
def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=60).read()
def wait_idle():
    while True:
        try:
            q = get("/queue")
            if not q["queue_running"] and not q["queue_pending"]: return
        except Exception: pass
        time.sleep(15)
def run(e):
    name = e["name"]
    wait_idle()
    OPTS.write_text(json.dumps(e["opts"]))
    p = json.load(open(HERE / e["api"])); p = p.get("prompt", p)
    p["15"]["inputs"]["run_name"] = "seam-" + name
    if e.get("seed") is not None: p["29"]["inputs"]["value"] = int(e["seed"])
    t0 = time.time()
    try: pid = json.loads(post("/prompt", {"prompt": p}))["prompt_id"]
    except urllib.error.HTTPError as err:
        log("REJECTED", name, err.read()[:600]); return
    log("queued", name, "opts", json.dumps(e["opts"]), "seed", e.get("seed"), pid)
    while True:
        time.sleep(20)
        try: h = get("/history/" + pid)
        except Exception: continue
        if pid in h:
            st = h[pid].get("status", {})
            log("finished", name, st.get("status_str"), "secs", int(time.time() - t0))
            if st.get("status_str") == "success": (HERE / "done" / name).write_text(str(int(time.time() - t0)))
            else: log("  messages", json.dumps(st.get("messages", [])[-1:])[:800])
            return
(HERE / "done").mkdir(exist_ok=True); (HERE / "failed").mkdir(exist_ok=True)
while True:
    todo = [e for e in json.load(open(sys.argv[1])) if not (HERE / "done" / e["name"]).exists() and not (HERE / "failed" / e["name"]).exists()]
    if not todo: break
    e = todo[0]
    try: run(e)
    except Exception as exc: log("ERROR", e["name"], repr(exc)[:300])
    if not (HERE / "done" / e["name"]).exists(): (HERE / "failed" / e["name"]).write_text("")
OPTS.unlink(missing_ok=True)
log("ALL DONE (override file removed; hook stays until takes.py is restored)")

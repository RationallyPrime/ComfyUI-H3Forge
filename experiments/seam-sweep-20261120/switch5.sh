#!/bin/bash
# switch5.sh <queuefile> <marker> <takename>: at the job boundary after done/<marker>, apply hook5, queue t-<takename>-seq first, reboot ComfyUI, restart the driver.
cd /workspace/seam-gpu
PY=/workspace/runpod-slim/ComfyUI/.venv-cu128/bin/python
L=/workspace/seam-gpu/switch5.log; Q=$1
echo "$(date -u +%T) waiting for done/$2" >> $L
while [ ! -e done/$2 ]; do sleep 5; done
pkill -f "driver5.py" ; sleep 1
curl -s -X POST localhost:8188/interrupt >/dev/null
curl -s -X POST localhost:8188/queue -H 'Content-Type: application/json' -d '{"clear":true}' >/dev/null
sleep 5
rm -f /tmp/h3forge_seam.json
$PY hook5.py >> $L 2>&1 || { echo "HOOK FAILED, restarting driver unchanged" >> $L; nohup $PY driver5.py $Q >> driver.out 2>&1 & exit 1; }
$PY - "$Q" "$3" >> $L 2>&1 <<'P'
import json, sys
qf, take = sys.argv[1], sys.argv[2]
q = json.load(open(qf)); n = f"t-{take}-seq"
if not any(e["name"] == n for e in q):
    q.insert(0, {"name": n, "api": f"take-{take}.json", "opts": {"handoff": "lead", "lead": 0.5, "sequential": [85, 149]}, "seed": 42})
    json.dump(q, open(qf, "w"), indent=1)
print("queue head", q[0]["name"], len(q))
P
curl -s -X POST localhost:8188/v2/manager/reboot >/dev/null
echo "$(date -u +%T) reboot sent" >> $L
sleep 20
until curl -s -m 5 localhost:8188/system_stats >/dev/null; do sleep 5; done
echo "$(date -u +%T) comfy back" >> $L
sleep 10
nohup $PY driver5.py $Q >> driver.out 2>&1 &
echo "$(date -u +%T) driver restarted" >> $L

#!/bin/bash
# switch6.sh <queuefile> <marker> <takename> <k>: after done/<marker>, apply hook6, queue t-<takename>-early<k> first, reboot ComfyUI, restart the driver.
cd /workspace/seam-gpu
PY=/workspace/runpod-slim/ComfyUI/.venv-cu128/bin/python
L=/workspace/seam-gpu/switch6.log; Q=$1
echo "$(date -u +%T) waiting for done/$2" >> $L
while [ ! -e done/$2 ]; do sleep 5; done
pkill -f "driver5.py" ; sleep 1
curl -s -X POST localhost:8188/interrupt >/dev/null
curl -s -X POST localhost:8188/queue -H 'Content-Type: application/json' -d '{"clear":true}' >/dev/null
sleep 5
cp /workspace/runpod-slim/ComfyUI/custom_nodes/ComfyUI-MMH3-ClipLoops/__init__.py /workspace/backups/seam-sweep-20261120/__init__.py.hook5
$PY hook6.py >> $L 2>&1 || { echo "HOOK FAILED, restarting driver unchanged" >> $L; cp /workspace/backups/seam-sweep-20261120/__init__.py.hook5 /workspace/runpod-slim/ComfyUI/custom_nodes/ComfyUI-MMH3-ClipLoops/__init__.py; nohup $PY driver5.py $Q >> driver.out 2>&1 & exit 1; }
$PY - "$Q" "$3" "$4" >> $L 2>&1 <<'P'
import json, sys
qf, take, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
q = json.load(open(qf)); n = f"t-{take}-early{k}"
if not any(e["name"] == n for e in q):
    q.insert(0, {"name": n, "api": f"take-{take}.json", "opts": {"handoff": "lead", "lead": 0.5, "early": k, "sequential": [85, 149]}, "seed": 42})
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

#!/bin/bash
# Applies hook4 at a job boundary: waits for marker $1, stops the driver, reboots ComfyUI, adds baseline canaries, restarts the driver.
cd /workspace/seam-gpu
PY=/workspace/runpod-slim/ComfyUI/.venv-cu128/bin/python
L=/workspace/seam-gpu/switch4.log
echo "$(date -u +%T) waiting for done/$1" >> $L
while [ ! -e done/$1 ]; do sleep 5; done
pkill -f "driver5.py" ; sleep 1
curl -s -X POST localhost:8188/interrupt >/dev/null
curl -s -X POST localhost:8188/queue -H 'Content-Type: application/json' -d '{"clear":true}' >/dev/null
sleep 5
rm -f /tmp/h3forge_seam.json
$PY hook4.py >> $L 2>&1 || { echo "HOOK FAILED, restarting driver unchanged" >> $L; nohup $PY driver5.py queue-sweep.json >> driver.out 2>&1 & exit 1; }
$PY - >> $L 2>&1 <<'E'
import json
q = json.load(open('queue-sweep.json'))
if not any(e['name'].startswith('bl-') for e in q):
    api = 'battle-brief-baseline-api.json'
    fl = {"handoff": "off", "blend": "overlap-linear", "stagger": False, "freenoise": False}
    new = [{"name": "bl-bb-flowlong-t06-s42", "api": api, "opts": {**fl, "renoise_until": 0.6}, "seed": 42},
           {"name": "bl-bb-flat-s42", "api": api, "opts": {"handoff": "off", "blend": "flat", "stagger": False, "freenoise": False}, "seed": 42},
           {"name": "bl-bb-flowlong-t08-s42", "api": api, "opts": {**fl, "renoise_until": 0.8}, "seed": 42}]
    json.dump(new + q, open('queue-sweep.json', 'w'), indent=1); print('queue now', len(new + q))
E
curl -s -X POST localhost:8188/v2/manager/reboot >/dev/null
echo "$(date -u +%T) reboot sent" >> $L
sleep 20
until curl -s -m 5 localhost:8188/system_stats >/dev/null; do sleep 5; done
echo "$(date -u +%T) comfy back" >> $L
sleep 10
nohup $PY driver5.py queue-sweep.json >> driver.out 2>&1 &
echo "$(date -u +%T) driver restarted" >> $L

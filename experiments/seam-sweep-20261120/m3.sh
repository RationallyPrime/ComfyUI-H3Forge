cd /workspace/seam-gpu; tail -5 driver.log | cut -c1-150; O=/workspace/runpod-slim/ComfyUI/output/MMH3_sequences
for d in "$@"; do [ -f $O/seam-${d}_001/shot_0001.mp4 ] && { echo "### $d"; /workspace/runpod-slim/ComfyUI/.venv-cu128/bin/python seam_metrics.py $O/seam-${d}_001/shot_0001.mp4 2>&1 | grep -v "per-second\|frames"; }; done

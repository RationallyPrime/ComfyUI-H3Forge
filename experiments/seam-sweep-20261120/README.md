# Seam sweep experiments, 2026-11-20

Not part of H3Forge. These are the throwaway scripts used on two rented pods to compare
window hand-over methods on one take long takes, kept here so they are not lost when the
pods go. Nothing in this folder is imported by the node pack.

The hooks patch the ClipLoops node pack in place on a pod
(`custom_nodes/ComfyUI-MMH3-ClipLoops/takes.py` and `__init__.py`); each asserts the text it
replaces, so it fails loudly on a different version. Back the two files up first.

| File | What it does |
|---|---|
| `hook3.py`, `hook4.py` | Read `/tmp/h3forge_seam.json` (`handoff`, `lead`, `blend`, `stagger`, `freenoise`, `renoise_until`) and pass it to the H3Forge context windows, so one graph can render as handoff, plain blend, a FlowLong-style renoise, or flat averaging. |
| `hook5.py` | Sequential hand-over: render the take, then per window bound freeze everything before it and redo the rest. Clean on moving subjects, about 3x the render time. |
| `hook6.py` | The same idea for the first `k` steps only (`{"early": k, "sequential": [bounds]}`). `k=16` of 30 was clean on one take at 2.2x; `k=8` was not. Needs `hook5.py`. |
| `switch4.sh`, `switch5.sh`, `switch6.sh` | Apply a hook and restart ComfyUI. |
| `driver5.py`, `guard.py` | Queue runner (one job at a time, writes the override file per job) and a guard that keeps the override at the kit default for other jobs. |
| `seam_metrics.py`, `m3.sh` | Overlap sharpness, flicker and largest frame-to-frame difference per take. |
| `p11.py`, `p12.py`, `p21.py` | The prompt texts that removed the hand-over jump without any of the above. |

## What was learned

With one-way handoff (lead 0.5, the default since #27) a take can still jump by one frame at
the hand-over when the subject travels. Each window renders its beat as if it were a
standalone clip, so the jump is the difference between where the earlier beat ends and where
the later beat would naturally begin.

The fix that costs nothing is in the prompt:

1. The setting block is identical in every beat.
2. Each beat ends on a held pose.
3. Each beat's first sentences state the pose, the place and the size in the frame that the
   previous beat ends on, ideally as the same passage word for word.

That was clean at both hand-overs on three 3 x 9 s takes and on a 2 x 15 s take with two
subjects (`p21.py`). Sequential and early-step hand-over also work, at 2.2 to 3 times the
render time, and were not taken further.

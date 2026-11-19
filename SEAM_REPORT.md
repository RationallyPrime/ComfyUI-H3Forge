# Measure disagreement between adjacent H3 context windows

This experiment adds a video overlap probe to both Forge context nodes. It captures the velocity each window predicted for its shared rows, then compares both with the velocity the wrapper actually returned. This includes all contributors when three windows cover a latent. It uses the current staggered plan and reuses the existing model calls.

## Running the probe

In the cockpit's `takes.py`, `windowed()` must pass `overlap_probe="<directory for this take>"` to the Forge context node. Use a fresh directory per take because `probe.jsonl` is append-only. An empty directory string disables the probe by default. The optional `overlap_probe_steps` string defaults to `0,3,7,15,29`. An empty step string records metrics only. The step-zero context plan prints the directory and selected steps.

Each JSON line describes one adjacent window pair in one model evaluation. Window ordinals and latent indices are zero-based. The recorded `overlap_range` is the half-open intersection of the input windows, including shared context that a prompt's output ownership rules may exclude from fusion. The first entry in `per_latent_rel_l2` is the earliest shared latent and the last entry is the latest.

All ratios use the L2 norm of the velocity difference divided by the L2 norm of the pair's arithmetic mean. Identical inputs give zero. A nonzero difference with a zero denominator gives JSON `null`. The spatial pyramid averages 2×2 blocks twice, independently per latent and channel. The high and mid bands are the residuals after expanding each block mean back over its footprint. A selected-step record links to a safetensors file containing `x0_a`, `x0_b` and `x0_fused` as float32 tensors in [B, C, T, H, W] order. They use `x - sigma * v` at the diffusion wrapper, before ComfyUI applies an outer denoise mask.

Read the measurements with the report tool.

```sh
python3 tools/probe_report.py <directory-for-this-take>/probe.jsonl
```

The tool follows each pair by ordinal as its range moves. It reports the first measured step whose low-band ratio equals or exceeds the previous measured step's. Repeated evaluations at the same step are averaged, with the count shown.

## CPU measurements

The two-window fake model predicted velocities [1, 1, 1] and [1, 3, 5] over global latents [3, 6). The expected relative L2 was √(20/14). The report returned the following values.

| Step | Low | Mid | High | Overall |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.19523 | 0.00000 | 0.00000 | 1.19523 |
| 1 | 1.19523 | 0.00000 | 0.00000 | 1.19523 |

The first non-decrease was step 1. Per-latent disagreement rose from 0 to 1 to 1.33333, matching the injected difference at the end of the earlier window. Relative L2 to the returned fused velocity was 1.02598 for window A and 0.26968 for window B. The test read back all three step-zero x0 tensors at sigma 0.75 and checked their values. Both modes made two model calls per evaluation and returned byte-identical video and audio. A second fixture moved the first overlap from [3, 10) to [4, 10) and checked the fused comparison in rows shared by three windows.

| Check | Result |
| --- | --- |
| Untouched base `05792b7` | 227 passed, 11 failed |
| Branch with probe | 231 passed, the same 11 test IDs failed |
| Focused probe, context node and connected timeline tests | 10 passed |
| Direct comparison of probe-off with the base wrapper | Byte-identical video and audio in 12 seeded CPU evaluations across float32 and bfloat16, two steps and three plans |
| Ruff 0.12.12 | Passed for `h3forge tests __init__.py` and `tools/probe_report.py` |

An independent local review of implementation `a2fbf06` found no material issues and reran all four probe-specific tests successfully.

The inherited failures are two native-control cases with an unresolved `comfy` name and nine NAG cases with an unresolved `ComfyAttention` name in the local native-source harness.

## Render evidence and cost

**no GPU receipt**. Neither Battle Brief nor sparring hall was rendered. The synthetic inputs establish that the probe can locate an injected difference. Whether either real take disagrees mainly in the low band early or the high band late, and at which end of the overlap, remains unmeasured. There are no new flicker, sharpness, frame-difference, motion-continuity or render-time measurements.

Enabling the probe adds overlap copies, two-level spatial reductions, a transfer of scalar metrics and one JSON line per adjacent overlap per model evaluation. Selected steps also transfer and write three float32 overlap tensors, for 12 bytes per overlap tensor element across the three snapshots, plus the file header. GPU memory and elapsed-time overhead have not been measured.

# Blackwell test matrix

Use the same seed, prompt, input conditioning, resolution, frame count, sampler, steps, CFG/guidance, and model files for every row.

For pipe-timeline tests, keep the exact segment count and delimiter placement fixed. Each segment is an independent encoding and occupies an equal portion of the target video timeline.

| ID | Sparse | FETA | Context | Purpose |
|---|---|---|---|---|
| B0 | off | off | off | Native baseline |
| B1 | dense patch | off | off | Patch equivalence |
| A1 | flex | off | off | Sparse topology |
| F1 | dense | on | off | FETA isolation |
| C1 | off | off | on | Window isolation |
| AF | flex | on | off | Sparse + FETA |
| AC | flex | off | on | Sparse + windows |
| ALL | flex | on | on | Full stack |

For each run record:

- first-run wall time;
- second-run wall time (compile amortized);
- peak allocated/reserved VRAM;
- H3Forge sparse/dense/FETA call counts;
- block-mask cache hit/miss/eviction counts;
- output hash if deterministic enough to be useful;
- visual seam score (0–5);
- identity drift score (0–5);
- motion quality score (0–5);
- lip-sync / AV-sync score (0–5);
- audio intelligibility score (0–5);
- notes and failure trace.

## Escalation order

1. Prove B1 against B0.
2. Prove A1 can complete with `strict=true`.
3. Compare A1 quality and steady-state speed to B0.
4. Prove F1 independently.
5. Prove C1 independently at a length that actually invokes >1 window.
6. Run AC before ALL.
7. Introduce FETA last.

## Sparse sweep

Once A1 is sane, useful first sweep:

| temporal_window | spatial_radius | bridge_stride |
|---:|---:|---:|
| 20 (0.5 s) | 6 | 20 |
| 40 (1.0 s) | 8 | 40 |
| 60 (1.5 s) | 8 | 40 |
| 80 (2.0 s) | 12 | 40 |
| 80 (2.0 s) | 12 | 80 |

Track both route quality and actual speed. A mask that is theoretically sparse but compiles into an inefficient block pattern is not a win.

H3 timeline units are 40 Hz. Do not read `temporal_window` or `bridge_stride` as decoded-video frame counts.

## Context sweep

Keep overlap around 20–30% initially:

| window | overlap |
|---:|---:|
| 17 | 4 |
| 25 | 5 |
| 33 | 8 |
| 41 | 10 |

The best point will depend heavily on target resolution because each video latent frame contains all spatial patch rows.

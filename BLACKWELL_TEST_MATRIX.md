# Blackwell test matrix

Use the same seed, prompt, input conditioning, resolution, frame count, sampler, steps, CFG/guidance, and model files for every row.

For pipe-timeline tests, keep the exact segment count, delimiter placement, `global_prompt`, and `segment_durations` fixed. Each segment is an independent encoding and occupies its requested relative portion of the target video timeline (equal portions when durations are empty); each beat gets exclusive output ownership at its reported native-grid frame cuts; neighboring video is available as context, and all windows see the complete shared audio timeline. Retain the step-zero context-plan line so the actual window count, stagger phase/state, stride, minimum adjacent overlap, blend mode, latent-visit ratio, and prompt assignment are part of the receipt.

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

Sweep overlap around 20–32% initially; `25 / 8` is the current node default:

| window | overlap | note |
|---:|---:|---|
| 17 | 4 | lower-window comparison |
| 25 | 5 | retained pre-parity comparison |
| 25 | 8 | current default |
| 33 | 8 | larger-window comparison |
| 41 | 10 | larger-window comparison |

The best point will depend heavily on target resolution because each video latent frame contains all spatial patch rows.

## NAG acceptance matrix

Use the same fixed-seed workflow as the runs above. All NAG runs use `BasicGuider` at CFG 1 unless stated otherwise.

| ID | Guidance | Sparse | Purpose |
|---|---|---|---|
| N-A | BasicGuider CFG 1 | off | Native reference |
| N-B | NAG-Lite | off | Isolate guidance |
| N-C | NAG-Lite | on | Composition with H3Forge sparse |
| N-D | Shared-softmax NAG (`mode=faithful_selective`) | off/on | Compare denominator choice with the same attention topology |
| N-E | Ordinary CFG | off | Cost/quality reference |

Starting values: `nag_scale 3.0`, `nag_tau 2.5`, `nag_alpha 0.15`, `nag_sigma_end 0.70`, blocks `8–28`, `video_strength 1.0`, `audio_strength 0.5`, `strict true`. Find stable tau/alpha first, then tune only the scale.

Negative prompts should separately test:

- visual object suppression;
- style suppression;
- quality defects (blur, malformed hands);
- camera/static-motion suppression;
- music suppression;
- speech or voice-property suppression;
- environmental audio suppression.

For each run record wall time, peak VRAM, attention-call counts (`nag=` in the run summary), voice identity, lip sync, event synchronization, and fixed-seed visual differences against N-A.

Performance target (a target, not a prediction): NAG-Lite under 20% overhead versus N-A. Selective faithful NAG will land materially higher — H3 is a packed single-stream transformer, structurally closer to Flux's expensive NAG case than to Wan's cheap one.

## KJNodes composition checks

- Add `MiniMax H3 Token Counter` to every diagnostic workflow and record the packed token count per run.
- `MiniMax H3 Chunk FeedForward` is expected to be output-identical; verify once against B0 with fixed seeds, then leave it on.
- `MiniMax H3 Low VRAM Attention` equivalence gate: run `KJ Low VRAM Attention + H3Forge sparse + FETA disabled` against A1 before allowing the combination into larger stacks. KJ head grouping invokes the attention override per head group, which is only guaranteed consistent with FETA off.
- Do not include FirstBlockCache or Spectrum in any H3Forge/NAG proving run; benchmark them in isolation later.

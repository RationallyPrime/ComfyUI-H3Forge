# Re-predict H3 video windows from a shared overlap estimate

This experiment tests whether showing both windows a cleaner version of their fused overlap helps them agree on the surrounding frames. Both context nodes expose `consensus`, off by default. The local CPU tests establish the exchange, output selection and extra-call limit. **no GPU receipt**. The effect on either take is unmeasured.

## What changes during sampling

Above `consensus_until`, the normal first pass saves each writer's video velocity only on co-written rows. For each contiguous overlap between a pair of windows, Forge computes `d = ||v_a - v_b|| / ||v_fused||` using the actual staggered plan and its write intervals. An exact agreement scores zero, including when both predictions are zero. Opposing predictions that cancel in the fused estimate score infinity.

When a nonzero score reaches the threshold, each window touching the flagged span receives the same pass-one estimate at lower noise. For a row with original mask strength `m`, its sigma goes from `m * sigma` to `lead * m * sigma`. The exchanged input is `x + (lead - 1) * m * sigma * v_fused`, the flow interpolation between the estimated clean and noise endpoints. The pass-two mask is `lead * m` there. The local ComfyUI core at `c5751c8` takes the maximum mask strength over each 2×2 patch in `mask_row_values`, uses it for the packed row's timestep in `MiniMaxH3Model._forward`, and caps the clean label at the condition timestep. The outer `forward` applies the caller's original velocity mask after Forge returns.

Flagged rows keep their pass-one velocity. Elsewhere, each window that ran again contributes its new prediction with its original blend weight. Every audio contribution comes from pass one.

The sampler banks credit from completed first-pass window groups. A step re-predicts the whole set of readers of flagged rows only when the cumulative fraction covers all of them. On a three-window run with every overlap flagged and the default 40% cap, the first full exchange can occur on the third forward. This keeps the last window eligible even when both seams disagree repeatedly. The step-zero log gives the settings, eligible forwards log their overlap scores and cumulative call counts, and the sampler prints final totals. Measuring disagreement still costs overlap storage and reductions on a quiet step.

## Cockpit invocation

In ComfyUI-MMH3-ClipLoops, `takes.py`'s `windowed()` must pass `consensus=True` to its existing `H3ForgeContextWindows.patch(...)` call for the experimental arm. Omitting the argument selects the control. The other four optional arguments default to `consensus_threshold=0.15`, `consensus_lead=0.6`, `consensus_until=0.5`, and `consensus_max_fraction=0.4`.

## CPU measurements

| Check | Result |
|---|---|
| Default and explicit off | Staggered pyramid fusion with windowed audio returned the exact video and audio bytes recorded separately from main at `05792b7` for the bfloat16 fixture. |
| Agreement and cutoff | Identical predictions, sub-threshold disagreement, sigma at or below the cutoff, and a zero budget each produced 3 first-pass calls and 0 re-predictions. |
| Selective exchange | At sigma 0.8 the two test overlaps scored 0.66667 and 0. With the test budget at 1.0, the first two of three windows ran again. Their shared full-strength rows moved to sigma 0.48. The test also checked fractional and pinned input masks, quiet-overlap weighting, and exact pass-one audio. |
| Default call budget | Across six forwards, observed extra calls per forward were 0, 0, 3, 0, 3, 0. The run used 6 extra calls for 18 first-pass calls (33.3%). A second invocation of the same sampler started fresh. |
| Independent worker check | With the installed core's option-merging code, one persistent worker used 6 extra calls for 15 first-pass calls. Two concurrent workers used 12 for 30. Every window participated equally in each check. |

The 14 focused tests also cover three-way overlaps and option propagation through both nodes. Ruff 0.12.12 passed `ruff check h3forge tests __init__.py`. With `H3FORGE_COMFY_SOURCE=~/Projects/comfyui-local`, untouched main had **227 passed, 11 failed**. The implementation had **241 passed, 11 failed**, with identical failing test IDs. Two ControlNet cases encounter an unbound `comfy` in the source-extraction harness and nine NAG cases encounter an unbound `ComfyAttention`.

## Render measurements outstanding

| Take | Supplied baseline overlap flicker | Supplied baseline overlap sharpness | Consensus render |
|---|---|---|---|
| Battle Brief | 0.49 / 0.29 | 424 / 447 | Unmeasured |
| Sparring hall | 0.40 / 0.34 | 642 / 669 | Unmeasured |

Neither take has experimental per-step disagreement, pass-two counts, elapsed render time or frame metrics. The Battle Brief flicker target of 0.10 in both overlaps, overlap sharpness relative to neighbouring solo spans, solo-span drift, peak frame differences, the framing dissolve, and Sparring hall lunge continuity all remain open. The measured CPU call fractions describe synthetic window predictions. Wall-clock overhead against the supplied 10–12 minute render baseline is unknown.

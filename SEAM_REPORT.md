# Brief 06: Spatial Band-Split Fusion for Video Context Overlaps

Branch: `seam/06-band-split-fusion`

## Summary

Adds optional spatial Laplacian-pyramid band-split fusion for overlapping video context windows in `h3forge/context.py`, exposed as optional inputs on both `H3ForgeContextWindows` and `H3ForgeTimelineContextWindows` (`h3forge/nodes.py`).

Instead of averaging every spatial frequency across the full overlap width, overlapping video predictions (`v_out`) are decomposed per latent frame and channel into `band_fusion` spatial bands over latent `(H, W)`. The lowest (coarse) band keeps the existing blend weights (`pyramid` by default) unchanged, while higher (detail) bands switch over progressively narrower transitions down to `band_crossfade` latents centred on the overlap middle.

- **No extra model calls**: window execution is unchanged.
- **Video overlaps only**: solo video latents and the entire audio stream are untouched (audio is bit-identical whether band fusion is on or off).
- **Default off (`band_fusion=0`)**: executes the original overlap-add path verbatim and is bit-identical to `main`.

## Design

### 1. Spatial Laplacian pyramid (`laplacian_pyramid`, `_separable_blur_2d`)
- Uses a fixed separable 5-tap Burt-Adelson binomial blur `[1, 4, 6, 4, 1] / 16` along `W` then `H` with `replicate` padding (safe for arbitrary odd or tiny `(H, W)` down to `1x1`) and à trous dilation `2**level` across pyramid levels.
- Every band stays at full latent `(H, W)` in `float32`, ordered from coarsest (`bands[0]`) to finest (`bands[levels - 1]`), with exact sum reconstruction (`sum(bands) == x.float()`).
- Spatially constant fields stay entirely in `bands[0]`; Nyquist detail is isolated in `bands[-1]`.

### 2. Per-band transition weights (`band_overlap_weights`, `fuse_band_overlap`)
- **Band `0` (lowest)**: normalized existing blend weights (`pyramid`, `overlap-linear`, or `flat`), unchanged.
- **Band `levels - 1` (highest)**: linear crossfade of width `band_crossfade` latents centred on the overlap midpoint (`0` = hard switch).
- **Intermediate bands (`0 < b < levels - 1`)**: linear crossfade whose width linearly interpolates between the full overlap length and `band_crossfade` by `b / (levels - 1)`.
- Weights are guaranteed non-negative and sum to `1.0` on every overlap latent for every band (including 3+-way overlaps when `overlap > window / 2` and carved-out rescued pipe-prompt beats).
- Overlap slices are buffered per contiguous overlap segment and freed immediately once that segment's owners finish, keeping peak extra memory to the active overlap (~5 MB at 544x960) rather than the full clip.

### 3. Adaptive disagreement variant (`band_adaptive`)
- For bands `b > 0`, measures per-latent, per-pixel local disagreement between overlapping windows using a 3x3 replicate-padded spatial average of channel-mean `|B_a - B_b|` normalized by local band magnitude `|B_a| + |B_b|` (scale-invariant in `[0, 1]`).
- Maps the ratio through a smoothstep gate between `0.15` and `0.50`: where windows agree (`<= 0.15`), keeps the plain low-band blend weights; where they disagree (`>= 0.50`), applies the narrow band transition. Effective weights `(1 - g) * w_plain + g * w_narrow` sum to `1.0` at every spatial location and latent.

### 4. High-band stagger hold (`band_stagger_hold`)
- When `stagger=True` and `band_stagger_hold=True` (default), once `sigma <= 0.8` (with `1e-6` float32 tolerance so `torch.tensor([0.8])` triggers), high-band transitions lock to the phase-invariant midpoint `0.5 * (starts_max[i + 1] + starts_0[i] + window)` and reference width `starts_0[i] + window - starts_max[i + 1]`.
- Because `window_starts` is non-decreasing in phase and `max_phase <= overlap`, `starts_max[i + 1] <= starts_0[i] + window` always holds, so the held midpoint lies inside every stagger phase's overlap.
- This holds the high-band owner fixed across steps while the windows themselves still stagger for the low band and audio (no extra model calls, audio untouched). Note: because the windows still move for the low band, each window's temporal context still shifts slightly by `phase`, even though the high-band owner index per latent does not flip across steps. Set `band_stagger_hold=False` to test without the hold.

## Options & Cockpit Integration

Added to `ContextPolicy` (`h3forge/context.py`) and `_context_optional_inputs()` (`h3forge/nodes.py`, optional on both `H3ForgeContextWindows` and `H3ForgeTimelineContextWindows` so existing API graphs remain valid):

| Option | Type | Default | Meaning |
|---|---|---|---|
| `band_fusion` | `INT` | `0` | `0` = off; `2` or `3` (any int `>= 2`) = Laplacian pyramid levels |
| `band_crossfade` | `INT` | `2` | Highest-band transition width in video latents (`0` = hard switch) |
| `band_adaptive` | `BOOLEAN` | `False` | Gate narrow high-band transitions by local spatial disagreement |
| `band_stagger_hold` | `BOOLEAN` | `True` | Hold high-band transition midpoint once `sigma <= 0.8` when `stagger=True` |

### Step-0 log receipt
The existing step-0 `[H3Forge] context plan ...` log line now always appends band-fusion status:
- Off: `band_fusion=off`
- Single-window clip: `band_fusion=<N>(inactive:single-window)`
- Active: `band_fusion=<N> band_crossfade=<K> band_adaptive=<on|off> band_stagger_hold=<on(sigma<=0.8)|off> band_overlap_latents=<M>`

### What the cockpit must pass
In `ComfyUI-MMH3-ClipLoops` (`takes.py`, `windowed()`), pass the optional inputs to the context node (`H3ForgeContextWindows` / `H3ForgeTimelineContextWindows`):
- `band_fusion`: `2` or `3` (default `0`)
- `band_crossfade`: `2` (default `2`)
- `band_adaptive`: `False` or `True` (default `False`)
- `band_stagger_hold`: `True` or `False` (default `True`)

## What Was Measured

- **Unit tests** (`H3FORGE_COMFY_SOURCE=~/Projects/comfyui-local .venv/bin/python -m pytest -q`):
  - Untouched `main` baseline in this environment: `227 passed, 11 failed` (pre-existing incompatibility between the newer `~/Projects/comfyui-local` source and AST-extracted `definitions(...)` tests: 2 in `tests/test_audit_repairs.py` due to `comfy.model_prefetch`, 9 in `tests/test_nag.py` due to `ComfyAttention`).
  - With this branch: **`445 passed, 11 failed`** (+218 new test cases passing, exact same 11 pre-existing failures, zero regressions).
  - Targeted test files (`tests/test_context_math.py`, `tests/test_nodes.py`): **`342 passed, 0 failed`**.
  - New tests cover:
    - Exact Laplacian pyramid reconstruction across levels `1..4` and shapes `(1,1)` to `(34,60)`, constant preservation, and Nyquist separation.
    - Per-band overlap weights summing to `1.0` on every overlap latent across `pyramid` / `overlap-linear` / `flat`, 2-way and 3-way overlaps, and width narrowing across bands.
    - Bit-identical output (`torch.equal`) at `band_fusion=0` in `float32` and `bfloat16` against plain overlap-add, regardless of auxiliary band options.
    - Synthetic two-window tests (low bands agree, high bands differ) proving the fused high band and frame match window A / window B away from the crossfade while solo video and audio stay bit-identical.
    - Synthetic mixed low+high disagreement tests proving the low band keeps the plain blend while high-frequency amplitude is preserved.
    - Adaptive gating tests proving agreeing spatial regions stay on plain blending, disagreeing regions use the narrow switch, and the gate is scale-invariant.
    - Stagger hold tests on the baseline `202/85/21` geometry proving high-band ownership locks at `sigma <= 0.8` (including `float32` `0.8`) while the low band and audio continue staggering, and moves when `band_stagger_hold=False` or `sigma > 0.8`.
    - Segmented pipe-prompt (`blend` with rescued short beats and `exclusive`) preservation and step-0 `[H3Forge]` logging.
    - Node `INPUT_TYPES`, optional placement, forwarding, and validation on both context nodes.
- **Lint & bytecode compilation**:
  - `ruff check --select E4,E7,E9,F --line-length 120 h3forge tests __init__.py .github/scripts`: passed (0 errors).
  - Strict `ruff check` on all new/modified lines: 0 warnings introduced.
  - `python3 -m compileall -q h3forge tests __init__.py`: passed.

## What Was Not Measured

**No GPU receipt.** No GPU pod was attached to this session. The following GPU runs remain to be executed on Battle Brief and Sparring Hall (seed 42, 544x960, euler/simple, 30 steps, shift 12 and 3, measured with `tools/seam_metrics.py`):
1. `band_fusion=2, band_crossfade=2, band_adaptive=False, band_stagger_hold=True`
2. `band_fusion=3, band_crossfade=2, band_adaptive=False, band_stagger_hold=True`
3. `band_fusion=2, band_crossfade=2, band_adaptive=True, band_stagger_hold=True`
4. `band_fusion=3, band_crossfade=2, band_adaptive=True, band_stagger_hold=True`
5. Ablation without stagger hold: `band_fusion=2, band_crossfade=2, band_adaptive=False, band_stagger_hold=False`

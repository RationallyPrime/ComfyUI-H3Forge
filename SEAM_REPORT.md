# Seam Experiment 03: Positions and Cadence (`stagger_cadence`, `position_mode`)

Branch: `seam/03-positions-cadence` (branched from `main` at `05792b7`)

## Summary

Adds two independent, opt-in options to `ContextPolicy` and both context nodes (`H3ForgeContextWindows` and `H3ForgeTimelineContextWindows`, via `_context_optional_inputs`) to test whether overlap softness/flicker comes from off-cadence staggered window starts or extrapolated RoPE time positions past H3's 107-latent trained range. Neither option adds model calls.

With defaults (`stagger_cadence=False`, `position_mode="global"`), behavior and outputs are bit-identical to `main`.

## Core `position_ids` Time Coordinate Unit (verified in `comfy/ldm/minimax/model.py`)

In `PackedLayout.__init__`:

- **Time coordinate unit**: **1.0 = 1 audio latent frame at 40 Hz (25 ms)** (`FRAME_RESCALE = 5.0 / 3.0` audio ticks per 24 fps video pixel frame).
- **Prefix layout**: text tokens occupy `[0, text_len)`, optional reference blocks occupy `[text_len, origin)` where `origin = float(text_len) + sum(_ref_t_span(blk))`, and target audio and target video both start at `origin`.
- **Target audio** (`_audio_grid`): `t = origin + arange(audio_t)` duplicated across left/right stereo channels (step `1.0` tick per audio latent).
- **Target video** (`_video_grid`, `_video_t_spans`, `FRAME_PER_TOKEN = (1, 4, 4, 4, 4)`): each 5-latent / 17-pixel-frame VAE cycle spans `17 * (5 / 3) = 85 / 3 ≈ 28.3333` ticks (`0, 5/3, 25/3, 45/3, 65/3` within a cycle).
- **Trained range**: 107 video latents (362 pixel frames, 15.08 s) corresponds to `[0, 362 * 5 / 3] = [0, 603.3333]` ticks relative to `origin`.

## What Changed

### 1. `stagger_cadence` (`BOOLEAN`, default `False`) — `h3forge/context.py`, `h3forge/nodes.py`

- Restricts `max_stagger_phase` and `stagger_phase` to multiples of `_LATENT_CADENCE` (5 latents) and keeps `_shifted_starts(..., snap=True)` active across every stagger phase so movable window starts stay on single-frame (`k % 5 == 0`) VAE cycle boundaries.
- Keeps `_stagger_layouts`, `stagger_layout_count`, `_blended_segment_windows`, and `make_freenoise_wrapper` aligned on the same phase-0 anchor. When `stagger_cadence=True`, `_window_anchor` never falls back to off-cadence `spread`; instead it prefers a cadence-floored anchor over the rounded anchor when doing so unlocks more distinct cadence layouts (e.g. on the baseline `202/85/21` plan, `[0, 55, 117]` unlocks 2 cadence layouts `(0, 55, 117)` and `(0, 60, 117)` instead of collapsing to 1 at `[0, 60, 117]`, compared with 7 off-cadence layouts when `stagger_cadence=False`).
- Note on fixed tail windows: `final_start = total - window` remains fixed to preserve full coverage and constant window shape (`(total - window) % 5 == 0` puts the tail on cadence as well, e.g. `427/107`; on `202/85`, `117 % 5 == 2` stays the single off-cadence tail start).

### 2. `position_mode` (`"global"` | `"rebased"` | `"wrapped"`, default `"global"`) — `h3forge/layout.py`, `h3forge/context.py`, `h3forge/nodes.py`

- **`global`** (default): unchanged global target positions transplanted from `full_layout`.
- **`rebased`**: subtracts `(v0 // 5) * (85 / 3)` ticks from both target video and target audio `position_ids[:, 0]`, so each window's video starts at `origin` plus its exact offset inside the 5-latent cycle and target audio shifts by the exact same constant, preserving A/V alignment while keeping every window inside the trained range.
- **`wrapped`**: keeps global positions (`shift = 0`) whenever the window's centre relative to `origin` lies within `[0, 362 * 5 / 3]`; otherwise subtracts the smallest integer multiple of `85 / 3` ticks (capped at `v0 // 5` cycles) that brings the window's centre into `[0, 603.3333]`. On `202/85/21`, windows 0 (`[0, 85)`) and 1 (`[60, 145)`) stay global (`shift = 0`), and window 2 (`[117, 202)`) shifts by `-11 * 85 / 3 = -311.67` ticks.
- **Audio context**: full-timeline audio spans the whole clip and would collide with text/refs when shifted (or misalign with shifted video). `ContextPolicy` rejects `position_mode != "global"` combined with `audio_context="full"`, and `audio_context="auto"` automatically activates the existing `window_audio` path whenever `position_mode != "global"`. Passing `audio_context="window"` explicitly is also supported and recommended for clarity in GPU tests.
- **Prefix / sparse attention**: prefix rows (`text`, `ref_img`, `ref_audio`, `cond`, `cond_audio`) and spatial axes `position_ids[:, 1:]` are untouched; `rebased`/`wrapped` are intended for T2VA/Ref2VA (FL2VA/I2VA keyframes remain anchored in global clip time). `_h3forge_video_offset` and `_h3forge_audio_offset` remain global so sparse attention masks in `h3forge/attention.py` continue to work without extra recompiles.

### 3. Step-0 Logging — `h3forge/context.py`

The existing single step-0 `[H3Forge] context plan ...` line now includes:
- `stagger_cadence=on|off`
- `stagger_layouts=<N>` (when staggering is active)
- `position_mode=global|rebased|wrapped`
- `position_shifts=<s0,s1,...>` (in 40 Hz ticks when `position_mode != "global"`)

## Cockpit (`ComfyUI-MMH3-ClipLoops`) Arguments for GPU Testing

In `takes.py` `windowed()`, pass the new optional inputs on `H3ForgeContextWindows` / `H3ForgeTimelineContextWindows`:

1. `stagger_cadence` alone:
   `stagger_cadence=True, position_mode="global"`
2. `rebased` alone (must use windowed audio):
   `stagger_cadence=False, position_mode="rebased", audio_context="window"`
3. Both:
   `stagger_cadence=True, position_mode="rebased", audio_context="window"`
4. Optional middle ground (`wrapped`):
   `stagger_cadence=False, position_mode="wrapped", audio_context="window"` (and with `stagger_cadence=True`)

When evaluating `rebased` on Battle Brief and Sparring Hall, inspect windows 2 and 3 in their solo spans as well as overlaps, because rebasing changes the RoPE coordinates seen in those solo spans.

## Verification

### Unit Tests and Lint (CPU, macOS)

- Lint (`ruff 0.12.12`): `.venv/bin/ruff check h3forge tests __init__.py` — **passed** (0 errors).
- Bytecode compile: `.venv/bin/python -m compileall -q h3forge tests __init__.py` — **passed**.
- Pytest (`PYTHONPATH=. H3FORGE_COMFY_SOURCE=~/Projects/comfyui-local .venv/bin/python -m pytest -q`):
  - **261 passed** (+34 new test cases added across `tests/test_context_math.py`, `tests/test_layout_windows.py`, `tests/test_audit_repairs.py`, `tests/test_nodes.py`).
  - **11 pre-existing environmental failures** (identical to untouched `main` `05792b7` against local `~/Projects/comfyui-local`, which references `comfy.model_prefetch.pause_malloc_graph` and `ComfyAttention` not injected by `tests/core_source.py`: 2 in `tests/test_audit_repairs.py::test_native_control_wrappers_receive_global_control_slices` and 9 in `tests/test_nag.py`). Zero new failures.
- New tests verify:
  - `stagger_cadence=True` keeps all phases and movable starts on multiples of 5 (and tail too when `(total - window) % 5 == 0`), preserves full coverage and requested overlap, matches `stagger_layout_count`, and keeps the FreeNoise anchor in lockstep with step 0.
  - `position_mode="rebased"` starts video at `origin + cycle_offset`, shifts stereo audio by the exact same tick offset (preserving A/V alignment across on- and off-cadence starts with and without refs), leaves prefix/spatial positions untouched, and preserves global `_h3forge_video_offset` / `_h3forge_audio_offset`.
  - `position_mode="wrapped"` keeps windows with centre inside `[0, 603.33]` ticks at exact global positions and subtracts a cycle multiple for later windows so their centres lie inside the trained range.
  - `_make_block_mask` in `h3forge/attention.py` continues to work and shares the same cached mask across `global`, `rebased`, and `wrapped`.
  - Default policy (`stagger_cadence=False`, `position_mode="global"`) is bit-identical to `main`.
  - Step-0 `[H3Forge] context plan` log reports all active settings, layout counts, position shifts, and windowed audio selection.
  - Both context nodes expose `stagger_cadence` and `position_mode` as optional inputs with off/global defaults and reject `audio_context="full"` with non-global position modes.

### GPU Metrics

**No GPU receipt.** No GPU pod was attached to this session; Battle Brief and Sparring Hall renders (`stagger_cadence` alone, `rebased` alone, both, and `wrapped`) and `tools/seam_metrics.py` measurements have not been run yet.

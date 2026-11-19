# Seam Experiment 04: Coarse Layout Pass (`layout_guide`)

Branch: `seam/04-coarse-layout` (from `main` at `05792b7`)

## Summary

Adds an opt-in **coarse layout pass** (`layout_guide`, `layout_guide_until`) to `ContextPolicy` and both context window nodes (`H3ForgeContextWindows` and `H3ForgeTimelineContextWindows`).

On high-noise steps (`sigma > layout_guide_until`), `make_context_wrapper` runs one extra sparse global forward pass across a subset of video latents spanning the entire clip within a single window budget, preserving each selected latent's true global RoPE coordinates. It then interpolates that sparse video velocity across the full clip timeline and replaces only the low spatial frequency band of the normal windowed video prediction with `lerp(windowed_low, sparse_low, layout_guide)`. High spatial frequencies and all audio predictions remain strictly from the normal windowed pass.

When `layout_guide == 0.0` (the default) or `sigma <= layout_guide_until`, the sparse pass is skipped and outputs are bit-identical to `main`.

## What Changed

- **`h3forge/layout.py`**:
  - Added `video_frame_positions(layout)` to extract per-latent-frame global timestamps.
  - Generalized `clone_window_layout` with an optional `video_indices` argument (mutually exclusive with `video_range`) to transplant exact global `position_ids` for non-contiguous video latent sets while preserving prefix (`text`, `cond`, `ref_img`, `ref_audio`) and audio coordinates. Sets `_h3forge_video_indices` on the cloned layout.
- **`h3forge/context.py`**:
  - Added `LAYOUT_GUIDE_LOWPASS_RADIUS = 3` (separable 2D Gaussian low-pass radius in latent H/W; `sigma = max(radius / 2, 0.5)`, `replicate` padding so constant fields are preserved with zero edge darkening).
  - Added `layout_guide: float = 0.0` and `layout_guide_until: float = 0.7` to `ContextPolicy` with `[0, 1]` finite validation.
  - Added `sparse_video_indices(total, window)`: includes every H3 5-latent cycle's single-frame latent (`0, 5, 10, ...`), guarantees first (`0`) and last (`total - 1`) latents are covered, and fills remaining budget up to `window` with evenly spaced four-frame latents (for 202 latents / window 85: all 41 single-frame latents + 44 four-frame latents = 85 latents spanning `0..201`). When a very long clip has more single-frame latents than `window`, subsamples evenly while keeping first and last latents.
  - Added `interpolate_sparse_video(sparse, indices, total_t, times=...)`: linear temporal interpolation using physical H3 timestamps (`video_frame_positions`), returning exact values (no rounding drift) at the sparse anchor rows.
  - Added `spatial_lowpass` and `fuse_layout_guide` for low-band replacement: `(windowed - windowed_low) + lerp(windowed_low, sparse_low, layout_guide)`.
  - Updated `make_context_wrapper`:
    - Runs the sparse global pass with full clip audio (`audio_range=(0, audio_t)`), index-selected `denoise_mask`, and `control_window(..., video_indices=...)`; discards the sparse pass's audio output.
    - Leaves the normal windowed pass and audio fusion untouched.
    - Skips the guide cleanly on segmented (pipe-prompt) runs and single-window runs.
    - Emits one `[H3Forge]` log line at step 0 reporting `layout_guide`, `until`, active/skipped status, sparse latent count and span, `lowpass_radius`, and `guided_steps` (e.g. `[H3Forge] layout_guide=0.50 until=0.70 active=yes sigma=1.000 sparse_latents=85/202 span=0..201 lowpass_radius=3 guided_steps=23/30`, or `[H3Forge] layout_guide=0.00 (off)`).
- **`h3forge/control.py`**:
  - Extended `control_window` with optional `video_indices` so native `MiniMaxH3FunControlPatch` receives matching sparse control latents during the sparse global pass and restores full-clip control state afterward.
- **`h3forge/attention.py`**:
  - Added a `"sparse-layout-guide"` check in `_can_sparse` (accepted under `strict=True` alongside `first-dense-layers` / `first-dense-steps`) so sparse `flex_sliding` attention does not apply a contiguous-frame local mask to a non-contiguous global layout; the sparse pass runs dense attention over its `<= window` latents, giving full-clip visibility.
- **`h3forge/nodes.py`**:
  - Exposed optional inputs `layout_guide` (FLOAT `0.0..1.0`, default `0.0`) and `layout_guide_until` (FLOAT `0.0..1.0`, default `0.7`) in `_context_optional_inputs()` so both `H3ForgeContextWindows` and `H3ForgeTimelineContextWindows` accept them without breaking existing API graphs.
- **Tests**:
  - `tests/test_layout_windows.py`: non-contiguous `clone_window_layout` transplant tests across T2VA/I2VA/FL2VA/Ref2VA/keyframes+refs and argument validation.
  - `tests/test_context_math.py`: `sparse_video_indices` coverage/bounds (including `202/85` and `singles > window`), `interpolate_sparse_video` exact anchor preservation and physical-time interpolation, `spatial_lowpass` / `fuse_layout_guide` band separation, `ContextPolicy` validation, `layout_guide=0` bit-identical proof, threshold cutoff (`sigma <= layout_guide_until`) skip proof, end-to-end `fake_minimax` low-band lerp + audio discard + step-0 log receipt tests at `0.5` and `1.0`, segmented skip test, and `control_window` non-contiguous slicing test.
  - `tests/test_nodes.py`: optional input exposure/defaults and policy plumbing on both context nodes.
  - `tests/test_audit_repairs.py`: strict-mode sparse attention dense fallback on `_h3forge_video_indices` layouts.

## Cockpit Integration Note (`ComfyUI-MMH3-ClipLoops`)

This PR does not modify `ComfyUI-MMH3-ClipLoops`. To test or enable this feature from `takes.py` (`windowed()`), pass the new optional inputs on the `H3ForgeContextWindows` / `H3ForgeTimelineContextWindows` node:

```python
"layout_guide": 0.5,         # FLOAT in [0.0, 1.0]; 0.0 = off (test 0.5 and 1.0)
"layout_guide_until": 0.7,   # FLOAT in [0.0, 1.0]; default 0.7
```

## What Was Measured (CPU Verification)

- **Pytest (without `H3FORGE_COMFY_SOURCE`)**: `220 passed, 43 skipped` (0 failed).
- **Pytest (with `H3FORGE_COMFY_SOURCE=~/Projects/comfyui-local`)**: `252 passed, 11 failed` (up from `227 passed, 11 failed` on untouched `main` `05792b7` — +25 new passing test cases, 0 regressions). The 11 pre-existing failures are environmental mismatches between `tests/core_source.py` and the newer local ComfyUI checkout (`NameError: name 'ComfyAttention' is not defined` in `tests/test_nag.py` [9 tests], and `NameError: name 'comfy' is not defined` in `tests/test_audit_repairs.py::test_native_control_wrappers_receive_global_control_slices` [2 tests]).
- **Lint (`ruff` default rules `E4,E7,E9,F`, line-length 120)**: `All checks passed!`.
- **Bytecode compile (`python3 -m compileall -q h3forge tests __init__.py`)**: passed cleanly.

## What Was Not Measured

**No GPU receipt.** No GPU pod was attached to this session.

To complete evaluation on a GPU pod against the baseline (`seam-brief/api-prompts/battle-brief-baseline-api.json` and `sparring-hall-baseline-api.json`):

1. Run both baseline takes at `layout_guide=0.5` and `layout_guide=1.0` with `layout_guide_until=0.7` (passing the options via a temporary hook in `takes.py` `windowed()`).
2. Confirm the step-0 `[H3Forge] layout_guide=...` log receipt shows `active=yes sparse_latents=85/202 span=0..201`.
3. Measure sharpness, flicker, and max frame diff with `python3 tools/seam_metrics.py baseline.mp4 yours.mp4`.
4. Visually inspect Battle Brief for whether the middle window still frames the room closer than windows 1 and 3, and inspect a Sparring Hall contact sheet for motion continuity across both joins (checking for any time-lapse artifacts from the sparse pass).
5. Record actual render time vs baseline. Expected overhead: 1 extra window-sized forward per guided step (+33% per guided step on a 3-window plan; with `shift=12`, 30 steps, and `layout_guide_until=0.7`, roughly 22–23 of 30 steps are above 0.7, so ~+25% wall-clock time).

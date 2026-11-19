# Seam Experiment 08: Neighbour as a Keyframe (`seam/08-neighbour-keyframe`)

## Summary

Adds optional video-only neighbour keyframe conditioning (`neighbour_keyframe` and `neighbour_keyframe_until`) to `ContextPolicy`, `make_context_wrapper`, `make_freenoise_wrapper`, and both context nodes (`H3ForgeContextWindows` and `H3ForgeTimelineContextWindows`, via `_context_optional_inputs`):

- `neighbour_keyframe`: `"off"` (default), `"first"`, or `"both"`.
- `neighbour_keyframe_until`: float sigma in `[0.0, 1.0]` (default `0.3`). Neighbour keyframes are supplied while `sigma >= neighbour_keyframe_until` and omitted below it.
- Default `"off"` is bit-identical to `main`. Audio fusion is untouched.
- At step 0, `make_context_wrapper` emits one `[H3Forge]` log line stating the option value, `until`, whether it was active, the cadence-restricted stagger setting, the step-0 `visual_cond_noise_aug`, and the leading anchors.

### Cockpit integration (`ComfyUI-MMH3-ClipLoops`)

To test this branch from `takes.py` `windowed()`, pass the new optional widget inputs to the Forge context node:

- `neighbour_keyframe="first"` (or `"both"`)
- `neighbour_keyframe_until=0.3`

---

## Core Investigation (Three Required Questions)

All line references below are against ComfyUI core in `~/Projects/comfyui-local`.

### 1. How keyframe rows are packed and positioned, and whether a keyframe can be supplied directly as a latent

**Yes — keyframes are supplied directly as normalized video latents; no VAE decode/encode is needed.**

- **Same normalized latent space as the sampler:** `MiniMaxH3VideoVAE.encode` (`comfy/ldm/minimax/vae.py:965-982`) outputs normalized `[1, 24, T, H/16, W/16]` latents using `LATENTS_MEAN`/`LATENTS_STD`. `MiniMaxH3Video` / `MiniMaxH3AV` (`comfy/latent_formats.py:622-674`) uses `scale_factor = 1.0` with no video rescaling in `process_latent_in`/`process_latent_out` (`comfy/model_base.py:2248-2252` only rescales audio). Under `CONST` flow matching (`comfy/model_sampling.py:86-92`), the velocity returned inside `DIFFUSION_MODEL` wrappers satisfies `x0_est = video_x - sigma * v_out` (with `m * sigma` when a per-row `denoise_mask` is active; `comfy/ldm/minimax/model.py:705-706, 752-765`). That 1-latent slice `[1, 24, 1, H_lat, W_lat]` is already in the exact space consumed by `MiniMaxH3Model._cond_video_rows` (`comfy/ldm/minimax/model.py:567-580`).
- **Packing and positioning:** `PackedLayout.__init__` (`comfy/ldm/minimax/model.py:386-504`) lays out the packed sequence as `[text | keyframes (cond / cond_audio) | refs (ref_img / ref_audio) | audio | video]`:
  - The target timeline origin is `cursor = float(text_len) + sum(_ref_t_span(blk) for blk in refs or ())` (`model.py:401-403`).
  - Each keyframe entry `kf` in `payload["keyframes"]` places its video rows at `cond_t = cursor + FRAME_RESCALE * kf["resolved_frame_index"]` (`model.py:408-418`, `FRAME_RESCALE = 5/3`) using `_video_grid(vt, frame, cond_t)` (`model.py:133-137`), sets `img_update = False`, and appends segment `("cond", vt * frame_rows)`.
  - `MiniMaxH3._cond_latents` (`comfy/model_base.py:2303-2307`) orders `payload["cond_video_latents"]` as `[keyframe video latents..., ref video latents...]`, matching `~layout.img_update` in `_embed_and_pack` (`model.py:604-608, 625-637`).
  - Target video rows in `final_layer` are selected by `next(s for s in layout.segments if s[2] == "video")` (`model.py:806`), so extra `"cond"` rows never alter output shapes.
- **Conditioning noise augmentation semantics:** In `model.py:34, 570-580, 746-750`, `payload["visual_cond_noise_aug"]` (`minimax_visual_cond_noise_aug` in `comfy/model_base.py:2280-2281`) is in **`t`-space (cleanliness)**, defaulting to `VISUAL_COND_TIMESTEP = 0.999`:
  - `_cond_video_rows` perturbs every cond latent as `r = aug * r + (1.0 - aug) * noise` (`model.py:578`).
  - `_forward` assigns `seg_t["cond"] = max(t_v, vis_aug)` where `t_v = 1.0 - sigma_v` (`model.py:742, 749`).
  - Because shift-12 spends the first half of a 30-step run at `sigma_v` in `[0.92, 1.0]`, setting `aug = 1 - sigma` would zero out step 0 (`aug = 0`) and leave `< 5%` signal during the early steps where seams form. Instead, `neighbour_keyframe_noise_aug` maps `aug = min(0.999, 1.0 - 0.5 * sigma)` (never raising an existing lower user value), giving `0.50` at `sigma = 1.0` and `0.85` at `sigma = 0.3` while keeping `seg_t["cond"] == aug >= t_v`.

### 2. Generalising `clone_window_layout` vs. forcing uniform keyframe rows

**Chosen: generalise `clone_window_layout` with an optional `extra_keyframes` parameter (`h3forge/layout.py:49-125`).**

Forcing every window to carry the same number of keyframe rows does not work cleanly here:
1. Before window 0 runs at step 0, no estimate exists for window 0 (and in `both` mode, no previous-step estimate exists for any window at step 0), so injecting a dummy/zero keyframe would corrupt window 0 rather than acting as a no-op.
2. `clone_window_layout` previously copied `full_layout.position_ids[fa:fb]` onto every non-target segment (`h3forge/layout.py:90`), which would overwrite each window's distinct keyframe time coordinate even if dummy rows were added to `full_layout`.

Instead, `clone_window_layout(..., extra_keyframes=...)` accepts `(descriptor, anchor_video_latent)` pairs for the current window only:
- When `extra_keyframes` is `None`/empty, the strict segment-structure check is unchanged.
- When `extra_keyframes` is provided, `PackedLayout` is built with `all_keyframes = shared_keyframes + extra_keyframes`; the shared segments (text, any user keyframes, refs, audio, video) are still validated 1-to-1 against `full_layout.segments`, and each extra `"cond"` segment receives the exact global RoPE slice `full_layout.position_ids[src.video_start + anchor * frame_rows : src.video_start + (anchor + 1) * frame_rows]`. This stays exact even when pipe-prompt segments have different text lengths or when `refs` shift the timeline origin.
- `H3ForgeAttention` (`h3forge/attention.py:157-165, 211-225`) already includes `layout.segments` in `_mask_specialization` and treats all tokens before `audio_start` (including `"cond"`) as dense global conditioning (`gq | gk`).

### 3. Interaction with the location RefMod bridge on the model line

`MiniMaxH3RefModContinuumBridge` (`battle-brief-baseline-api.json` node 192) is a third-party custom node not present in `~/Projects/comfyui-local`, so its source could not be inspected locally. Based on ComfyUI core's H3 extension points (`comfy/ldm/minimax/model.py`, `comfy/ldm/minimax/controlnet.py`, `comfy_extras/nodes_minimax_h3.py`):

- **If RefMod patches DiT blocks or attention (`patches_replace["dit"]` / `optimized_attention_override`)** (consistent with node 192's `retention` / `curve_*` / `scramble_*` / `max_total_tokens` inputs, and analogous to `MiniMaxH3FunControl` in `comfy/ldm/minimax/controlnet.py:62-69`): our `local_layout` is a real `PackedLayout` with consistent `segments`, `img_pos`, `img_update`, and `seq_len`, published to `transformer_options["minimax_h3_layout"]` (`model.py:736`). Target video rows stay `img_update == True` and keyframe rows are `img_update == False`.
- **If RefMod wraps `DIFFUSION_MODEL` and injects `refs` / `cond_video_latents` into `minimax_payload`:**
  - In `battle-brief-baseline-api.json`, node 192 sits upstream of Forge, so its wrapper runs outside `make_context_wrapper`. `_inject_neighbour_keyframes` inserts neighbour keyframe latents *before* any existing `refs` latents in `cond_video_latents` and leaves `refs` intact, preserving `PackedLayout`'s `[keyframes | refs]` order.
  - **Caveat to verify on GPU:** `payload["visual_cond_noise_aug"]` in core is a single scalar shared by both `"cond"` and `"ref_img"` (`model.py:570-578, 746-750`). If RefMod feeds tokens through `payload["refs"]` / `cond_video_latents`, lowering `visual_cond_noise_aug` for neighbour keyframes will also noise-augment RefMod's `ref_img` rows while `sigma >= neighbour_keyframe_until`. (If RefMod injects pre-encoded tokens in block/attention hooks instead, `visual_cond_noise_aug` does not affect it.)

---

## Implementation Notes

- **Cadence-aligned stagger and overlap anchors (`h3forge/context.py`):**
  - When `neighbour_keyframe != "off"`, `max_stagger_phase`, `stagger_phase`, `window_starts`, `_blended_segment_windows`, and `make_freenoise_wrapper` run with `cadence_only=True`: stagger shifts are multiples of 5 (`_LATENT_CADENCE`) and interior starts stay snapped to the 5-latent cadence.
  - Because H3 clip lengths follow `5k + 2` (e.g. `202` latents) while windows are multiples of 5 (`85`), the final window start (`202 - 85 = 117`) is off-cadence (`117 % 5 == 2`, a 4-frame latent). `_first_neighbour_anchor` selects the first cadence-0 latent in the overlap (`v0` whenever `v0 % 5 == 0`, e.g. `60` for window 1, and `120` inside `[117, 145)` for window 2). Symmetrically, `_last_neighbour_anchor` selects the last cadence-0 latent in the overlap (`80` for window 0, `140` for window 1). Both fall back to the overlap edge if an overlap is smaller than 5 latents.
- **Mode `both` cache:** A small per-wrapper dict caches window `k+1`'s detached 1-latent `x0` estimate(s) for window `k` on the next step (covering cadence-0 overlap anchors across stagger shifts). It clears on `step == 0`, clip shape/window-count changes, sigma increases (new run / refine pass), or once `sigma < neighbour_keyframe_until`.

---

## Verification

### GPU evaluation

**No GPU receipt** — no GPU pod was attached to this session. The two baseline takes (`battle-brief-baseline-api.json` and `sparring-hall-baseline-api.json`) for `first` and `both`, render-time overhead (1–2 extra single-frame `cond` rows per window), RefMod bridge interaction, and checking for window-start "restart" pops remain to be measured on a GPU pod.

### CPU unit tests & lint

- Baseline untouched `main` (`PYTHONPATH=. H3FORGE_COMFY_SOURCE=~/Projects/comfyui-local .venv/bin/python -m pytest -q`): **227 passed, 11 failed**.
  - The 11 pre-existing failures are environmental due to `~/Projects/comfyui-local` being newer than CI's pinned ComfyUI commit (`250b2e9551`):
    - 2 in `tests/test_audit_repairs.py` (`test_native_control_wrappers_receive_global_control_slices[True/False]` — `NameError: name 'comfy' is not defined` in `pause_malloc_graph()`).
    - 9 in `tests/test_nag.py` (`NameError: name 'ComfyAttention' is not defined` in `Attention.__init__`).
- After this change: **243 passed, 11 failed** (the exact same 11 environmental failures; 16 new tests added, all passing):
  - `tests/test_layout_windows.py`: `clone_window_layout` with 1 and 2 `extra_keyframes` across T2VA, I2VA, Ref2VA, keyframes+refs, unequal pipe text lengths, and error cases.
  - `tests/test_context_math.py`: policy validation, bit-identity with `neighbour_keyframe="off"`, `first` mode `x0` extraction + `cond_video_latents` ordering + `visual_cond_noise_aug` + `until` cutoff + step-0 log line, `both` mode previous-step cache + run-reset + moving stagger, cadence-only stagger and FreeNoise alignment, and packing contract against real `comfy/ldm/minimax/model.py` (`PackedLayout`, `patchify_video`).
  - `tests/test_nodes.py`: both `H3ForgeContextWindows` and `H3ForgeTimelineContextWindows` expose and forward `neighbour_keyframe` and `neighbour_keyframe_until`.
- Lint (`uvx ruff==0.12.12 check h3forge tests __init__.py`): **All checks passed!**
- Bytecode check (`python3 -m compileall -q h3forge tests __init__.py`): **Passed.**

# ComfyUI-H3Forge

Experimental MiniMax-H3 inference surgery for native ComfyUI:

1. **H3-native sliding/radial sparse attention** using PyTorch FlexAttention.
2. **Video-only FETA-style off-diagonal attention enrichment** derived from Enhance-A-Video.
3. **Synchronized audio/video context windows** with overlap-add blending and absolute H3 RoPE preservation.
4. **Pipe-delimited timeline prompting** with independently encoded, optionally unequal prompt spans and a reusable global anchor.
5. **Reference-aware pipe prompting** that reuses ComfyUI's native Ref2VA encoder for every segment.
6. **Experimental H3 NAG-Lite** — Normalized Attention Guidance adapted to H3's packed single stream via a negative-text sidecar.

The project is deliberately a custom-node patch layer. It does **not** fork or modify files under `ComfyUI/comfy/`.

## Status

The GPU integration gate has been passed on Blackwell hardware. The retained long-form receipt is a successful **60.417-second, 1344 × 768, 24 fps synchronized audio/video clip**: 1,450 decoded output frames, denoised as a 427-frame H3 video latent (ComfyUI's `17k + 5` frame grid; five latent frames per 17 output frames) in one sampler execution, using the official `minimax_h3_fl2va_pruned_int8_convrot.safetensors` checkpoint. It used strict sparse attention (`40 / 8 / 40`) and strict chained context windows (`25 / 5`, staggered pyramid blend), peaking **under 50 GB of VRAM**.

Here, "chained" means overlapping latent A/V context windows evaluated inside every denoising step. It does not mean rendering several clips and feeding decoded pixels from one clip into the next. Peak denoising memory is governed mainly by the active window, while wall time and total work continue to grow with the number of windows. Native ComfyUI currently exposes lengths up to 3,600 frames (about 150 seconds at 24 fps), but H3Forge only claims the retained 60.417-second run as verified; longer runs remain an explicit quality, seam, and runtime test.

Systematic numbers against `BLACKWELL_TEST_MATRIX.md` (sparse sweeps, seam/identity scoring, NAG acceptance) are still being collected. Treat the knob defaults in this README as working starting points, not tuned optima.

## What is implemented

### H3 Forge — Sliding Attention + FETA

Patches MiniMax-H3's `optimized_attention_override` and stamps the 50 DiT block indices using ComfyUI's public patch APIs.

The sparse policy is modality aware:

- text / conditioning / reference prefix remains globally reachable;
- target audio ↔ audio uses a local temporal band;
- target video ↔ audio uses the same H3 physical time coordinate;
- same-time video ↔ video stays spatially dense;
- cross-time video ↔ video is limited by both temporal distance and a spatial patch radius;
- optional dilated bridge times provide long-range K/V routes outside the local band;
- early layers and an early fraction of denoising can remain dense.

The H3 temporal mapping is **not** guessed from token indices. Video time uses MiniMax-H3's `1,4,4,4,4` latent-token cadence and audio uses its native one-step grid.

Both FlexAttention and block-mask construction are compiled. The block mask is built once with `H=None` because the policy is head-independent, then broadcast across H3's 56 heads. This avoids the eager `B × H × S × S` boolean grid that otherwise OOMs before the sparse kernel can run. If FlexAttention declines and `strict=false`, H3Forge falls back to ComfyUI's configured dense attention backend and records the reason.

### H3 FETA

The optional enrichment path follows Enhance-A-Video's scalar FETA estimator but only samples MiniMax-H3's **target-video rows**:

- same spatial sites are compared across time;
- only a configurable sample of heads and spatial sites is used;
- no full packed-sequence attention map is materialized;
- the gain is applied only to target-video attention output rows;
- target audio, text, references, and conditioning are left untouched.

The default gain cap is intentionally conservative (`1.15`).

### H3 Forge — Chained A/V Context Windows

This is overlap-add context denoising rather than "generate clip A, then feed its pixels to clip B".

For every denoising step it:

1. creates overlapping target-video latent windows;
2. maps each video interval to the physically overlapping H3 audio-latent interval;
3. constructs a window-local `PackedLayout`;
4. transplants the **original global audio/video position IDs** into that layout;
5. evaluates H3 jointly on the synchronized A/V window;
6. pyramid-blends video and audio predictions into their full latent canvases;
7. optionally shifts interior window boundaries between sampler steps.

That absolute-position transplant is important: a context window beginning at latent frame 26 must not pretend it begins at H3 frame zero and restart the `1,4,4,4,4` RoPE cadence.

### H3 Forge — Pipe Timeline Prompt

This node splits a prompt on `|`, encodes every segment independently with MiniMax's Qwen3-VL text encoder, and maps the segments in order across the target timeline. It is not a decorative delimiter passed to one global text encoding.

Two optional inputs make the timeline less toy-like:

- `global_prompt` is repeated inside every independent segment encoding. Put the identity/style/location anchors that truly apply throughout the video here instead of maintaining identical copies by hand.
- `segment_durations` accepts exactly one positive number per segment, separated by commas or newlines. The numbers are relative durations, so `2,18,40` means the same thing whether you think of them as seconds, frames, or beats. They do not have to add up to the output length. Leave the field empty for equal spans.

All segment embeddings are padded to one token length before sampling, preserving one compiled H3 context shape. Each synchronized A/V context window uses the **complete encoding of the one segment covering its midpoint** — contextualized token slots from independently encoded prompts do not correspond to one another, so hidden states are never interpolated. Around a prompt boundary, adjacent windows generated under different prompts crossfade in **output space** through the context-window overlap-add blend, which is where blending is semantically sound.

Segments must share the same conditioning structure: differing multimodal inserts or presentation tags across segments are rejected at encode time rather than silently stamped with segment 1's metadata.

Segments must not outnumber the context windows that can select them: if a segment's requested span contains no window midpoint, H3Forge warns at the first step (or aborts in strict mode) instead of letting that prompt silently vanish. This can also happen when an unequal segment is shorter than the context window's reachable midpoint range. Use fewer segments, lengthen the affected duration, or use a smaller `window_frames`.

Use it with `H3 Forge — Chained A/V Context Windows` and an `Empty MiniMax H3 AV Latent`. A single segment is valid and behaves like ordinary global conditioning. Escape a literal pipe as `\|`.

Every segment is encoded independently. Write each local segment as a self-contained, valid H3 prompt in MiniMax's official structured format (`integrated_multimodal_description` / `overall_soundscape` / `non_diegetic_music`). Put concrete identity, wardrobe, location, lighting, voice, and style anchors in `global_prompt`, or repeat them manually when they change between segments. Do not rely on cross-segment shorthand such as “same person”, “continues”, or “remains unchanged”; the local segment encoder cannot see a previous segment. For ordinary 5–15 second work, MiniMax's native `[Shot N] At ...` timing syntax inside one prompt may make pipe prompting unnecessary; pipe scheduling earns its keep on the long-form context-window path where separate local forwards really do need different prompt contexts.

At sampler step zero, the context node prints one compact plan containing the actual latent length, window count, effective stagger phase, window/overlap settings, total video-latent visits, and a run-length-compressed prompt-to-window assignment. `video_latent_visits` is an overlap accounting ratio, not a wall-time or VRAM prediction.

### H3 Forge — Reference Pipe Timeline Prompt

This is the image-reference counterpart to the text-only pipe node. It splits on `|`, invokes ComfyUI's native `MiniMaxH3ReferenceToVideo` encoder independently for every self-contained segment with the same one-to-four reference images, validates that their multimodal token-tag structure matches, and returns both the combined positive conditioning and native AV latent. It supports the same optional `global_prompt` and `segment_durations` inputs. The first segment's identical native reference payload supplies the global reference prefix; every local context window receives the complete reference-aware Qwen encoding selected for its midpoint.

Use full Ref2VA prompt grammar inside **every** segment (`subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, `non_diegetic_music`) and keep reference labels and subject definitions identical. This node intentionally does not expose reference video or reference audio inputs yet; use the native node directly when those modalities matter.

### H3 Forge — Normalized Attention Guidance (experimental)

H3's released checkpoints are guidance-distilled: a normal `BasicGuider` workflow already runs one forward per step and pays no CFG cost, but negative prompts do nothing at CFG 1. This node restores meaningful negative-prompt control **without a second complete H3 transformer pass**.

H3 is structurally on the expensive side of the NAG divide: unlike Wan's external text cross-attention (where NAG costs roughly 12%), H3's text, references, audio, and video share one packed self-attention per block, like Flux (where faithful NAG costs closer to 87%). So this node implements **H3 NAG-Lite**, not faithful NAG:

1. the negative prompt is encoded once and passed through H3's text preprocessing;
2. at selected DiT blocks, the current positive target audio/video queries attend to the positive text K/V and to the negative sidecar text K/V;
3. the exact NAG formula (`guided = pos·scale − neg·(scale−1)`, L1-renormalized with cap `tau`, alpha-blended) is applied to those two text-conditioned contributions;
4. only the **delta** is injected into target audio/video attention rows before the output projection;
5. A/V↔A/V self-attention, the MLP, and the rest of the positive packed stream are untouched.

The added attention cost is roughly `target rows × text length` per selected block instead of `packed length²`.

**Documented approximations** (why this is named NAG-Lite): the sidecar text state is frozen at the refined embedding rather than re-evolved through earlier blocks; it skips per-step modulated norms and RoPE on sidecar keys; and in `lite` mode the standalone text attention does not share the packed softmax denominator with A/V keys. `mode=faithful_selective` removes the last approximation by rerunning full-key attention for the target rows with the text partition swapped — materially more expensive, useful as a quality ceiling for the selected blocks.

Knobs: start at `nag_scale 3.0` (not Wan's 11 — H3 is distilled, single-stream, and jointly generates speech and imagery, so aggressive attention extrapolation has more opportunities to damage identity, voice, or sync). Find stable `nag_tau`/`nag_alpha`, then leave them fixed and tune only the scale. `nag_sigma_end` stops NAG once sigma falls below it, saving compute in the late schedule. `first_block`/`last_block` select the DiT blocks; `video_strength`/`audio_strength` scale the injected delta per modality.

The node composes with `H3 Forge — Sliding Attention + FETA` in either wiring order (both configure one shared H3Forge runtime) and works standalone. Do not stack generic ComfyUI-NAG on top.

## Requirements

- A recent native ComfyUI build with `MiniMaxH3Model` and `PackedLayout` under `comfy.ldm.minimax.model`.
- PyTorch with `torch.nn.attention.flex_attention` for sparse mode.
- CUDA strongly recommended; sparse FlexAttention is not intended as a CPU execution path.
- No extra Python package is required by H3Forge itself.

The code was authored against the native ComfyUI MiniMax-H3 implementation and most recently rechecked live with ComfyUI `0.34.0`, Python `3.12.3`, and PyTorch `2.13.0+cu130` on 2026-09-03.

## Install

Extract or clone into ComfyUI's custom nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/RationallyPrime/ComfyUI-H3Forge.git
```

or place the extracted `ComfyUI-H3Forge/` directory there, then restart ComfyUI.

Five nodes should appear:

- `H3 Forge — Sliding Attention + FETA`
- `H3 Forge — Chained A/V Context Windows`
- `H3 Forge — Pipe Timeline Prompt`
- `H3 Forge — Reference Pipe Timeline Prompt`
- `H3 Forge — Normalized Attention Guidance` (experimental)

The attention, context, and NAG nodes accept and return `MODEL`; insert them after the H3 model loader and before sampling. They can be wired in any order — the attention and NAG nodes configure one shared H3Forge runtime. The text-only pipe node accepts MiniMax's `CLIP` and returns positive `CONDITIONING`. The reference pipe node additionally accepts the video/audio VAEs and one-to-four images, returning both positive `CONDITIONING` and the native AV `LATENT`. The NAG node additionally takes negative `CONDITIONING`.

For the recommended feed-forward memory reduction, also install [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes):

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/kijai/ComfyUI-KJNodes.git
```

Restart ComfyUI after installing either custom-node package.

## GPU bring-up protocol

This protocol was used for the initial Blackwell bring-up and remains the recommended path when validating a new GPU, driver, or ComfyUI build. Do **not** turn every knob on for the first run. Use a known-good H3 workflow and keep seed/prompt/resolution/steps fixed.

### Run 0 — baseline

No H3Forge nodes and no KJNodes chunker: this is the native reference (`B0`) that every later equivalence check compares against. Record:

- wall time;
- peak VRAM;
- output video;
- audio intelligibility;
- lip sync / audiovisual event sync.

### Run 1 — wiring-only dense mode

Add `H3 Forge — Sliding Attention + FETA` with:

```text
mode                  dense
feta_enabled          false
first_dense_layers    2
first_dense_fraction  0.15
strict                true
```

This should be visually equivalent to baseline apart from ordinary backend nondeterminism. It proves the patch plumbing does not perturb H3.

### Run 2 — sparse attention only

Suggested starting values:

```text
mode                  flex_sliding
temporal_window       40
spatial_radius        8
bridge_stride         40
first_dense_layers    2
first_dense_fraction  0.15
feta_enabled          false
strict                true
```

Expect a compilation hit on the first sparse invocation. Judge steady-state time separately from first-run compile time.

Watch the console for H3Forge's run summary. In strict mode a sparse-kernel contract failure aborts instead of silently poisoning the comparison.

### Run 3 — FETA only

Use dense mode and enable FETA:

```text
mode              dense
feta_enabled      true
feta_strength     2.0
feta_max_gain     1.10
feta_first_layer  6
feta_last_layer   42
```

Compare motion amplitude, temporal consistency, faces/hands, speech mouth shapes, and audio sync. If useful, raise `feta_max_gain` toward `1.15` before touching the weight.

### Run 4 — context windows only

Start with a generation whose target video latent length is larger than the window:

```text
window_frames   25
overlap_frames  5
stagger         true
blend           pyramid
strict          true
```

Look specifically for:

- visible window seams;
- rhythm or speech discontinuities at window boundaries;
- subject identity jumps;
- camera-motion resets;
- uncovered/zeroed audio or video spans.

The implementation maintains separate audio and video weight accumulators, so every target element must have positive overlap weight.

### Run 5 — combined

Only after Runs 2–4 each behave independently:

```text
H3 attention: flex_sliding + conservative FETA
H3 context:   25 / 5 / stagger / pyramid
```

If the combined result regresses, disable FETA first. Sparse routing and context windows change the model's information topology; FETA is an amplifier and should be the last variable introduced.

## Knob semantics

### Attention

`temporal_window`
: Maximum H3-time separation for ordinary target-stream attention, measured on H3's 40 Hz timeline rather than in decoded frames. H3 audio advances by 1 tick; video advances by the native cumulative `1,4,4,4,4` cadence on that same axis. The default `40` is one second. For comparison, `12` is only 0.3 seconds and is an aggressive research setting.

`spatial_radius`
: Patch-grid Chebyshev radius for **cross-time video↔video** attention. Same-time video attention remains spatially dense.

`bridge_stride`
: Every N H3 timeline ticks, target K/V rows become temporal bridge keys available outside the local band. The default `40` is one second; `0` disables bridges.

`first_dense_layers`
: Keeps the first N H3 DiT blocks dense.

`first_dense_fraction`
: Keeps the first fraction of sampler steps dense when ComfyUI exposes the schedule in `transformer_options`.

### FETA

`feta_strength`
: The Enhance-A-Video-style additive weight in the `(T + weight)` gain estimator. This is **not** a direct output multiplier.

`feta_max_gain`
: Hard safety cap on the resulting target-video attention-output multiplier.

### Context

`window_frames`
: Number of **H3 video latent frames** in each context invocation, not decoded output frames.

`overlap_frames`
: Video-latent overlap. Audio overlap is derived from physical H3 time rather than copied index-for-index.

`stagger`
: Moves interior window boundaries by at most the overlap amount across sampling steps while preserving complete coverage.

### Timeline prompt

`global_prompt`
: An optional shared anchor included in every segment before that segment is independently encoded. It saves repetition; it is not a separate globally attended token bank.

`segment_durations`
: Comma- or newline-delimited positive relative durations, with exactly one value per `|` segment. Empty means equal spans. Selection remains hard and midpoint-based per context window; the overlap-add blend performs the boundary crossfade in output space.

### NAG

`nag_scale`
: Attention-space extrapolation strength between positive and negative text contributions. Start at `3.0`; H3 is distilled and single-stream, so do not import Wan-scale defaults.

`nag_tau`
: L1-norm cap on the guided feature relative to the positive feature. Find a stable value, then leave it fixed and tune `nag_scale`.

`nag_alpha`
: Blend of the guided feature back toward the positive feature.

`nag_sigma_end`
: NAG applies while the current sigma is at or above this value; below it, blocks run untouched.

`first_block` / `last_block`
: Inclusive DiT block range that receives the sidecar delta.

`video_strength` / `audio_strength`
: Per-modality multipliers on the injected delta for target video and audio rows.

## Important compatibility notes

- **Do not stack `SolAttnH3` and H3Forge attention/NAG nodes on the same model.** Both own `optimized_attention_override`; H3Forge prints a warning if it replaces an existing override. Choose SolAttn or H3Forge for a given run.
- H3Forge Context Windows *can* be used without H3Forge Attention.
- FETA can be tested with `mode=dense`.
- `strict=true` is recommended for development / first GPU tests. Use `strict=false` only when you explicitly prefer dense fallback over an aborted generation.
- This is inference experimentation, not a claim that MiniMax trained H3 with this exact sparse topology.
- MiniMax describes native sparse-attention training, while the released ComfyUI inference path is dense. Start from the one-second default and treat shorter windows as an explicit quality/speed sweep.

### Composing with ComfyUI-KJNodes

Kijai's KJNodes ships several native-H3 utilities. Only the feed-forward chunker belongs in the default H3Forge model chain:

```text
selected H3 model
  → MiniMax H3 Chunk FeedForward
  → H3 Forge — Chained A/V Context Windows
  → H3 Forge — Sliding Attention + FETA
  → scheduler and guider
```

- **MiniMax H3 Chunk FeedForward** — chunks the packed-token rows of each SwiGLU feed-forward block. Those rows are independent, and INT8 activation quantization is per-token, so the operation is intended to match the unchunked model while reducing peak activation memory. Start with KJNodes' defaults, `chunks=2` and `seq_threshold=4096`. H3Forge patches context/attention behavior at different seams, so the chunker composes with sparse and context-window runs, but it is gated rather than assumed: keep it **out** of the native baseline (Run 0 / `B0`), then verify it once against that unchunked baseline with fixed seeds (`BLACKWELL_TEST_MATRIX.md`, KJNodes composition checks). Leave it enabled only after that equivalence check passes; otherwise a chunker discrepancy would be misattributed to H3Forge in every later comparison.
- **MiniMax H3 Token Counter** — optional diagnostics only. It passes the latent and conditioning through while reporting the true packed count for text, references/keyframes, audio, and video. Add it when investigating attention cost or kernel limits; it does not need to occupy the canonical model chain.
- **MiniMax H3 Low VRAM Attention** — experimental and intentionally excluded from the canonical H3Forge workflow. It replaces H3 block/attention forwards and may split the attention override into head groups. Before combining it with H3Forge, require a fixed-seed equivalence run with FETA disabled; sampled FETA gain is not guaranteed to remain global across separate head-group calls.

### Approximate accelerators

- **ComfyUI-MiniMaxH3-FirstBlockCache** replaces/skips the remaining block stack and warns against combining it with another `double_block` replacement — that is a direct conflict with H3Forge's block stamping and context surgery. Do not combine; benchmark it in isolation with fixed seeds.
- **ComfyUI-Spectrum-MiniMax-H3** forecasts post-transformer features and skips selected transformer evaluations; its own documentation notes it changes the denoising trajectory. Keep it out of any workflow whose purpose is proving H3Forge or NAG behavior.

## Test locally

The included CPU tests cover scheduler coverage, unequal prompt-span routing, global-anchor propagation, context-plan reporting, blend-edge correction, block-mask cache keying, bridge semantics, FETA gain routing, Ref2VA/I2VA/FL2VA window transplants (against a faithful fake `PackedLayout`), NAG math and gating, node composition, and sampler-step resolution:

```bash
PYTHONPATH=. python -m pytest -q tests
```

Static syntax check:

```bash
python -m compileall -q .
```

The CPU tests cannot validate GPU/model correctness on their own — that requires a loaded H3 checkpoint and a CUDA device, and has now been exercised in live GPU sessions (see [Status](#status)). They remain the fast pre-GPU gate for the scheduler, mask, and blend math.

## Design lineage and nearby work

H3Forge is an original implementation informed by native ComfyUI's MiniMax-H3 packed layout and wrapper seams, WanVideoWrapper's context scheduling, and Enhance-A-Video/FETA's off-diagonal temporal-attention gain. The H3 ecosystem now has several useful neighboring projects:

- [ComfyUI-SolAttn-H3](https://github.com/quzopl/ComfyUI-SolAttn-H3) demonstrates disciplined sparse-attention integration, named fallback reasons, self-tests, and per-run telemetry. H3Forge keeps its own H3-aware sparse topology and now applies the same principle of reporting the actual context plan it ran.
- [ComfyUI-YCNodes-MiniMax-H3](https://github.com/yichengup/ComfyUI-YCNodes-MiniMax-H3) introduced a practical global/local Prompt Relay UI with explicit segment lengths. [T8's MiniMax H3 nodes](https://github.com/T8mars/comfyui-minimax-h3-audio-T8) go further with validated frame/second/percent ranges and absolute-timeline projection across sequential long-video segments. H3Forge independently adopts the small composable part that fits its different mechanism: a repeated global anchor and exact unequal relative spans for independently encoded context-window prompts.
- [ComfyUI-MMH3Tools](https://github.com/ckinpdx/ComfyUI-MMH3Tools) and [ComfyUI-H3-Toolkit](https://github.com/wordbrew/ComfyUI-H3-Toolkit) are stronger at H3 frame/audio-grid utilities, workflow validation, and staged long-video mechanics. H3Forge continues to use its own synchronized per-denoise A/V windows with transplanted absolute positions rather than importing a sequential continuation engine.
- [ComfyUI-MiniMax-H3-LongMedia](https://github.com/vizart-vj/ComfyUI-MiniMax-H3-LongMedia) is a fuller end-user long-media system with sequential execution, continuation overlap, streaming/offload, and a memory governor. H3Forge remains the smaller inference-surgery layer and composes with KJNodes' feed-forward chunker instead of duplicating that application shell.
- [H3-Optimizations](https://github.com/Zironic/H3-Optimizations) provides unusually thorough GPU diagnostics and benchmark harnesses. H3Forge's retained artifact and Blackwell matrix serve the same evidence goal, but its benchmark coverage is still less mature.

These are design comparisons, not copied source. In particular, T8 is GPL-3.0-or-later while H3Forge is Apache-2.0; H3Forge's duration routing and reporting here were implemented independently against its existing data path.

No upstream model weights are included.

## License

Apache-2.0. See `LICENSE`.

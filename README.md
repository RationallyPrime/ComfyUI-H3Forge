# ComfyUI-H3Forge

Experimental MiniMax-H3 inference surgery for native ComfyUI:

1. **H3-native sliding/radial sparse attention** using PyTorch FlexAttention.
2. **Video-only FETA-style off-diagonal attention enrichment** derived from Enhance-A-Video.
3. **Video context windows with a shared full audio timeline**, FP32 overlap fusion, and absolute H3 RoPE preservation.
4. **Pipe-delimited timeline prompting** with independently encoded, optionally unequal prompt spans and a reusable global anchor.
5. **Image/video/voice reference prompting** that prepares shared native Ref2VA references once.
6. **Experimental H3 NAG-Lite** — Normalized Attention Guidance adapted to H3's packed single stream via a negative-text sidecar.

The project is deliberately a custom-node patch layer. It does **not** fork or modify files under `ComfyUI/comfy/`.

## Status

Earlier releases passed a Blackwell integration run. The retained long-form receipt is a successful **60.417-second, 1344 × 768, 24 fps synchronized audio/video clip**: 1,450 decoded output frames, denoised as a 427-frame H3 video latent (ComfyUI's `17k + 5` frame grid; five latent frames per 17 output frames) in one sampler execution, using the official `minimax_h3_fl2va_pruned_int8_convrot.safetensors` checkpoint. It used strict sparse attention (`40 / 8 / 40`) and strict chained context windows (`25 / 5`), peaking **under 50 GB of VRAM**. That run predates the current fusion-parity change: its then-named `pyramid` mode was the edge ramp now exposed as `overlap-linear`, so it is not yet a GPU receipt for the full-window pyramid.

Here, "chained" means overlapping latent A/V context windows evaluated inside every denoising step. It does not mean rendering several clips and feeding decoded pixels from one clip into the next. The active video window, complete audio timeline, and reference prefix govern denoiser memory. Wall time and work grow with window count. Decoded output still occupies a full CPU image buffer; 1,450 float32 RGB frames at 1344×768 are about 16.7 GiB. Native ComfyUI currently exposes lengths up to 3,600 frames (about 150 seconds at 24 fps), but H3Forge only claims the retained 60.417-second run as verified; longer runs remain an explicit quality, seam, and runtime test.

The 0.3 repair details and current validation are recorded in [audit-repairs.md](docs/audit-repairs.md). The earlier 60-second clip predates these changes and is not a quality result for the current revision.

Systematic numbers against `BLACKWELL_TEST_MATRIX.md` (sparse sweeps, seam/identity scoring, NAG acceptance) are still being collected. Treat the knob defaults in this README as working starting points, not tuned optima.

## What is implemented

### H3 Forge — Sliding Attention + FETA

Patches MiniMax-H3's `optimized_attention_override` and stamps the 50 DiT block indices using ComfyUI's public patch APIs.

The sparse policy is modality aware:

- text / conditioning / reference prefix remains globally reachable;
- target audio ↔ audio stays fully visible across both stereo channels, preserving earlier utterances;
- target video ↔ audio uses the same H3 physical time coordinate;
- same-time video ↔ video stays spatially dense;
- cross-time video ↔ video is limited by both temporal distance and a spatial patch radius;
- optional dilated bridge times provide long-range K/V routes outside the local band;
- early layers and an early fraction of denoising can remain dense.

The H3 temporal mapping is **not** guessed from token indices. Video time uses MiniMax-H3's `1,4,4,4,4` latent-token cadence and audio uses its native one-step grid. Both use 40 Hz ticks; `temporal_window` limits video and cross-modal links, while audio self-attention is uncapped.

Both FlexAttention and block-mask construction are compiled. The block mask builder uses a private compiled code object and `H=None` because the policy is head-independent, then broadcast across H3's 56 heads. This avoids the eager `B × H × S × S` boolean grid that otherwise OOMs before the sparse kernel can run. Unsupported sparse contracts abort in strict mode; deliberately dense initial layers/steps remain allowed. Non-strict attention can fall back to the configured dense backend and records the reason. Context-window execution failures always abort the generation.

One compiled `flex_attention` runner is kept per concrete attention shape and mask specialization (segment table plus `temporal_window`, `spatial_radius`, `bridge_stride`), in an LRU between runs. Active sampling retains its working shapes to prevent recompilation on every denoising step; the caches are trimmed when the run ends. Each runner wraps its own private copy of `flex_attention`'s code object, because Dynamo counts recompiles per code object rather than per `torch.compile` wrapper: with one shared code object, a session that met more distinct shapes than Dynamo's budget (and a prompt edit is a new shape, since text tokens are part of it) would silently fall back to eager `flex_attention` and its full `S × S` score matrix. With a code object per runner and the mask's guarded values in the runner key, a runner never meets a second mask specialization, so it never approaches its budget. Evicting a runner resets its code object's Dynamo state and returns the code object to a pool for the next runner, so the LRU bounds retained Dynamo entries and private code objects. It does not evict PyTorch's process-wide Inductor/PyCodeCache modules or unload CUDA kernels; total backend compilation memory is outside this cache's ownership and is not bounded by it.

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

1. chooses overlapping target-video windows;
2. includes the complete shared stereo audio latent in each forward, so earlier utterances remain directly visible;
3. constructs a local `PackedLayout` containing only the selected prompt's actual tokens;
4. transplants global video/audio positions and shared reference/keyframe positions;
5. evaluates H3 jointly and projects each prediction onto the video's local interval and its matching audio interval;
6. accumulates and normalizes predictions in FP32 before returning the model dtype.

A single prompt can stagger interior boundaries using ordered-halving phases. Multi-segment prompts have fixed, exclusive output intervals on the native video-token grid; their windows include neighboring video for context, but only write predictions into the assigned interval. This prevents one beat from disappearing merely because no old window midpoint selected it.

Native Fun ControlNet composes on either side of the context node: its complete control latent is prepared once and sliced by each global video interval. Forge preserves existing block-hook dependencies and keeps base-model NAG/FETA out of the control network's attention.

Absolute positions matter: a window beginning at latent 26 must retain that global frame's native cadence. Full audio visibility does not make an unlimited-memory model; reference size, audio length, and decoded output continue to consume resources.

### H3 Forge — Pipe Timeline Prompt

This node splits a prompt on `|`, encodes every segment independently with MiniMax's Qwen3-VL text encoder, and maps the segments in order across the target timeline. It is not a decorative delimiter passed to one global text encoding.

Two optional inputs make the timeline less toy-like:

- `global_prompt` is repeated inside every independent segment encoding. Put the identity/style/location anchors that truly apply throughout the video here instead of maintaining identical copies by hand.
- `segment_durations` accepts exactly one positive number per segment, separated by commas or newlines. The numbers are relative durations, so `2,18,40` means the same thing whether you think of them as seconds, frames, or beats. They do not have to add up to the output length. Leave the field empty for equal spans.

Segments are separated by the node's `delimiter`, which defaults to `|` so existing workflows keep working. MiniMax spells its own special tokens `<|cutoff|>`, `<|lyrics_start|>` and so on, so text between `<|` and `|>` is never split whatever the delimiter is; without that rule a single `<|cutoff|>` silently turns one prompt into three segments. A literal delimiter is escaped with a backslash, and a delimiter containing a backslash, `<` or `>` is rejected. For prompts that carry MiniMax tokens, `|||` or `%%%` read more clearly than a bare pipe.

Each segment keeps its native token length and is refined independently. No zero-padding tokens enter the refiner or DiT. Reference and target positions use one common timeline origin, even when the text lengths differ. Native lengths may require additional compiled shapes after a prompt edit.

Durations are projected to the nearest boundary of H3's `1,4,4,4,4` decoded-frame cadence. Every representable segment gets model evaluations and exclusive video/audio output ownership. The step-zero plan reports the actual `prompt_frame_cuts`; a sub-grid segment raises before the first denoiser forward. Windows can include neighboring video for context while contributing output only inside their assigned beat. Predictions from different prompts are not blended across the boundary.

This controls conditioning on the native latent grid. A frame-perfect editorial cut between independently generated shots belongs in the video edit; a diffusion model and temporal VAE do not guarantee a photographic hard cut just because the conditioning changes.

Shared reference/keyframe payload **values** must match across segments. Per-segment text lengths and their text tags are retained; differing reference payloads fail visibly.

Use it with `H3 Forge — Chained A/V Context Windows` and an `Empty MiniMax H3 AV Latent`. A single segment is valid and behaves like ordinary global conditioning. Escape a literal pipe as `\|`.

Every segment is encoded independently. Write each local segment as a self-contained, valid H3 prompt in MiniMax's official structured format (`integrated_multimodal_description` / `overall_soundscape` / `non_diegetic_music`). Put concrete identity, wardrobe, location, lighting, voice, and style anchors in `global_prompt`, or repeat them manually when they change between segments. Do not rely on cross-segment shorthand such as “same person”, “continues”, or “remains unchanged”; the local segment encoder cannot see a previous segment. For ordinary 5–15 second work, MiniMax's native `[Shot N] At ...` timing syntax inside one prompt may make pipe prompting unnecessary; pipe scheduling earns its keep on the long-form context-window path where separate local forwards really do need different prompt contexts.

At sampler step zero, the context node prints one compact plan containing the actual latent length, window count, effective stagger phase (and the phase bound when staggering), window/overlap and stride settings, minimum realized adjacent overlap, blend mode, stagger state, total video-latent visits, and a run-length-compressed prompt-to-window assignment. `video_latent_visits` is an overlap accounting ratio, not a wall-time or VRAM prediction.

### H3 Forge — Reference Pipe Timeline Prompt

This node uses native Ref2VA preparation once and independently encodes every prompt against the same prepared presentation. It supports up to nine images, three reference videos, three paired video soundtracks, and three standalone audio references. Video soundtracks pair with the same-numbered video. Connect the video VAE for visual references and the audio VAE for voice/soundtrack references; image-only work does not require an audio VAE. A single segment is valid, and there is no arbitrary eight-segment ceiling.

Use full Ref2VA grammar inside each segment (`subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, `non_diegetic_music`) with stable `<Picture i>`, `<Video i>`, and `<Audio i>` labels. All segments share one reference payload. Native `MiniMaxH3AddGuide` can add image, clip, or audio anchors at specific frames when that is the desired control.

### H3 Forge — Normalized Attention Guidance (experimental)

H3's released checkpoints are guidance-distilled: a normal `BasicGuider` workflow already runs one forward per step and pays no CFG cost, but negative prompts do nothing at CFG 1. This node applies experimental negative-text guidance **without a second complete H3 transformer pass**. Its effect on a particular visual or audio defect needs a controlled comparison.

H3 is structurally on the expensive side of the NAG divide: unlike Wan's external text cross-attention (where NAG costs roughly 12%), H3's text, references, audio, and video share one packed self-attention per block, like Flux (where faithful NAG costs closer to 87%). So this node implements **H3 NAG-Lite**, not faithful NAG:

1. the negative prompt is encoded and refined once;
2. each selected block applies its current native timestep modulation, Q/K normalization and rotary positions to the negative text; projected keys/values are cached within that sigma;
3. the exact NAG formula (`guided = pos·scale − neg·(scale−1)`, L1-renormalized with cap `tau`, alpha-blended) is applied to those two text-conditioned contributions;
4. only the **delta** is injected into target audio/video attention rows before the output projection;
5. A/V↔A/V self-attention, the MLP, and the rest of the positive packed stream are untouched.

In `lite` mode the added attention cost is roughly `target rows × text length` per selected block. `faithful_selective` evaluates an additional packed attention operation with the configured visibility rule at each selected block.

**Remaining approximations:** the sidecar starts from the frozen refined embedding at every selected block rather than evolving through all earlier blocks. In `lite` mode its text attention has a separate softmax denominator. `faithful_selective` swaps the text in the full key stream and reuses the positive attention result; it preserves the configured backend and sparse visibility rule. This mode removes the denominator approximation, while retaining frozen sidecar states. It is a comparison mode, not an established quality ceiling.

Knobs: start at `nag_scale 3.0` (not Wan's 11 — H3 is distilled, single-stream, and jointly generates speech and imagery, so aggressive attention extrapolation has more opportunities to damage identity, voice, or sync). Find stable `nag_tau`/`nag_alpha`, then leave them fixed and tune only the scale. `nag_sigma_end` stops NAG once sigma falls below it, saving compute in the late schedule. `first_block`/`last_block` select the DiT blocks; `video_strength`/`audio_strength` scale the injected delta per modality.

The node composes with `H3 Forge — Sliding Attention + FETA` in either wiring order (both configure one shared H3Forge runtime) and works standalone. Do not stack generic ComfyUI-NAG on top.

## Requirements

- A recent native ComfyUI build with `MiniMaxH3Model` and `PackedLayout` under `comfy.ldm.minimax.model`.
- PyTorch with `torch.nn.attention.flex_attention` for sparse mode.
- CUDA strongly recommended; sparse FlexAttention is not intended as a CPU execution path.
- No extra Python package is required by H3Forge itself.

Core-contract CI is pinned to ComfyUI `250b2e9551a7bc7a8ebb5beb07e0fecd2983e04a`. Use a build containing the native H3 `rope_rotation_table` and Fun ControlNet wrapper contracts. GPU validation uses PyTorch `2.13.0+cu130`; CPU CI exercises the public Torch 2.8 API surface separately.

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

The attention, context, and NAG nodes accept and return `MODEL`; insert them after the H3 model loader and before sampling. They can be wired in any order — the attention and NAG nodes configure one shared H3Forge runtime. The text-only pipe node accepts MiniMax's `CLIP` and returns positive `CONDITIONING`. The reference pipe node additionally accepts the appropriate VAEs and image, video, or audio references, returning both positive `CONDITIONING` and the native AV `LATENT`. The NAG node additionally takes negative `CONDITIONING`.

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
overlap_frames  8
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
H3 context:   25 / 8 / stagger / pyramid
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
: Requested minimum video-latent overlap before staggering; the node default is `8` for a `25`-latent window. Audio overlap is derived from physical H3 time rather than copied index-for-index.

`stagger`
: Moves interior windows for single-prompt runs using bounded ordered-halving phases. The first and last windows remain anchored and the requested overlap remains covered. Multi-segment output ownership stays fixed; its context plan reports `stagger=off`.

`blend`
: `pyramid` applies weights `1,2,...,peak,...,2,1` across each complete window before normalized overlap-add. `overlap-linear` preserves H3Forge's former Kijai-style edge ramp, including first/last boundary handling. `flat` gives every covered prediction equal weight.

Existing saved workflows whose blend is `pyramid` intentionally acquire the new full-window triangle. Select `overlap-linear` to retain the former H3Forge weighting.

### Timeline prompt

`global_prompt`
: An optional shared anchor included in every segment before that segment is independently encoded. It saves repetition; it is not a separate globally attended token bank.

`segment_durations`
: Comma- or newline-delimited positive relative durations, with exactly one value per `|` segment. Empty means equal spans. Each beat owns a contiguous interval at the nearest native token boundaries, reported as decoded `prompt_frame_cuts`. Only predictions for the same beat are fused; sub-grid beats are rejected.

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
- `strict=true` is recommended for development / first GPU tests. Non-strict attention permits dense fallback. A context execution error always aborts instead of retrying a different whole-clip job.
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

The tests cover scheduler coverage, exclusive beat ownership, FP32 fusion, shared audio visibility, reference payload equality, native-length conditioning, sparse-mask cache behavior, and both node/control wiring orders. The integration tests extract the specific contracts from a real Comfy checkout without loading weights; NAG projection tests execute its actual Attention and DiTBlock classes with toy weights and a CPU reference for the fused rotary kernel.

```bash
H3FORGE_COMFY_SOURCE=/path/to/ComfyUI PYTHONPATH=. python -m pytest -q tests
ruff check h3forge tests __init__.py .github/scripts
```

Without `H3FORGE_COMFY_SOURCE`, native-contract tests explicitly skip. CI supplies the pinned checkout and runs them. CPU assertions establish mechanics, not image quality; retained GPU results and their limits live in [audit-repairs.md](docs/audit-repairs.md).

## Review policy

This hobby project uses one review round, fixes to that round, and green CI before merge. Repairs do not trigger an automatic new review round. The former seven-round review router and scheduled nudges have been removed; CI and merge announcements remain.

## Design lineage and nearby work

H3Forge is an original implementation informed by native ComfyUI's MiniMax-H3 packed layout and wrapper seams, [ComfyUI's context-window pyramid and ordered-halving sequence](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/context_windows.py), WanVideoWrapper's context scheduling, and Enhance-A-Video/FETA's off-diagonal temporal-attention gain. The H3 ecosystem now has several useful neighboring projects:

- [ComfyUI-SolAttn-H3](https://github.com/quzopl/ComfyUI-SolAttn-H3) demonstrates disciplined sparse-attention integration, named fallback reasons, self-tests, and per-run telemetry. H3Forge keeps its own H3-aware sparse topology and now applies the same principle of reporting the actual context plan it ran.
- [ComfyUI-YCNodes-MiniMax-H3](https://github.com/yichengup/ComfyUI-YCNodes-MiniMax-H3) introduced a practical global/local Prompt Relay UI with explicit segment lengths. [T8's MiniMax H3 nodes](https://github.com/T8mars/comfyui-minimax-h3-audio-T8) go further with validated frame/second/percent ranges and absolute-timeline projection across sequential long-video segments. H3Forge independently adopts the small composable part that fits its different mechanism: a repeated global anchor and exact unequal relative spans for independently encoded context-window prompts.
- [ComfyUI-MMH3Tools](https://github.com/ckinpdx/ComfyUI-MMH3Tools) and [ComfyUI-H3-Toolkit](https://github.com/wordbrew/ComfyUI-H3-Toolkit) are stronger at H3 frame/audio-grid utilities, workflow validation, and staged long-video mechanics. H3Forge continues to use its own synchronized per-denoise A/V windows with transplanted absolute positions rather than importing a sequential continuation engine.
- [ComfyUI-MiniMax-H3-LongMedia](https://github.com/vizart-vj/ComfyUI-MiniMax-H3-LongMedia) is a fuller end-user long-media system with sequential execution, continuation overlap, streaming/offload, and a memory governor. H3Forge remains the smaller inference-surgery layer and composes with KJNodes' feed-forward chunker instead of duplicating that application shell.
- [H3-Optimizations](https://github.com/Zironic/H3-Optimizations) provides unusually thorough GPU diagnostics and benchmark harnesses. H3Forge's retained artifact and Blackwell matrix serve the same evidence goal, but its benchmark coverage is still less mature.

These are design comparisons, not copied source. In particular, T8 is GPL-3.0-or-later while H3Forge is Apache-2.0; H3Forge's duration routing and reporting here were implemented independently against its existing data path.

No upstream model weights are included.

## License

Apache-2.0. See `LICENSE`.

# ComfyUI-H3Forge

Experimental MiniMax-H3 inference surgery for native ComfyUI:

1. **H3-native sliding/radial sparse attention** using PyTorch FlexAttention.
2. **Video-only FETA-style off-diagonal attention enrichment** derived from Enhance-A-Video.
3. **Synchronized audio/video context windows** with overlap-add blending and absolute H3 RoPE preservation.
4. **Pipe-delimited timeline prompting** with independently encoded, equal-time prompt segments.
5. **Experimental H3 NAG-Lite** — Normalized Attention Guidance adapted to H3's packed single stream via a negative-text sidecar.

The project is deliberately a custom-node patch layer. It does **not** fork or modify files under `ComfyUI/comfy/`.

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

This node splits a prompt on `|`, encodes every segment independently with MiniMax's Qwen3-VL text encoder, and maps the segments in order across equal portions of the target timeline. It is not a decorative delimiter passed to one global text encoding.

All segment embeddings are padded to one token length before sampling, preserving one compiled H3 context shape. Each synchronized A/V context window uses the **complete encoding of the one segment covering its midpoint** — contextualized token slots from independently encoded prompts do not correspond to one another, so hidden states are never interpolated. Around a prompt boundary, adjacent windows generated under different prompts crossfade in **output space** through the context-window overlap-add blend, which is where blending is semantically sound.

Segments must share the same conditioning structure: differing multimodal inserts or presentation tags across segments are rejected at encode time rather than silently stamped with segment 1's metadata.

Segments must not outnumber the context windows that can select them: if a segment's equal-time span contains no window midpoint, H3Forge warns at the first step (or aborts in strict mode) instead of letting that prompt silently vanish. Use fewer segments or a smaller `window_frames`.

Use it with `H3 Forge — Chained A/V Context Windows` and an `Empty MiniMax H3 AV Latent`. A single segment is valid and behaves like ordinary global conditioning. Escape a literal pipe as `\|`.

Every segment is encoded independently. Write each segment as a self-contained, valid H3 prompt in MiniMax's official structured format (`integrated_multimodal_description` / `overall_soundscape` / `non_diegetic_music`), and repeat concrete identity, wardrobe, location, lighting, voice, and style anchors inside every segment. Do not use cross-segment shorthand such as “same person”, “continues”, or “remains unchanged”; those words refer to context the segment encoder cannot see. For ordinary 5–15 second work, MiniMax's native `[Shot N] At ...` timing syntax inside one prompt may make pipe prompting unnecessary; pipe scheduling earns its keep on the long-form context-window path where separate local forwards really do need different prompt contexts.

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

The code was authored against the ComfyUI MiniMax-H3 implementation current on 2026-08-21.

## Install

Extract or clone into ComfyUI's custom nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/RationallyPrime/ComfyUI-H3Forge.git
```

or place the extracted `ComfyUI-H3Forge/` directory there, then restart ComfyUI.

Four nodes should appear:

- `H3 Forge — Sliding Attention + FETA`
- `H3 Forge — Chained A/V Context Windows`
- `H3 Forge — Pipe Timeline Prompt`
- `H3 Forge — Normalized Attention Guidance` (experimental)

The attention, context, and NAG nodes accept and return `MODEL`; insert them after the H3 model loader and before sampling. They can be wired in any order — the attention and NAG nodes configure one shared H3Forge runtime. The pipe-timeline node accepts MiniMax's `CLIP` and returns the positive `CONDITIONING` used by the guider; the NAG node additionally takes the negative `CONDITIONING`.

## First Blackwell bring-up

Do **not** turn every knob on for the first run. Use a known-good H3 workflow and keep seed/prompt/resolution/steps fixed.

### Run 0 — baseline

No H3Forge nodes. Record:

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

Kijai's KJNodes ships focused native-H3 nodes that compose well with H3Forge and belong in the canonical workflow:

- **MiniMax H3 Chunk FeedForward** — chunks SwiGLU along packed-token rows; rows are independent, so this is intended to be mathematically exact while reducing peak activation memory. Recommended by default alongside H3Forge.
- **MiniMax H3 Token Counter** — reports the true packed token count (text, keyframes/references, audio, video). Put it in every diagnostic workflow; packed sequence length is the quantity that actually predicts attention cost.
- **MiniMax H3 Low VRAM Attention** — composes through `optimized_attention`, so it can run under H3Forge sparse mode, but its head grouping invokes the override once per head group. Ordinary sparse attention is mathematically separable across head groups; H3Forge's *sampled* FETA gain is not, and could differ per group. For a clean equivalence test run `KJ Low VRAM Attention + H3Forge sparse + FETA disabled`, and treat that combination as the supported mode until FETA computes one global gain across groups.

### Approximate accelerators

- **ComfyUI-MiniMaxH3-FirstBlockCache** replaces/skips the remaining block stack and warns against combining it with another `double_block` replacement — that is a direct conflict with H3Forge's block stamping and context surgery. Do not combine; benchmark it in isolation with fixed seeds.
- **ComfyUI-Spectrum-MiniMax-H3** forecasts post-transformer features and skips selected transformer evaluations; its own documentation notes it changes the denoising trajectory. Keep it out of any workflow whose purpose is proving H3Forge or NAG behavior.

## Test locally

The included CPU tests cover scheduler coverage, blend-edge correction, block-mask cache keying, bridge semantics, FETA gain routing, Ref2VA/I2VA/FL2VA window transplants (against a faithful fake `PackedLayout`), NAG math and gating, node composition, and sampler-step resolution:

```bash
PYTHONPATH=. python -m pytest -q tests
```

Static syntax check:

```bash
python -m compileall -q .
```

GPU/model correctness cannot be validated without a loaded H3 checkpoint and CUDA device. The first Blackwell session is therefore the integration gate.

## Design lineage

H3Forge is an original implementation informed by:

- native ComfyUI MiniMax-H3 packed layout and wrapper seams;
- the sparse-attention integration pattern demonstrated by ComfyUI-SolAttn-H3;
- WanVideoWrapper's context-window scheduling ideas;
- Enhance-A-Video/FETA's off-diagonal temporal-attention gain.

No upstream model weights are included.

## License

Apache-2.0. See `LICENSE`.

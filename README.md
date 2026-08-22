# ComfyUI-H3Forge

Experimental MiniMax-H3 inference surgery for native ComfyUI:

1. **H3-native sliding/radial sparse attention** using PyTorch FlexAttention.
2. **Video-only FETA-style off-diagonal attention enrichment** derived from Enhance-A-Video.
3. **Synchronized audio/video context windows** with overlap-add blending and absolute H3 RoPE preservation.
4. **Pipe-delimited timeline prompting** with independently encoded, equal-time prompt segments.

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

All segment embeddings are padded to one token length before sampling, preserving one compiled H3 context shape. Each synchronized A/V context window receives the weighted mixture of the prompt segments that physically overlap it, so a window crossing a prompt boundary blends the two contexts instead of imposing a hard latent seam.

Use it with `H3 Forge — Chained A/V Context Windows` and an `Empty MiniMax H3 AV Latent`. A single segment is valid and behaves like ordinary global conditioning. Escape a literal pipe as `\|`.

Every segment is encoded independently. Repeat concrete identity, wardrobe, location, lighting, and style anchors inside every segment. Do not use cross-segment shorthand such as “same person”, “continues”, or “remains unchanged”; those words refer to context the segment encoder cannot see.

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

Three nodes should appear:

- `H3 Forge — Sliding Attention + FETA`
- `H3 Forge — Chained A/V Context Windows`
- `H3 Forge — Pipe Timeline Prompt`

The attention and context nodes accept and return `MODEL`; insert them after the H3 model loader and before sampling. They can be wired in either order. The pipe-timeline node accepts MiniMax's `CLIP` and returns the positive `CONDITIONING` used by the guider.

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

## Important compatibility notes

- **Do not stack `SolAttnH3` and `H3ForgeAttention` on the same model.** Both own `optimized_attention_override`; H3Forge prints a warning if it replaces an existing override.
- H3Forge Context Windows *can* be used without H3Forge Attention.
- FETA can be tested with `mode=dense`.
- `strict=true` is recommended for development / first GPU tests. Use `strict=false` only when you explicitly prefer dense fallback over an aborted generation.
- This is inference experimentation, not a claim that MiniMax trained H3 with this exact sparse topology.
- MiniMax describes native sparse-attention training, while the released ComfyUI inference path is dense. Start from the one-second default and treat shorter windows as an explicit quality/speed sweep.

## Test locally

The included CPU tests cover scheduler coverage, blending positivity, and sampler-step resolution:

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

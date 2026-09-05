# H3Forge 0.3 audit repairs

This change addresses the fifteen findings in the 2026-09-05 audit of H3Forge
`f5f6e6eae0533cad38400cab15d9a9ae0d8e8d85`. Native contracts are checked against
ComfyUI `250b2e9551a7bc7a8ebb5beb07e0fecd2983e04a`. This is an experimental
filmmaking tool: implementation correctness, successful rendering, and perceptual
quality are separate results.

## Findings and resulting behavior

| Finding | Repair | Evidence |
| --- | --- | --- |
| F01 — lost node settings | Reacquire the live transformer-options dictionary after core installs replacement hooks. Bind each cloned model's settings to that dictionary. | Both Attention/NAG orders exercise core's real copying setter. |
| F02 — overwritten ControlNet hooks | Compose block stamps with existing hooks and forward their model, device and cleanup dependencies. Guidance state is active only inside the base DiT block. | Prior hooks run before/after the original block; control attention sees no base-block guidance state. |
| F03 — local control shape and timing | Prepare native Fun ControlNet latents at full-clip shape, expose the corresponding global slice during a window, and restore the full cache afterward. | Core's actual control wrapper and executor run in both orders; full preparation occurs once and subsequent windows receive different global slices. |
| F04 — padded text changes real tokens | Preserve each independently encoded prompt's actual length through refinement, tags and local packed layouts. | Unequal-length contexts reach native layouts without dummy attention keys. |
| F05 — failed windows trigger a full-clip retry | Raise the original failure with window and prompt location. | A failing local forward runs once, including with strict mode off. |
| F06 — strict sparse silently declines | Raise for unsupported sparse contracts; preserve explicitly dense initial layers/steps and non-DiT calls. | Length mismatch fails; the configured dense schedule succeeds. |
| F07 — inaccurate overlap sums | Accumulate weights, predictions and normalization in FP32, then cast once at the model boundary. | Identical BF16 predictions remain identical; the former FP16 overflow case remains finite and exact. |
| F08 — mismatched negative keys | Use native timestep modulation, Q/K normalization and rotary positioning for the negative sidecar. Selective NAG reuses the positive result and its configured dense/sparse visibility rule for swapped text keys. | Actual native DiT/attention projection comparison, per-sigma cache checks and unequal negative-length mask checks. |
| F09 — shared payload values ignored | Compare the actual native reference/keyframe/noise-augmentation values when combining independently prepared segments. | Different reference tensors with identical metadata keys are rejected. |
| F10 — short beats disappear | Assign every prompt an exclusive output interval on the native video-token grid. Give every interval model forwards; log decoded-frame cuts. | The audit's 2/18/40 example covers all three beats, as does a multi-beat clip shorter than one context window. |
| F11 — dialogue history ends at each window | Pass the complete target audio timeline into every video window, retaining global positions. Project and fuse audio outputs only into the corresponding physical-time interval. | Native and two-window dialogue renders, plus focused audio ownership checks; render receipts below. |
| F12 — reference modalities absent | Expose nine image, three video, three paired soundtrack and three standalone audio inputs. Require only the VAEs needed by the supplied media. Allow one segment and remove the arbitrary eight-segment ceiling. | Input contract and native reference preparation checks; loaded-model reference receipt below. |
| F13 — repeated shared preparation | Call native reference preparation once, reuse its prepared tokenizer presentation and shared latent payload, encode texts independently, and retain core's already-refined primary text. | One native preparation call for multiple independent prompts; actual loaded-model reference path. |
| F14 — doubles hide core behavior | Pin and load the small real core definitions under test. CI always supplies that checkout. | Native patcher, wrapper, layout and DiT contracts supplement the math tests; real H3 renders exercise the GPU boundary. |
| F15 — recurring review rounds | Remove the scheduled review router and its repeated-review instructions. Preserve CI and the existing merge-announcement action. | Workflow source now follows one review round, its fixes, green CI, then merge. |

Native prompt lengths also require distinct sparse kernel shapes. A sampling run
retains its working kernels and masks across steps, then trims the caches to eight
runners and 32 masks. Mask construction uses a private compiler code object so
prompt edits cannot exhaust a shared recompile budget and enter eager quadratic
mask construction. These bounds cover Forge-owned caches, not PyTorch's global
backend caches.

## Validation

Local contract and math suite: **128 passed** on Torch 2.8.0, with the pinned core
checkout supplied through `H3FORGE_COMFY_SOURCE`. Ruff and `git diff --check` pass.
Core-dependent tests explicitly skip when the checkout is absent; CI does not
permit that omission.

GPU validation uses a separate checkout on an RTX PRO 6000 Blackwell, Torch
2.13.0+cu130, the pinned core above, the pruned INT8 ConvRot H3 FL2VA checkpoint,
Qwen3-VL 32B NVFP4 text encoder, and native H3 video/audio VAEs. The controlled
dialogue comparisons use seed 12345, 20 `res_multistep` steps, 672×384, 24 fps,
and no LoRA. The original runtime checkout is preserved.

The pinned core's model allocation recorder produced fatal `cudaFreeAsync`
errors in normal asynchronous NAG/sparse execution. Synchronous debugging
completed, but a local pause/resume experiment also failed and was removed.
The supported launch uses `--disable-comfy-compiler`; Forge fails with that
instruction if the recorder is active. Its own sparse kernels remain compiled.
Success under synchronous debugging is not counted as asynchronous acceptance.

The final runtime source matches commit `3958401078c761c95af99cc8257292972abfc932`.
The complete source hashes, prompt IDs, media hashes, frame counts, execution
receipts and ASR words are in [audit-repairs-validation.json](audit-repairs-validation.json).
The [API workflows](audit-repair-workflows) retain the seven final-candidate graphs.
Two initial native/sparse controls predate the allocator guard and compiler-cache
lifetime changes; their separate source archive hash is recorded explicitly.

| Case | Result |
| --- | --- |
| Native baseline, 226 frames / 9.417 s | Completed; “This one is finished.” transcribed once. |
| Strict sparse, two 42/10 video windows, same prompt/seed | Completed: 1,632 sparse calls, 368 scheduled dense calls. The same line was transcribed once. Each window received all 377 audio ticks. |
| Dense attention with two context windows | Completed; the same line was transcribed once, giving a context-only control. |
| NAG-Lite followed by Attention | Completed in normal asynchronous execution: 714 NAG calls and 1,632 sparse calls. The requested line was transcribed once. |
| Shared-softmax NAG followed by Attention | Completed immediately after NAG-Lite in the same process: 714 NAG calls and 1,632 sparse calls, with separate positive and swapped-text masks. The line was transcribed once. |
| Three unequal beats, durations 2 / 3 / 4.4 | Completed with three native text lengths and three windows; frame cuts `0,47,120,226`. Sampled frames show chisel inspection, arms folded with the chisel on the bench, and a turn toward the window. |
| Continuous seam-crossing sentence | Completed; the full sentence was transcribed once from 1.92–6.58 s, spanning the full 3.542–5.875 s overlap region. |
| 481-frame / 20.042 s continuous take | Completed with two 80/10 windows and all 802 audio ticks visible to both. “This one is finished.” appears once at 7.28–8.70 s; no further speech was recognized through the end. The second video window begins at 8.708 s. |
| One segment with an image and verified spoken-audio reference | Completed: 124 frames / 5.167 s, two 25/8 windows. Generated the new line “The edge is ready.” once at 3.30–4.62 s. |

All nine retained acceptance files contain the expected 672×384 video-frame count
and a 32 kHz audio stream. ASR uses Faster Whisper `base.en`, CPU int8, beam size
five, no VAD and no previous-text conditioning. Word timestamps establish that
the continuous sentence crosses both overlap edges; they do not establish
perfect lip-sync or a perceptually inaudible join. Contact sheets were inspected;
these results are not a human listening panel or a motion-quality benchmark.

The first short seam attempt spoke after the overlap and is not counted as a
crossing result. An initial reference crop preceded the baseline's speech; it
was replaced with the verified 5.8–8.1 s crop and rerendered. The successful
reference receipt uses that corrected audio asset, containing “This one is
finished.”, and the first decoded frame of the native baseline. Reference asset
hashes are retained with the local evidence. No voice-similarity score or NAG
music-suppression improvement is claimed.

The local `MiniMax-audit-fixes-evidence` directory retains API graphs, source
snapshots, histories, logs, all generated MP4s, reference inputs, contact sheets
and transcripts, including unsuccessful and non-decisive trials.

## Practical boundaries

- Segment durations control conditioning ownership at native video-token
  boundaries. They do not promise an editorial hard cut or exact spoken-word
  onset. Separately generated shots still need deliberate editing.
- Full audio context removes the missing-input horizon between video windows.
  It does not prove stable voices or seamless dialogue for arbitrary durations.
  Audio attention and full output canvases grow with clip length.
- NAG remains a frozen-negative-text approximation: the negative hidden state
  does not evolve through all previous DiT blocks. Lite mode additionally uses
  separate text-only softmax denominators. Successful guidance calls alone do
  not establish useful suppression or better picture/sound quality.
- Native control-wrapper tests establish cache and hook composition. They do
  not establish perceptual motion-control quality with trained Fun ControlNet
  weights.
- Image/voice references expose core's conditioning capability. They do not
  repair or evaluate the separately trained likeness LoRA.

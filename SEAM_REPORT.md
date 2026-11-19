# H3Forge Seam Euler — local video repair loops

Adds the `H3Forge Seam Euler` SAMPLER node and the optional `seam_repair` switch to both context nodes. Existing graphs default to `seam_repair=False` and the sampler defaults to `repairs=0`. Zero repairs delegates to ComfyUI’s Euler implementation.

**no GPU receipt**. This branch establishes the sampler behavior on CPU. Its effect on Battle Brief and Sparring hall remains unmeasured.

## How a step works

With `seam_repair=True`, the wrapper measures each pair of windows only where both actually write video. It divides the L2 norm of their velocity difference by the L2 norm of the final fused video velocity over the same half-open latent range. This follows the current stagger phase, includes multiple contributors, and excludes holes owned by another prompt. Identical predictions score zero. Exact cancellation of differing predictions scores infinity. Repeated measurements of the same range within one model evaluation use the maximum.

After the Euler step, a range qualifies when its disagreement exceeds `repair_threshold` (default 0.15) and the starting sigma exceeds `repair_until` (0.4). Qualifying ranges share one extra model call. Their clean estimate comes from the model’s fused denoised output. The sampler re-noises the video to the starting sigma in each selected range, with a smoothstep transition over `repair_ramp` latents (4) on each side. The wrapper passes each row’s actual noise level through H3’s native video mask and scales its returned velocity for ComfyUI’s scalar-sigma conversion. The sampler then advances each row by its own remaining distance. Both the local core and the CI pin implement the row timestep inside `MiniMaxH3Model._forward`. No whole-latent fallback was needed.

Each retry sees the original step’s audio input at its original sigma. Only the ordinary Euler call supplies the audio update. Video outside the selected ranges and ramps retains its base-step result. An existing inpaint mask protects an entire 2×2 H3 token if it contains a partially or fully pinned pixel. The sampler publishes one callback per base step, including the revised clean video estimate used by the cockpit’s denoised-output socket.

`repair_noise=fresh` is the default. It uses a private CPU generator keyed by run seed, step and retry. The fresh ramp accounts for independent residuals so its variance matches the sigma sent to H3. `repair_noise=original` reuses the actual input noise after the FreeNoise shuffle. It interpolates the correlated residual directly.

## CPU results

The suite used the checkout’s core Euler function on a fake model. With zero repairs, output, model inputs and callbacks matched exactly. A separate window-fusion test matched the video and audio byte hashes recorded from untouched `05792b7` with diagnostics both off and on.

| ComfyUI source | Untouched branch | Implementation |
| --- | --- | --- |
| Owner’s checkout, `c5751c81` | 227 passed, 11 failed | 245 passed, the same 11 failed |
| CI pin, `250b2e95` | — | 256 passed |

The 11 inherited failures are the two native control-wrapper cases and nine NAG cases. The source-extraction fixtures lack `comfy` and `ComfyAttention` for the newer core. The failing test IDs matched the untouched run. Ruff 0.12.12 passed `ruff check h3forge tests __init__.py`.

Independent review of implementation head `95eeb44` verified the fresh-stream and ramp fixes. At seed 42, the first fresh repair matched zero of 3,280,320 corresponding values in the initial video-noise prefix. For a step from 1.00 to 0.98, one million independent draws at the half-ramp gave a standard deviation of 0.989956 against a target of 0.99.

### Fresh versus original noise

This CPU probe used two four-latent windows on a six-latent video, flat fusion, FreeNoise, seed 42 and overlap `[2,4)`. The packed input held video of shape `[1,2,6,2,2]` and stereo audio of shape `[1,1,2,30]`. Sampling started from 0.8 times the shuffled unit-normal noise. The fake video velocity was `0.25 * video + video[:, :, :1].mean()`, so the two windows could disagree about the same rows. The sigma schedule was `[0.8, 0.6, 0.4, 0.2, 0]`, threshold 0.15, cutoff 0.4, ramp one latent, and budget 0.5.

| Noise | Repairs allowed per step | Extra calls / 4 base | Disagreement before → after each extra call | Last base-call disagreement |
| --- | ---: | ---: | --- | ---: |
| Euler control | 0 | 0 | — | 3.2223 |
| Fresh | 1 | 2 | Step 0 3.0231 → 0.0032 · step 1 0.2411 → 0.9371 | 1.3558 |
| Fresh | 2 | 2 | Step 0 3.0231 → 0.0032 · step 1 0.2411 → 0.9371 | 1.3558 |
| Original | 1 | 2 | Step 0 3.0231 → 3.2919 · step 1 3.5789 → 3.0148 | 2.7748 |
| Original | 2 | 2 | Step 0 3.0231 → 3.2919 · step 0 3.2919 → 3.1650 | 2.8014 |

Fresh noise had lower final disagreement on this fake model. Individual repairs could also increase disagreement. A control using the mean of the entire window in place of its first latent produced equal window predictions to six decimals and spent zero extra calls in every arm. Neither probe measures rendered quality.

## Cost and cockpit selection

`repairs` permits 0–4 retries per step. `repair_budget` defaults to 0.3 of the scheduled base model calls. The sampler checks the integer cap before each retry. For the 30-step, three-window baseline, the cap is 9 extra model calls, each evaluating all three windows. At CFG 1 that is at most 27 extra window forwards on top of 90 base forwards. Diagnostics also cost overlap copies and reductions. Wall time and peak GPU memory have not been measured. The step-zero log records settings and the cap. Retry lines record the step, running extra-call count, latent ranges and disagreement before and after.

In a ComfyUI graph, enable `seam_repair` on either Forge context node and connect the sampler output to `SamplerCustomAdvanced.sampler`. For the supplied cockpit prompts, node 49’s speed preset chooses a sampler **name**. The cockpit would need to expose a Seam Euler choice, pass `seam_repair=True` in `takes.py::windowed()`, and resolve that generation choice in its package-level `sample()` to `nodes.NODE_CLASS_MAPPINGS["H3ForgeSeamEuler"]().get_sampler(repairs=n, repair_threshold=0.15, repair_until=0.4, repair_ramp=4, repair_budget=0.3)[0]`. Its current `comfy.samplers.sampler_object(name)` lookup only covers registered core names. The cockpit was read only.

Neither baseline take was rendered at repairs 1 or 2. Overlap flicker and sharpness, frame-difference peaks, solo-span change, sparring-lunge continuity, render time and the better noise source for H3 all remain open.

# One-way handoff for H3 video context windows

Both Forge context nodes now let a single window write each shared video latent. All windows still attend to their full input span and run once per denoising step. The existing audio fusion, prompt assignment and FreeNoise window plan are preserved.

| `handoff` | Video written in the overlap | Context seen by the reader |
| --- | --- | --- |
| `off` (default) | Existing weighted blend | Original noisy rows |
| `earlier` | Earlier eligible window, through the overlap end | Original noisy rows |
| `later` | Later eligible window, from the overlap start | Original noisy rows |
| `lead` | Same owner as `earlier` | Following windows see the owner's estimate at a fraction of the incoming noise level |

For the cockpit experiment, `ComfyUI-MMH3-ClipLoops/takes.py` function `windowed()` must pass `handoff="earlier"`, `handoff="later"`, or `handoff="lead", lead=0.5` alongside its existing Forge arguments. The step-zero `[H3Forge] context plan` line reports the selected mode and, for `lead`, its value. The default control can omit both arguments.

## What the next window receives

Native H3 receives the video timestep as `1000 * sigma`. Its inner forward returns the signed flow velocity and derives each packed video row's timestep from the supplied mask, with clean rows pinned at the visual conditioning timestep. On the installed core (`c5751c81`), the outer forward multiplies the fused velocity by the caller's original mask after the diffusion wrappers finish.

For an incoming row strength `m`, Forge advances the owner's prediction from `s = m * sigma` to `s' = lead * s`. Interpolating its clean and noise estimates simplifies to `x' = x + (s' - s) * v_owner`. The reader receives that input and a local mask strength of `lead * m`. Zero supplies the clean estimate. One retains the original input and mask. Only the owner's prediction enters the video output. In a triple overlap every reader uses the same owner. Exclusive and rescued prompt beats retain their write intervals.

## CPU verification

| Source used by the tests | Untouched Forge `05792b7` | Implementation `23b5278` |
| --- | --- | --- |
| Installed ComfyUI `c5751c81` | 227 passed, 11 failed | 241 passed, same 11 failed |
| ComfyUI source pinned by CI, `250b2e95` | — | 252 passed |

The installed-core failures have identical pytest node IDs before and after the change. Two native control-wrapper cases encounter a missing `comfy` binding and nine NAG cases encounter a missing `ComfyAttention` binding in the source-extraction harness. Both runs used the local CPU test environment. Ruff 0.12.12 passed `ruff check h3forge tests __init__.py`.

- On the 202 / 85 / 21 plan, each ownership mode assigned exactly one writer to all 202 latents at every supported stagger phase and with stagger disabled. With stagger off the starts were 0, 60 and 117. The owner changed at latents 85 and 145 for `earlier` and `lead`, or at 60 and 117 for `later`. With stagger on, the middle start moved from 58 on step 0 to 64 on step 1.
- A deterministic fake-model forward matched the recorded main-branch video and audio bytes with the option omitted and with explicit `off`. Synthetic audio predictions also fused identically across the ownership modes.
- The recording executor observed three forwards. For `lead` values 0, 0.5 and 1 it checked the next window's actual rows and mask, including incoming strengths 0, 0.5 and 1. Filling the reader's discarded velocity rows with NaNs left every fused video row finite and exactly equal to its owner's output.
- Independent local review accepted implementation commit `23b5278` and checked 72 additional segmented ownership cases.

## Render evidence

**no GPU receipt**. No Battle Brief or Sparring hall renders were made. Overlap flicker and sharpness, peak frame difference and its location, solo-span drift, lunge continuity, peak VRAM and render time remain unmeasured. The CPU tests establish where ownership changes and what H3 receives. They cannot determine whether the dissolve improves or becomes a visible step.

This implementation uses hard ownership boundaries and applies `lead` at every sampled sigma. The visual comparisons against a one- or two-latent crossfade, high-sigma-only lead, and fixed versus staggered seams remain open. The three-window plan still costs three model forwards per step. Lead also copies the reader's local video and mask, updates its shared rows and invokes H3's per-row timestep path. Its wall-clock cost relative to the baseline needs a render.

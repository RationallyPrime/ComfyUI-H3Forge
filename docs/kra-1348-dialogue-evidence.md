# KRA-1348: controlled dialogue comparison

2026-09-04, one RTX PRO 6000 Blackwell Server Edition (97,887 MiB),
ComfyUI `12d5279438bfefc058a269eae805ceab6047777f`.

All four generations used the same workshop prompt containing
`<d>[English] This one is finished.</d>`, seed 12345, 226 output frames,
672×384, 20 `res_multistep` steps, no turbo or identity LoRA, and one context
window (67 video latents, window limit 80). FETA was disabled. The initial
three arms ran H3Forge `627a634b57e3f39df71608764dc15cffd4a58688`.
The candidate ran the merged #11/#12 base `731174ca0f4afd2a8cd0b87e63a1692bcc70d0a5`
plus this PR's fixed audio-pair mask clause.

| Arm | ComfyUI prompt ID | Transcript occurrences |
| --- | --- | --- |
| Sparse, temporal window 40 | `28cde64f-8bc0-4f32-a62e-eff8def872ca` | 4 |
| Attention bypassed | `3b26d2d1-160d-4986-a55d-3ec8c5912f7f` | 1 |
| Sparse, temporal window 200 | `f73ae90e-10c6-4113-ad7c-c19a05c78486` | 2 |
| Sparse 40, full audio self-attention | `50ab1ec5-17b7-404f-bdee-2bde3c874a21` | 1 |

Transcripts were generated locally with Faster Whisper `base.en`, CPU int8,
beam size 5, `condition_on_previous_text=False`, no VAD. The sparse-40
transcript repeats the line at 0.00, 2.42, 4.50 and 6.84 seconds; sparse-200 at
0.00 and 6.36 seconds. The bypass and candidate transcripts each contain only
“This one is finished.” These are ASR observations, not a broad human-rated
speech benchmark. All four ComfyUI executions succeeded. The candidate file
was also inspected: 226 AV1 video frames at 672×384 plus 32 kHz AAC audio,
9.417 seconds total, and a valid workshop image.

The 1000-tick arm suggested by the issue was rejected before execution by the
node's maximum of 256. No generation occurred for that request.

## Chosen rule

Audio-to-audio pairs are always visible across both stereo channels. Text and
reference prefix visibility, video temporal/spatial sparsity, cross-modal
visibility, bridge keys, head broadcasting, and all policy defaults remain as
before. The exemption is fixed, so no widget, additional policy field, or cache
specialization is needed. Tests explicitly check distant, non-bridge stereo
pairs and continued video/cross-modal masking.

KRA-1349's source investigation established that both modalities already use
40 Hz ticks: no 3.19× rescaling is needed. The original cap was one second of
audio self-attention. Raising it reduced repetition, while removing it for the
small audio segment matched the dense-control transcript at this seed.

The candidate is not proof of general dialogue, identity, lip-sync, or
long-duration quality equivalence. The short generation isolates the reported
failure; the separate large-shape attention benchmark measures the cost of the
changed mask without conflating model loading, decoding, or compilation.

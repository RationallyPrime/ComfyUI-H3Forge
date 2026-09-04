# H3 packing investigation: KRA-1349 and KRA-1351

Verified 2026-09-04 against the installed ComfyUI source at
`12d5279438bfefc058a269eae805ceab6047777f` and upstream
`6e3c0bda5a756ec334df449cdc7d4a4685631e91`.
The relevant authority is `comfy/ldm/minimax/model.py`: `_audio_grid`,
`_video_t_grid`, `_video_t_spans`, and `PackedLayout.__init__`.
The diagnostic executed these definitions directly from each source file,
without loading model weights. Both versions produced identical results.
H3Forge baseline: `731174ca0f4afd2a8cd0b87e63a1692bcc70d0a5` (PRs #11 and #12 merged).

## KRA-1349: no time-unit change

Audio latents are **40 Hz**, with positions `origin + arange(audio_t)` repeated
channel-major for stereo. Video uses the `1,4,4,4,4` output-frame cadence scaled
by `5/3` into the same 40 Hz axis. Target audio and video share one origin after
text and reference spans. H3Forge subtracts that common origin implicitly in
its reconstructed target times, so the target-target temporal distances remain
consistent. Its bridge buckets are relative to the target origin.

For a text length of 12 and no references:

- Text endpoints: `0, 11`.
- First audio positions: `12, 13, 14, 15, 16, 17, 18, 19`.
- First video-frame positions: `12, 13.666667, 20.333333, 27, 33.666667, 40.333333, 42, 48.666667`.

Adding one reference image and 40 reference-audio latents moves the target
origin to 53. The grids retain the same increments and relative distances.
The float32 video reconstruction differs from the float64 source by at most
`0.000163` ticks over 427 video latents.

The old comment saying an 80-latent window contains 141–143 audio latents was
wrong. Eighty video latents span 272 output frames, or 11.333 seconds. The
physical overlap mapper produces **454–455 audio latents**. At 427 video latents
and 2417 audio latents, the original starts `[0,69,139,208,278,347]` produce
ranges `[0,454)`, `[390,844)`, `[786,1241)`, `[1176,1631)`, `[1573,2027)`,
`[1963,2417)`. The tail reaches the final audio latent.

Therefore KRA-1349 resolves without changing production time arithmetic.
The original 3.19× discrepancy was derived from an incorrect comment, not the
real packing. `temporal_window=40` means one second for both target modalities.
This establishes the units for KRA-1348; it does not cure its visibility cap.

## KRA-1351: preserve the existing global-coordinate contract

For T2VA, I2VA, FL2VA, and Ref2VA, the diagnostic compared all prefix position
rows and every retained target-video row for every window of a 427-latent run.
Prefix rows were exactly equal to the full layout; target rows were exactly
equal to their global slices, including starts that are not multiples of five.
Keyframes keep their resolved global frame anchors. Reference positions depend
on the common text length and reference descriptors, not the window start.

`combine_conditioning_segments` pads all independently encoded contexts to one
maximum token length and uses the first padded context as the primary. Thus
the supported pipe path gives `clone_window_layout` the same text length for
every window. Reconstructing its prefix regenerates the same global positions.
The prefix-to-later-target distance grows exactly as it does for that same target
in the full run; it is not an additional displacement introduced by windowing.

No prefix shift, target rebasing, generalized segment adapter, or conditioning
mode refusal is warranted by this evidence. The test double previously put text
and prefix coordinates at zero and accepted integer placeholders for descriptors.
It now models nonzero text/prefix origins and actual descriptor fields; tests pin
the prefix-to-target relationship and unequal-encoding padding. Spatial positions
remain explicitly synthetic in the CPU double. The diagnostic against the real
source also checked all spatial coordinates.

This resolves the alleged transplant defect on the supported path. It does not
claim that sparse attention or long-window generation is perceptually equivalent
to full attention; those are separate questions, including KRA-1348.

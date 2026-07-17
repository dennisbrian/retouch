# Color Fidelity Working Queue

**Source:** `RESEARCH_COLOR_RETOUCH_MASSIVE_WINS_2026_07_17.md`
**Status:** Active implementation queue. This is intentionally separate from
the harmony/body task list while those changes are being stabilized.

## Current Slice: K4 + K12

### 1. K4 Real White Balance

**Outcome:** replace hue-rotation pseudo-WB with chromatic adaptation in linear
RGB. A neutral pixel must move under a Kelvin correction and a 6500K / zero
tint call must remain byte-identical.

**Implementation:**

- Add a small CAT16 / von-Kries implementation with sRGB transfer functions
  and Planckian white-point mapping.
- Retire the LCh rotation implementation behind the existing
  `white_balance_kelvin` and `white_balance_tint` controls.
- Keep the current no-op defaults and preserve float32 execution through the
  grade path.
- Test identity, neutral/cast correction, tint direction, gamut safety, and
  skin-hue stability over a Kelvin sweep.
- Audit recipe intent before retaining any non-default Kelvin or tint value:
  this control now corrects an estimated source illuminant to D65, rather than
  acting as a creative warm/cool filter. Preserve intentional scene warmth
  with the grading/film controls, not a fabricated source CCT.

**Owner:** K4 agent. Owns `retouch/white_balance.py`, `retouch/grading.py`,
`retouch/engine.py`, and its dedicated tests.

### 2. K12 Blue-Noise Export Dither

**Outcome:** add deterministic, zero-mean, sub-LSB blue-noise dither only at a
float-to-8-bit delivery boundary. Do not add noise to 16-bit output or to
intermediate uint8 compatibility boundaries.

**Implementation:**

- Add a tiled 64x64 blue-noise-like threshold matrix and a dedicated dithered
  uint8 conversion helper.
- Prove uint8 input stays byte-identical, output stays in gamut, noise is
  bounded to half an LSB, and smooth ramps avoid coherent quantization bands.
- Keep engine wiring out of this subtask; integrate only after K4 completes so
  the final boundary is selected once.

**Owner:** K12 agent. Owns `retouch/precision.py` and dither-specific tests.

### 3. Integration and Visual QA

**Outcome:** select the single final float-to-8-bit boundary after K4, then
wire K12 there without perturbing internal compatibility conversions.

**Gates:**

- Current default renders remain byte-identical except for the deliberate
  dithered 8-bit delivery pixel pattern.
- White-costume / gray-backdrop cast panel visibly neutralizes with WB.
- H5 banding delta does not regress on long face and body gradients.
- 16-bit PNG output receives no dither.
- Do not claim K12 is fully shipped until the engine has one true float-to-8
  bit delivery boundary. Today both face and global-only paths quantize before
  their final return, and applying the helper later would be byte-identical on
  an existing uint8 image. The helper and its regressions are ready; the
  float-output/export refactor is the required integration follow-up.

**Owner:** integration pass after both agents report.

## Next Queue

1. **K2 hue-linearity:** correct the skin hue-line operation before expanding
   `hue_unify` usage further.
2. **Shared light-direction wiring:** consume `lighting.py` in sculpt/relight
   with an unknown-confidence identity fallback.
3. **K1 CAM16 substrate:** start only when K5/K10 needs the appearance model.
4. **K7 Kubelka-Munk skin optics:** strategic moat; schedule only with a full
   dedicated research/implementation week.
5. **K12 delivery-boundary refactor:** retain float output to the exporter,
   then call the dither helper only for an 8-bit write. Keep 16-bit paths
   un-dithered and preserve their source precision.

## Explicit Holds

- Do not start body parity auto-tuning without exposed-body assets and a
  stable full-engine runtime.
- Keep full-face foundation unmixing parked; classical alpha unmixing is not a
  safe general makeup-layer solver.

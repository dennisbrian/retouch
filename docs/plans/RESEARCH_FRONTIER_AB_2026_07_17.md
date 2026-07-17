# RESEARCH — Frontier AB-series: delivery-boundary deep dive + next axes

**Date:** 2026-07-17 · **Author:** Fable research pass (read-only on production code)
**Owner ask:** continue researching the `feat/color-science-k9-fix-and-frontier`
line beyond `RESEARCH_FRONTIER_AA_2026_07_17.md`.
**Method:** direct code verification at HEAD + working tree (every line anchor
below re-checked this pass), plus targeted web verification of the new items'
method sources (§Sources; verify exact citations when implementing).
**Rules inherited:** classical/deterministic, unit-testable, no learned models,
tone-fair, no absolute intensity thresholds, no identity-changing defaults.

---

## 1. Working-tree status (delta since the AA doc)

- **Uncommitted adoption tail of the wire-up slice:** two new recipes in
  `recipes.py` (`cosplay_porcelain_protected_v1`, `cosplay_flash_rescue_v1`)
  adopt the freshly wired `hb_even` / `hb_shift` / `mole_protect` /
  `film_highlight_purity` controls, with matching resolve tests in
  `test_recipe_integration.py`. `scripts/review/visual_qa_wireup_slice.py`
  (untracked) is the slice's visual-QA driver. Not yet committed.
- **Re-verified open at HEAD (grep, this pass):** `to_uint8_dithered` still has
  zero non-test callers; `relight.py` still has zero `light_direction`
  consumers; no Abney/hue-linearity code exists anywhere in `retouch/`.

The AA doc's status table stands. This doc does two things: (§2) turns the one
remaining K-item — the K12 delivery boundary — from a warning into an
implementation-grade map, and (§3) adds four new AB items the K/X/Y/Z/AA
series do not cover.

---

## 2. Deep dive — the K12 delivery boundary, mapped

**Headline finding: 16-bit export currently ships 8-bit information.** The
engine quantizes to uint8 before returning, so every "16-bit" deliverable is a
container upgrade, not a precision upgrade — and `write_image_16bit` already
*logs a warning saying exactly this* when fed uint8 (`io.py:129-130`). That
warning fires on every GUI PNG-16 export today. All of E1/F1's float-resident
internal work is discarded one function call before the exporter.

### 2a. The boundary as it exists (all anchors re-verified)

| Exit | Site | Behavior today |
|---|---|---|
| Engine, face path | `engine.py:1721`, `engine.py:1979` | `np.clip(result_f, 0, 255).astype(np.uint8)` before return |
| Engine, global-only paths | `engine.py:2263`, `engine.py:2534`, `engine.py:2646` | `to_uint8(result)` before return |
| CLI export | `cli.py:276` → `write_image_with_icc` (`io.py:498`) | Exporter is **already float-capable** — proper float→uint16 at `io.py:538-539` — but always receives uint8, so 16-bit goes through the ×257 upscale at `io.py:541-542` |
| GUI export | `gui.py:559-580` | Bypasses the ICC helper entirely: raw `cv2.imwrite` (no ICC embed, JPEG at OpenCV default 4:2:0); PNG-16 via `write_image_16bit` on uint8 (×257, warning path) |
| GUI compare strip | `gui.py:973-974` | `cv2.imencode('.jpg', …, quality 92)` — 4:2:0, see AB1 |

### 2b. Latent mine: two conflicting float range contracts

The two exporters disagree about what "float image" means:

- `write_image_16bit` (`io.py:118-121`): float **[0, 255]**, scales ×257.
- `write_image_with_icc` 16-bit path (`io.py:539`): float **[0, 1]**, scales
  ×65535 (and its 8-bit docstring also says float [0, 1]).

Harmless today only because both receive uint8. The moment the engine starts
returning float (the whole point of the K12 refactor), routing engine-float
[0, 255] into `write_image_with_icc` blows out the render by ×255. The
refactor must unify this contract *first* or it will ship a spectacular
regression. Recommendation: standardize on the engine's native float [0, 255]
(documented at `io.py:118`), make `write_image_with_icc` accept it, and delete
the [0, 1] assumption — one convention, asserted in tests at both exporters.

### 2c. Refactor spec (implementation-grade)

1. **Engine float output mode.** `process(..., output_dtype="uint8"|"float32")`
   (or a float-carrying `ProcessingResult` field): skip the final quantize at
   the five exit sites above when float is requested. Default stays uint8 —
   byte-identical for all existing callers.
2. **Unify the float contract** per §2b before any wiring.
3. **8-bit writes** call `to_uint8_dithered` (`precision.py:181`) — the only
   place it is ever called. **16-bit writes** use the existing float→uint16
   paths, un-dithered (already correct per TODO gates).
4. **Route GUI export through `write_image_with_icc`** — closes the ICC-embed
   gap and the PNG-16 warning path in the same move.
5. **Acceptance:** existing `TODO_COLOR_FIDELITY` §3 gates, plus one direct
   proof precision survived: a smooth synthetic ramp rendered to 16-bit PNG
   must contain **> 256 unique levels per channel** (today it cannot).

**Effort:** ~2–3 days. **Risk:** low once §2b is done first; the default path
is byte-identical by construction.

---

## 3. New research — AB-series (not covered by K/C/E/X/Y/Z/AA)

| # | Item | What it unlocks | Effort | Deps |
|---|---|---|---|---|
| **AB1** | 4:4:4 delivery chroma | Stops re-destroying the chroma edges AA1 restores; sharper lips/costume edges in every JPEG/WebP deliverable | ~½ d | rides §2 |
| **AB2** | K2 correction data + oracle | Upgrades K2 from a direction to a buildable spec (dataset named, insertion point located) | folds into K2 | none |
| **AB3** | Flyaway-strand cleanup | The #1 remaining manual retouch step; wigs shed — cosplay-corpus gold | ~1 wk | Y1 ✅ |
| **AB4** | Purple-fringe / lateral-CA cleanup | Removes the violet halo on backlit/high-contrast edges before skin/eye ops touch it | ~3–5 d | Z7 detector |

### AB1 — 4:4:4 delivery chroma (rider on the §2 slice)

**Gap:** every JPEG/WebP the pipeline delivers is chroma-subsampled 4:2:0 by
encoder default — OpenCV `imwrite` (GUI export path, and no sampling flag
passed) and Pillow JPEG save (CLI path via `write_image_with_icc`, no
`subsampling` argument) both default to 4:2:0. AA1 exists because 4:2:0
*input* bleeds chroma at lip/blush/costume edges; today the export path
re-introduces the identical defect on the way out. AA1 + AB1 together close
the chroma loop (restore on ingest, preserve on delivery).

**Fix:** Pillow `save(..., subsampling=0)`; OpenCV
`IMWRITE_JPEG_SAMPLING_FACTOR = IMWRITE_JPEG_SAMPLING_FACTOR_444` (available
since OpenCV 4.6). Also apply to the GUI compare strip (`gui.py:973-974`) —
that strip is the visual-QA surface, and today it softens exactly the edges
the slider work is judged on.

**Tests:** encode→decode round trip on saturated synthetic edges — chroma MSE
drops vs 4:2:0 at equal quality; file-size delta measured and documented
(expect ~5–15% larger; acceptable for a delivery format).

### AB2 — K2 hue-linearity: name the data, locate the insertion point

The K-queue's K2 ("Abney correction for the skin hue-line ops") has been
direction-only. Making it buildable:

- **Data:** the standard constant-perceived-hue oracles are **Hung & Berns
  1995** (published dataset, Zenodo record 3367463; constant luminance) and
  **Ebner & Fairchild 1998** (varying luminance — the right oracle for our
  concern, since the shipped `hue_unify` drift is L-dependent). Both are the
  datasets every hue-uniform space (IPT, OKLab itself) was evaluated against.
- **Shape:** fit **once, offline**, a small Δh(h, L) correction table over the
  skin quadrant (orange-red branch only — keeps the table tiny) from the
  Ebner–Fairchild loci mapped into OKLab; ship as literals. At runtime,
  convert to corrected-hue coordinates, do the existing constant-hue pull,
  convert back. Insertion point: `skin.py:1266` (`unify_hue_line`) and the
  locus-pull code path that shares its OKLCh conversion.
- **Tests:** Ebner–Fairchild skin-branch loci map to constant corrected hue
  within tolerance; Fitzpatrick I–VI synthetic ramp hue drift measurably drops
  vs the current path; byte-identity with correction disabled.

### AB3 — Flyaway-strand cleanup (the wig item) ⭐ new capability

**Gap:** stray-hair cleanup is the most-cited remaining manual step in pro
retouch workflows, and the cosplay corpus is worst-case: **wigs shed** —
strands across the face, over skin, and against clean backdrops. Nothing in
K/X/Y/Z/AA touches it; commercial tools ship it as a learned-model feature,
but the classical pipeline is proven in a neighboring field.

**Method (transferred from dermoscopy hair removal, a mature classical
literature):** multi-scale curvilinear matched filters (derivative-of-Gaussian
/ black top-hat) detect thin dark-or-light ridges; hysteresis + length/curvature
gating separates strands from skin texture; inpainting fills the strand mask.
We hold strictly better cards than dermoscopy: a hair mask, a face mask,
landmarks, and a **live PatchMatch engine** (Y1, shipped) for the fill.

**Scope guard (identity):** operate only in the band *outside* the hair
silhouette — skin and backdrop within a dilation of the hair mask, plus the
face region. Never thin the hair mass, never touch the hairline shape. Opt-in;
strength parameter = maximum strand width in face-width-relative units (no
absolute pixel thresholds; detection contrast is margin-over-local-baseline,
tone-fair per the standing rule).

**Tests:** synthetic strands composited over flat and textured backgrounds are
removed with no residual ridge energy at the strand scale; hair-mass region
byte-identical; strand-free image byte-identical; darker-skin fixtures in the
detection sweep (contrast gate is relative, must not silently fail on Fitz V–VI).

### AB4 — Purple-fringe / lateral-CA cleanup (input-side rider on Z7)

**Gap:** axial CA / purple fringing — the violet halo at high-contrast
backlit edges (convention windows, dark wigs against bright backdrops,
specular armor edges) — sits upstream of eye/hair/costume ops and reads as a
skin-adjacent color defect none of our ops model.

**Honesty tier:** unlike AA's items this literature is heuristic/patent-tier,
not IPOL-canonical — treat it as bounded defect removal, not color science.
Method sketch from the published/heuristic art: detect fringe pixels by
*geometry + color jointly* (blue-violet chroma spike immediately adjacent to a
clipped or high-gradient luminance edge), then replace fringe chroma with an
estimate from the nearest unaffected pixels along the gradient direction.

**NO-GO edge:** never a global purple desaturation — the gate is adjacency to
a luminance edge/clip, so iris color, purple wigs, and purple costumes are
untouched by construction. Sequenced after Z7's detector exists (shares the
provenance/edge analysis machinery).

---

## 4. Lighting wiring is a ~1-day item, not a research item (spec)

Re-verified: `sculpt()` **already accepts** explicit `light_azimuth` /
`light_elevation` (`relight.py:422-423`); its only call site
(`perf_optimizations.py:658`) simply doesn't pass them, so sculpt falls back
to its internal blurred-L-gradient estimate. Meanwhile every `FaceContext`
carries the confidence-rated `LightDirection` (`detection.py:59`,
`engine.py:2868`) with zero consumers.

**Wiring:** at the call site, if the user gave no explicit azimuth and
`face_ctx.light_direction.is_known` (optionally require `source ==
"catchlight"` for the conservative first cut), convert the 2D unit vector to
azimuth via `atan2` and pass it; leave elevation `None` (the 2D estimate has
none). Coherence rule: an explicit user `relight_azimuth` wins for both
`relight()` and `sculpt()` so the two never shade in different directions.
Byte-identical when the estimate is `unknown`. No `relight.py` changes needed.

---

## 5. Recommended attack order (v2 — supersedes AA §3)

1. **The delivery-fidelity slice:** §2 K12 boundary refactor (contract unify →
   engine float output → dither at 8-bit write → GUI through the ICC helper)
   **+ AB1** 4:4:4 chroma **+ AA8** rides along. One PR closes: true 16-bit,
   banding-free 8-bit, ICC-correct GUI export, chroma-preserving delivery.
2. **§4 lighting wiring** — one day, closes a shipped-quality incoherence.
3. **AA5 + AA6 calibration/QA pack** — unchanged from AA.
4. **K2 with AB2's data** — now spec-complete; grows more urgent as
   `hue_unify` adoption spreads (two more recipes in this very working tree).
5. **AA2 BGU**, then flagships per owner preference (X1/X3/Y3). **AB3** is the
   next new user-visible capability after the flagships; **AB4/AA1** remain
   fillers behind Z7.

## Bounds / NO-GO (inherited + new)

- AB1 changes encoder settings only — zero pixel-pipeline changes; if a
  deployment needs small files over edge fidelity, quality stays the knob,
  subsampling does not silently return.
- AB2's correction table is fitted offline from published perceptual data and
  shipped as literals — no runtime fitting.
- AB3 never edits inside the hair silhouette or moves the hairline; detection
  thresholds are relative-to-local-baseline (tone-fair), never absolute.
- AB4 never triggers on color alone — geometric adjacency to a luminance
  edge/clip is a hard precondition.

## Sources

- OpenCV issue #22052 + PR #22064 — `IMWRITE_JPEG_SAMPLING_FACTOR`, default
  4:2:0 behavior (AB1).
- Pillow *Image file formats* / *JpegPresets* docs — JPEG `subsampling`
  default behavior (AB1).
- Hung & Berns 1995, *Determination of constant Hue Loci…* — dataset on
  Zenodo, record 3367463; Ebner & Fairchild 1998, *Development and Testing of
  a Color Space (IPT) with Improved Hue Uniformity*, CIC6 (AB2 oracles).
- Dermoscopy hair-removal literature: cascaded Hough / top-hat + harmonic
  inpainting (PMC9777124) and multi-scale curvilinear matched-filter +
  coherence-transport inpainting line of work (AB3 method transfer).
- Purple-fringe art: US patents 7,577,292 / 8,797,418 / 8,786,709 (adjacency
  + estimated-chroma replacement); `mjambon/purple-fringe` heuristic tool
  (AB4; heuristic-tier, noted as such).

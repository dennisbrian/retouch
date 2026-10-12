# RESEARCH — Structure, Healing & Workflow Intelligence (Y-series)

**Date:** 2026-07-17 · **Author:** Fable research pass (read-only)
**Owner ask:** a third research axis beyond the two same-day docs
(`RESEARCH_COLOR_RETOUCH_MASSIVE_WINS_2026_07_17.md`,
`RESEARCH_SKIN_COLOR_FRONTIER_X_2026_07_17.md`).
**Coverage logic:** those two own color *rendering* and skin *optics*. What
remains unexplored is **structure (geometry), reconstruction (healing), and
decision intelligence (auto-grading, delivery)** — verified absent at HEAD by
grep: no perspective correction, no PatchMatch, no Poisson compositing, no HDR
gain-map export, no cast-shadow rescue, no closed-loop auto-grade (F9
`image_analyzer.py` is open-loop heuristics), no character palette lock.
**Rules inherited:** classical/deterministic, no learned models, tone-fair, no
identity-changing defaults. No file overlap with in-flight agents (K4/K12,
X-series, body/marks fix agent).

---

## 0. TL;DR ranking

| # | Item | What it unlocks | Effort | Risk |
|---|---|---|---|---|
| **Y1** | **PatchMatch content-aware heal** | Photoshop-grade fill for blemishes/strays/backdrop — no model download | ~1–1.5 wk | low |
| **Y2** | **Perspective / focal-length face correction** | The "85mm look" from close-up shots — kills selfie jaw/nose distortion | ~1.5–2 wk | med |
| **Y3** | **Closed-loop auto-colorist solver** | One-click grade that *converges on targets* instead of suggesting deltas | ~1–1.5 wk | low |
| **Y4** | **Character palette lock (cosplay moat)** | Wig/costume held to the character's canonical colors, shading preserved | ~1–1.5 wk | med |
| **Y5** | **Flash cast-shadow rescue** | Softens hard chin/nose/brim shadows from on-camera flash | ~1.5 wk | med-high |
| **Y6** | **HDR gain-map export (ISO 21496-1)** | Catchlights/speculars genuinely *glow* on HDR phones/displays | ~1 wk | low |
| **Y7** | **Band-limited chroma processing** | Color blotch evening with zero luma-texture cost | ~2–3 d | low |
| **Y8** | **Poisson (gradient-domain) blending for heal/transplant** | Kills residual seams in every clone/heal composite | ~2–3 d | low |

Attack order: **Y1 + Y8 as one healing slice** (Y8 is Y1's compositor), then
**Y3** (it multiplies the value of every shipped color op), then Y2/Y4 by
owner preference (visible-wow vs cosplay-moat), Y6/Y7 as fillers, Y5 last
(highest visual-QA burden).

---

## Y1 — PatchMatch content-aware heal ⭐ the reconstruction ceiling, classically

**Gap:** every heal path today is Telea diffusion (`blemish.py`,
`backdrop.py`, `heal_region`) — fine at r≤5px, smeary beyond. The LaMa track
(B3) is wired to a model that was never downloaded, so its practical fallback
is still Telea. Large blemishes, stray-hair clusters, backdrop logos, and the
skin regions under removed marks all read "healed" today.

**Method (Barnes et al. 2009 — the algorithm behind Photoshop's CAF):**
randomized nearest-neighbor field over patches (5–7px), propagation +
random-search iterations, coarse-to-fine over a pyramid; synthesize the hole
from real image patches → real *texture*, not diffusion soup. Constrain the
source region to the same semantic mask (skin fills from skin, backdrop from
backdrop — masks already exist) so it never pulls costume into a cheek.

**Codeable plan:** `retouch/patchmatch.py` — vectorized NumPy NNF (the inner
loop vectorizes over the hole pixels; at proxy resolution holes are small),
pyramid levels from proxy→full, Y8's Poisson blend as the compositor. Wire as
`heal_engine="patchmatch"` beside Telea in the heals loop and
`FreckleRemover`'s inpaint. Deterministic via fixed RNG seed.

**Tests:** synthetic texture (noise, fabric weave, skin high-band) with a
punched hole → spectral distance of fill vs surroundings beats Telea by a
measured margin; pore-spectrum QA (`detect_pore_spectrum_distance`) on healed
skin patches; golden no-op when mask empty.

## Y2 — Perspective / focal-length face correction ⭐ the visible-wow item

**Gap:** close-camera portraits (selfies, tight con-floor shots) carry
perspective distortion — enlarged nose, narrowed jaw, pushed-back ears. Users
read it as "unflattering," not as geometry. Nothing in the engine addresses
it; `FaceReshaper` is 2D liquify with no camera model.

**Method (Fried et al., SIGGRAPH 2016 — "Perspective-aware Manipulation of
Portrait Photos" — warp-based, no NN):** MediaPipe's 478 landmarks carry
metric-normalized z on a canonical 3D face. Fit a weak-perspective camera
(subject distance from inter-ocular size), re-project the mesh as if shot
from a longer distance/focal length, take the per-vertex 2D displacement,
extend it smoothly beyond the face (face-anchored falloff into hair/body,
exactly the liquify infrastructure), warp.

**Product shape:** one slider `perspective_correct` (0 = off default, 100 =
full 85mm-equivalent), auto-estimated distance shown as a hint. Off by
default — geometry is a creative control per the standing non-goal rules.

**Tests:** synthetic 3D-rendered head at 30cm vs 1.5m — corrected 30cm render
converges toward the 1.5m ground truth (landmark RMS); identity guard —
landmark *topology* unchanged, correction monotonic in slider; profile faces
fade out via the existing yaw guard.

## Y3 — Closed-loop auto-colorist solver ⭐ multiplies every shipped op

**Gap:** F9 (`image_analyzer.py`) is **open-loop**: measure once, suggest
params from heuristics, hope. Nothing verifies the suggestion landed; a
too-dark render stays too dark.

**Method:** feedback optimization at proxy resolution. Objective = weighted
perceptual targets, all already implemented as metrics: skin at its preferred
locus (C1 tables + K8 ΔE00), neutral anchor after WB (K4, in flight), tonal
distribution targets (percentile L bands per look family), σ_C uniformity,
banding/clip guards (H5, `detect_clipping`) as constraints. Optimizer:
coordinate descent or Nelder–Mead over the ~6–10 continuous grade params
(exposure, contrast, WB, saturation, locus strengths) — each eval is a
proxy-res render of *only the global stages*, ~50–100ms, so a full solve is
2–5s. Deterministic given the image.

**Product shape:** `--auto-grade [look-family]`: solve, then hand the user
the *resolved params* (they remain editable — it's a starting point solver,
not a black box). F9 keeps its role as the initializer (its suggestion seeds
the search — nothing is thrown away).

**Tests:** cast/underexposed corpus → post-solve skin ΔE-to-locus and
neutral-error drop below thresholds on every image; solver idempotent
(re-solving a solved image moves <ε); constraint violations never increase.

## Y4 — Character palette lock (the cosplay moat feature)

**Gap:** cosplayers care intensely that the wig/costume matches the
character's canonical colors, and venue light + WB correction destroys that.
`harmonizer.py` snaps hues to *templates*; `style_transfer` moves global
looks; nothing can hold "this wig = exactly Miku teal" while keeping the
shading real.

**Method:** treat wig/costume regions as dyed materials: reflectance =
dye color × shading (multiplicative). Per region (hair parse / cloth mask /
user brush): estimate the region's current dye chromaticity (robust median in
OKLCh, shading-free by construction since shading is ~pure L), compute the
transform dye→target (reference swatch: eyedropper from character sheet, or a
saved per-character palette JSON), apply as chroma/hue rotation with L
untouched and the specular layer (X3, or today's `extract_specular`)
explicitly protected so highlights stay illuminant-colored — that is what
keeps recolors looking like *material* instead of paint-bucket.

**Tests:** synthetic shaded cylinder in dye A → recolor to dye B matches a
ground-truth re-render (hue/chroma within tolerance at every shading level);
speculars unshifted; skin/face pixels provably untouched (region containment).

## Y5 — Flash cast-shadow rescue

**Gap:** on-camera flash carves hard shadows (under chin, nose, hat brim,
glasses) — the single ugliest artifact of convention photography. Only tonal
`shadows`/`shadow_lift` exist: they lift *everything* dark, shadow or not.

**Method (classical, two stages):** (1) detect *cast* shadow on skin: region
where luminance drops sharply but chromaticity ratios stay near-invariant
(Finlayson-style illuminant-invariant cue), boundary = sharp L gradient
co-located with the chroma-invariant transition, geometry-gated to below-nose/
below-chin/brow zones via landmarks; (2) soften: attenuate the shadow-boundary
gradients (gradient-domain edit, shares Y8's Poisson solve) + partial lift of
the shadow interior toward ambient, bounded, never to zero (a de-shadowed
face reads AI-fake). Confidence-gated like `lighting.py`: no confident
detection → byte-identical.

**Risk note:** highest visual-QA burden in this doc (false positives = lifted
real form shadows = flat face). Ship behind a default-off param with the H
harmony/QA metrics watching; mandatory montage review.

## Y6 — HDR gain-map export (ISO 21496-1)

**Gap:** output is SDR 8-bit. Every 2025+ iPhone/Pixel/high-end display
renders gain-map JPEGs with real highlight headroom — catchlights, rim light,
and speculars *emit*. For a portrait product this is disproportionate wow per
engineering hour, and none of the competitor desktop tools ship it.

**Method:** the engine already has float scene data pre-tonemap. Gain map =
log2 ratio between an HDR rendition (highlight rolloff relaxed, headroom up
to +2–3 stops on speculars/emissives) and the SDR base, downsampled ×4,
embedded per ISO 21496-1 / Adobe gain-map container (MPF dual-image JPEG —
`libultrahdr` bindings or pure-Python container writing). SDR base remains
byte-identical to today's output → zero regression risk for SDR viewers.

**Tests:** container parses in reference viewers; SDR base unchanged
(checksum); gain-map values bounded; round-trip decode reconstructs the HDR
rendition within tolerance.

## Y7 — Band-limited chroma processing (cheap, colorist-standard)

**Gap:** `chroma_even` compresses chroma *variance* globally;
blotch-scale color mottle (broken capillaries, uneven flush) lives in a
specific spatial band. Colorists fix it with chroma-only midband blur —
luminance texture untouched, so it cannot plastic-ify.

**Method:** split a/b (or OKLab a/b) channels into the existing 3-band
structure (reuse `FrequencySeparator` on chroma planes), attenuate only the
mid band of chroma, leave L bands alone. ~60 LOC on shipped machinery, one
param (`chroma_blotch_smooth`). Composes under X2 later (hemoglobin-selective
is strictly better) but ships value now.

**Tests:** synthetic luma-texture + chroma-blotch image → blotch chroma
variance drops, luma high-band energy byte-stable; I–VI sweep.

## Y8 — Poisson / gradient-domain compositing for heal & transplant

**Gap:** `texture_transplant`, heals, and inpaint results composite via alpha
(`blend_masked`/`inpaint_and_blend`) — low-frequency mismatch between donor
and target leaves visible "sticker" seams that the seam QA detector
occasionally flags. Gradient-domain (Pérez 2003 seamless cloning) composites
*gradients* and solves for absolute levels — seams vanish by construction.

**Method:** `cv2.seamlessClone` for the simple cases; small multigrid Poisson
solve (or Fourier solve on rectangular ROIs) where mixed gradients are needed
(keep target texture where donor is flat). Adopt inside `texture_transplant`
and the heal paths as `blend="poisson"`. ~2–3 days, mostly tests.

**Tests:** donor patch with deliberate +10L offset composites seam-free
(gradient at boundary below threshold) vs visible step under alpha blend;
`detect_seam` scores improve on the transplant fixtures.

---

## Bounds / NO-GO

- **No neural fill, no generative reconstruction** — Y1 is explicitly the
  classical ceiling; if it ever falls short, that is a *future* B3/LaMa
  decision, not this doc's.
- **Y2 ships off-by-default** and never composes into "beauty" recipes —
  geometry stays a deliberate creative act (standing anatomy non-goal).
- **No burst/multi-frame work** — P5 stays parked (dependency-blocked).
- **Y4 never touches skin pixels** — palette lock is a material op; skin
  color remains owned by the C1/X2 skin stack.
- **Y5 never removes shadows fully** — bounded softening only.

## Sequencing vs everything in flight

- Zero file overlap with: K4/K12 agents (`white_balance.py`, export
  boundary), X-series scopes (`spectral.py`, `skin_chromophore.py`,
  `specular.py`, `film.py`), body/marks fix agent (`engine.py` body stage,
  `marks.py`, `harmony.py`, `parsing.py`). Y-work lands in new modules
  (`patchmatch.py`, `perspective.py`, `autograde.py`, `palette_lock.py`,
  `gainmap.py`) + `blemish.py`/`skin.py` integration points.
- Y6 wants K12's export-boundary refactor merged first (same exit point).
- Y3 wants K4 merged (neutral-anchor target) but can start with the WB term
  stubbed.
- Y8 before or with Y1 (its compositor). Y7 anytime. Y5 last.

## Sources (canonical; verify exact citations when implementing)

- Barnes, Shechtman, Finkelstein, Goldman — *PatchMatch*, SIGGRAPH 2009 (Y1).
- Fried, Shechtman, Goldman, Finkelstein — *Perspective-aware Manipulation of
  Portrait Photos*, SIGGRAPH 2016 (Y2).
- Pérez, Gangnet, Blake — *Poisson Image Editing*, SIGGRAPH 2003 (Y8, Y5).
- Finlayson, Hordley, Drew — illuminant-invariant shadow detection line of
  work (Y5).
- ISO 21496-1 / Adobe gain map specification; Google `libultrahdr` (Y6).
- Nelder & Mead 1965; standard derivative-free optimization (Y3).

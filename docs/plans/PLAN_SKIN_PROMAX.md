# Skin — Pro Max Frontier (new *axes*, not recombinations of R1–R15)

**Status:** 📋 RESEARCH (2026-07-11, Opus) · **Parent:** MASTER_PLAN Phase 2 skin work.
**Purpose:** The R1–R15 / M1–M6 / I1–I5 sweep declared skin "exhausted — anything further is
recombination." That's true *within the axes those docs chose*: reflectance/color (R7/R10),
form (R9/C2), finish (R12), signal-processing bands (R13), measurement (M), and the additive/inverse
move (I). This doc challenges the exhaustion claim by naming **axes none of them operated on.** Each
section below is a *dimension* the prior work never varied, not a new recipe inside an old one.

The test I held every item to: *"Could this have been written as a bullet under an existing R/M/I item?"*
If yes, it's cut. What remains are genuinely new degrees of freedom.

---

## P1 — Skin as a layered translucent medium (full SSS forward model) ⭐⭐ moonshot
**New axis: depth.** R7/R10 split color into melanin+hemoglobin *concentrations* in one plane. R9 splits
albedo×shading. But real skin is a **stack**: oil/surface, epidermis (melanin), papillary dermis
(oxy-hemoglobin), reticular dermis (deoxy-hemoglobin, collagen scatter). Light does subsurface transport
*between* these layers. Every prior idea treats skin as one or two 2-D maps; none models the **vertical
transport**.

The move: a lightweight **Kubelka–Munk / dipole-diffusion forward model** parameterized by per-pixel
layer concentrations (melanin fraction, blood fraction, oxygenation, dermal scatter). Fit the model to
the observed pixel (inverse-render), *edit the physiological parameters* (reduce blood, keep melanin,
raise oxygenation for a "healthy flush" instead of "redness"), then *forward-render* back to RGB. This
is strictly more powerful than R7's linear unmixing: it explains **why** the R7 maps look the way they
do and lets you make edits that are physiologically impossible to express as 2-D map arithmetic (e.g.
"same total blood, more oxygenated" — the difference between sallow and radiant). No consumer tool, and
essentially no prosumer tool, has a forward skin-optics model. This is the true ceiling of the "physics"
line R7→R9→R12 was climbing.

**Why not a recombination:** R7 is *concentration maps in one plane*; this is *transport across depth*.
You cannot express oxygenation-at-constant-total-blood in R7's basis.

---

## P2 — Perceptual-loss / adaptation-aware processing ⭐ novel
**New axis: the observer.** Every R/M/I item optimizes a *pixel* quantity (ΔE, variance, spectrum). None
optimizes what the **human visual system actually perceives** after its own adaptation. Two consequences
the sweep missed:

- **CSF-weighted smoothing budget.** The contrast sensitivity function means blotch at ~0.5–2 cycles/deg
  is *far* more visible than either fine pore noise or very-low-freq shading. Current frequency work
  spends smoothing uniformly across the band. Re-weight the smoothing budget by the CSF at the image's
  *expected viewing distance/size* (thumbnail vs print differ enormously) — spend where the eye looks,
  preserve where it can't tell. This is M4/R13 done in *perceptual* frequency, not cycles/pixel.
- **Local-adaptation-aware evening.** The eye adapts locally; a blotch next to bright background is less
  visible than the same blotch mid-cheek. Weight S3/blotch correction by local retinal contrast
  (surround luminance), not absolute pixel deviation. You get identical *perceived* evenness for less
  actual smoothing → less plastic.

**Why not a recombination:** R13/M4 measure and act in physical spatial frequency; this reparameterizes
the entire budget by the *observer's transfer function and adaptation state*. Different objective.

---

## P3 — Output-conditioned processing (the render target changes the retouch) ⭐ novel
**New axis: the destination.** The whole pipeline produces *one* answer regardless of where the image
goes. But the correct retouch for a 400px Instagram thumbnail, a 4K phone screen, an A2 print, and a
backlit OLED are **measurably different** (viewing distance, gamut, surround, whether pores are even
resolvable). Nothing in R/M/I conditions on output medium.

The move: an **output profile** (size, gamut, medium, surround) that reparameterizes the *existing*
knobs — texture re-injection scale (R13c already gestures at this for export, but only scale), smoothing
CSF budget (P2), gamut-aware saturation ceiling, print-dot-gain-aware micro-contrast. One tuning, N
correct renders. This is the natural home for R13c's export-scale idea generalized to *every* medium
axis, and it's what "pro max" literally means to a working retoucher delivering to many channels.

**Why not a recombination:** R13c retargets one parameter (texture scale) for print resolution; this is a
*conditioning variable* threaded through the whole skin stack.

---

## P4 — Skin↔makeup unmixing (the cosplay-critical blind spot) ⭐⭐ high-moat
**New axis: what's on the skin vs the skin.** A3/R11 handle *makeup-aware smoothing* (don't smooth across
paint edges) and face-paint repair. But no idea **separates the makeup layer from the skin layer** as an
unmixing problem. For the cosplay audience this is the whole game: foundation, body paint, contour, and
prosthetic edges sit *on top of* real skin, and the retoucher wants to fix one without disturbing the
other.

The move: model the observed face as `skin_reflectance ⊗ makeup_layer` (semi-transparent overlay with
its own color + coverage α) and **unmix** them — using the P1 physiological model as the skin prior
(real skin obeys the melanin/blood constraints; makeup doesn't). Then you can: even the *foundation
coverage* without touching skin tone underneath; fix *skin* (redness, oil) *through* light makeup;
detect where foundation is *caking* in pores/creases (coverage-α spiking at high freq) and fix only
that; verify contour/paint symmetry independent of face shading. This is the unmixing sibling of P1 and
the single most defensible cosplay-skin feature imaginable — literally nobody separates the paint from
the person.

**Why not a recombination:** A3/R11 *mask around* makeup; this *decomposes* it as a distinct optical
layer with its own parameters. Masking ≠ unmixing.

---

## P5 — Temporal / burst coherence ⭐ novel (the un-opened axis, region-agnostic but skin-critical)
**New axis: time.** The entire sweep is single-frame. But cosplayers shoot *bursts and sets*, and the
plastic tell compounds: retouch each frame independently and the skin **shimmers** — blotch appears in
frame 1, gone in frame 2, back in frame 3, because per-frame QA back-off (A5) makes independent
decisions. The set-level R14 idea learns a *profile* but still processes frames independently.

The move: **temporally-coherent skin decisions** across a burst/set — shared blotch/mole registry
(fix the same features the same way every frame), motion-aware texture donor (S6/R13b picks a *stable*
donor across frames, not a per-frame best that flickers), and coherent strength (A5 back-off decided
once per subject-per-set, not per frame). Turns a burst into a *consistent* edit. This is the axis the
skin-frontier doc explicitly flagged as un-opened; it's here because skin is where incoherence is most
visible.

**Why not a recombination:** R14 shares a *static profile*; this enforces *frame-to-frame consistency of
the actual pixel decisions*. Profile-sharing ≠ temporal coherence.

---

## P6 — Closed-loop self-calibration (perceptual-preference learning) ⭐ novel
**New axis: the tuning process itself.** Every strength/threshold in the R-series is *human-set*. M6
proposes a skin *score*; A5 backs *off* on a detector trip. But nothing **learns the mapping from image
features → preferred strength** from data. This is the loop M6 gestures at but never closes.

The move: collect a small set of (input, human-chosen-strength) pairs — the owner's own edits are the
training signal, free and perfectly on-brand — and fit a light model (not a deep net; a calibrated
regressor on F9's analyzer features + M1 ITA band + M3/M4 metrics) that predicts the *strength the owner
would pick*. Now F10 "smart default" is trained on the actual aesthetic, not heuristics. This is the
mechanism that makes "pro max" *personal* — it converges to the owner's taste instead of a generic mean.
Distinct from F9/F10, which are hand-authored heuristic maps.

**Why not a recombination:** F9/F10 are hand-tuned rules; this *learns* the rule from preference data and
closes the M6 measurement→action loop. Heuristic ≠ learned.

---

## P7 — Cross-region skin consistency (neck/ears/hands/chest as one tone system) ⭐ novel
**New axis: the whole visible skin, not just the face patch.** S1 does body skin and `body_match_face`
matches *tone*. But the sweep never modeled all visible skin as **one illuminated 3-D surface** with a
shared albedo/light solution. The tells it misses: face-retouched-but-hands-aren't (the classic
give-away), neck-in-shadow reading as a different *person's* skin, ear translucency (ears are lit from
behind — a strong life cue) getting flattened, décolletage sun-damage inconsistent with an evened face.

The move: solve **one** skin-tone + light model across *all* skin regions (face, neck, ears, hands,
chest via the person mask + a coarse body-part prior) and apply corrections that keep them *mutually
consistent* — the same albedo target, the same light direction (E-COMMON-1 again), preserved
ear-translucency, hands evened *to the face's* result. `body_match_face` matches a number; this matches a
*model*.

**Why not a recombination:** S1 matches face↔body *statistics*; this is a *joint* albedo+light solve over
all skin regions at once. Statistic-matching ≠ joint model.

---

## P8 — Aging-vector control (principled age up/down, not wrinkle-soften) ⭐ novel
**New axis: a semantic direction.** S5/R13d *soften wrinkles* — one symptom. But "look younger/older" is
a **coordinated multi-cue vector**: wrinkle depth, but also skin *translucency* (drops with age),
*chroma uniformity* (rises then falls), *volume/AO* (fat pad loss), *sebum/gloss*, and *edge definition*
(vermilion, jaw). Editing wrinkles alone gives the "smoothed but still old" result. Nothing in the sweep
models the *joint* aging axis.

The move: define an **aging vector** in the space of the metrics we already compute (M-series + the P1
physiological params + translucency from P1 + AO from I3), so a single "vitality/age" slider moves all
cues *together and proportionally* — the way age actually reads. This is the top-level creative control
the whole M+I+R machinery was building toward without naming it: not "reduce wrinkles 40%," but "read 5
years younger, coherently." Consumes almost everything else in the backlog as its components.

**Why not a recombination:** S5 is one cue; this is the *coordinated vector* across all cues, which only
becomes expressible once M/I/P1 exist. The whole > the parts.

---

## Ranking & recommendation

| Rank | Item | New axis | Why | Effort | Blocked by |
|---|---|---|---|---|---|
| 1 | **P4 skin↔makeup unmixing** | on-skin vs skin | Highest moat; cosplay-critical; nobody separates paint from person | ~3–4 wk | P1 prior (soft), A3 masks |
| 2 | **P1 layered SSS forward model** | depth/transport | The true ceiling of the physics line; unlocks P4/P7/P8 | ~4–6 wk (moonshot) | R7 (soft) |
| 3 | **P2 perceptual/CSF-aware budget** | the observer | Less smoothing for equal perceived evenness → less plastic; cheap | ~1–1.5 wk | R13/M4 |
| 4 | **P6 preference learning** | the tuning loop | Makes it converge to the *owner's* taste; closes M6 | ~2 wk | F9·M-series |
| 5 | **P3 output-conditioned render** | the destination | One tune, N correct deliveries — what "pro" means | ~1.5 wk | R13c |
| 6 | **P7 cross-region consistency** | all visible skin | Kills the face-done-hands-aren't tell | ~2 wk | S1·E-COMMON-1 |
| 7 | **P5 temporal/burst coherence** | time | Stops set-level shimmer; the flagged un-opened axis | ~2–3 wk | R14·A5 |
| 8 | **P8 aging-vector control** | semantic direction | The top-level slider everything else composes into | ~1.5 wk | M·I·P1 |

**Standout recommendation:** **P4 (skin↔makeup unmixing)** is the one to fund — it's the highest-moat,
most audience-aligned idea in the entire skin backlog (the cosplay thesis is *literally* about painted
faces, and no competitor separates the paint), and it motivates **P1** underneath it as the skin prior.
If P1's full forward model is too heavy to start, P4 can bootstrap on R7's linear unmixing as an interim
skin prior and upgrade to P1 later. Second pick is **P2** — cheap, immediate, and it attacks the
plastic-skin problem from the one angle (the observer's transfer function) that R9/R13 never used.

**Honest note — is *this* the real exhaustion?** These eight are genuine new axes (depth, observer,
destination, on-skin-layer, time, tuning-loop, all-skin-joint, semantic-vector), not recombinations —
that's the bar I held them to. Below this, I believe skin *is* actually mapped: physics (R7/R9/R12/**P1**),
signal (R13/**P2**), measurement (M/**P6**), audience (R11/**P4**), the additive frontier (I), and now
delivery (**P3**), scope (**P7**), time (**P5**), and semantics (**P8**). The only frontier I can see past
this is leaving classical processing entirely — a learned generative skin *prior* (diffusion/NeRF-style)
that hallucinates plausible skin, which the project has deliberately deferred (A4 parked, classical-only
until A1 evidence). That's a different philosophy, not a next item. Say the word and I'll scope it as an
explicit fork.

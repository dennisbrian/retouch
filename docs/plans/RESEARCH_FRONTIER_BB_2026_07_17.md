# RESEARCH — Frontier BB-series: verified code findings + next-source deep dive

**Date:** 2026-07-17 · **Author:** Fable research pass (read-only on production code)
**Owner ask:** continue the research workstream (K/X/Y/Z/AA lineage) toward
further algorithm enhancement, internet-sourced where internal docs are
exhausted.
**Method:** grep/read verification at HEAD (`5097560`), plus primary-source
fetches for the AA5/AA6 targets (citations in §Sources — exact numbers pulled
from the papers, not recalled).
**Rules inherited:** classical/deterministic, unit-testable, no learned models,
tone-fair, no absolute intensity thresholds, no identity-changing defaults.

**Delivery update:** BB-F1 shipped in `fd095ad` with dark-versus-bright and
tiny-mask regression coverage. The table retains the former gate as audit
evidence; current code derives its ramp only from pixels inside the sclera mask.

---

## 0. TL;DR

Two workstreams came out of this pass:

1. **Three verified tone-fairness gate findings** in shipped code that the
   `No Absolute Intensity Thresholds` rule in CLAUDE.md forbids (§1). One is
   worth fixing now; two are minor. These were *verified by reading*, not
   grep-pattern-matched — one candidate (teeth colour math) was checked and
   **cleared** because it is neutral-relative, which is the endorsed pattern.
2. **Primary-source numbers** now in hand for AA5 (redness / teeth / sclera
   perceptual bands) and AA6 (Kee–Farid), so those items graduate from "cited"
   to "implementable with real targets" (§2). Plus two genuinely new axes
   (BB1 limbal-ring restoration, BB2 C2PA provenance-preserving export) that
   fall out of what's already in the tree (§3).

---

## 1. Verified code findings (fairness gates) — fold into the wire-up backlog

These are absolute intensity/luminance gates on tone-scaling signals. Per the
CLAUDE.md fairness rule they systematically bias against darker or dimly-lit
subjects. Confidence is from reading the surrounding code, not the grep hit.

| # | Site | Gate | Severity | Note |
|---|---|---|---|---|
| BB-F1 | `eyes.py` (pre-fix) | `bright_sclera = (lab[:,:,0] > 100.0)` | **Fixed** | The former gate gave no correction to dim sclera. `fd095ad` replaces it with a soft ramp relative to the median inside the sclera mask, including tiny masks against a bright background. |
| BB-F2 | `teeth.py:121` | `(l_channel > 80)` "not dark void" floor | Minor | AND-ed *under* a correct relative primary (`l_channel > l_median * 0.9`). The relative term already excludes the void in most cases; the absolute floor is a redundant belt that can clip genuinely-detected-but-dim teeth. Replace with `l_median * k` or drop. |
| BB-F3 | `eyes.py:117` | `(l_chan - 220.0)` iris-specular gate | Minor | Specular catchlights are near-white regardless of tone, so real-world bias is small — but it is still an absolute constant on a per-face signal. Gate on margin over the iris's own bright percentile if touched. |

**Cleared on inspection (do NOT "fix"):** `teeth.py:80-85` colour math
(`lab[...,2] - 128`, `lab[...,1] - 128`) and `_detect_teeth`'s
`l_median * 0.9` / `s_median * 1.2` are **neutral-relative and median-relative**
— exactly the tone-adaptive pattern the rule endorses. `teeth.py` is mostly a
*good example* of the rule; only its one redundant `> 80` floor strays.

**Recommendation:** BB-F1 is shipped as a standalone tone-adaptive fix,
verified with deterministic regression tests. Fold BB-F2/BB-F3 into the next
fairness sweep; they are latent, low-blast-radius.

---

## 2. AA5 / AA6 — now with primary-source numbers (were "cited", now buildable)

### AA6 — Kee–Farid perceived-retouching score (exact statistic set confirmed)

The AA doc guessed "4 geometric + 4 photometric incl. SSIM." The paper (PNAS
2011, verified) is more specific, and the specifics *change the build*:

- **Geometric (4):** mean & σ of the warp-field magnitude **projected onto the
  local luminance gradient**, computed separately over **face** and over
  **body** (body regions weighted: bust/waist/thigh ×2, head/hair ×0.5, else
  ×1). Projection onto the gradient matters — motion *along* an edge is
  perceptually invisible, so raw displacement magnitude would over-count. The
  engine has the raw displacement field; it must dot it with the luma gradient
  before summarizing.
- **Photometric (4):** mean & σ of **SSIM over the face** (luminance channel
  only, **brightness term excluded**, β=γ=1, C₂=0.03², C₃=C₂/2), and mean & σ
  of a **9×9 linear-filter frequency-response parameter D** over the face
  (D>0 = net blur, D<0 = net sharpen).
- **Fit:** SVR with an RBF kernel on **390 observers × 468 image pairs**,
  leave-one-out R≈0.80. **Caveat for us:** an SVR is a learned model, which
  our NO-GO forbids at runtime. The AA doc's escape hatch holds — compute the
  8-vector deterministically (we own every input exactly), then fit a **fixed
  monotone mapping once offline** and ship stored coefficients. Do **not**
  ship the SVR; ship a monotone polynomial/isotonic fit on our own corpus and
  treat the 8-vector itself as the primary QA output.

**Net:** AA6 stays high-value and the statistics are pinned. Downgrade the
"1–5 number" to secondary; the **8-vector is the deliverable**, the scalar is a
convenience fit we control.

### AA5 redness bound (Re et al. 2011 — numbers confirmed)

The oxygenated-blood manipulation moved skin by **ΔE 2.4 between the two
oxygenation masks**, ±100% giving endpoints at **ΔE 4.8**. Measured 2AFC
thresholds:

- **Colour detection:** ΔE **0.67** (SEM 0.09) — observers notice a redness
  change here.
- **Health / attractiveness perception shift:** ΔE **1.44 / 1.38** (SEM ~0.16)
  — the change starts *reading* as healthier only past here.

**For `hb_shift`/`hb_even`:** express the edit cap as a **ΔE budget in the
oxygenated-blood direction**, not a raw strength. A sensible band: stay under
~ΔE **2.4** (the full physiological oxygenation swing) as a hard cap; a change
below ΔE 0.67 is imperceptible (no-op territory); the sweet spot for a
*visible-but-natural* health lift is ΔE **1.4–2.0**. This gives `hb_shift` a
principled maximum instead of a taste-scaled slider.

### AA5 teeth bound (WIO thresholds — confirmed, and they matter for BB-F2's sibling)

`teeth.py::whiten` today lightens L* by up to `+15` and pulls b* toward neutral
with **no perceptual ceiling** — that is the "chiclet teeth at high strength"
gap (a *quality* gap, distinct from the BB-F2 fairness floor). Confirmed
thresholds from the tooth-whiteness study (Journal of Dentistry 2017, 60
observers, 3 sites — note: **3 sites, not the "five-country" claim in the AA
doc; correct that citation**):

- **50:50 perceptibility:** ~**0.1–2.7** whiteness-index units (task-dependent).
- **50:50 acceptability:** ~**2.3–4.5** units.

A companion WID study gives cleaner single numbers (WPT ≈ **0.72**, WAT ≈
**2.62**). **Action:** compute a WIO/WID delta pre/post whiten and cap the edit
so ΔW stays inside the acceptability band (≈ ≤2.6 for a natural look, hard-stop
before ~4.5). Turns `strength` into a target-seeking whiteness delta with a
built-in ceiling — no chiclets. WIO formula needs one more fetch to pin exact
coefficients before implementing.

### AA5 sclera (unchanged direction, now paired with BB-F1)

The BB-F1 fairness fix and the AA5 sclera work touch the **same function**
(`_enhance_whites`). Sequence them together: fix the absolute gate (BB-F1) and
add the de-yellow-first-then-bounded-brighten target (AA5) in one pass, since
both rewrite the same L*/b* handling. The Provine/Russell direction (de-yellow
b* toward the face's own axis first, brighten second, both bounded, never
"super-white = healthy") is already captured in the AA doc.

---

## 3. New axes that fall out of the current tree

### BB1 — Limbal-ring restoration (rides the existing iris machinery)

**Finding:** `_sculpt_iris` already builds a `limbal_mask`
(`eyes.py:270`, an annulus at normalized iris radius ~0.9) and *darkens* it
(`- limbal_mask * 30 * strength`). So the geometry and mask for a limbal ring
already exist and ship. The literature (Peshek et al. 2011; multiple
replications) establishes the dark limbal ring as a validated youth/health cue
whose prominence **declines with age** — restoring a naturally-present-but-
faded ring is a legitimate, identity-preserving enhancement (unlike inventing
one on a dark iris where it was never visible).

**Method:** the darkening is already there; the gap is that it is **strength-
scaled with no upper bound and no visibility gate**. Make it target-seeking:
measure the existing ring contrast (limbal annulus L* vs iris-body L* within
the mask), and only restore *toward the subject's own likely-original*
contrast, capped so it never exceeds a natural maximum and **never fabricates a
ring where the iris is too dark for one to have existed** (the Peshek caveat:
limbal rings are undetectable in very dark irises — a hard confidence gate,
and a fairness point). Effort: small — it is a bound + gate on code that runs.

**Tests:** ring contrast moves monotonically with strength up to the cap then
saturates; dark-iris fixture (low iris-body L*) → gated to no-op; zero-strength
byte-identity.

### BB2 — C2PA / Content Credentials provenance-preserving export (workflow, not pixels)

**Why now:** C2PA is past spec and shipping in the exact tools this engine sits
beside — Photoshop, Capture One, DaVinci sign edit history; Pixel 10 and Leica
M11-P sign at capture; **EU AI Act Art. 50 transparency obligations take full
effect 2026-08-02** (weeks out). An automated retouch engine that **strips**
capture-time Content Credentials on export is now a real workflow liability for
pro users, independent of image quality.

**Two tiers:**
1. **Preserve (cheap, do first):** on export, if the source carries a C2PA
   manifest, don't silently drop it. At minimum warn; better, pass it through.
   Rides the K12 delivery-boundary refactor (single export choke point) — the
   right place to hook manifest handling.
2. **Assert (later, opt-in):** emit our own signed edit-history assertion
   (which ops ran, non-AI/deterministic provenance) via the MIT-licensed
   `c2pa-rs` / `c2patool`. This is a genuine differentiator: a retouch engine
   that produces *verifiable, non-generative* provenance in a market where
   "was this AI?" is the question regulators are now asking.

**Scope guard:** pixels unchanged; this is metadata/IO only. `c2patool` is an
external CLI (subprocess), not a Python runtime model — stays within the
no-learned-models rule trivially (it is a signing tool).

**Tests:** round-trip a fixture with a known manifest → manifest present and
valid post-export; source with no manifest → export unchanged; the assertion
path emits a manifest that `c2patool inspect` validates.

---

## 4. Recommended attack order (delta from the AA doc)

1. **K12 delivery boundary** (unchanged from AA) — and note it is now the hook
   point for **BB2 tier-1** (manifest passthrough) and AA8 (gain-map HDR).
2. **AA5 + AA6 with the confirmed numbers** — redness ΔE budget, teeth WIO cap
   (also closes BB-F2's sibling quality gap), sclera de-yellow (pairs with
   BB-F1), Kee–Farid 8-vector. All now have real targets, not placeholders.
3. **BB1 limbal cap+gate** — small, rides shipped iris code, real perceptual
   backing.
4. **AA2 BGU proxy upgrade** — still the architecture-level fidelity play.
5. Flagships per owner preference (X1 spectral / X3 dichromatic), AA7 after Z3.

## Bounds / NO-GO (inherited + new)

- **AA6:** the 1–5 scalar is a fixed offline monotone fit on our corpus — the
  paper's SVR is a **test oracle only**, never a runtime dependency. The
  8-vector is the primary product; it scores the *edit*, never the *person*.
- **AA5 redness/teeth/sclera:** all caps are bounded deltas from the input
  (ΔE / ΔW / b*-axis), QA bands and recipe ceilings — never silent auto-edits,
  never "white/red = healthy" claims on the subject.
- **BB1:** restore toward the subject's own likely-original ring only; hard
  confidence gate forbids fabricating a ring on a dark iris (fairness + the
  Peshek detectability caveat).
- **BB2:** metadata/IO only, no pixel change; `c2patool` is a signing CLI, not
  a model. Never assert non-AI provenance on an edit path that isn't actually
  deterministic.

## Sources

- Kee & Farid — *A perceptual metric for photo retouching*, PNAS 108(50), 2011
  (AA6; 8-statistic set, SVR-RBF, 390 observers / 468 pairs, R≈0.80 verified
  from the PMC full text).
- Re, Whitehead, Xiao, Perrett — *Oxygenated-Blood Colour Change Thresholds for
  Perceived Facial Redness, Health, and Attractiveness*, PLOS ONE 2011
  (AA5 redness; mask ΔE 2.4, detection ΔE 0.67, health/attractiveness ΔE
  1.44/1.38 verified from the article).
- Luo, Westland, Li et al. — *Investigation of the perceptual thresholds of
  tooth whiteness*, J. Dentistry 2017 (AA5 teeth; 50:50 PT 0.1–2.7, AT 2.3–4.5,
  60 observers / 3 sites — corrects the AA doc's "five-country" wording); WID
  companion thresholds WPT 0.72 / WAT 2.62.
- Peshek, Semmaknejad, Hoffman, Foley — *Preliminary Evidence that the Limbal
  Ring Influences Facial Attractiveness*, Evol. Psychology 2011; replications
  incl. Brown et al. (BB1; dark-iris detectability caveat).
- Newson, Faraj, Galerne, Delon — *Realistic Film Grain Rendering*, IPOL 2017
  (AA3 rider; Boolean-model coverage E=1−exp(−λπr²), Gaussian-approx mean/var
  — PDF saved locally for exact kernel factor when AA3 is implemented).
- C2PA Specification 2.4; `contentauthenticity/c2pa-rs` (MIT) + `c2patool`;
  ISO 21496-1:2025; EU AI Act Art. 50 (effective 2026-08-02) (BB2).

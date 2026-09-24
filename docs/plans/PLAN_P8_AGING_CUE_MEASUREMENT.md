# P8.M — Aging-cue measurement study (Implementation Plan)

Date: 2026-09-09. Baseline: `ee88d3d`.
Companion to [RESEARCH_P8_AGING_VECTOR_2026_09_09.md](RESEARCH_P8_AGING_VECTOR_2026_09_09.md),
which returned **NO-GO for an aging slider / GO for a bounded measurement study**.
This plan implements only that study's §6.

**What this plan is not.** It does not build an aging vector, does not fit
coefficients, does not couple cues, and adds no user-facing control, no
`ParamSpec`, no GUI element and no recipe key. Nothing here touches the render
path. If this plan is executed in full, the product behaves identically; only
new measurement functions and a report exist. That is deliberate — the research
found the vector uncalibratable for lack of any age supervision, and this study
exists to decide whether the cues are even *separable from lighting* before
anyone spends corpus money.

---

## 1. Objective and decision gate

**Objective:** establish, for the four aging cues that are measurable without
P1/I3, how much each moves across the existing corpus — and specifically how
much it moves with **lighting alone**, which upper-bounds any age signal that
could later be claimed.

**The gate (step 4).** For each cue, compare within-subject spread across
lighting strata against between-subject spread. If lighting-induced variance is
comparable to or larger than between-subject variance, that cue is not
recoverable from a single photograph, and P8 should be **closed**, not parked.
This gate can fail all four cues; that is a legitimate and useful outcome.

**Explicitly out of scope:** translucency and volume/AO. Both require P1/I3.
Neither can be inferred from `decompose_intrinsic`, whose scalar shading map is
documented not to recover the underlying physics (research §2, §6).

---

## 2. Slice M0 — Manifest `age` stratum (schema only, ~0.5 day)

**File:** `retouch/corpus_manifest.py`

Add to `DEFAULT_STRATA_VOCABULARY` (currently 10 dimensions, `:60-83`):

```python
"age_band": ("under_25", "25_34", "35_44", "45_54", "55_plus", "unknown"),
"age_source": ("self_reported", "annotator_estimated", "unknown"),
```

Constraints:
- **Do NOT add either to `DEFAULT_REQUIRED_STRATA`.** It stays
  `("skin_tone", "lighting", "face_scale", "pose")`. Making age required would
  invalidate every existing manifest at once.
- `age_source` is mandatory *whenever* `age_band` is set to anything but
  `unknown` — an annotator-estimated band is a much weaker label than a
  self-reported one and the distinction must survive into any later analysis.
  Enforce in the validator as a conditional issue, not a required stratum.
- `CORPUS_MANIFEST_SCHEMA_VERSION` is currently `3`. Adding *optional*
  vocabulary entries is backward compatible; **do not bump** unless the
  conditional-`age_source` rule is implemented as a hard validation error, in
  which case bump to `4` and update the three `schema_version` write sites
  (`:528`, `:836`, `:859`).

**Tests:** `tests/test_corpus_manifest.py` — a manifest with no age fields
still validates clean (regression); `age_band` set without `age_source` raises
the conditional issue; unknown age value is rejected by the existing
`unknown_*` vocabulary path.

**Honest note:** this slice creates a *place* to put labels. It does not create
labels, and populating it is the expensive, unscheduled part (research §4).

---

## 3. Slice M1 — Two cue estimators (~2–3 days)

**New file:** `retouch/aging_cues.py` (measurement-only leaf module; imports
from `specular`/`parsing` only — no engine import, no circular risk).

### M1.a Gloss scalar

Reduce the existing specular layer rather than re-deriving it. `extract_specular`
(`specular.py:48`) is already the repo's reference tone-invariant implementation
— margin above the face's own diffuse baseline, verified across a synthetic
Fitzpatrick I–VI sweep in `tests/test_specular_finish.py::TestExtractSpecularToneInvariance`.

```python
def gloss_state(img_bgr, skin_mask, *, rho: float = 0.95) -> GlossState
```

Returns a dataclass mirroring `SkinHomogeneityState`'s shape (value fields +
`pixel_count` + `confidence`):
- `gloss_coverage` — fraction of skin pixels with non-zero specular response.
- `gloss_intensity` — mean specular value over responding pixels only
  (separating coverage from intensity; a broad sheen and a small hot highlight
  are different cues and must not average into one number).
- `confidence` — degrade with small `pixel_count`, mirroring the existing
  metric's convention.

**Critical constraint:** the returned numbers must never be normalized against
a fixed intensity. All scaling is relative to the face's own baseline, which
`extract_specular` already handles — do not add a post-hoc absolute threshold
on its output.

### M1.b Wrinkle-energy scalar

The research found wrinkle *actuators* but no *estimator* (§2). The actuator
already contains the detector: `skin._wrinkle_soften_masked` uses multi-
orientation black-hat morphology to find ridges, then attenuates them. This
slice measures that same ridge signal and discards the attenuation.

```python
def wrinkle_energy(img_bgr, regions, *, face_width=None) -> Dict[str, WrinkleEnergy]
```

Per-zone, keyed by the zone names that already exist and are **verified
populated** in `parsing.py:909-911` from real landmark rings: `forehead`,
`nasolabial_l/r`, `crows_feet_l/r`, `neck`.

- Reuse `_build_dimensional_mask` for zone masks (same path the op uses).
- Ridge signal: black-hat at multiple orientations on the L channel, as in the
  op.
- **Normalize per zone by that zone's own local contrast** (e.g. ridge energy
  ÷ zone L standard deviation). This is the tone-invariance requirement and the
  single most important line in this slice: raw ridge amplitude scales with
  shadow contrast, which falls with increasing melanin. An un-normalized
  wrinkle energy would systematically under-report on Fitzpatrick V–VI.
- **Scale-invariance:** the op derives `wrinkle_length` from
  `face_height = h_img * 0.6` — an image-relative guess that is wrong for any
  crop that is not a tight portrait. The estimator must take `face_width`
  explicitly (as `apply_sss` does) and derive its structuring-element length
  from that, or measurements are not comparable across `face_scale` strata.
  Do not inherit the op's approximation.

### Tests (`tests/test_aging_cues.py`)

- **Monotonicity:** synthetic ridges of increasing amplitude → increasing
  `wrinkle_energy`; increasing synthetic highlight → increasing gloss.
- **Tone-invariance (the one that matters):** the same synthetic ridge/highlight
  pattern composited onto a Fitzpatrick I–VI albedo sweep must return
  approximately equal values. Follow the existing
  `TestExtractSpecularToneInvariance` pattern. **A failure here blocks the
  slice** — it means the estimator would encode skin tone as age.
- **Scale-invariance:** same face at 2× resolution → approximately equal energy.
- **Empty/degenerate:** `None` mask, <32 px zone, missing region attr → zero
  value with `confidence == 0.0`, never a crash (the `hasattr` guard in the op
  shows zones can legitimately be absent).

---

## 4. Slice M2 — Stratified sweep + report (~2 days)

**New file:** `scripts/qa/aging_cue_sweep.py`, following the established
pattern of `scripts/qa/post_epsilon_faceop_sweep.py`.

- Input: the **dev split only**, via `corpus_manifest`. Not `locked_test` —
  it has no age labels to certify against and burning it on an exploratory
  sweep is exactly what corpus governance v3 exists to prevent.
- Measure per face: `wrinkle_energy` (per zone), `gloss_state`,
  `skin_homogeneity_state.chroma_blotch_std`, and `facial_feature_contrast`
  as the edge-definition proxy.
- Emit JSON (raw per-face rows, so conclusions are re-derivable without a
  re-run — the eye-gate study's `sweep_full.json` precedent, which proved its
  worth when the doc's rounded recommendation turned out wrong) plus a markdown
  summary stratified by `lighting`, `skin_tone`, `face_scale`, `pose`.
- Run with `.venv/bin/python` and `RETOUCH_MEDIAPIPE_BACKEND=legacy`
  (bare `python3` has no working detection — every face op would silently
  no-op and the sweep would report zeros as data).

**Report:** `docs/plans/RESEARCH_P8_CUE_VARIANCE_<date>.md` — must state the
step-4 verdict per cue explicitly, including "not separable from lighting"
where that is what the numbers say.

---

## 5. Risks

| Risk | Mitigation |
|---|---|
| Estimator encodes skin tone as age | Tone-invariance test is a blocking gate, not a nice-to-have. Normalize by each zone's own contrast. |
| Makeup confounds every cue | P4 unmix is wired (`engine.py:1160`), but this sweep measures **unretouched input**. Report `makeup` tag alongside; do not attempt correction in this study. |
| Corpus too small to conclude | ~5–15 real people, zero subjects assigned. The sweep may be underpowered for the within-subject arm — say so in the report rather than reporting a spread as a finding. |
| Sweep zeros read as real data | Runtime-doctor check + assert non-zero detection count before writing results (the MediaPipe env-drift failure mode). |
| Study drifts into building the vector | This plan's §1 scope statement is the contract. Fitting coefficients requires new authorization and, first, age labels. |

---

## 6. Sequencing

M0 → M1 → M2, strictly. M0 is independent and safe to land alone. M1 is the
real work and is where the blocking tone-invariance gate sits. M2 is worthless
without M1's tests green — an unvalidated estimator produces confident numbers
about nothing.

Total: ~1 week. At the end there is a report and a decision, not a feature.

# Tier A Plan — Skin Retouch Competitive Advantage (Stages A1–A5)

**Date:** 2026-07-02
**Type:** Competitive research + advantage plan (planning only, no code)
**Question:** among the best available skin-retouch tools, where can this system genuinely win — and what do we build to make that advantage real?
**Builds on:** `PLAN_SKIN_PRO.md` (S1–S6 capability stages). This doc decides *positioning*; S1–S6 remain the capability backbone.

---

## Part 1 — The competitive field (2026)

| Tool | Model | Skin strength | Structural weakness |
|---|---|---|---|
| **Evoto AI** | Desktop app, cloud processing, **credit-based** | Best-in-class texture-preserving skin, volume batch (weddings/schools) | Credits get expensive at volume (~$500 ≈ 4 weddings); cloud = privacy + offline problem; not scriptable; generalist portrait focus |
| **Retouch4me** | Per-plugin suite (~$150 ea.) inside Photoshop/host | Single-task neural nets are excellent: Dodge&Burn, Heal, Skin Tone, Eye Vessels | No unified pipeline; needs Photoshop; per-plugin cost stacks; no color grading; no recipes |
| **Aperty (Skylum)** | Desktop subscription | Newer all-in-one portrait editor, solid sliders | Generalist; young; convenience-tier depth |
| **PortraitPro / Portraiture** | Perpetual license plugins | Fast classic smoothing | Smoothing-era tech; plastic at high settings |
| **Photoshop manual** | The pro baseline | Unlimited ceiling (manual freq-sep + D&B) | Hours per image; skill-gated; neural skin filter is weak |
| **Meitu/Wink/XHS-class apps** | Mobile, free/cheap | Instant, stylized, huge in CN cosplay community | Phone-res output; no RAW/16-bit; no batch; style presets not editable |

Consensus in current comparisons: **Evoto wins on overall quality, Retouch4me on cost-practicality**; pros commonly run Evoto for the bulk and Retouch4me for hero shots. That's the bar.

## Part 2 — What the best do that we must match (table stakes)

1. **Evoto-class texture-preserving smoothing** — never blur pores. Our path: guided-filter core (P1) + `texture_opacity`/`micro_restore` + S6 texture transplant. Already close architecturally; needs tuning against their output (A2).
2. **Retouch4me-class micro dodge & burn and skin-tone evening** — planned as S2/S3; the classical band-pass method reproduces most of the neural result because the *problem* is linear (luminance/chroma unevenness at a known frequency band).
3. **Volume batch reliability** — we already have folder→ZIP batch + per-face detection; F2 sessions + F9 adaptation make it per-image smart, which Evoto charges credits for.

## Part 3 — The openings (what NONE of them do)

1. **🎯 Body skin parity.** Evoto/R4me are face-first; body skin evening (legs/arms/décolletage matched to the retouched face) is weak-to-absent everywhere. S1 (`body_match_face`) is a real first — and cosplay is the genre where it matters most.
2. **🎯 Cosplay/anime awareness.** No tool on the market knows what a wig lace line is, protects body paint, preserves drawn-on makeup edges, or produces anime-cel porcelain (our flatten/quantize/unify primitives are already unique). The CN mobile apps stylize but can't deliver print-res pro output. **This niche is unowned at pro quality.**
3. **🎯 Recipes-as-code + local + no credits.** Photographers resent Evoto's credit burn and R4me's plugin stacking. One-time local tool, shareable JSON recipes, scriptable CLI batch — a structural business advantage, not just a feature.
4. **🎯 Self-checking output.** No competitor runs artifact QA on its own results. F11 detectors + auto-back-off = a marketable "no plastic skin" guarantee (A5).
5. **Privacy/offline** — cosplay shoots often involve unpublished costumes/embargoed characters; local processing is a real selling point vs cloud.

**Advantage thesis:** *the cosplay & anime-portrait skin specialist — Evoto-class face skin, first-in-market body skin, anime primitives nobody has, local and scriptable with no credits, and the only tool that checks its own output.*

---

## Part 4 — The plan (Stages A1–A5)

### Stage A1 — Competitive benchmark harness (~4 days, do FIRST)
Evidence before engineering:
1. Build a 30-image benchmark corpus: 10 studio portraits, 10 cosplay (incl. the 爻一爻-class moody sets, legs/arms visible), 10 hard cases (dark skin tones, freckles — must survive!, high-ISO, profile faces).
2. Process the corpus through trials of Evoto + Retouch4me (D&B, Heal, Skin Tone) + our current `natural`/portrait recipes; store all outputs versioned in `test_output/competitive/`.
3. Metrics scored per output (extends `benchmark.py`): blotch metric (band-passed L std — from S2 plan), texture retention (high-band energy ratio), tone uniformity (skin chroma variance), plus a blind A/B sheet (contact-sheet generator already exists in GUI batch tab).
4. Output: `COMPETITIVE_BASELINE.md` — where we objectively stand per metric, per image class. Re-run after every S/A stage; the deltas ARE the progress report.

### Stage A2 — Match-the-best face quality (~1 week, rides S2/S3)
Tune, don't invent: run S2 (micro-D&B) and S3 (color evening) parameter sweeps against the R4me/Evoto outputs from A1; acceptance = **blind A/B indistinguishable-or-preferred on ≥7/10 studio portraits**. Freckle/mole preservation is the differentiating sub-case (neural tools sometimes eat them): protected-feature mask (dark compact spots with stable edges → exclude from S2/S3/blemish) — a visible quality win pros check first.

### Stage A3 — Cosplay-exclusive skin moat (~2 weeks, after S1 + T2 masks exist)
The features that make switching pointless for our audience:
1. **Makeup-aware smoothing:** drawn eyeliner/eyeshadow/lip-line edges are high-chroma edges *inside* the skin mask — detect and protect them during smoothing/D&B (chroma-edge mask subtracted from processing masks). Anime makeup survives at smooth 80.
2. **Wig-lace / hairline blend:** the lace-front transition band (hair_mask boundary ∩ forehead) gets dedicated blend smoothing — the #1 cosplay-specific flaw in every competitor.
3. **Body-paint & tattoo protection** (from S1's high-chroma exclusion) promoted to a first-class toggle with its own mask preview.
4. **Stocking/fishnet handling:** periodic-texture detection over leg regions → skip skin ops (moiré/pattern destruction guard) or "through-stocking" tone-even mode.
5. **Shoot-consistency lock:** per-shoot skin reference (mean face LAB of a chosen hero frame, stored via `style_library.py`) — every image in the batch pulls face+body skin toward the same target. Series consistency is something even Evoto doesn't offer explicitly.

### Stage A4 — Neural boosters where classical plateaus (~2–3 weeks, optional, after A1 evidence)
Only if A1/A2 show classical gaps: (a) stray-hair-on-skin segmentation (R4me Heal's edge), (b) skin-defect segmentation to upgrade the blemish detector's anomaly heuristic. Small U-Net-class ONNX models, local, download-on-first-use (P4 model-fetch), classical fallback always. Honest note: sourcing/training these is real effort (datasets + training runs) — hence evidence-gated, last, and skippable if A2 closes the gap.

### Stage A5 — "No plastic skin" guarantee (~3 days, after F11 + F9)
F11 detectors (plastic-skin, halo, banding) + auto-back-off: if a skin QA check fails, reduce the responsible strengths (S2/S3/smooth) stepwise and re-render the face crop until green, annotate the result ("auto-adjusted to preserve texture"). Batch mode gains a QA column. No competitor makes this claim; we can, with receipts.

---

## Sequencing

```
A1 (benchmark, 4d) ──► S2/S3 + A2 (match the best) ──► S1 (body skin) ──► A3 (cosplay moat) ──► A5 (guarantee) ──► [A4 if evidence demands]
```
Master-sequence placement: A1 can run **immediately** (needs nothing). A2 rides S2/S3 (Tier S). A3 needs S1 + ideally T2's makeup region work. A5 needs F11+F9. Fold into the authoritative sequence in `PLAN_TIERP_PERF_ARCH_SHIP.md` when you commit — suggested: A1 now; A2 with Tier S; A3 after S1; A5 after F11/F9; A4 floats.

## Verification
1. A1 corpus + metrics committed; `COMPETITIVE_BASELINE.md` regenerated per stage — objective trend line vs the best.
2. A2 acceptance: blind A/B ≥7/10 vs R4me D&B on studio portraits; freckles 100% preserved on the freckle test images.
3. A3 acceptance: cosplay corpus — anime makeup intact at smooth 80, lace line invisible at 100%, stockinged legs artifact-free, 20-shot batch tone-consistent within ΔE 3 face-to-face.
4. A5 acceptance: force plastic-skin failure (smooth 100 + texture 0) → auto-back-off produces QA-green output.

## Open items
- Requires purchases for A1: Evoto trial credits + R4me trial plugins (small, one-time research cost).
- `anime_crystal_void` dead keys — unchanged (fix vehicle: T1).
- ~~Master sequence update pending placement decision.~~ **Resolved 2026-07-02 in `MASTER_PLAN.md`: A1 = Phase 0 (immediate, parallel to P1); A2 = Phase 2 with S2/S3; A5 = Phase 5 after F9; A3 = Phase 6 after S1+T2; A4 = Phase 6, optional and evidence-gated.**

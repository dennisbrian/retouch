# Autonomous Image Quality & Regression Lab

Unit tests verify software correctness. This lab answers the other question:
**"tests are green… but did the photo actually get better?"**

It renders a tagged benchmark corpus through the engine, measures
region-aware deltas against a frozen baseline, and triages every
(case, recipe) as **PASS / REVIEW / REGRESSION** so the owner inspects only
the images that need a human eye.

## Quick start

```bash
scripts/dev/quality-lab corpus --build-synthetic   # seed corpus (one-time)
scripts/dev/quality-lab baseline                   # freeze stable reference
# ... make a change ...
scripts/dev/quality-lab run --pr-aware origin/main HEAD -o pr.json
scripts/dev/quality-lab compare test_output/quality_lab/reports/pr.json
scripts/dev/quality-lab benchmark test_output/quality_lab/reports/pr.json
```

Every command accepts `--json` for machine-readable output. `compare` and
`benchmark` exit 1 on REGRESSION (CI-gateable).

## Concepts

### Corpus
`quality_lab/corpus_manifest.json` (tracked) registers cases:
`{id, tags, input}`. Images live under `test_output/quality_lab/<id>/input.png`
(gitignored — **never commit private photos**). The seed corpus is synthetic:
procedural portraits tagged `skin`, `dark-skin`, `face_paint`, `wig`,
`glasses`, `eyes`, `blemish`, `makeup`, plus one frozen-landmark face from
`tests/golden_face_fixture.py`.

Synthetic cases have no detectable face; region masks come from the frozen
landmark fixture (real anatomical topology, same `FaceParser` code path as
production). Cases are annotated `synthetic_fixture_face` in reports.

To add real photos: drop `input.png` into `test_output/quality_lab/<id>/`
and append a case entry to `quality_lab/corpus_manifest.json`.

### Baseline
`baseline` renders every (case, recipe), records metrics + output hashes +
provenance (commit, cv2/numpy/mediapipe versions, recipes) into
`test_output/quality_lab/reports/baseline.json`. Re-run it only when main
advances and you intend the new outputs to become the reference.

### Metrics
- **Identity** (candidate vs baseline): SSIM, ΔE76 — *regression evidence,
  not quality*. A perfect do-nothing edit scores 1.0; that's why absolute
  checks exist too.
- **Absolute** (output vs input): luminance drift, skin chroma drift,
  highlight/shadow clip-fraction ratios, texture retention (Laplacian
  energy ratio), edge retention (Canny density ratio).
- **Region-aware**: all absolute checks run per region
  (FACE/SKIN/EYES/EYEBROWS/LIPS/HAIR/NECK/BACKGROUND) using real
  `FaceRegions` masks. A skin improvement that destroys eyelashes shows up
  as EYES texture/edge loss, not as a hidden global average.
- Regions covering < 0.2% of the frame are reported but not triaged
  (means over a few hundred px are noise).

### Triage
Per (case, recipe): REGRESSION if any check breaches its `regress` limit
(or the render errors), REVIEW at the `warn` limit, else PASS. The
comparison report lists `human_attention` — the cases to actually look at.

### PR-aware selection
`run --pr-aware BASE HEAD` maps the diff to corpus tags via
`IMPACT_TAG_MAP` in `scripts/dev/quality_lab/core.py` and scales recipes by
`pr_governor` risk (LOW→natural, MEDIUM→+cosplay_clear_v1, HIGH→+portrait).
Docs-only PRs select zero cases and exit 0 without rendering.

### Performance
`benchmark` compares per-case time and peak RAM against baseline. Renders
under 2 s are skipped (ratio noise dominates at that scale).

## Threshold governance

`quality_lab/thresholds.json` is a **tracked, review-gated engineering
artifact**. The lab never writes it; agents must never loosen it to make a
candidate pass. CI guard:

```bash
scripts/dev/quality-lab check-thresholds origin/main   # exit 1 if loosened
```

A loosening (higher `warn`/`regress` for `direction=high`, lower for
`direction=low`) must arrive as its own reviewed PR with justification.

## Layout

```
quality_lab/corpus_manifest.json   tracked — case registry
quality_lab/thresholds.json        tracked — triage policy
scripts/dev/quality_lab/           implementation (stdlib CLI + lazy heavy deps)
scripts/dev/quality-lab            bash wrapper (uses .venv when present)
test_output/quality_lab/           gitignored — images, renders, reports
```

## Known limits

- Synthetic corpus catches catastrophic regressions and pipeline breakage;
  it does NOT replace the real-photo corpus for aesthetic judgment.
- No LPIPS/learned perceptual metric (dependency cost not justified yet —
  cv2/numpy only).
- Identity checks require baseline re-freeze after intentional output
  changes, or every subsequent PR flags as changed.

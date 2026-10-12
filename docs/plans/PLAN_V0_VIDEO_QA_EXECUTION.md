# Plan — V0 Execution: Corpus, Baselines, Thresholds

**Status:** 🔄 IN PROGRESS — companion to `PLAN_VIDEO_FACE_RETOUCH.md` (V0)
and `RESEARCH_VIDEO_RETOUCH_ALGORITHMS.md`. Authorizes review-only scripts
under `scripts/review/` and documentation; no image-pipeline or video-pipeline
code.

## What exists (2026-07-18)

- `scripts/review/video_qa_report.py` — compares source/result clips in
  track-normalized 128×128 face crops. Reports `track_coverage`,
  `mean_effect_energy`, `mean_effect_flicker`, `p95_effect_flicker`,
  `mean_source_motion`, and `effect_flicker_to_motion` (the headline signal:
  how much the retouch delta changes per frame relative to how much the face
  itself moved). Requires a single-face track JSON
  (`{"frames": [{"frame", "face": [x,y,w,h], "confidence"}]}`).
- `tests/test_video_qa_report.py` — 4 passing tests.
- `docs/review/VIDEO_CORPUS_MANIFEST_TEMPLATE.md` — per-clip consent,
  retention, coverage-tag, and review record; facial video never enters the
  repo.

## Gaps blocking the first real measurement

1. ✅ **Track producer** — `scripts/review/extract_face_track.py` (2026-07-18):
   reuses `retouch.detection.FaceDetector` per frame on a 1280-wide proxy,
   scales the largest face box back to source resolution, emits the harness
   contract. Tests in `tests/test_extract_face_track.py`.
2. **Corpus clips.** First clip in hand (local `DSCF4322.MOV`, 4K 29.97 fps,
   210 frames ≈ 7 s, frontal steady portrait; 100% track coverage). Needs a
   completed manifest record; further clips must cover the remaining tags.
3. ✅ **Baseline render tooling** — `scripts/review/render_video_naive.py`
   (2026-07-18): deliberately naive per-frame engine render, review-only.

### First calibration result (DSCF4322, null render)

The plan predicted `effect_flicker_to_motion ≈ 0` on a null render. Measured:
**0.138** (mean effect energy 2.60, mean effect flicker 1.26 against source
motion 9.13) — the lossy `mp4v` re-encode alone injects that much apparent
"effect". Consequences: (a) every threshold must be read as margin above the
per-clip codec floor, so archive a null run beside every baseline run; and
(b) evaluate a higher-quality/lossless intermediate codec for renders before
V1 so codec noise stops eating the measurement budget.

## Step order

1. **Corpus assembly** (user-driven): 8–12 clips, 5–15 s, ≥720p, each with a
   completed manifest. Across the corpus, every coverage tag in the template
   must be ticked at least once, with ≥3 distinct skin-tone groups including
   Fitzpatrick V–VI (the repo tone-invariance rule applies to QA corpora, not
   just code), and audio present on at least two clips for later A/V checks.
2. **Track extraction script** (`extract_face_track.py`, review-only).
3. **Calibration runs** before trusting any number:
   - *Null render* — result = source re-encoded with no retouch. Establishes
     the codec-noise floor for every metric; `effect_flicker_to_motion`
     should be ≈ 0. Archive as `test_output/video_qa/null_<clip>/`.
   - *Track dropout sanity* — verify `track_coverage` responds correctly when
     confidence dips below `--min-confidence`.
4. **Problem baseline:** naive per-frame render of each corpus clip through
   the image engine, harness report + human review (the report's human-gates
   checklist) archived per clip. This is the number the whole video track
   exists to beat — record it before designing fixes.
5. **Threshold setting:** only after ≥8 human-reviewed runs, derive
   acceptance thresholds from the reviewed pass/fail split (not from
   intuition) and record them with provenance in
   `docs/review/VIDEO_QA_THRESHOLDS.md`. This plan deliberately contains no
   threshold numbers.
6. **Research bake-offs** (defined in `RESEARCH_VIDEO_RETOUCH_ALGORITHMS.md`
   §5, run only after 1–5): (a) mask cadence — per-frame parsing vs
   every-N + track-warp vs landmark+colour prior; (b) smoothing formulation —
   current frequency separation at proxy vs guided-filter
   downsample/upsample with high-pass reinjection.

## Metric coverage vs the master plan

The master plan names five objective measurements. Current status:

| Planned metric | Covered today? | Planned extension |
|---|---|---|
| Track loss | ✅ `track_coverage` | none needed for V0 |
| Effect displacement | ⚠️ folded into `effect_flicker` | needs landmark (not box) tracks; version the track JSON contract before extending it |
| Frame-to-frame skin-colour delta | ❌ | mean OKLCh delta over a skin region — requires a mask input |
| Texture-energy delta | ❌ | high-frequency band energy ratio per crop, source vs result |
| Mask-edge error | ❌ | requires mask input + human-marked edge frames |

Known conflation to keep in mind when reading reports: `effect_flicker` sums
tone, texture, and geometry instability into one L1 number. That is fine for
ranking runs and detecting regressions; it is not diagnostic. The extensions
above split it — add them only when a baseline exists to validate them
against.

## Authorization boundary

Same as the master plan: review-only scripts and docs. No `retouch/video/`
modules, no engine changes, no committed facial video. Two failed attempts at
the same QA metric or script → stop and escalate, per repo convention.

## V0 exit (restated concretely)

- Corpus of manifest-registered clips covering the full tag matrix.
- Null-render calibration + naive-render baseline reports archived per clip.
- `VIDEO_QA_THRESHOLDS.md` with human-review provenance.
- No image-pipeline files changed.

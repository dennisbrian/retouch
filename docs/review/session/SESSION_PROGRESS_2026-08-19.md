# Session Progress Report — 2026-08-19

**Date:** 2026-08-19
**Session:** Progress check-in — reviewed 9 commits landed 2026-08-16 → 2026-08-18
**Branch:** `feat/color-science-k9-fix-and-frontier`
**Tests:** 4,516 passed, 11 skipped, 0 failed (per `58d5fbb` commit message; full suite, not re-run this session)

---

## TL;DR

Three days of work since the last session report (`96bbcfa`, 2026-08-15): a foundational OKlab color-science bug fix, two detection-recall fixes, a 6-finding security hardening pass, three more tone-invariance conversions, and a GUI refactor that cut `gui.py` from 6522 to 5318 LOC. This is a major-upgrade arc — one of the fixes (OKlab matrix) had been silently skewing every skin-hue calculation since `59f9e31`.

---

## Commits this arc (`96bbcfa` → `58d5fbb`, oldest first)

| Commit | Area | What shipped |
|---|---|---|
| `66622be` | gui | Folder-picker buttons (batch tab), Recipe Cookbook wired to batch dropdown, gradio_client schema patch (bare-bool `additionalProperties` crash), `.venv/` gitignored |
| `e8d93b6` | golden | Re-blessed `outdoor_harsh_sun_v1` snapshot for the `96bbcfa` `acc_hair_only` render change; full suite green |
| `46bfd2c` | perf | `build_face_anchored_body_mask` label loop: O(N·L) → O(N) via `np.bincount`/`np.isin`; 5.7x faster (8.7s → 1.5s, 24MP/1603 components); byte-identical output |
| `a7208ce` | watch-folder | State-file paths no longer trusted blind: `record.path` must resolve under `input_dir`, `job.output_path` under a constructor-captured `output_root`. Violations quarantine (`.corrupt-*`) instead of silent skip |
| `aaaa1e5` | engine/tests | Deleted ~190 LOC unreachable non-registry branch in `_run_global_phases`; moved `_run_qa` body to `qa_detectors.run_qa()`; batch LUT cache invalidation at batch start; session-scoped 2048px test fixture (355s → 20s fast lane) |
| `d63994b` | tests/gui | Regression tests locking in `96bbcfa`'s render changes (safe-auto bands, ICC ingest, hair-mask plumbing, draft-path upscale); gui split step 1 (`gui_advanced.py`, ~580 LOC); CI macOS matrix added |
| `f3afabf` | gui/watch-folder/ci | gui split steps 2-3 (`gui_shoot.py`, `gui_batch.py`; `gui.py` 6015 → 5318 LOC); `flock`-based interprocess locking for watch-folder state saves (concurrent-writer test reproduces the pre-fix lost update); bench pip cache key fix |
| `cc61e67` | color/detection/skin | **OKlab matrix fix** (see below); RetinaFace `min_confidence` + BGR-feed fixes; small-crop landmark upscale retry; BiSeNet NaN/shape guards; yaw dampening for profile reshape; 3 more skin ops converted to tone-relative gates; pore-FFT + color-drift QA perf |
| `58d5fbb` | security/detection | 6-finding security hardening pass; full-res landmark pass now always runs (≤4096px) instead of only on zero-face fallback; engine entry-guard for malformed input; K3 gamut-compression perf investigated, not changed (documented why) |

---

## The headline fix: OKlab matrix (`cc61e67`)

`_OKLAB_M1` row 3 has been corrupted since `59f9e31`. Effect: white mapped to `a=0.022/b=-0.040` instead of achromatic; skin hue was off by ~60° vs true Oklab; chroma inflated ~1.6x. `SKIN_LOCI` hue targets were authored against *true* Oklab, so they become correct again with the matrix fixed (verified: real-skin hue lands 38-49°, fair subject `L=0.69/C=0.05`).

Regression anchors added (red `h=29.2`, blue a/b, white achromatic, skin swatches). White/gray tolerances tightened from `0.03/0.05` to `1e-3` — the old loose bounds are exactly what let this ship undetected.

**Why this matters more than a typical bug fix:** every skin-tone-relative color decision in the pipeline (hue targeting, chroma limits, specular/undereye/hue-line gates) sits downstream of this matrix. A ~60° hue error at that layer is large enough to plausibly explain subtler color-grading complaints that would otherwise look unrelated.

---

## Detection recall — two real misses fixed

1. `min_confidence` was never passed to `RetinaFace.detect_faces` — pip's internal default (0.9) silently dropped every face scoring 0.4–0.89. The engine-level threshold existed but was dead code (never reached the detector).
2. RetinaFace was fed RGB when the installed package (0.0.18) expects BGR internally and reverses channels itself — verified against the installed preprocess source, not assumed from docs.
3. Separately (`58d5fbb`): the full-resolution landmark pass previously only ran when the 1024px pass found *zero* faces. Small faces next to one big face were routinely missed because the 1024 pass "succeeded" (found the big face) and skipped the full-res retry. Now the full-res pass always runs for images ≤4096px and merges via bucket-based dedup.

Combined, these were likely a meaningful chunk of the "~4% undetected faces" limitation noted in CLAUDE.md — worth re-measuring against that figure once there's a representative eval set.

---

## Security hardening (`58d5fbb`) — 6 findings, all local-first / untrusted-file preconditions

| Finding | Fix |
|---|---|
| Loaded session JSON could steer file reads via `img_paths` | `gui.py` strips any `*path*` key on load |
| Hand-edited watch-folder state could pair a spoofed `output_root` with an arbitrary write target | Containment now pinned to the runtime `output_root` captured at `queue_jobs`, not self-attested state |
| `.cube` LUT with `LUT_3D_SIZE=512` → ~3.2 GB allocation before first render | Capped at 256, matching `.3dl` |
| Zip export used a predictable stamped tmpfile name (symlink race) | `tempfile.mkstemp` (O_EXCL) |
| Job-id path traversal via dataframe row ids | `_path_for` rejects ids outside `[A-Za-z0-9_.-]` |
| `torch.load` on NAFNet export path (crafted `.pth` = code exec) | `weights_only=True` |

---

## Tone-invariance sweep — 3 more ops closed out

Per CLAUDE.md's no-absolute-intensity-threshold rule: `unify_hue_line`, `smooth_undereye_shadow`, and `apply_specular_bloom` moved from fixed absolute L thresholds (e.g. `L>220`, `L>40/L<200`) to median-baseline relative margins. Continues the July tone-invariance work noted in the 08-15 report (specular, freckle, skin-whiten, lip-gloss, sclera) — undereye shadow and specular bloom were the ones still sitting on absolutes that skipped/misfired on darker skin tones.

---

## GUI refactor

`gui.py`: 6522 → 5318 LOC across two commits, split into `gui_advanced.py` (~580 LOC), `gui_shoot.py` (523 LOC), `gui_batch.py` (302 LOC). Same discipline both times: explicit re-exports, late-bound `gui.X` seams so existing test monkeypatches keep hitting the right module, registries (`_process_input_components`, `_recipe_output_components`) stay textually in `gui.py` under the existing AST-drift guard.

---

## Perf

- `build_face_anchored_body_mask`: 5.7x faster (8.7s → 1.5s, 24MP), byte-identical output (65/65 randomized + structural equality checks).
- Pore FFT: switched to `rfft2` float, ~4x faster, band fraction matches to 1e-3.
- Color-drift QA: deterministic 50k subsample before CIEDE2000 (24MP delta_e: 2.5s → a fraction of that).
- K3 gamut-compression: investigated, **not changed** — BLAS-bound bisection, subset gating can't be made byte-exact (gamut-corner ridges any affordable grid misses), 7.7% of real-photo pixels genuinely hit the knee. Documented as a future semantic-gating opportunity that would change golden hashes.

---

## Verification posture this session

This was a review/reporting session, not an implementation session — no code changes. Findings above are drawn from commit messages (which this repo's convention treats as the detail-of-record) and `git log --stat`/`-p` on the 9-commit range, not re-derived from scratch. Test/golden-hash claims are quoted from the commits, not independently re-run today.

---

## Afternoon session — detection recall & precision study (same day, after `60166c6`)

Executive summary of the follow-up research tranche (full detail in
`docs/plans/RESEARCH_DETECTION_RECALL_2026_08_19.md`; scripts in
`scripts/qa/detection_recall_*.py`, artifacts in
`test_output/detection_recall_study/`):

- **Subject recall re-measured on the 83-image DSCF corpus: 98.8%** (82/83,
  engine path = legacy FaceMesh @ 2048 proxy). The "~4%" CLAUDE.md figure was
  stale for subjects. Dual-scale detect (2048 ∪ 1024) reaches 100% on this
  corpus.
- **The one miss (DSCF4598) is FP-suppression**: an anime-poster
  false-positive counts as the face, so the zero-face-gated tiled fallback
  never fires; the real subject gets no face work (mean |Δ| 1.60 vs 4.31 for
  detected subjects in the same shoot, measured with real `RetouchEngine`
  runs at `cosplay` strength).
- **Precision is the real convention-corpus problem**: 18/100 S1 detections
  are MediaPipe fires on anime posters (Laplacian-var forensics: real faces
  p10=499 vs poster FPs p90=256 — zero overlap on this corpus; 10-frame
  geometry lock confirms same-poster re-detection). Engine-verified harm at
  `cosplay`: poster scenery receives visible face work (p99 |Δ|=29, max=91).
- **RetinaFace is not in `requirements/base.txt`** — the `cc61e67` F1/F2
  fixes are inert in the pinned `.venv` (opportunistic import falls back to
  MediaPipe-only silently). Production detection IS MediaPipe-only; the
  Tasks-path architecture never executes on the supported runtime.
- CLAUDE.md "Known Limitations" rewritten with the measured figures.
- No engine code changed at study time; `tests/test_detection.py` 80
  passed; all 9 study scripts compile.

### Follow-up #1 implemented same day: dual-scale person-gated detect augment

The research doc's top follow-up was implemented after two pre-checks
reshaped the design: a naive 2048∪1024 union imports 5 new poster FPs
(including DSCF4454's "recovered face", a poster all along — tex 65.5), so
the landed version gates 1024-pass additions on the engine's own
person-segmenter mask (measured poles: posters 0.000 vs subjects 1.000
coverage). `retouch/detection.py::_dual_scale_augment_legacy` + 8 unit
tests (`tests/test_detection_dual_scale.py`). Verified on the full corpus:
83/83 subject recall (was 82/83), exactly 1 new detection (the DSCF4598
subject, RF-confirmed), zero new FPs, median detect 22 ms. DSCF4598 subject
now receives face work (1.60 → 3.36 mean |Δ| at cosplay strength). Full
suite: 4,532 passed, 11 skipped, 0 failed. The 18 pre-existing poster FPs
are deliberately untouched (follow-up #3, needs non-convention validation
first).

---

## Evening session — poster-FP veto validation + RetinaFace dead-end proof

Follow-ups #3/#4 from the detection study, closed with measurements
(`docs/plans/RESEARCH_POSTERFP_VETO_2026_08_19.md`):

- **Safe partial veto validated (not implemented):** joint rule
  (person-coverage<0.5) OR (coverage<0.9 AND texture<80) kills 14/18 poster
  FPs with 0/82 real-face collateral, stable across threshold plateau
  T=50–350. Texture-only veto proven UNSAFE (real-face Lap-var floor is
  1.0 under blur σ=4, 9.1 under makeup smoothing — below poster baselines).
  Chromophore skinlike-fraction rejected (overlaps real faces both
  directions; 3 real faces sit at ≤0.05). 4 on-person poster FPs are
  unvetoable by any measured feature. Awaiting owner sign-off + group-shot
  re-validation (single-subject corpus showed real-face coverage ≡ 1.000;
  occluded group-shot faces will sit lower).
- **RetinaFace structurally dead in pinned runtime:** `pip --dry-run
  retina-face==0.0.18` resolves to protobuf 6.33.6 + TF 2.20 + keras 3.10
  + numpy 2 + cv2 5 — violating every core pin (pb<4, mp==0.10.5, numpy<2,
  cv2<4.12). System-python cross-check confirms it only works where
  protobuf 6 already broke the mp pin. Recommended: document
  MediaPipe-only; F1/F2 remain documented-not-live. dist-name footgun
  recorded: package is `retina-face`, import `retinaface`, and bare
  `retinaface` on PyPI is a different abandoned series.
- 5 evidence scripts committed (`scripts/qa/posterfp_*.py`); CLAUDE.md
  limitations updated.

---

## Next session

- **Poster-FP veto decision:** owner call on whether 14/18 (zero collateral,
  documented residual 4) is worth shipping now, or wait for a parsing-level
  signal (BiSeNet skin-mask plausibility per detected face, available
  downstream where a poster would fail face-region parsing anyway). If
  shipping: needs group-shot re-validation first — the "real-face coverage
  ≡ 1.000" property is single-subject framing; occluded group-shot faces
  will sit lower.
- RetinaFace path stays documented-not-live; do not re-attempt as a quick
  dependency addition.
- Dual-scale augment is legacy-path only; if the Tasks backend is ever
  certified for production, port the person-gated augment there too.
- The 08-15 report commit gap is closed (`7893319`); research session
  committed (`8342dbb`); dual-scale implementation committed (`8811320`).
- No other open threads flagged in the commit messages themselves.

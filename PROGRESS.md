# Progress (as of 2026-07-10)

> Snapshot of `MASTER_PLAN.md` status. Source of truth for stage receipts remains
> `MASTER_PLAN.md`; this file is a condensed progress view.

## Headline

- **~95% of plan scope complete.**
- All **6 phases ✅ COMPLETE** + P4.a ✅. 37 numbered stage rows, all `✅ DONE`.
- **11/11 flagship recipes**, **85 recipes** total.
- Core engine is **feature-complete and unit-tested**.

## Phase status

| Phase | Status |
|---|---|
| Phase 1 (foundations) | ✅ |
| Phase 2 (manual tools) | ✅ (except A2, blocked on A1) |
| Phase 3 (creative ecosystem) | ✅ |
| Phase 4 (fidelity: C3/F4/F4.b/F5/F6/F7/H1/H2) | ✅ |
| Phase 5 (harmonize/F9/F10/A5) | ✅ |
| Phase 6 (T1/T2/T3/T4/T5/A3/A4) | ✅ |
| P4.a (model-fetch infra) | ✅ |
| Phase 7 (P4 distribution hardening) | 🔴 not started |

## Leftover (~5%)

### 1. Phase 7 — P4 distribution hardening (0% started)
- Code signing, update-check, diagnostics, Windows build.
- Packaging/shipping work, not algorithm.

### 2. A1 / A2 — owner-gated tuning (blocked)
- **A1**: owner must supply Evoto / R4me / PixCake reference corpus (dev cannot do).
- **A2**: tune S2/S3/C1/C2 + freckle protection + E3 response-curve remaps to beat
  Retouch4me — blocked on A1.

### 3. Visual-QA verification passes PENDING (no new code — real-image runs only)
- F5 liquify (`geometry.py`, Visual-Critical: No-Edge-Tearing / Natural-Output / No-Halo / Background-line gates).
- F10 smart (`--smart` 20-photo mixed-folder acceptance per PLAN_TIERQ §4).
- 2026-07-09 `clear`/showcase family (`frequency`/`skin`/`freckle`/`under-eye`/`eye` modules).
- `masterwork_v1` — commit cites QA on `DSCF6102`, but that file is **not present** in
  `test_output/` (only DSCF4xxx / DSCF7xxx / DSCF8007 exist). Verify or fix the note.

### 4. Known bug, still-unfixed (documented, decision pending)
- `anime_cinematic_v1`: `relight_azimuth` / `relight_elevation` nested under `"skin"`
  instead of top level → silently ignored (resolves to defaults 0°/30° since before
  this session). Fix requires updating the recipe **and** `test_recipe_validation.py`'s
  `_VALID_KEYS` (which builds from `spec.recipe_key`, not `spec.engine_recipe_key`).

## Recipe generation artifacts
- `test_output/masterwork_v1/DSCF8007.jpg` + `_compare.jpg` + `.session.json` generated
  (denoise-first pipeline; ~120s CPU-pinned denoise on 6K).
- Full 85-recipe sweep on `DSCF8007.jpg` was **aborted** by the user; re-runnable via
  `scripts/recipes/recipe_sweep.py` or a per-recipe CLI loop. Output subdirs
  `test_output/recipes/<recipe>/` are partially populated.

## Last updated
2026-07-10 — `MASTER_PLAN.md` RESUME + rows #11 / showcase-family updated in commit
`0822b9c`.

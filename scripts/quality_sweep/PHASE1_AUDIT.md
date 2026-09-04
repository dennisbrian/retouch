# Phase 1 — Golden Fixture Audit (mutation-tested)

Repo: `/Applications/htdocs/retouch`  ·  branch `feat/color-science-k9-fix-and-frontier`
All mutations applied to a clean tree and reverted via `git checkout --`; tree verified CLEAN after each.

## Verdict

**The face harness is genuinely strong. The base harness is close to worthless.
Coverage is far narrower than "golden fixture harness" implies: 3 of 128 recipes
(2.3% of all; 2.9% of the 103 grain-free recipes eligible for snapshotting).**

## Mutation kill matrix

| # | Mutation | Target | base harness | face harness |
|---|----------|--------|--------------|--------------|
| M1 | OKLab M1 row0 +0.002 (**0.5%**) | `retouch/color_science.py` (live) | SURVIVED | **KILLED** |
| M2 | OKLab M1 row1 −0.10 (**15%**) | `retouch/color_science.py` (live) | SURVIVED | **KILLED** |
| M3 | IED landmark scaling ×1.15 | `tests/golden_face_fixture.py` | SURVIVED | **KILLED** |
| M4 | sRGB→XYZ luma row −0.10 (15%) | `retouch/color_space.py` (**dead**) | SURVIVED | SURVIVED |

The face harness kills a **0.5%** perturbation of a live color matrix. That is a
sharp fixture. Every base-harness cell is SURVIVED.

### Retracted intermediate measurement
My first run mutated `_SRGB_TO_XYZ` in `color_space.py` and both harnesses survived.
I initially read that as a coverage gap. **That was wrong and is retracted.**
`_SRGB_TO_XYZ` feeds only `bgr_to_prophoto()`, which has **no callers** in the
pipeline (`color_context.py:62` enforces sRGB-only working space). It is dead code,
so the mutation was uninformative rather than damning. M4 is retained above
*labelled as dead code* to document the difference. The real live-matrix probes
are M1/M2 against `_OKLAB_M1`, and the face harness kills both.

## Confirmed weaknesses

1. **`natural` on the base harness asserts nothing.** Its snapshot
   `f5e23f232162160b` is byte-identical to the hash of the raw synthetic input —
   verified directly. The recipe is a no-op on a face-less 64×64 gradient, so that
   parametrization locks in "the pipeline did nothing." (The other two base recipes
   *do* transform the image; they are simply insensitive to face-path and OKLab changes.)

2. **`test_snapshot_count_matches_recipes` is vacuous.** It audits
   `_get_recipe_names()` against snapshots — but that function returns the same
   hard-coded 3-name sample it is supposed to be auditing. It can never fail, and
   never notices the other **125** recipes.

3. **Deleting the snapshot file is green, not red.** A missing key auto-creates the
   snapshot and calls `pytest.skip`. Removing `golden_pipeline_face_snapshots.json`
   yields `5 passed, 3 skipped` — exit 0. The safety net silently re-arms itself
   around whatever the code currently does.

4. **Coverage is 3/128 recipes (2.3%).** Breaking that down precisely: 25 recipes are
   grain-excluded by design (`grain_strength > 0`, because `apply_film_grain` uses
   `seed=None` and is non-deterministic), leaving **103 eligible** recipes. Of those,
   **3 are covered = 2.9% of eligible; 100 eligible recipes are untested.**

## Non-findings (checked, cleared)

- **Slow tests do run.** No `-m "not slow"` in `pyproject.toml`, `conftest.py`, or
  CI; `.github/workflows/test.yml` runs the full `tests/` tree. Both golden
  harnesses execute in CI. 13 passed / 2.1s at baseline.

## Recommended fixes (not applied — audit only)

- Make a missing snapshot **fail**, not skip, unless `--update-snapshot` is passed.
- Drop `natural` from the base harness or give it an image with a detectable face.
- Make the count test enumerate `RECIPES`, not the 3-name sample.
- Raise face-harness recipe coverage well above 3.

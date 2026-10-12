# Improvements & Housekeeping Plan

Purpose
 - Capture small, high-value improvements and a low-risk rollout plan that respects the rules in `AGENTS.md`.
 - Provide a short checklist for maintainers so changes stay safe (no heavy test runs unless needed).

Guiding principles
 - Follow AGENTS.md: Visual Fidelity > Correctness > Performance > Elegance.
 - Small, reversible changes first. Avoid touching Visual-Critical modules (frequency/skin/grading/parsing/geometry) unless Visual QA is run.
 - Prefer read-only scans and tooling (visibility) before automated fixes.

Quick wins (do these first)
 1) Add a fast static-scan for Auto-CRITICAL rules
    - Script: `scripts/agents_scan.py` (planned) — runs regex/AST checks for bare `except:`, suspicious `.astype(np.uint8)` usage, `cv2.GaussianBlur`, BiSeNet mask dtype hints, and multiprocessing pickling (`FaceContext`/`ctx`).
    - Benefit: surface likely violations without running tests.

 2) Add a fast local verifier
    - Script: `scripts/verify_fast.sh` — runs `python -m py_compile` + flake8 (if available) + `scripts/agents_scan.py`.
    - Benefit: quick pre-commit check that doesn't run the full 1.6k tests.

3) Add a lightweight CI job (optional short-term)
    - GitHub Actions job that runs only `py_compile` + linters + `scripts/agents_scan.py` on PRs.
    - Keep the heavy pytest job optional / gated to a nightly or separate job.

Medium-impact improvements
 - Add `.pre-commit-config.yaml` (black, isort, flake8) to reduce noisy diffs.
 - Add `CONTRIBUTING.md` / `DEV-SETUP.md` with fast verification steps and how to run subsets of tests.
 - Centralize colorspace/dtype helpers in `retouch/color_utils.py` if repeated conversion patterns cause bugs.

Visual-Critical work (must follow Visual QA)
 - Any edits to `retouch/frequency.py`, `retouch/skin.py`, `retouch/grading.py`, `retouch/parsing.py`, `retouch/geometry.py` require: running the pipeline on reference images, diff gating, and documenting PASS/FAIL per `docs/VISUAL_QA.md`.

Branching and commit conventions
 - Use small feature branches for each change, prefixed `housekeeping/` or `improve/` (example: `housekeeping/fast-checks`, `improve/colorspace-utils`).
 - PR checklist (short):
   - Summary of change
   - Files touched
   - Tests run (or `verify_fast.sh` run)
   - For Visual-Critical changes: Visual QA results and image diffs

Checklist template for each change
 - [ ] Branch created: `housekeeping/<what>`
 - [ ] `python -m py_compile` passes for modified files
 - [ ] `scripts/verify_fast.sh` ran locally with no new findings (or findings documented)
 - [ ] PR description includes AGENTS.md references and tags claims as `[VERIFIED]` / `[ASSUMED]` / `[STALE]` where appropriate

Next suggested action (I can do now)
 - Create `scripts/agents_scan.py` and `scripts/verify_fast.sh` and commit them on branch `housekeeping/fast-checks` (read-only reporting only). This is low-risk and gives immediate visibility.

If you want that, tell me whether to create the branch `housekeeping/fast-checks` and I will implement the two scripts and push the commit locally.

---
Reference: AGENTS.md — follow its Auto-CRITICAL list and Visual QA rules when making code changes.

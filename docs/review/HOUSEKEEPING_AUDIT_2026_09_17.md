# Repo Housekeeping & Hygiene Audit — 2026-09-17

Document-only audit. **No files changed, nothing deleted, nothing committed.**
Findings + recommended actions only. Owner decides what to act on.

Scope: working tree, git state, untracked/disk artifacts, lockfiles, docs
staleness. Method: direct `git` / `du` / `ls` inspection on 2026-09-17.

---

## 1. Working Tree

| Item | State | Assessment |
|---|---|---|
| `paintx-blue-cosplay-sketch.png` (207 KB) | untracked, root | Stray sketch. Referenced by nothing in git. Move to `docs/assets/` or delete. |
| Stash `stash@{0}` (10 files, +160/−18) | WIP on `feat/color-science-k9-fix-and-frontier`, dated 2026-08-27 (3 weeks old) | See §2. |
| Everything else | clean | Tracked tree clean; no modified files. |

## 2. Stash Risk — the one real decision

`stash@{0}` touches `gui.py`, `retouch/engine.py`, `retouch/eye_enhancement.py`,
`retouch/eyes.py`, `retouch/params.py`, `retouch/parsing.py`,
`retouch/perf_optimizations.py` + 3 test files.

- **Stale:** base commit `6504109` (2026-08-19). Since then `main` moved 10+
  commits including `fafed66` "fix(eyes): recalibrate eye-occlusion gate
  thresholds" — same files (`eyes.py`, `eye_enhancement.py`).
- **Conflicts:** `git apply --check` fails — patch does not apply to current
  `main` (conflicts in `gui.py`, `engine.py`, `eye_enhancement.py`, …).
- **Not silently droppable:** 160 insertions of possibly-unshipped work
  (parsing feather + eye work + tests).
- Recommendation: `git stash branch <newbranch> stash@{0}` to rebase it onto
  current `main`, resolve, then decide. Or export: `git stash show -p stash@{0}`
  is already saved at `/tmp/stash.patch` (362 lines) — `/tmp` is volatile, copy
  somewhere durable if you keep it.

## 3. `test_output/` — 3.4 GB, mixed tracked/untracked

- 111 JSON/CSV artifacts **tracked in git** (fa02 pilot manifests, gate
  reports, summaries — all small, few-KB each; biggest 56 KB). Fine.
- The 3.4 GB is all *image payload* (fa02_texture_representation 808 MB,
  recipe_sweeps 149 MB, 11 pilot dirs ~70–86 MB each, …) — **untracked**,
  already covered by `test_output/` in `.gitignore`. No git hygiene problem,
  but a disk-hygiene question:
  - `fa02_pilot_DSCF*` dirs are from a 2026-09-06-ish pilot; if the FA-02
    gate is closed (readiness JSON last touched in `427ae49`), these ~850 MB
    of renders are probably regenerable. Candidates for archive/delete —
    **owner call, not acted on.**

## 4. Root Strays

| File | Assessment |
|---|---|
| `qa_run_all_recipes.py`, `test_visual_qa.py` | Tracked root-level scripts that arguably belong in `scripts/qa/`. Cosmetic; both tracked and referenced in history — leave unless a move is wanted. |
| `requirements-py312-migration.txt` | Header explicitly says "not consumed by pyproject or uv.lock" — intentional candidate-input file. Keep. |
| `paintx-blue-cosplay-sketch.png` | See §1. |

## 5. Tooling Config Directories

- `.quadcodeai/` (5 MB) — ignored by `.gitignore`. Fine.
- `.antigravitycli/7e883abe-….json` — **tracked** but 0 B on disk
  (79 B in index). A tracked empty session file from some tool. Harmless;
  removing from index would be tidy (`git rm --cached`), cosmetic.
- `.ai/` (24 KB, 6 md files) — tracked, looks like intentional project-state
  docs for another assistant. Last meaningful sync unknown; if stale vs
  CLAUDE.md it could mislead. Worth a skim, not urgent.

## 6. Lockfiles

Two lockfiles exist: `pylock.toml` (2026-08-13) and `uv.lock` (2026-09-08).
`uv.lock` is the newer/active one; `pylock.toml` is a pip-era artifact
(880 KB). Both are tracked. If uv is the sole resolver now, `pylock.toml`
is dead weight in the repo — candidate for deletion (owner call).

## 7. Unpushed Commits

`main` is ahead of `origin/main` by **10 commits** (latest `6795b7b`
"feat(engine): wire K1/K5/K6/K7, BB3/BB6"). Repo convention is linear `main`;
push when ready.

## 8. Docs Hygiene (light pass)

- `docs/review/` and `docs/plans/` are well-populated and date-stamped — good.
- `AUDIT_EYE_OCCLUSION_WORKINGTREE_2026_08_27.md` (2026-08-27) likely overlaps
  with the stashed working-tree state in §2 — the stash and that audit doc are
  probably the same body of work. Cross-check before resolving the stash.
- Precedent: `DOCUMENTATION_AUDIT_2026-07-15.md` self-marked "superseded" —
  good pattern; no new stale-audit offenders found in this pass.

## 9. Cache Junk

`__pycache__` present in root, `retouch/`, `tests/`, `scripts/qa/` — all
covered by `.gitignore` (`__pycache__/`, `*.pyc`). No `.DS_Store` anywhere.
No action needed.

---

## Summary — priority order

1. **Stash `stash@{0}`** — 3-week-old WIP, conflicts with current main,
   overlaps later eye-threshold commits. Resolve via `git stash branch`, don't
   drop blind. (Action recommended)
2. **10 unpushed commits on `main`** — push when convenient. (Action recommended)
3. `paintx-blue-cosplay-sketch.png` — move or delete. (Trivial)
4. `test_output/` 3.4 GB image payload — archive/delete old pilot renders if
   FA-02 is closed. (Disk only, no git impact)
5. `pylock.toml` — delete if uv-only now. (Owner call)
6. `.antigravitycli` empty tracked file — `git rm --cached` for tidiness.
   (Cosmetic)
7. `.ai/` docs — skim for staleness vs CLAUDE.md. (Low)

None of the above performed — this document is the entire deliverable.

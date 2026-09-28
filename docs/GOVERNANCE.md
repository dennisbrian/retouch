# Repository Governance

Lightweight guardrails for high-velocity, AI-assisted development. Goal:
velocity **without** documentation drift, architecture drift, broken
onboarding instructions, or repo entropy. Not a gatekeeper — a lint.

One command runs everything:

```bash
scripts/dev/governance             # full audit (~10 s, collects test count)
scripts/dev/governance --ci        # fast PR checks, ERRORs only (<1 s)
scripts/dev/governance --stats     # print generated repo stats
scripts/dev/governance --write-stats   # regenerate docs/REPO_STATS.md
scripts/dev/governance --strict    # WARNINGs also exit non-zero
```

Exit codes: `0` pass, `1` check failure, `2` setup error.

## Severity model

| Level | Meaning | Blocks CI? |
|---|---|---|
| ERROR | Dangerous / inconsistent (broken links, required module missing from arch docs, global-python commands in onboarding docs) | Yes |
| WARNING | Likely stale (unindexed doc, unlisted non-required module, stale freshness stamp) | Only with `--strict` |
| INFO | Housekeeping opportunity | Never |

CI runs `--ci` (ERRORs only) on every PR and push to main. A nightly workflow
runs the full audit with `--strict` so warnings stay visible without slowing
PR feedback.

## Checks

1. **Repo stats** — module count, test count (via `.venv/bin/python -m pytest
   --collect-only`), source LOC, git commit, timestamp. Written to
   `docs/REPO_STATS.md`. **Never hand-maintain these numbers** — human docs
   link to `docs/REPO_STATS.md` instead of restating them.
2. **Doc-link validation** — every relative Markdown link in `docs/`,
   `CLAUDE.md`, `README.md`, `AGENTS.md` must resolve (code spans ignored).
3. **Doc-index coverage** — every `.md` must be listed in `docs/INDEX.md`,
   matched by an INDEX wildcard/dir entry, or explicitly excluded.
4. **Architecture drift** — every `retouch/*.py` module must be named in a
   canonical architecture doc. Missing a module in `ARCH_REQUIRED` = ERROR;
   anything else = WARNING.
5. **Stale metadata** — `CLAUDE.md` "Last Updated" older than 14 days = WARNING.
6. **Command validation** — onboarding docs must use `.venv/bin/python -m
   pytest` (or `scripts/dev/test`), never bare `python3 -m pytest`, because the
   global interpreter is unpinned and breaks golden hashes.

## Policy tables (edit in `scripts/dev/governance.py`)

- `INDEX_EXCLUDE_*` — docs intentionally not indexed (weekly `TODO_WEEK_*`,
  append-only session logs).
- `CANONICAL_ARCH_DOCS` — the three architecture docs the drift check trusts.
- `ARCH_ALIASES` — when a module is represented by a class name instead of its
  file name (e.g. `params.py` ↔ `ParamSpec`).
- `ARCH_REQUIRED` — significant modules whose absence is an ERROR. Add new
  significant modules here when they ship.
- `COMMAND_CHECK_FILES` / `PINNED_CMD_PATTERNS` — where global-python commands
  are forbidden.

## Canonical documentation policy

One authoritative doc per architectural truth:

| Truth | Canonical doc | Others must |
|---|---|---|
| Pipeline stages / module map | `docs/architecture/ARCHITECTURE.md` | Link to it, or document a clearly different concern |
| Repo size stats | `docs/REPO_STATS.md` (generated) | Link, never restate numbers |
| Doc map | `docs/INDEX.md` | — |
| Dev workflow | `docs/CONTRIBUTING.md` | — |
| Agent conventions | `CLAUDE.md` | — |

`docs/PIPELINE.md` and `docs/PIPELINE_ARCHITECTURE.md` predate the canonical
`docs/architecture/PIPELINE_FLOW.md`. They are kept for their distinct content
(blast-radius notes; bilingual stage walkthrough) but carry a banner pointing
to the canonical flow doc. Do not add new pipeline-truth to the legacy docs.

## PR governance

`.github/PULL_REQUEST_TEMPLATE.md` asks for purpose, modules affected, tests,
compatibility, performance, and docs/architecture impact. For trivial changes
(typo, comment, config) write "trivial" in one line — the template is a
reminder, not a form.

## CI

`.github/workflows/governance.yml`:

- **PR / push to main** — `scripts/dev/governance --ci` (fast, ERRORs only).
- **Nightly + manual** — full audit with `--strict`; regenerates
  `docs/REPO_STATS.md` and fails if the checked-in copy is stale.



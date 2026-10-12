# AI Engineering Control Plane

Coordination layer for many concurrent coding agents. Sits **before** the
PR governor: the control plane decides *what should be worked on*; the PR
governor (`scripts/dev/pr_governor.py`) decides *what is safe to merge*;
post-merge validation (`pr_governor.py post-merge`) catches bad interactions.

```
IDEAS → control-plane (intake/dedupe/deps/leases/plan) → READY TASKS
      → agents (claim + intent manifest) → PRs → PR governor → merge queue
      → main → post-merge validation
```

Stdlib-only Python, repo-local state. No database, no services.

## State

- `control-plane/tasks/TASK-NNNN.json` — one file per task (human-editable;
  atomic writes; unknown keys ignored on load so the schema can grow).
- `control-plane/health.json` — circuit-breaker state.
- Both are committed so every agent/clone sees the same queue.
  (*If the queue should be local-only, gitignore `control-plane/`.*)

## Task schema

See `scripts/dev/control_plane/core.py::Task`. Fields: `id, title, type,
priority, status, subsystem, risk_hint, budget, dependencies, blocked_by,
conflicts_with, files_hint, modules_hint, acceptance_criteria,
required_tests, required_docs, assigned_agent, branch, pr, intent_manifest,
heartbeat, base_commit, created_at, started_at, completed_at`.

- Types: `feature bugfix research refactor performance docs test maintenance hotfix`
- States: `BACKLOG → READY → CLAIMED → IN_PROGRESS → VALIDATING → PR_OPEN → MERGED`,
  plus `BLOCKED / FAILED / CANCELLED`. Terminal states do not reopen.
- Priorities: `P0` hotfix … `P4` research. P0 always schedulable, even during
  a circuit-breaker pause.
- Budgets: `SMALL MEDIUM LARGE RESEARCH` — churn ceilings in
  `BUDGET_CHURN_CEILING`; overshoot produces a rescope warning, not a block.

## Commands

All commands accept `--json` (before or after the subcommand).

| Command | Purpose |
|---|---|
| `intake "title" [--type --subsystem --priority --accept "a;b"]` | Create a BACKLOG task; runs duplicate check; suggests decomposition only when the work spans subsystems |
| `list [--state A,B]` / `ready` | Queue views; `ready` = dependencies satisfied, priority-ordered |
| `preflight TASK` | Full pre-start check: cycles, deps, leases, duplicates, open-PR file overlap, base freshness, circuit breaker → `READY TO START` / `CAUTION` / `BLOCKED` (exit 1) |
| `claim TASK --agent NAME [--branch --force]` | Marks CLAIMED, writes intent manifest, takes leases, captures base commit |
| `heartbeat TASK [--note working|validating|waiting|blocked|done]` | Refreshes the lease; note may advance the state |
| `block TASK DEP...` / `unblock TASK` | Explicit dependencies; cycle-creating blocks are rejected |
| `release TASK` | Back to READY (leases lapse) |
| `complete TASK [--pr N]` | MERGED |
| `fail TASK` | FAILED — feeds the circuit breaker |
| `status` | Active work, pressure, blocked, stale, health |
| `plan [--max N]` | Recommended safe concurrent batch + HOLD/BLOCKED lists |
| `conflicts` | Pairwise MEDIUM/HIGH conflicts among active tasks (exit 1 on HIGH) |
| `drift TASK [--files ...]` | Scope-drift vs intent manifest (exit 1 on HIGH) |
| `health [--pause --resume]` | Circuit breaker status / manual override |
| `attention` | The short list for the human owner: stale tasks, cycles, failures, pause |

## Agent contract

1. `plan` (or `ready`) → pick a task; do not grab arbitrary backlog items.
2. `claim` with a stable agent name; read the printed **intent manifest**:
   `expected_subsystems`, `must_not_change`, `expected_tests`.
3. Work on the printed branch. `heartbeat` periodically — a task silent for
   72h is flagged stale in `status`/`attention` and its leases lapse.
4. Acceptance criteria were fixed at intake/claim time. Passing tests alone
   does not redefine success.
5. Before opening a PR: `drift TASK`. HIGH drift or protected-area violation
   (pipeline/params/recipes touched without being in scope) needs a written
   justification in the PR body.
6. `complete --pr N` after merge; `fail` on abandon-with-prejudice.

Stop and escalate (do not improvise) when: requirements ambiguous, an
architecture decision is needed, backwards compatibility cannot be preserved,
tests contradict documented behavior, or two active tasks conflict
fundamentally.

## Lease model

Per-subsystem, from `SUBSYSTEMS` in `core.py` (path prefixes auto-infer the
subsystem from `files_hint`):

- `SHARED` — docs, tests, metadata: no warnings.
- `CAUTION` — recipes, gui, cli, export, skin, grading: preflight CAUTION.
- `EXCLUSIVE` — pipeline, params, face-analysis, governance: preflight BLOCKED.

Leases expire `LEASE_HOURS` (72) after the last heartbeat. Coordination, not
locks: `--force` overrides a block and records the cautions.

## Duplicate detection

`duplicate_verdict` compares synonym-expanded token sets (title vs live tasks,
branch names, open PR titles). Verdicts: `DUPLICATE` (score ≥ 0.8 against a
live task), `POSSIBLE_OVERLAP`, `UNIQUE`. Weak similarity never blocks; a
DUPLICATE verdict blocks `claim` unless `--force`.

## Parallel planner

`plan` takes ready tasks in priority order and greedily admits those with no
HIGH pair-conflict (shared EXCLUSIVE subsystem or explicit `conflicts_with`)
against the batch and active work. A soft share of slots
(`HOUSEKEEPING_SHARE`, default 30%) is kept for docs/test/maintenance/
refactor/performance work so features don't consume all bandwidth.

## Circuit breaker

`evaluate_health` pauses new non-urgent feature work when, in a 72h window:
≥3 FAILED tasks, or open PRs > 12, or any CAUTION/EXCLUSIVE subsystem has >2
active tasks. During a pause only P0/hotfix/test/maintenance work passes
preflight. `health --pause/--resume` is the human override.

## Limits & honest notes

- Duplicate detection is lexical (tokens + synonyms), not semantic. It will
  miss paraphrases and over-fire on generic wording; it never hard-blocks on
  weak evidence.
- Open-PR overlap needs `gh` auth; without it those checks silently degrade
  to repo-local only (branches).
- `drift` without `--files` diffs the task branch against `base_commit`;
  on the wrong branch the file list is garbage — pass `--files` explicitly.
- The planner does not start agents; it prints a batch. Supervisor agents
  consume `--json` and run `claim` themselves.
- Queue files are shared state: two agents claiming the same task
  concurrently is a last-write-wins race. Coordinate via branches/PRs as
  usual; this is a coordination aid, not a lock server.

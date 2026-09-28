# Autonomous maintenance

`scripts/dev/maintenance` observes repository entropy and records evidence in
`control-plane/maintenance.json` (local, ignored). It uses existing governance
checks for links, index coverage, architecture drift, and guide freshness;
Control Plane tasks for repair work; and Attention Router alerts for decisions.
It does not replace the PR Governor, Quality Lab, release train, or their gates.

## Commands

```bash
scripts/dev/maintenance scan --json
scripts/dev/maintenance status --json
scripts/dev/maintenance debt --json
scripts/dev/maintenance hotspots --json
scripts/dev/maintenance plan --json
scripts/dev/maintenance campaign flaky-tests --max-tasks 2
scripts/dev/maintenance accept DEBT-... --until 2026-12-31 --reason 'legacy recipe support'
scripts/dev/maintenance suppress DEBT-... --until 2026-12-31 --reason 'known vendor issue'
scripts/dev/maintenance resolve DEBT-... --reason 'fixed in PR 123'
scripts/dev/maintenance heal index          # preview
scripts/dev/maintenance heal index --apply  # repair eligible index links in this worktree
```

`scan` updates observations, creates at most three maintenance tasks from
recurring D2/D3 findings, and syncs consequential alerts. `scan --no-tasks`
still updates evidence. `plan` is read-only. Campaigns create at most three
tasks, one per category/subsystem, from already observed findings. All task
creation uses the Control Plane's task schema and duplicate check. The Control
Plane schedules and leases the work; no maintenance command claims an agent.

## Debt and severity

The register stores stable `id`, title, category, severity, evidence, subsystem,
first/last seen, occurrences, trend, effort, suggested action, related tasks/PRs,
status, suppression expiry, and accepted reason. Observations are kept for trend
analysis. A repeat scan on the same UTC day updates the observation without
inflating recurrence. Three increasing numeric observations promote D2/D3 by one
level, capped at D4. A resolved finding that returns becomes `REGRESSED` and
gains one severity level. Missing findings become `RESOLVED` on the next scan.

| Level | Meaning | Default route |
| --- | --- | --- |
| D0 | Informational | No task or alert |
| D1 | Housekeeping | Register / digest |
| D2 | Maintenance | Bounded Control Plane task after recurrence |
| D3 | Engineering risk | Task; Attention Router after three observations |
| D4 | Architectural debt | Owner decision; no automatic refactor |
| D5 | Immediate health risk | Owner decision with critical attention |

Size and commit frequency alone never establish D4/D5. The 30-day hotspot
counter is a commit-touch measure; it does not prove regressions, agent
contention, or review cost. The scanner records a decomposition recommendation
only after investigation. Size needs a worsening trend or a matching hotspot
before it becomes a task candidate; hotspot frequency needs worsening or
regression evidence. D4/D5 require owner architecture/health judgment.

## Sources and evidence limits

- Governance supplies concrete documentation and architecture findings.
- Python AST supplies module/function line spans. Dated TODO/FIXME/HACK comments
  are checked against a configurable age threshold; undated comments are not
  called stale.
- Git commit history supplies a 30-day changed-file count.
- Exact AST bodies of functions spanning at least 20 lines in different
  `retouch/` modules identify concrete duplicate-helper candidates. Similar
  looking code is not reported without exact body evidence.
- `requirements*.txt` is checked for conflicting declarations only. Unused,
  abandoned, and outdated dependencies need package usage or registry evidence;
  the scanner makes no claim without it.
- Optional `test_output/maintenance/test-runs.json` is a JSON array of runner
  observations with `test`, `outcome` (`passed`, `failed`, `skipped`, `xfailed`),
  `at`, `duration_seconds`, and optional `retries`. Flakiness requires three
  runs with at least two failures and at least one pass. The scanner never
  quarantines or weakens a test. Runtime uses the median of at least three
  observations. Without this feed, flakiness and performance health are
  `UNKNOWN`, not `GOOD`.
- The latest Quality Lab `kind: perf` benchmark report supplies measured
  render-time and RAM ratios for recurring performance debt. The Quality Lab
  remains the source of its pass/review/regression verdict. It and the release
  train do not supply a durable per-test history; the scanner does not infer one.

## Budget, interest, and lifecycle

The suggested maintenance share starts at 20%, grows by three points per active
D3+ item and two points per worsening trend, and caps at 40%. This is an
advisory planning input, not a quota or an automatic pause on feature work.
Evidence such as repeated CI retries or hotspot touches represents ongoing
cost; the tool reports those concrete counts instead of inventing token or
human-minute estimates. Control Plane's own housekeeping share remains the
enforced planning mechanism.

States are `OPEN`, `PLANNED`, `IN_PROGRESS`, `ACCEPTED`, `SUPPRESSED`,
`RESOLVED`, and `REGRESSED`. Accepted/suppressed debt needs a reason and expiry.
It stays quiet until expiry or a substantial numeric evidence increase. The
reason and action are appended to history. An explicit `resolve` is auditable;
the next scan marks it `REGRESSED` if the signal remains. A task is considered
planned when created and in progress when its Control Plane task becomes active.
If a task finishes while the finding persists, it reopens. A follow-up scan
should confirm the cause disappeared.

## Safe repair boundary

`heal index` only adds links for existing, unindexed `docs/` files. It previews
by default and modifies the worktree only with `--apply`. Normal governance,
PR Governor, and relevant Quality Lab review still apply before merge. Other
repairs remain tasks. Architecture refactors, API removals, behavioral edits,
test weakening, major dependency upgrades, migrations, pipeline reordering,
and destructive cleanup are outside automatic repair eligibility.

## Failure modes

The register is local and ignored; copies or new worktrees start fresh unless
its state is deliberately transferred. Git history is branch-local. A missing
test feed prevents flakiness claims. Heuristic size signals can be noisy, so
tasks require recurrence and are capped. The scanner has no scheduler; invoke
`scan` from an existing supervisor or workflow. It does not claim PR validation
or auto-merge. The engineering system is production infrastructure: its
governance, task, attention, and release components need the same monitoring
and review discipline as application code.

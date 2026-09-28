# Owner attention router

`scripts/dev/attention` turns consequential engineering signals into a short owner brief. It is a local, stdlib Python CLI. It does not start agents, merge PRs, change quality thresholds, or decide architecture. One owner can inspect the entire policy and JSON state.

## Commands

```bash
scripts/dev/attention                  # daily brief
scripts/dev/attention --json           # same brief, stable schema 1
scripts/dev/attention brief --json
scripts/dev/attention decisions --json
scripts/dev/attention metrics --json
scripts/dev/attention policies --json
scripts/dev/attention record ITEM_ID defer --until 2026-10-01T00:00:00+00:00 --reason "Await corpus review"
scripts/dev/attention record ITEM_ID dismiss --reason "Known false positive" --policy-key quality-lab:visual-review
scripts/dev/attention record ITEM_ID supersede --reason "New evidence" --policy-key quality-lab:visual-review
scripts/dev/attention record ITEM_ID temporary --temporary-until 2026-10-01T00:00:00+00:00 --reason "Trial decision"
scripts/dev/attention record ITEM_ID deeper-evidence --reason "Need source/output pairs"
scripts/dev/attention ingest alerts.json
scripts/dev/attention daily 2026-09-28 42 --budget 30
```

The `ingest` file is a JSON array. Each object requires `source`, `subject`, `kind`, `title`; optional fields are `severity` (`critical`, `error`, `warning`, `info`, `success`), `priority` (`P0`–`P4`), `blocking` (number of tasks), `minutes`, `evidence`, `parent` (parent alert ID), and `resolved`. Reingesting a matching source/subject/kind updates one item. Use `resolved: true` when a producer has fixed its own alert. Never ingest successful routine CI, ordinary generated stats, known harmless warnings, or automatically resolved events.

## Existing signals

The router reads control-plane tasks, stale heartbeats, dependency cycles and health pauses; `.git/governance-state.json` post-merge failures; and Quality Lab comparison reports under `test_output/quality_lab/reports/`. It excludes Quality Lab PASS rows. Other systems can feed the `ingest` contract until they produce a stable local JSON report. Missing optional reports do not generate alerts. Reports are read-only.

## Levels and ordering

| Level | Meaning | Handling |
|---|---|---|
| A0 | Routine or resolved | Suppress |
| A1 | Low consequence | Batch when budget permits |
| A2 | Warning needing judgment | Daily brief |
| A3 | Error or blocked work | Prioritize |
| A4 | Error blocking work | Always show |
| A5 | Critical failure | Always show immediately |

Ordering uses level, blocked-task count, and task priority, then stable ID. A4/A5 bypass the daily budget. Parent alerts suppress child alerts when both appear. Identical producer keys collapse into one item. The router never rewards alert generation.

## Budget, batching, and debt

The default budget is 30 estimated owner minutes per day (`brief --budget N`). The brief selects the highest-scoring items that fit; A4/A5 always appear. Deferred items remain visible in JSON and contribute to `attention_debt_minutes`. Batches group selected decisions by source and kind; batching does not erase item IDs or evidence. Estimated minutes are planning values, not measured human time.

Record daily demand with `attention daily DATE MINUTES --budget N`. When demand exceeds budget for three consecutive UTC dates, the circuit breaker pauses new P3/P4 **feature** intake through `control-plane intake`. Hotfixes, recovery, critical fixes, and tasks near completion remain eligible. The brief reports the pause. An owner may resume normal intake by recording a day within budget or raising its recorded budget; the daily ledger retains that correction.

## Decision lifecycle and memory

`record` requires an item ID, action, and reason. Actions are `escalate`, `defer`, `dismiss`, `override`, `supersede`, `temporary`, `deeper-evidence`, and `decide`. Defer requires an ISO deadline; temporary requires an ISO expiry. Every action appends an audit entry to `decisions` and owner interventions also append to `overrides`. Dismiss/decide/override/supersede close the matching item. Escalate raises its level; defer and temporary hide it until their deadline; deeper-evidence leaves it open. The owner can use any action despite classification or budget.

`--policy-key SOURCE:KIND` extracts a reusable dismiss/defer rule from a decision. New policy entries supersede earlier active entries for the same key, preserving `superseded_by` and the old reason. `supersede --policy-key` retires the current rule without replacing it. Only dismiss and defer policies auto-resolve matching alerts. A policy can expire via `--temporary-until` or `--until`; the owner can inspect policy history with `policies --json`. Policy reuse is deterministic, not an agent inference.

The JSON state is `control-plane/attention.json` (gitignored). It contains alerts, decisions, overrides, policies, and daily budget entries. Writes use a temporary file and rename. It is local shared state, without concurrent writer locking; coordinate owner edits and back up the file when audit history matters. The state is not a legal audit log or tamper-proof store.

## Metrics and limits

`metrics --json` reports daily human decision count, estimated minutes, deferred count, repeat-question rate, policy reuse rate, explicitly marked false-positive count (`record --false-positive`), dismissed alert count, auto-resolved alerts, attention debt, alert volume, A0–A5 distribution, and overload state. `average_time_to_decision_hours` uses ingested alerts with a known first-seen timestamp and reports its sample count; it is `null` without such alerts. `blocking_tasks_unblocked_per_decision` is `null` until producers provide causal unblock links. The router does not invent measurements. Decisions/day should be few, prepared, and consequential; zero is not a target.

The live adapters cover only signals with reliable local structure. Quality Lab REVIEW still requires visual judgment. No network service, database, broker, or UI is required. The router does not infer that a warning is harmless from wording; producers must set `resolved` or the owner must dismiss it. Reused policies need owner review when conditions change. JSON schema 1 is additive; consumers should ignore unknown fields.

## Integration path

Ideas → control plane → agents → PR governor → merge queue → Quality Lab → release train → health monitor. Their actionable signals flow to this router, then decision memory and the human owner. Owner decisions inform control-plane intake, policy, architecture, and agent rules. Supervisors consume `attention brief --json` and `attention decisions --json`; they must honor owner overrides and keep A4/A5 visible.

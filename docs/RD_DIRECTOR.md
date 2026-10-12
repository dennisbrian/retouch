# R&D Director

Proposes **what to build next** from evidence gathered across the other
subsystems. Advisory only — the human approves every stage transition and
creates the actual Control Plane tasks. The director never writes code,
never creates tasks, never rewrites the roadmap.

```text
Benchmarks / Quality Lab / Maintenance / Attention / Research docs
        ↓
   R&D DIRECTOR            ← scripts/dev/rd
        ↓
Opportunities → Experiments → Approved work
        ↓
   Control Plane (human files the task)
```

## Commands

```bash
scripts/dev/rd brief                     # dashboard (default)
scripts/dev/rd evidence [--json]         # weakness signals from quality-lab / attention / maintenance
scripts/dev/rd add TITLE --kind RESEARCH --problem "..." [--evidence ...] [--effort M]
scripts/dev/rd advance OPP-001 EVIDENCE --reason "3 research notes found"
scripts/dev/rd check "idea text"         # negative-result memory lookup BEFORE starting work
scripts/dev/rd reject "what was tried" --result "what happened" [--retry-unless "..."]
scripts/dev/rd hyp "hypothesis" --success ">20% fewer failures" [--opp OPP-001]
scripts/dev/rd hyp-update HYP-001 --status EXPERIMENTING --experiment EXP-041
scripts/dev/rd log-work FEATURE "new slider" [--share 0.3]
scripts/dev/rd portfolio                 # work mix vs target allocation
scripts/dev/rd opportunities|hypotheses|negatives [--json]
```

## Lifecycle (enforced)

```text
IDEA → EVIDENCE → EXPERIMENT → RESULT → DECISION → IMPLEMENTATION → SHIPPED
                 ↘ REJECTED (from any stage, with reason)
```

`IMPLEMENTATION`/`SHIPPED` require recorded evidence — idea→coding directly
is blocked by the CLI. This is the guard against "agent finds cool library,
17 agents start coding".

## Negative-result memory

`rd reject` records what was tried and failed, with a `retry_unless`
condition. `rd check <idea>` fuzzy-matches new ideas against it so agents
stop rediscovering the same bad idea every three weeks.

## Portfolio balance

Target allocation (advisory, not a quota): QUALITY 35%, FEATURE 25%,
PERFORMANCE 15%, MAINTENANCE 15%, RESEARCH 10%. `log-work` records what
engineering actually did; `portfolio` and `brief` emit a STRATEGY WARNING
when any bucket exceeds its target by ≥30 points (e.g. 95% features while
quality debt climbs).

## State & tests

- State: `control-plane/rd.json` (schema 1, atomic writes, never committed
  if it contains session-specific paths — same treatment as other
  control-plane state).
- Tests: `tests/test_rd_director.py`.
- Stdlib only; reads quality-lab reports, attention signals, and
  maintenance debt read-only via `scripts.dev.*` cores.

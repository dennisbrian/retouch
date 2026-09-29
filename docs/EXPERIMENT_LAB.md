# Autonomous Experiment Laboratory

The Experiment Laboratory turns a registered R&D hypothesis into evidence before
production engineering starts. It creates a durable manifest under
`experiments/EXP-####/manifest.json`. Prototype code and private images remain in
ignored `test_output/experiment_lab/` worktrees. Only a qualified winner can
create a Control Plane implementation task. Prototype code is never copied or
merged by this tool.

## Lifecycle and commands

```bash
scripts/dev/experiment similar "Can local wig-edge confidence preserve hair?" --json
scripts/dev/experiment create --hypothesis HYP-001 --title 'Wig edge confidence' \
  --question 'Can local confidence preserve wig edges without background harm?' --json
scripts/dev/experiment design EXP-0001 /path/to/design.json --json
scripts/dev/experiment candidate EXP-0001 --approach 'gradient confidence' \
  --mechanism 'local gradient mask' --rationale 'uses local image structure' --patch /path/to/a.patch
scripts/dev/experiment candidate EXP-0001 --approach 'segmentation fusion' \
  --mechanism 'mask boundary fusion' --rationale 'combines semantic and edge evidence'
scripts/dev/experiment prepare EXP-0001 A --json
scripts/dev/experiment run EXP-0001 A --split development --json
scripts/dev/experiment run EXP-0001 A --split validation --json
scripts/dev/experiment compare EXP-0001 --json
scripts/dev/experiment run EXP-0001 A --split holdout --json
scripts/dev/experiment compare EXP-0001 --json
scripts/dev/experiment results EXP-0001 --json
scripts/dev/experiment promote EXP-0001 --json
```

`list`, `show`, `status`, `history`, and `negatives` support supervisor agents.
`eliminate` records a reason when an agent stops an unpromising approach.
`archive` retains evidence. `cleanup` previews old prototype worktrees; deletion
requires `--apply --discard-prototypes` after the retention period. Each command
accepts `--json` after the command or before it.

States are `PROPOSED`, `DESIGNED`, `RUNNING`, `ANALYZING`, `COMPLETED`,
`INCONCLUSIVE`, `REJECTED`, `PROMOTED`, and `ARCHIVED`. Candidate states also
include `READY`, `MEASURED`, and `ELIMINATED`.

## Freeze design before candidate code

`design` accepts a JSON object such as:

```json
{
  "base_commit": "FULL_40_CHARACTER_GIT_SHA",
  "baseline_report": "/absolute/path/to/test_output/quality_lab/reports/baseline.json",
  "corpus_manifest": "/absolute/path/to/quality_lab/corpus_manifest.json",
  "recipes": ["natural", "cosplay_clear_v1"],
  "success_criteria": {
    "metric": "regions.HAIR.edge_retention",
    "direction": "higher",
    "min_delta": 0.05,
    "min_positive_fraction": 0.8,
    "quality_tolerance": 0.01
  },
  "failure_criteria": {
    "max_regression_fraction": 0.02,
    "max_runtime_ratio": 1.10,
    "max_ram_ratio": 1.15
  },
  "constraints": {"max_loc_added": 500, "max_changed_files": 8},
  "reject_all_if": "Every candidate harms background separation or fails the holdout."
}
```

The report must have been made by Quality Lab from the exact clean base commit.
Its recipe list, input and output hashes, corpus manifest, and pinned thresholds
must match. Design records their hashes and cannot be edited through the CLI
after acceptance. `min_delta` is an absolute paired metric improvement in the
specified direction. It is not a percent or a claim of perceptual quality by
itself. At least 80% of paired case-recipes must improve by default. Choose a
target metric relevant to the hypothesis and review the other
region metrics as regression evidence.

The lab assigns cases deterministically by subject when `subject` is present,
or by case ID otherwise. You may provide an explicit `split` object containing
`development`, `validation`, and `holdout` case-ID arrays; it must assign every
case once and keep subjects together. Holdout results are unavailable until
development and validation pass and `compare` selects finalists. A corpus with
fewer than six independent groups has no qualified holdout. Production
promotion also requires at least two holdout cases marked
`"evidence_class": "real_photo"` in the corpus manifest. The current synthetic
seed corpus cannot by itself qualify a production winner.

## Candidate isolation and budgets

`prepare` makes a detached Git worktree at the frozen commit. A candidate may
provide a patch or an agent may edit that worktree. Only `retouch/`, `presets/`,
`cli.py`, and `gui.py` may change. Tests, benchmark policy, corpus, release
infrastructure, docs, dependencies, and experiment metadata cannot be changed
by a candidate run. Before each run, the lab verifies HEAD, changed paths,
threshold hash, and a code fingerprint. Candidates use the same recipes,
environment pins, input copies, metric definitions, and frozen baseline.

| Budget | Candidates | Refinement rounds per candidate | Total runner time |
| --- | ---: | ---: | ---: |
| TINY | 2 | 1 | 15 minutes |
| SMALL | 3 | 2 | 1 hour |
| MEDIUM | 4 | 3 | 3 hours |
| LARGE | 5 | 3 | 8 hours |

TINY uses at most three development cases and is exploratory only. Runner
time is bounded by the remaining budget. Each refinement must change code;
validation and holdout must use the same fingerprint as development. Holdout
may run only once per candidate. Severe development regressions or budget
breaches eliminate a candidate early. More candidate approaches are justified
only when their mechanisms differ; the CLI rejects near-identical mechanism
descriptions, while human/agent review still checks real design diversity.

## Comparison and evidence

Quality Lab supplies per-case image metrics, visual regression verdicts, time,
and RAM. The lab reports paired target deltas with sample size and spread,
plus a regression matrix by candidate and split. It does not collapse visual
quality into similarity alone. Runtime/RAM gates require Quality Lab cases with
stable baseline time of at least two seconds; otherwise performance evidence
is missing and the candidate cannot pass. Complexity records files, added LOC,
and touched subsystems, with no new dependency allowed. LOC is a guardrail,
not a quality score.

`compare` returns `NO_WINNER` when measured candidates fail, `INCONCLUSIVE`
when evidence is missing or tradeoffs remain, and `WINNER` only when frozen
criteria pass across development, validation, and a qualified real-photo
holdout. It shows the Pareto frontier for target improvement, runtime, LOC,
and regressions. A near-quality tie may favor the faster simpler option when
the frozen tolerance permits it. Negative findings remain in the manifest and
are also copied into R&D Director negative-result memory. Similarity checks
search previous experiments, R&D negatives, research document titles, owner
decision memory, and recent PR titles when GitHub is available;
retrying a duplicate requires new evidence.

Where visual judgment matters, `blind` copies candidate and baseline images
to neutral filenames under ignored output space. The identity key remains
separate until `review` records preferences from a JSON vote array such as
`[{"case":"case-id","recipe":"natural","preferred":"X"}]`. Blinding
reduces naming bias but does not conceal visual artifacts or stop an operator
from inspecting local files.

## Integration and safety

`promote` creates a normal Control Plane task with evidence and required
validation. Production agents must reimplement the proven behavior cleanly,
then pass PR Governor, Quality Lab, merge queue, and release gates. Clear
winners enter Attention Router at A2; unresolved tradeoffs enter at A3, or A4
when the frozen design marks `architecture_review_required`. A5 is reserved for
severe production evidence from the existing health and release systems; an
experiment result alone cannot establish a production incident. Routine
progress and no-winner outcomes do not ask the owner to decide. `status --json`
exposes running, analyzing, completed, winner, no-winner, promotion, compute,
candidate, negative-reuse, and human-review counts for Mission Control. There
is no Mission Control module in this checkout, so this is an integration
contract rather than a UI change.

The current baseline report is from a dirty tree and an older commit. It must
be regenerated from a clean fixed base before a real experiment can begin.
Worktrees isolate Git changes, not arbitrary Python process access to the host;
run only trusted candidate code. Private corpus files and rendered images stay
ignored. The CLI has no scheduler or model that writes candidate algorithms;
agents use the candidate specification and worktree to implement each approach.
Only evidence and decisions are durable.

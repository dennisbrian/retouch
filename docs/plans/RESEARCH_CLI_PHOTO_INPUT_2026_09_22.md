# CLI photo input: research and improvement proposal

Date: 2026-09-22 (Asia/Kuala_Lumpur)

Status: research baseline plus implementation addendum; no photo-quality claim

## 1. Recommendation

Improve the step between choosing photos and starting retouching: a shared,
inspectable **input plan**. It should say exactly which files were selected,
why others were excluded, which decoder each file needs, and where every
artifact will go. Both dry-run and execution should consume that plan.

The existing CLI already has important safeguards. The highest-value next step
is making those safeguards visible before a run, then making photo selection
more convenient. Adding more formats or increasing worker counts comes later.

Suggested order:

1. Make dry-run reflect real destination checks and fix selection inconsistencies.
2. Support several selected photos and explicit include/exclude rules.
3. Add staged format/decode checks with actionable per-file reasons.
4. Persist the selection and results so reruns are understandable and resumable.
5. Calibrate resource estimates using decoded dimensions and actual processing.

The implementation addendum at the end records the now-landed CLI workflow.

## 2. Scope and evidence boundary

Reviewed the checked-out source at `cdfd6fe5b65794e963480117d696ff0b8aff828a`
on `feat/color-science-k9-fix-and-frontier`, including the existing dirty
worktree, and compared CLI/I/O code with newer local `main` at
`6795b7bf2e9f5ecf9b0ea60cdde3d14cf0781fb7`. No branch switch or fetch.

Evidence below is static source inspection, test-assertion inspection, and
primary documentation. Example failure paths are source-derived, not executed
reproductions. Existing test files are not evidence of a passing run today.

The initial research pass made no production edits or photo experiments. The
implementation addendum was subsequently applied as a bounded CLI change and
verified with focused syntax, planner, helper, and CLI compatibility checks.
No face-smoothing quality is claimed.

## 3. What already exists

| Capability | Current evidence | Keep / clarify |
| --- | --- | --- |
| One file or one folder, optional recursion | `cli.py:389–398, 651–666, 837–845` | Preserve simple existing commands |
| Sorted discovery and case-insensitive extension filtering | `find_images()` sorts and lowercases suffixes | Do not describe these as missing |
| Recursive output hierarchy | `_destination_for_image()`, `cli.py:401–420` | Keep relative subfolders, not a flattened batch |
| Original protection and artifact collision checks | `_assert_safe_destination()` / `_preflight_destinations()`, `cli.py:423–517` | Covers outputs, comparisons, session paths, and filesystem aliases; `--force` must not override source protection |
| Recursive output-tree containment check | `cli.py:459–469` | Output inside the input tree is already rejected |
| Orientation, ICC context, RAW selection | `retouch/io.py:491–614, 739–815` | Surface these existing decisions in preflight; do not invent a second decoder |
| Save/reuse processing settings | `cli.py:117–174, 713–724`; `retouch/session.py:311–344` | A session is not a completed-batch ledger |
| Destination-volume capacity check on newer main | `main:cli.py::_check_disk_space()` | Already implemented there; absent in this older checkout |

On newer main, disk checking groups sibling destinations by device or probes
the nearest existing ancestor of an explicit output directory. Its estimate
uses input bytes × 2, doubled when comparisons are enabled, with 5 GiB headroom.
Although its docstring calls it warn-only, its low-space branch prints a warning
and exits 1 unless bypassed. Report actual behaviour, not that stale wording.

Newer main also contains worker teardown changes and a `--preflight-check`
system-integrity/throughput mode. That mode is not a per-photo input plan and
can instantiate the engine. This proposal must not conflate those meanings.

## 4. Source-backed gaps

### A. Dry-run does not preview the run's actual safety decision

`cli.py:894–900` returns after listing filenames, settings, and workers.
Output resolution and `_preflight_destinations()` happen later at `904–922`.
The same ordering remains on inspected main, including its later disk check.

Consequently, a dry-run can accept a JPEG with no separate output folder even
though execution would reject writing back to that source. It also omits
planned destinations, collision reasons, and existing-output dispositions.

It is not uniformly a metadata-only operation either: `--extract-look` reads
reference images before the dry-run branch (`876–892`), while `--color-ref`
loading happens after it through `_finalize_params()`. Distinguish inspecting
arguments from resolving image-derived settings. Do not promise that the current
dry-run is dependency-free: the module imports the engine and imaging libraries
at startup.

**Proposal:** one planner used by preview and execution. Planning never creates
output directories or starts a retouch engine. Reference-derived settings not
computed at that level must be labelled unresolved. An explicitly requested
report file is the only permitted planning write.

### B. Selection has avoidable inconsistencies

The positional argument uses `nargs="?"`: one optional input, not a list.
Passing several selected/dragged files is therefore not supported. A shell glob
expanding to multiple paths also exceeds that contract; a quoted glob is treated
as a literal path, not expanded by the application.

The initial input existence check uses `Path(args.input)` without `expanduser()`.
A literal quoted `"~/Pictures/Event"` is not expanded, whereas the output path is
expanded later. Ordinary spaces already work when the shell supplies one correctly
quoted argument; the problem is not general space handling.

`find_images()` accepts any explicit existing file without extension screening.
Folder discovery screens by suffix but does not require `is_file()`. Thus a
directory named `archive.jpg` can become a candidate, while a directly supplied
text file is listed by dry-run. This explicit-file/folder distinction could be
intentional content probing, but there is no probe in that dry-run to justify it.

There are no include/exclude or hidden/generated-artifact policies in this
function. An old comparison or retouched image inside a searched source tree can
be selected. Existing nested-output rejection does not filter historical exports
already present elsewhere in that tree. Python's path matching includes dotfiles;
filesystem error and symlink behaviour also varies with Python version.
[Python pathlib documentation](https://docs.python.org/3.13/library/pathlib.html#comparison-to-the-glob-module)

**Proposal:** consistent file-type checks; normalized paths with original spelling
retained for diagnostics; explicit selection policies; visible exclusions; and
well-defined link handling. Never silently drop an explicitly named photo merely
because its name contains `retouched` or `compare`.

### C. RAW+JPEG pairs are a selection/naming decision, not just a collision

Both `DSCF0001.RAF` and `DSCF0001.JPG` can be discovered. With JPEG delivery they
claim the same destination and correctly fail current preflight. `--force` is
not a solution.

**Proposal:** show the pair before rendering. Preserve both by explicit,
deterministic source-extension disambiguation, or let the user request RAW-only
or JPEG-only selection. Do not silently prefer RAW: the camera JPEG and RAW
development are different renditions. Basename matching alone is not proof that
files across different cards are the same photograph.

Reuse the idea in `retouch/batch_processor.py:81–128`, which already plans
extension-disambiguated destinations. Do not import that entire processing
subsystem or silently change the main CLI's existing naming rules.

### D. A recognised suffix is not a verified input

`IMAGE_EXTENSIONS` lists ordinary images, RAW families, and EXR. Actual decoder
selection happens inside I/O after execution starts. RAW support is optional
(`pyproject.toml:55`), and `raf2jpeg` resolution happens during conversion
(`retouch/io.py:631–718`). In the serial path the engine is created before the
first input is decoded (`cli.py:956–984`).

HEIC/HEIF and AVIF are not in the inspected folder-discovery allowlist. This does
not prove that every directly supplied file of those types will fail on every
machine: available Pillow plugins are a separate matter. Do not claim support
merely by adding an extension to the list.

EXR needs special caution: newer main adds an EXR branch to `read_image_16bit()`,
but inspected `imread_engine_with_context()` routes only RAW suffixes through
that helper. Ordinary EXR CLI input still takes the non-RAW path. This is a
source-level wiring concern, not a tested claim that every EXR fails. The separate
[EXR/range study](RESEARCH_RETOUCH_IMPROVEMENTS_2026_09_21.md#4-colorexport-findings-from-the-newest-main-commit)
also prevents treating header recognition as color-delivery certification.

**Proposal:** report separate statuses for discovery, decoder availability,
header inspection, full decode, and color/precision conformance. Never collapse
`not_checked` or `unavailable` into `passed`.

Pillow opens images lazily. Its `verify()` attempts integrity checks without
decoding pixels, and the file must be reopened afterward; neither header success
nor `verify()` replaces a full decode. Keep decompression-bomb protections and
explicit pixel limits rather than disabling them for convenience.
[Pillow Image documentation](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.verify)

### E. Input transformations need to be visible

The current non-RAW path applies orientation, converts tagged input into working
sRGB, records assumed-sRGB for untagged input, reduces high-bit input to its
current 8-bit working contract, and flattens non-opaque alpha on white in linear
light (`retouch/io.py:491–614`). These are material photo-input decisions.

`output_format()` (`867–885`) preserves PNG and WebP under `same`, but otherwise
chooses JPEG, including TIFF/BMP/RAW/EXR. The help and batch guide's “same as input”
wording is therefore too broad. `--bit-depth 16` can force a PNG container without
proving end-to-end 16-bit processing.

**Proposal:** the plan shows detected source format, orientation, dimensions,
source/working precision, ICC assumption/conversion, alpha policy, requested and
applicable decoder, and resolved output format. Before actual decoding, label
transformations as planned, not already applied. No silent color/decode fallback.

Older color documents contain historical implementation gaps. Current code
already pins RAW linear gamma before explicit sRGB encoding and performs
native-profile conversion; do not reopen those as absent features based only
on the old text. This turn does not certify their visual correctness.

### F. Existing output is not evidence that the input was completed

Workers skip when the main output exists and `--force` is absent
(`cli.py:213–215`), without validating its pixels, parameters, or required
comparison/session artifacts. The serial path decodes first and then checks
existence (`956–1010`); the pool task checks before decode, but its worker
initializer can already have constructed an engine. These paths have different
cost and failure ordering for the same nominal skip.

The CLI collects failures and prints totals, but its normal end has no
`failed > 0` nonzero-exit decision (`1088–1099`). Handled failures, including
counted QA failures, can therefore end with exit 0. This is not a claim that
all failures exit 0: preflight errors and uncaught exceptions behave differently.

**Proposal:** distinguish `skipped_existing_unverified` from verified resume;
use stable source IDs/relative paths rather than basenames in result records;
and align serial/pool row outcomes and aggregate exit semantics. Do not silently
overwrite mismatched existing outputs.

Reuse the existing proposed hash/atomic-write/resume contract in
[full-v2 engineering plan, sections 8–9](PLAN_AMGDAY32026_FULL_V2_ENGINEERING_2026_09_01.md#84-atomic-write-protocol).
It is a design reference, not implemented general-CLI resume. Session JSON
stores settings and a short image hash; it is not a complete input/result manifest.

### G. Compressed size and CPU count do not describe the input workload

Workers default to half the CPU count (`cli.py:703`). `--max-dim` resizing occurs
after decoding, and comparison handling can retain a full-resolution copy.
Therefore that flag does not itself cap initial decode memory. Main's disk
estimate uses compressed input size, not resolved format, dimensions, or depth.

**Proposal:** estimate resources from dimensions, decode dtype, comparison
buffers, engine/model overhead, and concurrency. Report a range and assumptions,
not an exact RAM promise. Skip dispositions should affect resource estimates.
The queue budget helper at `retouch/batch_processor.py:131–153` is a reuse lead,
not a calibrated CLI peak-memory bound. Do not change worker defaults until an
approved native-resolution study measures them.

## 5. Proposed input-plan contract

Keep these stages separate:

```text
explicit paths / folders / saved selection
  -> deterministic discovery + exclusion reasons
  -> source identity + decoder/header observations
  -> destination/artifact safety + resource estimates
  -> display or save immutable plan
  -> recheck source/destination state -> decode -> retouch -> result ledger
```

| Level | Intended work | Must not imply |
| --- | --- | --- |
| Paths | Parse, traverse, stat, classify, plan destinations, report collisions | Decodable photos or usable face models |
| Headers | Inspect supported container headers/metadata and decoder capability | Full pixel decode, correct ICC conversion, or face detection |
| Decode, explicitly requested | Use the real configured I/O route, one bounded image at a time, no retouch | Visual quality, successful later export, or immutable files |
| Execution | Revalidate and use the same planned sources/destinations | That an old plan guarantees current filesystem state |

RAW header inspection is decoder-specific and may do more work than a lightweight
JPEG header read. Record unavailable observations; do not invoke full RAW
development silently inside a “paths only” plan.

Minimum row data: stable row ID, original input token, normalized path/root-relative
path, selection origin, file kind, size/mtime, exclusion/rejection reason,
inspection level, decoder requested/planned/actual, metadata observations,
planned artifacts, existing-output disposition, and blocking/advisory findings.
Record schema version, source revision, selection policy, and configuration hash
at job level. Avoid embedding photo pixels, GPS, or unnecessary EXIF in reports.
Provide relative-path export/redaction for reports shared outside the workstation.

Size/mtime is a cheap stale-plan check, not a content identity guarantee. A later
resume mode needs content hashes plus settings/runtime/artifact validation.
Recheck originals and destinations at use time; preflight is not a sandbox or
a defence against all filesystem races. Never delete sources or old outputs.

## 6. Selection interface proposal, not implemented syntax

Preserve existing single-input commands. First add multiple explicit paths;
keep recipe-list/help modes working with no input. Review ambiguity with optional
`--save-session [PATH]`; simply changing one `nargs` value is not the full design.
Use standard argument parsing, not shell evaluation or manual splitting on spaces.
[Python argparse](https://docs.python.org/3.11/library/argparse.html#nargs)

Candidate interactions for owner review:

```text
# PROPOSED ONLY: several dragged/selected photos, planned without rendering
cli.py --output "/photos/event-retouched" --dry-run -- "/photos/A 01.jpg" "/photos/B 02.RAF"

# PROPOSED ONLY: select RAW files and explain each excluded file
cli.py "/photos/event" --recursive --include "*.RAF" --exclude "exports/**" --dry-run

# PROPOSED ONLY: replay an explicit selection, without treating it as completed work
cli.py --input-list "selected-photos.json" --output "/photos/event-retouched" --dry-run
```

Design defaults to settle before implementation:

- Multiple roots: preserve a root identifier plus relative path; require unique
  root aliases or reject conflicts. Do not flatten two memory cards together.
- Patterns: defined relative to each root, with an explicit case policy and
  include-then-exclude precedence. Explicit paths remain literal, not auto-globs.
- Hidden/generated files: proposed exclusion for discovered hidden/sidecar files;
  generated-output exclusion should rely on known artifacts/manifests or explicit
  patterns. Filename heuristics alone get a warning, not silent removal. An
  explicit file must receive its own visible decision even if a policy rejects it.
- Links/duplicates: propose no discovered directory-link traversal by default;
  report outside-root file links and cycles. Deduplicate repeated references to
  the same filesystem object visibly; do not deduplicate distinct photos by
  perceptual similarity or silently rewrite Unicode filenames.
- Input-list: versioned JSON with literal paths and a documented base directory
  is a better first contract than a file interpreted as arbitrary CLI arguments.
  No nested includes, command execution, or implicit environment substitution.
- Piped lists: defer until needed; use NUL-delimited paths rather than whitespace
  splitting so unusual filenames survive. This follows established
  [GNU filename-handling guidance](https://www.gnu.org/software/findutils/manual/html_node/find_html/Safe-File-Name-Handling.html).
- RAW+JPEG: default remains explicit conflict reporting until the user chooses
  a pair policy or approves disambiguated output names.

Prefer an explicit sibling output directory for event work. Do not auto-create
one during planning, remove the existing overwrite guards, silently select
`--global-only`, or change the user's established face-aware recipe convention.

## 7. Bounded implementation and validation plan

The implementation addendum below applies the CLI workflow portions of these
tranches. The table remains the acceptance checklist for later RAW/EXR,
native-resolution resource, and visual-delivery validation; those are not
claimed complete merely because planning and hashing now exist.

| Tranche | Scope | Required evidence before claiming completion |
| --- | --- | --- |
| 1: Truthful planning | Shared planner; input path/file-kind fixes; dry-run safety parity; clarify actual output formats | Preview and execution agree on every planned artifact and blocking reason; planning initializes no engine and writes nothing by default |
| 2: Better selection | Multiple paths; include/exclude; explicit multi-root and RAW+JPEG policy | Stable ordering, visible exclusions, no lost explicitly selected photos, unchanged single-input compatibility |
| 3: Capability inspection | Header/decode levels; decoder and color/precision observations | Real decoder routes; unavailable stays distinct from passed; bounded resource use; no silent fallback |
| 4: Reliable reruns | Per-file ledger, aggregate exit truth, then verified resume | Corrupt/stale outputs cannot count as complete; serial/pool outcome parity; interruption recovery under the existing atomic-write design |
| 5: Capacity calibration | Memory-aware scheduling and destination-format estimates | Approved native RAW/JPEG cases, measured peak memory and disk usage; no unsupported speed/quality promises |

Future fixture matrix should include:

- Paths with spaces, Unicode, quotes, literal glob characters, a leading dash,
  quoted tilde, repeated paths, hardlinks, symlinks, missing/unreadable roots,
  directory names ending in `.jpg`, and files removed after planning.
- Mixed uppercase/lowercase images, sidecars, hidden files, existing comparisons,
  two cards with identical basenames, RAW+JPEG pairs, and nested output roots.
- Truncated/zero-byte/renamed files, decoder absent vs unsupported content,
  EXIF orientations 1–8, CMYK/gray ICC, untagged JPEG, transparent PNG, high-bit
  TIFF/PNG, multi-frame inputs, very large dimensions, and RAW/EXR path parity.
- Existing main output but missing comparison/session; changed source/settings;
  changed output after planning; per-file failure; all-skipped; and serial/pool
  equivalence. Decide multi-frame policy explicitly, not by accidental first-frame use.

Existing helper tests cover important destination protections. The current
`test_unsupported_format` in `tests/test_cli.py:138–147` expects dry-run exit 0
for a text file, with a comment claiming it was skipped; the source actually
lists it. Future tests must assert row dispositions and reasons, not just success
codes. `tests/test_cli_integration.py:444–464` checks that dry-run does not write
images, but does not establish destination-validation parity.

The fixtures described above were not used for visual-quality claims. Focused
planner and CLI compatibility checks are recorded in the implementation
addendum below.

## 8. Documentation follow-up

When implementation is separately approved, update the batch guide to distinguish
face-aware commands from its current global-only quick start, document real
`same` format mapping, show safe sibling destinations and selection examples,
and separate input planning from main's system `--preflight-check`.

Python compatibility is part of the design: the repository declares Python 3.9+
and its inspected test matrix targets 3.9–3.11. Do not require newer pathlib
features such as `Path.walk()` or `recurse_symlinks` without an explicit version
decision. Available older primitives already cover path expansion and file-kind
checks. [Python 3.9 pathlib](https://docs.python.org/3.9/library/pathlib.html)

Bottom line: **make photo selection and preflight trustworthy first; add richer
selection second; treat new formats and faster concurrency as separate, verified
capabilities.**

## 9. Implementation addendum — 2026-09-22

The bounded CLI implementation now includes:

- `retouch/cli_input.py`: deterministic multi-token discovery, literal JSON
  input lists, hidden-file policy, include/exclude filters, duplicate reporting,
  RAW+JPEG `error`/`raw-only`/`jpeg-only`/`suffix` policy, header checks, optional
  full decoder checks, working-memory estimates, planned artifacts, and a
  hash-backed result ledger covering all planned artifacts;
- shared destination planning for dry-run and execution, including the existing
  source-overwrite and collision guards;
- `--input-plan` for a JSON selection/result record and `--resume-plan` for
  verified source/output/config matching; stale or mismatched rows rerender;
- `--input-check {paths,headers,decode}`, explicit `--multi-frame-policy`, opt-in
  `--max-input-pixels`, and
  `--ram-budget-gib` worker capping, plus destination-volume disk preflight
  with `--skip-disk-check`; and
- aggregate nonzero exit status when one or more files fail.

Dry-run remains non-mutating and informative: it prints blocking output
conditions instead of creating directories or returning a render success. A
normal run still refuses those conditions before starting the engine. Existing
single-file/folder commands and face-aware defaults remain compatible.

Focused evidence after implementation: `11 passed` planner tests,
`11 passed, 1 skipped` CLI-helper tests, and `41 passed` across
`tests/test_cli.py` and `tests/test_cli_integration.py`. Syntax compilation for
`cli.py` and `retouch/cli_input.py` passed. No full repository suite, RAW decode,
real-photo render, or visual-quality comparison was performed in this tranche.
EXR/HDR delivery and new HEIC/AVIF support remain intentionally deferred to a
separate color/decoder contract; the implementation does not claim those
formats are supported merely because a suffix is recognized.

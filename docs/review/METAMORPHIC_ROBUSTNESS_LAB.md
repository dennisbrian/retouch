# Metamorphic Robustness Lab

The lab is a cross-cutting QA gate for image-processing stages. It generates
controlled variants for rotation, proxy resize, JPEG quality, exposure,
white-balance shift, and bit-depth roundtrip. Each variant carries an inverse
transform; the lab compares the restored result with the baseline and can also
compare stage decisions supplied by a decision provider.

Run a diagnostic global-only report with:

```bash
python scripts/qa/metamorphic_lab.py \
  docs/reference_targets/chang_e_cosplay_tamed_shine.jpg \
  --recipe natural --global-only --max-dim 1200 \
  --output /tmp/metamorphic.json
```

`global_only: true` is explicitly non-certifying for face-aware behavior. A
face-aware run requires the GUI-attached native MediaPipe environment and must
record independent `face_aware_run`, `automatic_pass`, `human_review`, and
`final_certified` fields before it can gate shipping. Thresholds are
configurable because different stages have different legitimate sensitivity;
the report always retains the raw error metrics and reason for failure.

Initial diagnostic smoke result on the bundled Chang'e reference at 600px
(`global_only: true`): rotation, JPEG, exposure, white-balance, and bit-depth
roundtrip passed; proxy resize flagged p95 error `25.0` against the default
`24.0` threshold. This is an actionable robustness finding, not a reason to
weaken the threshold and not face-aware certification.

## Verification status

The local Gradio composition graph has been built and checked: the Advanced
Retouch editor, apply/undo/redo/reset actions, snapshot comparison, mask
overlay, and export action all have registered backend event bindings. The
focused Advanced Retouch tests also cover a deterministic semantic-mask
intersection and snapshot/export round trip.

The default system environment remains blocked by MediaPipe's native
`DrishtiMetalHelper` service initialization, but the pinned MediaPipe 0.10.5
CPU/XNNPACK environment now passes the real-image probe and face-aware
Advanced Retouch runner. The successful evidence has `global_only: false`,
`face_aware_run: true`, and `automatic_pass: true`; human review remains
separate and pending.

Verified command (pinned isolated environment):

```bash
RETUCH_GPU=0 RETOUCH_MEDIAPIPE_BACKEND=legacy \
python scripts/qa/face_aware_runtime_probe.py test_output/DSCF8007.jpg
python scripts/qa/advanced_retouch_visual_qa.py test_output/DSCF8007.jpg \
  --output /tmp/advanced-retouch-face-aware
```

Observed result: one face, 478 landmarks, six rendered Advanced Retouch
cases, `global_only: false`, and `automatic_pass: true`.

## Advanced Retouch visual evidence

The active feature has a dedicated evidence runner:

```bash
python scripts/qa/advanced_retouch_visual_qa.py \
  /path/to/portrait.jpg \
  --output /tmp/advanced-retouch-qa
```

The default run exercises semantic intersection, per-face selection, healing,
removal fallback, and subtle face reshape, then writes per-case outputs,
`contact_sheet.jpg`, and `manifest.json`. A headless smoke run may add
`--global-only`, but that manifest is explicitly non-certifying.

Repository-native portrait sample used for the current gate:

```bash
python scripts/qa/advanced_retouch_visual_qa.py \
  test_output/DSCF8007.jpg \
  --max-dim 1200 \
  --output /tmp/advanced-retouch-dscf8007
```

`DSCF8007.jpg` is a clear single-portrait sample with wig/hair detail,
hands crossing the body, and mixed warm/cool venue lighting. The current run
records `status: blocked` because MediaPipe 0.10.35 aborts natively in
`DrishtiMetalHelper` before face detection; this is an environment/runtime
failure, not an absent sample or a passing certification result.

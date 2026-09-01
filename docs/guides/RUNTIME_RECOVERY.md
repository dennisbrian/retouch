# Runtime recovery and Python 3.12 migration

**Audit date:** 2026-08-15
**Last verified:** 2026-08-31 on macOS 26.5 arm64

**Scope:** dependency declarations, isolated-environment recovery, and the
separate migration track. This guide does not change application Python
modules or claim that a GUI launch is face-aware.

## Decision

Retouch has two deliberately separate runtime tracks:

| Track | Interpreter | Dependency source | Face backend | Status |
|---|---|---|---|---|
| Legacy recovery | CPython 3.11 (3.9 is not installable — see [Onnxruntime wheel gap](#onnxruntime-wheel-gap-and-the-311-recovery-track) below) | `pyproject.toml` + locked `uv.lock` | MediaPipe 0.10.5 legacy CPU `mp.solutions.face_mesh` | The current application track; must be installed in isolation |
| Python 3.12 migration | CPython 3.12 | `requirements-py312-migration.txt`, then a separately generated migration lock | MediaPipe Tasks `mp.tasks.vision.FaceLandmarker` | Candidate only; the current application modules are not certified against it |

The repository does not currently carry a Python patch-version declaration.
The 3.11 choice above is the only installable recovery interpreter for the
legacy lock on the verified platforms (macOS arm64, 2026-08-31), not a claim
that every Python 3.11 patch release is equivalent.

Do not use the Python 3.12 candidate file with `uv sync`. The project lock is
the legacy lock and intentionally remains unchanged until the Tasks/API port
has its own verified resolution and acceptance evidence.

### Onnxruntime wheel gap and the 3.11 recovery track

The original recovery target was CPython 3.9.6, matching the project's
`requires-python = ">=3.9"`. That path is **not installable** with the current
`uv.lock`:

- The lock maps `onnxruntime==1.20.1` to `python_full_version < '3.10'`, but
  onnxruntime 1.20.1 publishes only `cp310`/`cp311`/`cp312`/`cp313` wheels —
  there is no `cp39` wheel for any platform. `uv sync --locked` hard-fails on
  CPython 3.9 with: *"onnxruntime (v1.20.1) only has wheels with the following
  Python implementation tags: cp310, cp311, cp312, cp313"*.
- The same gap affects the 3.10 branch: `onnxruntime==1.24.3` is mapped to
  `python_full_version == '3.10.*'` but ships only `cp311`+ wheels.

The only onnxruntime pin in the lock that resolves is `1.26.0` (mapped to
`python_full_version >= '3.11'`). On CPython 3.11 the lock still preserves
every other legacy contract pin exactly — `mediapipe==0.10.5`,
`protobuf==3.20.3`, `opencv-contrib-python==4.11.0.86`, `numpy==1.26.4` —
so 3.11 is the recovery interpreter. This is not a re-resolution: the lock is
used unchanged with `--locked`. The single documented deviation is onnxruntime
itself (`1.26.0` instead of the un-installable `1.20.1`), which still
satisfies the declared `onnxruntime>=1.16.0` constraint.

Verified on 2026-08-31: `uv sync --locked --extra gui --extra dev` on
CPython 3.11.15 resolves 147 packages, the dependency gate passes, the runtime
doctor reports `mode: face-aware` / `backend: mediapipe_legacy`, the detector
probe initializes, and a real portrait render produces `global_only: false`,
`face_aware_run: true`, `automatic_pass: true`.

### MediaPipe 0.10.5 universal2 wheel and `pip check`

The `mediapipe==0.10.5` wheel tagged `cp311-cp311-macosx_13_0_x86_64` is a
universal2 build: its native libraries (`_framework_bindings…so`,
`_pywrap_*.so`) are fat `x86_64 arm64` binaries verified with `lipo`. On an
arm64 Mac the WHEEL metadata's `Tag` line advertises only `x86_64`, so
`pip check` reports *"mediapipe 0.10.5 is not supported on this platform"*.

This is a false alarm. The runtime imports cleanly, `FaceMesh` instantiates,
the XNNPACK delegate starts, and the portrait probe returns 478 landmarks. The
runtime doctor propagates the cosmetic failure as `pip_check: error /
pip_check_failed`; treat it as expected on this interpreter and rely on the
detector probe and the face-aware render as the real capability gates.

## Runtime Doctor

Run the doctor before interpreting a face-detection failure. Its default mode
is side-effect-free: it checks declared and installed versions, `pip check`,
duplicate OpenCV distributions, model hashes and controlled URLs, parser and
ONNX provider availability, and the effective Face-aware/Global-only mode.
It never downloads models.

```bash
PYTHONNOUSERSITE=1 RETOUCH_CACHE_DIR=/private/tmp/retouch-runtime-cache \
  python3 -m retouch.runtime_doctor --compact
```

`--probe-detector` is an explicit native initialization check. Run it only in
the isolated legacy environment and a GUI-attached/native session; a native
MediaPipe abort can terminate the subprocess before Python can catch it.

```bash
PYTHONNOUSERSITE=1 RETOUCH_GPU=0 RETOUCH_MEDIAPIPE_BACKEND=legacy \
  "$LEGACY_ENV/bin/python" -s -m retouch.runtime_doctor --probe-detector
```

The report's static provider list is not proof that a model session used that
provider. Active provider claims require a real session or render evidence.

## Legacy recovery contract

The active project declaration is the contract in `pyproject.toml`:

```text
mediapipe==0.10.5
protobuf>=3.20,<4
opencv-contrib-python>=4.8.0,<4.12
numpy>=1.24.0,<2
onnxruntime>=1.16.0
```

For the Python 3.11 resolution recorded in `uv.lock`, the relevant resolved
versions are:

```text
mediapipe              0.10.5
protobuf               3.20.3
opencv-contrib-python  4.11.0.86
numpy                  1.26.4
onnxruntime            1.26.0   # lock's >=3.11 pin (1.20.1 has no cp39 wheel)
gradio                 4.44.1   # GUI/dev extra
```

The lock contains MediaPipe 0.10.5 wheels for CPython 3.9, 3.10, and 3.11.
The 3.9 and 3.10 wheels cannot be installed because their onnxruntime
counterparts (1.20.1 and 1.24.3 respectively) have no cp39/cp310 wheels — see
[Onnxruntime wheel gap](#onnxruntime-wheel-gap-and-the-311-recovery-track)
above. The 3.11 resolution is the installable legacy track and preserves all
four legacy contract pins; onnxruntime moves to 1.26.0 within the declared
`>=1.16.0` range. Python 3.12 still belongs to the migration track below.

### Isolated recovery commands

These commands keep the environment, bytecode, package caches, and model
cache outside the checkout. They also prevent user-site packages—especially a
different TensorFlow/protobuf installation—from contaminating the pinned
runtime.

```bash
cd /Applications/htdocs/retouch

export RETOUCH_RECOVERY_ROOT=/private/tmp/retouch-runtime-recovery
export LEGACY_ENV="$RETOUCH_RECOVERY_ROOT/venv-legacy-py311"
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="$RETOUCH_RECOVERY_ROOT/pycache"
export UV_CACHE_DIR="$RETOUCH_RECOVERY_ROOT/uv-cache"
export XDG_CACHE_HOME="$RETOUCH_RECOVERY_ROOT/xdg-cache"
export MPLCONFIGDIR="$RETOUCH_RECOVERY_ROOT/matplotlib"
export RETOUCH_CACHE_DIR="$RETOUCH_RECOVERY_ROOT/retouch-cache"
mkdir -p "$RETOUCH_RECOVERY_ROOT" "$PYTHONPYCACHEPREFIX" "$UV_CACHE_DIR" \
  "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$RETOUCH_CACHE_DIR"

# CPython 3.11 is the only interpreter on which the legacy lock installs
# (see the onnxruntime wheel gap above). Downloads it only when absent.
uv python install 3.11
uv venv --python 3.11 "$LEGACY_ENV"

# --locked is required: do not re-resolve the legacy project during recovery.
UV_PROJECT_ENVIRONMENT="$LEGACY_ENV" uv sync --locked --extra gui --extra dev

# Use the environment's interpreter explicitly; do not fall through to a
# globally installed `python3` or user-site package set. uv venvs do not
# contain pip, so seed it to make the doctor's pip check meaningful.
uv pip install --python "$LEGACY_ENV/bin/python" pip
PYTHONNOUSERSITE=1 "$LEGACY_ENV/bin/python" -s -m pip check
```

For a desktop packaging environment, use the same commands with
`--extra desktop` (and add `--extra dev` only when the test tools are needed).
The packaging script also prefers `uv sync --locked`; its pip fallback points
to `requirements/desktop.txt`, which is not currently tracked in this
checkout. That fallback is not an alternative recovery source of truth.

### Version and duplicate-package gate

Run this before interpreting any detector failure. It checks the exact
legacy versions and rejects the common duplicate-OpenCV installation that
causes NumPy requirement conflicts.

```bash
PYTHONNOUSERSITE=1 "$LEGACY_ENV/bin/python" -s - <<'PY'
from importlib import metadata
from packaging.version import Version

expected = {
    "mediapipe": "0.10.5",
    "protobuf": "3.20.3",
    "opencv-contrib-python": "4.11.0.86",
    "numpy": "1.26.4",
}
for name, wanted in expected.items():
    actual = metadata.version(name)
    if actual != wanted:
        raise SystemExit(f"{name}: expected {wanted}, got {actual}")

opencv_dists = sorted(
    dist.metadata["Name"].lower()
    for dist in metadata.distributions()
    if (dist.metadata["Name"] or "").lower().startswith("opencv-")
)
if opencv_dists != ["opencv-contrib-python"]:
    raise SystemExit(f"duplicate or unexpected OpenCV distributions: {opencv_dists}")

if not Version(metadata.version("protobuf")) < Version("4"):
    raise SystemExit("protobuf is outside the legacy <4 contract")
if not Version(metadata.version("numpy")) < Version("2"):
    raise SystemExit("NumPy is outside the legacy <2 contract")

# onnxruntime: the lock's 1.20.1 pin (Python <3.10) has no cp39 wheel and is
# not installable; the 3.11 track resolves 1.26.0. Both satisfy the declared
# onnxruntime>=1.16.0, so the gate enforces the declared floor, not a pin.
ort = metadata.version("onnxruntime")
if Version(ort) < Version("1.16.0"):
    raise SystemExit(f"onnxruntime {ort} is below the declared >=1.16.0 floor")

print("legacy dependency gate: pass (onnxruntime", ort + ")")
PY
```

`pip check` in this environment reports *"mediapipe 0.10.5 is not supported on
this platform"* because of the universal2 wheel's x86_64-only WHEEL tag — see
[MediaPipe 0.10.5 universal2 wheel and `pip check`](#mediapipe-0105-universal2-wheel-and-pip-check).
That single line is expected on arm64; the detector probe and face-aware
render below are the authoritative capability gates. `pip check` must report
nothing beyond that known cosmetic line.

### Face-aware and render acceptance

Use a consented real portrait at `PORTRAIT`; the repository does not assume
that a checked-in test image is appropriate for certification.

```bash
export PORTRAIT=/absolute/path/to/consented-portrait.jpg

PYTHONNOUSERSITE=1 RETOUCH_GPU=0 RETOUCH_MEDIAPIPE_BACKEND=legacy \
  "$LEGACY_ENV/bin/python" -s scripts/qa/face_aware_runtime_probe.py "$PORTRAIT"

export LEGACY_EVIDENCE="$RETOUCH_RECOVERY_ROOT/legacy-face-aware-evidence"
rm -rf "$LEGACY_EVIDENCE"
PYTHONNOUSERSITE=1 RETOUCH_GPU=0 RETOUCH_MEDIAPIPE_BACKEND=legacy \
  "$LEGACY_ENV/bin/python" -s scripts/qa/advanced_retouch_visual_qa.py \
  "$PORTRAIT" --output "$LEGACY_EVIDENCE"

PYTHONNOUSERSITE=1 "$LEGACY_ENV/bin/python" -s - <<'PY'
import json
import os
from pathlib import Path

manifest = json.loads(
    (Path(os.environ["LEGACY_EVIDENCE"]) / "manifest.json")
    .read_text(encoding="utf-8")
)
required = {
    "global_only": False,
    "face_aware_run": True,
    "automatic_pass": True,
}
for key, wanted in required.items():
    if manifest.get(key) != wanted:
        raise SystemExit(f"manifest {key}={manifest.get(key)!r}, expected {wanted!r}")
print("face-aware evidence gate: pass")
PY
```

The runtime probe must report `status: "pass"`, `backend: "legacy_cpu"`,
and at least 468 landmarks per detected face. The render must create a
non-empty output/contact sheet and a manifest with `global_only: false`,
`face_aware_run: true`, and `automatic_pass: true`. The runner intentionally
leaves `human_review` separate; these checks are not final visual
certification.

If a model-backed Tasks path is exercised, verify only the models required by
that path and keep the manifest as the authority for availability and hashes:

```bash
PYTHONNOUSERSITE=1 "$LEGACY_ENV/bin/python" -s scripts/qa/verify_models.py \
  --model face_landmarker \
  --model selfie_segmenter
```

Entries in `models/manifest.json` with `availability: "unavailable"` and no
controlled release URL are not clean-install dependencies. They must remain
reported as unavailable until a release artifact and URL are supplied.

## Python 3.12 migration track

`requirements-py312-migration.txt` is a candidate input, not a lock and not a
claim that the current GUI or detector is available on Python 3.12. It moves
the package families forward, uses the MediaPipe Tasks API, and deliberately
does not carry the legacy `protobuf<4` constraint. Its modern protobuf range
is only a candidate; the resolver must still confirm a metadata-compatible
MediaPipe and ONNX Runtime combination.

Create the migration environment outside the project and compile a separate,
hash-bearing lock. The compile step is the availability check; if the index,
platform, or wheel set cannot satisfy the input, stop and record the failure
instead of weakening the constraints or modifying `uv.lock`.

```bash
cd /Applications/htdocs/retouch

export RETOUCH_RECOVERY_ROOT=/private/tmp/retouch-runtime-recovery
export MIGRATION_ENV="$RETOUCH_RECOVERY_ROOT/venv-py312-migration"
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="$RETOUCH_RECOVERY_ROOT/pycache-py312"
export UV_CACHE_DIR="$RETOUCH_RECOVERY_ROOT/uv-cache-py312"
mkdir -p "$RETOUCH_RECOVERY_ROOT" "$PYTHONPYCACHEPREFIX" "$UV_CACHE_DIR"

uv python install 3.12
uv venv --python 3.12 "$MIGRATION_ENV"

export MIGRATION_LOCK="$RETOUCH_RECOVERY_ROOT/requirements-py312-migration.lock.txt"
uv pip compile requirements-py312-migration.txt \
  --python-version 3.12 \
  --python-platform macos \
  --generate-hashes \
  --output-file "$MIGRATION_LOCK"
uv pip sync --python "$MIGRATION_ENV/bin/python" "$MIGRATION_LOCK"

# uv-managed venvs do not include pip; seed it so pip check can run.
uv pip install --python "$MIGRATION_ENV/bin/python" pip
PYTHONNOUSERSITE=1 "$MIGRATION_ENV/bin/python" -s -m pip check
PYTHONNOUSERSITE=1 "$MIGRATION_ENV/bin/python" -s - <<'PY'
import mediapipe as mp
import onnxruntime as ort

assert hasattr(mp, "tasks")
assert hasattr(mp.tasks, "vision")
assert hasattr(mp.tasks.vision, "FaceLandmarker")
print("Tasks API: available")
print("ONNX execution providers:", ort.get_available_providers())
PY
```

This only proves that the candidate packages resolve, import, and expose the
Tasks symbol. It does not prove native macOS construction, model loading,
current `retouch/detection.py` compatibility, GUI behavior, or output
quality. In particular, do not call the current legacy runtime probe a
successful migration result: the current application code contains a
version-specific macOS guard for MediaPipe 0.10.35, and the migration needs a
separate API-port test seam.

Migration acceptance requires all of the following after that port exists:

1. The generated migration lock is checked in as a separate, reviewed
   artifact; `pip check` is clean in a fresh Python 3.12 environment.
2. `mp.tasks.vision.FaceLandmarker` constructs with the verified local
   `face_landmarker.task` model and closes cleanly on the target OS.
3. ONNX providers are printed and the selected provider is verified for each
   model; no provider is assumed merely because it is installed.
4. A real portrait probe and render complete with `global_only: false`, a
   non-empty output/contact sheet, and `face_aware_run: true` plus
   `automatic_pass: true` in the manifest.
5. Human review remains a separate gate before a certification or release
   claim.

## What the current audit does and does not prove

Verified in the repository on the audit date:

- `pyproject.toml` declares the legacy MediaPipe/protobuf/OpenCV/NumPy
  bounds above.
- `uv.lock` resolves the legacy package set above and has no MediaPipe 0.10.5
  CPython 3.12 wheel entry.
- The build wrapper prefers `uv sync --locked --extra desktop`.
- No tracked `requirements*.txt` files existed before this migration input;
  older docs and the build fallback still reference untracked
  `requirements/...` paths.

Verified in an isolated environment on 2026-08-31 (macOS 26.5 arm64,
CPython 3.11.15), using `uv sync --locked` with the lock unmodified:

- The dependency gate passes: `mediapipe 0.10.5`, `protobuf 3.20.3`,
  `opencv-contrib-python 4.11.0.86`, `numpy 1.26.4`, `onnxruntime 1.26.0`,
  and exactly one OpenCV distribution.
- The runtime doctor reports `mode: face-aware` with
  `backend: mediapipe_legacy`; `--probe-detector` returns
  `detector_initialized` with `probe_state: initialized`.
- `scripts/qa/face_aware_runtime_probe.py` on a real portrait returns
  `status: pass`, `backend: legacy_cpu`, and 478 landmarks.
- `scripts/qa/advanced_retouch_visual_qa.py` completes with
  `passed: true` and `global_only: false`, and its manifest carries
  `face_aware_run: true` and `automatic_pass: true` alongside a non-empty
  output directory and contact sheet.
- `pip check` reports only the known MediaPipe universal2 wheel-tag line and
  nothing else.

Not verified by this documentation slice:

- Linux or Windows recovery, or any Intel-x86_64 macOS recovery run.
- The CPython 3.9 and 3.10 lock branches on any platform: they cannot be
  installed while `onnxruntime` 1.20.1/1.24.3 lack cp39/cp310 wheels.
- `FaceDetector.available=True`, a real portrait render, or a
  `global_only: false` manifest in the current contaminated interpreter.
- Native MediaPipe Tasks startup on macOS, Gradio 5 API compatibility, ONNX
  provider correctness, or human visual review. Active ONNX provider use is
  still unproven: the doctor reports `active_provider_verified: false` on the
  static provider list alone.
- Distribution of models whose manifest entries have no controlled URL.
- Reproducibility after a reboot: the recovery root used for verification
  lives under `/private/tmp`, which is not durable storage.

For the migration design, the authoritative references are the [MediaPipe
Face Landmarker Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python),
[MediaPipe package metadata](https://pypi.org/project/mediapipe/), the [Gradio
changelog](https://www.gradio.app/changelog), and the [ONNX Runtime install
matrix](https://onnxruntime.ai/docs/install/).

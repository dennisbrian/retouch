"""Eye-occlusion gate calibration — per-eye signal sweep over the DSCF corpus.

Protocol: docs/plans/RESEARCH_EYE_OCCLUSION_2026_08_26.md (§4 arms, §5 metrics).

For every image in the 83-image DSCF corpus, measures the three gate signals
per eye on both arms:

  Arm A — BiSeNet available (models/resnet18.onnx, handedness-fixed masks)
  Arm B — BiSeNet unavailable (``parser._sess = None``, landmark-fallback
          masks; the hair signal is structurally dead here — §B1)

Signals (imported from retouch.eye_visibility so the sweep measures exactly
what the shipped gate computes):

  1. EAR       — 6-point eyelid aspect ratio (landmark-only; arm-invariant)
  2. contrast  — tone-adaptive iris-vs-sclera ratio vs the eye's own p95 L
  3. hair      — weighted hair-mask overlap over iris and eye

Also records the gate's live per-eye decision (``_should_gate_eye``) at the
current provisional thresholds, and saves a per-eye crop to
``test_output/eye_occlusion_study/crops/`` for human labeling.

Images are pre-shrunk to max-dim 2048 (the detection-study proxy scale; EAR
is a ratio and contrast is relative, so both are scale-stable — noted as a
study condition, not hidden).

    .venv/bin/python scripts/qa/eye_occlusion_sweep.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RETOUCH_GPU", "0")
os.environ.setdefault("RETOUCH_MEDIAPIPE_BACKEND", "legacy")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from retouch.detection import FaceDetector  # noqa: E402
from retouch.parsing import FaceParser  # noqa: E402
from retouch.eye_visibility import (  # noqa: E402
    _EAR_INDICES,
    _contrast_for_side,
    _ear_for_side,
    _should_gate_eye,
    _weighted_overlap,
)

CORPUS = ROOT / "test_output" / "detection_recall_study" / "corpus.txt"
OUT_DIR = ROOT / "test_output" / "eye_occlusion_study"
CROP_DIR = OUT_DIR / "crops"
MAX_DIM = 2048
SIDES = ("left", "right")  # camera-viewer convention (parsing.py LEFT_EYE = 33-cluster)


def shrink(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = MAX_DIM / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def largest_face(faces):
    return max(faces, key=lambda f: f.bbox[2] * f.bbox[3])


def eye_crop(img: np.ndarray, landmarks, side: str) -> np.ndarray | None:
    h, w = img.shape[:2]
    pts = []
    for i in _EAR_INDICES[side]:
        lm = landmarks[i] if not hasattr(landmarks, "landmark") else landmarks.landmark[i]
        pts.append((lm.x * w, lm.y * h))
    pts = np.asarray(pts)
    cx, cy = pts.mean(axis=0)
    span = max(pts[:, 0].ptp(), pts[:, 1].ptp(), 24.0)
    half = int(span * 1.6)
    x1, y1 = max(0, int(cx - half)), max(0, int(cy - half))
    x2, y2 = min(w, int(cx + half)), min(h, int(cy + half))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def measure_arm(img, landmarks, regions):
    h, w = img.shape[:2]
    out = {}
    for side in SIDES:
        eye = getattr(regions, f"{side}_eye", None)
        iris = getattr(regions, f"{side}_iris", None)
        hair = getattr(regions, "hair", None)
        ear = _ear_for_side(landmarks, side, w, h)
        contrast = _contrast_for_side(img, eye, iris)
        hair_iris = hair_eye = None
        if isinstance(hair, np.ndarray) and isinstance(iris, np.ndarray) and hair.shape == iris.shape:
            hair_iris = _weighted_overlap(hair, iris)
            if isinstance(eye, np.ndarray) and eye.shape == hair.shape:
                hair_eye = _weighted_overlap(hair, eye)
        out[side] = {
            "ear": None if ear is None else round(ear, 4),
            "contrast": None if contrast is None else round(contrast, 4),
            "hair_over_iris": None if hair_iris is None else round(hair_iris, 4),
            "hair_over_eye": None if hair_eye is None else round(hair_eye, 4),
            "eye_px": int(np.count_nonzero(np.asarray(eye) > 0.25)) if isinstance(eye, np.ndarray) else 0,
            "iris_px": int(np.count_nonzero(np.asarray(iris) > 0.25)) if isinstance(iris, np.ndarray) else 0,
            "gated": bool(_should_gate_eye(ear, contrast, eye, iris, hair)),
        }
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)

    rels = [ln.strip() for ln in CORPUS.read_text().splitlines() if ln.strip()]
    det = FaceDetector(max_faces=25, min_confidence=0.4)

    parser_a = FaceParser()
    assert parser_a._sess is not None, "Arm A requires BiSeNet; resnet18.onnx missing?"
    parser_b = FaceParser()
    parser_b._sess = None  # force landmark-fallback arm (golden-fixture pattern)

    rows = {}
    t0 = time.time()
    for n, rel in enumerate(rels, 1):
        img = cv2.imread(str(ROOT / rel))
        if img is None:
            rows[rel] = {"error": "unreadable"}
            continue
        img = shrink(img)
        faces = det.detect(img)
        if not faces:
            rows[rel] = {"error": "no_face"}
            print(f"[{n}/{len(rels)}] {rel}: NO FACE", flush=True)
            continue
        fd = largest_face(faces)
        row = {"n_faces": len(faces), "ied": round(fd.ied, 1), "bbox": list(fd.bbox)}
        for arm, parser in (("A", parser_a), ("B", parser_b)):
            regions = parser.parse(fd.landmarks, img, tuple(fd.bbox), ied=fd.ied)
            row[f"arm_{arm}"] = measure_arm(img, fd.landmarks, regions)
        stem = Path(rel).stem
        for side in SIDES:
            crop = eye_crop(img, fd.landmarks, side)
            if crop is not None:
                cv2.imwrite(str(CROP_DIR / f"{stem}_{side}.png"), crop)
        rows[rel] = row
        a = row["arm_A"]
        print(
            f"[{n}/{len(rels)}] {stem}: EAR L={a['left']['ear']} R={a['right']['ear']} "
            f"contrast L={a['left']['contrast']} R={a['right']['contrast']} "
            f"gated L={a['left']['gated']} R={a['right']['gated']}",
            flush=True,
        )

    meta = {
        "date": "2026-08-31",
        "corpus": str(CORPUS.relative_to(ROOT)),
        "max_dim": MAX_DIM,
        "interpreter": sys.executable,
        "person_mask": "not supplied (None); arm-B hair mask is empty by construction",
        "side_convention": "camera-viewer (parsing.py LEFT_EYE = 33-cluster)",
        "elapsed_s": round(time.time() - t0, 1),
    }

    # Per-protocol artifact names (§7.3): one file per signal, arm-keyed.
    def signal_file(name: str, keys: tuple) -> None:
        out = {"meta": meta, "rows": {}}
        for rel, row in rows.items():
            if "error" in row:
                out["rows"][rel] = row
                continue
            out["rows"][rel] = {
                arm: {side: {k: row[f"arm_{arm}"][side][k] for k in keys} for side in SIDES}
                for arm in ("A", "B")
            }
        (OUT_DIR / name).write_text(json.dumps(out, indent=1))

    signal_file("ear_sweep.json", ("ear", "gated"))
    signal_file("contrast_sweep.json", ("contrast", "eye_px", "iris_px", "gated"))
    signal_file("hair_sweep.json", ("hair_over_iris", "hair_over_eye", "gated"))
    (OUT_DIR / "sweep_full.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    print(f"done in {meta['elapsed_s']}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()

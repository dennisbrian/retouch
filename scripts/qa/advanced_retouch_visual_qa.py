#!/usr/bin/env python3
"""Generate real-image QA evidence for the Advanced Retouch workspace.

The default run is face-aware and therefore requires the native MediaPipe
runtime. ``--global-only`` is available only as a headless diagnostic: it
does not exercise semantic masks, face selection, or reshape and is marked
non-certifying in the manifest.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retouch.advanced_retouch import apply_advanced_edit  # noqa: E402
from retouch.batch_processor import generate_contact_sheet  # noqa: E402
from retouch.certification import certification_fields  # noqa: E402
from retouch.io import imread_exif, resize_for_processing  # noqa: E402
from retouch.model_fetch import model_exists  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Portrait image to inspect")
    parser.add_argument("-o", "--output", required=True, help="Evidence directory")
    parser.add_argument("--max-dim", type=int, default=1600)
    parser.add_argument(
        "--global-only",
        action="store_true",
        help="Diagnostic local edits only; explicitly non-certifying",
    )
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _face_brush_mask(shape: Tuple[int, int], faces: Iterable[Any], selection: Optional[int] = None) -> np.ndarray:
    """Create a soft deterministic brush mask around one or all detected faces."""
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for index, face in enumerate(faces):
        if selection is not None and index != selection:
            continue
        x, y, w, h = [int(value) for value in face.bbox]
        center = (x + w // 2, y + h // 2)
        axes = (max(2, int(w * 0.58)), max(2, int(h * 0.62)))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=3).astype(np.float32) / 255.0


def _editor(image_rgb: np.ndarray, mask: np.ndarray) -> Mapping[str, Any]:
    layer = np.zeros((*mask.shape, 4), dtype=np.uint8)
    layer[:, :, 3] = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    return {"background": image_rgb, "layers": [layer]}


def _write(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
        raise IOError(f"Could not write {path}")


def _run_case(
    name: str,
    current_rgb: np.ndarray,
    editor: Mapping[str, Any],
    mode: str,
    operation: str,
    strength: float,
    semantic: str,
    selection: str,
    detector: Any,
    parser: Any,
    output_dir: Path,
    **kwargs: Any,
) -> Dict[str, Any]:
    try:
        result = apply_advanced_edit(
            current_rgb,
            editor,
            mode,
            operation,
            strength,
            semantic,
            selection,
            kwargs.pop("reshape", {}),
            detector,
            parser,
            **kwargs,
        )
        output_path = output_dir / f"{name}.jpg"
        _write(output_path, result.image_rgb)
        return {
            "name": name,
            "status": "rendered",
            "mode": mode,
            "operation": operation,
            "semantic": semantic,
            "selection": selection,
            "output": output_path.name,
            "message": result.status,
        }
    except Exception as exc:  # noqa: BLE001 - preserve evidence for review
        return {
            "name": name,
            "status": "error",
            "mode": mode,
            "operation": operation,
            "semantic": semantic,
            "selection": selection,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    # MediaPipe can abort in native code on headless macOS before Python can
    # catch an exception. Keep the parent process responsible for a truthful
    # blocked manifest so a failed probe never masquerades as missing evidence.
    if not args.global_only and not args._worker:
        worker = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                str(input_path),
                "--output", str(output_dir),
                "--max-dim", str(args.max_dim),
                "--_worker",
            ],
            check=False,
        )
        manifest_path = output_dir / "manifest.json"
        if worker.returncode == 0:
            return 0
        if manifest_path.exists():
            return worker.returncode
        error = f"Face-aware worker exited before writing evidence (exit code {worker.returncode})."
        manifest = {
            "version": 1,
            "input": str(input_path),
            "scale": None,
            "global_only": False,
            "status": "blocked",
            "error": error,
            "face_count": None,
            "cases": [],
        }
        manifest.update(certification_fields(face_aware_run=False, automatic_pass=False, human_review="pending"))
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(output_dir), "status": "blocked", "error": error}))
        return 2

    image_bgr = imread_exif(input_path)
    image_bgr, scale = resize_for_processing(image_bgr, args.max_dim)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    engine = None
    detector = parser = None
    faces: List[Any] = []
    blocked_error = None
    if not args.global_only:
        try:
            from retouch import RetouchEngine

            engine = RetouchEngine()
            detector, parser = engine._detector, engine._parser
            faces = list(detector.detect(image_bgr))
            if not faces:
                unavailable_reason = getattr(detector, "unavailable_reason", None)
                blocked_error = (
                    f"Face-aware detector unavailable: {unavailable_reason}"
                    if unavailable_reason
                    else "No faces detected; face-aware evidence is incomplete."
                )
        except Exception as exc:  # noqa: BLE001 - write a useful blocked report
            blocked_error = f"{type(exc).__name__}: {exc}"

    try:
        if blocked_error:
            manifest = {
                "version": 1,
                "input": str(input_path),
                "scale": scale,
                "global_only": False,
                "status": "blocked",
                "error": blocked_error,
                "face_count": len(faces),
                "cases": [],
            }
            manifest.update(certification_fields(face_aware_run=False, automatic_pass=False, human_review="pending"))
            (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(json.dumps({"output": str(output_dir), "status": "blocked", "error": blocked_error}))
            return 2

        all_faces_mask = _face_brush_mask(image_rgb.shape[:2], faces)
        selected_mask = _face_brush_mask(image_rgb.shape[:2], faces, 0 if faces else None)
        if args.global_only:
            all_faces_mask = np.ones(image_rgb.shape[:2], dtype=np.float32)
            selected_mask = all_faces_mask

        if args.global_only:
            cases = [
                ("warmth_brush", _editor(image_rgb, all_faces_mask), "Adjust", "warmth", 25, "None", "All faces", {}),
                ("smooth_brush", _editor(image_rgb, all_faces_mask), "Adjust", "smooth", 18, "None", "All faces", {}),
                ("heal_brush", _editor(image_rgb, selected_mask), "Heal", "exposure", 0, "None", "All faces", {}),
                ("remove_brush", _editor(image_rgb, selected_mask), "Remove", "exposure", 0, "None", "All faces", {}),
            ]
        else:
            cases = [
                ("warmth_skin_all", _editor(image_rgb, all_faces_mask), "Adjust", "warmth", 25, "None", "All faces", {}),
                ("smooth_skin_all", _editor(image_rgb, all_faces_mask), "Adjust", "smooth", 18, "Skin", "All faces", {}),
                ("clarity_face_0", _editor(image_rgb, selected_mask), "Adjust", "clarity", 16, "Face", "0", {}),
                ("heal_face_0", _editor(image_rgb, selected_mask), "Heal", "exposure", 0, "None", "0", {}),
                ("remove_face_0", _editor(image_rgb, selected_mask), "Remove", "exposure", 0, "None", "0", {}),
                ("reshape_face_0", _editor(image_rgb, selected_mask), "Reshape", "exposure", 0, "None", "0", {
                    "eye_size": 3, "eye_distance": 0, "nose_width": 0, "nose_length": 0,
                    "jaw_width": -4, "chin_length": 0, "mouth_size": 0, "smile": 0, "forehead": 0,
                }),
            ]
        rows = []
        for name, editor, mode, operation, strength, semantic, selection, reshape in cases:
            rows.append(_run_case(
                name, image_rgb, editor, mode, operation, strength, semantic, selection,
                detector, parser, output_dir, heal_method="telea",
                remove_engine="Telea fallback", reshape=reshape,
            ))

        outputs = [output_dir / row["output"] for row in rows if row.get("output")]
        if outputs:
            generate_contact_sheet(outputs, output_dir / "contact_sheet.jpg", cols=3, cell_size=420)
        automatic_pass = bool(outputs) and all(row["status"] == "rendered" for row in rows)
        manifest = {
            "version": 1,
            "input": str(input_path),
            "scale": scale,
            "global_only": bool(args.global_only),
            "status": "complete",
            "face_count": len(faces),
            "optional_models": {
                "lama_inpaint": model_exists("lama_inpaint"),
            },
            "cases": rows,
            "contact_sheet": "contact_sheet.jpg" if outputs else None,
            "passed": automatic_pass,
        }
        manifest.update(certification_fields(
            face_aware_run=not bool(args.global_only),
            automatic_pass=automatic_pass,
            human_review="pending",
        ))
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(output_dir), "passed": manifest["passed"], "global_only": manifest["global_only"]}))
        return 0 if manifest["passed"] else 1
    finally:
        if engine is not None:
            engine.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main(_parser().parse_args()))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)

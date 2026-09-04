#!/usr/bin/env python3
# ruff: noqa: I001, UP045
"""Compare one source image with competitor and Retouch outputs.

The utility is deliberately product-neutral.  It registers every edited image
to the source with seven stable face landmarks, reports full-frame and face
crop change metrics, and writes inspectable contact sheets.  Metrics describe
how an output differs from the source; they are not quality or preference
scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import ExifTags, Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retouch.detection import FaceData, FaceDetector


ANCHOR_INDICES = (33, 133, 263, 362, 1, 61, 291)
LABEL_HEIGHT = 42


def _safe_name(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in value
    )


def _parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("candidate name cannot be empty")
    return name, Path(raw_path).expanduser().resolve()


def _read_image(path: Path, max_dim: int) -> tuple[np.ndarray, dict[str, Any]]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read image: {path}")
    original_height, original_width = image.shape[:2]
    scale = min(1.0, float(max_dim) / max(original_height, original_width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (round(original_width * scale), round(original_height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return image, {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "file_bytes": path.stat().st_size,
        "original_shape": [original_height, original_width],
        "analysis_shape": [int(image.shape[0]), int(image.shape[1])],
        "analysis_scale": scale,
        **_read_metadata(path),
    }


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            named_exif = {
                ExifTags.TAGS.get(tag, str(tag)): value
                for tag, value in exif.items()
                if isinstance(value, (str, int, float))
            }
            return {
                "format": image.format,
                "mode": image.mode,
                "icc_profile_bytes": len(image.info.get("icc_profile", b"")),
                "software": named_exif.get("Software"),
                "datetime": named_exif.get("DateTime"),
                "datetime_original": named_exif.get("DateTimeOriginal"),
            }
    except Exception as exc:  # noqa: BLE001 - metadata is supporting evidence
        return {"metadata_error": f"{type(exc).__name__}: {exc}"}


def _largest_face(detector: FaceDetector, image: np.ndarray) -> tuple[FaceData, int]:
    faces = detector.detect(image)
    if not faces:
        raise ValueError("no face detected")
    face = max(faces, key=lambda item: item.bbox[2] * item.bbox[3])
    return face, len(faces)


def _landmark_points(face: FaceData, width: int, height: int) -> np.ndarray:
    return np.asarray(
        [
            (landmark.x * width, landmark.y * height)
            for landmark in face.landmarks.landmark
        ],
        dtype=np.float32,
    )


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float32)])
    return homogeneous @ matrix.T


def _expanded_face_mask(
    shape: Sequence[int], face: FaceData, padding: float = 0.35
) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    x, y, box_width, box_height = face.bbox
    pad_x = round(box_width * padding)
    pad_y = round(box_height * padding)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(width, x + box_width + pad_x), min(height, y + box_height + pad_y)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def _fit_to_source(
    source_image: np.ndarray,
    source_face: FaceData,
    edited_image: np.ndarray,
    edited_face: FaceData,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source_height, source_width = source_image.shape[:2]
    edited_height, edited_width = edited_image.shape[:2]
    source_points = _landmark_points(source_face, source_width, source_height)
    edited_points = _landmark_points(edited_face, edited_width, edited_height)
    if len(source_points) != len(edited_points):
        raise ValueError(
            f"landmark count mismatch: source={len(source_points)}, edited={len(edited_points)}"
        )

    source_anchors = source_points[list(ANCHOR_INDICES)]
    edited_anchors = edited_points[list(ANCHOR_INDICES)]
    matrix, inliers = cv2.estimateAffinePartial2D(
        edited_anchors,
        source_anchors,
        method=cv2.LMEDS,
    )
    if matrix is None:
        raise ValueError("similarity registration failed")

    warped = cv2.warpAffine(
        edited_image,
        matrix,
        (source_width, source_height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    valid = cv2.warpAffine(
        np.full((edited_height, edited_width), 255, dtype=np.uint8),
        matrix,
        (source_width, source_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    aligned_points = _transform_points(edited_points, matrix)
    residuals = np.linalg.norm(aligned_points - source_points, axis=1)
    source_ied = max(float(source_face.ied), 1.0)
    anchor_residuals = residuals[list(ANCHOR_INDICES)]
    return (
        warped,
        valid,
        {
            "matrix_edited_to_source": matrix.round(8).tolist(),
            "anchor_inliers": int(np.count_nonzero(inliers))
            if inliers is not None
            else None,
            "anchor_count": len(ANCHOR_INDICES),
            "anchor_reprojection_mean_px": float(np.mean(anchor_residuals)),
            "all_landmarks_median_px": float(np.median(residuals)),
            "all_landmarks_p95_px": float(np.percentile(residuals, 95)),
            "all_landmarks_median_ied": float(np.median(residuals) / source_ied),
            "all_landmarks_p95_ied": float(np.percentile(residuals, 95) / source_ied),
            "interpretation": (
                "Residual landmark displacement after a global similarity fit. It can reflect "
                "non-rigid face changes or landmark-detector drift; it is not an identity score."
            ),
        },
    )


def _metric_row(
    source: np.ndarray, edited: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    selected = mask > 0
    if not np.any(selected):
        raise ValueError("metric mask is empty")

    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    edited_lab = cv2.cvtColor(edited, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_delta = edited_lab - source_lab
    lab_mae = np.mean(np.abs(lab_delta[selected]), axis=0)

    difference = edited.astype(np.float32) - source.astype(np.float32)
    mse = float(np.mean(np.square(difference[selected])))
    abs_max = np.max(np.abs(difference), axis=2)
    return {
        "pixel_count": int(np.count_nonzero(selected)),
        "lab8_mae": [float(value) for value in lab_mae],
        "psnr_db": 99.0
        if mse <= 0.0
        else float(20.0 * math.log10(255.0 / math.sqrt(mse))),
        "fraction_pixels_abs_delta_gt8": float(np.mean(abs_max[selected] > 8.0)),
    }


def _resize_long_edge(image: np.ndarray, long_edge: int) -> np.ndarray:
    scale = float(long_edge) / max(image.shape[:2])
    return cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
    )


def _appearance_view(image: np.ndarray, face: Optional[FaceData]) -> np.ndarray:
    if face is None:
        return _resize_long_edge(image, 1024)
    mask = _expanded_face_mask(image.shape, face)
    crop = _face_crop(image, mask)
    # Both sides receive the same canonical resample.  This avoids the
    # numerator/denominator interpolation bias of measuring HF energy after a
    # one-sided registration warp.
    return cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)


def _appearance_summary(image: np.ndarray) -> dict[str, float]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    high = gray - cv2.GaussianBlur(gray, (0, 0), 2.0)
    return {
        "lab8_luma_mean": float(np.mean(lab[:, :, 0])),
        "hsv8_saturation_mean": float(np.mean(hsv[:, :, 1])),
        "hf_std": float(np.std(high)),
    }


def _appearance_change(source: np.ndarray, edited: np.ndarray) -> dict[str, Any]:
    source_summary = _appearance_summary(source)
    edited_summary = _appearance_summary(edited)
    source_hf = source_summary["hf_std"]
    edited_hf = edited_summary["hf_std"]
    hf_ratio = (
        1.0 if max(source_hf, edited_hf) < 1e-6 else edited_hf / max(source_hf, 1e-6)
    )
    return {
        "source": source_summary,
        "edited": edited_summary,
        "lab8_luma_delta": edited_summary["lab8_luma_mean"]
        - source_summary["lab8_luma_mean"],
        "hsv8_saturation_delta": (
            edited_summary["hsv8_saturation_mean"]
            - source_summary["hsv8_saturation_mean"]
        ),
        "hf_std_ratio": hf_ratio,
        "method": (
            "Independent symmetric resampling: 1024px long-edge full views or 512x512 "
            "expanded-face crops. No one-sided registration warp is used for these summaries."
        ),
    }


def _label_panel(image: np.ndarray, label: str) -> np.ndarray:
    panel = cv2.copyMakeBorder(
        image,
        LABEL_HEIGHT,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(25, 25, 25),
    )
    cv2.putText(
        panel,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return panel


def _fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
    )
    canvas = np.full((height, width, 3), 22, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _face_crop(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return image
    return image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def _write_visuals(
    output_dir: Path,
    source: np.ndarray,
    face_mask: np.ndarray,
    aligned: Mapping[str, np.ndarray],
) -> dict[str, str]:
    names = ["source", *aligned.keys()]
    images = [source, *aligned.values()]
    full_panels = [
        _label_panel(_fit_panel(image, 420, 630), name)
        for name, image in zip(names, images)
    ]
    full_sheet = np.hstack(full_panels)
    full_path = output_dir / "full_comparison.jpg"
    if not cv2.imwrite(str(full_path), full_sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise OSError(f"failed to write {full_path}")

    source_crop = _face_crop(source, face_mask)
    detail_rows: list[np.ndarray] = []
    for name, image in aligned.items():
        edited_crop = _face_crop(image, face_mask)
        difference = cv2.absdiff(source_crop, edited_crop)
        difference = np.clip(difference.astype(np.float32) * 4.0, 0, 255).astype(
            np.uint8
        )
        row = np.hstack(
            [
                _label_panel(_fit_panel(source_crop, 420, 420), "source face"),
                _label_panel(_fit_panel(edited_crop, 420, 420), f"{name} face"),
                _label_panel(_fit_panel(difference, 420, 420), f"{name} abs diff x4"),
            ]
        )
        detail_rows.append(row)
    detail_sheet = np.vstack(detail_rows)
    detail_path = output_dir / "face_detail_comparison.jpg"
    if not cv2.imwrite(str(detail_path), detail_sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise OSError(f"failed to write {detail_path}")
    return {
        "full_comparison": str(full_path),
        "face_detail_comparison": str(detail_path),
    }


def _write_blinded_visuals(
    output_dir: Path,
    source: np.ndarray,
    face_mask: np.ndarray,
    aligned: Mapping[str, np.ndarray],
    seed: str,
    source_hash: str,
) -> tuple[dict[str, str], dict[str, str]]:
    order = list(aligned.items())
    random.Random(f"{seed}:{source_hash}").shuffle(order)
    labelled = [
        (chr(ord("A") + index), name, image)
        for index, (name, image) in enumerate(order)
    ]

    full_panels = [_label_panel(_fit_panel(source, 420, 630), "reference source")]
    full_panels.extend(
        _label_panel(_fit_panel(image, 420, 630), label) for label, _, image in labelled
    )
    full_path = output_dir / "blinded_full_review.jpg"
    if not cv2.imwrite(
        str(full_path), np.hstack(full_panels), [cv2.IMWRITE_JPEG_QUALITY, 94]
    ):
        raise OSError(f"failed to write {full_path}")

    source_crop = _face_crop(source, face_mask)
    face_panels = [_label_panel(_fit_panel(source_crop, 420, 420), "reference source")]
    face_panels.extend(
        _label_panel(_fit_panel(_face_crop(image, face_mask), 420, 420), label)
        for label, _, image in labelled
    )
    face_path = output_dir / "blinded_face_review.jpg"
    if not cv2.imwrite(
        str(face_path), np.hstack(face_panels), [cv2.IMWRITE_JPEG_QUALITY, 94]
    ):
        raise OSError(f"failed to write {face_path}")

    key = {label: name for label, name, _ in labelled}
    key_path = output_dir / "blinded_key.json"
    key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")
    return {
        "blinded_full_review": str(full_path),
        "blinded_face_review": str(face_path),
        "blinded_key": str(key_path),
    }, key


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("competitor", type=Path)
    parser.add_argument("--competitor-name", default="competitor")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        type=_parse_candidate,
        metavar="NAME=PATH",
        help="Additional output to compare; repeat for multiple Retouch recipes.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-dim", type=int, default=1600)
    parser.add_argument(
        "--blind-seed",
        help="Also write randomized labelled full/face review sheets plus a separate key.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    source_path = args.source.expanduser().resolve()
    competitor_path = args.competitor.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    named_paths = [(args.competitor_name, competitor_path), *args.candidate]
    if len({name for name, _ in named_paths}) != len(named_paths):
        raise ValueError("comparison names must be unique")
    for path in [source_path, *(path for _, path in named_paths)]:
        if not path.is_file():
            raise FileNotFoundError(path)

    source, source_record = _read_image(source_path, args.max_dim)
    records: list[dict[str, Any]] = []
    aligned_images: dict[str, np.ndarray] = {}

    with FaceDetector(allow_unavailable=False) as detector:
        source_face, source_face_count = _largest_face(detector, source)
        face_mask = _expanded_face_mask(source.shape, source_face)
        source_full_view = _appearance_view(source, None)
        source_face_view = _appearance_view(source, source_face)
        for name, path in named_paths:
            edited, metadata = _read_image(path, args.max_dim)
            edited_face, edited_face_count = _largest_face(detector, edited)
            aligned, valid, registration = _fit_to_source(
                source,
                source_face,
                edited,
                edited_face,
            )
            full_mask = np.where(valid > 0, 255, 0).astype(np.uint8)
            registered_face_mask = cv2.bitwise_and(face_mask, full_mask)
            aligned_images[name] = aligned
            records.append(
                {
                    "name": name,
                    "metadata": metadata,
                    "detected_face_count": edited_face_count,
                    "selected_face_bbox": list(edited_face.bbox),
                    "registration": registration,
                    "appearance": {
                        "full": _appearance_change(
                            source_full_view, _appearance_view(edited, None)
                        ),
                        "face": _appearance_change(
                            source_face_view, _appearance_view(edited, edited_face)
                        ),
                    },
                    "metrics": {
                        "full": _metric_row(source, aligned, full_mask),
                        "face": _metric_row(source, aligned, registered_face_mask),
                    },
                }
            )

    visuals = _write_visuals(output_dir, source, face_mask, aligned_images)
    blinded_key = None
    if args.blind_seed:
        blinded_visuals, blinded_key = _write_blinded_visuals(
            output_dir,
            source,
            face_mask,
            aligned_images,
            args.blind_seed,
            source_record["sha256"],
        )
        visuals.update(blinded_visuals)
    report = {
        "schema_version": 2,
        "metric_boundary": (
            "Metrics measure source-to-output change after face-landmark similarity registration. "
            "Appearance summaries use symmetric canonical resampling to avoid one-sided HF bias. "
            "Neither family measures beauty, quality, fidelity, or human preference."
        ),
        "source": {
            **source_record,
            "detected_face_count": source_face_count,
            "selected_face_bbox": list(source_face.bbox),
            "selected_face_ied": float(source_face.ied),
        },
        "anchor_indices": list(ANCHOR_INDICES),
        "comparisons": records,
        "visuals": visuals,
    }
    if blinded_key is not None:
        label_text = "/".join(blinded_key)
        report["blinded_review"] = {
            "seed": args.blind_seed,
            "labels": list(blinded_key),
            "key_file": visuals["blinded_key"],
            "instructions": (
                f"Review without opening the key. Rank {label_text} separately for intended "
                "correction, identity, texture, feature realism, and overall preference."
            ),
        }
    report_path = output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "done",
                "output": str(output_dir),
                "metrics": str(report_path),
                "visuals": visuals,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

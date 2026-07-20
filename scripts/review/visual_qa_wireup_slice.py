#!/usr/bin/env python3
"""Render inspectable proof panels for the AA wire-up slice.

This is a review-only harness. It avoids ``RetouchEngine`` because local
MediaPipe/Metal startup can abort before face processing. The X2 skin mask is
therefore a visible Haar-face approximation, not a production segmentation
mask. Eye/brow/lip exclusion holes are pose-aware: they anchor on detected or
explicitly passed eye centres (``--eye-centers``) and rotate with the
inter-eye axis, because upright box-fraction holes miss on tilted poses and
let the smoother paint over an iris. PatchMatch is demonstrated on an
injected compact cheek defect so the repair has known ground truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from retouch.chromophore_v2 import (  # noqa: E402
    decompose_chromophores_v2,
    reduce_hemoglobin_variance,
    shift_hemoglobin,
)
from retouch.film import FilmDensityEngine  # noqa: E402
from retouch.frequency import FrequencySeparator  # noqa: E402
from retouch.heal import heal_region  # noqa: E402


def _resize(img: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def _face_box(img: np.ndarray) -> tuple[int, int, int, int]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        raise RuntimeError("Haar QA face approximation found no face")
    return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))


def _detect_eyes(
    img: np.ndarray, box: tuple[int, int, int, int]
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Detect the two eye centres inside the face box, or None if ambiguous."""
    x, y, w, h = box
    gray = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    side = max(14, w // 14)
    found = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(side, side))
    centres = [(x + ex + ew // 2, y + ey + eh // 2) for ex, ey, ew, eh in found]
    pairs = [
        (centres[i], centres[j])
        for i in range(len(centres))
        for j in range(i + 1, len(centres))
        if 0.25 * w <= np.hypot(centres[i][0] - centres[j][0], centres[i][1] - centres[j][1]) <= 0.75 * w
    ]
    return pairs[0] if len(pairs) == 1 else None


def _skin_qa_mask(
    shape: tuple[int, int],
    box: tuple[int, int, int, int],
    eye_centers: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> np.ndarray:
    """Return a conservative, visibly-reviewable central-face approximation.

    With ``eye_centers`` the eye/brow exclusion holes are rotated ellipses
    placed on the actual eyes, sized from the inter-eye distance, so they hold
    on tilted / high-angle poses. Without them the legacy upright box-fraction
    holes are used — those are only valid on upright frontal faces.
    """
    x, y, w, h = box
    mask = np.zeros(shape, dtype=np.float32)
    center = (x + w // 2, y + int(h * 0.63))
    mouth = (x + w // 2, y + int(h * 0.79))
    cv2.ellipse(mask, center, (int(w * 0.31), int(h * 0.29)), 0, 0, 360, 1.0, -1)
    # Do not apply a skin-colour effect across the eyes, brows or lips.
    if eye_centers is not None:
        (x1e, y1e), (x2e, y2e) = eye_centers
        d = float(np.hypot(x2e - x1e, y2e - y1e))
        angle = float(np.degrees(np.arctan2(y2e - y1e, x2e - x1e)))
        # Face-up = perpendicular to the inter-eye axis, pointing away from the mouth.
        axis = np.array([x2e - x1e, y2e - y1e], dtype=np.float32) / max(d, 1.0)
        up = np.array([axis[1], -axis[0]], dtype=np.float32)
        mid_to_up = np.array([(x1e + x2e) / 2.0 - mouth[0], (y1e + y2e) / 2.0 - mouth[1]], np.float32)
        if float(np.dot(up, mid_to_up)) < 0.0:
            up = -up
        for ex, ey in ((x1e, y1e), (x2e, y2e)):
            # Eye + lid/lash coverage, aligned with the head tilt.
            cv2.ellipse(mask, (int(ex), int(ey)), (int(d * 0.40), int(d * 0.26)), angle, 0, 360, 0.0, -1)
            # Brow band above the eye along the same axis.
            bx, by = int(ex + up[0] * d * 0.30), int(ey + up[1] * d * 0.30)
            cv2.ellipse(mask, (bx, by), (int(d * 0.42), int(d * 0.16)), angle, 0, 360, 0.0, -1)
        # Lip hole at the pose-estimated mouth (eye midpoint + 1.1·d face-down).
        mx = int((x1e + x2e) / 2.0 - up[0] * d * 1.1)
        my = int((y1e + y2e) / 2.0 - up[1] * d * 1.1)
        cv2.ellipse(mask, (mx, my), (int(d * 0.45), int(d * 0.22)), angle, 0, 360, 0.0, -1)
    else:
        for cx in (x + int(w * 0.34), x + int(w * 0.66)):
            cv2.ellipse(mask, (cx, y + int(h * 0.57)), (int(w * 0.12), int(h * 0.05)), 0, 0, 360, 0.0, -1)
        cv2.ellipse(mask, mouth, (int(w * 0.17), int(h * 0.07)), 0, 0, 360, 0.0, -1)
    return cv2.GaussianBlur(mask, (0, 0), 2.5)


def _face_crop(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    x0, y0 = max(0, x - w // 6), max(0, y - h // 10)
    x1, y1 = min(img.shape[1], x + w + w // 6), min(img.shape[0], y + h + h // 5)
    return img[y0:y1, x0:x1]


def _title(img: np.ndarray, label: str) -> np.ndarray:
    bar = np.full((38, img.shape[1], 3), (24, 24, 24), dtype=np.uint8)
    cv2.putText(bar, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
    return np.vstack((bar, img))


def _stack(title: str, panels: list[tuple[str, np.ndarray]]) -> np.ndarray:
    height = min(panel.shape[0] for _, panel in panels)
    resized = [
        cv2.resize(panel, (round(panel.shape[1] * height / panel.shape[0]), height), interpolation=cv2.INTER_AREA)
        for _, panel in panels
    ]
    sep = np.full((height, 4, 3), 180, dtype=np.uint8)
    row = resized[0]
    for panel in resized[1:]:
        row = np.hstack((row, sep, panel))
    return _title(row, title)


def _film_params(purity: float) -> dict[str, float | bool]:
    return {
        "enable": True,
        "strength": 1.0,
        "toe_r": 0.0, "toe_g": 0.0, "toe_b": 0.0,
        "shoulder_r": 0.0, "shoulder_g": 0.0, "shoulder_b": 0.0,
        "midpoint": 0.5, "gamma": 1.0,
        "crosstalk_cy_mg": 0.0, "crosstalk_cy_ye": 0.0, "crosstalk_mg_ye": 0.0,
        "tonemap_strength": 0.0, "tonemap_toe": 0.1, "tonemap_shoulder": 0.15,
        "skew": 0.0, "highlight_purity": purity,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", default=ROOT / "test_output" / "DSCF8007_baseline.jpg")
    parser.add_argument("--out", type=Path, default=ROOT / "test_output" / "visual_qa_wireup_2026_07_17")
    parser.add_argument("--max-dim", type=int, default=1200)
    parser.add_argument(
        "--face-box",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        help="Verified QA face box after resizing; bypasses Haar detection.",
    )
    parser.add_argument(
        "--eye-centers",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Verified eye centres after resizing; bypasses Haar eye detection "
        "and anchors the pose-aware eye/brow exclusion holes.",
    )
    args = parser.parse_args()

    image = cv2.imread(str(args.source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read {args.source}")
    image = _resize(image, args.max_dim)
    box = tuple(args.face_box) if args.face_box else _face_box(image)
    x, y, w, h = box
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > image.shape[1] or y + h > image.shape[0]:
        raise ValueError(f"face box {box} is outside resized image {image.shape[1]}x{image.shape[0]}")
    if args.eye_centers:
        e = args.eye_centers
        eye_centers = ((e[0], e[1]), (e[2], e[3]))
    else:
        eye_centers = _detect_eyes(image, box)
    if eye_centers is None:
        print(
            "WARNING: no verified eye centres (auto-detection ambiguous). Falling back to "
            "upright box-fraction eye holes — INVALID on tilted/high-angle poses; the "
            "smoothing panels must not be used as eye-safety evidence."
        )
    skin_mask = _skin_qa_mask(image.shape[:2], box, eye_centers)
    args.out.mkdir(parents=True, exist_ok=True)

    # X4: same film parameters, only highlight purity differs.
    film = FilmDensityEngine()
    x4_base = film.apply(image, _film_params(0.0))
    x4_on = film.apply(image, _film_params(0.85))
    x4_diff = np.clip(np.abs(x4_on.astype(np.float32) - x4_base.astype(np.float32)) * 8.0, 0, 255).astype(np.uint8)
    x4_panel = _stack(
        "X4 HIGHLIGHT PURITY: zero -> 0.85 (right: 8x absolute difference)",
        [("BASE", x4_base), ("PURITY 0.85", x4_on), ("DIFF", x4_diff)],
    )
    cv2.imwrite(str(args.out / "x4_highlight_purity.jpg"), x4_panel, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # X2: direct v2 operations using a conservative QA face mask.
    dec = decompose_chromophores_v2(image.astype(np.float32), skin_mask=skin_mask)
    x2_even = reduce_hemoglobin_variance(image.astype(np.float32), 0.75, skin_mask=skin_mask, decomposition=dec)
    dec_even = decompose_chromophores_v2(x2_even, skin_mask=skin_mask, axes_rgb=dec.axes_rgb)
    x2_on = shift_hemoglobin(x2_even, -0.35, skin_mask=skin_mask, decomposition=dec_even)
    x2_u8 = np.clip(x2_on, 0, 255).astype(np.uint8)
    mask_view = cv2.applyColorMap(np.clip(skin_mask * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    mask_overlay = cv2.addWeighted(image, 0.68, mask_view, 0.32, 0.0)
    x2_diff = np.clip(np.abs(x2_u8.astype(np.float32) - image.astype(np.float32)) * 7.0, 0, 255).astype(np.uint8)
    x2_panel = _stack(
        "X2 HB EVEN 0.75 + HB SHIFT -0.35 (QA mask is shown; not production segmentation)",
        [("BEFORE", _face_crop(image, box)), ("X2 ON", _face_crop(x2_u8, box)), ("MASK", _face_crop(mask_overlay, box)), ("DIFF", _face_crop(x2_diff, box))],
    )
    cv2.imwrite(str(args.out / "x2_hemoglobin_v2.jpg"), x2_panel, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Frequency smoothing: this is the production separator/combine path. The
    # only QA substitute is the conservative Haar-derived face mask above.
    separator = FrequencySeparator()
    layers = separator.separate(image, face_width=float(box[2]))
    smoothed = separator.combine(
        layers,
        skin_mask=skin_mask,
        smooth_strength=0.58,
        mid_reduction=0.38,
        texture_opacity=0.86,
        face_width=float(box[2]),
        smooth_engine="anisotropic",
    )
    showcase_smoothed = separator.combine(
        layers,
        skin_mask=skin_mask,
        smooth_strength=0.82,
        mid_reduction=0.58,
        texture_opacity=0.78,
        face_width=float(box[2]),
        smooth_engine="anisotropic",
    )
    demo_smoothed = separator.combine(
        layers,
        skin_mask=skin_mask,
        smooth_strength=0.95,
        mid_reduction=0.80,
        texture_opacity=0.45,
        face_width=float(box[2]),
        smooth_engine="anisotropic",
    )
    # Self-check: the excluded eyes must be (near-)unchanged in every render.
    # Reported per eye so a mask that misses one eye cannot pass silently.
    eye_deltas: dict[str, list[float]] = {}
    if eye_centers is not None:
        r = max(8, int(np.hypot(eye_centers[1][0] - eye_centers[0][0],
                                eye_centers[1][1] - eye_centers[0][1]) * 0.15))
        for label, render in (("client", smoothed), ("showcase", showcase_smoothed), ("demo", demo_smoothed)):
            deltas = []
            for ex, ey in eye_centers:
                patch = np.abs(
                    render[ey - r:ey + r, ex - r:ex + r].astype(np.float32)
                    - image[ey - r:ey + r, ex - r:ex + r].astype(np.float32)
                )
                deltas.append(float(patch.mean()))
            eye_deltas[label] = deltas
            status = "OK" if max(deltas) < 1.5 else "EYE LEAK — mask does not protect the eyes"
            print(f"eye-delta [{label}]: {deltas[0]:.2f} / {deltas[1]:.2f}  ({status})")

    smooth_diff = np.clip(
        np.abs(showcase_smoothed.astype(np.float32) - image.astype(np.float32)) * 5.0,
        0,
        255,
    ).astype(np.uint8)
    mask_note = (
        "pose-aware eye/brow/lip holes" if eye_centers is not None
        else "LEGACY upright holes — eye safety UNVERIFIED"
    )
    smoothing_panel = _stack(
        f"FREQUENCY SMOOTHING: client 0.58/0.38, showcase 0.82/0.58 (QA mask; {mask_note})",
        [("BEFORE", _face_crop(image, box)), ("CLIENT", _face_crop(smoothed, box)), ("SHOWCASE", _face_crop(showcase_smoothed, box)), ("DIFF", _face_crop(smooth_diff, box))],
    )
    cv2.imwrite(str(args.out / "frequency_smoothing.jpg"), smoothing_panel, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(args.out / "frequency_smoothing_preview.jpg"), smoothed, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(args.out / "frequency_smoothing_showcase_preview.jpg"), showcase_smoothed, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(args.out / "frequency_smoothing_demo_preview.jpg"), demo_smoothed, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(
        str(args.out / "frequency_smoothing_compare.jpg"),
        np.hstack((image, smoothed)),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    cv2.imwrite(
        str(args.out / "frequency_smoothing_showcase_compare.jpg"),
        np.hstack((image, showcase_smoothed)),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    cv2.imwrite(
        str(args.out / "frequency_smoothing_demo_compare.jpg"),
        np.hstack((image, demo_smoothed)),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    # PatchMatch: inject a compact synthetic defect inside a real cheek, then
    # repair it from masked skin exemplars. The exact hole mask isolates the
    # reconstruction quality from detector scaling in this preview harness.
    injected = image.copy()
    x, y, w, h = box
    defect = (x + int(w * 0.65), y + int(h * 0.66))
    defect_radius = max(5, w // 33)
    cv2.circle(injected, defect, defect_radius, (25, 25, 25), -1)
    repair_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(repair_mask, defect, defect_radius, 255, -1)
    restored = heal_region(
        injected,
        repair_mask,
        method="patchmatch",
        source_mask=skin_mask,
        patch_size=7,
        iterations=5,
        seamless=False,
    )
    d = defect_radius
    patch_region = np.s_[max(0, defect[1] - d):defect[1] + d + 1, max(0, defect[0] - d):defect[0] + d + 1]
    patch_injected_error = float(np.mean(np.abs(injected[patch_region].astype(np.float32) - image[patch_region].astype(np.float32))))
    patch_restored_error = float(np.mean(np.abs(restored[patch_region].astype(np.float32) - image[patch_region].astype(np.float32))))
    patch_diff = np.clip(np.abs(restored.astype(np.float32) - injected.astype(np.float32)) * 5.0, 0, 255).astype(np.uint8)
    patch_panel = _stack(
        "PATCHMATCH HEAL: real cheek with injected compact defect (known-ground-truth QA mask)",
        [("ORIGINAL", _face_crop(image, box)), ("INJECTED", _face_crop(injected, box)), ("PATCHMATCH", _face_crop(restored, box)), ("DIFF", _face_crop(patch_diff, box))],
    )
    cv2.imwrite(str(args.out / "patchmatch_auto_heal.jpg"), patch_panel, [cv2.IMWRITE_JPEG_QUALITY, 95])

    contact_width = max(x4_panel.shape[1], x2_panel.shape[1], smoothing_panel.shape[1], patch_panel.shape[1])
    contact_panels = [
        cv2.resize(panel, (contact_width, round(panel.shape[0] * contact_width / panel.shape[1])), interpolation=cv2.INTER_AREA)
        for panel in (x4_panel, x2_panel, smoothing_panel, patch_panel)
    ]
    contact = np.vstack(contact_panels)
    cv2.imwrite(str(args.out / "contact_sheet.jpg"), contact, [cv2.IMWRITE_JPEG_QUALITY, 95])

    mel_after = decompose_chromophores_v2(x2_on, skin_mask=skin_mask, axes_rgb=dec.axes_rgb).melanin
    active = skin_mask > 0.5
    report = (
        "# Wire-up visual QA\n\n"
        f"Source: `{args.source}`\n\n"
        "- X4 is a real full-frame film-engine A/B: only `highlight_purity` changes.\n"
        "- X2 uses a visible Haar-derived QA mask because this local environment cannot reliably start MediaPipe.\n"
        "- Frequency smoothing uses the production anisotropic separator/combine path with the QA mask "
        f"({'pose-aware eye/brow/lip exclusion holes anchored on verified eye centres' if eye_centers is not None else 'LEGACY upright holes — eye safety UNVERIFIED on this pose'}).\n"
        "- PatchMatch starts from a known injected compact defect and known QA mask on a real cheek.\n"
        + "".join(
            f"- Frequency smoothing `{label}` per-eye mean |delta| (must stay near zero): "
            f"`{d[0]:.2f}` / `{d[1]:.2f}`.\n"
            for label, d in eye_deltas.items()
        )
        + f"- X2 masked max melanin-coordinate delta: `{float(np.max(np.abs(mel_after[active] - dec.melanin[active]))):.6f}`.\n"
        f"- X4 mean absolute BGR delta: `{float(np.mean(np.abs(x4_on.astype(np.float32) - x4_base.astype(np.float32)))):.4f}`.\n"
        f"- PatchMatch local MAE vs the known clean cheek: injected `{patch_injected_error:.3f}` -> restored `{patch_restored_error:.3f}`.\n"
    )
    (args.out / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Wrote visual QA outputs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

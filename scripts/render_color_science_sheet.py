#!/usr/bin/env python3
"""Regenerate the color-science (K3/K8/K9) review renders + combined sheet.

Saved so the sheet is reproducible (the original was ad-hoc). Uses a single
shared portrait source and a synthetic rainbow ramp for the gamut panels.

Usage:
    python3 scripts/render_color_science_sheet.py [SOURCE.jpg] [OUT_DIR]
"""
import os
import sys

import cv2
import numpy as np

from retouch.color_science import (
    apply_subtractive_saturation,
    bgr_to_oklab,
    oklab_to_bgr,
    oklab_to_oklch,
    oklch_to_oklab,
    find_gamut_intersection,
)


def _label(img, text, org=(14, 34)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, 0.7,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, 0.7,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _rainbow(h, w):
    hue = (np.linspace(0, 179, w, dtype=np.float32)[None, :]
           .repeat(h, axis=0)).astype(np.uint8)
    hsv = np.stack([hue, np.full_like(hue, 255), np.full_like(hue, 255)], -1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _additive(img_bgr, amount):
    """HSV additive saturation, mirrors grading._F_adjust_saturation."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[:, :, 1]
    hsv[:, :, 1] = np.clip(s * (1.0 + amount * (1.0 - s / 255.0)), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _naive_sat_clip(img_bgr, amount):
    """Naive HSV saturation with hard clip (produces hue shift/posterize)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * (1.0 + amount)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _gamut_compress_bgr(img_bgr, amount):
    """Boost saturation then roll out-of-gamut chroma back to the boundary."""
    boosted = _naive_sat_clip(img_bgr, amount)
    lab = bgr_to_oklab(boosted)
    c_max = find_gamut_intersection(lab)
    oklch = oklab_to_oklch(lab)
    oklch[..., 1] = np.minimum(oklch[..., 1], np.maximum(c_max, 1e-4))
    return oklab_to_bgr(oklch_to_oklab(oklch))


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/Desktop/event/DSCF6102.jpg")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else \
        "test_output/color_science_2026-07-11"
    os.makedirs(out_dir, exist_ok=True)

    photo = cv2.imread(src)
    if photo is None:
        raise SystemExit(f"could not read source: {src}")

    # The sheet is a fixed 2-column grid. Every panel is resized to (CELL_W, *)
    # so rows stack without ragged black gaps.
    CELL_W = 1080

    def fit_w(img, w=CELL_W):
        h = int(round(img.shape[0] * w / img.shape[1]))
        return cv2.resize(img, (w, h))

    # ---- Row 1 (K3): gamut compression on a saturation-boosted ramp ----
    ramp_wide = _rainbow(360, CELL_W)
    a = _naive_sat_clip(ramp_wide, 0.8)
    b = _gamut_compress_bgr(ramp_wide, 0.8)
    _label(a, "A: saturation-boosted (naive clip -> hue shift/posterize)")
    _label(b, "B: +gamut_compress (hue-stable, no posterize)")
    row1 = np.hstack([a, b])
    cv2.imwrite(os.path.join(out_dir, "K3_gamut_compress.png"), row1)

    # ---- Row 2 (K9): original / additive / subtractive on the ramp ----
    third = CELL_W * 2 // 3
    ramp3 = _rainbow(300, third)
    add = _additive(ramp3, 0.6)
    sub = apply_subtractive_saturation(ramp3, 0.6)
    _label(ramp3, "Original")
    _label(add, "Additive saturation (L ~flat)")
    _label(sub, "Subtractive saturation (film: darkens chroma)")
    row2 = np.hstack([ramp3, add, sub])
    cv2.imwrite(os.path.join(out_dir, "K9_subtractive_vs_additive.png"),
                np.hstack([ramp3, add, sub]))

    # ---- Row 3 (photo): original vs subtractive saturation ----
    photo_fit = fit_w(photo)
    photo_sub = apply_subtractive_saturation(photo_fit, 0.6)
    row3 = np.hstack([photo_fit, photo_sub])

    # K8 color-drift demo: additive vs subtractive on the real photo
    cv2.imwrite(os.path.join(out_dir, "K8_color_drift_demo.png"),
                np.hstack([_additive(photo_fit, 0.6), photo_sub]))

    # ---- combined REVIEW SHEET (uniform 2*CELL_W width) ----
    W = CELL_W * 2

    def fit_row(x):
        h = int(round(x.shape[0] * W / x.shape[1]))
        return cv2.resize(x, (W, h))

    sheet = np.vstack([fit_row(row1), fit_row(row2), fit_row(row3)])
    cv2.imwrite(os.path.join(out_dir, "color_science_REVIEW_SHEET.png"), sheet)
    print("wrote sheet + panels to", out_dir)


if __name__ == "__main__":
    main()

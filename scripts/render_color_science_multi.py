#!/usr/bin/env python3
"""Render the K3/K8/K9 color-science demos across MANY source photos.

Extends the single-source `render_color_science_sheet.py` so the user can
review the effects on diverse subjects (skin tones, costumes, colored venue
light). Writes per-image sheets + one combined contact sheet into
`test_output/color_science_2026-07-11/multi/`.

Usage:
    python3 scripts/render_color_science_multi.py
"""
import os

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
from retouch.qa_detectors import detect_color_drift


def _label(img, text, org=(14, 34), scale=0.7):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, scale,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, scale,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _additive(img_bgr, amount):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[:, :, 1]
    hsv[:, :, 1] = np.clip(s * (1.0 + amount * (1.0 - s / 255.0)), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _naive_sat_clip(img_bgr, amount):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * (1.0 + amount)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def _gamut_compress_bgr(img_bgr, amount):
    boosted = _naive_sat_clip(img_bgr, amount)
    lab = bgr_to_oklab(boosted)
    c_max = find_gamut_intersection(lab)
    oklch = oklab_to_oklch(lab)
    oklch[..., 1] = np.minimum(oklch[..., 1], np.maximum(c_max, 1e-4))
    return oklab_to_bgr(oklch_to_oklab(oklch))


def _hue_shift_oklch(img_bgr, deg):
    """Aggressive grade: rotate OKLCh hue by `deg` (for the K8 drift demo)."""
    lab = bgr_to_oklab(img_bgr)
    oklch = oklab_to_oklch(lab)
    oklch[..., 2] = (oklch[..., 2] + deg) % 360.0
    return oklab_to_bgr(oklch_to_oklab(oklch))


def fit_w(img, w):
    h = int(round(img.shape[0] * w / img.shape[1]))
    return cv2.resize(img, (w, h))


def build_sheet(src_path, out_dir, cell_w=900):
    photo = cv2.imread(src_path)
    if photo is None:
        return None
    stem = os.path.splitext(os.path.basename(src_path))[0]
    p = fit_w(photo, cell_w)

    # K9: original | additive | subtractive
    add = _additive(p, 0.6)
    sub = apply_subtractive_saturation(p, 0.6)
    _label(p, "Original")
    _label(add, "Additive +0.6 (L~flat)")
    _label(sub, "Subtractive +0.6 (film: darkens chroma)")
    row_k9 = np.hstack([p, add, sub])

    # K3: naive sat-boost (clips/hue-shift) | +gamut_compress
    boost = _naive_sat_clip(p, 0.9)
    comp = _gamut_compress_bgr(p, 0.9)
    _label(boost, "Sat-boost naive (clip -> hue shift)")
    _label(comp, "Same boost + K3 gamut_compress (hue-stable)")
    row_k3 = np.hstack([boost, comp])

    # K8: original | aggressive hue-shift; overlay drift detector result
    shifted = _hue_shift_oklch(p, 45)
    drift = detect_color_drift(shifted, None, p)
    _label(p, "Original")
    _label(shifted, f"Hue-shift +45deg  Δh={drift['deltaH_mean_deg']:.1f}°  "
                    f"flag={'YES' if drift['flagged'] else 'no'}", org=(14, 34))
    row_k8 = np.hstack([p, shifted])

    # stack (each row already cell_w*N wide; normalize to common width)
    W = row_k9.shape[1]

    def fit_row(x):
        h = int(round(x.shape[0] * W / x.shape[1]))
        return cv2.resize(x, (W, h))

    sheet = np.vstack([fit_row(row_k9), fit_row(row_k3), fit_row(row_k8)])
    out = os.path.join(out_dir, f"{stem}_sheet.png")
    cv2.imwrite(out, sheet)
    return out


def main():
    root = "test_output"
    srcs = [
        "test_output/DSCF4454.jpg",
        "test_output/DSCF4503.jpg",
        "test_output/DSCF4550.jpg",
        "test_output/DSCF8007.jpg",
        "test_output/DSCF8007_con_mixed_temp_v1.jpg",
        "test_output/DSCF8007_con_fluorescent_v1.jpg",
    ]
    srcs = [s for s in srcs if os.path.exists(s)]
    out_dir = "test_output/color_science_2026-07-11/multi"
    os.makedirs(out_dir, exist_ok=True)

    sheets = []
    for s in srcs:
        out = build_sheet(s, out_dir)
        if out:
            sheets.append(out)
            print("wrote", out)

    # Combined contact sheet (resize each to a common width, stack)
    if sheets:
        W = 1600
        cols = [fit_w(cv2.imread(s), W) for s in sheets]
        contact = np.vstack(cols)
        cv2.imwrite(os.path.join(out_dir, "contact_sheet.png"), contact)
        print("wrote", os.path.join(out_dir, "contact_sheet.png"))


if __name__ == "__main__":
    main()

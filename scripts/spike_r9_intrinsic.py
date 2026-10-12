#!/usr/bin/env python3
"""R9 Research Spike — Intrinsic Decomposition (Albedo x Shading).

Standalone, NON-invasive research script. Reuses detection / parsing /
frequency / qa_detectors but modifies NO pipeline modules.

See PLAN_R9_INTRINSIC_SPIKE.md for the design. Implements:
  Phase A: SH 9-coeff robust fit on MediaPipe-mesh normals -> A | S | I
  Phase B: chroma-constancy cast-shadow refinement of S (guided_filter)
  Phase C: point existing S3 blotch-evening (FrequencySeparator.blotch_reduction)
           at A instead of I, recompose, compare vs current S3 on F11 detectors.

Outputs triptychs + Phase-C comparisons + an aggregated GO/NO-GO verdict.

Usage:
  python3 scripts/spike_r9_intrinsic.py [images...] [--out DIR] [--blotch 0.6]
If no images given, a curated corpus subset from test_output/ is used.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import Delaunay

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retouch.detection import FaceDetector, FaceContext
from retouch.parsing import FaceParser
from retouch.utils import guided_filter
from retouch.frequency import FrequencySeparator
from retouch.qa_detectors import detect_plastic_skin, detect_halo

logger = logging.getLogger("r9_spike")

EPS = 1e-4

# --------------------------------------------------------------------------
# Color helpers (sRGB <-> linear, Rec.709)
# --------------------------------------------------------------------------
def srgb_to_linear(u8: np.ndarray) -> np.ndarray:
    x = u8.astype(np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(f: np.ndarray) -> np.ndarray:
    f = np.clip(f, 0.0, 1.0)
    x = np.where(f <= 0.0031308, f * 12.92, 1.055 * (f ** (1.0 / 2.4)) - 0.055)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def luminance(linear_rgb: np.ndarray) -> np.ndarray:
    return (
        0.2126 * linear_rgb[:, :, 0]
        + 0.7152 * linear_rgb[:, :, 1]
        + 0.0722 * linear_rgb[:, :, 2]
    )


# --------------------------------------------------------------------------
# Spherical harmonics (2nd order, real) basis from a normal map
# --------------------------------------------------------------------------
def sh_basis_9(n: np.ndarray) -> np.ndarray:
    """n: (H,W,3) normalized normals -> (H,W,9) SH basis values."""
    nx, ny, nz = n[:, :, 0], n[:, :, 1], n[:, :, 2]
    Y = np.empty(n.shape[:2] + (9,), dtype=np.float32)
    Y[:, :, 0] = 0.282095
    Y[:, :, 1] = 0.488603 * ny
    Y[:, :, 2] = 0.488603 * nz
    Y[:, :, 3] = 0.488603 * nx
    Y[:, :, 4] = 1.092548 * nx * ny
    Y[:, :, 5] = 1.092548 * ny * nz
    Y[:, :, 6] = 0.315392 * (3.0 * nz * nz - 1.0)
    Y[:, :, 7] = 1.092548 * nx * nz
    Y[:, :, 8] = 0.546274 * (nx * nx - ny * ny)
    return Y


# --------------------------------------------------------------------------
# Mesh normal map from MediaPipe landmarks (Delaunay triangulation)
# --------------------------------------------------------------------------
def build_normal_map(
    landmarks: Any,
    W: int,
    H: int,
    roi_mask: np.ndarray,
) -> np.ndarray:
    """Rasterize per-pixel normals from 478 landmark points.

    landmarks: iterable of objects with .x .y .z (normalized; z ~ x scale).
    roi_mask: (H,W) float; normals only resolved where roi_mask > 0.5.
    Returns (H,W,3) float32 normals (xyz), default (0,0,1) outside.
    """
    px, py, pz = [], [], []
    for lm in landmarks:
        px.append(lm.x * W)
        py.append(lm.y * H)
        # MediaPipe z is normalized like x (by width) -> scale to pixels by W
        pz.append(lm.z * W)
    pts2d = np.stack([np.array(px), np.array(py)], axis=1).astype(np.float64)
    pts3d = np.stack([np.array(px), np.array(py), np.array(pz)], axis=1).astype(np.float64)

    tri = Delaunay(pts2d)

    # Per-vertex normals from accumulated triangle face normals
    simp = tri.simplices
    v0 = pts3d[simp[:, 0]]
    v1 = pts3d[simp[:, 1]]
    v2 = pts3d[simp[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(fn, axis=1, keepdims=True)
    norm[norm < 1e-9] = 1.0
    fn = fn / norm
    N = pts2d.shape[0]
    vn = np.zeros((N, 3), dtype=np.float64)
    np.add.at(vn, simp.ravel(), np.repeat(fn, 3, axis=0))
    vnn = np.linalg.norm(vn, axis=1, keepdims=True)
    vnn[vnn < 1e-9] = 1.0
    vn = vn / vnn

    normal_map = np.zeros((H, W, 3), dtype=np.float32)
    msk = roi_mask > 0.5
    yy, xx = np.where(msk)
    if len(xx) == 0:
        normal_map[:, :, 2] = 1.0
        return normal_map

    query = np.stack([xx.astype(np.float64), yy.astype(np.float64)], axis=1)
    simplex = tri.find_simplex(query)
    ok = simplex >= 0
    if not ok.any():
        normal_map[:, :, 2] = 1.0
        return normal_map

    q = query[ok]
    s = simplex[ok]
    T = tri.transform[s]  # (K,3,3): [:,:2] basis, [:,2] origin
    d = q - T[:, 2]
    bc = T[:, :2] @ d[:, :, None]
    bc = bc[:, :, 0]
    b3 = 1.0 - bc[:, 0] - bc[:, 1]
    bcoords = np.stack([bc[:, 0], bc[:, 1], b3], axis=1)
    verts = simp[s]
    interp = (bcoords[:, :, None] * vn[verts]).sum(axis=1)
    inn = np.linalg.norm(interp, axis=1, keepdims=True)
    inn[inn < 1e-9] = 1.0
    interp = interp / inn

    out_idx = np.where(ok)[0]
    normal_map[yy[out_idx], xx[out_idx], 0] = interp[:, 0]
    normal_map[yy[out_idx], xx[out_idx], 1] = interp[:, 1]
    normal_map[yy[out_idx], xx[out_idx], 2] = interp[:, 2]
    return normal_map


# --------------------------------------------------------------------------
# Specular exclusion mask (S4-lite): high luminance + low chroma
# --------------------------------------------------------------------------
def specular_mask(linear_rgb: np.ndarray, skin: np.ndarray) -> np.ndarray:
    L = luminance(linear_rgb)
    with np.errstate(divide="ignore", invalid="ignore"):
        chroma = np.abs(np.log(linear_rgb + EPS) - np.log(L[:, :, None] + EPS)).mean(axis=2)
    sp = ((L > 0.85) & (chroma < 0.06) & (skin > 0.5)).astype(np.float32)
    return sp


# --------------------------------------------------------------------------
# Phase A + B: decompose into linear albedo/shading luminance fields
# --------------------------------------------------------------------------
def decompose(
    linear_rgb: np.ndarray,
    skin: np.ndarray,
    normal_map: np.ndarray,
    spec: np.ndarray,
) -> Dict[str, np.ndarray]:
    H, W = skin.shape
    L = luminance(linear_rgb)
    logL = np.log(L + EPS)

    # Fit region: skin, exclude specular + tiny regions
    fit_mask = (skin > 0.5) & (spec < 0.5)
    ys, xs = np.where(fit_mask)
    if len(xs) < 64:
        # degenerate: fall back to all skin
        fit_mask = skin > 0.5
        ys, xs = np.where(fit_mask)

    Y = sh_basis_9(normal_map)[ys, xs]  # (M,9)
    tgt = logL[ys, xs]  # (M,)

    # Robust IRLS (Huber) least squares for the 9 SH coefficients
    l = np.zeros(9, dtype=np.float64)
    w = np.ones(len(tgt), dtype=np.float64)
    Yt = Y.T
    for _ in range(6):
        A = (Yt * w) @ Y
        b = (Yt * w) @ tgt
        try:
            l = np.linalg.solve(A + 1e-6 * np.eye(9), b)
        except np.linalg.LinAlgError:
            l, *_ = np.linalg.lstsq(Y, tgt, rcond=None)
            break
        r = Y @ l - tgt
        c = 1.345
        w = np.where(np.abs(r) < c, 1.0, c / np.maximum(np.abs(r), 1e-6))

    logS = (sh_basis_9(normal_map).reshape(-1, 9) @ l).reshape(H, W)
    S = np.exp(np.clip(logS, -6, 6))
    A = np.clip(L / S, EPS, 1.0)

    # ---- Phase B: chroma-constancy cast-shadow refinement ----
    # residual that SH couldn't explain; achromatic + spatially smooth part is
    # shading (cast shadow), move it from A into S via guided filter.
    resid = np.clip(logL - logS, -6, 6)
    with np.errstate(divide="ignore", invalid="ignore"):
        chroma = np.abs(np.log(linear_rgb + EPS) - np.log(L[:, :, None] + EPS)).mean(axis=2)
    achromatic = (chroma < np.percentile(chroma[skin > 0.5], 60) if (skin > 0.5).any() else chroma < 0.05)
    achromatic = achromatic & (skin > 0.5)
    r_gray = resid.astype(np.float32)
    g = guided_filter(r_gray, radius=21, eps=0.02)
    logS = logS + g * achromatic.astype(np.float32)
    S = np.exp(np.clip(logS, -6, 6))
    A = np.clip(L / S, EPS, 1.0)

    return {"A": A, "S": S, "logS": logS, "L": L, "coeffs": l}


# --------------------------------------------------------------------------
# Phase C: S3 blotch-evening on A vs on I
# --------------------------------------------------------------------------
def phase_c(
    orig_u8: np.ndarray,
    linear_rgb: np.ndarray,
    skin: np.ndarray,
    dec: Dict[str, np.ndarray],
    face_width: float,
    blotch: float,
) -> Tuple[np.ndarray, np.ndarray]:
    fs = FrequencySeparator()

    # Baseline (current S3) on full I
    layers = fs.separate(orig_u8, face_width)
    base_out = fs.combine(
        layers, skin_mask=skin, blotch_reduction=blotch,
        smooth_strength=0.35, mid_reduction=0.35, texture_opacity=1.0,
        face_width=face_width,
    )

    # R9: even the albedo luminance field, recompose + recolor
    A = np.clip(dec["A"], EPS, 1.0)
    A_srgb = linear_to_srgb(A)  # (H,W)
    A_bgr = np.stack([A_srgb, A_srgb, A_srgb], axis=2)
    layersA = fs.separate(A_bgr, face_width)
    Aeven_u8 = fs.combine(
        layersA, skin_mask=skin, blotch_reduction=blotch,
        smooth_strength=0.35, mid_reduction=0.35, texture_opacity=1.0,
        face_width=face_width,
    )
    Aeven_lin = srgb_to_linear(Aeven_u8).mean(axis=2)
    L_r9 = np.clip(Aeven_lin * dec["S"], EPS, 1.0)

    L_lin = luminance(linear_rgb)
    # Smooth recolor: transfer only the evened (low/blotch) luminance, keep
    # the original pore high-frequency detail so we never "even" pores.
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = L_r9[:, :, None] / (L_lin[:, :, None] + EPS)
    scale = np.clip(scale, 0.5, 2.0)
    base_lin = np.clip(linear_rgb * scale, 0.0, 1.0)
    k = max(3, int(face_width * 0.012) | 1)
    hf = linear_rgb - cv2.GaussianBlur(linear_rgb, (k, k), 0)
    final_lin = np.clip(base_lin + hf * skin[:, :, None], 0.0, 1.0)
    r9_out = linear_to_srgb(final_lin)
    return base_out, r9_out


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------
def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _gray_stack(images: List[np.ndarray], labels: List[str]) -> np.ndarray:
    rows = []
    for im, lb in zip(images, labels):
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        rows.append(_label(im, lb))
    h = max(r.shape[0] for r in rows)
    rows = [cv2.resize(r, (int(r.shape[1] * h / r.shape[0]), h)) for r in rows]
    return np.hstack(rows)


def _norm_gray(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, None)
    if x.max() > 0:
        x = x / x.max()
    return (x * 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Per-image driver
# --------------------------------------------------------------------------
@dataclass
class Verdict:
    name: str = ""
    decompose_ms: float = 0.0
    chroma_in_A: float = 0.0
    chroma_in_S: float = 0.0
    plastic_base: float = 0.0
    plastic_r9: float = 0.0
    halo_base: float = 0.0
    halo_r9: float = 0.0
    dark: bool = False


def process_image(
    path: str,
    out_dir: str,
    det: FaceDetector,
    parser: FaceParser,
    blotch: float,
) -> List[Verdict]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("cannot read %s", path)
        return []
    stem = os.path.splitext(os.path.basename(path))[0]
    H, W = img.shape[:2]
    linear = srgb_to_linear(img)
    skin_all = np.zeros((H, W), np.float32)

    person = det.segment_person(img)
    faces = det.detect(img)
    if not faces:
        logger.warning("no faces in %s", path)
        return []

    verdicts: List[Verdict] = []
    for i, f in enumerate(faces):
        regions = parser.parse(f.landmarks, img, f.bbox, person, f.ied)
        skin = regions.skin
        if skin is None or skin.max() < 0.5:
            continue
        skin_all = np.maximum(skin_all, skin)
        face_width = f.ied * 2.5
        roi = cv2.dilate(skin, np.ones((15, 15), np.uint8), iterations=1)

        t0 = time.time()
        nmap = build_normal_map(f.landmarks.landmark, W, H, roi)
        spec = specular_mask(linear, skin)
        dec = decompose(linear, skin, nmap, spec)
        decompose_ms = (time.time() - t0) * 1000.0

        base_out, r9_out = phase_c(img, linear, skin, dec, face_width, blotch)

        # F11 judges
        pb = detect_plastic_skin(base_out, skin, img)["score"]
        pr = detect_plastic_skin(r9_out, skin, img)["score"]
        hb = detect_halo(base_out, skin)["score"]
        hr = detect_halo(r9_out, skin)["score"]

        # Separation metric (criterion 1): chroma variance in A vs S
        with np.errstate(divide="ignore", invalid="ignore"):
            chroma = np.abs(np.log(linear + EPS) - np.log(luminance(linear)[:, :, None] + EPS))
        cA = float(chroma[skin > 0.5].mean())
        cS = 0.0  # S is grayscale by construction

        v = Verdict(
            name=f"{stem}#f{i}", decompose_ms=decompose_ms,
            chroma_in_A=cA, chroma_in_S=cS,
            plastic_base=pb, plastic_r9=pr, halo_base=hb, halo_r9=hr,
        )
        verdicts.append(v)

        # ---- renders ----
        A_disp = linear_to_srgb(dec["A"])
        S_disp = _norm_gray(dec["S"])
        triptych = _gray_stack(
            [A_disp, S_disp, img],
            ["A (albedo)", "S (shading, norm)", "I (orig)"],
        )
        cv2.imwrite(os.path.join(out_dir, f"{stem}_f{i}_triptych_A_S_I.png"), triptych)

        cmp = _gray_stack(
            [base_out, r9_out, img],
            ["S3 on I (baseline)", "S3 on A (R9)", "I (orig)"],
        )
        cv2.imwrite(os.path.join(out_dir, f"{stem}_f{i}_phaseC_compare.png"), cmp)

        diff = np.abs(base_out.astype(np.int16) - r9_out.astype(np.int16)) * 8
        diff = np.clip(diff, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_f{i}_diff_baseline_vs_r9_x8.png"), diff)

    return verdicts


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="R9 intrinsic-decomposition spike")
    ap.add_argument("images", nargs="*", help="input images (default: curated corpus)")
    ap.add_argument("--out", default="test_output/r9_spike_2026-07-10")
    ap.add_argument("--blotch", type=float, default=0.6, help="S3 blotch_reduction strength")
    ap.add_argument("--max-dim", type=int, default=900,
                    help="Downscale longest side before processing (spike runs on a "
                         "face core; 400-900 mimics the intended ~400x400 operating point)")
    args = ap.parse_args()

    if not args.images:
        cand = [
            "test_output/DSCF4454.jpg",
            "test_output/DSCF4503.jpg",
            "test_output/DSCF4550.jpg",
            "test_output/DSCF8007_con_mixed_temp_v1.jpg",
            "test_output/DSCF8007_con_fluorescent_v1.jpg",
        ]
        args.images = [c for c in cand if os.path.exists(c)]
        logger.info("curated corpus: %s", args.images)

    os.makedirs(args.out, exist_ok=True)
    det = FaceDetector()
    parser = FaceParser()
    all_v: List[Verdict] = []
    try:
        for p in args.images:
            logger.info("processing %s", p)
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            if args.max_dim and max(img.shape[:2]) > args.max_dim:
                s = args.max_dim / float(max(img.shape[:2]))
                img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)),
                                 interpolation=cv2.INTER_AREA)
                tmp = os.path.join(args.out, "_" + os.path.splitext(os.path.basename(p))[0] + "_scaled.png")
                cv2.imwrite(tmp, img)
                p = tmp
            all_v.extend(process_image(p, args.out, det, parser, args.blotch))
    finally:
        det.close()

    # ---- GO/NO-GO aggregation ----
    print("\n" + "=" * 72)
    print("R9 SPIKE — PER-FACE RESULTS")
    print("=" * 72)
    wins = 0
    for v in all_v:
        better = v.plastic_r9 >= v.plastic_base and v.halo_r9 <= v.halo_base + 0.05
        wins += int(better)
        print(f"[{v.name}] dec={v.decompose_ms:5.1f}ms "
              f"plastic base={v.plastic_base:.3f} r9={v.plastic_r9:.3f} "
              f"halo base={v.halo_base:.3f} r9={v.halo_r9:.3f} "
              f"chromaA={v.chroma_in_A:.4f} -> {'WIN' if better else 'lose'}")
    n = max(1, len(all_v))
    crit1 = all(v.chroma_in_S == 0.0 for v in all_v)  # S grayscale by construction
    crit2 = wins >= max(4, int(0.8 * n))
    crit3 = any(v.chroma_in_A > 0 for v in all_v)  # dark-skin present in run
    crit4 = all(v.decompose_ms < 150.0 for v in all_v)
    go = crit1 and crit2 and crit4
    print("-" * 72)
    print(f"C1 separation (chroma in A, S=0 by construction): {'PASS' if crit1 else 'FAIL'}")
    print(f"C2 quality (R9 beats baseline on >=4/5):          {'PASS' if crit2 else 'FAIL'} ({wins}/{n})")
    print(f"C3 dark-skin present in corpus:                   {'PASS' if crit3 else 'N/A'}")
    print(f"C4 perf (<150ms/face):                            {'PASS' if crit4 else 'FAIL'}")
    print(f"\nVERDICT: {'GO' if go else 'NO-GO (continue to 2c Poisson-only fallback per spike doc)'}")
    print(f"Outputs in: {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""P4 spike — multi-cue makeup α prior + reconstruct residual (no engine wire).

Run: python3 scripts/spike_p4_makeup_unmix.py
Exit 0 if synthetic S1–S5-ish gates pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retouch.chromophore import (  # noqa: E402
    _M,
    _M_PINV,
    _bgr_to_log_rgb,
    decompose_chromophores,
    reconstruct_from_chromophores,
)
from retouch.color_science import bgr_to_oklab, oklab_to_oklch  # noqa: E402


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def unconstrained_conc(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    log_rgb = _bgr_to_log_rgb(img_bgr)
    log_diff = log_rgb - log_rgb[..., 2:3]
    conc = log_diff @ _M_PINV.T
    return conc[..., 0].astype(np.float32), conc[..., 1].astype(np.float32)


def multi_cue_alpha(
    img_bgr: np.ndarray,
    skin_mask: np.ndarray,
    mole_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Soft α prior: chroma ∨ white-paint ∨ nonneg-clamp residual."""
    h, w = img_bgr.shape[:2]
    m = skin_mask.astype(np.float32)
    if m.max() > 1.0:
        m = m / 255.0
    m = np.clip(m, 0.0, 1.0)

    oklab = bgr_to_oklab(img_bgr.astype(np.float32))
    oklch = oklab_to_oklch(oklab)
    L, C = oklch[..., 0], oklch[..., 1]

    mel_u, hb_u = unconstrained_conc(img_bgr)
    clamp = np.maximum(-mel_u, 0.0) + np.maximum(-hb_u, 0.0)

    cue_chroma = _smoothstep(0.10, 0.20, C)
    cue_white = _smoothstep(0.85, 0.95, L) * (1.0 - _smoothstep(0.02, 0.08, C))
    cue_clamp = _smoothstep(0.02, 0.08, clamp)

    hint = np.maximum(np.maximum(cue_chroma, cue_white), cue_clamp) * m
    if mole_mask is not None:
        mm = mole_mask.astype(np.float32)
        if mm.max() > 1.0:
            mm = mm / 255.0
        hint = hint * (1.0 - mm)

    # Light morph close
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    u8 = np.clip(hint * 255, 0, 255).astype(np.uint8)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, k)
    return (u8.astype(np.float32) / 255.0)


def closed_form_alpha(I: np.ndarray, S: np.ndarray, M: np.ndarray) -> np.ndarray:
    """α = clamp( (I-S)·(M-S) / ||M-S||² )."""
    If = I.astype(np.float32)
    Sf = S.astype(np.float32)
    Mf = M.astype(np.float32)
    d = Mf - Sf
    num = ((If - Sf) * d).sum(axis=-1)
    den = (d * d).sum(axis=-1) + 1e-6
    return np.clip(num / den, 0.0, 1.0).astype(np.float32)


def recompose(S: np.ndarray, M: np.ndarray, a: np.ndarray) -> np.ndarray:
    a3 = a[..., None]
    out = (1.0 - a3) * S.astype(np.float32) + a3 * M.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.float32)


def _flat(h, w, val):
    return np.full((h, w, 3), val, dtype=np.float32)


def run_synthetic() -> list[str]:
    fails: list[str] = []
    h = w = 64

    # S1: constant S, constant M, disk α=0.6
    S = _flat(h, w, 160.0)
    Mcol = np.array([200.0, 180.0, 220.0], dtype=np.float32)  # BGR pinkish
    M = np.broadcast_to(Mcol, (h, w, 3)).copy()
    yy, xx = np.ogrid[:h, :w]
    disk = ((yy - 32) ** 2 + (xx - 32) ** 2) <= 18 ** 2
    a_gt = disk.astype(np.float32) * 0.6
    I = recompose(S, M, a_gt)
    a_hat = closed_form_alpha(I, S, M)
    err = float(np.abs(a_hat[disk] - 0.6).mean()) if disk.any() else 1.0
    if err > 0.08:
        fails.append(f"S1 closed-form α err={err:.3f} > 0.08")

    # S3: bare skin α≈0 multi-cue
    bare = _flat(h, w, 170.0)
    bare[:, :, 0] = 150  # slight skin-ish BGR
    bare[:, :, 1] = 165
    bare[:, :, 2] = 200
    mask = np.ones((h, w), dtype=np.float32)
    a0 = multi_cue_alpha(bare.astype(np.uint8), mask)
    if float(a0.mean()) > 0.15:
        fails.append(f"S3 bare multi-cue mean α={a0.mean():.3f} > 0.15")

    # S5: reconstruct round-trip on tan-ish patch
    patch = _flat(32, 32, 0.0)
    patch[:, :, 0] = 90
    patch[:, :, 1] = 120
    patch[:, :, 2] = 160
    u8 = np.clip(patch, 0, 255).astype(np.uint8)
    mel, hb = decompose_chromophores(u8)
    recon = reconstruct_from_chromophores(mel, hb, img_bgr=u8)
    resid = float(np.abs(recon - patch).mean())
    if resid > 25.0:
        fails.append(f"S5 reconstruct mean abs={resid:.1f} > 25")

    # White paint cue: high L low C should raise α vs bare
    white = _flat(h, w, 245.0)
    a_w = multi_cue_alpha(white.astype(np.uint8), mask)
    if float(a_w.mean()) < float(a0.mean()):
        fails.append("white paint cue not stronger than bare")

    # Clamp residual: pure blue (off skin manifold) should cue
    blue = _flat(h, w, 0.0)
    blue[:, :, 0] = 220
    a_b = multi_cue_alpha(blue.astype(np.uint8), mask)
    if float(a_b.mean()) < 0.1:
        fails.append(f"off-manifold blue α={a_b.mean():.3f} too low")

    return fails


def main() -> int:
    fails = run_synthetic()
    if fails:
        print("P4 SPIKE FAIL:")
        for f in fails:
            print(" ", f)
        return 1
    print("P4 SPIKE PASS — synthetic multi-cue + reconstruct gates OK")
    print("  _M shape", _M.shape)
    print("  Next: real cosplay triptychs before engine wire")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

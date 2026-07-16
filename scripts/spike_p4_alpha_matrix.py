#!/usr/bin/env python3
"""P4 alpha-solve failure-mode matrix — evidence for PLAN_P4_MAKEUP_UNMIX.md §19.

Audits the CURRENT (post-2026-07-16 remediation) solver in
retouch/makeup_unmix.py across:

  tones        : Fitzpatrick I-VI representative diffuse swatches
  makeup types : foundation lighter / foundation darker / blush /
                 concealer / white cosplay paint / blue cosplay paint
  alpha fields : uniform (0.6 disk) / gradient (0->0.9 ramp in disk) /
                 patchy (soft blobs, ~0.7 peak)

Known ground truth: composites are built as I = (1-a)S + aM (the solver's
own forward model) plus low-freq skin mottling and sigma=3 sensor noise, so
measured error is attributable to the inverse solve, not model mismatch.
A separate shading variant and an ambiguity sweep (||M-S|| -> 0) are run at
the end.

Outputs:
  - metric table on stdout (seeded, deterministic)
  - visual panels test_output/spike_p4_alpha_matrix_<type>.png
    (rows = tones; cols = input | a_true | a_rec | S_rec | S_true)

Run: python3 scripts/spike_p4_alpha_matrix.py
Exit 0 always (reporting tool).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retouch.makeup_unmix import (  # noqa: E402
    apply_makeup_coverage_even,
    recompose,
    unmix_makeup,
)

H = W = 128
SEED = 42

TONES = {
    "I": (189, 208, 244),
    "II": (143, 180, 231),
    "III": (109, 152, 208),
    "IV": (87, 114, 165),
    "V": (62, 82, 122),
    "VI": (38, 51, 80),
}


def makeup_color(kind: str, s_bgr: np.ndarray) -> np.ndarray:
    """Makeup color M as a function of the base skin tone (BGR float)."""
    s = s_bgr.astype(np.float32)
    if kind == "fnd_light":
        return np.clip(s * 1.18 + np.array([0, 8, 12], np.float32), 0, 255)
    if kind == "fnd_dark":
        return np.clip(s * 0.80 + np.array([0, -2, 2], np.float32), 0, 255)
    if kind == "blush":
        return np.clip(0.55 * s + 0.45 * np.array([90, 90, 225], np.float32), 0, 255)
    if kind == "concealer":
        c = np.clip(s * 1.12, 0, 255)
        gray = float(c.mean())
        return np.clip(0.7 * c + 0.3 * gray, 0, 255)
    if kind == "paint_white":
        return np.array([245, 245, 245], np.float32)
    if kind == "paint_blue":
        return np.array([200, 90, 40], np.float32)
    raise ValueError(kind)


def base_skin(rng: np.random.RandomState, bgr: tuple) -> np.ndarray:
    """Flat tone + gentle low-frequency mottling (float32, no sensor noise)."""
    base = np.zeros((H, W, 3), np.float32) + np.array(bgr, np.float32)
    mottle = rng.normal(0, 1, (H // 8, W // 8)).astype(np.float32)
    mottle = cv2.resize(mottle, (W, H), interpolation=cv2.INTER_CUBIC)
    mottle = cv2.GaussianBlur(mottle, (0, 0), 3.0) * 6.0
    # Multiplicative mottling: same relative variation on every tone.
    return np.clip(base * (1.0 + mottle[..., None] * 0.01), 0, 255)


def alpha_field(kind: str, rng: np.random.RandomState) -> np.ndarray:
    yy, xx = np.ogrid[:H, :W]
    disk = (((yy - H // 2) ** 2 + (xx - W // 2) ** 2) <= 44 ** 2).astype(np.float32)
    if kind == "uniform":
        return disk * 0.6
    if kind == "gradient":
        ramp = np.broadcast_to(np.linspace(0.0, 0.9, W, dtype=np.float32), (H, W))
        return disk * ramp
    if kind == "patchy":
        field = rng.normal(0, 1, (H // 6, W // 6)).astype(np.float32)
        field = cv2.resize(field, (W, H), interpolation=cv2.INTER_CUBIC)
        field = cv2.GaussianBlur(field, (0, 0), 4.0)
        field = np.clip((field - field.mean()) / (field.std() + 1e-6), -3, 3)
        soft = np.clip(field * 0.5 + 0.5, 0, 1)  # ~half the disk covered
        return disk * soft * 0.7
    raise ValueError(kind)


def composite(S: np.ndarray, mcol: np.ndarray, a: np.ndarray,
              rng: np.random.RandomState, shade: bool = False) -> np.ndarray:
    a3 = a[..., None]
    img = (1.0 - a3) * S + a3 * mcol
    if shade:
        ramp = np.linspace(0.82, 1.12, H, dtype=np.float32)[:, None, None]
        img = img * ramp
    img = img + rng.normal(0, 3, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def to_bgr_u8(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = cv2.applyColorMap(
            np.clip(x * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO
        )
    return np.clip(x, 0, 255).astype(np.uint8)


def evaluate(img: np.ndarray, S_true: np.ndarray, a_true: np.ndarray) -> dict:
    mask = np.ones((H, W), np.float32)
    I = img.astype(np.float32)
    S_rec, M_rec, a_rec = unmix_makeup(img, mask)
    sup = a_true > 0.3
    bare = a_true < 0.05
    degenerate = bool(np.array_equal(S_rec, I))
    recon = recompose(S_rec, M_rec, a_rec)
    out = apply_makeup_coverage_even(img, mask, 0.5)
    dmg_bare = float(np.abs(out.astype(np.float32) - I)[bare].max()) if bare.any() else 0.0
    r = {
        "detect_in": float(a_rec[sup].mean()) if sup.any() else float("nan"),
        "false_out": float(a_rec[bare].mean()) if bare.any() else float("nan"),
        "iou": 0.0,
        "a_mae": float(np.abs(a_rec - a_true)[sup].mean()) if sup.any() else float("nan"),
        "s_mae": float(np.abs(S_rec - S_true)[sup].mean()) if sup.any() else float("nan"),
        "recon": float(np.abs(recon - I)[sup].mean()) if sup.any() else float("nan"),
        "dmg_bare": dmg_bare,
        "degen": degenerate,
        "a_rec": a_rec,
        "S_rec": S_rec,
    }
    pred = a_rec > 0.3
    union = (pred | sup).sum()
    r["iou"] = float((pred & sup).sum() / union) if union else 1.0
    return r


def main() -> int:
    out_dir = ROOT / "test_output"
    out_dir.mkdir(exist_ok=True)
    kinds = ["fnd_light", "fnd_dark", "blush", "concealer", "paint_white", "paint_blue"]
    patterns = ["uniform", "gradient", "patchy"]

    hdr = (f"{'type':<12} {'pat':<9} {'tone':<4} {'det_in':>6} {'fp_out':>6} "
           f"{'IoU':>5} {'aMAE':>5} {'S_MAE':>6} {'recon':>6} {'dmg':>5} {'degen':>5}")
    print(hdr)
    print("-" * len(hdr))

    panels: dict = {k: [] for k in kinds}
    for kind in kinds:
        for pat in patterns:
            for tone, bgr in TONES.items():
                rng = np.random.RandomState(SEED)
                S = base_skin(rng, bgr)
                mcol = makeup_color(kind, np.array(bgr, np.float32))
                a = alpha_field(pat, rng)
                img = composite(S, mcol, a, rng)
                r = evaluate(img, S, a)
                print(f"{kind:<12} {pat:<9} {tone:<4} {r['detect_in']:6.3f} "
                      f"{r['false_out']:6.3f} {r['iou']:5.2f} {r['a_mae']:5.2f} "
                      f"{r['s_mae']:6.1f} {r['recon']:6.1f} {r['dmg_bare']:5.0f} "
                      f"{'YES' if r['degen'] else '':>5}")
                if pat == "patchy":
                    row = np.hstack([
                        img,
                        to_bgr_u8(a),
                        to_bgr_u8(r["a_rec"]),
                        to_bgr_u8(r["S_rec"]),
                        to_bgr_u8(S),
                    ])
                    panels[kind].append(row)
            print()

    for kind, rows in panels.items():
        panel = np.vstack(rows)
        cv2.imwrite(str(out_dir / f"spike_p4_alpha_matrix_{kind}.png"), panel)
    print("panels: input | a_true | a_rec | S_rec | S_true (rows = tones I..VI)")

    # --- shading variant: fnd_light uniform under a 0.82-1.12 luminance ramp
    print("\n=== shading variant (fnd_light, uniform 0.6, x0.82-1.12 vertical ramp) ===")
    print(f"{'tone':<4} {'det_in':>6} {'fp_out':>6} {'IoU':>5} {'s_mae':>6} {'dmg':>5}")
    shade_rows = []
    for tone, bgr in TONES.items():
        rng = np.random.RandomState(SEED)
        S = base_skin(rng, bgr)
        mcol = makeup_color("fnd_light", np.array(bgr, np.float32))
        a = alpha_field("uniform", rng)
        img = composite(S, mcol, a, rng, shade=True)
        ramp = np.linspace(0.82, 1.12, H, dtype=np.float32)[:, None, None]
        r = evaluate(img, np.clip(S * ramp, 0, 255), a)
        print(f"{tone:<4} {r['detect_in']:6.3f} {r['false_out']:6.3f} "
              f"{r['iou']:5.2f} {r['s_mae']:6.1f} {r['dmg_bare']:5.0f}")
        shade_rows.append(np.hstack([img, to_bgr_u8(a), to_bgr_u8(r["a_rec"]),
                                     to_bgr_u8(r["S_rec"])]))
    cv2.imwrite(str(out_dir / "spike_p4_alpha_matrix_shading.png"),
                np.vstack(shade_rows))

    # --- ambiguity sweep: tone-matched foundation, ||M-S|| -> 0
    print("\n=== ambiguity sweep: foundation contrast ||M-S|| in {30,15,8,4} "
          "(uniform 0.6 disk) ===")
    print(f"{'tone':<4} {'|M-S|':>6} {'det_in':>6} {'fp_out':>6} {'degen':>5}")
    for tone, bgr in TONES.items():
        s_vec = np.array(bgr, np.float32)
        # direction: lighter + slightly warmer, unit-normalized
        d = np.array([0.45, 0.55, 0.70], np.float32)
        d = d / np.linalg.norm(d)
        for target in (30.0, 15.0, 8.0, 4.0):
            rng = np.random.RandomState(SEED)
            S = base_skin(rng, bgr)
            mcol = np.clip(s_vec + d * target, 0, 255)
            a = alpha_field("uniform", rng)
            img = composite(S, mcol, a, rng)
            r = evaluate(img, S, a)
            print(f"{tone:<4} {target:6.0f} {r['detect_in']:6.3f} "
                  f"{r['false_out']:6.3f} {'YES' if r['degen'] else '':>5}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

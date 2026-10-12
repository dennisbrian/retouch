"""S1 spike — audit the melanin/hemoglobin chromophore decomposition.

Read-only audit of retouch/chromophore.py. No production code is modified.
Seeded, re-runnable. Renders panels to test_output/spike_s1_*.png.

Run:
    python3 scripts/spike_s1_chromophore_audit.py

Sections:
    A. Closed-form reduction check (is the 3x2 pinv == the two log-ratios?)
    B. Circularity guard (reconstruct-based probe -> identity; DO NOT use it)
    C. Independent probe: Fitzpatrick/Monk swatches x a* reddening x WB gain
       -> crosstalk + tone-stability matrix (the fairness-critical numbers)
    D. Real-photo channel maps (visual plausibility)
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retouch.chromophore import (  # noqa: E402
    decompose_chromophores,
    reconstruct_from_chromophores,
    _M,
    _M_PINV,
)

RNG = np.random.default_rng(20260717)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_output")


# ---------------------------------------------------------------------------
# A. Closed-form reduction
# ---------------------------------------------------------------------------
def closed_form(img_bgr):
    """Advisor's predicted closed form. mel=10*log(G/B); hb=(logR-2logG+logB)/0.75."""
    rgb = img_bgr[..., ::-1].astype(np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 1e-4, 1.0)
    L = -np.log(rgb)
    lr, lg, lb = L[..., 0], L[..., 1], L[..., 2]
    # -log => darker channel = larger L. mel = 10*log(G/B) = 10*(lb-lg)? check sign.
    mel = _M_PINV[0, 0] * (lr - lb) + _M_PINV[0, 1] * (lg - lb)
    hb = _M_PINV[1, 0] * (lr - lb) + _M_PINV[1, 1] * (lg - lb)
    return np.clip(mel, 0, None), np.clip(hb, 0, None)


def section_a():
    print("\n=== A. Closed-form / matrix reduction ===")
    print("M =\n", _M)
    print("M_PINV (2x3) =\n", np.round(_M_PINV, 4))
    # Random RGB, compare production decompose vs explicit pinv-of-diff closed form.
    img = (RNG.uniform(10, 250, size=(64, 64, 3))).astype(np.float32)
    m0, h0 = decompose_chromophores(img)
    m1, h1 = closed_form(img)
    print("max|mel diff| =", float(np.abs(m0 - m1).max()))
    print("max|hb  diff| =", float(np.abs(h0 - h1).max()))
    # Report the effective per-channel weights on L=-log(rgb) (blue-differenced).
    # conc = (L - L_b) @ pinv.T ; expand into weights on (lr,lg,lb).
    w_mel = np.array([_M_PINV[0, 0], _M_PINV[0, 1], -(_M_PINV[0, 0] + _M_PINV[0, 1])])
    w_hb = np.array([_M_PINV[1, 0], _M_PINV[1, 1], -(_M_PINV[1, 0] + _M_PINV[1, 1])])
    print("melanin weights on (Lr,Lg,Lb) =", np.round(w_mel, 4))
    print("hb      weights on (Lr,Lg,Lb) =", np.round(w_hb, 4))
    print("  (Lx = -log(x); so +weight on Lb means brighter blue lowers that map)")
    return w_mel, w_hb


# ---------------------------------------------------------------------------
# B. Circularity guard
# ---------------------------------------------------------------------------
def section_b():
    print("\n=== B. Circularity guard (why we must NOT probe via reconstruct) ===")
    mel_gt = RNG.uniform(0.0, 1.0, size=(32, 32)).astype(np.float32)
    hb_gt = RNG.uniform(0.0, 1.0, size=(32, 32)).astype(np.float32)
    bgr = reconstruct_from_chromophores(mel_gt, hb_gt, c=0.0)
    mel_r, hb_r = decompose_chromophores(bgr)
    # correlation of recovered vs ground truth
    def corr(a, b):
        a = a.ravel(); b = b.ravel()
        return float(np.corrcoef(a, b)[0, 1])
    print("recon->decompose  mel corr =", round(corr(mel_gt, mel_r), 4),
          " hb corr =", round(corr(hb_gt, hb_r), 4))
    print("  -> ~1.0 by construction. Probes below use an INDEPENDENT skin model.")


# ---------------------------------------------------------------------------
# C. Independent probe: swatches x reddening x WB
# ---------------------------------------------------------------------------
# Published sRGB skin swatches. Monk Skin Tone scale hex values (Monk 2019,
# used by Google) span light->dark; a good independent Fitzpatrick-like ladder.
MONK_HEX = [
    "f6ede4",  # 1 lightest
    "f3e7db",  # 2
    "f7ead0",  # 3
    "eadaba",  # 4
    "d7bd96",  # 5
    "a07e56",  # 6
    "825c43",  # 7
    "604134",  # 8
    "3a312a",  # 9
    "292420",  # 10 darkest
]


def hex_to_bgr(h):
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    return np.array([b, g, r], dtype=np.float32)


def add_reddening_lab(bgr_px, da):
    """Independent hemoglobin perturbation: boost Lab a* by da (redness), 8-bit LAB."""
    px = np.clip(bgr_px, 0, 255).reshape(1, 1, 3).astype(np.uint8)
    lab = cv2.cvtColor(px, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[0, 0, 1] = np.clip(lab[0, 0, 1] + da, 0, 255)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out.reshape(3).astype(np.float32)


def apply_wb_gain(bgr_px, gain_bgr):
    return np.clip(bgr_px * np.asarray(gain_bgr, np.float32), 0, 255)


def decompose_px(bgr_px):
    tile = np.tile(bgr_px.reshape(1, 1, 3), (4, 4, 1)).astype(np.float32)
    m, h = decompose_chromophores(tile)
    return float(m.mean()), float(h.mean())


def section_c():
    print("\n=== C. Independent probe: crosstalk + tone stability ===")
    # Warm/cool WB as per-channel gain (mimics illuminant, ~+/-, energy-ish balanced)
    WB = {
        "neutral": (1.0, 1.0, 1.0),
        "warm+": (0.85, 1.0, 1.12),   # less blue, more red -> warm
        "cool+": (1.15, 1.0, 0.88),   # more blue, less red -> cool
    }
    REDDEN = [0.0, 10.0, 25.0]  # Lab a* deltas

    header = f"{'tone':>5} {'wb':>8} {'da*':>5} {'melanin':>9} {'hb':>7}"
    print(header)
    rows = []
    for ti, hx in enumerate(MONK_HEX, 1):
        base = hex_to_bgr(hx)
        for wbname, g in WB.items():
            for da in REDDEN:
                px = add_reddening_lab(base, da)
                px = apply_wb_gain(px, g)
                mel, hb = decompose_px(px)
                rows.append((ti, wbname, da, mel, hb))
                print(f"{ti:>5} {wbname:>8} {da:>5.0f} {mel:>9.4f} {hb:>7.4f}")

    rows = np.array([(r[0], {'neutral':0,'warm+':1,'cool+':2}[r[1]], r[2], r[3], r[4]) for r in rows])

    print("\n--- Crosstalk: does reddening (hb up) move melanin? (neutral WB only) ---")
    print(f"{'tone':>5} {'mel@da0':>8} {'mel@da25':>9} {'dMel':>7} | {'hb@da0':>7} {'hb@da25':>8} {'dHb':>7}")
    for ti in range(1, 11):
        sel0 = rows[(rows[:,0]==ti)&(rows[:,1]==0)&(rows[:,2]==0)]
        sel25 = rows[(rows[:,0]==ti)&(rows[:,1]==0)&(rows[:,2]==25)]
        if sel0.size and sel25.size:
            m0,h0 = sel0[0,3],sel0[0,4]; m25,h25 = sel25[0,3],sel25[0,4]
            print(f"{ti:>5} {m0:>8.4f} {m25:>9.4f} {m25-m0:>7.4f} | {h0:>7.4f} {h25:>8.4f} {h25-h0:>7.4f}")

    print("\n--- WB confound: does warm/cool WB move melanin at fixed chromophores? (da*=0) ---")
    print(f"{'tone':>5} {'mel_neu':>8} {'mel_warm':>9} {'mel_cool':>9} {'hb_neu':>7} {'hb_warm':>8} {'hb_cool':>8}")
    for ti in range(1, 11):
        def g(wb):
            s=rows[(rows[:,0]==ti)&(rows[:,1]==wb)&(rows[:,2]==0)]
            return (s[0,3],s[0,4]) if s.size else (np.nan,np.nan)
        mn,hn=g(0); mw,hw=g(1); mc,hc=g(2)
        print(f"{ti:>5} {mn:>8.4f} {mw:>9.4f} {mc:>9.4f} {hn:>7.4f} {hw:>8.4f} {hc:>8.4f}")

    print("\n--- Tone stability: raw hb magnitude across tones (neutral, da*=0) ---")
    print("  (vein_attenuate multiplies raw hb by gain=60, so absolute drift matters)")
    for ti in range(1, 11):
        s = rows[(rows[:,0]==ti)&(rows[:,1]==0)&(rows[:,2]==0)]
        print(f"  tone {ti:>2}: mel={s[0,3]:.4f} hb={s[0,4]:.4f}")


# ---------------------------------------------------------------------------
# D. Real-photo channel maps
# ---------------------------------------------------------------------------
def _norm8(x):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    y = np.clip((x - lo) / (hi - lo), 0, 1)
    return (y * 255).astype(np.uint8)


def section_d(files):
    print("\n=== D. Real-photo channel maps ===")
    for f in files:
        path = os.path.join(OUT, f)
        img = cv2.imread(path)
        if img is None:
            print("  skip (not found):", f); continue
        h, w = img.shape[:2]
        s = 1600.0 / max(h, w)
        if s < 1.0:
            img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
        mel, hb = decompose_chromophores(img)
        mel_v = cv2.applyColorMap(_norm8(mel), cv2.COLORMAP_INFERNO)
        hb_v = cv2.applyColorMap(_norm8(hb), cv2.COLORMAP_JET)
        # luminance for leak comparison
        lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # correlation of maps with luminance (leak indicator)
        def corr(a,b):
            return float(np.corrcoef(a.ravel().astype(np.float32), b.ravel().astype(np.float32))[0,1])
        print(f"  {f}: corr(mel,lum)={corr(mel,lum):+.3f}  corr(hb,lum)={corr(hb,lum):+.3f}"
              f"  mel[min,max]=[{mel.min():.2f},{mel.max():.2f}] hb[min,max]=[{hb.min():.2f},{hb.max():.2f}]")
        panel = np.hstack([img, mel_v, hb_v])
        outp = os.path.join(OUT, f"spike_s1_{os.path.splitext(f)[0]}.png")
        cv2.imwrite(outp, panel)
        print("    wrote", outp, "(orig | melanin=inferno | hemoglobin=jet)")


if __name__ == "__main__":
    section_a()
    section_b()
    section_c()
    section_d(["DSCF4550.jpg", "DSCF4551.jpg", "DSCF4552.jpg"])
    print("\nDone.")

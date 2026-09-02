"""Post-epsilon face-op re-audit + neck-gate characterisation (2026-09-02).

Background: f69ab1e fixed the `> 1.0` mask normaliser that divided float
region masks by 255 whenever feathering overshot 1.0 by one ulp. On the
affected portraits the face-core composite alpha over skin collapsed to
1/255, so smoothing / equalize / blemish / under-eye / relight / sculpt were
silently discarded. Every default strength tuned while ~24% of faces were
partially invisible, and every "op X does nothing on image Y" conclusion,
is now suspect in both directions.

Per corpus image this script records:

  * ``affected``      — would the PRE-fix normaliser have collapsed this face
                        (regions.skin max > 1.0 with the feather-source clip
                        disabled)? Also the same flag for regions.neck.
  * natural (shipping) vs natural (emulated pre-fix) — what changed for users
    at the default recipe; mae/p99 inside the skin hull.
  * per-op probes on the shipping path with cached contexts — mae_in/p99_in
    (skin hull), mae_out (leak), neck-band mae — for the six discarded ops
    plus skin_sss / whiten (both also route through the skin composite).
  * neck harmonisation isolated: equalize=60 with and without
    ``SkinProcessor.harmonize_neck`` (stubbed to identity) → the neck op's own
    contribution; plus the shipped 1.3 yaw switch and the fraction of the
    landmark-derived neck rectangle that survives the plane-distance gate.

Sheets: ``sheets/<stem>_natural.jpg`` (source | ship | pre-fix | heat) and
``sheets/<stem>_ops.jpg`` (per-op crop | heat rows).

    RETOUCH_GPU=0 RETOUCH_MEDIAPIPE_BACKEND=legacy .venv/bin/python \
        scripts/qa/post_epsilon_faceop_sweep.py [--images A,B] [--no-ops]
"""
from __future__ import annotations

import argparse
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

from retouch import parsing, perf_optimizations, utils  # noqa: E402
from retouch.engine import RetouchEngine  # noqa: E402
from retouch.parsing import FACE_OVAL, LEFT_UNDER_EYE, RIGHT_UNDER_EYE  # noqa: E402
from retouch.skin import SkinProcessor  # noqa: E402

CORPUS = ROOT / "test_output" / "detection_recall_study" / "corpus.txt"
OUT = ROOT / "test_output" / "post_epsilon_faceop_sweep"
SHEETS = OUT / "sheets"
MAX_DIM = 2048

# op -> probe value. All route through the face-core skin composite that the
# epsilon bug collapsed. blemish was not in the 08-31 audit's OPS table.
OPS = {
    "smooth": 70, "equalize": 60, "blemish": 60, "dark_circles": 60,
    "relight": 60, "sculpt": 50, "skin_sss": 60, "whiten": 50,
}
ALL_OFF = {k: 0 for k in OPS}
ALL_OFF.update({"undereye_darken_removal": 0, "redness_even": 0, "skin_hue_unify": 0,
                "skin_chroma_even": 0})

# --- pre-fix emulation -----------------------------------------------------
_ORIG_FEATHER = utils.feather_mask
_ORIG_NORM = perf_optimizations._norm_mask


def _feather_unclipped(mask, radius=None, sigma=None):
    """utils.feather_mask as it was before f69ab1e (no clip at the source)."""
    if mask is None or mask.size == 0:
        return mask
    m = mask.astype(np.float32)
    if m.max() > 1.5:
        m /= 255.0
    if sigma is None and radius is None:
        return m
    if sigma is None:
        sigma = max(radius / 3.0, 1.0)
    if radius is None:
        radius = int(sigma * 3)
    ksize = max(radius * 2 + 1, 3)
    from retouch.acceleration import accelerated_gaussian_blur
    return accelerated_gaussian_blur(m, ksize, sigma)


def _norm_mask_prefix(mask):
    """perf_optimizations._norm_mask before f69ab1e: `> 1.0` and no clip."""
    if mask is None:
        return None
    m = mask.astype(np.float32)
    if m.max() > 1.0:
        m /= 255.0
    return m


def set_prefix_emulation(on: bool) -> None:
    parsing.feather_mask = _feather_unclipped if on else _ORIG_FEATHER
    utils.feather_mask = _feather_unclipped if on else _ORIG_FEATHER
    perf_optimizations._norm_mask = _norm_mask_prefix if on else _ORIG_NORM


# --- helpers ----------------------------------------------------------------
def shrink(img):
    h, w = img.shape[:2]
    s = MAX_DIM / max(h, w)
    return img if s >= 1 else cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def u8(res):
    a = np.asarray(res)
    return a if a.dtype == np.uint8 else np.clip(a, 0, 255).astype(np.uint8)


def lms_of(fc):
    lm = fc.face_data.landmarks
    return lm.landmark if hasattr(lm, "landmark") else lm


def hull(lms, idx, w, h, dilate_px=0):
    out = np.zeros((h, w), np.uint8)
    pts = np.array([[int(lms[i].x * w), int(lms[i].y * h)] for i in idx], np.int32)
    cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2)
        out = cv2.dilate(out, k)
    return out


def face_geom(lms, w, h):
    xs = [int(l.x * w) for l in lms]
    ys = [int(l.y * h) for l in lms]
    return max(xs) - min(xs), max(ys) - min(ys)


def neck_rect_mask(lms, w, h):
    """The landmark-derived neck rectangle harmonize_neck falls back to."""
    face_w, face_h = face_geom(lms, w, h)
    cx, cy = int(lms[SkinProcessor.MEDIAPIPE_CHIN_IDX].x * w), int(lms[SkinProcessor.MEDIAPIPE_CHIN_IDX].y * h)
    m = np.zeros((h, w), np.uint8)
    y1, y2 = cy, min(int(cy + face_h * 1.5), h)
    x1, x2 = max(int(cx - face_w * 0.8), 0), min(int(cx + face_w * 0.8), w)
    if y2 > y1 and x2 > x1:
        m[y1:y2, x1:x2] = 1
    return m


def neck_depth_gate(lms, w, h):
    """Replicates skin.harmonize_neck's plane-distance gate in full-image coords.
    Returns (yaw_ratio, gate_applied, survive_fraction_of_neck_rect)."""
    d_left = abs(lms[6].x - lms[234].x)
    d_right = abs(lms[454].x - lms[6].x)
    yaw = max(d_left, d_right) / (min(d_left, d_right) + 1e-5)
    rect = neck_rect_mask(lms, w, h).astype(bool)
    if yaw > 1.3:
        return yaw, False, 1.0
    face_w, _ = face_geom(lms, w, h)
    p1 = np.array([lms[33].x * w, lms[33].y * h, lms[33].z * face_w])
    p2 = np.array([lms[263].x * w, lms[263].y * h, lms[263].z * face_w])
    p3 = np.array([lms[6].x * w, lms[6].y * h, lms[6].z * face_w])
    n = np.cross(p2 - p1, p3 - p1)
    n = n / (np.linalg.norm(n) + 1e-5)
    Y, X = np.ogrid[:h, :w]
    dist = np.abs(n[0] * (X - p1[0]) + n[1] * (Y - p1[1]))
    gate = dist < face_w * 0.15
    surv = float(gate[rect].mean()) if rect.any() else 0.0
    return yaw, True, surv


def stats(diff, mask):
    sel = diff[mask.astype(bool)]
    if sel.size == 0:
        return 0.0, 0.0, 0.0
    return float(sel.mean()), float(np.percentile(sel, 99)), float((sel > 6).mean())


def heat(d, gain=6.0):
    return cv2.applyColorMap(np.clip(d.astype(np.float32) * gain, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def crop_box(fc, shape, pad=0.35, down=0.55):
    h, w = shape[:2]
    x, y, bw, bh = fc.face_data.bbox
    x0, y0 = max(int(x - pad * bw), 0), max(int(y - pad * bh), 0)
    x1, y1 = min(int(x + bw * (1 + pad)), w), min(int(y + bh * (1 + pad + down)), h)  # include neck
    return x0, y0, x1, y1


def row(tiles, height=380, label=None):
    tiles = [cv2.resize(t, (int(t.shape[1] * height / t.shape[0]), height), interpolation=cv2.INTER_AREA) for t in tiles]
    sep = np.full((height, 3, 3), 220, np.uint8)
    out = []
    for i, t in enumerate(tiles):
        if i:
            out.append(sep)
        out.append(t)
    r = np.hstack(out)
    if label:
        cv2.putText(r, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return r


def stack(rows, path):
    W = max(r.shape[1] for r in rows)
    padded = []
    for i, r in enumerate(rows):
        if r.shape[1] < W:
            r = cv2.copyMakeBorder(r, 0, 0, 0, W - r.shape[1], cv2.BORDER_CONSTANT, value=(30, 30, 30))
        if i:
            padded.append(np.full((4, W, 3), 220, np.uint8))
        padded.append(r)
    cv2.imwrite(str(path), np.vstack(padded), [cv2.IMWRITE_JPEG_QUALITY, 86])


def absdiff(a, b):
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2).astype(np.uint8)


# --- main -------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None)
    ap.add_argument("--no-ops", action="store_true", help="skip per-op probes (flag + natural only)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    SHEETS.mkdir(parents=True, exist_ok=True)

    if args.images:
        paths = [ROOT / "test_output" / f"{s}.jpg" for s in args.images.split(",")]
    else:
        paths = [ROOT / l.strip() for l in CORPUS.read_text().splitlines() if l.strip()]

    eng = RetouchEngine()
    rows_out = []
    t_all = time.time()
    try:
        for p in paths:
            if not p.exists():
                continue
            stem = p.stem
            img = shrink(cv2.imread(str(p)))
            h, w = img.shape[:2]
            t0 = time.time()

            # 1. Pre-fix emulation: unclipped feathering + old _norm_mask.
            set_prefix_emulation(True)
            try:
                res_pre = eng.process(img, recipe="natural")
                ctx_pre = res_pre.face_contexts or []
                nat_pre = u8(res_pre)
            finally:
                set_prefix_emulation(False)
            if not ctx_pre:
                rows_out.append({"image": stem, "faces": 0})
                print(f"{stem}: no face", flush=True)
                continue
            fc0 = max(ctx_pre, key=lambda f: f.face_data.bbox[2] * f.face_data.bbox[3])
            skin_max = float(np.asarray(fc0.regions.skin).max()) if fc0.regions.skin is not None else 0.0
            neck_max = float(np.asarray(fc0.regions.neck).max()) if getattr(fc0.regions, "neck", None) is not None else 0.0
            affected = skin_max > 1.0

            # 2. Shipping path: fresh detection (clean contexts), natural as shipped.
            res_ship = eng.process(img, recipe="natural")
            ctx = res_ship.face_contexts or []
            nat_ship = u8(res_ship)
            fc = max(ctx, key=lambda f: f.face_data.bbox[2] * f.face_data.bbox[3])
            lms = lms_of(fc)
            skin_hull = hull(lms, FACE_OVAL, w, h, dilate_px=int(fc.face_data.ied * 0.9))
            neck_band = neck_rect_mask(lms, w, h) & (1 - hull(lms, FACE_OVAL, w, h, int(fc.face_data.ied * 0.3)))
            outside = 1 - (skin_hull | neck_band)
            ue_hull = hull(lms, LEFT_UNDER_EYE, w, h, int(fc.face_data.ied * 0.3)) | hull(lms, RIGHT_UNDER_EYE, w, h, int(fc.face_data.ied * 0.3))
            neck_reg = getattr(fc.regions, "neck", None)
            neck_reg_max = float(np.asarray(neck_reg).max()) if neck_reg is not None else 0.0

            d_nat = absdiff(nat_ship, nat_pre)
            d_in = absdiff(nat_ship, img)
            rec = {"image": stem, "faces": len(ctx), "ied": round(fc.face_data.ied, 1),
                   "affected": bool(affected), "skin_max_prefix": skin_max, "neck_max_prefix": neck_max}
            rec["natural_ship_vs_prefix"] = dict(zip(("mae_in", "p99_in", "frac_gt6_in"), stats(d_nat, skin_hull)))
            rec["natural_ship_vs_input"] = dict(zip(("mae_in", "p99_in", "frac_gt6_in"), stats(d_in, skin_hull)))
            rec["natural_prefix_vs_input"] = dict(zip(("mae_in", "p99_in", "frac_gt6_in"), stats(absdiff(nat_pre, img), skin_hull)))
            yaw, gate_on, surv = neck_depth_gate(lms, w, h)
            rec["neck"] = {"yaw_ratio6": round(yaw, 3), "depth_gate_applied": gate_on, "rect_survive_frac": round(surv, 3),
                           "bisenet_neck_used": neck_reg_max > 0.01, "neck_region_max": round(neck_reg_max, 4)}

            x0, y0, x1, y1 = crop_box(fc, img.shape)
            crop = lambda a: a[y0:y1, x0:x1]  # noqa: E731
            stack([row([crop(img), crop(nat_ship), crop(nat_pre), crop(heat(d_nat))], 420,
                       f"{stem} natural | src | SHIP | PRE-FIX emu | heat x6  affected={affected} skin_max={skin_max:.7f}")],
                  SHEETS / f"{stem}_natural.jpg")

            # 3. Per-op probes on the shipping path (cached contexts).
            if not args.no_ops:
                base = u8(eng.process(img, recipe="natural", face_contexts=ctx, **ALL_OFF))
                rec["ops"] = {}
                op_rows = []
                for op, val in OPS.items():
                    kw = dict(ALL_OFF)
                    kw[op] = val
                    out = u8(eng.process(img, recipe="natural", face_contexts=ctx, **kw))
                    d = absdiff(out, base)
                    mi, p99, fr = stats(d, skin_hull)
                    mo = float(d[outside.astype(bool)].mean()) if outside.any() else 0.0
                    mn = float(d[neck_band.astype(bool)].mean()) if neck_band.any() else 0.0
                    ue_m, ue_p99, ue_fr = stats(d, ue_hull)
                    cy_, cx_ = np.unravel_index(int(np.argmax(cv2.GaussianBlur(d.astype(np.float32), (0, 0), 5))), d.shape)
                    rec["ops"][op] = {"value": val, "mae_in": round(mi, 4), "p99_in": round(p99, 2),
                                      "frac_gt6_in": round(fr, 4), "mae_out": round(mo, 4), "mae_neck": round(mn, 4),
                                      "mae_ue": round(ue_m, 4), "p99_ue": round(ue_p99, 2),
                                      "p99_full": round(float(np.percentile(d, 99)), 2), "max_full": int(d.max()),
                                      "n_gt6_full": int((d > 6).sum()), "hotspot_xy": [int(cx_), int(cy_)]}
                    op_rows.append(row([crop(out), crop(heat(d))], 300, f"{op}={val} mae_in={mi:.2f} p99={p99:.0f} neck={mn:.2f} out={mo:.3f}"))
                    if op == "equalize":
                        # Neck isolation: same render with harmonize_neck stubbed.
                        orig = SkinProcessor.harmonize_neck
                        SkinProcessor.harmonize_neck = lambda self, img_bgr, *a, **k: img_bgr
                        try:
                            out_nn = u8(eng.process(img, recipe="natural", face_contexts=ctx, **kw))
                        finally:
                            SkinProcessor.harmonize_neck = orig
                        dn = absdiff(out, out_nn)
                        mn_iso = float(dn[neck_band.astype(bool)].mean()) if neck_band.any() else 0.0
                        mf_iso = float(dn[skin_hull.astype(bool)].mean())
                        rec["neck"]["harmonize_iso_mae_neck"] = round(mn_iso, 4)
                        rec["neck"]["harmonize_iso_mae_face"] = round(mf_iso, 4)
                        rec["neck"]["harmonize_iso_p99_neck"] = round(float(np.percentile(dn[neck_band.astype(bool)], 99)) if neck_band.any() else 0.0, 2)
                        op_rows.append(row([crop(out_nn), crop(heat(dn))], 300,
                                           f"neck-harmonize ISOLATED (eq60 minus eq60/no-neck) mae_neck={mn_iso:.2f} face={mf_iso:.2f} yaw={yaw:.2f} gate={gate_on} surv={surv:.2f}"))
                stack(op_rows, SHEETS / f"{stem}_ops.jpg")

            rows_out.append(rec)
            ns = rec["natural_ship_vs_prefix"]
            print(f"{stem:9s} aff={int(affected)} skin_max={skin_max:.7f} nat ship-vs-pre mae={ns['mae_in']:.2f} p99={ns['p99_in']:.0f}"
                  f" | yaw={yaw:.2f} gate={int(gate_on)} surv={surv:.2f}"
                  + (" | " + " ".join(f"{k}={v['mae_in']:.2f}" for k, v in rec.get("ops", {}).items()) if "ops" in rec else "")
                  + f"  ({time.time()-t0:.0f}s)", flush=True)
            (OUT / "sweep.json").write_text(json.dumps(rows_out, indent=1))
    finally:
        close = getattr(eng, "close", None)
        if callable(close):
            close()
    (OUT / "sweep.json").write_text(json.dumps(rows_out, indent=1))
    det = [r for r in rows_out if r.get("faces")]
    print(f"\nDONE {len(det)}/{len(rows_out)} faces in {time.time()-t_all:.0f}s; affected={sum(r['affected'] for r in det)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

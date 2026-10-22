"""Legacy ``blemish`` vs ``spot_heal`` on planted spots (Phase 1.1 of
docs/plans/PLAN_FACE_QUALITY_ROADMAP_2026_10_22.md).

For each photo (scaled to ``--long-side``) and each exposure (as shot, and a
simulated 0.45x linear-light capture), pimples and moles are planted on every
face's skin in linear light (a transmittance, so they read the same on every
skin tone; positions from tests/golden_identity.pick_mark_sites). Each setting
is rendered on the planted and the unplanted photo with the same detected
faces, everything else held at the recipe's values, and scored against the
setting with both ops off:

- pimple fixed (soft pimples and stronger, larger inflamed ones): its low-pass Lab residual (planted render vs unplanted render)
  is at most half of what it is with both ops off;
- mole kept: at least half of that residual is left;
- other skin changed: face-skin pixels farther than 3 spot radii from every
  planted spot that moved by more than 3 levels vs both ops off, as a share of
  face skin (real spots the op heals count here too).

Photos are read from the command line and are never written anywhere except
the optional ``--sheet`` folder.

Usage:
    RETOUCH_GPU=0 RETOUCH_MEDIAPIPE_BACKEND=legacy .venv/bin/python \\
        scripts/qa/blemish_remeasure.py PHOTO [PHOTO ...] [--json out.json] [--sheet DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests import golden_identity as gi  # noqa: E402

PIMPLE_TRANS = np.array([0.80, 0.70, 0.93])  # BGR, hemoglobin-like (as tests/test_spot_heal_auto.py)
PIMPLE_R_FW = 0.007
# A stronger, larger inflamed pimple, so the legacy pass (a contrast
# threshold) gets a spot well above its floor too.
STRONG_TRANS = np.array([0.62, 0.45, 0.85])
STRONG_R_FW = 0.010
MOLE_R_FW = 0.008
PIMPLES_PER_FACE = 8  # half of them strong
MOLES_PER_FACE = 4
MIN_FACE_W = 120  # px at the working size; smaller faces get sub-pixel spots

SETTINGS = {
    "off": dict(blemish=0, spot_heal=0),
    "legacy_30": dict(blemish=30, spot_heal=0),
    "spot_30": dict(blemish=0, spot_heal=30),
    "legacy_55": dict(blemish=55, spot_heal=0),
    "spot_55": dict(blemish=0, spot_heal=55),
}


def plant(img, marks):
    out = img.copy()
    for m in marks:
        if m.kind in ("pimple", "strong"):
            trans = STRONG_TRANS if m.kind == "strong" else PIMPLE_TRANS
            lin = gi._srgb_to_lin(out)
            h, w = out.shape[:2]
            R = int(np.ceil(m.r * 3)) + 1
            y1, y2, x1, x2 = max(0, m.y - R), min(h, m.y + R + 1), max(0, m.x - R), min(w, m.x + R + 1)
            yy, xx = np.mgrid[y1:y2, x1:x2]
            prof = np.exp(-0.5 * (np.hypot(xx - m.x, yy - m.y) / (0.6 * m.r)) ** 2)
            lin[y1:y2, x1:x2] *= trans[None, None, :] ** prof[:, :, None]
            out = np.clip(gi._lin_to_srgb(lin) + 0.5, 0, 255).astype(np.uint8)
        else:
            out = gi.plant_mark(out, m)
    return out


def plan(ctxs, shape, seed):
    marks, faces = [], []
    for i, c in enumerate(ctxs):
        fw = float(c.face_data.bbox[2])
        if fw < MIN_FACE_W:
            continue
        sites = gi.pick_mark_sites(c, shape, PIMPLES_PER_FACE + MOLES_PER_FACE, min_gap_fw=0.09,
                                   seed=seed + i)
        for j, (x, y) in enumerate(sites):
            kind = ("pimple", "strong", "mole")[j % 3]
            r_fw = {"pimple": PIMPLE_R_FW, "strong": STRONG_R_FW, "mole": MOLE_R_FW}[kind]
            marks.append(gi.Mark(kind, x, y, max(1.5, r_fw * fw)))
        faces.append(i)
    return marks, faces


def darker(img, gain=0.45):
    return np.clip(gi._lin_to_srgb(gi._srgb_to_lin(img) * gain) + 0.5, 0, 255).astype(np.uint8)


def skin_union(ctxs, faces, shape):
    m = np.zeros(shape, bool)
    for i in faces:
        m |= gi.face_skin(ctxs[i], shape)
    return m


def crop_sheet(path, title, panels, ctx, marks):
    x, y, w, h = (int(v) for v in ctx.face_data.bbox)
    H, W = panels[0][1].shape[:2]
    y1, y2, x1, x2 = max(0, y), min(H, y + h), max(0, x), min(W, x + w)
    tiles = []
    for name, im in panels:
        t = np.ascontiguousarray(im[y1:y2, x1:x2])
        scale = 420.0 / t.shape[0]
        t = cv2.resize(t, (int(t.shape[1] * scale), 420), interpolation=cv2.INTER_AREA)
        for m in marks:
            if x1 <= m.x < x2 and y1 <= m.y < y2:
                c = (int((m.x - x1) * scale), int((m.y - y1) * scale))
                rad = int(m.r * scale * 3)
                cv2.circle(t, c, rad, (255, 128, 0) if m.kind == "mole" else (0, 255, 255), 1)
        cv2.putText(t, name, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(t)
    sheet = np.hstack(tiles)
    cv2.putText(sheet, title, (6, sheet.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("photos", nargs="+", type=Path)
    ap.add_argument("--recipe", default="natural")
    ap.add_argument("--long-side", type=int, default=2000)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--sheet", type=Path, default=None, help="write face-crop sheets here")
    args = ap.parse_args(argv)

    from retouch.engine import RetouchEngine

    engine = RetouchEngine()
    rows = []
    t0 = time.time()
    for pi, path in enumerate(args.photos):
        src = cv2.imread(str(path), cv2.IMREAD_COLOR)
        s = args.long_side / max(src.shape[:2])
        if s < 1:
            src = cv2.resize(src, (int(src.shape[1] * s) // 2 * 2, int(src.shape[0] * s) // 2 * 2),
                             interpolation=cv2.INTER_AREA)
        for exposure, img in (("as_shot", src), ("darker_0.45x", darker(src))):
            first = engine.process(img, recipe=args.recipe)
            ctxs = first.face_contexts or []
            marks, faces = plan(ctxs, img.shape[:2], seed=17 * pi)
            if not marks:
                print(f"{path.name} {exposure}: no face >= {MIN_FACE_W}px", flush=True)
                continue
            planted = plant(img, marks)
            skin = skin_union(ctxs, faces, img.shape[:2])
            near = np.zeros(img.shape[:2], np.uint8)
            for m in marks:
                cv2.circle(near, (m.x, m.y), int(np.ceil(3 * m.r)), 1, -1)
            far_skin = skin & (near == 0)
            renders = {}
            for name, kw in SETTINGS.items():
                np.random.seed(0)
                p = engine.process(planted, recipe=args.recipe, face_contexts=ctxs, **kw)
                np.random.seed(0)
                c = engine.process(img, recipe=args.recipe, face_contexts=ctxs, **kw)
                renders[name] = (np.asarray(p, np.uint8), np.asarray(c, np.uint8))
            base_res = [gi.mark_residual(*renders["off"], m) for m in marks]
            for name in SETTINGS:
                if name == "off":
                    continue
                p, c = renders[name]
                res = [gi.mark_residual(p, c, m) / max(b, 1e-6) for m, b in zip(marks, base_res)]
                pim = [r for r, m in zip(res, marks) if m.kind in ("pimple", "strong")]
                strong = [r for r, m in zip(res, marks) if m.kind == "strong"]
                mol = [r for r, m in zip(res, marks) if m.kind == "mole"]
                ch = (np.abs(p.astype(np.int16) - renders["off"][0].astype(np.int16)).max(2) > 3)
                row = {"photo": path.name, "exposure": exposure, "setting": name,
                       "faces": len(faces), "face_w": [int(ctxs[i].face_data.bbox[2]) for i in faces],
                       "pimples": len(pim), "moles": len(mol),
                       "pimples_fixed": int(sum(r <= 0.5 for r in pim)),
                       "strong": len(strong), "strong_fixed": int(sum(r <= 0.5 for r in strong)),
                       "moles_kept": int(sum(r >= 0.5 for r in mol)),
                       "other_skin_px": int((ch & far_skin).sum()), "skin_px": int(far_skin.sum()),
                       "pimple_residual_median": float(np.median(pim)),
                       "mole_residual_median": float(np.median(mol)) if mol else None}
                rows.append(row)
                print(f"[{time.time() - t0:.0f}s] {path.name[:24]} {exposure} fw {row['face_w']} {name}: pimples "
                      f"{row['pimples_fixed']}/{row['pimples']} fixed (strong {row['strong_fixed']}/{row['strong']}), moles {row['moles_kept']}/{row['moles']} "
                      f"kept, other skin {100.0 * row['other_skin_px'] / max(row['skin_px'], 1):.2f}%", flush=True)
            if args.sheet:
                args.sheet.mkdir(parents=True, exist_ok=True)
                big = max(faces, key=lambda i: ctxs[i].face_data.bbox[2])
                panels = [("planted input", planted)] + [(n, renders[n][0]) for n in SETTINGS]
                crop_sheet(args.sheet / f"{pi:02d}_{exposure}.jpg",
                           f"{path.name[:40]} {exposure} ({args.recipe}; planted spots are simulated)",
                           panels, ctxs[big], marks)

    print("\nTotals (all photos):")
    for exposure in ("as_shot", "darker_0.45x", None):
        for name in [n for n in SETTINGS if n != "off"]:
            sel = [r for r in rows if r["setting"] == name and (exposure is None or r["exposure"] == exposure)]
            if not sel:
                continue
            pf, pn = sum(r["pimples_fixed"] for r in sel), sum(r["pimples"] for r in sel)
            sf, sn_ = sum(r["strong_fixed"] for r in sel), sum(r["strong"] for r in sel)
            mk, mn = sum(r["moles_kept"] for r in sel), sum(r["moles"] for r in sel)
            sp, sn = sum(r["other_skin_px"] for r in sel), sum(r["skin_px"] for r in sel)
            print(f"  {exposure or 'both'} {name}: pimples fixed {pf}/{pn} (strong {sf}/{sn_}), moles kept {mk}/{mn}, "
                  f"other skin changed {100.0 * sp / max(sn, 1):.2f}% of {sn} px")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

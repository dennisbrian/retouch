"""Identity scorecard on your own photos, plus blind A/B face-crop sheets.

Runs the per-face scorecard from tests/golden_identity.py (geometry drift,
lightness shift in stops, survival of planted moles/freckles, AA6 face SSIM
and D) for each variant on each photo, as shot and at a simulated 0.45x
linear-light exposure. Variants are ``name=recipe[:param=value,...]``.

With ``--ab A,B --sheets DIR`` it also writes one sheet per face with the two
variants' face crops side by side in a random left/right order, and a
``key.json`` saying which side is which, so the owner can judge blind.

Photos are only read; nothing is written except --json and --sheets output.

Usage:
    RETOUCH_GPU=0 RETOUCH_MEDIAPIPE_BACKEND=legacy .venv/bin/python \\
        scripts/qa/identity_photos.py PHOTO [PHOTO ...] \\
        --variant natural=natural --variant v2=natural_v2 \\
        [--ab natural,v2 --sheets DIR] [--json out.json] [--no-darker]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests import golden_identity as gi  # noqa: E402

MIN_FACE_W = 120  # px at the working size


def parse_variant(text):
    name, _, rest = text.partition("=")
    recipe, _, kv = rest.partition(":")
    kw = {}
    for item in filter(None, kv.split(",")):
        k, _, v = item.partition("=")
        try:
            kw[k] = int(v)
        except ValueError:
            try:
                kw[k] = float(v)
            except ValueError:
                kw[k] = v
    return name, recipe, kw


def darker(img, gain=0.45):
    return np.clip(gi._lin_to_srgb(gi._srgb_to_lin(img) * gain) + 0.5, 0, 255).astype(np.uint8)


def face_crop(img, ctx, pad=0.25, height=560):
    x, y, w, h = (int(v) for v in ctx.face_data.bbox)
    H, W = img.shape[:2]
    px, py = int(pad * w), int(pad * h)
    t = img[max(0, y - py):min(H, y + h + py), max(0, x - px):min(W, x + w + px)]
    s = height / t.shape[0]
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LANCZOS4
    return cv2.resize(np.ascontiguousarray(t), (int(t.shape[1] * s), height), interpolation=interp)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("photos", nargs="+", type=Path)
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--long-side", type=int, default=2000)
    ap.add_argument("--no-darker", action="store_true")
    ap.add_argument("--ab", default="", help="two variant names for the blind sheets")
    ap.add_argument("--sheets", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    from retouch.engine import RetouchEngine

    variants = [parse_variant(v) for v in args.variant]
    ab = [a for a in args.ab.split(",") if a]
    engine = RetouchEngine()
    rng = random.Random(20261022)
    rows, key = [], {}
    t0 = time.time()
    for pi, path in enumerate(args.photos):
        src = cv2.imread(str(path), cv2.IMREAD_COLOR)
        s = args.long_side / max(src.shape[:2])
        if s < 1:
            src = cv2.resize(src, (int(src.shape[1] * s) // 2 * 2, int(src.shape[0] * s) // 2 * 2),
                             interpolation=cv2.INTER_AREA)
        exposures = [("as_shot", src)] + ([] if args.no_darker else [("darker_0.45x", darker(src))])
        for exposure, img in exposures:
            ctxs = engine.process(img, recipe=variants[0][1]).face_contexts or []
            faces = [i for i, c in enumerate(ctxs) if c.face_data.bbox[2] >= MIN_FACE_W]
            if not faces:
                print(f"{path.name}: no face >= {MIN_FACE_W}px", flush=True)
                continue
            plan = gi.plan_marks(ctxs, img.shape[:2])
            planted = gi.plant_all(img, [plan[i] for i in faces])
            bases = {i: gi.geometry_baseline(engine._detector, img, ctxs[i].face_data.bbox) for i in faces}
            outs = {}
            for name, recipe, kw in variants:
                np.random.seed(0)
                out = np.asarray(engine.process(img, recipe=recipe, face_contexts=ctxs, **kw), np.uint8)
                np.random.seed(0)
                outp = np.asarray(engine.process(planted, recipe=recipe, face_contexts=ctxs, **kw), np.uint8)
                outs[name] = out
                for i in faces:
                    sc = gi.face_scorecard(engine._detector, img, out, ctxs[i], bases[i], plan[i], planted, outp)
                    sc.update(photo=path.name, exposure=exposure, face=i, variant=name,
                              face_w=int(ctxs[i].face_data.bbox[2]))
                    rows.append(sc)
                    print(f"[{time.time() - t0:.0f}s] {path.name[:24]} {exposure} face {i} {name}: "
                          f"geo {sc['geo_ratio'] if sc['geo_ratio'] is None else round(sc['geo_ratio'], 2)} "
                          f"L {sc['l_stops']:+.3f} stop dE {sc['dE']:.2f} marks {sc['marks_kept']}/"
                          f"{sc['marks_total']} (moles {sc['moles_kept']}) aa6 {sc['aa6_ssim']:.3f}/"
                          f"{sc['aa6_d']:+.3f} flags {sc['identity_flags']}", flush=True)
            if len(ab) == 2 and args.sheets and exposure == "as_shot":
                args.sheets.mkdir(parents=True, exist_ok=True)
                for i in faces:
                    left, right = (ab if rng.random() < 0.5 else ab[::-1])
                    tag = f"{pi:02d}_face{i}"
                    a, b = face_crop(outs[left], ctxs[i]), face_crop(outs[right], ctxs[i])
                    gap = np.full((a.shape[0], 12, 3), 255, np.uint8)
                    sheet = np.hstack([a, gap, b])
                    for txt, x in (("A", 8), ("B", a.shape[1] + 20)):
                        cv2.putText(sheet, txt, (x, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
                    cv2.imwrite(str(args.sheets / f"{tag}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    key[tag] = {"A": left, "B": right, "photo": path.name}
    print("\nPer variant (all photos, all exposures):")
    for name, _, _ in variants:
        sel = [r for r in rows if r["variant"] == name]
        if not sel:
            continue
        geo = [r["geo_ratio"] for r in sel if r["geo_ratio"] is not None]
        stops = [r["l_stops"] for r in sel if r["l_stops"] is not None]
        kept, total = sum(r["marks_kept"] or 0 for r in sel), sum(r["marks_total"] for r in sel)
        moles = [tuple(map(int, r["moles_kept"].split("/"))) for r in sel if r["moles_kept"]]
        print(f"  {name}: {len(sel)} faces | geometry flags {sum('geo' in r['identity_flags'] for r in sel)} "
              f"(max {max(geo):.2f}x) | lightness median {float(np.median(stops)):+.3f} stop, flags "
              f"{sum('l' in r['identity_flags'] for r in sel)} | marks kept {kept}/{total} "
              f"(moles {sum(m[0] for m in moles)}/{sum(m[1] for m in moles)}) | aa6 ssim median "
              f"{float(np.median([r['aa6_ssim'] for r in sel])):.3f}, D median "
              f"{float(np.median([r['aa6_d'] for r in sel])):+.3f}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1, default=str))
    if key and args.sheets:
        (args.sheets / "key.json").write_text(json.dumps(key, indent=1))


if __name__ == "__main__":
    main()

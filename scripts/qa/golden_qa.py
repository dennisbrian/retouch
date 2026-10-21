"""Golden hard-case QA sweep: every curated recipe on every tests/golden case.

For each case and recipe it renders the recipe twice with the same detected
faces, once as is and once with every skin op off, and reports:

- faces retouched vs people in the photo,
- skin leak (changed pixels outside the grown skin mask, share of skin area),
- planted object change (clip, rose, frames, poster inside the face zone),
- SSIM against the approved reference (reference recipes only),
- per-face identity columns (tests/golden_identity.py): landmark geometry
  drift, face-skin lightness shift in stops next to colour drift, survival
  of moles/freckles planted on the input, and AA6 face SSIM / high-pass
  change D, plus the number of people the recipe skipped.

Identity columns are a report: they never fail the run (the leak, object,
face-count and SSIM checks still do).

The thresholds are the ones tests/test_golden_qa.py enforces in CI; CI
runs a sample, this script runs everything.

Usage:
    RETOUCH_GPU=0 RETOUCH_MEDIAPIPE_BACKEND=legacy \\
        .venv/bin/python scripts/qa/golden_qa.py [--recipes a,b] [--cases x,y] \\
        [--json out.json] [--approve] [--no-identity] [--no-marks]

``--approve`` (re)writes tests/golden/references/ for REFERENCE_RECIPES from
the current renders. Only run it after looking at the renders.
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

from tests import golden_identity as gi
from tests import golden_qa as gq


def render(engine, img, recipe, **kw):
    np.random.seed(0)  # grain/sparkle use the global RNG
    return engine.process(img, recipe=recipe, **kw)


class IdentityCache:
    """Per case: geometry baselines and planted marks, shared by every recipe."""

    def __init__(self, engine, marks: bool = True):
        self.engine = engine
        self.marks = marks
        self._base = {}
        self._marks = {}

    def baseline(self, case, i, ctx):
        key = (case.name, i)
        if key not in self._base:
            self._base[key] = gi.geometry_baseline(self.engine._detector, case.image, ctx.face_data.bbox)
        return self._base[key]

    def planted(self, case, ctxs):
        if not self.marks:
            return None, [[] for _ in ctxs]
        if case.name not in self._marks:
            plan = gi.plan_marks(ctxs, case.image.shape[:2])
            self._marks[case.name] = (gi.plant_all(case.image, plan), plan)
        return self._marks[case.name]


def identity_rows(engine, case, recipe, first, ctxs, cache: IdentityCache):
    """One identity scorecard per face (see tests/golden_identity.py)."""
    planted, plan = cache.planted(case, ctxs)
    out_planted = render(engine, planted, recipe, face_contexts=ctxs) if planted is not None else None
    faces = []
    for i, ctx in enumerate(ctxs):
        sc = gi.face_scorecard(engine._detector, case.image, gq.to_u8(first), ctx, cache.baseline(case, i, ctx),
                               plan[i] if i < len(plan) else (), planted,
                               gq.to_u8(out_planted) if out_planted is not None else None)
        sc["face"] = i
        faces.append(sc)
    return faces


def identity_summary(row) -> str:
    faces = row.get("identity") or []
    if not faces:
        return f"skipped {row.get('skipped', 0)}"
    geo = [f["geo_ratio"] for f in faces if f["geo_ratio"] is not None]
    stops = [f["l_stops"] for f in faces if f["l_stops"] is not None]
    kept = sum(f["marks_kept"] or 0 for f in faces)
    total = sum(f["marks_total"] for f in faces)
    ssim = [f["aa6_ssim"] for f in faces if f["aa6_ssim"] is not None]
    parts = [f"geo {max(geo):.2f}x" if geo else "geo -",
             f"L {max(stops, key=abs):+.2f} stop" if stops else "L -",
             f"marks {kept}/{total}" if total else "marks -",
             f"aa6 ssim {min(ssim):.3f}" if ssim else "aa6 -",
             f"skipped {row.get('skipped', 0)}"]
    flags = sorted({fl for f in faces for fl in f["identity_flags"]})
    return " ".join(parts) + (f" ID={','.join(flags)}" if flags else "")


def check_case_recipe(engine, case, recipe, contexts=None, identity: IdentityCache = None):
    first = render(engine, case.image, recipe, face_contexts=contexts) if contexts is not None \
        else render(engine, case.image, recipe)
    row = {"case": case.name, "recipe": recipe, "persons": case.persons,
           "faces": int(getattr(first, "face_count", 0))}
    ctxs = getattr(first, "face_contexts", None) or []
    if ctxs:
        hold = {**gq.NO_GRAIN, **gq.BODY_OPS_OFF}
        on = render(engine, case.image, recipe, face_contexts=ctxs, **hold)
        off = render(engine, case.image, recipe, face_contexts=ctxs, **gq.skin_ops_off(), **hold)
        row["skin_leak"], row["leak_px"] = gq.skin_leak(on, off, first, include_hair=gq.edits_hair(on))
        objs = case.objects
        row["object_change"] = gq.object_change(on, off, objs, first) if objs is not None else None
    else:
        on = first
        row["skin_leak"], row["leak_px"], row["object_change"] = 0.0, 0, None
    ref = gq.reference_path(case.name, recipe)
    row["ssim"] = gq.ssim(gq.to_u8(first), cv2.imread(str(ref))) if ref.exists() else None
    fails = []
    if row["faces"] != case.persons:
        fails.append("faces")
    if row["skin_leak"] > gq.SKIN_LEAK_MAX:
        fails.append("skin_leak")
    if row["object_change"] is not None and row["object_change"] > gq.OBJECT_CHANGE_MAX:
        fails.append("object")
    row["known"] = []
    if case.known_gap and "faces" in fails:
        row["known"].append("faces")
    if case.object_gap and "object" in fails:
        row["known"].append("object")
    fails = [f for f in fails if f not in row["known"]]
    if row["ssim"] is not None and row["ssim"] < gq.SSIM_MIN:
        fails.append("ssim")
    row["fails"] = fails
    row["skipped"] = max(0, case.persons - row["faces"])
    if identity is not None and ctxs:
        row["identity"] = identity_rows(engine, case, recipe, first, ctxs, identity)
    return row, first, ctxs


def main(argv=None):
    ap = argparse.ArgumentParser(description="Golden hard-case QA sweep")
    ap.add_argument("--recipes", default="", help="comma list (default: all curated)")
    ap.add_argument("--cases", default="", help="comma list (default: all)")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--approve", action="store_true", help="write reference outputs")
    ap.add_argument("--dump", type=Path, default=None, help="write every render here")
    ap.add_argument("--no-identity", action="store_true", help="skip the per-face identity columns")
    ap.add_argument("--no-marks", action="store_true",
                    help="skip mark survival (saves one render per case and recipe)")
    args = ap.parse_args(argv)

    from retouch.engine import RetouchEngine
    from retouch.recipes import CURATED_RECIPE_NAMES

    recipes = [r for r in args.recipes.split(",") if r] or list(CURATED_RECIPE_NAMES)
    cases = gq.load_cases()
    names = [c for c in args.cases.split(",") if c] or sorted(cases)
    engine = RetouchEngine()
    identity = None if args.no_identity else IdentityCache(engine, marks=not args.no_marks)
    rows = []
    t0 = time.time()
    total = len(recipes) * len(names)
    for ci, name in enumerate(names):
        case = cases[name]
        for ri, recipe in enumerate(recipes):
            row, out, _ = check_case_recipe(engine, case, recipe, identity=identity)
            rows.append(row)
            if args.dump:
                args.dump.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.dump / f"{name}__{recipe}.jpg"), gq.to_u8(out))
            if args.approve and recipe in gq.REFERENCE_RECIPES:
                gq.REFERENCES.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(gq.reference_path(name, recipe)), gq.to_u8(out),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
            done = ci * len(recipes) + ri + 1
            if row["fails"] or done % 25 == 0:
                print(f"[{done}/{total} {time.time() - t0:.0f}s] {name} {recipe} "
                      f"faces {row['faces']}/{row['persons']} leak {row['skin_leak']:.4f} "
                      f"obj {row['object_change']} ssim {row['ssim']} FAIL={','.join(row['fails']) or '-'}"
                      f"{' known=' + ','.join(row['known']) if row['known'] else ''}"
                      f"{' | ' + identity_summary(row) if identity is not None else ''}",
                      flush=True)
    failing = sorted({r["recipe"] for r in rows if r["fails"]})
    known = sum(bool(r["known"]) for r in rows)
    print(f"\n{len(rows)} renders, {sum(bool(r['fails']) for r in rows)} failing "
          f"({known} more hit a documented known gap); "
          f"{len(failing)}/{len(recipes)} recipes fail at least one case")
    for recipe in failing:
        bad = [f"{r['case']}:{'+'.join(r['fails'])}" for r in rows if r["recipe"] == recipe and r["fails"]]
        print(f"  {recipe}: {', '.join(bad)}")
    if identity is not None:
        print("\nIdentity (report only; per face: geometry drift > tolerance, |lightness| > "
              f"{gi.LIGHTNESS_MAX_STOPS} stop, a planted mark below {gi.MARK_KEPT_MIN:.0%} of its residual):")
        for recipe in recipes:
            faces = [f for r in rows if r["recipe"] == recipe for f in r.get("identity") or []]
            if not faces:
                continue
            n = {k: sum(k in f["identity_flags"] for f in faces) for k in ("geo", "l", "mark")}
            kept = sum(f["marks_kept"] or 0 for f in faces)
            total = sum(f["marks_total"] for f in faces)
            stops = [f["l_stops"] for f in faces if f["l_stops"] is not None]
            skipped = sum(r.get("skipped", 0) for r in rows if r["recipe"] == recipe)
            print(f"  {recipe}: {len(faces)} faces | geometry {n['geo']} | lightness {n['l']} "
                  f"(median {float(np.median(stops)):+.2f} stop) | marks {kept}/{total} kept, "
                  f"{n['mark']} faces lost one | skipped people {skipped}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1, default=float))
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())

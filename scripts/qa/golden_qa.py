"""Golden hard-case QA sweep: every curated recipe on every tests/golden case.

For each case and recipe it renders the recipe twice with the same detected
faces, once as is and once with every skin op off, and reports:

- faces retouched vs people in the photo,
- skin leak (changed pixels outside the grown skin mask, share of skin area),
- planted object change (clip, rose, frames, poster inside the face zone),
- SSIM against the approved reference (reference recipes only).

The thresholds are the ones tests/test_golden_qa.py enforces in CI; CI
runs a sample, this script runs everything.

Usage:
    RETOUCH_GPU=0 RETOUCH_MEDIAPIPE_BACKEND=legacy \\
        .venv/bin/python scripts/qa/golden_qa.py [--recipes a,b] [--cases x,y] \\
        [--json out.json] [--approve]

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

from tests import golden_qa as gq


def render(engine, img, recipe, **kw):
    np.random.seed(0)  # grain/sparkle use the global RNG
    return engine.process(img, recipe=recipe, **kw)


def check_case_recipe(engine, case, recipe, contexts=None):
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
    return row, first, ctxs


def main(argv=None):
    ap = argparse.ArgumentParser(description="Golden hard-case QA sweep")
    ap.add_argument("--recipes", default="", help="comma list (default: all curated)")
    ap.add_argument("--cases", default="", help="comma list (default: all)")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--approve", action="store_true", help="write reference outputs")
    ap.add_argument("--dump", type=Path, default=None, help="write every render here")
    args = ap.parse_args(argv)

    from retouch.engine import RetouchEngine
    from retouch.recipes import CURATED_RECIPE_NAMES

    recipes = [r for r in args.recipes.split(",") if r] or list(CURATED_RECIPE_NAMES)
    cases = gq.load_cases()
    names = [c for c in args.cases.split(",") if c] or sorted(cases)
    engine = RetouchEngine()
    rows = []
    t0 = time.time()
    total = len(recipes) * len(names)
    for ci, name in enumerate(names):
        case = cases[name]
        for ri, recipe in enumerate(recipes):
            row, out, _ = check_case_recipe(engine, case, recipe)
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
                      f"{' known=' + ','.join(row['known']) if row['known'] else ''}",
                      flush=True)
    failing = sorted({r["recipe"] for r in rows if r["fails"]})
    known = sum(bool(r["known"]) for r in rows)
    print(f"\n{len(rows)} renders, {sum(bool(r['fails']) for r in rows)} failing "
          f"({known} more hit a documented known gap); "
          f"{len(failing)}/{len(recipes)} recipes fail at least one case")
    for recipe in failing:
        bad = [f"{r['case']}:{'+'.join(r['fails'])}" for r in rows if r["recipe"] == recipe and r["fails"]]
        print(f"  {recipe}: {', '.join(bad)}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1, default=float))
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())

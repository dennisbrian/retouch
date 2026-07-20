import os, sys, time, traceback
import cv2
import numpy as np
from retouch.recipes import RECIPES
from retouch.engine import RetouchEngine

SRC = os.path.expanduser("~/Desktop/duotian nikke/DSCF7585.jpg")
OUT = os.path.expanduser("~/Desktop/qa_recipes_DSCF7585")
os.makedirs(OUT, exist_ok=True)

img = cv2.imread(SRC)
assert img is not None, f"cannot read {SRC}"
print(f"source {SRC} shape={img.shape}")

engine = RetouchEngine()
recipes = sorted(RECIPES.keys())
print(f"running {len(recipes)} recipes")

results = {}
total = len(recipes)
for i, name in enumerate(recipes, 1):
    t0 = time.time()
    try:
        out = engine.process(img.copy(), recipe=name)
        arr = np.asarray(out)
        path = os.path.join(OUT, f"{i:03d}_{name}.jpg")
        cv2.imwrite(path, arr)
        dt = time.time() - t0
        results[name] = (True, dt, path)
        print(f"[{i}/{total}] OK   {name:35s} {dt:6.2f}s")
    except Exception as e:
        results[name] = (False, time.time() - t0, repr(e))
        print(f"[{i}/{total}] FAIL {name:35s} {repr(e)}")
        traceback.print_exc()

fails = [n for n,(ok,_,_) in results.items() if not ok]
print(f"\nDONE: {len(recipes)-len(fails)}/{total} ok, {len(fails)} fail")
if fails:
    print("FAILURES:", fails)
for n in fails:
    print("  ", n, results[n][2])

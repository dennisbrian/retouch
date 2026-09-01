"""Render slimming with the yaw gate BYPASSED at several measured yaws, so a
human can judge where the symmetric jaw warp actually starts to look wrong.
Output: test_output/yaw_gate_study/slimming_sheet.jpg (orig | warped per row)."""
from __future__ import annotations
import os, sys, json
from pathlib import Path
os.environ.setdefault("RETOUCH_GPU", "0"); os.environ.setdefault("RETOUCH_MEDIAPIPE_BACKEND", "legacy")
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from retouch import geometry
from retouch.engine import RetouchEngine
OUT = ROOT / "test_output" / "yaw_gate_study"
PICK = sys.argv[1:] or ["DSCF4568", "DSCF4563", "DSCF4574", "DSCF4551", "DSCF7204", "DSCF4599"]
STRENGTH = int(os.environ.get("SLIM", "70")); OP = os.environ.get("OP", "slimming")
geometry.FaceReshaper._yaw_dampen_factor = staticmethod(lambda lm: 1.0)  # bypass gate

def shrink(img, m=2048):
    h, w = img.shape[:2]; s = m / max(h, w)
    return img if s >= 1 else cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

sweep = {r["image"]: r for r in json.load(open(OUT / "sweep.json"))}
eng = RetouchEngine(); tiles = []
for name in PICK:
    img = shrink(cv2.imread(str(ROOT / "test_output" / f"{name}.jpg")))
    base = np.asarray(eng.process(img, recipe="natural", **{OP: 0}))
    warp = np.asarray(eng.process(img, recipe="natural", **{OP: STRENGTH}))
    faces = eng._detector.detect(img)
    x, y, w, h = max(faces, key=lambda f: f.bbox[2]*f.bbox[3]).bbox if faces else (0, 0, img.shape[1], img.shape[0])
    pad = int(0.25 * w); x0, y0 = max(0, x-pad), max(0, y + int(0.35*h)); x1, y1 = min(img.shape[1], x+w+pad), min(img.shape[0], y+h+pad)
    a, b = base[y0:y1, x0:x1], warp[y0:y1, x0:x1]
    sw = sweep[name + ".jpg"]; MODE = "gate off" if OP == "slimming" else "new band"
    d = np.abs(a.astype(int)-b.astype(int)).sum(-1); print(name, "zyaw", sw["zyaw_temple_deg"], "ratio6", sw["ratio_bridge6"], "changed px", int((d>6).sum()), flush=True)
    heat = cv2.applyColorMap(np.clip(d * 3, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    tile = np.hstack([a, b, heat]); tile = cv2.resize(tile, (1500, int(1500*tile.shape[0]/tile.shape[1])))
    cv2.putText(tile, f"{name} zyaw_corr={sw['zyaw_corr_deg']} r6={sw['ratio_bridge6']} {OP}={STRENGTH} {MODE} | orig | warp | diff x3", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    tiles.append(tile)
W = max(t.shape[1] for t in tiles); sheet = np.vstack([cv2.copyMakeBorder(t, 0, 0, 0, W-t.shape[1], cv2.BORDER_CONSTANT) for t in tiles])
cv2.imwrite(str(OUT / f"{OP}_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80]); print("wrote", OUT / f"{OP}_sheet.jpg", sheet.shape)

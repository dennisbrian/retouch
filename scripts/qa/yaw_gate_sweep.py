"""Yaw-gate calibration sweep (2026-09-02).

Audit finding 2 (RESEARCH_FACEOP_VISUAL_AUDIT_2026_08_31 §2): the nose-bridge /
temple x-ratio yaw proxy (lm 6 vs 234/454) zeroes slimming and sculpt on ordinary
portrait poses. Same shape as the EAR miscalibration: good signal, wrong band.

For every corpus face this records:
  * the shipped ratio (lm 6) and the same ratio from other midline points
    (nose tip 1, nasion 168, chin 152, forehead 10) — protrusion sensitivity;
  * eye-width ratio (33-133 vs 362-263) — the audit's "mild turn" reference;
  * 3D yaw angle from landmark depth (MediaPipe z, same scale as x) on the
    temple, cheek and outer-eye-corner vectors — an independent yaw estimate;
  * what each shipped band does today: geometry smoothstep 1.3→1.6, sculpt
    linear 1.5→1.7, skin neck hard-gate 1.3.

Images are shrunk to max-dim 2048 (proxy scale; ratios are scale-invariant).
Run with .venv/bin/python (mediapipe 0.10.5, legacy backend).
"""
from __future__ import annotations
import json, math, os, sys, time
from pathlib import Path
os.environ.setdefault("RETOUCH_GPU", "0")
os.environ.setdefault("RETOUCH_MEDIAPIPE_BACKEND", "legacy")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from retouch.detection import FaceDetector  # noqa: E402
from retouch.geometry import _YAW_DAMPEN_START, _YAW_DAMPEN_END  # noqa: E402

CORPUS = ROOT / "test_output" / "detection_recall_study" / "corpus.txt"
OUT = ROOT / "test_output" / "yaw_gate_study"
MAX_DIM = 2048
MIDLINE = {"bridge6": 6, "tip1": 1, "nasion168": 168, "chin152": 152, "forehead10": 10}
LATERAL = {"temple": (234, 454), "cheek": (93, 323), "eye_outer": (33, 263)}


def shrink(img):
    h, w = img.shape[:2]; s = MAX_DIM / max(h, w)
    return img if s >= 1 else cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def ratio(lm, mid, l, r):
    dl = abs(lm[mid].x - lm[l].x); dr = abs(lm[r].x - lm[mid].x)
    return max(dl, dr) / (min(dl, dr) + 1e-5)


def smoothstep_damp(rv, a=_YAW_DAMPEN_START, b=_YAW_DAMPEN_END):
    if rv <= a: return 1.0
    if rv >= b: return 0.0
    t = (rv - a) / (b - a); return t * t * (3 - 2 * t)


def z_yaw_deg(lm, l, r, w, h):
    # MediaPipe legacy z is normalised on the same scale as x (image width).
    dx = (lm[r].x - lm[l].x); dz = (lm[r].z - lm[l].z)
    return math.degrees(math.atan2(dz, dx))


# Generic 6-point head model (classic solvePnP head-pose set; mm, arbitrary origin).
_PNP_MODEL = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
                       (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)], np.float64)
_PNP_IDX = (1, 152, 33, 263, 61, 291)  # nose tip, chin, L/R outer eye, L/R mouth corner


def pnp_yaw_deg(lm, w, h):
    """Yaw from 2D pixel landmarks only (independent of MediaPipe z and of crop remap)."""
    pts = np.array([(lm[i].x * w, lm[i].y * h) for i in _PNP_IDX], np.float64)
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], np.float64)
    ok, rvec, _ = cv2.solvePnP(_PNP_MODEL, pts, K, np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return float("nan")
    R, _ = cv2.Rodrigues(rvec)
    return math.degrees(math.atan2(-R[2, 0], math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))


def _tag_remap():
    """Tag remapped landmarks with the crop→full x-scale so z (left in crop units
    by FaceDetector._remap_landmarks) can be rescaled to the same units as x."""
    orig = FaceDetector._remap_landmarks
    def tagged(landmarks, ox, oy, cw, ch, full_w, full_h):
        out = orig(landmarks, ox, oy, cw, ch, full_w, full_h)
        for l in out: l._zscale = cw / full_w
        return out
    FaceDetector._remap_landmarks = staticmethod(tagged)


def main():
    OUT.mkdir(parents=True, exist_ok=True); _tag_remap()
    det = FaceDetector(max_faces=25, min_confidence=0.4)
    rows = []; t0 = time.time()
    for line in CORPUS.read_text().splitlines():
        p = ROOT / line.strip()
        if not p.exists(): continue
        img = shrink(cv2.imread(str(p))); h, w = img.shape[:2]
        faces = det.detect(img)
        if not faces: rows.append({"image": p.name, "faces": 0}); continue
        f = max(faces, key=lambda f: f.bbox[2] * f.bbox[3]); lm = f.landmarks.landmark if hasattr(f.landmarks, "landmark") else f.landmarks
        r = {"image": p.name, "faces": len(faces), "bbox_w": f.bbox[2], "ied": round(f.ied, 1),
             "z_present": bool(abs(lm[6].z) + abs(lm[234].z) > 1e-6)}
        for name, idx in MIDLINE.items():
            r[f"ratio_{name}"] = round(ratio(lm, idx, 234, 454), 3)
        r["ratio_bridge6_cheek"] = round(ratio(lm, 6, 93, 323), 3)
        ewl = abs(lm[33].x - lm[133].x); ewr = abs(lm[362].x - lm[263].x)
        r["eye_width_ratio"] = round(max(ewl, ewr) / (min(ewl, ewr) + 1e-5), 3)
        zs = getattr(lm[0], "_zscale", 1.0); r["remapped"] = zs != 1.0; r["zscale"] = round(zs, 3)
        for name, (l, rr) in LATERAL.items():
            r[f"zyaw_{name}_deg"] = round(z_yaw_deg(lm, l, rr, w, h), 1)
        dx = lm[454].x - lm[234].x; dz = (lm[454].z - lm[234].z) * zs
        r["zyaw_corr_deg"] = round(math.degrees(math.atan2(dz, dx)), 1)
        r["pnp_yaw_deg"] = round(pnp_yaw_deg(lm, w, h), 1)
        rb = r["ratio_bridge6"]
        r["damp_geometry"] = round(smoothstep_damp(rb), 3)
        r["damp_sculpt"] = round(1 - min(max((rb - 1.5) / 0.2, 0), 1), 3)
        r["neck_gate_open"] = rb <= 1.3
        rows.append(r)
        print(f"{p.name:14s} ratio6={rb:5.2f} tip={r['ratio_tip1']:5.2f} nasion={r['ratio_nasion168']:5.2f} "
              f"eyeW={r['eye_width_ratio']:4.2f} zyaw_t={r['zyaw_temple_deg']:6.1f} zyaw_c={r['zyaw_cheek_deg']:6.1f} "
              f"zcorr={r['zyaw_corr_deg']:6.1f}{'R' if r['remapped'] else ' '} dampG={r['damp_geometry']:.2f}", flush=True)
    (OUT / "sweep.json").write_text(json.dumps(rows, indent=1))
    det_rows = [r for r in rows if r.get("faces")]
    print(f"\n{len(det_rows)}/{len(rows)} faces in {time.time()-t0:.0f}s; z_present on {sum(r['z_present'] for r in det_rows)}")
    for key, thr in (("damp_geometry", 1.0), ("damp_geometry", 0.5), ("damp_sculpt", 1.0), ("damp_sculpt", 0.5)):
        n = sum(1 for r in det_rows if r[key] < thr); print(f"{key} < {thr}: {n}/{len(det_rows)}")
    print("neck gate closed:", sum(1 for r in det_rows if not r["neck_gate_open"]), "/", len(det_rows))


if __name__ == "__main__":
    main()

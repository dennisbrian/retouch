"""OFAT sensitivity screening: which parameters actually move face pixels?

Whole-frame SSIM dilutes face-local edits to nothing, so every metric here
is restricted to the face-region mask. A parameter that cannot clear the
threshold on a cropped face at max strength is inert for sweep purposes.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import cv2, numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CORPUS = Path("/Users/dennis/Library/CloudStorage/GoogleDrive-alexanderlooi88@gmail.com"
              "/My Drive/Photography/Remielle/Photos/Total")

CANDIDATES = {
    "smooth": (0.0, 1.0), "skin_unify": (0.0, 1.0), "skin_unify_hue": (0.0, 1.0),
    "skin_hue_unify": (0.0, 1.0), "relight": (0.0, 1.0), "sharpen": (0.0, 1.0),
    "whiten": (0.0, 1.0), "blemish": (0.0, 1.0), "dark_circles": (0.0, 1.0),
    "eye_enhance": (0.0, 1.0), "lip_enhance": (0.0, 1.0), "clarity": (0.0, 1.0),
    "texture_opacity": (0.0, 1.0), "contrast": (0.0, 1.0), "saturation": (0.0, 1.0),
    "vibrance": (0.0, 1.0), "glow": (0.0, 1.0), "shine_removal": (0.0, 1.0),
    "face_exposure": (0.0, 1.0), "sculpt": (0.0, 1.0),
}


def face_crop(img):
    from retouch.detection import FaceDetector
    faces = FaceDetector().detect(img)
    if not faces:
        return None
    x, y, w, h = faces[0].bbox
    pad = int(max(w, h) * 1.1)
    cx, cy = x + w // 2, y + h // 2
    return img[max(0, cy-pad):cy+pad, max(0, cx-pad):cx+pad]


def face_mask(crop):
    """Skin-region mask from the landmark fixture path (ONNX-free)."""
    sys.path.insert(0, str(REPO / "tests"))
    from retouch.detection import FaceDetector
    from retouch.parsing import FaceParser
    faces = FaceDetector().detect(crop)
    if not faces:
        return None
    parser = FaceParser(); parser._sess = None
    regions = parser._landmark_fallback_only(
        faces[0].landmarks, crop, person_mask=None, ied=faces[0].ied)
    m = getattr(regions, "skin", None)
    if m is None:
        return None
    return (np.asarray(m) > 0.5)


def masked_diff(a, b, mask):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).mean(axis=2)
    return float(d[mask].mean()) if mask is not None and mask.any() else float(d.mean())


def main():
    from retouch.engine import RetouchEngine
    img = cv2.imread(str(sorted(CORPUS.glob("*.jpg"))[0]))
    img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    crop = face_crop(img)
    mask = face_mask(crop)
    cov = float(mask.mean()) if mask is not None else float("nan")
    print(f"crop={crop.shape} skin_mask_coverage={cov:.3f}")

    e = RetouchEngine()
    base = np.asarray(e.process(crop, recipe="natural"))
    rows = []
    for p, (lo, hi) in CANDIDATES.items():
        try:
            o_lo = np.asarray(e.process(crop, recipe="natural", **{p: lo}))
            o_hi = np.asarray(e.process(crop, recipe="natural", **{p: hi}))
        except Exception as ex:
            rows.append((p, None, f"{type(ex).__name__}"))
            continue
        # sensitivity = how far min-strength and max-strength diverge
        rows.append((p, masked_diff(o_lo, o_hi, mask), ""))

    rows.sort(key=lambda r: (r[1] is None, -(r[1] or 0)))
    print(f"\n{'param':22s} {'masked |lo-hi| diff':>20s}   note")
    for p, v, note in rows:
        print(f"{p:22s} {('%.5f'%v) if v is not None else 'ERR':>20s}   {note}")

    live = [p for p, v, _ in rows if v is not None and v >= 0.05]
    inert = [p for p, v, _ in rows if v is not None and v < 0.05]
    print(f"\nLIVE  (>=0.05): {live}")
    print(f"INERT (<0.05) : {inert}")
    Path(__file__).parent.joinpath("screening.json").write_text(json.dumps(
        {"live": live, "inert": inert,
         "raw": {p: v for p, v, _ in rows}, "skin_mask_coverage": cov}, indent=2))


if __name__ == "__main__":
    main()

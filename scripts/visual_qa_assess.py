"""Numeric Visual-QA: artifact localization + synthetic functional proof.

Cannot rely on eyeballing (no vision in session), so:
 1. backdrop / fabric: split frame into SUBJECT (central) vs BACKGROUND
    (border strips). A correct feature changes background, NOT subject.
 2. wrinkle / sclera: draw synthetic defects on a real render, then run the
    feature and confirm it attenuated them (proves the op works end-to-end).
"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retouch.engine import RetouchEngine

SRC = "test_output/masterwork_v1/DSCF8007.jpg"
OUT = "test_output/visual_qa"


def load():
    img = cv2.imread(SRC, cv2.IMREAD_COLOR)
    md = 1280
    h, w = img.shape[:2]
    if max(h, w) > md:
        s = md / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), cv2.INTER_AREA)
    return img


def bg_strips(img):
    h, w = img.shape[:2]
    return [img[0:int(h*0.12)], img[int(h*0.88):h],
            img[:, 0:int(w*0.12)], img[:, int(w*0.88):w]]


def split(img):
    h, w = img.shape[:2]
    subj = img[int(h*0.2):int(h*0.8), int(w*0.25):int(w*0.75)]
    return subj, bg_strips(img)


def mae(a, b):
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def main():
    img = load()
    eng = RetouchEngine()
    base = eng.process(img)

    print("== LOCALIZATION (subject vs background) ==")
    for name, kw in [("backdrop", dict(backdrop_cleanup=60)),
                     ("fabric", dict(fabric_wrinkle_smooth=60))]:
        on = eng.process(img, **kw)
        s_b, s_on = split(base)[0], split(on)[0]
        gb0, gb1 = split(base)[1], split(on)[1]
        g_b = np.concatenate([s.ravel() for s in gb0])
        g_on = np.concatenate([s.ravel() for s in gb1])
        print(f"{name:9s} subject MAE={mae(s_b, s_on):.4f}  background MAE={mae(g_b, g_on):.4f}")

    print("\n== FUNCTIONAL (synthetic defects on real render) ==")
    # wrinkles: dark horizontal-ish lines on forehead band of the BASE render
    wr = base.copy()
    h, w = wr.shape[:2]
    fy1, fy2 = int(h*0.18), int(h*0.30)
    for yy in range(fy1, fy2, 6):
        cv2.line(wr, (int(w*0.35), yy), (int(w*0.65), yy+3), (60, 60, 60), 1)
    out_wr = eng.process(wr, wrinkle_soften_forehead=80)
    # measure mean L in the wrinkled band before/after
    lab_b = cv2.cvtColor(wr, cv2.COLOR_BGR2LAB).astype(np.float32)[:, :, 0]
    lab_a = cv2.cvtColor(out_wr, cv2.COLOR_BGR2LAB).astype(np.float32)[:, :, 0]
    band_b = lab_b[fy1:fy2, int(w*0.35):int(w*0.65)]
    band_a = lab_a[fy1:fy2, int(w*0.35):int(w*0.65)]
    print(f"wrinkle forehead: band mean L before={band_b.mean():.1f} after={band_a.mean():.1f} "
          f"(raised={band_a.mean() > band_b.mean()})  whole MAE={mae(wr, out_wr):.3f}")

    # sclera vessels: red squiggles inside an eye region (use landmarks via parser)
    from retouch.parsing import FaceParser
    parser = FaceParser()
    # need a face bbox + landmarks: reuse engine internals lightly
    from retouch.detection import RetinaFaceDetector
    det = RetinaFaceDetector()
    faces = det.detect(img)
    if faces:
        f = faces[0]
        # landmarks via mediapipe through engine? use parser with dummy landmarks not possible.
        # Instead place vessels near eye using retinaface bbox eyes approx.
        x, y, fw, fh = f["box"]
        eye_y = int(y + fh*0.42)
        eye_lx = int(x + fw*0.35)
        eye_rx = int(x + fw*0.65)
        vr = base.copy()
        for (ex,) in [(eye_lx,), (eye_rx,)]:
            cv2.line(vr, (ex-8, eye_y), (ex+8, eye_y+2), (40, 30, 120), 1)
            cv2.line(vr, (ex-6, eye_y+4), (ex+6, eye_y+6), (40, 30, 120), 1)
        out_vr = eng.process(vr, eye_sclera_vessel_remove=80, eye_enhance=20)
        # redness = a-channel deviation; compare inside small eye patch
        def redness(im, cx, cy):
            lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
            a = lab[cy-6:cy+6, cx-8:cx+8, 1]
            b = lab[cy-6:cy+6, cx-8:cx+8, 2]
            return float(np.mean(np.sqrt(a**2 + b**2)))
        rb = redness(vr, eye_lx, eye_y); ra = redness(out_vr, eye_lx, eye_y)
        print(f"sclera vessel: eye redness(before={rb:.1f} after={ra:.1f}) reduced={ra < rb}  whole MAE={mae(vr, out_vr):.3f}")
        cv2.imwrite(os.path.join(OUT, "synthetic_wrinkle_on.jpg"), out_wr)
        cv2.imwrite(os.path.join(OUT, "synthetic_vessel_on.jpg"), out_vr)
    else:
        print("sclera: no face detected for synthetic test")

    print("\nDone.")


if __name__ == "__main__":
    main()

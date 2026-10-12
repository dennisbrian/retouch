"""Focused eye QA: render eye-region zoom crops for the eye features.

- Real-photo eye zoom for eye_enhance / catchlight / whiten / iris (visible).
- Synthetic red vessel overlay on the sclera to PROVE eye_sclera_vessel_remove
  attenuates (real eyes often have no visible vessels, so the live op is a
  near-no-op — the synthetic proof shows the mechanism works).
Outputs: test_output/visual_qa_eyes/
"""
import os, sys, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retouch.engine import RetouchEngine
from retouch.detection import FaceDetector

JPEG = "test_output/masterwork_v1/DSCF8007.jpg"
RAF = "/Users/dennis/Pictures/2025/2025-08-09/_DSF1853.RAF"
OUT = "test_output/visual_qa_eyes"
MAXD = 1280


def load(path, is_raf=False):
    if is_raf:
        import rawpy
        raw = rawpy.imread(path)
        rgb = raw.postprocess(use_camera_wb=True, output_color=rawpy.ColorSpace.sRGB,
                               highlight_mode=rawpy.HighlightMode.ReconstructDefault)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    if max(h, w) > MAXD:
        s = MAXD / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), cv2.INTER_AREA)
    return img


def eye_boxes(lm, w, h, pad=2.2):
    """Return (left_box, right_box) around iris landmarks 468/473."""
    pts = [(int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)) for i in (468, 473)]
    boxes = []
    for (cx, cy) in pts:
        r = int(w * 0.05 * pad)
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)
        boxes.append((x1, y1, x2, y2))
    return boxes[0], boxes[1]


def crop(img, box):
    x1, y1, x2, y2 = box
    return img[y1:y2, x1:x2]


def montage(base, on, title):
    diff = np.clip(np.abs(on.astype(np.float32) - base.astype(np.float32)) * 4.0, 0, 255).astype(np.uint8)
    top = np.hstack([base, on])
    bottom = np.hstack([diff, np.zeros_like(diff)])
    m = np.vstack([top, np.zeros((8, top.shape[1], 3), np.uint8), bottom])
    cv2.putText(m, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(m, "base | on        | amplified diff", (10, m.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    return m


def draw_vessels(img, box):
    """Draw synthetic red squiggles on the sclera (outside the iris) for proof."""
    x1, y1, x2, y2 = box
    out = img.copy()
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    for dx in (-int((x2 - x1) * 0.28), int((x2 - x1) * 0.28)):
        for dy in (-6, 6):
            px, py = cx + dx, cy + dy
            cv2.line(out, (px - 10, py), (px + 10, py + 3), (40, 30, 120), 1)
            cv2.line(out, (px - 6, py + 5), (px + 6, py + 7), (40, 30, 120), 1)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    eng = RetouchEngine()
    cases = {
        "eye_enhance": dict(eye_enhance=60),
        "eye_full": dict(eye_enhance=40, eye_sclera_vessel_remove=60, catchlight=0.4,
                         whiten=0.3),
        "sclera_vessel": dict(eye_enhance=20, eye_sclera_vessel_remove=80),
    }
    for src_name, path, is_raf in [("jpeg", JPEG, False), ("raf", RAF, True)]:
        img = load(path, is_raf)
        det = FaceDetector()
        faces = det.detect(img)
        if not faces:
            print(f"{src_name}: no face detected, skipping"); continue
        f = faces[0]
        lbox, rbox = eye_boxes(f.landmarks, img.shape[1], img.shape[0])
        base = eng.process(img)
        cv2.imwrite(os.path.join(OUT, f"{src_name}_baseline.jpg"), base)
        for name, kw in cases.items():
            out = eng.process(img, **kw)
            # synthetic vessel proof on THIS source
            vimg = draw_vessels(img, lbox)
            vout = eng.process(vimg, eye_enhance=20, eye_sclera_vessel_remove=90)
            # save eye crops + montages
            for tag, b, o in [("L", base, out), ("R", base, out)]:
                box = lbox if tag == "L" else rbox
                bm = montage(crop(b, box), crop(o, box), f"{src_name} {name} eye{tag}")
                cv2.imwrite(os.path.join(OUT, f"{src_name}_{name}_eye{tag}_montage.jpg"), bm)
            # vessel proof montage (left eye)
            vm = montage(crop(vimg, lbox), crop(vout, lbox), f"{src_name} sclera vessel PROOF")
            cv2.imwrite(os.path.join(OUT, f"{src_name}_sclera_vessel_proof_L.jpg"), vm)
            cv2.imwrite(os.path.join(OUT, f"{src_name}_{name}_on.jpg"), out)
            d = float(np.mean(np.abs(out.astype(np.float32) - base.astype(np.float32))))
            print(f"{src_name:5s} {name:14s} OK mean_delta={d:.3f}")
    print(f"\nOutputs in {OUT}/")


if __name__ == "__main__":
    main()

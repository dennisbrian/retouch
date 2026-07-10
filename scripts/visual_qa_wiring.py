"""Render the wiring-debt fix recipes to visible before/after montages.

T3 body_reshape, A3 cosplay, and face_exposure were previously unreachable
via recipes. Render each (vs baseline) on the JPEG reference + RAF so the
user can eyeball them. Outputs in test_output/visual_qa_wiring/.
"""
import os, sys, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retouch.engine import RetouchEngine

JPEG = "test_output/masterwork_v1/DSCF8007.jpg"
RAF = "/Users/dennis/Pictures/2025/2025-08-09/_DSF1853.RAF"
OUT = "test_output/visual_qa_wiring"


def load(path, is_raf=False, maxd=1280):
    if is_raf:
        import rawpy
        raw = rawpy.imread(path)
        rgb = raw.postprocess(use_camera_wb=True, output_color=rawpy.ColorSpace.sRGB,
                               highlight_mode=rawpy.HighlightMode.ReconstructDefault)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    if max(h, w) > maxd:
        s = maxd / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), cv2.INTER_AREA)
    return img


def montage(base, on, title):
    diff = np.clip(np.abs(on.astype(np.float32) - base.astype(np.float32)) * 4.0, 0, 255).astype(np.uint8)
    top = np.hstack([base, on])
    bottom = np.hstack([diff, np.zeros_like(diff)])
    m = np.vstack([top, np.zeros((8, top.shape[1], 3), np.uint8), bottom])
    cv2.putText(m, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(m, "base | on        | amplified diff", (10, m.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    eng = RetouchEngine()
    recipes = {
        "portrait_face_exposure": dict(recipe="portrait"),
        "body_reshape_demo": dict(recipe="body_reshape_demo_v1"),
        "cosplay_wiring_demo": dict(recipe="cosplay_wiring_demo_v1"),
    }
    for src_name, path, is_raf in [("jpeg", JPEG, False), ("raf", RAF, True)]:
        img = load(path, is_raf)
        base = eng.process(img)
        cv2.imwrite(os.path.join(OUT, f"{src_name}_baseline.jpg"), base)
        for name, kw in recipes.items():
            try:
                out = eng.process(img, **kw)
                cv2.imwrite(os.path.join(OUT, f"{src_name}_{name}_on.jpg"), out)
                m = montage(base, out, f"{src_name} {name}")
                cv2.imwrite(os.path.join(OUT, f"{src_name}_{name}_montage.jpg"), m)
                delta = float(np.mean(np.abs(out.astype(np.float32) - base.astype(np.float32))))
                print(f"{src_name:5s} {name:22s} OK  mean_delta={delta:.3f}")
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"{src_name:5s} {name:22s} FAILED: {e}")
    print(f"\nOutputs in {OUT}/")


if __name__ == "__main__":
    main()

"""Visual QA for the "everything auto" backlog features (#1-#4, #6, #7).

Renders each new feature ON vs a clean baseline on the real reference photo
`test_output/masterwork_v1/DSCF8007.jpg` and writes before/after montages to
`test_output/visual_qa/`. Run: `python3 scripts/visual_qa_backlog.py`
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retouch.engine import RetouchEngine  # noqa: E402

SRC = "test_output/masterwork_v1/DSCF8007.jpg"
OUT = "test_output/visual_qa"
STRENGTH = 60


def load() -> np.ndarray:
    img = cv2.imread(SRC, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {SRC}")
    # downscale for fast artifact QA (features are resolution-relative)
    maxd = 1280
    h, w = img.shape[:2]
    if max(h, w) > maxd:
        s = maxd / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def center_crop(img: np.ndarray, fw: float = 0.55, fh: float = 0.62) -> np.ndarray:
    h, w = img.shape[:2]
    cw, ch = int(w * fw), int(h * fh)
    x1, y1 = (w - cw) // 2, max(0, (h - ch) // 2 - int(h * 0.04))
    return img[y1:y1 + ch, x1:x1 + cw]


def montage(base: np.ndarray, on: np.ndarray, title: str) -> np.ndarray:
    # amplified abs-diff to reveal subtle background/fabric changes
    diff = np.clip((np.abs(on.astype(np.float32) - base.astype(np.float32)) * 4.0), 0, 255).astype(np.uint8)
    top = np.hstack([base, on])
    bottom = np.hstack([diff, np.zeros_like(diff)])
    pad = np.zeros((8, top.shape[1], 3), dtype=np.uint8)
    m = np.vstack([top, pad, bottom])
    # title via simple text
    cv2.putText(m, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(m, "base | on        | amplified diff", (10, m.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    return m


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    img = load()
    engine = RetouchEngine()

    # baseline
    t0 = time.time()
    base = engine.process(img)
    print(f"baseline rendered in {time.time() - t0:.1f}s")
    cv2.imwrite(os.path.join(OUT, "baseline.jpg"), base)
    cv2.imwrite(os.path.join(OUT, "baseline_zoom.jpg"), center_crop(base))
    cv2.imwrite(os.path.join(OUT, "src_zoom.jpg"), center_crop(img))

    cases = {
        "sclera_vessel": dict(eye_enhance=20, eye_sclera_vessel_remove=STRENGTH),
        "backdrop": dict(backdrop_cleanup=STRENGTH),
        "fabric": dict(fabric_wrinkle_smooth=STRENGTH),
        "wrinkle_forehead": dict(wrinkle_soften_forehead=STRENGTH),
        "wrinkle_nasolabial": dict(wrinkle_soften_nasolabial=STRENGTH),
        "wrinkle_neck": dict(wrinkle_soften_neck=STRENGTH),
        "reshape_jaw_lr": dict(reshape_jaw_width_l=40, reshape_jaw_width_r=-20),
        "reshape_nose_lr": dict(reshape_nose_width_l=30, reshape_nose_width_r=-15),
        "reshape_eye_lr": dict(reshape_eye_size_l=25, reshape_eye_size_r=-10),
        "reshape_neck": dict(reshape_neck_width=30, reshape_neck_length=20),
        "auto_body": dict(auto_body_reshape=STRENGTH),
    }

    for name, kwargs in cases.items():
        try:
            t0 = time.time()
            out = engine.process(img, **kwargs)
            dt = time.time() - t0
            cv2.imwrite(os.path.join(OUT, f"{name}_on.jpg"), out)
            cv2.imwrite(os.path.join(OUT, f"{name}_on_zoom.jpg"), center_crop(out))
            m = montage(center_crop(base), center_crop(out), name)
            cv2.imwrite(os.path.join(OUT, f"{name}_montage.jpg"), m)
            # numeric delta in whole frame (mean abs diff)
            delta = float(np.mean(np.abs(out.astype(np.float32) - base.astype(np.float32))))
            print(f"{name:20s} OK {dt:5.1f}s  mean|Δ|={delta:.3f}  kwargs={kwargs}")
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"{name:20s} FAILED: {e}")

    # combined: everything auto at once
    try:
        combo = dict(
            eye_sclera_vessel_remove=40, backdrop_cleanup=50, fabric_wrinkle_smooth=50,
            wrinkle_soften_forehead=40, wrinkle_soften_nasolabial=30,
            reshape_jaw_width_l=20, reshape_jaw_width_r=-10, reshape_neck_width=20,
            auto_body_reshape=40,
        )
        out = engine.process(img, **combo)
        cv2.imwrite(os.path.join(OUT, "combo_on.jpg"), out)
        cv2.imwrite(os.path.join(OUT, "combo_on_zoom.jpg"), center_crop(out))
        m = montage(center_crop(base), center_crop(out), "combo everything-auto")
        cv2.imwrite(os.path.join(OUT, "combo_montage.jpg"), m)
        print("combo OK")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"combo FAILED: {e}")

    print(f"\nOutputs in {OUT}/")


if __name__ == "__main__":
    main()

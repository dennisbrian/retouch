"""C3 spike — soft BiSeNet logits vs binary+Gaussian vs guided-filter feathering.

Research-only. Does NOT touch engine code. Outputs:
  - printed measurements (uncertainty-band widths, calibration stats, timings, memory)
  - test_output/spike_c3_<img>_<region>.png side-by-side visual comparisons

Method mirrors retouch/parsing.py::FaceParser.parse exactly for preprocessing
(30% padded bbox crop -> 512x512 -> ImageNet-normalized -> ONNX 'input').
"""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retouch.detection import FaceDetector
from retouch.parsing import FaceParser
from retouch.utils import feather_mask

TEST_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_output")

# BiSeNet CelebAMask-HQ class indices (as used in parsing.py)
CLS_BG, CLS_SKIN, CLS_LBROW, CLS_RBROW, CLS_LEYE, CLS_REYE = 0, 1, 2, 3, 4, 5
CLS_NOSE, CLS_MOUTH, CLS_ULIP, CLS_LLIP = 10, 11, 12, 13
CLS_NECK, CLS_CLOTH, CLS_HAIR = 14, 16, 17


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax over axis 0. logits: (19, H, W)."""
    m = logits.max(axis=0, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=0, keepdims=True)


def load_and_downscale(path: str, max_dim: int = 1600) -> np.ndarray:
    img = cv2.imread(path)
    h, w = img.shape[:2]
    s = max_dim / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def run_bisenet(parser: FaceParser, img_bgr: np.ndarray, bbox):
    """Replicate parsing.py preprocessing; return logits (19,512,512) + crop box."""
    h_img, w_img = img_bgr.shape[:2]
    x, y, w, h = bbox
    pad_x, pad_y = int(w * 0.3), int(h * 0.3)
    cx1, cy1 = max(0, x - pad_x), max(0, y - pad_y)
    cx2, cy2 = min(w_img, x + w + pad_x), min(h_img, y + h + pad_y)
    crop = img_bgr[cy1:cy2, cx1:cx2]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_resized = cv2.resize(crop_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
    crop_f = crop_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    crop_input = np.transpose((crop_f - mean) / std, (2, 0, 1))[np.newaxis]
    outs = parser._sess.run(None, {"input": crop_input})
    logits = outs[0][0]  # (19, 512, 512)
    return logits, (cx1, cy1, cx2, cy2), crop


def boundary_between(pred: np.ndarray, cls_a, cls_b) -> np.ndarray:
    """Pixels on the boundary between class-set a and class-set b (bool, 512x512)."""
    a = np.isin(pred, cls_a).astype(np.uint8)
    b = np.isin(pred, cls_b).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    return (cv2.dilate(a, k) & cv2.dilate(b, k) & ((a | b) > 0)).astype(bool)


def band_width_stats(pred, maxprob, cls_a, cls_b, corridor=15, thresh=0.9):
    """Estimate uncertainty band width (px in 512-space) along an A|B boundary.

    Width = (# uncertain px within `corridor` px of the boundary) / boundary length.
    """
    bnd = boundary_between(pred, cls_a, cls_b)
    n_bnd = int(bnd.sum())
    if n_bnd < 20:
        return None
    dist = cv2.distanceTransform((~bnd).astype(np.uint8), cv2.DIST_L2, 3)
    corridor_mask = dist <= corridor
    uncertain = (maxprob < thresh) & corridor_mask
    # also restrict uncertain px to those whose argmax is in {a,b} (avoid counting
    # nearby other-class confusion into this boundary's number)
    in_ab = np.isin(pred, list(cls_a) + list(cls_b))
    uncertain &= in_ab
    width = uncertain.sum() / max(n_bnd / 2.0, 1.0)  # boundary counted from both sides
    probs_at_bnd = maxprob[bnd]
    return {
        "boundary_px": n_bnd,
        "band_width_px512": float(width),
        "maxprob_at_boundary_median": float(np.median(probs_at_bnd)),
        "maxprob_at_boundary_p25": float(np.percentile(probs_at_bnd, 25)),
        "frac_uncertain_in_corridor": float(uncertain.sum() / max(corridor_mask.sum(), 1)),
    }


def build_masks_at_crop_res(logits, prob, pred, cls_set, crop_bgr, feather_r):
    """Build the 3 competing masks at crop resolution (cw x ch).

    Returns dict: current (binary->NEAREST->Gaussian), soft (prob->LINEAR),
    guided (binary->LINEAR->guidedFilter vs image).
    """
    ch, cw = crop_bgr.shape[:2]
    binary512 = np.isin(pred, cls_set).astype(np.float32)
    prob512 = prob[list(cls_set)].sum(axis=0).astype(np.float32)

    t0 = time.perf_counter()
    # (a) CURRENT: NEAREST upsample of argmax binary, then Gaussian feather
    cur = cv2.resize(binary512.astype(np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    cur = feather_mask(cur, radius=feather_r)
    t_cur = time.perf_counter() - t0

    t0 = time.perf_counter()
    # (b) SOFT: per-class probability, bilinear upsample, no blur
    soft = cv2.resize(prob512, (cw, ch), interpolation=cv2.INTER_LINEAR)
    t_soft = time.perf_counter() - t0

    t0 = time.perf_counter()
    # (c) GUIDED: binary upsampled bilinear, guided-filtered against the image
    gsrc = cv2.resize(binary512, (cw, ch), interpolation=cv2.INTER_LINEAR)
    guide = crop_bgr.astype(np.float32) / 255.0
    try:
        gd = cv2.ximgproc.guidedFilter(guide, gsrc, radius=feather_r, eps=1e-3)
    except Exception:
        from retouch.utils import guided_filter as gf
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gd = gf(gsrc, radius=feather_r, eps=1e-3, guide=gray, max_dim=None)
    gd = np.clip(gd, 0.0, 1.0)
    t_gd = time.perf_counter() - t0

    # (d) HYBRID: soft prob refined by guided filter (best-of-both candidate)
    t0 = time.perf_counter()
    try:
        hyb = cv2.ximgproc.guidedFilter(guide, soft, radius=max(feather_r // 2, 2), eps=1e-3)
    except Exception:
        hyb = soft
    hyb = np.clip(hyb, 0.0, 1.0)
    t_hyb = time.perf_counter() - t0

    return {"current": cur, "soft": soft, "guided": gd, "soft+guided": hyb}, \
           {"current": t_cur, "soft": t_soft, "guided": t_gd, "soft+guided": t_hyb}


def find_region_crop(pred, cls_set, crop_shape, want="top"):
    """Pick a zoom window (in crop coords) centered on the class boundary.

    want='top' -> upper edge of the class region (hairline for skin);
    want='center' -> bbox center (lips).
    """
    ch, cw = crop_shape[:2]
    m = np.isin(pred, cls_set)
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    sy, sx = ch / 512.0, cw / 512.0
    if want == "top":
        y_edge = int(np.percentile(ys, 1) * sy)
        x_c = int(np.median(xs[ys < np.percentile(ys, 8)]) * sx)
    else:
        y_edge = int((ys.min() + ys.max()) / 2 * sy)
        x_c = int((xs.min() + xs.max()) / 2 * sx)
    half = max(int(0.11 * max(ch, cw)), 70)
    y1, y2 = max(0, y_edge - half), min(ch, y_edge + half)
    x1, x2 = max(0, x_c - half), min(cw, x_c + half)
    return y1, y2, x1, x2


def render_panel(crop_bgr, masks, zoom, out_path, title):
    """Side-by-side: [image | current | soft | guided | soft+guided] as
    two rows: top = mask grayscale, bottom = image*mask composite (halo test)."""
    y1, y2, x1, x2 = zoom
    img_z = crop_bgr[y1:y2, x1:x2]
    hz, wz = img_z.shape[:2]
    scale = max(1, int(360 / max(hz, 1)))
    big = lambda a: cv2.resize(a, (wz * scale, hz * scale), interpolation=cv2.INTER_NEAREST)

    label_h = 26
    panels_top, panels_bot, labels = [], [], []
    panels_top.append(big(img_z))
    panels_bot.append(big(img_z))
    labels.append("image")
    for name, m in masks.items():
        mz = m[y1:y2, x1:x2]
        g = (np.clip(mz, 0, 1) * 255).astype(np.uint8)
        panels_top.append(big(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)))
        comp = (img_z.astype(np.float32) * mz[..., None]).astype(np.uint8)
        panels_bot.append(big(comp))
        labels.append(name)

    def strip(panels):
        return np.concatenate(panels, axis=1)

    top, bot = strip(panels_top), strip(panels_bot)
    W = top.shape[1]
    header = np.zeros((label_h, W, 3), np.uint8)
    pw = top.shape[1] // len(labels)
    for i, lab in enumerate(labels):
        cv2.putText(header, lab, (i * pw + 6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    title_bar = np.zeros((label_h, W, 3), np.uint8)
    cv2.putText(title_bar, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    out = np.concatenate([title_bar, header, top, bot], axis=0)
    cv2.imwrite(out_path, out)
    print(f"  wrote {out_path}  ({out.shape[1]}x{out.shape[0]})")


def main():
    detector = FaceDetector(max_faces=4)
    parser = FaceParser()
    assert parser._sess is not None, "BiSeNet ONNX session failed to load"

    boundaries = {
        "hairline (skin|hair)": ([CLS_SKIN], [CLS_HAIR]),
        "jawline (skin|bg+neck+cloth)": ([CLS_SKIN], [CLS_BG, CLS_NECK, CLS_CLOTH]),
        "lip vermilion (lips|skin)": ([CLS_ULIP, CLS_LLIP], [CLS_SKIN]),
        "eye contour (eyes|skin)": ([CLS_LEYE, CLS_REYE], [CLS_SKIN]),
    }

    for fname in ["DSCF8007.jpg", "DSCF7204.jpg", "DSCF4550.jpg"]:
        path = os.path.join(TEST_OUT, fname)
        if not os.path.exists(path):
            print(f"skip {fname} (missing)")
            continue
        img = load_and_downscale(path, 1600)
        faces = detector.detect(img)
        if not faces:
            print(f"{fname}: no faces detected")
            continue
        face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        print(f"\n=== {fname}  img={img.shape[1]}x{img.shape[0]}  bbox={face.bbox}  ied={face.ied:.1f}")

        t0 = time.perf_counter()
        logits, (cx1, cy1, cx2, cy2), crop = run_bisenet(parser, img, face.bbox)
        t_inf = time.perf_counter() - t0
        cw, ch = cx2 - cx1, cy2 - cy1
        px_scale = ((cw / 512.0) + (ch / 512.0)) / 2.0

        t0 = time.perf_counter()
        prob = softmax(logits)
        t_sm = time.perf_counter() - t0
        pred = np.argmax(logits, axis=0).astype(np.uint8)
        maxprob = prob.max(axis=0)

        feather = max(int(face.ied * 0.08), 3)
        print(f"  inference {t_inf*1000:.0f}ms | softmax(19x512x512) {t_sm*1000:.1f}ms | "
              f"crop {cw}x{ch} (1 crop px512 = {px_scale:.2f} img px) | feather r={feather} (lips/eyes r={feather//2})")
        print(f"  logits bytes={logits.nbytes/1e6:.1f}MB f32(19,512,512) | softmax f32 same | "
              f"argmax u8={pred.nbytes/1e6:.2f}MB | maxprob-only f32={maxprob.nbytes/1e6:.1f}MB")
        print(f"  global: frac maxprob<0.9 = {float((maxprob<0.9).mean()):.3f}, "
              f"<0.5 = {float((maxprob<0.5).mean()):.4f}, median maxprob = {float(np.median(maxprob)):.4f}")

        for name, (ca, cb) in boundaries.items():
            st = band_width_stats(pred, maxprob, ca, cb)
            if st is None:
                print(f"  {name}: boundary not found")
                continue
            w512 = st["band_width_px512"]
            print(f"  {name}: band(maxprob<0.9) ~{w512:.1f}px512 = {w512*px_scale:.1f} img px | "
                  f"maxprob@edge med={st['maxprob_at_boundary_median']:.3f} p25={st['maxprob_at_boundary_p25']:.3f} | "
                  f"boundary len={st['boundary_px']}")

        # Visual comparisons: hairline (skin class upper edge) + lips
        stem = os.path.splitext(fname)[0]
        for region, cls_set, want, r in [
            ("hairline", (CLS_SKIN,), "top", feather),
            ("lips", (CLS_ULIP, CLS_LLIP), "center", feather // 2),
        ]:
            masks, times = build_masks_at_crop_res(logits, prob, pred, list(cls_set), crop, max(r, 2))
            zoom = find_region_crop(pred, list(cls_set), crop.shape, want=want)
            if zoom is None:
                print(f"  {region}: class absent, skipping render")
                continue
            print(f"  {region} mask timings (crop {cw}x{ch}): " +
                  ", ".join(f"{k}={v*1000:.1f}ms" for k, v in times.items()))
            out = os.path.join(TEST_OUT, f"spike_c3_{stem}_{region}.png")
            render_panel(crop, masks, zoom, out,
                         f"{stem} {region}: mask (top) / image*mask halo test (bottom)")

        # Probability profile across the hairline: is the transition graded or a step?
        bnd = boundary_between(pred, [CLS_SKIN], [CLS_HAIR])
        ys, xs = np.where(bnd)
        if len(ys) > 0:
            j = len(ys) // 2
            y0, x0 = int(ys[j]), int(xs[j])
            lo, hi = max(0, y0 - 8), min(512, y0 + 9)
            prof = prob[CLS_SKIN, lo:hi, x0]
            print(f"  skin-prob profile through hairline @512({x0},{y0}) rows {lo}..{hi-1}: " +
                  " ".join(f"{p:.2f}" for p in prof))

    print("\nDone.")


if __name__ == "__main__":
    main()

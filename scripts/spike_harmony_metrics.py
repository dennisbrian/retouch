#!/usr/bin/env python3
"""Facial-harmony objective spike — PLAN_BLEMISH_POLICY_HARMONY.md evidence.

Measures candidate harmony metrics on real portraits and validates that they
discriminate the failure modes they are designed to catch:

  H1  Texture Parity Ratio (TPR): robust high-band energy of body skin /
      face skin. Should sit near ~1 on an untouched photo, collapse toward 0
      (or blow up) when only the face is smoothed, and return toward the
      original ratio when face AND body are smoothed consistently
      (the "porcelain parity" reference-image trait).
  H2  Pore-band spectral fraction per region (scale-invariant FFT fraction,
      reusing qa_detectors' pore band) — same discrimination check.
  H3  Specular-shape preservation: extract_specular before/after smoothing;
      measures retained specular energy + area (dimensionality guard).
  H4  Identity-mark retention: freckle.classify_anomalies before/after;
      fraction of beauty_marks whose centroid survives processing.
  H5  Banding guard sanity: qa_detectors.detect_banding on the smoothed
      renders (porcelain style must not introduce posterization).

All measures are within-subject relative (ratios against the same face's own
baseline) — no absolute intensity thresholds (tone-invariance rule).

Run: python3 scripts/spike_harmony_metrics.py [image ...]
Writes test_output/spike_harmony_<name>.png (masks + high-band visualization).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retouch.detection import FaceDetector  # noqa: E402
from retouch.parsing import FaceParser  # noqa: E402
from retouch.color_space import bgr_to_lch, skin_mask_lch  # noqa: E402
from retouch.specular import extract_specular  # noqa: E402
from retouch.freckle import FreckleRemover  # noqa: E402
from retouch.qa_detectors import detect_banding  # noqa: E402
from retouch.utils import guided_filter, normalize_mask  # noqa: E402

DEFAULTS = [
    ROOT / "test_output" / "DSCF4550.jpg",
    ROOT / "test_output" / "DSCF4454.jpg",
    ROOT / "test_output" / "DSCF7204.jpg",
    ROOT / "test_output" / "DSCF7011.jpg",
]

MIN_REGION_PX = 4000  # don't report spectra from tiny regions


def robust_highband_energy(gray_f: np.ndarray, mask: np.ndarray,
                           sigma: float) -> float:
    """MAD-derived robust std of the high band inside mask.

    High band = gray - Gaussian(sigma). sigma is scaled to face width by the
    caller so the band means the same physical detail scale on face and body.
    Relative measure per-region; no absolute thresholds.
    """
    low = cv2.GaussianBlur(gray_f, (0, 0), sigma)
    high = gray_f - low
    vals = high[mask > 0.5]
    if vals.size < MIN_REGION_PX:
        return float("nan")
    mad = float(np.median(np.abs(vals - np.median(vals))))
    return 1.4826 * mad


def pore_fraction_masked(gray_f: np.ndarray, mask: np.ndarray,
                         patch: int = 32) -> float:
    """Mean pore-band FFT power fraction over patches fully inside mask.

    Patch-based so the FFT sees only region content (a global FFT would mix
    face/body/background). Pore band per qa_detectors: period 2-8 px.
    """
    h, w = gray_f.shape
    fracs = []
    for y in range(0, h - patch, patch // 2):
        for x in range(0, w - patch, patch // 2):
            m = mask[y:y + patch, x:x + patch]
            if (m > 0.5).mean() < 0.98:
                continue
            g = gray_f[y:y + patch, x:x + patch]
            f = g - g.mean()
            F = np.fft.fftshift(np.fft.fft2(f * np.hanning(patch)[:, None]
                                            * np.hanning(patch)[None, :]))
            power = F.real ** 2 + F.imag ** 2
            yy, xx = np.mgrid[0:patch, 0:patch]
            r = np.sqrt(((xx - patch / 2) / patch) ** 2
                        + ((yy - patch / 2) / patch) ** 2)
            band = (r >= 1.0 / 8.0) & (r <= 0.5)
            tot = float(power.sum())
            if tot > 1e-9:
                fracs.append(float(power[band].sum()) / tot)
    return float(np.mean(fracs)) if fracs else float("nan")


def smooth_region(img_u8: np.ndarray, mask: np.ndarray,
                  radius: int, eps: float = 0.01) -> np.ndarray:
    """Guided-filter smooth blended through mask (porcelain-style op proxy)."""
    f = img_u8.astype(np.float32) / 255.0
    sm = np.zeros_like(f)
    for c in range(3):
        sm[:, :, c] = guided_filter(f[:, :, c], radius, eps, guide=None,
                                    max_dim=1200)
    m3 = mask[:, :, None]
    out = f * (1 - m3) + sm * m3
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def build_body_mask(img_u8: np.ndarray, face_oval: np.ndarray,
                    skin: np.ndarray, hair: np.ndarray | None) -> np.ndarray:
    """Simplified _stage_body_skin mask: LCH skin, minus face hull, minus hair."""
    lch = bgr_to_lch(img_u8)
    body = skin_mask_lch(lch, hue_center=25.0, hue_tolerance=25.0,
                         chroma_min=8.0).astype(np.float32)
    fo = (normalize_mask(face_oval) > 0.3).astype(np.uint8)
    contours, _ = cv2.findContours(fo, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        pts = np.vstack(contours)
        if len(pts) >= 3:
            hull = cv2.convexHull(pts)
            interior = np.zeros_like(fo)
            cv2.fillConvexPoly(interior, hull, 1)
            body *= (1.0 - interior.astype(np.float32))
    if hair is not None:
        body *= (1.0 - np.clip(normalize_mask(hair), 0, 1))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    body = cv2.morphologyEx(body, cv2.MORPH_OPEN, k)
    body = cv2.erode(body, k)  # stay off edges/occlusion boundaries
    return body


def specular_shape(img_u8: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """(total specular energy, specular area fraction) inside mask."""
    spec = extract_specular(img_u8, skin_mask=mask)
    sel = mask > 0.5
    if sel.sum() == 0:
        return float("nan"), float("nan")
    s = spec[sel]
    energy = float(s.mean())
    # area where specular exceeds the region's own 95th pct/2 — relative gate
    thr = float(np.percentile(s, 95)) * 0.5
    area = float((s > max(thr, 1e-6)).mean())
    return energy, area


def mark_retention(orig_u8: np.ndarray, proc_u8: np.ndarray,
                   skin: np.ndarray) -> tuple[int, int, float]:
    """Beauty marks in orig still detected (within r=6px) in processed."""
    fr = FreckleRemover()
    before = [c for c in fr.classify_anomalies(orig_u8, face_mask=skin,
                                               confidence_threshold=0.6)
              if c.classification == "beauty_mark"]
    after = fr.classify_anomalies(proc_u8, face_mask=skin,
                                  confidence_threshold=0.4)
    kept = 0
    for b in before:
        for a in after:
            d = np.hypot(a.centroid[0] - b.centroid[0],
                         a.centroid[1] - b.centroid[1])
            if d <= 6.0:
                kept += 1
                break
    n = len(before)
    return n, kept, (kept / n if n else float("nan"))


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]] or DEFAULTS
    det = FaceDetector()
    parser = FaceParser()
    out_dir = ROOT / "test_output"

    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"{path.name}: unreadable, skipped")
            continue
        scale = 1600.0 / max(img.shape[:2])
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
        faces = det.detect(img)
        if not faces:
            print(f"{path.name}: no face")
            continue
        f = faces[0]
        regions = parser.parse(f.landmarks, img, f.bbox, ied=f.ied)
        skin = regions.skin
        if skin is None or skin.max() < 0.01:
            print(f"{path.name}: no skin mask")
            continue
        face_mask = (normalize_mask(skin) > 0.5).astype(np.float32)
        # erode face mask off edges so hairline/brows don't pollute spectra
        face_mask = cv2.erode(face_mask, np.ones((7, 7), np.uint8))
        neck = regions.neck
        hair = getattr(regions, "hair", None)
        body = build_body_mask(img, regions.face_oval
                               if regions.face_oval is not None else skin,
                               skin, hair)
        if neck is not None:
            body = np.clip(body + (normalize_mask(neck) > 0.5), 0, 1)
        body_px = int((body > 0.5).sum())
        face_px = int((face_mask > 0.5).sum())
        face_w = float(f.bbox[2])
        sigma = max(1.5, face_w / 300.0)  # high-band scale tied to face size

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # ---- Renders: face-only smoothed vs face+body smoothed ----
        r_gf = max(4, int(face_w * 0.02))
        face_only = smooth_region(img, cv2.GaussianBlur(face_mask, (0, 0), 3),
                                  r_gf)
        both = smooth_region(face_only,
                             cv2.GaussianBlur(body.astype(np.float32),
                                              (0, 0), 3), r_gf)

        def report(tag: str, im: np.ndarray) -> dict:
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
            ef = robust_highband_energy(g, face_mask, sigma)
            eb = robust_highband_energy(g, body, sigma)
            pf = pore_fraction_masked(g, face_mask)
            pb = pore_fraction_masked(g, body)
            tpr = eb / ef if ef and not np.isnan(ef) and ef > 1e-6 else float("nan")
            se, sa = specular_shape(im, face_mask)
            band = detect_banding(im)
            print(f"  [{tag:10s}] E_face={ef:6.3f} E_body={eb:6.3f} "
                  f"TPR={tpr:5.2f} | poreF_face={pf:.4f} poreF_body={pb:.4f} "
                  f"| spec E={se:6.2f} area={sa:.3f} "
                  f"| banding score={band.get('score', 0):.3f} "
                  f"flagged={band.get('flagged')}")
            return {"ef": ef, "eb": eb, "tpr": tpr, "pf": pf, "pb": pb,
                    "se": se, "sa": sa}

        print(f"\n{path.name}: face_px={face_px} body_px={body_px} "
              f"face_w={face_w:.0f} sigma={sigma:.2f} r_gf={r_gf}")
        if body_px < MIN_REGION_PX:
            print("  (body region too small — TPR not reliable on this asset)")
        m0 = report("original", img)
        m1 = report("face-only", face_only)
        m2 = report("face+body", both)

        # Harmony deltas
        if not np.isnan(m0["tpr"]):
            print(f"  TPR drift: orig->face-only {m0['tpr']:.2f}->{m1['tpr']:.2f}"
                  f" | orig->face+body {m0['tpr']:.2f}->{m2['tpr']:.2f}")
        n, kept, ratio = mark_retention(img, face_only, face_mask)
        print(f"  beauty-mark retention (face-only smooth): {kept}/{n}"
              f" = {ratio if not np.isnan(ratio) else 'n/a'}")

        # Visualization panel: image | face mask | body mask | high band x8
        low = cv2.GaussianBlur(gray, (0, 0), sigma)
        hb = np.clip(np.abs(gray - low) * 8, 0, 255).astype(np.uint8)
        panel = np.hstack([
            img,
            cv2.cvtColor((face_mask * 255).astype(np.uint8),
                         cv2.COLOR_GRAY2BGR),
            cv2.cvtColor((body * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(hb, cv2.COLOR_GRAY2BGR),
        ])
        out = out_dir / f"spike_harmony_{path.stem}.png"
        cv2.imwrite(str(out), panel)
        print(f"  panel -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

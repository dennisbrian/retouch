"""FA-02 eligibility metric validity research (RESEARCH ONLY).

Answers one question: is ``legacy_highpass_std_px2`` (the production
``fa02_texture_eligibility.highpass_std`` at its fixed 2px sigma) measuring
recoverable facial micro-texture, or is it confounded by capture scale and
sensor noise? Follows on from the d8e2171 native-resolution survey, whose
headline finding was: 1/11 faces crosses the 3.5 threshold, that face is the
smallest and grainiest in the corpus (owner-annotated as ISO grain, not
structure), and face width is strongly inverse to highpass_std across the
whole corpus.

This script does NOT:
* touch retouch/fa02_texture_eligibility.py or any production threshold;
* wire any candidate metric into the engine;
* derive or recommend a new production threshold (n=9 is enough to detect
  metric pathology, not to calibrate a gate -- see the acceptance criteria
  in this round's task brief);
* sharpen, denoise, or resize any image except as an explicit, labelled
  diagnostic treatment written only under /tmp.

Inputs: the 9 (canvas, support, face_width_px) triples extracted by
fa02_dump_faces.py to /tmp/fa02_scale_noise/crops/ -- the EXACT
pre_smooth_canvas + effective_support the production gate measured on the
native-resolution survey (verified byte-identical: recomputing
legacy_highpass_std_px2 from the dump reproduces the survey's
4.639751879330601 for DSCF1606 to full float precision). Two faces from
nahida_DSCF6754 are NOT included: that image's 2 faces were processed inside
FaceProcessorPool's spawned worker subprocess, where the in-process
monkeypatch used to capture these triples does not apply. This is disclosed,
not silently dropped -- n=9, not 11, for every experiment in this script.

Experiments (see module-level functions):
    1. corpus_correlations   -- item 2: face scale / noise / metric correlations
    2. rescale_experiment    -- item 3: same-content multi-scale response
    3. noise_denoise_blur    -- item 4: controlled single-variable treatments
    4. candidate_comparison  -- item 5: legacy vs face-scaled vs noise-subtracted
    5. semantic_region_check -- item 6: eyebrow (structural) vs smooth cheek

Usage:
    .venv/bin/python scripts/qa/fa02_metric_validity_experiment.py \
        --out test_output/fa02_metric_validity_report.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "qa"))

from retouch.fa02_texture_eligibility import (  # noqa: E402
    HIGHPASS_SIGMA_PX,
    MIN_HIGHPASS_STD,
    estimate_noise_sigma,
)
from fa02_texture_metric_candidates import METRIC_REGISTRY  # noqa: E402

CROPS_DIR = Path("/tmp/fa02_scale_noise/crops")
DIAG_DIR = Path("/tmp/fa02_scale_noise/diagnostic_treatments")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_faces() -> List[Dict[str, Any]]:
    faces = []
    for path in sorted(glob.glob(str(CROPS_DIR / "*.npz"))):
        d = np.load(path)
        stem = Path(path).stem  # "<asset_id>_face<i>"
        faces.append({
            "id": stem,
            "canvas": d["canvas"].astype(np.float32),
            "support": (d["support"] > 0.5).astype(np.uint8),
            "face_width_px": float(d["face_width_px"]),
        })
    return faces


# ---------------------------------------------------------------------------
# Item 2: corpus correlations
# ---------------------------------------------------------------------------


def _spearman(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 3:
        return None
    xr = _rank(x)
    yr = _rank(y)
    n = len(x)
    d2 = sum((a - b) ** 2 for a, b in zip(xr, yr))
    return 1.0 - (6.0 * d2) / (n * (n ** 2 - 1))


def _rank(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def corpus_correlations(faces: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    for f in faces:
        canvas, support, fw = f["canvas"], f["support"], f["face_width_px"]
        hp = legacy = METRIC_REGISTRY["legacy_highpass_std_px2"](canvas, support, fw)
        noise = estimate_noise_sigma(canvas, mask=support)
        lap = cv2.Laplacian(
            cv2.cvtColor(np.clip(canvas, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY),
            cv2.CV_64F,
        )
        lapvar = float(lap[support > 0].var()) if support.sum() > 0 else None
        support_px = int(support.sum())
        # face_width_px is IED*2.5 (a linear scale), not literally the crop
        # width -- "face area" is reported as its square, the pipeline's own
        # scale-squared convention, not a measured bounding-box area.
        face_area_proxy = fw ** 2
        rows.append({
            "id": f["id"],
            "face_width_px": fw,
            "face_area_proxy_px2": face_area_proxy,
            "support_px": support_px,
            "legacy_highpass_std_px2": hp,
            "noise_sigma": noise,
            "laplacian_variance": lapvar,
            "iso": "unavailable (no EXIF ISO tag on any source in this corpus)",
            "exposure": "unavailable (no EXIF ExposureTime tag on any source in this corpus)",
            "compression": "JPEG (old-style); no re-encode/compression-artifact estimate computed",
        })

    def _col(key):
        return [r[key] for r in rows if r[key] is not None]

    hp_vals = [r["legacy_highpass_std_px2"] for r in rows]
    fw_vals = [r["face_width_px"] for r in rows]
    area_vals = [r["face_area_proxy_px2"] for r in rows]
    noise_vals = [r["noise_sigma"] for r in rows]
    lap_vals = [r["laplacian_variance"] for r in rows]

    correlations = {
        "highpass_vs_face_width": _spearman(hp_vals, fw_vals),
        "highpass_vs_face_area_proxy": _spearman(hp_vals, area_vals),
        "highpass_vs_noise_sigma": _spearman(hp_vals, noise_vals),
        "highpass_vs_laplacian_variance": _spearman(hp_vals, lap_vals),
        "n": len(rows),
        "note": (
            "n=9. All four correlated quantities (highpass_std, noise_sigma, "
            "laplacian_variance) share the SAME fixed-pixel-scale measurement "
            "convention as legacy_highpass_std_px2 itself, so these "
            "correlations cannot cleanly separate 'scale confound' from "
            "'definitional artifact of comparing same-scale metrics to each "
            "other'. They are reported as a first-pass signal only; the "
            "controlled single-variable experiments (rescale_experiment, "
            "noise_denoise_blur) carry the actual evidentiary weight. "
            "ALSO NOTE: face_area_proxy_px2 = face_width_px ** 2 is a "
            "monotone transform of face_width_px, not an independent "
            "measurement -- its rank correlation with highpass_std is "
            "therefore IDENTICAL to highpass_vs_face_width by construction, "
            "not a second confirming data point. It is retained in the "
            "per-row output for readability, not as independent evidence."
        ),
    }
    return {"rows": rows, "correlations": correlations}


# ---------------------------------------------------------------------------
# Item 3: controlled rescaling of the SAME content
# ---------------------------------------------------------------------------


SCALE_FACTORS = (0.5, 0.75, 1.0, 1.5, 2.0)


def rescale_experiment(faces: List[Dict[str, Any]]) -> Dict[str, Any]:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for f in faces:
        canvas, support, fw = f["canvas"], f["support"], f["face_width_px"]
        per_face = {"id": f["id"], "base_face_width_px": fw, "scales": {}}
        for scale in SCALE_FACTORS:
            # Two interpolators only matter for DOWNSCALE (AREA vs LINEAR
            # decimation genuinely differ there); for scale >= 1.0 cv2 has no
            # meaningful choice to compare, so only one entry is emitted --
            # emitting both under a scale >= 1.0 loop previously produced
            # two identically-computed keys (both used INTER_LINEAR
            # regardless of the loop label), a labelling bug, not a second
            # measurement.
            if scale < 1.0:
                interp_variants = (("area", cv2.INTER_AREA), ("linear", cv2.INTER_LINEAR))
            else:
                interp_variants = (("linear", cv2.INTER_LINEAR),)
            for interp_name, interp in interp_variants:
                h, w = canvas.shape[:2]
                new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
                # Diagnostic-only resample. Never sharpened, never denoised.
                resampled = cv2.resize(canvas, new_size, interpolation=interp)
                resampled_mask = cv2.resize(
                    support.astype(np.float32), new_size, interpolation=cv2.INTER_LINEAR,
                )
                resampled_mask = (resampled_mask > 0.5).astype(np.uint8)
                rescaled_fw = fw * scale

                legacy_val = METRIC_REGISTRY["legacy_highpass_std_px2"](
                    resampled, resampled_mask, rescaled_fw,
                )
                # Face-scaled candidate uses the RESCALED face width, so a
                # perfectly scale-invariant metric should return ~constant
                # values across this row.
                scaled_val = METRIC_REGISTRY["face_scaled_highpass_std"](
                    resampled, resampled_mask, rescaled_fw,
                )
                key = f"{scale}x_{interp_name}"
                per_face["scales"][key] = {
                    "shape": list(resampled.shape[:2]),
                    "support_px": int(resampled_mask.sum()),
                    "legacy_highpass_std_px2": legacy_val,
                    "face_scaled_highpass_std": scaled_val,
                }
        results.append(per_face)

    # Summary: does legacy cross 3.5 purely from a scale change on content
    # that did NOT cross it at 1.0x, or vice versa?
    crossings = []
    for pf in results:
        base_val = pf["scales"].get("1.0x_linear", {}).get("legacy_highpass_std_px2")
        if base_val is None:
            continue
        for key, entry in pf["scales"].items():
            val = entry["legacy_highpass_std_px2"]
            if val is None:
                continue
            base_side = base_val >= MIN_HIGHPASS_STD
            this_side = val >= MIN_HIGHPASS_STD
            if base_side != this_side:
                crossings.append({
                    "id": pf["id"], "scale_key": key,
                    "base_highpass_std": base_val, "scaled_highpass_std": val,
                    "base_eligible": base_side, "scaled_eligible": this_side,
                })

    return {
        "per_face": results,
        "eligibility_flips_from_pure_rescale": crossings,
        "note": (
            "Pure rescale of identical content should not change the "
            "eligible/abstained verdict if the metric were scale-invariant. "
            "Any entry in eligibility_flips_from_pure_rescale is direct "
            "evidence legacy_highpass_std_px2 is NOT scale invariant "
            "(no sharpening/denoising was applied -- resize only)."
        ),
    }


# ---------------------------------------------------------------------------
# Item 4: noise / denoise / blur, single-variable
# ---------------------------------------------------------------------------


def _add_gaussian_noise(canvas: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=canvas.shape).astype(np.float32)
    return np.clip(canvas + noise, 0.0, 255.0)


def _find_noise_crossing_sigma(
    canvas: np.ndarray, mask: np.ndarray, fw: float, threshold: float, seed: int,
    lo: float = 0.0, hi: float = 20.0, iters: int = 14,
) -> Optional[float]:
    """Binary search the ADDED sigma at which legacy_highpass_std_px2 first
    crosses `threshold`. Returns None if hi never crosses."""
    metric = METRIC_REGISTRY["legacy_highpass_std_px2"]
    base = metric(canvas, mask, fw)
    if base is not None and base >= threshold:
        return 0.0  # already eligible with no added noise
    hi_val = metric(_add_gaussian_noise(canvas, hi, seed), mask, fw)
    if hi_val is None or hi_val < threshold:
        return None  # does not cross even at hi
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        val = metric(_add_gaussian_noise(canvas, mid, seed), mask, fw)
        if val is not None and val >= threshold:
            hi = mid
        else:
            lo = mid
    return hi


def noise_denoise_blur(faces: List[Dict[str, Any]]) -> Dict[str, Any]:
    from retouch.fa02_texture_eligibility import MAX_NOISE_SIGMA

    rows = []
    for f in faces:
        canvas, support, fw = f["canvas"], f["support"], f["face_width_px"]
        legacy_fn = METRIC_REGISTRY["legacy_highpass_std_px2"]
        base_val = legacy_fn(canvas, support, fw)
        base_noise = estimate_noise_sigma(canvas, mask=support)

        # A: mild structure-preserving denoise (bilateral filter -- edge
        # aware, the standard "preserve structure, remove noise" baseline).
        canvas_u8 = np.clip(canvas, 0, 255).astype(np.uint8)
        denoised = cv2.bilateralFilter(canvas_u8, d=5, sigmaColor=25, sigmaSpace=25).astype(np.float32)
        denoised_val = legacy_fn(denoised, support, fw)

        # B: synthetic sensor-like noise, added at the corpus's own median
        # noise_sigma (representative magnitude, not tuned to cross 3.5).
        added_sigma = 3.0
        noised = _add_gaussian_noise(canvas, added_sigma, seed=zlib.crc32(f["id"].encode()))
        noised_val = legacy_fn(noised, support, fw)
        noised_noise_est = estimate_noise_sigma(noised, mask=support)

        # C: mild blur (structure AND noise both attenuated -- included as a
        # contrast case, not a proposed treatment).
        blurred = cv2.GaussianBlur(canvas, (0, 0), sigmaX=1.5).astype(np.float32)
        blurred_val = legacy_fn(blurred, support, fw)

        # Crossing search: minimum added noise sigma that alone pushes this
        # face's score across 3.5, and whether that sigma is below the
        # gate's OWN noise ceiling (MAX_NOISE_SIGMA=6.0) -- if so, the two
        # gates are mutually inconsistent (noise alone can pass gate 2 while
        # failing to trip gate 3's own ceiling).
        crossing_sigma = _find_noise_crossing_sigma(
            canvas, support, fw, MIN_HIGHPASS_STD, seed=zlib.crc32(f["id"].encode()),
        )

        rows.append({
            "id": f["id"],
            "face_width_px": fw,
            "D_no_modification": base_val,
            "A_bilateral_denoise": denoised_val,
            "A_effect": (None if base_val is None or denoised_val is None
                         else denoised_val - base_val),
            "B_added_noise_sigma3": noised_val,
            "B_effect": (None if base_val is None or noised_val is None
                         else noised_val - base_val),
            "B_measured_noise_after": noised_noise_est,
            "C_mild_blur_sigma1.5": blurred_val,
            "C_effect": (None if base_val is None or blurred_val is None
                         else blurred_val - base_val),
            "noise_sigma_at_which_metric_crosses_3.5": crossing_sigma,
            "crossing_sigma_below_max_noise_sigma_ceiling": (
                None if crossing_sigma is None
                else bool(crossing_sigma < MAX_NOISE_SIGMA)
            ),
            "base_noise_sigma": base_noise,
            # Mechanism check for the gate-inconsistency finding: does
            # estimate_noise_sigma (the gate's OWN noise measurement) even
            # detect the injected noise at the crossing point? If it
            # under-reports, that's WHY gate 3 fails to catch what gate 2
            # reacts to, not just an unlucky threshold gap.
            "measured_noise_sigma_after_B_treatment": noised_noise_est,
            "added_noise_sigma_B_treatment": added_sigma,
            "expected_combined_sigma_if_measured_correctly": float(
                np.sqrt(base_noise ** 2 + added_sigma ** 2)
            ) if base_noise is not None else None,
        })

    flips_below_ceiling = [
        r for r in rows
        if r["crossing_sigma_below_max_noise_sigma_ceiling"] is True
    ]

    return {
        "rows": rows,
        "max_noise_sigma_ceiling": MAX_NOISE_SIGMA,
        "faces_where_noise_alone_crosses_3.5_below_the_noise_ceiling": [
            r["id"] for r in flips_below_ceiling
        ],
        "note": (
            "If a face's 'noise_sigma_at_which_metric_crosses_3.5' is below "
            "MAX_NOISE_SIGMA (6.0), added noise alone can make the face "
            "ELIGIBLE per gate 2 while never tripping gate 3's own noise "
            "ceiling -- the two gates would be mutually inconsistent. This "
            "is evaluated per the frozen production MAX_NOISE_SIGMA, not a "
            "tuned value."
        ),
    }


# ---------------------------------------------------------------------------
# Item 5: candidate comparison (research only)
# ---------------------------------------------------------------------------


def candidate_comparison(faces: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    for f in faces:
        canvas, support, fw = f["canvas"], f["support"], f["face_width_px"]
        row = {"id": f["id"], "face_width_px": fw}
        for name, fn in METRIC_REGISTRY.items():
            row[name] = fn(canvas, support, fw)
        rows.append(row)

    # Does each candidate still separate DSCF1606 (known noise-confounded)
    # from the rest of the corpus the same way legacy does?
    dscf1606 = next((r for r in rows if "DSCF1606" in r["id"]), None)
    others = [r for r in rows if "DSCF1606" not in r["id"]]
    separation = {}
    if dscf1606 is not None and others:
        for name in METRIC_REGISTRY:
            dscf1606_val = dscf1606.get(name)
            other_vals = [r.get(name) for r in others if r.get(name) is not None]
            if dscf1606_val is None or not other_vals:
                continue
            separation[name] = {
                "dscf1606_value": dscf1606_val,
                "rest_of_corpus_max": max(other_vals),
                "dscf1606_still_the_max": dscf1606_val >= max(other_vals),
            }

    return {
        "rows": rows,
        "dscf1606_separation_by_metric": separation,
        "note": (
            "dscf1606_still_the_max=True means that candidate metric ALSO "
            "ranks the known noise-confounded face highest -- i.e. it did "
            "NOT fix the confound. False means the candidate demoted "
            "DSCF1606 relative to the rest of the corpus, consistent with "
            "correctly discounting noise. This compares metrics on their "
            "OWN terms (ranking), not by tuning either to a target pass "
            "rate -- no threshold is fitted here."
        ),
    }


# ---------------------------------------------------------------------------
# Item 6: semantic region check (model-derived, not hand-labelled)
# ---------------------------------------------------------------------------


def semantic_region_check() -> Dict[str, Any]:
    """Compare eyebrow-interior (structural positive, model-derived from
    BiSeNet/landmark eyebrow regions) against interior smooth-cheek
    (detail-poor negative) within the SAME faces, using face_contexts from a
    fresh engine.process() call (this needs `regions.left_eyebrow` /
    `right_eyebrow`, which the earlier gate-input dump does not carry).

    Both region sets are DERIVED from the pipeline's own segmentation/
    landmark output, not hand-authored -- consistent with this session's
    standing rule against fabricating ground truth. Provenance is recorded
    per row.
    """
    import logging
    logging.basicConfig(level=logging.WARNING)

    from retouch.engine import RetouchEngine
    from fa02_eligibility_survey import DEV_CORPUS, _resolve_desktop_path

    engine = RetouchEngine()
    rows = []
    for entry in DEV_CORPUS:
        path = _resolve_desktop_path(entry["folder"], entry["file"])
        if path is None:
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        # max_dim kept modest here -- this experiment only needs region
        # SEPARATION, not the exact eligibility number, and full native-res
        # face_contexts extraction for all 10 assets a second time is not
        # worth the runtime. Reported explicitly so it's not confused with
        # the native-resolution numbers elsewhere in this report.
        result = engine.process(img, recipe="natural", max_dim=2048)
        if not result.face_contexts:
            continue
        for fc_idx, fc in enumerate(result.face_contexts):
            regions = fc.regions
            face_image = fc.face_image
            if face_image is None or regions is None:
                continue
            brow = None
            for attr in ("left_eyebrow", "right_eyebrow"):
                m = getattr(regions, attr, None)
                if m is None:
                    continue
                brow = m if brow is None else np.clip(brow + m, 0, 1)
            skin = getattr(regions, "skin", None)
            if brow is None or skin is None:
                continue
            brow_mask = (np.asarray(brow) > 0.5).astype(np.uint8)
            if brow_mask.sum() < 50:
                continue
            # Detail-poor negative: skin interior, eroded away from any
            # feature boundary so it can't pick up brow/eye/lip edges.
            skin_f = np.asarray(skin, dtype=np.float32)
            if skin_f.max() > 1.5:
                skin_f = skin_f / 255.0
            skin_bin = (skin_f > 0.5).astype(np.uint8)
            kernel = np.ones((15, 15), np.uint8)
            cheek_mask = cv2.erode(skin_bin, kernel, iterations=2)
            cheek_mask = np.clip(cheek_mask.astype(np.int32) - brow_mask.astype(np.int32), 0, 1).astype(np.uint8)
            if cheek_mask.sum() < 50:
                continue

            # Boundary-free brow arm: a raw BiSeNet eyebrow mask is thin, so
            # a large fraction of its pixels sit ON its own hair<->skin
            # boundary -- a fixed 2px high-pass responds to ANY edge, so a
            # brow-vs-cheek comparison using the raw mask cannot separate
            # "detects hair microstructure" from "detects mask boundary".
            # Erode the brow mask itself (small kernel -- brows are thin, a
            # 15x15 kernel would erase them entirely) to get an interior-only
            # arm with the same boundary-free property as the cheek mask.
            brow_eroded_mask = cv2.erode(brow_mask, np.ones((3, 3), np.uint8), iterations=1)
            brow_eroded_support_px = int(brow_eroded_mask.sum())

            fw = fc.face_data.ied * 2.5
            entry_row = {
                "asset_id": entry["asset_id"], "face_index": fc_idx,
                "face_width_px": fw,
                "brow_support_px": int(brow_mask.sum()),
                "brow_eroded_support_px": brow_eroded_support_px,
                "cheek_support_px": int(cheek_mask.sum()),
                "region_provenance": "model_derived (BiSeNet/landmark eyebrow + eroded skin interior)",
            }
            for name, fn in METRIC_REGISTRY.items():
                entry_row[f"{name}_brow"] = fn(face_image, brow_mask, fw)
                entry_row[f"{name}_cheek"] = fn(face_image, cheek_mask, fw)
                entry_row[f"{name}_brow_eroded"] = (
                    fn(face_image, brow_eroded_mask, fw)
                    if brow_eroded_support_px >= 20 else None
                )
            rows.append(entry_row)

    def _summarize(brow_key_suffix: str) -> Dict[str, Any]:
        summary = {}
        for name in METRIC_REGISTRY:
            brow_key, cheek_key = f"{name}_{brow_key_suffix}", f"{name}_cheek"
            pairs = [
                (r[brow_key], r[cheek_key]) for r in rows
                if r.get(brow_key) is not None and r.get(cheek_key) is not None
            ]
            if not pairs:
                continue
            brow_higher = sum(1 for b, c in pairs if b > c)
            summary[name] = {
                "n_faces": len(pairs),
                "brow_ranked_higher_than_cheek_count": brow_higher,
                "brow_ranked_higher_fraction": brow_higher / len(pairs),
            }
        return summary

    separation_summary = _summarize("brow")
    # The discriminating comparison: a raw eyebrow mask is thin, so most of
    # its pixels sit ON its own hair<->skin segmentation boundary, and a
    # fixed 2px high-pass responds to ANY edge (items 3-4 already establish
    # this metric is largely an edge/scale response). brow_eroded removes
    # that boundary from the brow arm the same way cheek_mask already
    # removes it from the negative arm -- if separation survives erosion,
    # it reflects genuine interior hair microstructure, not mask-edge
    # leakage. If it collapses toward 0.5, the raw-brow result above was
    # measuring the segmentation boundary, not hair.
    separation_summary_eroded = _summarize("brow_eroded")

    return {
        "rows": rows,
        "separation_summary_raw_brow_mask": separation_summary,
        "separation_summary_eroded_brow_mask": separation_summary_eroded,
        "note": (
            "A metric that responds to genuine structural micro-detail "
            "should rank eyebrow-interior (hair) above smooth-cheek-interior "
            "within the SAME face more often than not. "
            "brow_ranked_higher_fraction near 1.0 is the desired behaviour; "
            "near 0.5 means the metric does not discriminate; well below 0.5 "
            "would mean it is anti-correlated with structure. Measured at "
            "max_dim=2048 (region-separation check only, not a native-"
            "resolution eligibility measurement -- see the function "
            "docstring). "
            "IMPORTANT: separation_summary_raw_brow_mask uses the raw, thin "
            "BiSeNet eyebrow mask, most of whose pixels sit on its own "
            "hair<->skin boundary -- so a high separation score there is "
            "consistent with 'detects any edge' and does NOT by itself "
            "establish hair-microstructure sensitivity. "
            "separation_summary_eroded_brow_mask (an interior-only, "
            "boundary-free brow arm, matching the cheek arm's own "
            "boundary-free construction) is the discriminating comparison -- "
            "trust this one over the raw-mask summary."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "test_output" / "fa02_metric_validity_report.json"),
    )
    args = parser.parse_args()

    faces = _load_faces()
    if not faces:
        print(f"No dumped face crops found under {CROPS_DIR}. Run "
              f"fa02_dump_faces.py first.", file=sys.stderr)
        return 1

    print(f"Loaded {len(faces)} face crops from {CROPS_DIR}")

    report: Dict[str, Any] = {
        "corpus_note": (
            f"n={len(faces)} (single-face dev-split assets extracted via "
            "in-process monkeypatch of evaluate_face_eligibility; "
            "nahida_DSCF6754's 2 faces are NOT included -- they were "
            "processed inside FaceProcessorPool's spawned worker subprocess, "
            "where the extraction hook does not apply). This is a metric-"
            "pathology-detection sample, not a calibration corpus -- no "
            "production threshold may be derived from these results."
        ),
        "production_threshold_frozen_at": MIN_HIGHPASS_STD,
        "production_highpass_sigma_px": HIGHPASS_SIGMA_PX,
    }

    print("Running corpus_correlations (item 2)...")
    report["item2_corpus_correlations"] = corpus_correlations(faces)

    print("Running rescale_experiment (item 3)...")
    report["item3_rescale_experiment"] = rescale_experiment(faces)

    print("Running noise_denoise_blur (item 4)...")
    report["item4_noise_denoise_blur"] = noise_denoise_blur(faces)

    print("Running candidate_comparison (item 5)...")
    report["item5_candidate_comparison"] = candidate_comparison(faces)

    print("Running semantic_region_check (item 6)...")
    report["item6_semantic_region_check"] = semantic_region_check()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport written to {out_path}")

    corr = report["item2_corpus_correlations"]["correlations"]
    print(f"\nSpearman highpass_std vs face_width: {corr['highpass_vs_face_width']}")
    print(f"Spearman highpass_std vs noise_sigma: {corr['highpass_vs_noise_sigma']}")
    flips = report["item3_rescale_experiment"]["eligibility_flips_from_pure_rescale"]
    print(f"Eligibility flips from pure rescale: {len(flips)}")
    noise_flips = report["item4_noise_denoise_blur"]["faces_where_noise_alone_crosses_3.5_below_the_noise_ceiling"]
    print(f"Faces where added noise alone crosses 3.5 below the noise ceiling: {noise_flips}")
    sep = report["item5_candidate_comparison"]["dscf1606_separation_by_metric"]
    print(f"DSCF1606-still-max by metric: { {k: v['dscf1606_still_the_max'] for k, v in sep.items()} }")
    region_sep_raw = report["item6_semantic_region_check"]["separation_summary_raw_brow_mask"]
    region_sep_eroded = report["item6_semantic_region_check"]["separation_summary_eroded_brow_mask"]
    print(f"Brow-ranked-higher fraction (RAW brow mask, edge-confounded): { {k: v['brow_ranked_higher_fraction'] for k, v in region_sep_raw.items()} }")
    print(f"Brow-ranked-higher fraction (ERODED brow mask, discriminating): { {k: v['brow_ranked_higher_fraction'] for k, v in region_sep_eroded.items()} }")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

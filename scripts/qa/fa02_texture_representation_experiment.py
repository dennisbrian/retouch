"""Offline FA-02 experiment: saved canvases + accepted supports, no detection.

See docs/plans/EXPERIMENT_FA02_TEXTURE_REPRESENTATIONS_2026_09_06.md.
The controls subcommand generates mathematical fixtures, NOT portrait evidence.
All images/arrays are source-coordinate BGR in [0,255]; filters use float32.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from retouch.frequency import FrequencySeparator  # noqa: E402
from retouch.utils import adaptive_ksize  # noqa: E402

ARMS = ("A0_disabled", "A1_raw", "A2_dog", "A3_multiscale",
        "A4_orientation", "A5_retouch_frequency")
DEFAULT_CONFIG = {
    "gain": 0.1, "sigmas_at_ied_200": [0.6, 1.2, 2.4, 4.8, 9.6],
    "minimum_sigma_px": 0.6, "truncate": 3.0, "dog_index": 0,
    "multiscale_weights": [1.0, 0.5, 0.0, 0.0],
    "frequency_mid_gain": 0.5, "frequency_high_gain": 1.0,
    "orientation_attenuation": 0.5, "orientation_gradient_scale": 10.0,
    "tensor_sigma_at_ied_200": 1.2, "guard_feather_px": 4.0,
    "noise_reliability": "off_all_arms", "border": "REFLECT_101",
    "color": "encoded_BGR_float32_0_255", "clip": [0, 255],
    "signed_display_range": 2.0, "broad_diagnostic_sigma_px": 9.6,
    "opencv_threads": 1,
}
MASKS = ("allow", "corrected", "protected")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_config(cfg):
    if set(cfg) != set(DEFAULT_CONFIG):
        raise ValueError("Config keys must exactly match the documented config")
    numeric = [cfg[k] for k in ("gain", "minimum_sigma_px", "truncate",
               "tensor_sigma_at_ied_200", "orientation_gradient_scale",
               "guard_feather_px", "signed_display_range", "broad_diagnostic_sigma_px")]
    sigmas = cfg["sigmas_at_ied_200"]
    weights = cfg["multiscale_weights"]
    if not all(np.isfinite(numeric)) or any(x <= 0 for x in numeric[1:]):
        raise ValueError("Positive finite filter/display scales required")
    if not 0 <= cfg["gain"] <= 1 or len(sigmas) < 2 or len(weights) != len(sigmas)-1:
        raise ValueError("Invalid gain or band dimensions")
    if not all(np.isfinite(sigmas)) or any(x <= 0 for x in sigmas) or sorted(set(sigmas)) != sigmas:
        raise ValueError("Sigmas must be positive, finite and strictly increasing")
    if not isinstance(cfg["dog_index"], int) or not 0 <= cfg["dog_index"] < len(weights):
        raise ValueError("Invalid DoG index")
    bounded = weights + [cfg[k] for k in ("frequency_mid_gain", "frequency_high_gain",
                                        "orientation_attenuation")]
    if not all(np.isfinite(bounded)) or any(not 0 <= x <= 1 for x in bounded):
        raise ValueError("Band gains/attenuation must be in [0,1]")
    for key in ("noise_reliability", "border", "color", "clip", "opencv_threads"):
        if cfg[key] != DEFAULT_CONFIG[key]:
            raise ValueError(f"This first experiment freezes {key}")


def gaussian(img, sigma, truncate=3.0):
    radius = int(math.ceil(sigma * truncate))
    return cv2.GaussianBlur(img, (2*radius+1, 2*radius+1), sigma,
                            borderType=cv2.BORDER_REFLECT_101)


def luma(bgr):
    return bgr[..., 0]*0.114 + bgr[..., 1]*0.587 + bgr[..., 2]*0.299


def geometry(case, cfg):
    scale = case["inter_eye_distance_px"] / 200.0
    sigmas = [max(cfg["minimum_sigma_px"], s*scale) for s in cfg["sigmas_at_ied_200"]]
    tensor_sigma = max(cfg["minimum_sigma_px"], cfg["tensor_sigma_at_ied_200"]*scale)
    k_low = adaptive_ksize(case["face_width_px"], factor=0.12, minimum=5)
    k_mid = adaptive_ksize(case["face_width_px"], factor=0.04, minimum=3)
    # All specified bands, even unselected ones, get a COMMON conservative guard.
    # A4 takes a gradient of each band (one more pixel); tensor has Sobel support.
    radius = max(math.ceil(max(sigmas)*cfg["truncate"])+1,
                 math.ceil(tensor_sigma*cfg["truncate"])+1, k_low//2, k_mid//2)
    return {"sigmas_px": sigmas, "tensor_sigma_px": tensor_sigma,
            "frequency_k_low": k_low, "frequency_k_mid": k_mid,
            "common_radius_px": radius, "collapsed_bands": [
                i for i in range(len(sigmas)-1) if sigmas[i] == sigmas[i+1]]}


def map_external_supports(arr, geo, cfg):
    """Never cut holes into X/S; gate coefficients/output AFTER extraction.

    Square finite-filter support => Chebyshev distance, not Euclidean dilation.
    The fade is OUTSIDE the conservative exclusion radius, so it cannot leak in.
    Border exclusion makes scored output independent of reflected crop padding.
    """
    forbidden = ((arr["corrected"] > 0) | (arr["protected"] > 0)).astype(np.uint8)
    forbidden[[0, -1], :] = 1
    forbidden[:, [0, -1]] = 1
    distance = cv2.distanceTransform(1-forbidden, cv2.DIST_C, 3)
    guard = np.clip((distance-geo["common_radius_px"])/cfg["guard_feather_px"], 0, 1)
    return (arr["allow"]*guard).astype(np.float32)


def gradients(y):
    return (cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3, scale=1/8),
            cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3, scale=1/8))


def orientation_field(x, geo, cfg):
    gx, gy = gradients(luma(x))
    sigma = geo["tensor_sigma_px"]
    xx, yy, xy = [gaussian(z, sigma, cfg["truncate"]) for z in (gx*gx, gy*gy, gx*gy)]
    gap = np.sqrt(np.maximum((xx-yy)**2 + 4*xy*xy, 0))
    coherence = gap/(xx+yy+1e-8)
    theta = 0.5*np.arctan2(2*xy, xx-yy)  # dominant gradient NORMAL, not hair tangent
    strength = np.sqrt(np.maximum(xx+yy, 0))
    strength = strength/(strength+cfg["orientation_gradient_scale"])
    return theta, coherence, strength


def extract_bands(x, s, arm, geo, cfg):
    """Signed, same-coordinate differences. No coefficient denoising in any arm."""
    if arm == ARMS[0]:
        return np.zeros_like(s), {}
    residual = x-s
    if arm == ARMS[1]:
        return residual, {"raw": residual}
    if arm == ARMS[5]:
        # Actual existing Retouch separator, not a reimplementation or combine().
        # separate is linear float32: F(X-S) == F(X)-F(S), tested separately.
        layers = FrequencySeparator().separate(residual, geo["face_width_px"])
        bands = {"frequency_mid": layers.mid, "frequency_high": layers.high}
        return cfg["frequency_mid_gain"]*layers.mid + cfg["frequency_high_gain"]*layers.high, bands
    indices = ([cfg["dog_index"]] if arm == ARMS[2] else
               [i for i, w in enumerate(cfg["multiscale_weights"]) if w != 0])
    levels = {i: gaussian(residual, geo["sigmas_px"][i], cfg["truncate"])
              for i in sorted(set(indices + [i+1 for i in indices]))}
    out = np.zeros_like(s)
    bands = {}
    if arm == ARMS[4]:
        theta, coherence, strength = orientation_field(x, geo, cfg)
        bands.update(theta=theta, coherence=coherence, edge_strength=strength)
    for i in indices:
        band = levels[i]-levels[i+1]
        weight = 1.0 if arm == ARMS[2] else cfg["multiscale_weights"][i]
        bands[f"band_{i}"] = band
        if arm == ARMS[4]:
            gx, gy = gradients(luma(band))
            alignment = (gx*np.cos(theta)+gy*np.sin(theta))**2/(gx*gx+gy*gy+1e-8)
            # Bounded attenuation of edge-normal detail. Isotropic path >= 0.5
            # at the frozen setting. This can also suppress real hairs: score it.
            gate = 1-cfg["orientation_attenuation"]*coherence*strength*alignment
            bands[f"orientation_gate_{i}"] = gate
            band = band*gate[..., None]
        out += weight*band
    return out, bands


def restore(x, s, arm, geo, cfg, eligibility):
    detail, bands = extract_bands(x, s, arm, geo, cfg)
    requested_delta = cfg["gain"]*eligibility[..., None]*detail
    unclipped = s+requested_delta
    output = np.clip(unclipped, *cfg["clip"]).astype(np.float32)
    return {"output": output, "delta": output-s, "requested_delta": requested_delta,
            "detail": detail, "clipped_pixels": np.any(output != unclipped, axis=2), **bands}


def load_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def validate_manifest(manifest, base):
    if manifest.get("schema_version") != 1 or not manifest.get("cases"):
        raise ValueError("Expected FA-02 schema_version=1 and nonempty cases")
    if manifest.get("purpose") not in ("harness_validation", "development_pilot", "locked_comparison"):
        raise ValueError("Explicit experiment purpose required")
    people, sessions, ids, hashes = {}, {}, set(), {}
    for case in manifest["cases"]:
        cid = case["id"]
        if not cid or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in cid) or cid in ids:
            raise ValueError("Case IDs must be unique filesystem-safe tokens")
        ids.add(cid)
        if case["split"] not in ("validation", "dev", "calibration", "locked_test"):
            raise ValueError("Unknown split")
        if case["kind"] not in ("analytic_control", "portrait"):
            raise ValueError("Unknown case kind")
        if manifest["purpose"] == "harness_validation" and case["kind"] != "analytic_control":
            raise ValueError("Harness validation cannot relabel portraits as controls")
        if case["kind"] == "portrait":
            if case["split"] == "validation" or not case.get("person_ids") or not case.get("session_id"):
                raise ValueError("Portrait requires subject/session IDs and an explicit real split")
            for field in ("owner_mapping_reference", "support_acceptance_reference", "native_source_reference",
                          "smoothing_provenance", "color_profile", "source_sha256"):
                if not case.get(field):
                    raise ValueError(f"Portrait missing {field}")
            if case.get("support_status") != "externally_accepted" or case.get("resampled") is not False:
                raise ValueError("Portrait needs accepted supports and native, unresampled pixels")
            for table, keys in ((people, case["person_ids"]), (sessions, [case["session_id"]])):
                for key in keys:
                    if not key or table.setdefault(key, case["split"]) != case["split"]:
                        raise ValueError("Subject/session crosses splits")
            if case.get("previously_inspected") and case["split"] == "locked_test":
                raise ValueError("Previously inspected anchors cannot enter locked_test")
            source_path = (base / case["native_source_reference"]).resolve()
            if digest(source_path) != case["source_sha256"]:
                raise ValueError("Original source checksum mismatch")
        elif case["split"] != "validation" or manifest["purpose"] != "harness_validation":
            raise ValueError("Analytic controls are validation only, never portrait test evidence")
        if manifest["purpose"] == "locked_comparison" and case["split"] != "locked_test":
            raise ValueError("Final comparisons accept locked_test only")
        if manifest["purpose"] == "development_pilot" and case["split"] not in ("dev", "calibration"):
            raise ValueError("Development pilot accepts dev/calibration only")
        if not all(np.isfinite(case[k]) and case[k] > 0 for k in ("inter_eye_distance_px", "face_width_px")):
            raise ValueError("Positive native scales required")
        path = (base/case["arrays"]).resolve()
        if digest(path) != case["arrays_sha256"]:
            raise ValueError("Canvas/support checksum mismatch")
        if hashes.setdefault(case["arrays_sha256"], case["split"]) != case["split"]:
            raise ValueError("Same arrays cross splits")
        arr = load_arrays(path)
        shape = arr["X"].shape
        if len(shape) != 3 or shape[2] != 3 or min(shape[:2]) < 16:
            raise ValueError("Expected HxWx3 canvases")
        for key in ("X", "S"):
            if arr[key].dtype != np.float32 or arr[key].shape != shape or not np.isfinite(arr[key]).all() or arr[key].min() < 0 or arr[key].max() > 255:
                raise ValueError("X/S must be finite matching float32 BGR in [0,255]")
        for key in MASKS:
            if arr[key].shape != shape[:2] or not np.isfinite(arr[key]).all() or arr[key].min() < 0 or arr[key].max() > 1:
                raise ValueError("External masks must be matching HxW arrays in [0,1]")
        if "nuisance" in arr and (arr["nuisance"].shape != shape or not np.isfinite(arr["nuisance"]).all()):
            raise ValueError("Nuisance truth must match the canvas")
        if np.any((arr["corrected"] > 0) & (arr["protected"] > 0)):
            raise ValueError("Corrected and protected supports must be disjoint")
        if len(case["crop_xywh"]) != 4 or case["crop_xywh"][2:] != [shape[1], shape[0]]:
            raise ValueError("Crop metadata does not match canvases")
        for roi in case.get("rois", []):
            if not roi["id"] or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in roi["id"]):
                raise ValueError("ROI IDs must be filesystem-safe")
            x, y, w, h = roi["xywh"]
            if min(x, y) < 0 or min(w, h) <= 0 or x+w > shape[1] or y+h > shape[0]:
                raise ValueError("ROI outside fixed canvas")
    for case in manifest["cases"]:
        if case.get("paired_clean_id") and case["paired_clean_id"] not in ids:
            raise ValueError("Missing paired clean control")
        if case.get("paired_clean_id"):
            clean_case = next(c for c in manifest["cases"] if c["id"] == case["paired_clean_id"])
            a, b = [load_arrays(base/c["arrays"]) for c in (case, clean_case)]
            if not all(np.array_equal(a[k], b[k]) for k in ("S",)+MASKS) or "nuisance" not in a:
                raise ValueError("Paired nuisance controls require identical S/supports and known nuisance")
            if any(case[k] != clean_case[k] for k in ("inter_eye_distance_px", "face_width_px", "crop_xywh", "split")):
                raise ValueError("Paired controls require identical coordinate/scale/split metadata")


def rms(value):
    return float(np.sqrt(np.mean(np.square(value, dtype=np.float64)))) if value.size else None


def projection(value, reference, mask):
    a, b = value[mask].astype(np.float64), reference[mask].astype(np.float64)
    denominator = float(np.sum(b*b))
    return float(np.sum(a*b)/denominator) if denominator > 1e-8 else None


def contrast(y, center, radius):
    yy, xx = np.indices(y.shape)
    r2 = (xx-center[0])**2 + (yy-center[1])**2
    inner, outer = r2 <= radius*radius, (r2 >= (radius*2)**2) & (r2 <= (radius*3)**2)
    return float(y[outer].mean()-y[inner].mean()) if inner.any() and outer.any() else None


def profile(y, p0, p1, samples=65):
    xx = np.linspace(p0[0], p1[0], samples, dtype=np.float32)
    yy = np.linspace(p0[1], p1[1], samples, dtype=np.float32)
    return cv2.remap(y, xx[None], yy[None], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101).ravel()


def profile_shape(values):
    base = float((values[:5].mean()+values[-5:].mean())/2)
    darkness = base-values
    peak = int(np.argmax(darkness))
    height = float(darkness[peak])
    half = darkness > max(height*0.5, 1e-5)
    left = right = peak
    while left > 0 and half[left-1]:
        left -= 1
    while right+1 < len(values) and half[right+1]:
        right += 1
    minima = int(np.sum((values[1:-1] < values[:-2]) & (values[1:-1] < values[2:])))
    return {"contrast": height, "peak_sample": peak, "width_samples": right-left+1,
            "local_minima": minima}


def edge_shape(values):
    left, right = float(values[:5].mean()), float(values[-5:].mean())
    gradient = np.abs(np.diff(values))
    peak = int(np.argmax(gradient))
    return {"gradient_peak_sample": peak+0.5,
            "gradient_half_max_samples": int(np.sum(gradient > max(1e-5, gradient[peak]/2))),
            "undershoot": max(0.0, min(left, right)-float(values.min())),
            "overshoot": max(0.0, float(values.max())-max(left, right)),
            "endpoint_contrast": right-left}


def block_discontinuity(y, origin):
    bx, by = origin
    dx, dy = np.abs(np.diff(y, axis=1)), np.abs(np.diff(y, axis=0))
    ix = (np.arange(1, y.shape[1])-bx) % 8 == 0
    iy = (np.arange(1, y.shape[0])-by) % 8 == 0
    return float((dx[:, ix].mean()+dy[iy].mean()-dx[:, ~ix].mean()-dy[~iy].mean())/2)


def score(case, arr, result, eligibility, geo, cfg):
    delta, output = result["delta"], result["output"]
    dy = luma(delta)
    active = eligibility > 0
    forbidden = (arr["corrected"] > 0) | (arr["protected"] > 0)
    broad = gaussian(delta, cfg["broad_diagnostic_sigma_px"])
    chroma = np.stack((delta[..., 0]-dy, delta[..., 2]-dy), axis=-1)
    metrics = {"eligible_pixels": int(active.sum()), "weighted_coverage": float(eligibility.mean()),
               "eligible_fraction_of_allow": float(active.sum()/max(1, (arr["allow"] > 0).sum())),
               "delta_rms": rms(delta[active]), "delta_luma_mean": float(dy[active].mean()) if active.any() else None,
               "broad_luma_rms": rms(luma(broad)[active]), "chroma_delta_rms": rms(chroma[active]),
               "forbidden_max_abs_delta": float(np.abs(delta[forbidden]).max()) if forbidden.any() else None,
               "clipped_pixel_count": int(result["clipped_pixels"].sum()),
               "pore_truth": "analytic proxy" if case["kind"] == "analytic_control" else "external annotations only",
               "rois": [], "features": [], "profiles": [], "rings": []}
    for roi in case.get("rois", []):
        x, y, w, h = roi["xywh"]
        sl = np.s_[y:y+h, x:x+w]
        local = dy[sl]
        record = {"id": roi["id"], "tags": roi["tags"], "eligible_pixels": int(active[sl].sum()),
                  "delta_luma_rms": rms(local), "delta_luma_mean": float(local.mean()),
                  "broad_luma_rms": rms(luma(broad)[sl]), "chroma_delta_rms": rms(chroma[sl])}
        # This is a variance PROXY, not a biological texture/noise quality score.
        sx, so = luma(arr["S"])[sl], luma(output)[sl]
        record["flat_variance_change_proxy"] = float(np.var(so)-np.var(sx))
        metrics["rois"].append(record)
    for feature in case.get("pores", []):
        values = [contrast(luma(img), feature["center"], feature["radius"]) for img in (arr["X"], arr["S"], output)]
        denom = values[0]-values[1]
        px, py = feature["center"]
        radius = max(2, int(feature["radius"]*3))
        centers, minima = [], []
        for img in (arr["X"], arr["S"], output):
            patch = luma(img)[max(0, py-radius):py+radius+1, max(0, px-radius):px+radius+1]
            centers.append(list(map(int, np.unravel_index(np.argmin(patch), patch.shape))))
            inner = patch[1:-1, 1:-1]
            minima.append(int(np.sum((inner < patch[:-2, 1:-1]) & (inner < patch[2:, 1:-1]) &
                                     (inner < patch[1:-1, :-2]) & (inner < patch[1:-1, 2:]))))
        metrics["features"].append({"id": feature["id"], "kind": "pore_contrast",
            "eligible": bool(active[py, px]), "contrasts_X_S_O": values,
            "signed_recovery": (values[2]-values[1])/denom if abs(denom) > 1e-5 else None,
            "local_minimum_displacement_vs_X_px": math.dist(centers[0], centers[2]),
            "local_minima_counts_X_S_O": minima,
            "note": "Measured only inside supplied feature neighborhoods; not candidate generation"})
    for spec in case.get("profiles", []):
        lines = [profile(luma(img), spec["p0"], spec["p1"]) for img in (arr["X"], arr["S"], output)]
        shapes = [profile_shape(v) for v in lines]
        step = math.dist(spec["p0"], spec["p1"])/(len(lines[0])-1)
        metrics["profiles"].append({**spec, "X_S_O": [v.tolist() for v in lines],
            "shape_X_S_O": shapes, "profile_step_px": step,
            "position_shift_vs_X_px": (shapes[2]["peak_sample"]-shapes[0]["peak_sample"])*step,
            "width_change_vs_X_px": (shapes[2]["width_samples"]-shapes[0]["width_samples"])*step,
            "delta_max": float((lines[2]-lines[1]).max()), "delta_min": float((lines[2]-lines[1]).min())})
        if spec["kind"] == "edge":
            metrics["profiles"][-1]["edge_shape_X_S_O"] = [edge_shape(v) for v in lines]
    metrics["hair_trace_samples"] = []
    for trace in sorted({p["trace_id"] for p in metrics["profiles"] if "trace_id" in p}):
        profiles = [p for p in metrics["profiles"] if p.get("trace_id") == trace]
        observable = [p for p in profiles if p["shape_X_S_O"][0]["contrast"] > 0.1]
        metrics["hair_trace_samples"].append({"trace_id": trace, "observable_sections": len(observable),
            "retained_sections_at_half_source_contrast": sum(
                p["shape_X_S_O"][2]["contrast"] >= 0.5*p["shape_X_S_O"][0]["contrast"] for p in observable),
            "note": "Sparse trace samples, not continuous biological hair truth"})
    # Both before-gate coefficient contamination and AFTER-gate spatial leakage.
    for support in ("corrected", "protected"):
        mask = arr[support] > 0
        if not mask.any():
            continue
        distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_C, 3)
        for start in range(0, geo["common_radius_px"]+17, 4):
            ring = mask if start == 0 else (distance > start-4) & (distance <= start)
            if ring.any():
                metrics["rings"].append({"support": support, "outer_distance_px": start,
                    "pixels": int(ring.sum()), "delta_mean": float(dy[ring].mean()),
                    "delta_min": float(dy[ring].min()), "delta_max": float(dy[ring].max()),
                    "delta_rms": rms(dy[ring]), "pre_gate_detail_rms": rms(result["detail"][ring])})
        inner, outer = mask, (distance >= 3) & (distance <= 7)
        def local_contrast(img):
            yy = luma(img)
            return float(yy[outer].mean()-yy[inner].mean())
        cx, cs, co = [local_contrast(v) for v in (arr["X"], arr["S"], output)]
        metrics[support+"_signed_contrast_recovery"] = (co-cs)/(cx-cs) if abs(cx-cs) > 1e-5 else None
    if "nuisance" in arr:
        metrics["increment_projection_onto_nuisance_confoundable"] = projection(delta, arr["nuisance"], active)
    if "jpeg_origin_xy" in case:
        metrics["jpeg_block_excess_change_proxy"] = block_discontinuity(luma(output), case["jpeg_origin_xy"])-block_discontinuity(luma(arr["S"]), case["jpeg_origin_xy"])
    return metrics


def save_png(path, bgr):
    if not cv2.imwrite(str(path), np.clip(np.rint(bgr), 0, 255).astype(np.uint8)):
        raise IOError(f"Could not write {path}")


def signed_png(path, field, limit):
    # Fixed range across all arms/cases. Gray=0; white=positive; black=negative.
    save_png(path, 127.5 + np.clip(field/limit, -1, 1)*127.5)


def save_diagnostics(folder, case, arr, result, eligibility, metrics, cfg):
    folder.mkdir()
    np.savez_compressed(folder/"signed_arrays.npz", **result, eligibility=eligibility)
    save_png(folder/"output_native.png", result["output"])
    signed_png(folder/"recovered_signed_detail.png", result["delta"], cfg["signed_display_range"])
    signed_png(folder/"pre_gate_signed_detail.png", result["detail"], cfg["signed_display_range"]/cfg["gain"] if cfg["gain"] else 20)
    signed_png(folder/"broad_luma.png", gaussian(luma(result["delta"]), cfg["broad_diagnostic_sigma_px"]), cfg["signed_display_range"])
    dy = luma(result["delta"])
    signed_png(folder/"chroma_BminusY_RminusY.png", np.stack((result["delta"][..., 0]-dy, np.zeros_like(dy), result["delta"][..., 2]-dy), axis=-1), cfg["signed_display_range"])
    for key, value in result.items():
        if key.startswith(("band_", "frequency_")):
            signed_png(folder/(key+"_signed.png"), value, 20.0)
        elif key.startswith("orientation_gate") or key == "coherence":
            save_png(folder/(key+".png"), value*255)
    for roi in case.get("rois", []):
        x, y, w, h = roi["xywh"]
        sl = np.s_[y:y+h, x:x+w]
        save_png(folder/(roi["id"]+"_X_S_O_native.png"), np.concatenate((arr["X"][sl], arr["S"][sl], result["output"][sl]), axis=1))
        plane = dy[sl]
        window = np.outer(np.hanning(h), np.hanning(w))
        psd = np.abs(np.fft.fftshift(np.fft.fft2((plane-plane.mean())*window)))**2/(w*h)
        np.save(folder/(roi["id"]+"_delta_psd.npy"), psd)
        # Fixed log scale: -6..3 log10 intensity^2, not auto-normalized.
        save_png(folder/(roi["id"]+"_delta_psd.png"), (np.clip(np.log10(psd+1e-6), -6, 3)+6)*255/9)
    write_json(folder/"metrics.json", metrics)
    # Simple native-data SVG profiles; display charts never substitute for pixels.
    charts = []
    for item in metrics["profiles"]:
        values = item["X_S_O"]
        lo = min(min(v) for v in values)-1
        hi = max(max(v) for v in values)+1
        lines = []
        for vals, color in zip(values, ("black", "blue", "red")):
            points = " ".join(f"{20+i*8:.1f},{170-(v-lo)/(hi-lo)*140:.1f}" for i, v in enumerate(vals))
            lines.append(f'<polyline fill="none" stroke="{color}" points="{points}"/>')
        charts.append('<svg xmlns="http://www.w3.org/2000/svg" width="560" height="200">'
                      '<rect width="100%" height="100%" fill="white"/>'
                      f'<text x="20" y="18">{html.escape(item["id"])}: X black, S blue, O red; [{lo:.2f},{hi:.2f}]</text>'
                      + "".join(lines) + '</svg>')
    (folder/"profiles.html").write_text("<!doctype html><meta charset='utf-8'>"+"\n".join(charts))
    if metrics["rings"]:
        with (folder/"halo_rings.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics["rings"][0]))
            writer.writeheader()
            writer.writerows(metrics["rings"])
        points = metrics["rings"]
        width = max(p["outer_distance_px"] for p in points)
        paths = []
        for support, color in (("corrected", "red"), ("protected", "blue")):
            selected = [p for p in points if p["support"] == support]
            for field in ("delta_min", "delta_mean", "delta_max"):
                coords = " ".join(f'{20+p["outer_distance_px"]*500/width:.1f},{100-70*np.clip(p[field]/cfg["signed_display_range"], -1, 1):.1f}' for p in selected)
                paths.append(f'<polyline fill="none" stroke="{color}" points="{coords}"/>')
        (folder/"halo_profiles.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" width="560" height="210">'
            '<rect width="100%" height="100%" fill="white"/><path d="M20 100H530" stroke="gray"/>'
            '<text x="20" y="18">Ring min/mean/max; correction red, protection blue</text>'
            + "".join(paths) + f'<text x="20" y="200">0 to {width}px; fixed +/-{cfg["signed_display_range"]} intensity</text></svg>')


def peak_rss_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value/(1024**2 if sys.platform == "darwin" else 1024)


def worker(manifest_path, lock_path, case_id, arm, folder, repeats, measure_only=False):
    """Fresh process per arm/case. RSS excludes all diagnostic writing/scoring.

    Absolute high-water RSS includes interpreter, loaded X/S, imports and shared
    support mapping. Increment is above the pre-compute high-water, not allocated
    transform bytes. Small fixtures may show zero incremental RSS.
    """
    manifest = json.loads(manifest_path.read_text())
    cfg = json.loads(lock_path.read_text())["config"]
    cv2.setNumThreads(cfg["opencv_threads"])
    case = next(c for c in manifest["cases"] if c["id"] == case_id)
    arr = load_arrays(manifest_path.parent/case["arrays"])
    geo = geometry(case, cfg)
    geo["face_width_px"] = case["face_width_px"]
    started = time.perf_counter()
    eligibility = map_external_supports(arr, geo, cfg)
    support_ms = (time.perf_counter()-started)*1000
    before = peak_rss_mib()
    started = time.perf_counter()
    result = restore(arr["X"], arr["S"], arm, geo, cfg, eligibility)
    cold = (time.perf_counter()-started)*1000
    elapsed = []
    for _ in range(repeats):
        del result
        started = time.perf_counter()
        result = restore(arr["X"], arr["S"], arm, geo, cfg, eligibility)
        elapsed.append((time.perf_counter()-started)*1000)
    peak = peak_rss_mib()  # capture BEFORE any metrics/FFT/NPZ/PNG operations
    measurement = {"cold_compute_ms": cold, "warm_compute_ms": elapsed,
        "warm_median_ms": float(np.median(elapsed)), "warm_p95_ms": float(np.percentile(elapsed, 95)),
        "ms_per_megapixel": float(np.median(elapsed)/(arr["X"].shape[0]*arr["X"].shape[1]/1e6)),
        "support_mapping_ms_shared": support_ms, "pre_compute_peak_rss_mib": before,
        "peak_rss_mib": peak, "incremental_high_water_mib": max(0, peak-before),
        "rss_scope": "fresh process; imports+inputs+supports+compute; excludes metrics/output IO"}
    if measure_only:
        write_json(folder, {"arm": arm, "case_id": case_id, "geometry": geo, "measurement": measurement})
        return
    metrics = score(case, arr, result, eligibility, geo, cfg)
    metrics["measurement"] = measurement
    metrics.update(arm=arm, case_id=case_id, geometry=geo)
    save_diagnostics(folder, case, arr, result, eligibility, metrics, cfg)


def freeze_config(config_path, output, declaration, references):
    cfg = json.loads(config_path.read_text()) if config_path else dict(DEFAULT_CONFIG)
    validate_config(cfg)
    if output.exists():
        raise ValueError("Refusing to overwrite an existing config lock")
    write_json(output, {"schema_version": 1, "config": cfg, "config_sha256": canonical_hash(cfg),
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection": declaration, "development_references": references,
        "note": "A timestamp/hash audits settings; it does not prove no prior exposure."})


def transform_diagnostics(folder, cfg):
    """Finite discrete responses; A4 is stimulus-dependent, not a linear MTF."""
    folder.mkdir()
    n = 257
    case = {"inter_eye_distance_px": 200, "face_width_px": 500}
    geo = {**geometry(case, cfg), "face_width_px": 500}
    s = np.full((n, n, 3), 128, np.float32)
    x = s.copy()
    x[n//2, n//2] += 1
    spectra = []
    for arm in ARMS:
        detail, _ = extract_bands(x, s, arm, geo, cfg)
        response = detail[..., 0]
        spectrum = np.abs(np.fft.fft2(np.fft.ifftshift(response)))
        np.savez_compressed(folder/(arm+".npz"), impulse_response=response, amplitude_spectrum=spectrum)
        signed_png(folder/(arm+"_impulse.png"), response, 0.05)
        for i in range(n//2+1):
            spectra.append({"arm": arm, "frequency_cycles_per_px": i/n,
                "horizontal_amplitude": float(spectrum[0, i]),
                "impulse_horizontal_at_offset": i-n//2,
                "impulse_horizontal_value": float(response[n//2, i])})
    with (folder/"discrete_transfer.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(spectra[0]))
        writer.writeheader()
        writer.writerows(spectra)
    write_json(folder/"geometry.json", {**geo, "orientation_note": "A4 impulse response is stimulus-dependent, NOT a linear transfer function"})


def profile_run(manifest_path, lock_path, out, repeats):
    """Repeat compute measurements alone, without image rendering or other tests."""
    manifest = json.loads(manifest_path.read_text())
    lock = json.loads(lock_path.read_text())
    validate_manifest(manifest, manifest_path.parent)
    validate_config(lock["config"])
    if lock["config_sha256"] != canonical_hash(lock["config"]):
        raise ValueError("Config lock hash mismatch")
    if out.exists():
        raise ValueError("Timing output already exists")
    out.mkdir(parents=True)
    measurements = []
    for case in manifest["cases"]:
        for arm in ARMS:
            target = out/(case["id"]+"_"+arm+".json")
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "_worker",
                str(manifest_path.resolve()), str(lock_path.resolve()), case["id"], arm,
                str(target.resolve()), "--repeats", str(repeats), "--measure-only"], check=True)
            measurements.append(json.loads(target.read_text()))
        print(f"profiled {case['id']}", flush=True)
    write_json(out/"TIMINGS.json", {"config_sha256": lock["config_sha256"],
        "manifest_sha256": digest(manifest_path), "runner_sha256": digest(__file__),
        "platform": platform.platform(), "python": sys.version, "opencv": cv2.__version__,
        "note": "Separate serial timing-only run; no diagnostic rendering; native workload sizes in manifest",
        "measurements": measurements})


def run(manifest_path, lock_path, out, repeats):
    manifest = json.loads(manifest_path.read_text())
    lock = json.loads(lock_path.read_text())
    validate_config(lock["config"])
    if lock["config_sha256"] != canonical_hash(lock["config"]):
        raise ValueError("Config lock hash mismatch")
    validate_manifest(manifest, manifest_path.parent)
    if manifest["purpose"] == "locked_comparison":
        if lock["selection"] != "development_selected" or not lock["development_references"]:
            raise ValueError("Final comparison needs development selection provenance")
        if not manifest.get("owner_holdout_acceptance_reference"):
            raise ValueError("Final comparison needs owner confirmation of untouched holdout")
        if manifest.get("accepted_config_sha256") != lock["config_sha256"]:
            raise ValueError("Holdout approval must name the pre-frozen config hash")
    if out.exists():
        raise ValueError("Output already exists: choose a new immutable run directory")
    out.mkdir(parents=True)
    write_json(out/"manifest_snapshot.json", manifest)
    write_json(out/"config_lock.json", lock)
    write_json(out/"environment.json", {"python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "opencv": cv2.__version__, "opencv_threads": 1,
        "runner_sha256": digest(__file__), "frequency_sha256": digest(ROOT/"retouch/frequency.py"),
        "controls_sha256": digest(Path(__file__).with_name("fa02_texture_controls.py")),
        "manifest_sha256": digest(manifest_path), "measurement_repeats": repeats,
        "command": sys.argv, "rendering": "PNG rounded uint8; NPZ float32 authoritative; native dimensions"})
    cv2.setNumThreads(lock["config"]["opencv_threads"])
    transform_diagnostics(out/"transform_checks", lock["config"])
    summary = []
    for case in manifest["cases"]:
        folder = out/case["id"]
        folder.mkdir()
        arr = load_arrays(manifest_path.parent/case["arrays"])
        np.savez_compressed(folder/"inputs.npz", **arr)
        save_png(folder/"X_native.png", arr["X"])
        save_png(folder/"S_native.png", arr["S"])
        geo = geometry(case, lock["config"])
        residual = arr["X"]-arr["S"]
        signed_png(folder/"diagnostic_only_finest.png", residual-gaussian(residual, geo["sigmas_px"][0]), 20)
        signed_png(folder/"diagnostic_only_coarse.png", gaussian(residual, geo["sigmas_px"][-1]), 20)
        eligibility = map_external_supports(arr, geo, lock["config"])
        for key in MASKS:
            save_png(folder/(key+"_support.png"), arr[key]*255)
        save_png(folder/"common_eligibility.png", eligibility*255)
        for arm in ARMS:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "_worker",
                str(manifest_path.resolve()), str(lock_path.resolve()), case["id"], arm,
                str((folder/arm).resolve()), "--repeats", str(repeats)], check=True)
            metrics = json.loads((folder/arm/"metrics.json").read_text())
            summary.append(metrics)
        print(f"completed {case['id']} ({case['kind']}, {case['split']})", flush=True)
    # Paired clean controls isolate nuisance-induced restoration, avoiding
    # correlation between legitimate source texture and the corrupted source.
    for case in manifest["cases"]:
        if not case.get("paired_clean_id"):
            continue
        arr = load_arrays(manifest_path.parent/case["arrays"])
        clean_case = next(c for c in manifest["cases"] if c["id"] == case["paired_clean_id"])
        clean = load_arrays(manifest_path.parent/clean_case["arrays"])
        # These are controlled fixtures only: S and supports MUST match exactly.
        if not all(np.array_equal(arr[k], clean[k]) for k in ("S",)+MASKS):
            raise ValueError("Paired nuisance comparison requires identical S/supports")
        for arm in ARMS:
            folder = out/case["id"]/arm
            result = load_arrays(folder/"signed_arrays.npz")
            reference = load_arrays(out/case["paired_clean_id"]/arm/"signed_arrays.npz")
            causal = result["delta"]-reference["delta"]
            active = result["eligibility"] > 0
            metrics = next(m for m in summary if m["case_id"] == case["id"] and m["arm"] == arm)
            metrics["paired_nuisance"] = {"reference_case": case["paired_clean_id"],
                "increment_rms": rms(causal[active]),
                "increment_projection": projection(causal, arr["nuisance"], active),
                "pre_gate_coefficient_rms": rms((result["detail"]-reference["detail"])[active])}
            pre_gate = result["detail"]-reference["detail"]
            metrics["paired_nuisance"]["rings"] = []
            for support in ("corrected", "protected"):
                mask = arr[support] > 0
                if not mask.any():
                    continue
                distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_C, 3)
                for start in range(4, metrics["geometry"]["common_radius_px"]+17, 4):
                    ring = (distance > start-4) & (distance <= start)
                    if ring.any():
                        metrics["paired_nuisance"]["rings"].append({"support": support,
                            "outer_distance_px": start, "pre_gate_rms": rms(pre_gate[ring]),
                            "post_gate_rms": rms(causal[ring]), "post_gate_signed_mean": float(luma(causal)[ring].mean())})
            np.save(folder/"paired_nuisance_pre_gate.npy", pre_gate)
            signed_png(folder/"paired_nuisance_pre_gate.png", pre_gate, 20.0)
            np.save(folder/"paired_nuisance_delta.npy", causal)
            signed_png(folder/"paired_nuisance_signed.png", causal, lock["config"]["signed_display_range"])
            write_json(folder/"metrics.json", metrics)
    noise_cases = [c for c in manifest["cases"] if c.get("noise_seed") is not None]
    independent_noise = []
    if len(noise_cases) == 2 and noise_cases[0].get("paired_clean_id") == noise_cases[1].get("paired_clean_id"):
        for arm in ARMS:
            a, b = [load_arrays(out/c["id"]/arm/"signed_arrays.npz") for c in noise_cases]
            mask = (a["eligibility"] > 0) & (b["eligibility"] > 0)
            independent_noise.append({"arm": arm, "case_ids": [c["id"] for c in noise_cases],
                "independent_draw_increment_difference_rms": rms((a["delta"]-b["delta"])[mask])})
    write_json(out/"independent_noise.json", independent_noise)
    review = ['<!doctype html><meta charset="utf-8"><title>FA-02 native review</title>',
        '<h1>FA-02 native review</h1><p>No automatic quality ranking. Analytic controls are not portraits. '
        'Each strip is X | S | O at native pixel dimensions; scroll, do not fit-to-page. '
        'PNG is rounded 8-bit; float NPZ is authoritative. Profiles: X black, S blue, O red.</p>']
    for case in manifest["cases"]:
        review.append(f'<h2>{case["id"]} ({case["kind"]})</h2>')
        for roi in case.get("rois", []):
            review.append(f'<h3>{roi["id"]}</h3>')
            for arm in ARMS:
                prefix = f'{case["id"]}/{arm}'
                review.append(f'<p>{arm} <a href="{prefix}/profiles.html">profiles</a> '
                              f'<a href="{prefix}/metrics.json">metrics</a></p>'
                              f'<img src="{prefix}/{roi["id"]}_X_S_O_native.png" alt="X S O">')
    (out/"native_review.html").write_text("\n".join(review))
    write_json(out/"summary.json", {"purpose": manifest["purpose"], "results": summary,
        "portrait_quality_winner": None, "automatic_ranking": "disabled: owner native review required"})
    fields = ["case_id", "arm", "eligible_pixels", "weighted_coverage", "delta_rms", "broad_luma_rms",
              "chroma_delta_rms", "forbidden_max_abs_delta", "warm_median_ms", "peak_rss_mib"]
    with (out/"summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            flat = {**row, **row["measurement"]}
            writer.writerow({k: flat[k] for k in fields})
    write_json(out/"COMPLETE.json", {"cases": len(manifest["cases"]), "arms": len(ARMS),
        "config_sha256": lock["config_sha256"], "manifest_sha256": digest(manifest_path)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    controls = commands.add_parser("controls", help="Create mathematical validation fixtures only")
    controls.add_argument("output", type=Path)
    freeze = commands.add_parser("freeze", help="Freeze config BEFORE evaluating report patches")
    freeze.add_argument("output", type=Path)
    freeze.add_argument("--config", type=Path)
    freeze.add_argument("--selection", choices=("a_priori_validation", "development_selected"), default="a_priori_validation")
    freeze.add_argument("--development-reference", action="append", default=[])
    runner = commands.add_parser("run")
    runner.add_argument("manifest", type=Path)
    runner.add_argument("lock", type=Path)
    runner.add_argument("output", type=Path)
    runner.add_argument("--repeats", type=int, default=5)
    profiler = commands.add_parser("profile", help="Timing-only repeat, run without concurrent tests")
    profiler.add_argument("manifest", type=Path)
    profiler.add_argument("lock", type=Path)
    profiler.add_argument("output", type=Path)
    profiler.add_argument("--repeats", type=int, default=10)
    child = commands.add_parser("_worker")
    for arg in ("manifest", "lock"):
        child.add_argument(arg, type=Path)
    child.add_argument("case")
    child.add_argument("arm", choices=ARMS)
    child.add_argument("output", type=Path)
    child.add_argument("--repeats", type=int, default=5)
    child.add_argument("--measure-only", action="store_true")
    args = parser.parse_args(argv)
    if getattr(args, "repeats", 1) < 1:
        parser.error("At least one warm repeat required")
    if args.command == "controls":
        from fa02_texture_controls import create_controls
        create_controls(args.output)
    elif args.command == "freeze":
        freeze_config(args.config, args.output, args.selection, args.development_reference)
    elif args.command == "run":
        run(args.manifest, args.lock, args.output, args.repeats)
    elif args.command == "profile":
        profile_run(args.manifest, args.lock, args.output, args.repeats)
    else:
        worker(args.manifest, args.lock, args.case, args.arm, args.output, args.repeats, args.measure_only)


if __name__ == "__main__":
    main()

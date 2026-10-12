"""P5 analytical self-blend research harness. No production imports/wiring.

prepare DIR       -> tagged RGB ramps + manifest, never Photoshop goldens
validate DIR      -> independent two-input algebra, float/LUT/domain diagnostics
compare GOLDEN_DIR OUTPUT -> discover baseline.png + mode_025/050/075/100.png

Use p5_photoshop_capture.jsx in Photoshop to collect a capture. Missing captures
are an explicit NO-GO, not zero-error results. See the P5 research report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import struct
import zlib

import cv2
import numpy as np
from PIL import Image, ImageCms

MODES = ("multiply", "screen", "overlay", "hard_light", "soft_light",
         "color_dodge", "color_burn", "exclusion", "linear_light", "vivid_light")
OPACITIES = (0.25, 0.5, 0.75, 1.0)
MODELS = ("encoded", "linear", "encoded_blend_linear_opacity", "linear_blend_encoded_opacity")
ROUNDINGS = ("nearest_half_up", "floor")


def unit(value):
    a = np.asarray(value)
    if a.dtype not in (np.dtype("float32"), np.dtype("float64")):
        a = a.astype(np.float64)
    if a.size == 0 or not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
        raise ValueError("Expected nonempty finite normalized values in [0,1]")
    return a


def self_transfer(value, mode):
    """Reduced f(x)=B(x,x), explicitly clipped only where the operator requires."""
    x = unit(value)
    if mode == "multiply":
        return x*x
    if mode == "screen":
        return x*(2-x)
    if mode in ("overlay", "hard_light"):
        return np.where(x <= 0.5, 2*x*x, 1-2*(1-x)*(1-x))
    if mode == "soft_light":
        return np.where(x <= 0.5, x*x*(3-2*x), x+(2*x-1)*(np.sqrt(x)-x))
    if mode == "exclusion":
        return 2*x*(1-x)
    if mode == "linear_light":
        return np.clip(3*x-1, 0, 1)
    if mode == "color_dodge":
        y = np.ones_like(x)
        np.divide(x, 1-x, out=y, where=x < 0.5)
        return y
    if mode == "color_burn":
        y = np.zeros_like(x)
        selected = x > 0.5
        y[selected] = 2-1/x[selected]
        return y
    if mode == "vivid_light":
        y = np.zeros_like(x)
        lower = (x > 1/3) & (x <= 0.5)
        upper = (x > 0.5) & (x < 2/3)
        y[lower] = (3*x[lower]-1)/(2*x[lower])
        y[upper] = x[upper]/(2*(1-x[upper]))
        y[x >= 2/3] = 1
        return y
    raise ValueError(f"Unknown mode: {mode}")


def blend(backdrop, source, mode):
    """Independent two-input reference (W3C; LL/VL canonical candidates).

    Deliberately does not call self_transfer. Endpoint precedence matches W3C
    ColorDodge/ColorBurn, including the off-diagonal singular corners.
    """
    b, s = np.broadcast_arrays(unit(backdrop), unit(source))
    if mode == "multiply":
        return b*s
    if mode == "screen":
        return b+s-b*s
    if mode == "overlay":
        return blend(s, b, "hard_light")
    if mode == "hard_light":
        return np.where(s <= 0.5, b*(2*s), 1-(1-b)*(1-(2*s-1)))
    if mode == "soft_light":
        d = np.where(b <= 0.25, ((16*b-12)*b+4)*b, np.sqrt(b))
        return np.where(s <= 0.5, b-(1-2*s)*b*(1-b), b+(2*s-1)*(d-b))
    if mode == "color_dodge":
        y = np.zeros_like(b)
        # Saturate BEFORE division; even a subnormal denominator stays finite.
        saturated = (b != 0) & (b >= 1-s)
        y[saturated] = 1
        np.divide(b, 1-s, out=y, where=(b != 0) & ~saturated)
        return y
    if mode == "color_burn":
        y = np.zeros_like(b)
        selected = (b != 1) & (s > 1-b)
        ratio = np.zeros_like(b)
        np.divide(1-b, s, out=ratio, where=selected)
        y[selected] = 1-np.minimum(1, ratio[selected])
        y[b == 1] = 1
        return y
    if mode == "exclusion":
        return b+s-2*b*s
    if mode == "linear_light":
        return np.clip(b+2*s-1, 0, 1)
    if mode == "vivid_light":
        # Split before evaluation: inactive branches must never divide by zero.
        y = np.empty_like(b)
        low = s <= 0.5
        y[low] = blend(b[low], 2*s[low], "color_burn") if low.any() else 0
        high = ~low
        if high.any():
            y[high] = blend(b[high], 2*s[high]-1, "color_dodge")
        return y
    raise ValueError(f"Unknown mode: {mode}")


def decode(value, transfer="srgb", gamma=None):
    x = unit(value)
    if transfer == "srgb":
        return np.where(x <= 0.04045, x/12.92, ((x+0.055)/1.055)**2.4)
    if transfer == "gamma" and gamma is not None and np.isfinite(gamma) and gamma > 0:
        return x**gamma
    raise ValueError("Expected srgb or an explicit positive gamma transfer")


def encode(value, transfer="srgb", gamma=None):
    x = unit(value)
    if transfer == "srgb":
        return np.where(x <= 0.0031308, 12.92*x, 1.055*x**(1/2.4)-0.055)
    if transfer == "gamma" and gamma is not None and np.isfinite(gamma) and gamma > 0:
        return x**(1/gamma)
    raise ValueError("Expected srgb or an explicit positive gamma transfer")


def predict(value, mode, opacity=1.0, model="encoded", *, transfer="srgb", gamma=None, mask=None):
    """Opaque equal layers; blend domain and opacity domain are separate factors.

    No arbitrary ICC conversion, alpha compositing, Fill, Blend If or HDR. A mask
    here only multiplies opacity of an opaque source over an opaque backdrop.
    """
    x = unit(value)
    if mode not in MODES or model not in MODELS:
        raise ValueError("Unknown mode or domain model")
    if not np.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError("Opacity must be in [0,1]")
    if opacity == 0:
        return x.copy()
    p = opacity
    if mask is not None:
        m = unit(mask)
        if m.shape != x.shape[:-1]:
            raise ValueError("Mask must match RGB image spatial shape")
        p = opacity*m[..., None]
    if model in ("linear", "linear_blend_encoded_opacity"):
        mixed = encode(self_transfer(decode(x, transfer, gamma), mode), transfer, gamma)
    else:
        mixed = self_transfer(x, mode)
    if model in ("linear", "encoded_blend_linear_opacity"):
        result = encode((1-p)*decode(x, transfer, gamma)+p*decode(mixed, transfer, gamma), transfer, gamma)
    else:
        result = (1-p)*x+p*mixed
    # Exact identity outside masks; not an emergency clamp hiding invalid math.
    return np.where(p == 0, x, result)


def quantize(value, bits, rounding="nearest_half_up"):
    x = unit(value)
    if bits not in (8, 16) or rounding not in ROUNDINGS:
        raise ValueError("Expected 8/16 bits and supported rounding policy")
    scaled = x.astype(np.float64)*((1 << bits)-1)
    # Stabilize an exact half-DN tie displaced downward by one arithmetic ULP
    # (e.g. Dodge at 85/255). This policy is explicit, not Photoshop inference.
    q = np.floor(np.nextafter(scaled+0.5, np.inf)) if rounding == "nearest_half_up" else np.floor(scaled)
    return q.astype(np.uint8 if bits == 8 else np.uint16)


def errors(actual, reference, baseline=None, limit=10):
    if actual.shape != reference.shape or actual.size == 0:
        raise ValueError("Metrics require matching nonempty shapes")
    delta = actual.astype(np.float64)-reference.astype(np.float64)
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite metric input")
    absolute = np.abs(delta)
    pixel_error = absolute.max(axis=-1) if delta.ndim >= 3 else absolute
    if not 1 <= limit <= 20:
        raise ValueError("Mismatch location limit must be between 1 and 20")
    if baseline is not None and baseline.shape != actual.shape:
        raise ValueError("Input shape must match compared images")
    def locations(indices):
        result = []
        for index in indices[:limit]:
            coord = np.unravel_index(index, delta.shape)
            y, x, channel = coord if delta.ndim == 3 else (0, int(index), 0)
            result.append({"x": int(x), "y": int(y), "channel": "RGB"[channel],
                "input_DN": float(baseline[coord]) if baseline is not None else None,
                "predicted_DN": float(actual[coord]), "actual_DN": float(reference[coord]),
                "signed_error_DN": float(delta[coord])})
        return result
    maxima = np.flatnonzero(absolute.ravel() == absolute.max()) if absolute.max() > 0 else np.array([], dtype=int)
    return {"max_error_DN": float(absolute.max()), "MAE_DN": float(absolute.mean()),
            "RMSE_DN": float(np.sqrt(np.mean(delta*delta))),
            "pixels_over_1_DN": int((pixel_error > 1).sum()),
            "channel_samples_over_1_DN": int((absolute > 1).sum()),
            "pixel_count": int(pixel_error.size), "fraction_pixels_over_1_DN": float((pixel_error > 1).mean()),
            "first_mismatches": locations(np.flatnonzero(absolute.ravel() > 0)),
            "first_mismatches_over_1_DN": locations(np.flatnonzero(absolute.ravel() > 1)),
            "max_error_locations": locations(maxima), "max_error_location_count": int(maxima.size),
            "location_limit": limit, "location_order": "row-major y,x,RGB; signed error = predicted-actual"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def new_folder(path):
    path = Path(path)
    if path.exists():
        raise ValueError("Refusing to overwrite existing experiment artifacts")
    path.mkdir(parents=True)
    return path


def save_rgb(path, rgb, icc=None):
    ok, encoded = cv2.imencode(".png", rgb[..., ::-1])
    if not ok:
        raise IOError("PNG encoding failed")
    payload = encoded.tobytes()
    if icc:
        # Preserve uint16 RGB samples: insert an ICC chunk without PIL conversion.
        data = b"sRGB\0\0"+zlib.compress(icc)
        body = b"iCCP"+data
        chunk = struct.pack(">I", len(data))+body+struct.pack(">I", zlib.crc32(body)&0xffffffff)
        payload = payload[:33]+chunk+payload[33:]  # after PNG signature + IHDR
    Path(path).write_bytes(payload)


def load_rgb(path):
    path = Path(path)
    if path.suffix.lower() not in (".png", ".tif", ".tiff"):
        raise ValueError("Use lossless PNG/TIFF goldens, never screenshots/JPEG")
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None or a.ndim != 3 or a.shape[2] != 3 or a.dtype not in (np.uint8, np.uint16):
        raise ValueError("Expected opaque 8/16-bit three-channel RGB")
    with Image.open(path) as image:
        orientation = image.getexif().get(274, 1)
        if orientation != 1:
            raise ValueError("Orientation must be baked/normal; no silent rotation")
        icc = image.info.get("icc_profile", b"")
    return a[..., ::-1].copy(), hashlib.sha256(icc).hexdigest() if icc else None


def atlas(bits):
    if bits not in (8, 16):
        raise ValueError("Atlas requires 8 or 16 bits")
    maximum = (1 << bits)-1
    values = np.arange(maximum+1, dtype=np.uint32)
    if bits == 8:
        values = np.tile(values, 64)
    ramp = values.reshape(-1, 1024)
    zero, middle, white = np.zeros_like(ramp), np.full_like(ramp, (maximum+1)//2), np.full_like(ramp, maximum)
    tiles = [np.stack(c, axis=-1) for c in (
        (ramp, ramp, ramp), (ramp, zero, zero), (zero, ramp, zero), (zero, zero, ramp),
        (ramp, middle, white), (white, ramp, middle), (middle, white, ramp),
        (ramp, (ramp*73) % (maximum+1), (ramp*151) % (maximum+1)),
        (ramp//16, ramp//16, ramp//16),
        (maximum-ramp//16, maximum-ramp//16, maximum-ramp//16))]
    return np.concatenate(tiles).astype(np.uint8 if bits == 8 else np.uint16)


def prepare(folder):
    out = new_folder(folder)
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    records = []
    for bits in (8, 16):
        image = atlas(bits)
        path = out/f"rgb_atlas_{bits}.png"
        save_rgb(path, image, profile)
        recovered, profile_hash = load_rgb(path)
        if not np.array_equal(recovered, image):
            raise AssertionError("Fixture encoding changed samples")
        records.append({"path": path.name, "sha256": sha(path), "bits": bits,
                        "shape": list(image.shape), "icc_sha256": profile_hash,
                        "tile_height": image.shape[0]//10})
    write_json(out/"fixtures.json", {"producer": "analytic_fixture_not_photoshop",
        "profile": "sRGB IEC61966-2.1 (LittleCMS generated)", "fixtures": records,
        "tiles": ["gray", "red", "green", "blue", "red_context", "green_context", "blue_context", "mixed_rgb", "near_black", "near_white"],
        "modes": MODES, "opacities": OPACITIES,
        "goldens": [], "photoshop_parity": "NOT_MEASURED"})
    print(out/"fixtures.json")


def lookup(value, mode, size):
    if size < 2:
        raise ValueError("LUT needs at least two entries")
    x = unit(value)
    grid = np.linspace(0, 1, size)
    # No monotonic constraint: Exclusion's descending half is intentional.
    return np.interp(x, grid, self_transfer(grid, mode))


def validate(folder):
    out = new_folder(folder)
    grid = np.linspace(0, 1, 65536)
    x = np.unique(np.concatenate((grid, boundary_probes(), [.1, .2, .75, .9])))
    rows, luts = [], []
    # Subnormal products can correctly round to zero. Invalid/divide/overflow
    # remain errors; underflow is not an undefined or nonfinite endpoint.
    with np.errstate(divide="raise", invalid="raise", over="raise", under="ignore"):
        for mode in MODES:
            actual = self_transfer(x, mode)
            reference = blend(x, x, mode)
            np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-12)
            if not np.isfinite(actual).all() or actual.min() < 0 or actual.max() > 1:
                raise AssertionError("Invalid self-transfer output")
            row = {"mode": mode, "formula_max_abs_error": float(np.max(np.abs(actual-reference))),
                   "float32_max_error_8bit_DN": float(np.max(np.abs(self_transfer(x.astype(np.float32), mode)-actual))*255),
                   "decreasing_intervals": int((np.diff(actual) < -1e-12).sum())}
            for p in OPACITIES:
                expected = (1-p)*x+p*reference
                np.testing.assert_allclose(predict(x, mode, p), expected, rtol=0, atol=1e-12)
                row[f"opacity_{p}_max_abs_error"] = float(np.max(np.abs(predict(x, mode, p)-expected)))
            rows.append(row)
            for size in (4096, 65536):
                # Off-grid samples are essential: a LUT matching its own nodes proves little.
                probes = np.linspace(0, 1, 200003)
                luts.append({"mode": mode, "entries": size,
                    "max_error_8bit_DN": float(np.max(np.abs(lookup(probes, mode, size)-self_transfer(probes, mode)))*255),
                    "max_error_16bit_DN": float(np.max(np.abs(lookup(probes, mode, size)-self_transfer(probes, mode)))*65535)})
    points = []
    for bits, codes in ((8, (0, 1, 51, 63, 64, 65, 84, 85, 86, 127, 128, 129, 169, 170, 171, 192, 254, 255)),
                        (16, (0, 1, 16383, 16384, 16385, 21844, 21845, 21846, 32767, 32768, 32769, 43689, 43690, 43691, 65534, 65535))):
        for code in codes:
            for mode in MODES:
                v = self_transfer(np.array(code/((1 << bits)-1)), mode)
                points.append({"mode": mode, "bits": bits, "input_DN": code, "output_float_DN": float(v*((1 << bits)-1)),
                    "nearest_half_up": int(quantize(v, bits)), "floor": int(quantize(v, bits, "floor"))})
    # Domain hypotheses are not Photoshop measurements.
    domains = []
    for bits in (8, 16):
        gray = np.repeat(np.arange(1 << bits, dtype=np.uint8 if bits == 8 else np.uint16)[None, :, None], 3, axis=2)
        for sample_set, input_dn in (("gray_ramp", gray), ("rgb_atlas", atlas(bits))):
            src = input_dn.astype(np.float64)/((1 << bits)-1)
            for mode in MODES:
                for p in OPACITIES:
                    base = quantize(predict(src, mode, p), bits)
                    for model in MODELS[1:]:
                        predicted = quantize(predict(src, mode, p, model), bits)
                        domains.append({"mode": mode, "bits": bits, "sample_set": sample_set,
                                        "opacity": p, "model": model, **errors(predicted, base, input_dn)})
    write_json(out/"analytical_validation.json", {"producer": "analytic_not_photoshop",
        "samples_per_mode": len(x), "results": rows, "LUT_results": luts,
        "reference_points": points, "domain_differences_vs_encoded": domains,
        "photoshop_parity": "NOT_MEASURED", "production_integration": "NO_GO",
        "runner_sha256": sha(__file__), "numpy": np.__version__, "opencv": cv2.__version__})
    for name, records in (("reference_points", points), ("lut_errors", luts), ("domain_differences", domains)):
        with (out/(name+".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows([{k: json.dumps(v) if isinstance(v, (list, dict)) else v
                              for k, v in row.items()} for row in records])
    # Native analytic outputs: clearly named as predictions, not fake goldens.
    review = ['<!doctype html><meta charset="utf-8"><h1>Analytical predictions, NOT Photoshop goldens</h1>']
    ramp = np.repeat(np.linspace(0, 1, 1024)[None, :, None], 64, axis=0)
    ramp = np.repeat(ramp, 3, axis=2)
    for mode in MODES:
        variants = np.concatenate([predict(ramp, mode, p) for p in OPACITIES], axis=0)
        save_rgb(out/(mode+"_predicted.png"), quantize(variants, 8))
        curves = []
        for p, color in zip(OPACITIES, ("green", "blue", "orange", "red")):
            xx = np.linspace(0, 1, 513)
            yy = predict(xx, mode, p)
            coords = " ".join(f"{20+a*256:.3f},{276-b*256:.3f}" for a, b in zip(xx, yy))
            curves.append(f'<polyline fill="none" stroke="{color}" points="{coords}"/>')
        svg = '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="305"><rect width="100%" height="100%" fill="white"/><path d="M20 276L276 20" stroke="gray"/>'+"".join(curves)+f'<text x="20" y="298">{mode}</text></svg>'
        (out/(mode+".svg")).write_text(svg)
        review.append(f'<h2>{mode}</h2><img src="{mode}.svg"><p>25/50/75/100%, top to bottom</p><img src="{mode}_predicted.png">')
    (out/"analytic_review.html").write_text("\n".join(review))
    print(out/"analytical_validation.json")


def boundary_probes(dtype=np.float64):
    values = np.array([0, .25, 1/3, .5, 2/3, 1], dtype=dtype)
    with np.errstate(under="ignore"):
        return np.unique(np.concatenate((values, np.nextafter(values[:-1], dtype(1)),
                                        np.nextafter(values[1:], dtype(0)))))


def local_asset(root, name):
    path = (root/name).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("Capture assets must stay within the golden directory")
    return path


def discover_goldens(path):
    """No hand-authored case manifest. Partial sets are scored, never certified."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Golden directory/metadata does not exist: {path}")
    root = path if path.is_dir() else path.parent
    metadata_path = None
    if path.is_file():
        metadata_path = path
    else:
        found = [root/n for n in ("capture_metadata.json", "external_golden_metadata.json") if (root/n).exists()]
        if len(found) > 1:
            raise ValueError("Multiple metadata files: pass the intended JSON explicitly")
        if found:
            metadata_path = found[0]
    try:
        capture = json.loads(metadata_path.read_text()) if metadata_path else {}
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        # Broken provenance is not a reason to discard otherwise usable pixels.
        capture = {"metadata_read_error": str(error)}
    if not isinstance(capture, dict):
        raise ValueError("Capture metadata must be an object")
    def one_asset(stem, required=False):
        items = [p for p in root.iterdir() if p.stem.lower() == stem and p.suffix.lower() in (".png", ".tif", ".tiff")]
        if len(items) > 1:
            raise ValueError(f"Duplicate asset for {stem}")
        if required and not items:
            raise ValueError(f"Missing {stem}.png/.tif baseline")
        return items[0].name if items else None
    baseline = capture.get("baseline") or one_asset("baseline", required=True)
    normal = capture.get("normal_control") or one_asset("normal_100")
    records = []
    pattern = re.compile(r"("+"|".join(MODES)+r")_(025|050|075|100)$")
    keys = set()
    if "cases" in capture:
        candidates = capture["cases"]  # Legacy/script manifest also remains supported.
        if not isinstance(candidates, list):
            raise ValueError("Metadata cases must be a list")
    else:
        candidates = []
        for file in sorted(root.iterdir()):
            if file.suffix.lower() not in (".png", ".tif", ".tiff"):
                continue
            match = pattern.fullmatch(file.stem)
            if match:
                candidates.append({"mode": match[1], "opacity": int(match[2])/100, "file": file.name})
            elif any(file.stem.startswith(m+"_") for m in MODES):
                raise ValueError(f"Malformed golden filename: {file.name}; use e.g. multiply_050.png")
    for record in candidates:
        if not isinstance(record, dict) or not all(k in record for k in ("mode", "opacity", "file")):
            raise ValueError("Each golden case needs mode, opacity, and file")
        key = (record["mode"], record["opacity"])
        if key[0] not in MODES or key[1] not in OPACITIES or key in keys:
            raise ValueError("Unknown or duplicate mode/opacity golden")
        if not local_asset(root, record["file"]).is_file():
            raise ValueError(f"Missing listed golden: {record['file']}")
        keys.add(key)
        records.append(record)
    if not records:
        raise ValueError("No mode_NNN.png/.tif golden images found")
    missing = [{"mode": m, "opacity": p} for m in MODES for p in OPACITIES if (m, p) not in keys]
    records.sort(key=lambda r: (MODES.index(r["mode"]), r["opacity"]))
    capture = {**capture, "baseline": baseline, "normal_control": normal, "cases": records}
    return root, capture, metadata_path, missing


def assess_capture(capture, root, missing):
    """Invalid provenance DOWNGRADES evidence; incompatible pixel grids fail."""
    base, profile = load_rgb(local_asset(root, capture["baseline"]))
    bits = 8 if base.dtype == np.uint8 else 16
    issues = []
    def known(value):
        return isinstance(value, str) and value.strip().upper() not in ("", "UNKNOWN", "UNAVAILABLE", "NOT_RECORDED")
    if capture.get("producer") not in ("adobe_photoshop", "external_photoshop_exports"):
        issues.append("No Photoshop producer attestation")
    if capture.get("capture_status") != "complete":
        issues.append("Capture completion not attested")
    for field in ("photoshop_version", "document_profile", "capture_notes", "operator"):
        if not known(capture.get(field)):
            issues.append(f"Missing/unknown {field}")
    if capture.get("settings_confirmed") is not True:
        issues.append("Settings not confirmed")
    gamma = capture.get("blend_gamma_setting", {})
    if not isinstance(gamma, dict) or type(gamma.get("enabled")) is not bool:
        issues.append("Blend gamma enablement UNKNOWN/UNAVAILABLE")
    elif gamma["enabled"] and (type(gamma.get("gamma")) not in (int, float) or not np.isfinite(gamma["gamma"]) or gamma["gamma"] <= 0):
        issues.append("Enabled blend gamma has no valid numeric value")
    evidence = capture.get("settings_evidence")
    if not evidence or not local_asset(root, evidence).is_file():
        issues.append("No attached settings evidence for attestation")
    for key, expected in (("fill_opacity", 100), ("opaque_equal_layers", True),
                          ("blend_if", "disabled"), ("layer_effects", False), ("masks", "none")):
        if type(capture.get(key)) != type(expected) or capture[key] != expected:
            issues.append(f"Unconfirmed capture constraint: {key}")
    if capture.get("bits") != bits:
        issues.append("Missing/inconsistent bit-depth metadata")
    if profile is None:
        issues.append("No embedded baseline ICC profile")
    # A uniform image can pass every operator accidentally. A parity claim must
    # include the full gray ramp AND the pure R/G/B ramps at the captured depth.
    gray_pixels = np.all(base == base[..., :1], axis=-1)
    if np.unique(base[..., 0][gray_pixels]).size != 1 << bits:
        issues.append("Baseline lacks exhaustive grayscale input codes")
    for channel in range(3):
        others = [c for c in range(3) if c != channel]
        pure = np.all(base[..., others] == 0, axis=-1)
        if np.unique(base[..., channel][pure]).size != 1 << bits:
            issues.append(f"Baseline lacks exhaustive pure {'RGB'[channel]} ramp")
    if capture.get("source_fixture"):
        original, original_icc = load_rgb(local_asset(root, capture["source_fixture"]))
        if original_icc != profile or not np.array_equal(original, base):
            issues.append("Source fixture changed on import/baseline export (samples or ICC)")
    for record in capture["cases"]:
        image, icc = load_rgb(local_asset(root, record["file"]))
        if image.shape != base.shape or image.dtype != base.dtype:
            raise ValueError(f"Golden dimensions/bit depth differ: {record['file']}; no resampling permitted")
        if icc != profile:
            issues.append(f"ICC mismatch: {record['file']}")
    if capture.get("normal_control") and local_asset(root, capture["normal_control"]).is_file():
        normal, icc = load_rgb(local_asset(root, capture["normal_control"]))
        if icc != profile or not np.array_equal(normal, base):
            issues.append("Normal duplicate control differs from baseline")
    else:
        issues.append("Normal duplicate control missing")
    if missing:
        issues.append(f"Incomplete set: {len(missing)} of 40 mode/opacity pairs missing")
    return base, bits, issues


def compare(capture_path, folder, transfer="srgb", gamma=None, require_complete=False, save_error_images=False):
    root, capture, metadata_path, missing = discover_goldens(capture_path)
    if require_complete and missing:
        raise ValueError(f"Incomplete golden set: {len(missing)} missing")
    base, bits, issues = assess_capture(capture, root, missing)
    # An unknown TRC is still comparable, but cannot certify a domain hypothesis.
    profile_name = capture.get("document_profile", "UNKNOWN")
    if transfer == "srgb" and "srgb" not in profile_name.lower():
        issues.append("sRGB transfer is only an unverified hypothesis for this profile")
    if transfer == "gamma":
        issues.append("User-supplied power gamma is a hypothesis, not an ICC-profile verification")
    decode(np.array([.5]), transfer, gamma)
    out = new_folder(folder)
    source = base.astype(np.float64)/((1 << bits)-1)
    records, assets = [], []
    for item in capture["cases"]:
        actual, _ = load_rgb(local_asset(root, item["file"]))
        assets.append({"path": item["file"], "sha256": sha(local_asset(root, item["file"]))})
        for model in MODELS:
            normalized = predict(source, item["mode"], item["opacity"], model, transfer=transfer, gamma=gamma)
            for rounding in ROUNDINGS:
                predicted = quantize(normalized, bits, rounding)
                metrics = errors(predicted, actual, base)
                records.append({"mode": item["mode"], "opacity": item["opacity"],
                    "domain_hypothesis": model, "rounding": rounding, **metrics,
                    "within_1_DN": metrics["max_error_DN"] <= 1})
                if save_error_images:
                    stem = f'{item["mode"]}_{int(item["opacity"]*100):03d}_{model}_{rounding}'
                    difference = predicted.astype(np.int32)-actual.astype(np.int32)
                    np.save(out/(stem+"_error_DN.npy"), difference)
                    vis = np.clip(128+difference.astype(np.float64)*16, 0, 255).astype(np.uint8)
                    save_rgb(out/(stem+"_error_16x.png"), vis)
    summaries = []
    for model in MODELS:
        for rounding in ROUNDINGS:
            subset = [r for r in records if r["domain_hypothesis"] == model and r["rounding"] == rounding]
            summaries.append({"model": model, "rounding": rounding,
                "all_40_within_1_DN": len(subset) == 40 and all(r["within_1_DN"] for r in subset),
                "worst_max_error_DN": max(r["max_error_DN"] for r in subset)})
    verified = not issues and any(s["all_40_within_1_DN"] for s in summaries)
    report = {"capture_metadata": capture, "missing_cases": missing, "provenance_issues": issues,
        "evidence_class": "PHOTOSHOP_VERIFIED_CAPTURE_SCOPE_ONLY" if verified else "EXTERNAL_IMAGE_COMPARISON_PHOTOSHOP_UNVERIFIED",
        "provenance_attested": not issues, "photoshop_verified": verified,
        "capture_sha256": sha(metadata_path) if metadata_path else None,
        "baseline_sha256": sha(local_asset(root, capture["baseline"])),
        "source_fixture_sha256": sha(local_asset(root, capture["source_fixture"])) if capture.get("source_fixture") else None,
        "normal_sha256": sha(local_asset(root, capture["normal_control"])) if capture.get("normal_control") and local_asset(root, capture["normal_control"]).is_file() else None,
        "settings_evidence_sha256": sha(local_asset(root, capture["settings_evidence"])) if capture.get("settings_evidence") and local_asset(root, capture["settings_evidence"]).is_file() else None,
        "assets": assets, "bits": bits,
        "transfer_hypothesis": transfer, "gamma": gamma, "results": records,
        "domain_summaries": summaries, "runner_sha256": sha(__file__),
        "production_integration": "NO_GO_PENDING_CROSS_CONFIGURATION_REVIEW",
        "note": "Attestation is not cryptographic proof of origin. One passing capture does not establish other versions/profiles/bit depths or Fill semantics"}
    write_json(out/"photoshop_comparison.json", report)
    with (out/"photoshop_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows([{k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in records])
    print(out/"photoshop_comparison.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "validate"):
        commands.add_parser(name).add_argument("output", type=Path)
    command = commands.add_parser("compare")
    command.add_argument("capture", type=Path)
    command.add_argument("output", type=Path)
    command.add_argument("--transfer", choices=("srgb", "gamma"), default="srgb")
    command.add_argument("--gamma", type=float)
    command.add_argument("--require-complete", action="store_true")
    command.add_argument("--save-error-images", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.output)
    elif args.command == "validate":
        validate(args.output)
    else:
        compare(args.capture, args.output, args.transfer, args.gamma, args.require_complete, args.save_error_images)


if __name__ == "__main__":
    main()

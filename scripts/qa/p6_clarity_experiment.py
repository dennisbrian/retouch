"""P6 OFFLINE RESEARCH: characterize current clarity and bounded gain candidates.

No production wiring, learned models, detection, retouch recipes or source writes.
Constants below are declared before evaluation; seeds 11/29/47 are reporting seeds.
Native photo ROIs are one-subject exploratory evidence, not a held-out benchmark.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from retouch.grading import ColorGrader
from retouch.utils import bgr_f32_to_lab_f32, lab_f32_to_bgr_f32

VALUES = (0, 1, 2, 4, 6, 8, 10)
SEEDS = (11, 29, 47)
ARMS = ("disabled", "conversion_only", "production_current", "matched_current",
        "soft_floor", "variance_snr", "multiscale_snr", "snr_edge_cap", "edge_coherence")
ROIS = {
    "sky": (150, 120, 950, 620),
    "soft_background": (2950, 2650, 3350, 3050),
    "skin_nose": (1620, 1810, 1790, 1940),
    "skin_cheek": (1790, 1780, 1890, 1890),
    "hair": (1300, 1220, 1680, 1520),
    "eye_lash_makeup": (1420, 1560, 1930, 1770),
    "costume_fabric": (680, 3270, 1080, 3710),
    "hair_sky_boundary": (1770, 580, 2130, 980),
    "wig_background_boundary": (2020, 1900, 2440, 2340),
    "face_wig_boundary": (1430, 1890, 1690, 2150),
    "costume_boundary": (2460, 3440, 2860, 3880),
}
GRADER = ColorGrader()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def csv_write(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rgb_gray(x):
    return np.repeat(np.asarray(x, np.float32)[..., None], 3, axis=2)


def save(path, bgr):
    assert cv2.imwrite(str(path), np.floor(np.clip(bgr, 0, 1)*255+.5).astype(np.uint8))


def hp(x, sigma=2):
    return x-cv2.GaussianBlur(x, (0, 0), sigma)


def sigma_hh(x):
    """Global orthonormal 2x2 Haar-HH MAD. IID model, NOT reliable on all JPEGs."""
    a = x[:x.shape[0]//2*2, :x.shape[1]//2*2]
    hh = (a[::2, ::2]-a[1::2, ::2]-a[::2, 1::2]+a[1::2, 1::2])/2
    return float(np.median(np.abs(hh-np.median(hh)))/.67448975)


def gate_variance(detail, sigma):
    mean = cv2.blur(detail, (9, 9))
    variance = np.maximum(cv2.blur(detail*detail, (9, 9))-mean*mean, 0)
    # Conservative subtractive signal-variance estimate; 1.5 is fixed, not fitted.
    return np.clip((variance-1.5*sigma*sigma)/(variance+1e-12), 0, 1)


def prepare(image, window=None, eps=.02, noise_sigma=None):
    image = np.asarray(image, np.float32)
    lab = bgr_f32_to_lab_f32(image*255)
    lum = lab[..., 0]/255
    window = max(int(min(image.shape[:2])*.015), 5) if window is None else window
    base = GRADER._guided_filter(lum, lum, window, eps)
    detail = lum-base
    raw_sigma = sigma_hh(lum)
    sigma = max(raw_sigma if noise_sigma is None else noise_sigma, .25/255)
    g = gate_variance(detail, sigma)
    small = cv2.GaussianBlur(lum, (0, 0), 1.0)
    fine, middle = lum-small, small-base
    gf, gm = gate_variance(fine, sigma), gate_variance(middle, .30*sigma)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3, scale=1/8)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3, scale=1/8)
    xx, yy, xy = (cv2.blur(a, (9, 9)) for a in (gx*gx, gy*gy, gx*gy))
    coherence = np.sqrt((xx-yy)**2+4*xy*xy)/(xx+yy+1e-12)
    # Boundary envelope limiter only at strong local range; not a semantic mask.
    lo = cv2.erode(lum, np.ones((7, 7), np.uint8))
    hi = cv2.dilate(lum, np.ones((7, 7), np.uint8))
    fields = {
        "matched_current": detail,
        "soft_floor": np.sign(detail)*np.maximum(np.abs(detail)-2*sigma, 0),
        "variance_snr": detail*g,
        "multiscale_snr": fine*gf+middle*gm,
        "snr_edge_cap": detail*g,
        "edge_coherence": detail*g*np.clip(coherence/.5, 0, 1),
    }
    return dict(image=image, lab=lab, lum=lum, base=base, detail=detail, fields=fields,
                lo=lo, hi=hi, sigma=sigma, raw_sigma=raw_sigma, window=window, eps=eps,
                roundtrip=lab_f32_to_bgr_f32(lab)/255)


def apply(bundle, arm, recipe_value):
    """Candidate reconstructions share a delta-preserving Lab->RGB bridge.

    production_current separately retains the actual direct Lab round trip.
    matched_current isolates this reconstruction change from gain gating.
    """
    image = bundle["image"]
    if arm == "disabled" or (recipe_value == 0 and arm != "conversion_only"):
        return image.copy()
    if arm == "conversion_only":
        return bundle["roundtrip"].copy()
    strength = recipe_value/100
    if arm == "production_current":
        # Same operations/order as grading.py, independently checked against it.
        base = bundle["base"]*255
        detail = bundle["lab"][..., 0]-base
        new_l = np.clip(base+detail*(1+strength), 0, 255)
    else:
        new_l = bundle["lum"]+strength*bundle["fields"][arm]
        if arm == "snr_edge_cap":
            strong = bundle["hi"]-bundle["lo"] > 16/255
            new_l = np.where(strong, np.clip(new_l, bundle["lo"], bundle["hi"]), new_l)
        new_l = np.clip(new_l*255, 0, 255)
    lab = bundle["lab"].copy()
    lab[..., 0] = new_l
    rendered = lab_f32_to_bgr_f32(lab)/255
    if arm == "production_current":
        return rendered
    # Exact zero-gain bypass; no conversion floor mistaken for gated detail.
    delta = rendered-bundle["roundtrip"]
    result = np.clip(image+delta, 0, 1)
    return np.where((new_l == bundle["lab"][..., 0])[..., None], image, result).astype(np.float32)


def plane_residual(x):
    # Orthogonal centered coordinates: same least-squares plane without a solve.
    xx = np.linspace(-1, 1, x.shape[1])[None, :]
    yy = np.linspace(-1, 1, x.shape[0])[:, None]
    return x-x.mean()-np.mean(x*xx)/np.mean(xx*xx)*xx-np.mean(x*yy)/np.mean(yy*yy)*yy


def measures(image, origin=(0, 0)):
    # Display-referred luma in DN; no uint8 LAB conversion in the metric itself.
    y = image[..., 0]*.0722+image[..., 1]*.7152+image[..., 2]*.2126
    y = y.astype(np.float64)*255
    residual = plane_residual(y)
    high = hp(y)
    window = np.outer(np.hanning(y.shape[0]), np.hanning(y.shape[1]))
    spectrum = np.abs(np.fft.rfft2(residual*window))**2/(y.size*np.sum(window*window))
    fy = np.fft.fftfreq(y.shape[0])[:, None]
    fx = np.fft.rfftfreq(y.shape[1])[None, :]
    freq = np.sqrt(fx*fx+fy*fy)
    # One-sided PSD needs doubling except DC/Nyquist columns.
    spectrum[:, 1:(-1 if y.shape[1] % 2 == 0 else None)] *= 2
    result = dict(residual_std_DN=float(np.std(residual)), highpass_rms_DN=float(np.sqrt(np.mean(high*high))),
                  haar_MAD_DN=sigma_hh(y), mean_DN=float(y.mean()),
                  local_contrast_std_DN=float(np.std(high)),
                  clip_fraction=float(np.mean((image <= 0) | (image >= 1))))
    for name, low, upper in (("low", .01, .05), ("mid", .05, .15), ("high", .15, .5)):
        result[f"PSD_{name}_DN2"] = float(spectrum[(freq >= low) & (freq < upper)].sum())
    # Block-boundary contrast proxy, not a claimed full PSNR-B implementation.
    dx, dy = np.diff(y, axis=1), np.diff(y, axis=0)
    bx = (np.arange(dx.shape[1])+origin[0]) % 8 == 7
    by = (np.arange(dy.shape[0])+origin[1]) % 8 == 7
    result["block_excess_DN2"] = float(max(0, (np.mean(dx[:, bx]**2)+np.mean(dy[by]**2)
                                                  -np.mean(dx[:, ~bx]**2)-np.mean(dy[~by]**2))/2))
    return result


def quantized_change(out, source):
    """Explicit nearest-integer PNG diagnostic, not all production exporters."""
    qout = np.floor(np.clip(out, 0, 1)*255+.5)
    qsource = np.floor(np.clip(source, 0, 1)*255+.5)
    delta = qout-qsource
    y = qout[..., 0]*.0722+qout[..., 1]*.7152+qout[..., 2]*.2126
    return dict(changed_u8_pixel_fraction=float(np.mean(np.any(delta != 0, axis=-1))),
                quantized_change_rms_DN=float(np.sqrt(np.mean(delta*delta))),
                quantized_highpass_rms_DN=float(np.sqrt(np.mean(hp(y)**2))))


def synthetic(seed, size=384):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    flat = np.full((size, size), .5, np.float32)
    grad = (.42+.16*xx/(size-1)+.015*np.sin(yy/80)).astype(np.float32)
    step = np.where(xx < size//2, .3, .7).astype(np.float32)
    blurred = cv2.GaussianBlur(step, (0, 0), 2)
    hair = flat.copy()
    for pos in (45, 99, 153, 227, 301):
        hair -= .08*np.exp(-((xx-pos-.12*yy-3*np.sin(yy/47))/.8)**2).astype(np.float32)
    fabric = (.5+.025*np.sin(2*np.pi*xx/7)+.018*np.sin(2*np.pi*yy/11)).astype(np.float32)
    printed = (.45+.10*((xx//8+yy//8)%2)).astype(np.float32)
    pores = flat.copy()
    for px, py in rng.uniform(10, size-10, (170, 2)):
        pores -= .022*np.exp(-((xx-px)**2+(yy-py)**2)/(2*1.1**2)).astype(np.float32)
    gaussian = rng.normal(0, 2/255, flat.shape).astype(np.float32)
    encoded_poisson = rng.poisson(.21404114*5000, flat.shape).astype(np.float32)/5000
    encoded_poisson = 1.055*np.maximum(encoded_poisson, 0)**(1/2.4)-.055
    cases = [("flat", flat, flat, "smooth"), ("gaussian_2DN", flat+gaussian, flat, "noise"),
             ("gaussian_05DN", flat+gaussian*.25, flat, "noise"),
             ("poisson_like", encoded_poisson, flat, "noise"),
             ("correlated_noise", flat+cv2.GaussianBlur(gaussian, (0, 0), .8), flat, "noise"),
             ("gradient", grad, grad, "smooth"),
             ("quantized_gradient", np.round(grad*255/4)*4/255, grad, "quantization")]
    for quality in (40, 75):
        ok, encoded = cv2.imencode('.jpg', np.round(rgb_gray(grad)*255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, quality])
        assert ok
        cases.append((f"jpeg_gradient_q{quality}", cv2.imdecode(encoded, cv2.IMREAD_COLOR)[..., 0]/255, grad, "compression"))
    ok, encoded = cv2.imencode('.jpg', np.round(rgb_gray(blurred)*255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 40])
    assert ok
    cases.append(("jpeg_blurred_edge", cv2.imdecode(encoded, cv2.IMREAD_COLOR)[..., 0]/255, blurred, "edge"))
    for name, clean in (("hard_edge", step), ("blurred_edge", blurred), ("hair_lines", hair),
                        ("fabric", fabric), ("printed", printed), ("pores", pores)):
        cases.append((name, clean, clean, "edge" if 'edge' in name else "structure"))
        cases.append((name+"_noisy", clean+gaussian, clean, "edge" if 'edge' in name else "structure"))
    return [(name, rgb_gray(np.clip(observed, 0, 1)), rgb_gray(clean), kind) for name, observed, clean, kind in cases]


def evaluate(image, clean, kind, config, values=VALUES, arms=ARMS, metric_pad=12):
    bundle, truth = prepare(image, **config), prepare(clean, **config)
    # Keep all metric windows away from image boundaries; fixed for every arm.
    pad = metric_pad
    if min(image.shape[:2]) <= 2*pad+8:
        raise ValueError('Metric interior is too small')
    roi = (slice(pad, -pad), slice(pad, -pad))
    source_hp = hp(clean[roi][..., 0])
    denominator = float(np.sum(source_hp*source_hp))
    rows = []
    for value in values:
        for arm in arms:
            out, clean_out = apply(bundle, arm, value), apply(truth, arm, value)
            row = dict(arm=arm, recipe_clarity=value, effective_strength=value/100,
                       estimated_sigma_L_DN=bundle['raw_sigma']*255,
                       **measures(out[roi], (pad, pad)), **quantized_change(out[roi], image[roi]))
            diff = (out[roi]-clean_out[roi])*255
            row['paired_noise_rms_DN'] = float(np.sqrt(np.mean(diff*diff)))
            row['known_truth_error_rms_DN'] = float(np.sqrt(np.mean(((out[roi]-clean[roi])*255)**2)))
            row['clean_structure_gain'] = float(np.sum(hp(clean_out[roi][..., 0])*source_hp)/denominator) if denominator > 1e-8 else None
            row['increment_rgb_rms_DN'] = float(np.sqrt(np.mean(((out[roi]-apply(bundle, 'conversion_only' if arm == 'production_current' else 'disabled', value)[roi])*255)**2)))
            row['bypass_fraction'] = float(np.mean(np.max(np.abs(out[roi]-image[roi]), axis=-1) < 1e-8))
            if kind == 'edge':
                profile = out[pad:-pad, :, 0].mean(axis=0)
                baseline = clean[pad:-pad, :, 0].mean(axis=0)
                mid = len(baseline)//2
                low = float(np.median(baseline[pad:mid-20]))
                high = float(np.median(baseline[mid+20:-pad]))
                row['undershoot_DN'] = float(max(0, low-profile.min())*255)
                row['overshoot_DN'] = float(max(0, profile.max()-high)*255)
                row['reversed_slope_pixels'] = int(np.sum(np.diff(profile[pad:-pad]) < -1e-5))
                support = np.abs(profile-baseline)*255
                outside = np.abs(np.arange(len(profile))-len(profile)//2) > 6
                row['halo_area_DN_pixels'] = float(support[outside].sum())
            else:
                row.update(undershoot_DN=None, overshoot_DN=None, reversed_slope_pixels=None, halo_area_DN_pixels=None)
            rows.append(row)
    return rows


def mixed_detail_probe():
    """Matched noise-only counterfactual separates structure response IN noise.

    F(clean+noise)-F(flat+same_noise), projected onto known clean texture.
    This is controlled evidence unavailable for the real single JPEG. It does
    not use a clean estimate to gate the noisy image or tune any constants.
    """
    rows = []
    for seed in SEEDS:
        for name, image, clean, kind in synthetic(seed):
            if name not in ('hair_lines_noisy', 'fabric_noisy', 'printed_noisy', 'pores_noisy'):
                continue
            flat = np.full_like(clean, .5)
            noise_only = np.clip(flat+image-clean, 0, 1)
            b, counterfactual = prepare(image), prepare(noise_only)
            roi = (slice(12, -12), slice(12, -12))
            signal = hp((clean-flat)[roi][..., 0])
            norm = float(np.sum(signal*signal))
            for value in VALUES:
                for arm in ARMS:
                    response = hp((apply(b, arm, value)-apply(counterfactual, arm, value))[roi][..., 0])
                    rows.append(dict(seed=seed, fixture=name, recipe_clarity=value, arm=arm,
                        structure_in_noise_projection_gain=float(np.sum(response*signal)/norm),
                        nonprojected_response_rms_DN=float(np.sqrt(np.mean((response-signal*np.sum(response*signal)/norm)**2)))*255))
    return rows


def run(output, photo=None):
    output = Path(output)
    if output.exists():
        raise ValueError('Refusing to overwrite a research run')
    output.mkdir(parents=True)
    started = time.perf_counter()
    rows = []
    profiles = []
    for seed in SEEDS:
        for name, image, clean, kind in synthetic(seed):
            data = evaluate(image, clean, kind, {})
            rows.extend(dict(seed=seed, fixture=name, kind=kind, **r) for r in data)
            if seed == SEEDS[0] and name in ('hard_edge', 'blurred_edge', 'jpeg_blurred_edge'):
                b = prepare(image)
                for value in (4, 10):
                    for arm in ARMS:
                        profile = apply(b, arm, value)[12:-12, :, 0].mean(axis=0)*255
                        for x, dn in enumerate(profile):
                            profiles.append(dict(fixture=name, recipe_clarity=value, arm=arm,
                                                 x=x, input_DN=float(image[12:-12, x, 0].mean()*255),
                                                 clean_DN=float(clean[12:-12, x, 0].mean()*255), output_DN=float(dn)))
            if seed == SEEDS[0] and name in ('gaussian_2DN', 'jpeg_gradient_q40', 'hair_lines_noisy', 'pores_noisy', 'hard_edge'):
                b = prepare(image)
                save(output/f'{name}_input.png', image)
                for arm in ARMS:
                    save(output/f'{name}_{arm}_010.png', apply(b, arm, 10))
            print(f'synthetic {seed} {name}', flush=True)
    csv_write(output/'synthetic.csv', rows)
    csv_write(output/'edge_profiles.csv', profiles)
    mixed_rows = mixed_detail_probe()
    csv_write(output/'mixed_detail.csv', mixed_rows)
    sensitivity = []
    for name, image, clean, kind in synthetic(SEEDS[0]):
        if name not in ('gaussian_2DN', 'hard_edge', 'hair_lines_noisy', 'fabric_noisy', 'pores_noisy', 'jpeg_gradient_q40'):
            continue
        for window in (5, 15, 58):
            for eps in (.0008, .005, .02, .08):
                for r in evaluate(image, clean, kind, dict(window=window, eps=eps), values=(4, 10), arms=('production_current',), metric_pad=116):
                    sensitivity.append(dict(fixture=name, window=window, epsilon=eps, **r))
    csv_write(output/'parameter_sensitivity.csv', sensitivity)
    real_rows, real_profiles, parity = [], [], {}
    if photo:
        source = cv2.imread(str(photo), cv2.IMREAD_COLOR)
        if source is None or source.shape != (5850, 3879, 3):
            raise ValueError('Native DSCF1884 ROI coordinates require the established 3879x5850 source')
        image = source.astype(np.float32)/255
        frame_window = max(int(min(image.shape[:2])*.015), 5)
        # One full native current-op render verifies crop/halo padding and window.
        full = GRADER._F_add_clarity(image, .04)
        for name, (x0, y0, x1, y1) in ROIS.items():
            margin = 2*frame_window+16
            bx0, by0 = max(0, x0-margin), max(0, y0-margin)
            bx1, by1 = min(image.shape[1], x1+margin), min(image.shape[0], y1+margin)
            padded = image[by0:by1, bx0:bx1]
            crop = (slice(y0-by0, y1-by0), slice(x0-bx0, x1-bx0))
            b = prepare(padded, window=frame_window)
            parity[name] = float(np.max(np.abs(apply(b, 'production_current', 4)[crop]-full[y0:y1, x0:x1]))*255)
            save(output/f'DSCF1884_{name}_input.png', padded[crop])
            save(output/f'DSCF1884_{name}_detail_8x.png', rgb_gray(np.clip(.5+b['detail'][crop]*8, 0, 1)))
            for value in VALUES:
                for arm in ARMS:
                    out = apply(b, arm, value)[crop]
                    r = dict(roi=name, arm=arm, recipe_clarity=value, **measures(out, (x0, y0)),
                             **quantized_change(out, padded[crop]),
                             estimated_sigma_L_DN=b['raw_sigma']*255,
                             change_rms_DN=float(np.sqrt(np.mean(((out-padded[crop])*255)**2))),
                             bypass_fraction=float(np.mean(np.max(np.abs(out-padded[crop]), axis=-1) < 1e-8)))
                    real_rows.append(r)
                    if value in (4, 10):
                        save(output/f'DSCF1884_{name}_{arm}_{value:03d}.png', out)
                        if arm in ('production_current', 'variance_snr', 'multiscale_snr', 'snr_edge_cap'):
                            save(output/f'DSCF1884_{name}_{arm}_{value:03d}_delta_8x.png', .5+8*(out-padded[crop]))
                        if 'boundary' in name:
                            mid = out.shape[0]//2
                            profile = out[mid-2:mid+3].mean(axis=0)*255
                            source_profile = padded[crop][mid-2:mid+3].mean(axis=0)*255
                            for x in range(out.shape[1]):
                                for c, label in enumerate('BGR'):
                                    real_profiles.append(dict(roi=name, arm=arm, recipe_clarity=value,
                                        x=x+x0, y=mid+y0, channel=label,
                                        input_DN=float(source_profile[x, c]), output_DN=float(profile[x, c]),
                                        signed_delta_DN=float(profile[x, c]-source_profile[x, c])))
            print(f'real {name}', flush=True)
        csv_write(output/'real_rois.csv', real_rows)
        csv_write(output/'real_edge_profiles.csv', real_profiles)
    json_write(output/'manifest.json', dict(scope='P6 research only; no production changes', seeds=SEEDS,
        values=VALUES, arms=ARMS, candidate_constants=dict(noise_floor_L_DN=.25, soft_threshold_sigma=2,
        signal_variance_subtraction=1.5, variance_window=9, fine_sigma=1, middle_noise_factor=.30,
        coherence_scale=.5, strong_edge_range_L_DN=16, edge_envelope_window=7),
        tuning='Constants declared before first evaluation; no parameters selected on reporting seeds or DSCF1884. ROIs visually relabelled and supplemented, not optimized. This is not a validated calibration or public preregistration.',
        synthetic_rows=len(rows), mixed_detail_rows=len(mixed_rows), sensitivity_rows=len(sensitivity), real_rows=len(real_rows),
        native_crop_parity_max_DN=parity, photo=str(photo) if photo else None,
        photo_sha256=sha(photo) if photo else None, roi_xyxy=ROIS if photo else {},
        runner_sha256=sha(__file__), production_hashes={p:sha(ROOT/p) for p in
        ('retouch/grading.py','retouch/utils.py','retouch/engine.py','retouch/regions.py','retouch/skin.py','retouch/perf_optimizations.py')},
        numpy=np.__version__, opencv=cv2.__version__, elapsed_seconds=time.perf_counter()-started))
    print(output/'manifest.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--photo', type=Path)
    parser.add_argument('--mixed-detail-only', action='store_true', help='Independent fixed-parameter supplemental probe; output is a new CSV')
    args = parser.parse_args()
    if args.mixed_detail_only:
        if args.output.exists():
            raise ValueError('Refusing to overwrite a research probe')
        csv_write(args.output, mixed_detail_probe())
        print(args.output)
    else:
        run(args.output, args.photo)

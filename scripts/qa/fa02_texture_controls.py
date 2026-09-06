"""Deterministic analytic fixtures for FA-02; no photographs or inferred labels.

All supports are constructed from the injected signal definition, never detected.
These fixtures validate equations/diagnostics and cannot certify real pores/hair.
"""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from fa02_texture_representation_experiment import digest, gaussian, write_json


def create_controls(output: Path):
    if output.exists():
        raise ValueError("Refusing to overwrite existing controls")
    output.mkdir(parents=True)
    n = 512
    yy, xx = np.indices((n, n), dtype=np.float32)
    flat = np.zeros((n, n, 3), np.float32)+np.array([150, 170, 190], np.float32)
    clean = flat + (0.008*xx+0.003*yy)[..., None]
    pores = []
    # Analytic dark Gaussian dots, not generated or labelled biological pores.
    for y in (92, 116, 140, 164):
        for x in (92, 116, 140, 164):
            clean -= (4*np.exp(-((xx-x)**2+(yy-y)**2)/(2*0.9**2)))[..., None]
            pores.append({"id": f"dot_{x}_{y}", "center": [x, y], "radius": 1})
    profiles = []
    for index, base in enumerate((335, 357, 379)):
        curve = base + 3*np.sin(yy/38)
        window = ((yy > 72) & (yy < 176)).astype(np.float32)
        clean -= (3.5*np.exp(-((xx-curve)**2)/(2*0.65**2))*window)[..., None]
        for y in (96, 120, 144):
            x = float(base+3*np.sin(y/38))
            profiles.append({"id": f"line_{index}_{y}", "kind": "fine_hair_proxy",
                             "trace_id": f"line_{index}", "p0": [x-7, y], "p1": [x+7, y]})
    # A retained high-contrast edge supplies a JPEG ringing/edge-profile control.
    clean[350:400, 345:385] -= np.array([22, 25, 30], np.float32)
    profiles.append({"id": "retained_edge", "kind": "edge", "p0": [325, 375], "p1": [365, 375]})
    clean = clean.astype(np.float32)
    smooth = gaussian(clean, 2.0)
    allow = np.ones((n, n), np.float32)
    corrected = (((xx-250)**2+(yy-330)**2) <= 7**2).astype(np.float32)
    protected = np.zeros((n, n), np.float32)
    protected[210:234, 95:171] = 1
    common = {"allow": allow, "corrected": corrected, "protected": protected}
    rois = [
        {"id": "dot_signal", "xywh": [72, 72, 116, 116], "tags": ["analytic_pore_proxy"]},
        {"id": "line_signal", "xywh": [320, 72, 80, 116], "tags": ["analytic_fine_hair_proxy"]},
        {"id": "flat_signal", "xywh": [65, 340, 112, 100], "tags": ["flat", "noise_proxy"]},
        {"id": "spot_boundary", "xywh": [202, 282, 96, 96], "tags": ["correction", "halo"]},
        {"id": "held_edge_boundary", "xywh": [62, 180, 146, 84], "tags": ["makeup_edge_proxy", "halo"]},
        {"id": "retained_edge", "xywh": [315, 330, 100, 96], "tags": ["edge", "jpeg", "ringing"]},
    ]
    boundary_profiles = [
        {"id": "spot_normal", "kind": "correction", "p0": [198, 330], "p1": [302, 330]},
        {"id": "held_edge_normal", "kind": "makeup_edge_proxy", "p0": [135, 178], "p1": [135, 267]},
    ]
    cases = []

    def add(cid, x, s, *, nuisance=None, paired=None, masks=None, extra=None, size=n, annotations=True):
        arrays = {"X": x.astype(np.float32), "S": s.astype(np.float32), **(masks or common)}
        if nuisance is not None:
            arrays["nuisance"] = nuisance.astype(np.float32)
        path = output/(cid+".npz")
        np.savez_compressed(path, **arrays)
        case = {"id": cid, "kind": "analytic_control", "split": "validation",
            "person_ids": [], "session_id": "no_real_subjects",
            "arrays": path.name, "arrays_sha256": digest(path),
            "inter_eye_distance_px": 200.0*size/n, "face_width_px": 500.0*size/n,
            "scale_note": "nominal mathematical scale; no actual face/landmarks",
            "crop_xywh": [0, 0, size, size], "smoothing_provenance": "fixed saved S; analytic Gaussian sigma=2 at 512px unless condition states otherwise",
            "support_status": "analytic_definition", "support_acceptance_reference": "fixture equations, not human labels",
            "rois": rois if size == n else [{"id": "small_canvas", "xywh": [16, 16, size-32, size-32], "tags": ["small_sampling_proxy"]}],
            "pores": pores if annotations else [], "profiles": profiles+boundary_profiles if annotations else []}
        if paired:
            case["paired_clean_id"] = paired
        if extra:
            case.update(extra)
        cases.append(case)

    add("clean_signal", clean, smooth)
    add("constant_tone", flat+np.array([10, 8, 6], np.float32), flat, annotations=False)
    add("broad_ramp", flat+(10*(xx/(n-1)-0.5))[..., None], flat, annotations=False)
    spot = -40*corrected[..., None]*np.ones(3, np.float32)
    add("corrected_spot", clean+spot, smooth, nuisance=spot, paired="clean_signal")
    edge = np.zeros_like(clean)
    cv2.line(edge, (100, 228), (163, 214), (-35, -40, -45), 3, cv2.LINE_8)
    assert not np.any(edge[protected == 0])
    add("held_makeup_edge", clean+edge, smooth, nuisance=edge, paired="clean_signal")
    for seed in (1701, 2909):
        noise = np.random.default_rng(seed).normal(0, 2.5, clean.shape).astype(np.float32)
        add(f"noise_seed_{seed}", clean+noise, smooth, nuisance=noise, paired="clean_signal",
            extra={"noise_seed": seed, "noise_sigma_encoded": 2.5})
    for bx, by in ((0, 0), (3, 5)):
        padded = cv2.copyMakeBorder(np.rint(clean).astype(np.uint8), by, 0, bx, 0, cv2.BORDER_REFLECT_101)
        ok, encoded = cv2.imencode(".jpg", padded, [cv2.IMWRITE_JPEG_QUALITY, 30])
        assert ok
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)[by:by+n, bx:bx+n].astype(np.float32)
        add(f"jpeg_phase_{bx}_{by}", decoded, smooth, nuisance=decoded-clean, paired="clean_signal",
            extra={"jpeg_origin_xy": [-bx, -by], "jpeg_quality": 30,
                   "jpeg_note": "includes uint8 source rounding and codec chroma subsampling"})
    soft = gaussian(clean, 1.5)
    add("soft_focus", soft, gaussian(soft, 2.0),
        extra={"condition": "additional source blur sigma=1.5; not a real soft-focus portrait"})
    small = cv2.resize(clean, (128, 128), interpolation=cv2.INTER_AREA)
    masks = {k: cv2.resize(v, (128, 128), interpolation=cv2.INTER_NEAREST) for k, v in common.items()}
    add("small_sampling", small, gaussian(small, 0.6), masks=masks, size=128, annotations=False,
        extra={"condition": "analytic 4x downsampling; no upsampling; S Gaussian sigma=0.6"})
    write_json(output/"manifest.json", {"schema_version": 1, "purpose": "harness_validation",
        "corpus": "11 analytic signal canvases; zero photographs, zero real subjects, no held-out set",
        "parameter_selection": "a priori research settings, no fitting to these fixtures",
        "cases": cases})
    print(output/"manifest.json")

"""Render + measure: baseline / candidate reports for the Quality Lab."""
from __future__ import annotations

import gc
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from retouch import RetouchEngine
from retouch.detection import FaceDetector
from retouch.parsing import FaceParser

from . import metrics
from .core import (LAB_DIR, REPORTS_DIR, _engine_versions, _git_commit,
                   _git_dirty, _json_safe, _sha256)
from .regions import extract_region_masks

# Regions that get full per-region metric treatment (kept small: these are
# where edits actually land). Others are measured but only for drift.
DRIFT_REGIONS = ("FACE", "SKIN", "EYES", "HAIR", "LIPS", "NECK", "BACKGROUND")
TEXTURE_REGIONS = ("SKIN", "EYES", "HAIR", "LIPS")


def measure_output(inp: np.ndarray, out: np.ndarray,
                   masks: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Absolute-quality metrics between the INPUT and one output image."""
    report: Dict[str, Any] = {"global": {}, "regions": {}}
    report["global"] = {
        "luminance_mean_in": metrics.channel_stats(inp)["L"],
        "luminance_mean_out": metrics.channel_stats(out)["L"],
        "luminance_drift": abs(metrics.channel_stats(inp)["L"] - metrics.channel_stats(out)["L"]),
        "deltaE_mean": metrics.delta_e_mean(inp, out),
        "highlight_clip_in": metrics.clip_fraction(inp, "highlight"),
        "highlight_clip_out": metrics.clip_fraction(out, "highlight"),
        "shadow_clip_in": metrics.clip_fraction(inp, "shadow"),
        "shadow_clip_out": metrics.clip_fraction(out, "shadow"),
        "texture_retention": metrics.texture_retention(inp, out),
        "edge_retention": metrics.edge_retention(inp, out),
    }
    for name, mask in masks.items():
        if name not in DRIFT_REGIONS and name != "FULL":
            continue
        coverage = float((mask > 0.5).mean())
        if coverage < 0.002 and name != "FULL":
            # Sub-0.2% masks (a few hundred px): means over them are noise,
            # not signal. Record coverage only; skip drift metrics.
            report["regions"][name] = {"coverage": coverage, "skipped": "mask_too_small"}
            continue
        si, so = metrics.channel_stats(inp, mask), metrics.channel_stats(out, mask)
        entry: Dict[str, Any] = {
            "luminance_drift": abs(si["L"] - so["L"]),
            "chroma_drift": float(np.hypot(si["a"] - so["a"], si["b"] - so["b"])),
            "coverage": coverage,
        }
        if name in TEXTURE_REGIONS:
            entry["texture_retention"] = metrics.texture_retention(inp, out, mask)
            entry["edge_retention"] = metrics.edge_retention(inp, out, mask)
        report["regions"][name] = entry
    return report


def measure_identity(baseline: np.ndarray, candidate: np.ndarray,
                     masks: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Candidate-vs-baseline identity metrics (regression evidence)."""
    ident: Dict[str, Any] = {
        "ssim": metrics.ssim(baseline, candidate),
        "deltaE_mean": metrics.delta_e_mean(baseline, candidate),
        "regions": {},
    }
    for name in DRIFT_REGIONS:
        if name not in masks:
            continue
        ident["regions"][name] = {
            "deltaE_mean": metrics.delta_e_mean(baseline, candidate, masks[name]),
        }
    return ident


def render_cases(cases: Sequence[Dict[str, Any]], recipes: Sequence[str],
                 which: str) -> Dict[str, Any]:
    """Render every (case, recipe) through the engine, measure, collect.

    ``which`` is "baseline" or "candidate" — selects the output directory.
    Baselines additionally fingerprint the render for identity comparison.
    """
    engine = RetouchEngine()
    detector: Optional[FaceDetector] = None
    parser: Optional[FaceParser] = None
    results: Dict[str, Any] = {}
    try:
        for case in cases:
            case_id = case["id"]
            inp_path = Path(case["input"])
            if not inp_path.is_absolute():
                inp_path = Path(case.get("_root", ".")) / inp_path
            inp = cv2.imread(str(inp_path))
            if inp is None:
                results[case_id] = {"error": f"unreadable input: {inp_path}"}
                continue
            detector = detector or FaceDetector()
            parser = parser or FaceParser()
            masks, _ctx, notes = extract_region_masks(inp, detector, parser)
            case_out: Dict[str, Any] = {
                "tags": case.get("tags", []),
                "input_sha256": _sha256(inp_path),
                "notes": notes,
                "recipes": {},
            }
            for recipe in recipes:
                t0 = time.perf_counter()
                tracemalloc.start()
                try:
                    out = engine.process(inp.copy(), recipe=recipe)
                    err = None
                except Exception as exc:  # render failures are findings, not crashes
                    out, err = None, f"{type(exc).__name__}: {exc}"
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                elapsed = time.perf_counter() - t0
                gc.collect()
                entry: Dict[str, Any] = {
                    "time_s": round(elapsed, 3),
                    "peak_ram_mb": round(peak / 1e6, 1),
                }
                if err is not None:
                    entry["error"] = err
                else:
                    out_u8 = np.asarray(out)
                    if out_u8.dtype != np.uint8:
                        out_u8 = np.clip(out_u8, 0, 255).astype(np.uint8)
                    out_dir = LAB_DIR / case_id / which
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{recipe}.png"
                    cv2.imwrite(str(out_path), out_u8)
                    entry["output"] = str(out_path)
                    entry["output_sha256"] = _sha256(out_path)
                    entry["metrics"] = measure_output(inp, out_u8, masks)
                case_out["recipes"][recipe] = entry
            results[case_id] = case_out
    finally:
        if detector is not None:
            detector.close()
        if parser is not None:
            parser.close()
        engine.close()
    return results


def build_report(which: str, cases: Sequence[Dict[str, Any]],
                 recipes: Sequence[str], extra: Optional[Dict[str, Any]] = None
                 ) -> Dict[str, Any]:
    """Render + assemble a full report dict (does not write it)."""
    results = render_cases(cases, recipes, which)
    report = {
        "schema": 1,
        "kind": which,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {
            "commit": _git_commit(),
            "working_tree_dirty": _git_dirty(),
            "recipes": list(recipes),
            "engine_versions": _engine_versions(),
        },
        "cases": results,
    }
    if extra:
        report.update(extra)
    return _json_safe(report)


def write_report(report: Dict[str, Any], name: Optional[str] = None) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    name = name or f"{report['kind']}-{report['provenance']['commit'][:8]}-{int(time.time())}.json"
    path = REPORTS_DIR / name
    import json
    path.write_text(json.dumps(report, indent=2))
    return path

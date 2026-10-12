"""Engine Performance Benchmarking & Pre-Flight Quality Assurance.

Measures processing throughput (Mpx/s, FPS) and memory consumption across 1080p, 4K, and 8K
resolutions for engine pipelines.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class BenchmarkResult:
    resolution_label: str
    height: int
    width: int
    megapixels: float
    elapsed_seconds: float
    fps: float
    mpx_per_sec: float


def benchmark_engine_throughput(
    process_fn: Any,
    resolutions: Optional[Dict[str, Tuple[int, int]]] = None,
    iterations: int = 3,
) -> Dict[str, BenchmarkResult]:
    """Run throughput benchmark across standard image resolutions.

    Args:
        process_fn: Retouch process function accepting (H, W, 3) uint8 BGR image.
        resolutions: Resolution dict name -> (width, height).
        iterations: Number of test iterations to average.

    Returns:
        Dict mapping resolution names to BenchmarkResult dataclass instances.
    """
    if resolutions is None:
        resolutions = {
            "1080p": (1920, 1080),
            "4K": (3840, 2160),
        }

    results: Dict[str, BenchmarkResult] = {}

    for label, (w, h) in resolutions.items():
        dummy_img = np.full((h, w, 3), 128, dtype=np.uint8)
        
        # Warmup run
        try:
            _ = process_fn(dummy_img)
        except Exception:
            pass

        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = process_fn(dummy_img)
        t1 = time.perf_counter()

        elapsed = (t1 - t0) / float(iterations)
        mpx = (w * h) / 1e6
        fps = 1.0 / max(elapsed, 1e-6)
        mpx_per_sec = mpx / max(elapsed, 1e-6)

        results[label] = BenchmarkResult(
            resolution_label=label,
            height=h,
            width=w,
            megapixels=mpx,
            elapsed_seconds=elapsed,
            fps=fps,
            mpx_per_sec=mpx_per_sec,
        )

    return results


def run_preflight_checks(engine: Optional[Any] = None) -> Dict[str, bool]:
    """Run automated pre-flight system integrity checks.

    Returns:
        Dict of check names -> pass/fail boolean status.
    """
    checks: Dict[str, bool] = {}

    # Check 1: Color Science module imports & basic transforms
    try:
        from .color_science import bgr_to_oklab, oklab_to_bgr
        dummy = np.full((16, 16, 3), 128, dtype=np.uint8)
        _ = oklab_to_bgr(bgr_to_oklab(dummy))
        checks["color_science_oklab"] = True
    except Exception:
        checks["color_science_oklab"] = False

    # Check 2: QA detector readiness
    try:
        from .qa_detectors import run_all
        dummy = np.full((32, 32, 3), 150, dtype=np.uint8)
        _ = run_all(dummy)
        checks["qa_detectors_run_all"] = True
    except Exception:
        checks["qa_detectors_run_all"] = False

    # Check 3: Acceleration layer status
    try:
        from .acceleration import get_acceleration_backend
        _ = get_acceleration_backend()
        checks["acceleration_layer"] = True
    except Exception:
        checks["acceleration_layer"] = False

    return checks

#!/usr/bin/env python3
"""Benchmark runner for the retouch pipeline.

Discovers the benchmark tests in ``tests/benchmark_pipeline.py`` and
``tests/benchmark_modules.py``, runs them, and reports results in a
human-readable table. Saves a JSON report to ``benchmark_results.json``
for regression tracking.

Usage:
    python3 scripts/bench/benchmark.py                     # run everything
    python3 scripts/bench/benchmark.py --module pipeline   # only pipeline tests
    python3 scripts/bench/benchmark.py --module modules    # only module tests
    python3 scripts/bench/benchmark.py -k frequency        # only matching tests
    python3 scripts/bench/benchmark.py --iterations 50     # override default count
    python3 scripts/bench/benchmark.py --output my.json    # custom output path
    python3 scripts/bench/benchmark.py --no-save           # skip writing JSON
    python3 scripts/bench/benchmark.py --quiet             # only print the table

The runner uses unittest.mock at the engine boundary, so it does NOT require
MediaPipe / ONNX models on disk. It also uses synthetic images.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import os
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make ``tests`` and the project root importable so we can introspect
# benchmark classes.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TESTS_DIR = PROJECT_ROOT / "tests"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TESTS_DIR))


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


PIPELINE_FILE = "tests/benchmark_pipeline.py"
MODULES_FILE = "tests/benchmark_modules.py"


def discover_benchmarks(file_path: Path) -> List[Dict[str, Any]]:
    """Return a list of (class_name, method_name, file) for every benchmark."""
    # Import the module so we can use inspection rather than parsing pytest -collect
    import importlib.util

    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    if spec is None or spec.loader is None:
        return []
    module = importlib.util.module_from_spec(spec)
    # We don't actually execute the module body because the benchmarks
    # call into retouch.* at top of test methods; we only need the class
    # structure. Use exec to load class definitions.
    try:
        spec.loader.exec_module(module)
    except Exception:
        # If the module fails to import (e.g. mediapipe not installed),
        # fall back to AST-based discovery.
        return _discover_benchmarks_via_ast(file_path)

    out: List[Dict[str, Any]] = []
    for name in dir(module):
        obj = getattr(module, name)
        if not isinstance(obj, type):
            continue
        if not name.startswith("Test"):
            continue
        for attr in dir(obj):
            if attr.startswith("test_"):
                out.append(
                    {
                        "class": name,
                        "method": attr,
                        "file": str(file_path.relative_to(PROJECT_ROOT)),
                        "full_id": f"{name}::{attr}",
                    }
                )
    return out


def _discover_benchmarks_via_ast(file_path: Path) -> List[Dict[str, Any]]:
    """Fallback discovery using AST — only needs the test names, not imports."""
    import ast

    out: List[Dict[str, Any]] = []
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name.startswith("test_"):
                    out.append(
                        {
                            "class": node.name,
                            "method": child.name,
                            "file": str(file_path.relative_to(PROJECT_ROOT)),
                            "full_id": f"{node.name}::{child.name}",
                        }
                    )
    return out


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_pytest_subset(
    file_path: Path,
    test_filter: Optional[str],
    iterations_override: Optional[int],
    capture: bool = True,
) -> Dict[str, Any]:
    """Run a single benchmark file via pytest and parse output for timings.

    Returns a dict with keys: returncode, durations (list[(test, ms, p95)]).
    """
    cmd: List[str] = [
        sys.executable, "-m", "pytest", str(file_path),
        "-v", "-s", "-W", "ignore", "--tb=line", "--no-header",
    ]
    if test_filter:
        cmd.extend(["-k", test_filter])
    env = os.environ.copy()
    if iterations_override is not None:
        env["BENCHMARK_ITERATIONS"] = str(iterations_override)
    try:
        proc = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT), capture_output=capture, text=True,
            timeout=900, env=env,
        )
    except subprocess.TimeoutExpired:
        return {"returncode": -1, "durations": [], "stdout": "", "stderr": "timeout"}
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    durations = parse_benchmark_lines(stdout)
    return {
        "returncode": proc.returncode,
        "durations": durations,
        "stdout": stdout,
        "stderr": stderr,
    }


_BENCHMARK_RE = re.compile(
    r"\[benchmark\]\s+(?P<label>.+?)\s+median=\s*(?P<median>[\d.]+)\s*ms"
    r"\s+p95=\s*(?P<p95>[\d.]+)\s*ms\s+\(n=(?P<n>\d+)\)"
)
_MEMORY_RE = re.compile(
    r"\[benchmark\]\s+(?P<label>.+?):\s*baseline=(?P<baseline>[\d.]+)MB\s+"
    r"peak=(?P<peak>[\d.]+)MB\s+delta=\+(?P<delta>[\d.]+)MB"
)
_STAGE_RE = re.compile(
    r"\[benchmark\]\s+pipeline stage breakdown:\s+(?P<timings>\{.*?\})"
)
_BATCH_RE = re.compile(
    r"\[benchmark\]\s+batch avg per-image:\s*(?P<avg>[\d.]+)\s*ms"
)
_SKIN_QUALITY_RE = re.compile(
    r"\[benchmark\]\s+skin_quality_(?P<phase>\w+):\s+"
    r"blotch_std=(?P<blotch_std>[\d.]+)\s+chroma_std=(?P<chroma_std>[\d.]+)"
)
_QA_DETECTOR_RE = re.compile(
    r"\[benchmark\]\s+qa_detector_(?P<phase>\w+):\s+"
    r"banding=(?P<banding>[\d.]+)\s+"
    r"clipping=(?P<clipping>[\d.]+)\s+"
    r"plastic=(?P<plastic>[\d.]+)"
)


def parse_benchmark_lines(output: str) -> List[Dict[str, Any]]:
    """Parse ``[benchmark] ...`` lines from pytest -s output."""
    results: List[Dict[str, Any]] = []
    for line in output.splitlines():
        m = _BENCHMARK_RE.search(line)
        if m:
            results.append(
                {
                    "type": "timing",
                    "label": m.group("label").strip(),
                    "median_ms": float(m.group("median")),
                    "p95_ms": float(m.group("p95")),
                    "n": int(m.group("n")),
                }
            )
            continue
        m = _MEMORY_RE.search(line)
        if m:
            results.append(
                {
                    "type": "memory",
                    "label": m.group("label").strip(),
                    "baseline_mb": float(m.group("baseline")),
                    "peak_mb": float(m.group("peak")),
                    "delta_mb": float(m.group("delta")),
                }
            )
            continue
        m = _STAGE_RE.search(line)
        if m:
            try:
                timings = json.loads(m.group("timings").replace("'", '"'))
            except json.JSONDecodeError:
                timings = {}
            results.append({"type": "stages", "timings_ms": timings})
            continue
        m = _BATCH_RE.search(line)
        if m:
            results.append(
                {"type": "batch_avg", "avg_per_image_ms": float(m.group("avg"))}
            )
            continue
        m = _SKIN_QUALITY_RE.search(line)
        if m:
            results.append(
                {
                    "type": "skin_quality",
                    "phase": m.group("phase"),
                    "blotch_std": float(m.group("blotch_std")),
                    "chroma_std": float(m.group("chroma_std")),
                }
            )
            continue
        m = _QA_DETECTOR_RE.search(line)
        if m:
            results.append(
                {
                    "type": "qa_detector",
                    "phase": m.group("phase"),
                    "banding": float(m.group("banding")),
                    "clipping": float(m.group("clipping")),
                    "plastic": float(m.group("plastic")),
                }
            )
    return results


# ---------------------------------------------------------------------------
# Skin quality metrics
# ---------------------------------------------------------------------------


def skin_quality_metrics(
    img_bgr,
    skin_mask: Optional,
    face_width: float,
) -> Dict[str, float]:
    """Compute objective skin-quality metrics.

    Args:
        img_bgr: (H, W, 3) uint8 BGR image.
        skin_mask: (H, W) float mask [0, 1], or None.
        face_width: Face width in pixels (typically ied * 2.5).

    Returns:
        dict with keys "blotch_std" and "chroma_std", both floats.
        If skin_mask is None or empty, returns {"blotch_std": 0.0, "chroma_std": 0.0}.
    """
    import numpy as np
    import cv2
    from retouch.color_science import skin_chroma_std

    result = {"blotch_std": 0.0, "chroma_std": 0.0}

    # Handle empty/None mask
    if skin_mask is None or skin_mask.size == 0:
        return result
    if np.sum(skin_mask > 0.3) == 0:
        return result

    # Convert BGR to LAB
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = img_lab[..., 0]

    # Band-pass filter: DoG at sigma=fw/40 vs fw/12
    sigma_narrow = face_width / 40.0
    sigma_wide = face_width / 12.0
    blur_narrow = cv2.GaussianBlur(L, (0, 0), sigma_narrow)
    blur_wide = cv2.GaussianBlur(L, (0, 0), sigma_wide)
    band = blur_narrow - blur_wide

    # Compute std within skin_mask > 0.3
    mask_binary = (skin_mask > 0.3).astype(np.float32)
    masked_band = band * mask_binary
    valid_pixels = masked_band[mask_binary > 0.5]
    if valid_pixels.size > 0:
        result["blotch_std"] = float(np.std(valid_pixels))

    # Compute chroma std using the color_science function
    result["chroma_std"] = skin_chroma_std(img_bgr, skin_mask)

    return result


def qa_detector_metrics(
    img_bgr: np.ndarray,
    skin_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Run QA detectors and return their scores.

    Args:
        img_bgr: (H, W, 3) uint8 BGR image.
        skin_mask: Optional (H, W) float mask [0, 1].

    Returns:
        dict with keys "banding_score", "clipping_score", "plastic_skin_score".
        Returns 0.0 for each if detection fails.
    """
    from retouch import qa_detectors

    result: Dict[str, float] = {
        "banding_score": 0.0,
        "clipping_score": 0.0,
        "plastic_skin_score": 0.0,
    }

    try:
        banding = qa_detectors.detect_banding(img_bgr, skin_mask)
        result["banding_score"] = float(banding.get("score", 0.0))
    except Exception:
        pass

    try:
        clipping = qa_detectors.detect_clipping(img_bgr)
        result["clipping_score"] = float(clipping.get("score", 0.0))
    except Exception:
        pass

    try:
        plastic = qa_detectors.detect_plastic_skin(img_bgr, skin_mask)
        result["plastic_skin_score"] = float(plastic.get("score", 0.0))
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Peak-RSS guard for large-image grading (P2 row 3b)
# ---------------------------------------------------------------------------


def _peak_rss_mb() -> float:
    """Return current process peak RSS in MB (best-effort, cross-platform)."""
    try:
        import resource
        # ru_maxrss: kilobytes on Linux, bytes on macOS.
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return usage.ru_maxrss / (1024.0 * 1024.0)
        return usage.ru_maxrss / 1024.0
    except (ImportError, AttributeError):
        return 0.0


def assert_grading_peak_rss(
    dims: List[tuple],
    max_peak_mb: float = 4096.0,
) -> List[Dict[str, Any]]:
    """Run grading on large synthetic images and assert peak RSS stays bounded.

    P2 row 3b perf guard: the ``_large_sigma_blur`` compute-downsample helper
    must keep peak memory flat on 4K/6K frames. This builds a synthetic float32
    image at each requested dimension, runs a haze+glow-heavy preset through
    ``ColorGrader.grade``, and records peak RSS.

    Args:
        dims: List of (H, W) tuples to probe (e.g. [(2160, 3840), (3240, 5760)]).
        max_peak_mb: Hard ceiling; an AssertionError is raised if exceeded.

    Returns:
        List of per-dimension dicts with keys ``dim``, ``peak_mb``, ``passed``.
    """
    import numpy as np
    from retouch.grading import ColorGrader

    results: List[Dict[str, Any]] = []
    grader = ColorGrader()
    settings = {
        "haze": 0.5,
        "glow": 0.4,
        "orton_glow": 0.3,
    }
    for h, w in dims:
        img = (np.random.RandomState(0).rand(h, w, 3) * 255.0).astype(np.uint8)
        _ = grader.grade(img, settings, 1.0, skip_post_effects=False)
        peak = _peak_rss_mb()
        passed = peak <= max_peak_mb
        results.append(
            {
                "dim": f"{h}x{w}",
                "peak_mb": round(peak, 1),
                "passed": passed,
            }
        )
        print(
            f"[benchmark] grading peak-RSS {h}x{w}: "
            f"peak={peak:.1f}MB ceiling={max_peak_mb:.0f}MB "
            f"({'PASS' if passed else 'FAIL'})"
        )
        if not passed:
            raise AssertionError(
                f"Grading peak RSS {peak:.1f}MB exceeds ceiling "
                f"{max_peak_mb:.0f}MB at {h}x{w} — _large_sigma_blur downsample "
                f"guard may be bypassed."
            )
        del img
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_table(rows: List[Dict[str, Any]]) -> None:
    timing_rows = [r for r in rows if r.get("type") == "timing"]
    memory_rows = [r for r in rows if r.get("type") == "memory"]
    other_rows = [r for r in rows if r.get("type") not in ("timing", "memory")]

    if not rows:
        print("\nNo benchmark results captured.")
        return

    if timing_rows:
        name_w = max(len(r["label"]) for r in timing_rows)
        name_w = min(max(name_w, 40), 60)
        header = f"  {'Benchmark':<{name_w}}  {'Median (ms)':>12}  {'P95 (ms)':>10}  {'N':>4}"
        print()
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in timing_rows:
            print(
                f"  {r['label']:<{name_w}}  {r['median_ms']:>12.2f}  "
                f"{r['p95_ms']:>10.2f}  {r['n']:>4}"
            )
        print()

    if memory_rows:
        print()
        print("  Memory")
        print("  " + "-" * 60)
        for r in memory_rows:
            print(
                f"  {r['label']:<48s}  baseline={r['baseline_mb']:6.1f} MB  "
                f"peak={r['peak_mb']:6.1f} MB  delta=+{r['delta_mb']:5.1f} MB"
            )
        print()

    if other_rows:
        print()
        print("  Other (stage breakdowns, batch averages, etc.)")
        print("  " + "-" * 60)
        for r in other_rows:
            if r.get("type") == "stages":
                timings = r.get("timings_ms", {})
                if timings:
                    print("  pipeline stages (ms):")
                    for k, v in sorted(timings.items(), key=lambda kv: -kv[1]):
                        print(f"    {k:<22s}  {v:8.2f}")
            elif r.get("type") == "batch_avg":
                print(f"  batch avg per-image: {r['avg_per_image_ms']:.2f} ms")
        print()


def build_report(
    module_filter: str,
    iterations: Optional[int],
    rows: List[Dict[str, Any]],
    raw_runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "module_filter": module_filter,
        "iterations": iterations,
        "results": rows,
        "raw_runs": raw_runs,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retouch benchmarks and report timings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--module", "-m",
        choices=["pipeline", "modules", "all"],
        default="all",
        help="Which benchmark file(s) to run (default: all).",
    )
    parser.add_argument(
        "--iterations", "-i", type=int, default=None,
        help="Override iteration count (sets BENCHMARK_ITERATIONS env var).",
    )
    parser.add_argument(
        "-k", type=str, default=None,
        help="pytest -k filter (e.g. 'frequency' or 'eyes').",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="benchmark_results.json",
        help="Path to write the JSON report (default: benchmark_results.json).",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip writing the JSON report.",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress the per-test pytest output; only show the summary table.",
    )
    parser.add_argument(
        "--peak-rss", action="store_true",
        help="Run the P2 row 3b grading peak-RSS guard on 4K/6K synthetic images.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.iterations is not None and args.iterations < 1:
        print("  ! --iterations must be at least 1.")
        return 2

    if args.peak_rss:
        print(">>> P2 row 3b grading peak-RSS guard (4K / 6K)")
        dims = [(2160, 3840), (3240, 5760)]
        rss_rows = assert_grading_peak_rss(dims)
        print()
        print("  Peak-RSS guard")
        print("  " + "-" * 50)
        for r in rss_rows:
            status = "PASS" if r["passed"] else "FAIL"
            print(
                f"  {r['dim']:<12s}  peak={r['peak_mb']:7.1f} MB  {status}"
            )
        print()
        return 0

    selected_files: List[Path] = []
    if args.module in ("pipeline", "all"):
        selected_files.append(PROJECT_ROOT / PIPELINE_FILE)
    if args.module in ("modules", "all"):
        selected_files.append(PROJECT_ROOT / MODULES_FILE)

    discovered: List[Dict[str, Any]] = []
    for f in selected_files:
        if not f.exists():
            print(f"  ! Skipping {f} (not found)")
            continue
        discovered.extend(discover_benchmarks(f))

    if not discovered:
        print("  ! No benchmark tests discovered.")
        return 1

    print(f"Discovered {len(discovered)} benchmark test(s).")
    for f in selected_files:
        print(f"  - {f.relative_to(PROJECT_ROOT)}")

    raw_runs: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    had_failures = False

    for f in selected_files:
        if not f.exists():
            continue
        print(f"\n>>> Running {f.relative_to(PROJECT_ROOT)}")
        run = run_pytest_subset(
            f,
            test_filter=args.k,
            iterations_override=args.iterations,
            capture=True,
        )
        raw_runs.append(
            {
                "file": str(f.relative_to(PROJECT_ROOT)),
                "returncode": run["returncode"],
                "durations": run["durations"],
            }
        )
        all_rows.extend(run["durations"])

        if not args.quiet:
            if run["stdout"]:
                print(run["stdout"])
            if run["stderr"]:
                print("STDERR:", run["stderr"])

        if run["returncode"] != 0:
            had_failures = True
            print(f"  ! pytest exited with code {run['returncode']}")

    print_table(all_rows)

    report = build_report(args.module, args.iterations, all_rows, raw_runs)

    if not args.no_save:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.write_text(json.dumps(report, indent=2))
        print(f"Wrote report: {out_path}")

    return 1 if had_failures else 0


if __name__ == "__main__":
    sys.exit(main())

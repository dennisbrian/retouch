"""Shared controls for deterministic benchmark test execution."""

from __future__ import annotations

import os


def benchmark_iterations(default: int) -> int:
    """Return the CLI override when set, otherwise the test's default."""
    raw = os.environ.get("BENCHMARK_ITERATIONS")
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("BENCHMARK_ITERATIONS must be an integer") from exc
    if value < 1:
        raise ValueError("BENCHMARK_ITERATIONS must be at least 1")
    return value

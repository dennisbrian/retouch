# Performance Targets: Latency, Resolution & Memory Rules

This document outlines execution rules and constraints for the retouch pipeline to achieve near-real-time performance and prevent resource exhaustion.

## 1. Proxy Scaling & Resolution Thresholds
- **Proxy Downscaling Bound (`PROXY_MAX_DIM`)**:
  - Images with dimensions exceeding 2048px must be downscaled to a maximum dimension of 2048px for pipeline operations.
  - The final output and segmentation masks must be bilinearly upscaled back to the input resolution.
  - Target memory bound: **< 2.0 GB peak RAM** (e.g., reducing 7.5 GB to 1.84 GB for 24 MP canvases).
- **Fast Interactive Preview Mode (`fast=True`)**:
  - The Gradio interface must downscale images to a maximum dimension of 800px for slider operations.
  - Target latency bound: **< 150ms per frame** to ensure real-time slider updates.

## 2. Multi-Face Processing Parallelization
- **Single Face Execution**:
  - Skip parallel pools entirely to avoid subprocess initialization overhead.
- **Multi-Face Subprocess Pool (`FaceProcessorPool`)**:
  - Distribute per-face Stage 2 loops across a subprocess pool using `ProcessPoolExecutor` when 2 or more faces are detected.
- **Thread Pool Fallback**:
  - If subprocess instantiation fails or is unsupported, fall back to a `ThreadPoolExecutor` (capped at 4 workers) using thread-safe ROI slices.

## 3. Latency Targets (2048px Proxy)
- **Full Pipeline (All 7 Stages)**:
  - Target average execution time: **< 800ms per image** (on standard CPU runtimes).
- **Redundant Inference Bypassing**:
  - Ensure the pipeline reads and returns cached `FaceContext` objects during interactive slider updates, bypassing Stage 0 (detection) and Stage 2 (BiSeNet parsing) to save ~350ms of execution overhead.

## 4. Cache-Friendly Design Rules
- **Static Documentation**:
  - Avoid writing timestamps, dynamic TODO counts, or build versions inside any files under `.ai/`.
  - Maintain absolute static formatting to maximize model context cache hit rates.

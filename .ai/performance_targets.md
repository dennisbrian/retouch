# Performance Targets

## Proxy & Preview

- `PROXY_MAX_DIM=2048`: images >2048px downscaled → process → bilinear upscale. Peak RAM <2GB.
- `fast=True` (Gradio slider mode): downscale to 800px max-dim. Latency <150ms/frame.

## Multi-Face Parallelization

- 1 face: skip pools (subprocess overhead not worth it)
- ≥2 faces: `FaceProcessorPool` (`ProcessPoolExecutor`) per-face Stage 2
- Fallback: `ThreadPoolExecutor` (max 4 workers) with thread-safe ROI slices

## Latency (2048px proxy)

- Full 7-stage pipeline: <800ms avg CPU
- Cache: `FaceContext` skips Stage 0+2 on slider re-run (saves ~350ms)

## Cache-Friendly Docs

- Static only — no timestamps, dynamic counts, or versions in `.ai/` files

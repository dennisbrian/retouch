# Performance Tuning Guide

**Quick answer:** Use the GUI's **Render Preview** for interactive work; it uses `fast=True` and never exports. Use **Export Full Quality** for one final image; it forces `fast=False` and the Full quality tier. Use **Export All → Batch** for shoots. Batch processing? Use `--workers 4-8`.

---

## Three Resolution Layers

The pipeline has **three independent resolution controls** that don't affect output dimensions — only internal processing speed/quality.

| Layer | When Active | Internal Resolution | Speed | Memory | Use Case |
|-------|-------------|---------------------|-------|--------|----------|
| **Fast Preview** | `fast=True` | 800px | ~3-5× faster | ~600 MB | Interactive slider tuning in GUI |
| **Proxy Resolution** | Auto (input > 2048px) | 2048px | 2× faster | 1.84 GB | Large images (4K, 24MP) |
| **True Native** | `fast=False` + input ≤ 2048px | Full input | Baseline | 1.2 GB | Final export at full quality |

**Key insight:** Output file dimensions are ALWAYS preserved (unchanged). Internal processing resolution is a speed/memory optimization only.

---

## When to Use Each Mode

### Render Preview (`fast=True`) — GUI Interactive Path
- **Settings:** The `Render Preview` action sets `fast=True`; the legacy checkbox is not the final-export control.
- **Processing resolution:** 800px downscaled
- **Speed:** ~3-5× faster than native
- **Memory:** ~600 MB
- **Quality trade-off:** Softer skin details (pores, fine hair strands)
- **Use for:**
  - ✅ Interactive slider tuning (recipe selection, brightness, contrast)
  - ✅ Quick A/B testing of recipes
  - ✅ Real-time preview in GUI
- **NOT for:**
  - ❌ Final exports (too soft for print/large displays)

### Export Full Quality (`fast=False`) — GUI Final Path
- **Settings:** The `Export Full Quality` action sets `fast=False` and `quality="full"` for one image.
- **Metadata:** The writer preserves source ICC/EXIF intent for the exported derivative.
- **Use for:** Final single-image delivery.

### Proxy Resolution — Automatic for High-Res
- **When:** Input image > 2048px automatically activates
- **Processing resolution:** 2048px (downscaled before pipeline, upscaled after)
- **Speed:** 2× faster than native on 24MP images
- **Memory:** 4× reduction (7.5 GB → 1.84 GB on 24MP)
- **Quality trade-off:** Minimal (2048px still captures most face detail)
- **Example timings on 24MP image:**
  - With proxy: 3.09s total
  - Without proxy: 15.3s total
- **Use for:**
  - ✅ Processing 4K+ images on low-memory systems
  - ✅ Batch processing 100+ photos (faster throughput)
  - ✅ Interactive GUI work on high-res images
- **Automatic:** Enabled by default, no user action needed

### True Native (`fast=False`) — Final Export
- **Settings:** `fast=False` in GUI, or via CLI
- **Processing resolution:** Full input resolution
- **Speed:** Baseline (slowest)
- **Memory:** 1.2-7.5 GB depending on image size
- **Quality:** Maximum detail preservation
- **Use for:**
  - ✅ Final exports for print (300 DPI)
  - ✅ Large display work (4K monitors)
  - ✅ Critical portraits requiring maximum sharpness
- **Typical latency:** 2-5s for 1080p, 10-15s for 24MP

---

## Batch Processing Optimization

### Worker Count (`--workers` flag)

```bash
# Fast feedback (1-2 images)
python3 cli.py /photos -o /out --workers 1

# Moderate load (10-50 images)
python3 cli.py /photos -o /out --workers 4

# Heavy batch (100+ images)
python3 cli.py /photos -o /out --workers 8
```

**Decision matrix:**
- **1 worker:** Memory-constrained systems, or testing a single image
- **4 workers:** Good default for most machines (3-4 CPU cores)
- **8 workers:** High-end machines with 8+ cores, large batches

**Memory per worker:**
- With `fast=True`: ~600 MB each → 4.8 GB total for 8 workers
- With proxy (auto): ~1.84 GB each → 14.7 GB total for 8 workers
- Be mindful: 4 workers on a system with 8 GB RAM is tight

---

## Memory Management

### Out of Memory on High-Res?

**Symptom:** `MemoryError` or system freeze on 4K+ images

**Fix (in priority order):**
1. ✅ **Proxy resolution activates automatically** — should handle up to 24MP
2. ✅ **Use `fast=True`** in GUI for interactive work (reduces to 800px)
3. ✅ **Use `--workers 1`** in CLI (one image at a time)
4. ✅ **Reduce batch size** (process 10 images instead of 100)

### Typical Memory Usage

| Scenario | Memory |
|----------|--------|
| GUI with fast=True (1080p) | ~600 MB |
| GUI fast=False (1080p) | ~1.2 GB |
| CLI batch 4 workers (fast=True) | ~2.4 GB |
| 24MP raw with proxy (auto) | ~1.84 GB |
| 24MP raw without proxy | ~7.5 GB |

---

## GUI vs CLI Performance

| Operation | Tool | Speed | Notes |
|-----------|------|-------|-------|
| Interactive tuning | GUI with `fast=True` | 2-5s per preview | Responsive, soft details |
| Batch 100 photos | CLI with `--workers 4` | 5-15min total | Parallelized, faster throughput |
| Single final export | CLI with `--recipe` | 2-15s | Full quality, no preview overhead |

---

## AI Denoise (F7 / NAFNet) Cost

`ai_denoise > 0` runs the NAFNet-SIDD-width32 ONNX model pre-pipeline, tiled
at 512px. It is pinned to **CPUExecutionProvider** — CoreML silently
miscomputes this graph (measured max abs error ~1.7-2.4 on [0,1] data) and is
~90x slower due to graph fragmentation, so do not "optimize" it back onto
CoreML without re-verifying correctness.

| Input size | Wall time (M3 Pro, CPU) |
|------------|-------------------------|
| 512×512 tile | ~0.6s |
| 1560×1040 | ~10s |
| 6240×4160 (24MP) | ~80-90s |

Guidance: reserve `ai_denoise` for genuinely noisy (high-ISO) sources; it is
opt-in per image/recipe. For interactive GUI tuning, set it once and rely on
FaceContext caching — the denoise runs on every full `process()` call.

---

## Real-World Timings

### Example 1: Interactive Portrait Session (GUI)
```
1. Load 1080p portrait, fast=True
   → 2-3s preview with recipe
2. Adjust 5 sliders, each preview
   → 2-3s each (cache detection/parsing, so faster)
3. Turn off fast, export full quality
   → 2-3s final render
Total: ~20-25s interactive session
```

### Example 2: Batch Process 50 Portraits
```
CLI: python3 cli.py ./batch/ -o ./out --workers 4 --recipe anime_v2
→ ~5-10min total (5-15s per image × 50 ÷ 4 workers)
Memory: ~2.4 GB sustained
```

### Example 3: Process a 24MP RAW → Print
```
1. Load 24MP RAW (automatic proxy to 2048px)
2. Tune recipe in GUI (fast=True)
3. Export with fast=False
Total: ~6-8s (pipeline is intelligent about resolution)
```

---

## Monitoring Performance

### Check Actual Timings

**In GUI:** Look at the status message after processing:
- Shows total time, face count, processing mode

**In CLI:** Use `--verbose` flag:
```bash
python3 cli.py image.jpg -o out.jpg --verbose
# Prints: Detection: 150ms, Parsing: 200ms, Grading: 300ms, Total: 650ms
```

**In code:**
```python
from retouch import RetouchEngine
engine = RetouchEngine()
result = engine.process(img, recipe="anime_v2")
print(result.timings)  # Access per-stage breakdowns
```

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| **GUI feels sluggish** | fast=False on high-res | Enable `fast=True` in GUI (always on by default) |
| **Batch jobs taking forever** | Too few workers | Increase `--workers` to 4-8 |
| **Out of memory on 4K** | Proxy not kicking in | Ensure image is actually > 2048px; check available RAM |
| **Inconsistent timings** | Thermal throttling | Close other apps; run on power adapter if laptop |
| **Batch slower than expected** | Single-threaded code | Ensure `--workers > 1` and no other heavy processes |

---

## Best Practices

- ✅ Use `fast=True` in GUI (it's the default for a reason)
- ✅ Turn off fast only for final export
- ✅ Batch with `--workers 4-8` for 10+ images
- ✅ Monitor memory with `top` or Activity Monitor if unsure
- ✅ Use proxy resolution automatically (don't disable it)
- ✅ Test on representative images (portrait, group, high-res) before full batch

---

**Summary:** Start with defaults. They're tuned for 95% of workflows. Adjust only if you hit performance walls.

# Troubleshooting Guide

Quick fixes for common issues. If your problem isn't here, open an issue on GitHub.

---

## Installation & Setup

### ImportError: No module named 'retouch'
**Cause:** Package not installed or venv not activated.

**Fix:**
```bash
source venv/bin/activate
pip install -e .  # Install in editable mode
```

### Models downloading very slowly
**Cause:** Large files (~800 MB). First run downloads all models.

**Fix:**
1. ✅ Just wait (takes 5-15 min depending on connection)
2. Or manually download from S3 (see ARCHITECTURE.md for URLs)
3. Place in `models/` directory

**Check status:**
```bash
ls -lh models/
# Should have: face_landmarker.task, selfie_segmenter.tflite, resnet18.onnx
```

### CUDA/GPU not available (ONNX warning)
**Cause:** ONNX Runtime falling back to CPU (normal behavior).

**Fix:** GPU acceleration is optional. CPU-only is fine:
```bash
# No action needed — inference still works
python3 gui.py  # Runs on CPU, slower but correct
```

**To enable GPU:**
```bash
pip install onnxruntime-gpu
# Requires CUDA 11+ and cuDNN
```

---

## Face Detection & Processing

### No faces detected in image
**Symptom:** "0 faces found" message, no processing applied.

**Cause:** Image has extreme profile, heavy occlusion, or faces are tiny.

**Fix (priority order):**
1. ✅ Try `--global-only` flag (skips detection, applies only color grading)
   ```bash
   python3 cli.py image.jpg -o out.jpg --global-only
   ```
2. ✅ Verify face is front-ish (90°+ profile won't work)
3. ✅ Try a closer crop if faces are distant
4. ✅ Brighten image if it's very dark (detection struggles in low light)

**Example:**
```bash
python3 cli.py distant_portrait.jpg -o out.jpg --global-only --recipe anime_v2
# Applies grading even without face detection
```

### Faces detected but processing looks wrong
**Symptom:** Retouched face looks over-softened, colors are off, or artifacts appear.

**Causes & fixes:**
| Symptom | Cause | Fix |
|---------|-------|-----|
| Too soft/blurry | Using `fast=True` on high-res | Use `fast=False` for export, keep `fast=True` for GUI preview |
| Color cast (too warm/cool) | Wrong recipe or white balance | Try different recipe: `--recipe anime_v2` vs `--recipe cosplay` |
| Patchiness or halos | Frequency separation artifact | Use `--recipe classic_chrome` (less aggressive smoothing) |
| Eyes look weird | Over-enhancement | Reduce `eye_brightening` slider in GUI or CLI |

---

## Performance & Speed

### GUI feels sluggish / slow previews
**Symptom:** Preview takes 10+ seconds even for small images.

**Cause:** `fast=False` on high-res image, or running on low-end machine.

**Fix:**
1. ✅ **Enable fast mode** (GUI default — check it's ON)
   - Settings → Fast mode: ON
   - Cuts 3-5× faster (but softer details)
2. ✅ **For final export:** Turn off fast mode just before export
3. ✅ **Close other apps** to free RAM

**Typical timings (1080p image):**
- `fast=True`: 2-3s preview ✅
- `fast=False`: 5-10s final render

### Batch processing is very slow
**Symptom:** 50 images taking 30+ minutes to process.

**Cause:** Only 1 worker processing serially, or `fast=False` on high-res.

**Fix:**
```bash
# Increase workers (4-8 recommended for most machines)
python3 cli.py ./photos -o ./out --workers 4

# Or use fast mode for faster throughput
python3 cli.py ./photos -o ./out --workers 4 --fast

# For high-end machines (8+ cores)
python3 cli.py ./photos -o ./out --workers 8
```

**Memory check:**
```bash
# Monitor while running
top  # Watch %MEM column
# 4 workers × 600 MB = 2.4 GB typical
```

### Out of memory on 4K images
**Symptom:** `MemoryError` or system freezes on 24MP+ images.

**Cause:** Insufficient RAM or proxy resolution not activating.

**Fix (priority order):**
1. ✅ **Check available RAM:**
   ```bash
   free -h  # Linux/Mac
   ```
   Need ≥1.2 GB free (2+ GB for safety)

2. ✅ **Enable fast mode** (reduces to 800px)
   ```bash
   python3 cli.py image_4k.jpg -o out.jpg --fast
   ```

3. ✅ **Use --workers 1** (serial processing)
   ```bash
   python3 cli.py large_batch/ -o out/ --workers 1
   ```

4. ✅ **Verify proxy activates** (automatic for images > 2048px)
   - Image automatically downscaled to 2048px internally
   - Upscaled back to original on output
   - Check with `--verbose` flag

5. ❌ Last resort: close all other apps, restart machine

---

## Recipes & Presets

### Recipe not having any effect
**Symptom:** Image looks identical before/after, wrong color, or recipe silently ignored.

**Cause:** Recipe has errors, or live-editing a built-in recipe (overwrites not saved).

**Fix:**
1. ✅ **Try a different recipe** to verify engine works
   ```bash
   python3 cli.py image.jpg -o out1.jpg --recipe anime_v2
   python3 cli.py image.jpg -o out2.jpg --recipe classic_chrome
   # Compare out1.jpg and out2.jpg — should look different
   ```

2. ✅ **Check recipe keys** (might have typos)
   ```bash
   python3 -c "from retouch.recipes import RECIPES; print(RECIPES['anime_v2'].keys())"
   ```

3. ✅ **Save custom recipes to a file** (don't edit built-ins)
   - GUI: Export recipe → custom_recipe.json
   - Load: Import recipe from file

4. ✅ **Verify no dead keys** (silently ignored)
   ```bash
   python3 -m pytest tests/test_recipe_validation.py -v
   # Catches recipes with unimplemented parameters
   ```

### Custom recipe won't load
**Symptom:** "Invalid recipe" error, or recipe loads but parameters are ignored.

**Cause:** Malformed JSON, or recipe keys don't map to known parameters.

**Fix:**
1. ✅ **Validate JSON syntax**
   ```bash
   jq . custom_recipe.json  # Should print without error
   ```

2. ✅ **Check key names** (must match PROCESSING_PARAMS registry)
   ```bash
   python3 -c "from retouch.params import PROCESSING_PARAMS; print([s.name for s in PROCESSING_PARAMS])" | head -20
   ```

3. ✅ **Compare to working recipe**
   ```bash
   python3 -c "from retouch.recipes import RECIPES; import json; print(json.dumps(RECIPES['anime_v2'], indent=2))" | head -30
   ```

4. ✅ **Use recipe validator** to catch errors
   ```bash
   python3 -m pytest tests/test_recipe_validation.py::test_no_dead_recipe_keys -v
   ```

---

## GUI Issues

### GUI won't start
**Symptom:** Error when running `python3 gui.py`, or browser can't connect.

**Cause:** Port 7860 in use, dependencies missing, or models not downloaded.

**Fix:**
1. ✅ **Check port is free**
   ```bash
   lsof -i :7860  # If something appears, kill it
   kill -9 <PID>
   ```

2. ✅ **Reinstall dependencies**
   ```bash
   pip install --upgrade -r requirements-gui.txt
   ```

3. ✅ **Download models**
   ```bash
   python3 -c "from retouch import RetouchEngine; RetouchEngine()"
   ```

4. ✅ **Check browser compatibility** (Chrome/Safari/Firefox all work)

### GUI slider changes not applying
**Symptom:** Move slider, but preview doesn't update.

**Cause:** FaceContext caching issue or GPU memory fragmentation.

**Fix:**
1. ✅ **Reload image** (forces re-detection and cache invalidation)
   - Upload → Browse → select same image → OK

2. ✅ **Restart GUI**
   ```bash
   # Ctrl+C to stop, then:
   python3 gui.py
   ```

3. ✅ **Clear GPU cache** (if using ONNX GPU)
   ```bash
   python3 -c "import onnxruntime; onnxruntime.get_available_providers()"
   # Should show CUDAExecutionProvider if GPU is working
   ```

### Export freezes or takes forever
**Symptom:** Click "Export Full Quality" but nothing happens, or UI becomes unresponsive.

**Cause:** Large image + full resolution (`fast=False`) processing.

**Workaround:**
- ✅ Expected behavior: Export can take 15-30s for 4K images
- ✅ Check terminal for progress messages (if `--verbose` enabled)
- ✅ Use `fast=True` for faster preview, then export in CLI for batches

---

## CLI Issues

### CLI hangs on batch processing
**Symptom:** Process starts but never finishes, 100% CPU.

**Cause:** Model deadlock (rare), or worker pool issue.

**Fix:**
1. ✅ **Kill process** and retry with `--workers 1`
   ```bash
   Ctrl+C  # Stop current run
   python3 cli.py ./batch --workers 1 -o ./out
   ```

2. ✅ **Check for zombie processes**
   ```bash
   ps aux | grep python  # Look for stuck processes
   pkill -f "cli.py"     # Kill all Python CLI processes
   ```

3. ✅ **Reduce batch size** (process 10 images instead of 100)
   ```bash
   python3 cli.py ./batch --skip 100 --workers 4 -o ./out
   # Or manually split directory
   ```

### CLI crashes with "broken pipe" error
**Symptom:** Processing starts but crashes mid-run with broken pipe.

**Cause:** Worker process died (OOM, model error, or filesystem issue).

**Fix:**
1. ✅ **Retry with fewer workers**
   ```bash
   python3 cli.py ./batch --workers 2 -o ./out
   ```

2. ✅ **Check disk space**
   ```bash
   df -h .  # Need ≥1 GB free for output
   ```

3. ✅ **Enable verbose logging**
   ```bash
   python3 cli.py ./batch --verbose --workers 1 -o ./out 2>&1 | tee batch.log
   ```

---

## Color & Output Issues

### Output colors don't match input (too saturated/desaturated)
**Symptom:** Retouched image looks over-processed, colors are wrong.

**Cause:** Wrong color space, LUT loading failed, or recipe over-adjusts.

**Fix:**
1. ✅ **Verify input color space** (sRGB expected)
   ```bash
   identify image.jpg  # Shows color space (should be sRGB)
   ```

2. ✅ **Try different recipe**
   ```bash
   python3 cli.py image.jpg -o out1.jpg --recipe cosplay
   python3 cli.py image.jpg -o out2.jpg --recipe anime_v2
   # Compare — one may look better
   ```

3. ✅ **Disable color grading** (use --global-only)
   ```bash
   python3 cli.py image.jpg -o out_plain.jpg --global-only
   # Should look minimal but correct
   ```

4. ✅ **Check ICC profile embedding**
   ```bash
   # Output should have sRGB ICC profile
   identify -verbose out.jpg | grep -i "icc"
   ```

### Output is grayscale or very desaturated
**Symptom:** Color completely lost or severely reduced in output.

**Cause:** ICC profile mismatch, image decode error, or recipe desaturation too high.

**Fix:**
1. ✅ **Verify input has color**
   ```bash
   identify image.jpg  # Should show "sRGB" colorspace, not "Gray"
   ```

2. ✅ **Try without recipe** (test engine color handling)
   ```bash
   python3 cli.py image.jpg -o test.jpg --global-only
   # Should preserve colors even without recipe
   ```

3. ✅ **Reduce saturation slider** in GUI if recipe is desaturating

4. ✅ **Check output file** isn't corrupt
   ```bash
   file out.jpg  # Should say "JPEG image data"
   ```

---

## Logs & Debugging

### How do I enable debug logging?
**Use `--verbose` flag:**
```bash
python3 cli.py image.jpg -o out.jpg --verbose
# Prints: Detection: 150ms, Parsing: 200ms, Grading: 300ms, etc.
```

### Where are crash logs?
**Location:** Current working directory.

**Format:** `crash_YYYYMMDD_HHMMSS.log`

**Example:**
```bash
ls -la crash_*.log
tail -f crash_*.log  # Watch in real-time
```

### How to report a bug
1. **Reproduce the issue** with a specific image
2. **Enable debug logging** (`--verbose` or check crash logs)
3. **Note the environment:**
   ```bash
   python3 --version
   pip list | grep retouch
   ```
4. **Open an issue** with:
   - Image (or representative sample)
   - Exact command you ran
   - Verbose output / crash log
   - Environment (OS, Python version)

---

## Advanced Troubleshooting

### How to profile performance
```bash
python3 scripts/benchmark.py
# Runs standard benchmark suite, shows per-stage timings
```

### How to test a specific module
```bash
python3 -m pytest tests/test_grading.py -v  # Just grading module
python3 -m pytest tests/ -k detection -v    # All detection tests
```

### How to check test coverage
```bash
python3 -m pytest tests/ --cov=retouch --cov-report=html
open htmlcov/index.html  # Detailed coverage report
```

### How to validate all recipes
```bash
python3 -m pytest tests/test_recipe_validation.py -v
# Catches dead keys and malformed presets
```

---

## Still Stuck?

1. **Check ARCHITECTURE.md** for pipeline design & module responsibilities
2. **Check PERFORMANCE_TUNING.md** for speed/quality trade-offs
3. **Check CLAUDE.md § Debugging** for crash logging & common issues
4. **Check API.md** for Python API reference
5. **Open an issue** with debug logs, environment, and reproduction steps

---

**Last Updated:** 2026-07-02  
**Questions?** Open an issue or discussion on GitHub.

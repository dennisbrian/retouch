#!/usr/bin/env python3
"""Benchmark script for RetouchEngine.

Measures stage execution times and peak memory usage across different resolutions.
"""

import os
import sys
import time
import tracemalloc
import numpy as np
import cv2

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retouch import RetouchEngine


def run_benchmark(image_path, target_megapixels):
    print(f"\n--- Benchmarking at ~{target_megapixels} MP ---")
    if not os.path.exists(image_path):
        print(f"Error: Sample image not found at {image_path}")
        return

    # Load input
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Calculate scale factor for target Megapixels
    current_mp = (h * w) / 1000000.0
    scale = (target_megapixels / current_mp) ** 0.5

    target_h = int(h * scale)
    target_w = int(w * scale)
    img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    print(f"Resized input from {w}x{h} ({current_mp:.1f} MP) to {target_w}x{target_h} ({target_megapixels:.1f} MP)")

    # Start memory tracing
    tracemalloc.start()
    
    # Initialize engine
    engine = RetouchEngine()
    
    t0 = time.perf_counter()
    result = engine.process(img_resized, recipe="natural")
    total_time = (time.perf_counter() - t0) * 1000
    
    # Get memory stats
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    engine.close()

    # Print timings and memory
    print(f"Face count: {result.face_count}")
    print("Timings (ms):")
    for stage, ms in result.timings.items():
        print(f"  {stage:12}: {ms:7.1f} ms")
    print(f"Peak Memory usage: {peak_mem / (1024 * 1024):.1f} MB")


def warmup():
    """Warm up JIT/CoreML/GPU to avoid cold-start bias in timed runs."""
    engine = RetouchEngine()
    tiny = np.full((100, 100, 3), 128, dtype=np.uint8)
    engine.process(tiny, recipe="natural")
    engine.close()


def main():
    # Use one of the sample images in the repository
    sample_img = "test_output/DSCF4550.jpg"
    if not os.path.exists(sample_img):
        # Fallback to any other jpeg in test_output
        import glob
        jpgs = glob.glob("test_output/*.jpg")
        if jpgs:
            sample_img = jpgs[0]

    # Warm up before timing to avoid cold-start artifacts
    warmup()

    # Benchmark at 12 MP, 24 MP, and 50 MP (or lower if testing fast)
    run_benchmark(sample_img, 12.0)
    run_benchmark(sample_img, 24.0)
    # We do a quick 3 MP run to ensure fast testing too
    run_benchmark(sample_img, 3.0)


if __name__ == "__main__":
    main()

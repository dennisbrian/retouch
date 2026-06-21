#!/usr/bin/env python3
"""Benchmark script for RetouchEngine.

Measures stage execution times and peak memory usage across different resolutions.
"""

import os
import sys
import time
import gc
import numpy as np
import cv2

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retouch import RetouchEngine

# Try to use psutil for accurate process memory, fallback to resource module
try:
    import psutil
    def get_process_memory_mb():
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
except ImportError:
    import resource
    def get_process_memory_mb():
        # ru_maxrss is in kilobytes on Linux, bytes on macOS
        if sys.platform == 'darwin':
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_benchmark(engine, image_path, target_megapixels, iterations=3):
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
    
    # Use optimal interpolation flags
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    img_resized = cv2.resize(img, (target_w, target_h), interpolation=interp)
    print(f"Input size: {target_w}x{target_h} ({target_megapixels:.1f} MP)")

    # Force garbage collection before starting
    gc.collect()
    
    times = []
    peak_mem = 0.0
    base_mem = get_process_memory_mb()
    
    best_timings = None
    face_count = 0
    best_run_time = float('inf')
    
    for i in range(iterations):
        t0 = time.perf_counter()
        result = engine.process(img_resized, recipe="natural")
        t1 = time.perf_counter()
        
        elapsed = (t1 - t0) * 1000
        times.append(elapsed)
        
        # Capture timings and face count from the best run to avoid running a 4th time
        if elapsed < best_run_time:
            best_run_time = elapsed
            best_timings = result.timings.copy()
            face_count = result.face_count
        
        current_mem = get_process_memory_mb()
        if current_mem > peak_mem:
            peak_mem = current_mem
            
        print(f"  Run {i+1}/{iterations}: {elapsed:7.1f} ms")

    # Clean up result to free memory
    del result
    gc.collect()

    best_time = min(times)
    avg_time = sum(times) / len(times)
    mem_delta = peak_mem - base_mem

    print(f"Face count: {face_count}")
    print("Best Run Timings (ms):")
    if best_timings:
        for stage, ms in best_timings.items():
            print(f"  {stage:12}: {ms:7.1f} ms")

    print(f"Timing Summary: Best = {best_time:.1f} ms | Avg = {avg_time:.1f} ms")
    print(f"Peak Memory usage: {peak_mem:.1f} MB (Delta: +{mem_delta:.1f} MB)")


def main():
    # Use one of the sample images in the repository
    sample_img = "test_output/DSCF4550.jpg"
    if not os.path.exists(sample_img):
        import glob
        jpgs = glob.glob("test_output/*.jpg")
        if jpgs:
            sample_img = jpgs[0]
        else:
            print("No sample images found in test_output/. Exiting.")
            return

    print("Initializing RetouchEngine...")
    engine = RetouchEngine()
    
    # Warm up model loading and JIT caches
    print("Warming up...")
    tiny = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    engine.process(tiny, recipe="natural")

    # Benchmark at different resolutions
    run_benchmark(engine, sample_img, 3.0)
    run_benchmark(engine, sample_img, 12.0)
    run_benchmark(engine, sample_img, 24.0)

    engine.close()
    print("\nBenchmarking complete.")


if __name__ == "__main__":
    main()

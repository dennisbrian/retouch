#!/usr/bin/env python3
"""Peak-RSS and time characterisation for large editor stacks and canvases.

The compositor keeps float32 working memory bounded by ``band_rows`` x canvas
width, so the cost that grows without bound is the *source* pixel and mask
arrays held per layer, plus the flattened uint8 canvas. This measures both the
document limits (``MAX_SIDE`` = 30000 per side, ``MAX_SURFACE_PIXELS`` = 200 MP)
and the point where a deep stack or a maximum-size canvas approaches the
machine's memory.

Each case runs in its own process so ``ru_maxrss`` is that case's peak:

    .venv/bin/python scripts/dev/editor_scale_benchmark.py --all
    .venv/bin/python scripts/dev/editor_scale_benchmark.py --case canvas_14142 --json

No photo assets are needed; layers are synthetic fills. This is an editor
workload, not a Retouch recipe certification.
"""
import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from retouch.editor import Document, Layer, composite


def peak_mib():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 1)


# name -> (canvas_w, canvas_h, layer_count, layer_w, layer_h, save)
CASES = {
    'canvas_8000':   (8000, 8000, 1, 8000, 8000, False),      # 64 MP
    'canvas_12000':  (12000, 12000, 1, 12000, 12000, False),  # 144 MP
    'canvas_14142':  (14142, 14142, 1, 14142, 14142, False),  # ~200 MP, export limit
    'stack_16':      (4000, 3000, 16, 4000, 3000, False),     # 12 MP x 16 = 192 MP
    'stack_64':      (4000, 3000, 64, 1000, 750, False),      # 64 layers x 0.75 MP = 48 MP
    'stack_over_budget': (4000, 3000, 64, 4000, 3000, False), # 768 MP: must be refused
    'canvas_200mp_4': (14142, 14142, 4, 7071, 7071, False),   # 200 MP canvas, 4 x 50 MP
    'save_12000':    (12000, 12000, 1, 12000, 12000, True),   # max-size package write
}


def run_case(name, out_dir):
    canvas_w, canvas_h, count, layer_w, layer_h, do_save = CASES[name]
    report = {'case': name, 'canvas': [canvas_w, canvas_h], 'layers': count,
              'layer_pixels_mp': round(count * layer_w * layer_h / 1e6, 1),
              'surface_pixels_mp': round(canvas_w * canvas_h / 1e6, 1)}

    timer = time.perf_counter()
    layers = []
    for i in range(count):
        pixels = np.empty((layer_h, layer_w, 4), np.uint8)
        pixels[..., :3] = (i * 37 % 256, i * 91 % 256, i * 13 % 256)
        pixels[..., 3] = 255 if i % 2 else 200
        origin = (0, 0) if count == 1 else ((i * 53) % max(1, canvas_w - layer_w),
                                            (i * 71) % max(1, canvas_h - layer_h))
        layers.append(Layer(f'L{i}', pixels, origin=origin,
                            blend_mode=('Normal', 'Multiply', 'Screen')[i % 3]))
    report['build_s'] = round(time.perf_counter() - timer, 2)

    document = Document(canvas_w, canvas_h, layers)
    timer = time.perf_counter()
    try:
        image = composite(document)
    except ValueError as error:
        report['refused'] = str(error)
        report['peak_rss_mib'] = peak_mib()
        return report
    report['composite_s'] = round(time.perf_counter() - timer, 2)
    report['output_shape'] = list(image.shape)
    report['peak_rss_mib'] = peak_mib()

    if do_save:
        from retouch.editor import save_project, load_project
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        timer = time.perf_counter()
        path = save_project(document, Path(out_dir) / f'{name}.comp')
        report['save_s'] = round(time.perf_counter() - timer, 2)
        report['package_mib'] = round(
            sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / (1024 * 1024), 1)
        timer = time.perf_counter()
        load_project(path)
        report['reload_s'] = round(time.perf_counter() - timer, 2)
        report['peak_rss_mib'] = peak_mib()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--case', choices=sorted(CASES))
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--out', type=Path, default=Path('test_output/editor_scale'))
    args = parser.parse_args(argv)
    if args.all:
        args.out.mkdir(parents=True, exist_ok=True)
        reports = []
        for name in CASES:
            proc = subprocess.run(
                [sys.executable, __file__, '--case', name, '--json', '--out', str(args.out)],
                capture_output=True, text=True)
            if proc.returncode != 0:
                reports.append({'case': name, 'error': proc.stderr.strip().splitlines()[-1:]})
            else:
                reports.append(json.loads(proc.stdout))
        (args.out / 'report.json').write_text(json.dumps(reports, indent=2))
        print(json.dumps(reports, indent=2))
        return 0
    if not args.case:
        parser.error('pass --case NAME or --all')
    report = run_case(args.case, args.out)
    print(json.dumps(report) if args.json else json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())

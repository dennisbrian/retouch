#!/usr/bin/env python3
"""Real-photo reference check for the editor core (compositor port, milestone 1).

Given an original photo and a Retouch render of it at the same size, this
writes the Compositor handoff (``retouch.compositor.export_project``), reopens
it with ``retouch.editor.load_project``, flattens it, saves and reloads it, and
reports whether each step reproduces the expected pixels:

  * handoff composite == the render, exactly (masks are 0/255, opacity 1);
  * result layer hidden == the original, exactly;
  * result layer at 50% opacity == the analytic sRGB blend, within 1 level;
  * save/reload keeps every layer bit-identical.

Plus run time, peak RSS and SHA-256 of each flattened output, so later
milestones can pin them. Photos are not committed; pass your own.

    .venv/bin/python scripts/dev/compositor_reference.py \
        --before photo.jpg --after render.png --out /tmp/compositor-ref
"""
import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from retouch.compositor import export_project
from retouch.editor import composite, load_project, save_project
from retouch.io import imread_exif


def _rgb(path):
    return np.ascontiguousarray(imread_exif(path)[..., ::-1])


def _sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def check(before_path, after_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    before, after = _rgb(before_path), _rgb(after_path)
    if before.shape != after.shape:
        raise SystemExit(f'before {before.shape} and after {after.shape} differ in size')
    stem = Path(after_path).stem
    report = {'photo': stem, 'size': [before.shape[1], before.shape[0]]}
    timer = time.perf_counter()
    project = export_project(before, after, out_dir / (stem + '.comp'))
    report['handoff_s'] = round(time.perf_counter() - timer, 2)

    timer = time.perf_counter()
    doc = load_project(project)
    report['load_s'] = round(time.perf_counter() - timer, 2)
    timer = time.perf_counter()
    full = composite(doc)
    report['composite_s'] = round(time.perf_counter() - timer, 2)
    report['handoff_equals_render'] = bool(np.array_equal(full[..., :3], after))
    report['opaque'] = bool((full[..., 3] == 255).all())
    report['changed_pixels'] = int((doc.layers[1].mask > 0).sum())

    result = doc.layer(doc.active_layer_id)
    result.opacity = 0.5
    half = composite(doc)
    mask = result.mask[..., None].astype(np.float32) / 255 * 0.5
    expected = np.rint(after * mask + before * (1 - mask))
    report['half_opacity_max_error'] = int(np.abs(half[..., :3] - expected).max())

    result.visible = False
    hidden = composite(doc)
    report['hidden_equals_original'] = bool(np.array_equal(hidden[..., :3], before))
    result.visible, result.opacity = True, 1.0

    timer = time.perf_counter()
    saved = save_project(doc, out_dir / (stem + '-resaved.comp'))
    again = load_project(saved)
    report['save_reload_s'] = round(time.perf_counter() - timer, 2)
    report['round_trip_identical'] = all(
        (a.pixels is None and b.pixels is None or np.array_equal(a.pixels, b.pixels))
        and (a.mask is None and b.mask is None or np.array_equal(a.mask, b.mask))
        and (a.name, a.visible, a.opacity, a.blend_mode) == (b.name, b.visible, b.opacity, b.blend_mode)
        for a, b in zip(doc.layers, again.layers))
    report['sha256'] = {'full': _sha(full), 'half_opacity': _sha(half), 'hidden': _sha(hidden)}
    report['peak_rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024

    # Contact sheet for a visual check: original | 50% opacity | handoff.
    sheet = np.concatenate([before, half[..., :3], full[..., :3]], axis=1)
    scale = 1800 / sheet.shape[1]
    sheet = cv2.resize(sheet, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_dir / (stem + '-sheet.jpg')), sheet[..., ::-1],
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--before', required=True, action='append')
    parser.add_argument('--after', required=True, action='append')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    if len(args.before) != len(args.after):
        parser.error('pass one --after per --before')
    reports = [check(b, a, args.out) for b, a in zip(args.before, args.after)]
    (args.out / 'report.json').write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))
    ok = all(r['handoff_equals_render'] and r['hidden_equals_original']
             and r['half_opacity_max_error'] <= 1 and r['round_trip_identical']
             for r in reports)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

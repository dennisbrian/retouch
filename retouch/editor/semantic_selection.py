"""Automatic document selections using the existing detector/parser in a child."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .document import MAX_SURFACE_PIXELS

REGIONS = ('Skin', 'Hair', 'Eyes', 'Lips', 'Face', 'Person', 'Background')
MAX_INFERENCE_SIDE = 2048


def validate_options(region: str, feather: float) -> None:
    if region not in REGIONS:
        raise ValueError('Choose an available automatic selection region.')
    if (isinstance(feather, bool) or not isinstance(feather, (int, float))
            or not math.isfinite(feather) or not 0 <= feather <= 250):
        raise ValueError('Feather must be between 0 and 250 pixels.')


def infer_mask(rgb: np.ndarray, region: str, detector: Any, parser: Any) -> np.ndarray:
    """Use the Advanced Retouch masks, preserving their RGB/BGR contract."""
    from retouch.advanced_retouch import semantic_masks_for_selection

    validate_options(region, 0)
    if region in ('Person', 'Background'):
        mask = detector.segment_person(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        # FaceDetector returns a whole-frame mask if its segmenter is absent
        # or supplies no result. That fallback is not measured subject coverage.
        if mask is not None and np.asarray(mask).size and np.all(np.asarray(mask) == 1):
            raise ValueError('Person segmentation did not identify a boundary. The current selection is unchanged.')
        if mask is not None and region == 'Background':
            mask = 1 - np.asarray(mask, dtype=np.float32)
    else:
        masks = semantic_masks_for_selection(rgb, detector, parser, 'All faces')
        if not masks:
            raise ValueError('No faces detected. Try a clearer or closer photo.')
        mask = masks.get(region)
    if mask is None:
        raise ValueError(f'The {region.lower()} mask is unavailable for this photo.')
    mask = np.asarray(mask, dtype=np.float32)
    if mask.shape != rgb.shape[:2] or not np.isfinite(mask).all():
        raise ValueError('Automatic selection returned an invalid mask.')
    if mask.min() < 0 or mask.max() > 1:
        raise ValueError('Automatic selection returned out-of-range coverage.')
    if not mask.any():
        raise ValueError(f'No {region.lower()} region was found. The current selection is unchanged.')
    return mask


def render_worker(directory: Path, detector_factory: Any = None, parser_factory: Any = None) -> None:
    """Own native models on the child main thread and release them explicitly."""
    if detector_factory is None:
        from retouch.detection import FaceDetector
        detector_factory = FaceDetector
    if parser_factory is None:
        from retouch.parsing import FaceParser
        parser_factory = FaceParser
    settings = json.loads((directory / 'settings.json').read_text())
    validate_options(settings['region'], 0)
    with Image.open(directory / 'input.png') as image:
        rgb = np.asarray(image.convert('RGB'))
    detector = parser = None
    try:
        detector = detector_factory()
        if settings['region'] not in ('Person', 'Background'):
            parser = parser_factory()
        mask = infer_mask(rgb, settings['region'], detector, parser)
        Image.fromarray(np.rint(mask * 255).astype(np.uint8)).save(directory / 'mask.png')
    finally:
        try:
            if parser is not None:
                parser.close()
        finally:
            if detector is not None:
                detector.close()


def automatic_selection(rgba: np.ndarray, region: str, feather: float,
                        workspace: str, timeout: float = 120) -> np.ndarray:
    """Bound inference size, time and lifetime; return native document coverage."""
    validate_options(region, feather)
    if (rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8
            or rgba.shape[0] * rgba.shape[1] > MAX_SURFACE_PIXELS):
        raise ValueError('Automatic selection requires a supported RGBA canvas.')
    if not (rgba[..., 3] == 255).all():
        raise ValueError('Automatic selection currently requires an opaque composite.')
    height, width = rgba.shape[:2]
    factor = min(1, MAX_INFERENCE_SIDE / max(height, width))
    size = (max(1, round(width * factor)), max(1, round(height * factor)))
    rgb = rgba[..., :3]
    if factor < 1:
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    executable = os.environ.get('RETOUCH_EDITOR_PYTHON') or (
        None if getattr(sys, 'frozen', False) else sys.executable)
    if not executable:
        raise ValueError('Automatic selection is unavailable in this packaged runtime.')
    environment = os.environ.copy()
    environment.setdefault('RETOUCH_MEDIAPIPE_BACKEND', 'legacy')
    with tempfile.TemporaryDirectory(prefix='selection-', dir=workspace) as temporary:
        directory = Path(temporary)
        Image.fromarray(rgb).save(directory / 'input.png')
        (directory / 'settings.json').write_text(json.dumps({'region': region}))
        with (directory / 'worker.log').open('w') as log:
            try:
                result = subprocess.run(
                    [str(executable), '-u', '-m', 'retouch.editor.semantic_selection', str(directory)],
                    cwd=str(Path(__file__).resolve().parents[2]), env=environment,
                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
            except subprocess.TimeoutExpired as exc:
                raise ValueError('Automatic selection timed out; the current selection is unchanged.') from exc
        if result.returncode:
            error = directory / 'error.json'
            message = (json.loads(error.read_text())['error'] if error.exists()
                       else 'The face detector or parser could not finish. Check the installed models.')
            raise ValueError(message)
        with Image.open(directory / 'mask.png') as image:
            if image.mode != 'L' or image.size != size:
                raise ValueError('Automatic selection returned an invalid mask image.')
            mask = np.asarray(image).copy()
    if size != (width, height):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    if feather:
        mask = cv2.GaussianBlur(mask, (0, 0), feather / 2, borderType=cv2.BORDER_REPLICATE)
    if not mask.any():
        raise ValueError('No selectable coverage was found; the current selection is unchanged.')
    return mask


if __name__ == '__main__':
    arguments = argparse.ArgumentParser(description='Private automatic-selection worker')
    arguments.add_argument('directory', type=Path)
    directory = arguments.parse_args().directory
    try:
        render_worker(directory)
    except (ValueError, RuntimeError, OSError) as exc:
        (directory / 'error.json').write_text(json.dumps({'error': str(exc)}))
        raise SystemExit(1)

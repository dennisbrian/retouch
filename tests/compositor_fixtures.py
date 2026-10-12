"""Hand-written ``.comp`` packages for the editor-core tests.

Written straight from the public format (Compositor's
``docs/writing-comp-files.md`` at 11d8d7a), not with
``retouch.editor.save_project``, so the reader is checked against the format
rather than against our own writer. Every fixture is generated from literals:
nothing is downloaded and the bytes are reproducible.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

# Fixed ids so manifests are byte-stable between runs.
DOC_ID = '0C5E7A91-3B2D-4F6A-8E1C-9D0B7A6F5E4D'
IDS = [
    '6F1D3C2A-0B7E-4E8A-9C4D-2A1B3C4D5E6F',
    'A1B2C3D4-E5F6-4A7B-8C9D-0E1F2A3B4C5D',
    '11111111-2222-4333-8444-555555555555',
]


def transform(width, height, origin=(0, 0), flip_x=False, flip_y=False,
              sampling='High quality', rotation=0):
    return {'origin': list(origin), 'size': [width, height], 'rotation': rotation,
            'flipX': flip_x, 'flipY': flip_y, 'sampling': sampling}


def layer_record(index, name, width, height, image=True, **extra):
    layer_id = IDS[index]
    record = {'id': layer_id, 'name': name, 'isVisible': True, 'isGroup': False,
              'opacity': 1, 'blendMode': 'Normal',
              'transform': transform(width, height)}
    if image:
        record['imageFile'] = layer_id + '.png'
    record.update(extra)
    return record


def write_package(path, width, height, layers, images=None, masks=None,
                  version=11, **top):
    """Write a package. ``images``/``masks`` map layer index -> array."""
    path = Path(path)
    (path / 'images').mkdir(parents=True)
    for index, pixels in (images or {}).items():
        mode = 'RGBA' if pixels.shape[-1] == 4 else 'RGB'
        Image.fromarray(pixels, mode).save(path / 'images' / (IDS[index] + '.png'))
    for index, mask in (masks or {}).items():
        Image.fromarray(mask, 'L').save(path / 'images' / (IDS[index] + '.mask.png'))
    manifest = {'format': 'com.compositor.project', 'version': version,
                'colorSpace': 'sRGB', 'documentID': DOC_ID, 'width': width,
                'height': height, 'resolution': 72, 'layers': layers}
    if layers:
        manifest['activeLayerID'] = layers[-1]['id']
    manifest.update(top)
    (path / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return path


def solid(width, height, rgba):
    return np.tile(np.array(rgba, np.uint8), (height, width, 1))


def two_grays(path, top_opacity=1.0):
    """LayerAppearanceTests: an 80% gray layer over a 40% gray one, 4x4."""
    low, high = round(0.4 * 255), round(0.8 * 255)
    return write_package(
        path, 4, 4,
        [layer_record(0, 'Gray 40', 4, 4),
         layer_record(1, 'Gray 80', 4, 4, opacity=top_opacity)],
        images={0: solid(4, 4, (low, low, low, 255)),
                1: solid(4, 4, (high, high, high, 255))})


# LayerMaskTests: a red 2x2 layer, mask bytes [255, 0, 128, 255] row-major.
MASK_2X2 = np.array([[255, 0], [128, 255]], np.uint8)


def red_with_mask(path, mask=MASK_2X2, enabled=None, opacity=1.0, flip_x=False):
    extra = {'maskFile': IDS[0] + '.mask.png', 'opacity': opacity}
    if enabled is not None:
        extra['maskEnabled'] = enabled
    record = layer_record(0, 'Red', 2, 2, **extra)
    record['transform'] = transform(2, 2, flip_x=flip_x, sampling='Nearest')
    return write_package(path, 2, 2, [record],
                         images={0: solid(2, 2, (255, 0, 0, 255))},
                         masks={0: mask})

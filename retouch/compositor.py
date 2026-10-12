"""Editable, pixel-preserving handoff to Compositor's public project format.

Inputs are effective 8-bit working-sRGB RGB canvases, not engine previews.
Every handoff is a new project: manual edits in existing projects are never
overwritten. No desktop dependency is required to create or download it.
"""
from pathlib import Path
import json
import shutil
import uuid
import zipfile

import numpy as np
from PIL import Image


def export_project(base_rgb, result_rgb, destination):
    """Write a new .comp package; return its path, refusing existing targets."""
    base, result = np.asarray(base_rgb), np.asarray(result_rgb)
    for image in (base, result):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Compositor requires uint8 RGB working-sRGB canvases")
    if base.shape != result.shape:
        raise ValueError("Base and result must have the same canvas dimensions")
    h, w = base.shape[:2]
    # Conservative bridge budget for three images; current upstream limits
    # are dynamic and may be higher. Do not promise unbounded desktop memory.
    if not w or not h or max(w, h) > 30000 or w * h * 3 > 100000000:
        raise ValueError("Canvas exceeds this handoff's three-layer pixel budget")
    destination = Path(destination)
    if destination.suffix != '.comp':
        raise ValueError("Destination must end in .comp")
    destination.mkdir(exist_ok=False)
    try:
        images = destination / 'images'
        images.mkdir()
        layers = []
        def layer(name, pixels, visible=True, blend='Normal', mask=None):
            uid = str(uuid.uuid4()).upper()
            Image.fromarray(pixels).convert('RGBA').save(images / (uid + '.png'))
            record = dict(id=uid, name=name, imageFile=uid + '.png',
                          isVisible=visible, isGroup=False, opacity=1,
                          blendMode=blend, transform=dict(origin=[0, 0],
                          size=[w, h], rotation=0, flipX=False, flipY=False,
                          sampling='High quality'))
            if mask is not None:
                record.update(maskFile=uid + '.mask.png', maskEnabled=True)
                Image.fromarray(mask).save(images / record['maskFile'])
            layers.append(record)
        layer('Base — before manual edits', base)
        coverage = np.any(base != result, axis=2).astype(np.uint8) * 255
        layer('Retouch result — paint mask or adjust opacity', result, mask=coverage)
        active = layers[-1]['id']
        layer('Difference inspection — enable to review changes', base,
              visible=False, blend='Difference')
        manifest = dict(format='com.compositor.project', version=11,
                        colorSpace='sRGB', documentID=str(uuid.uuid4()).upper(),
                        width=w, height=h, resolution=72, activeLayerID=active,
                        layers=layers)
        temporary = destination / '.manifest.json.tmp'
        temporary.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        temporary.rename(destination / 'manifest.json')
    except Exception:
        shutil.rmtree(destination)
        raise
    return destination


def archive_project(project):
    """Package a newly created handoff for download, keeping the .comp folder."""
    project = Path(project)
    target = project.with_suffix('.comp.zip')
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(project.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(project.parent))
    return target

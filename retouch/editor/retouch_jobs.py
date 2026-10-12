"""Captured, revision-tagged recipe jobs for the layer editor.

Workers exchange pixels through private files and never mutate a document.
The UI installs a completed result only on its serialized editor queue.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .composite import composite
from .document import Layer


@dataclass
class RetouchJob:
    directory: Path
    document_id: str
    revision: int
    recipe: str
    strength: float
    status: str = 'queued'
    attached: bool = False
    finished: bool = False
    finalized: bool = False

    @property
    def stop(self):
        return self.directory / 'cancel.flag'

    def cancel(self):
        if self.status in ('queued', 'running') or (self.status == 'completed' and not self.attached):
            self.stop.touch(exist_ok=True)


def prepare_job(history, workspace, recipe, strength):
    from retouch.recipes import RECIPES
    if recipe not in RECIPES:
        raise ValueError('Choose an available recipe')
    if not np.isfinite(strength) or not 0 < strength <= 100:
        raise ValueError('Result strength must be greater than 0 and at most 100')
    document = history.document
    pixels = composite(document)
    if not (pixels[..., 3] == 255).all():
        raise ValueError('Recipe rendering currently requires an opaque composite')
    directory = Path(workspace) / ('recipe-' + uuid.uuid4().hex)
    directory.mkdir()
    Image.fromarray(pixels).save(directory / 'input.png')
    (directory / 'settings.json').write_text(json.dumps({'recipe': recipe}))
    return RetouchJob(directory, document.document_id, history.revision, recipe, float(strength))


def render_worker(directory, engine_factory=None):
    """Run the genuine engine on the child main thread; cancellation gates publish."""
    directory = Path(directory)
    stop = directory / 'cancel.flag'
    def checkpoint():
        if stop.exists():
            raise InterruptedError('Recipe job cancelled')
    checkpoint()
    settings = json.loads((directory / 'settings.json').read_text())
    with Image.open(directory / 'input.png') as image:
        rgba = np.asarray(image.convert('RGBA'))
    if engine_factory is None:
        from retouch.engine import RetouchEngine
        engine_factory = RetouchEngine
    print('Initializing Retouch engine', flush=True)
    engine = engine_factory()
    checkpoint()
    print('Rendering recipe: ' + settings['recipe'], flush=True)
    result = engine.process(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR),
                            recipe=settings['recipe'], quality='full')
    checkpoint()
    if result.dtype != np.uint8 or result.shape != rgba[..., :3].shape:
        raise ValueError('Engine returned unexpected pixel dimensions or precision')
    output = cv2.cvtColor(np.asarray(result), cv2.COLOR_BGR2RGBA)
    temporary = directory / 'result.pending.png'
    Image.fromarray(output).save(temporary)
    checkpoint()
    temporary.replace(directory / 'result.png')
    print('Render completed', flush=True)


def run_job(job, python=None, timeout=900):
    """Yield stage progress. Cancel kills only this job's child after a grace period."""
    if job.status != 'queued':
        raise ValueError('Recipe job has already started or ended')
    if job.stop.exists():
        job.status = 'cancelled'
        job.finished = True
        yield 'Recipe render cancelled before starting.'
        return
    executable = python or os.environ.get('RETOUCH_EDITOR_PYTHON') or (
        None if getattr(sys, 'frozen', False) else sys.executable)
    if not executable:
        job.status = 'failed'
        job.finished = True
        raise ValueError('Configure RETOUCH_EDITOR_PYTHON for recipe rendering')
    environment = os.environ.copy()
    environment.setdefault('RETOUCH_MEDIAPIPE_BACKEND', 'legacy')
    process = None
    started, stopping = time.monotonic(), None
    job.status = 'running'
    try:
        with (job.directory / 'worker.log').open('w') as log:
            process = subprocess.Popen(
                [str(executable), '-u', '-m', 'retouch.editor.retouch_jobs', str(job.directory)],
                cwd=str(Path(__file__).resolve().parents[2]), env=environment,
                stdout=log, stderr=subprocess.STDOUT)
            yield 'Rendering ' + job.recipe + '… You can edit other layers; changed revisions reject this result.'
            while process.poll() is None:
                if time.monotonic() - started > timeout:
                    job.stop.touch(exist_ok=True)
                    job.status = 'timed out'
                if job.stop.exists():
                    if stopping is None:
                        stopping = time.monotonic()
                    elif time.monotonic() - stopping > 2:
                        process.terminate()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                time.sleep(0.25)
                yield f'Recipe render {job.status} · {int(time.monotonic() - started)} seconds'
            if job.stop.exists():
                job.status = 'timed out' if job.status == 'timed out' else 'cancelled'
            elif process.returncode != 0 or not (job.directory / 'result.png').is_file():
                job.status = 'failed'
            else:
                job.status = 'completed'
            yield 'Recipe render ' + job.status + '.'
    finally:
        if process is not None and process.poll() is None:
            job.cancel()
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            job.status = 'cancelled'
        job.finished = True


def attach_result(history, job):
    """Fail closed on cancelled, stale, malformed or already consumed results."""
    if job.status != 'completed' or job.stop.exists() or job.attached:
        raise ValueError('No completed recipe result is available')
    document = history.document
    if (document.document_id, history.revision) != (job.document_id, job.revision):
        job.status = 'stale'
        raise ValueError('Document changed while rendering; rerun the recipe on the current revision')
    with Image.open(job.directory / 'result.png') as image:
        if image.size != (document.width, document.height):
            raise ValueError('Recipe result dimensions do not match the captured document')
        pixels = np.asarray(image.convert('RGBA'))
    if not (pixels[..., 3] == 255).all():
        raise ValueError('Recipe result is not opaque')
    layer = Layer('Retouch · ' + job.recipe, pixels=pixels, opacity=job.strength / 100,
                  mask=np.full((document.height, document.width), 255, np.uint8))
    history.add_layer(layer)
    job.attached = True
    return layer.id


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Private layer-editor recipe worker')
    parser.add_argument('directory', type=Path)
    render_worker(parser.parse_args().directory)

"""Automatic selection: model boundaries, worker cleanup, and failure safety."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from retouch.editor import semantic_selection as sem


class Detector:
    closed = False

    def detect(self, bgr):
        assert bgr[0, 0].tolist() == [30, 20, 10]
        return [SimpleNamespace(landmarks=i, bbox=(0, 0, 8, 6), ied=3) for i in range(2)]

    def segment_person(self, bgr):
        assert bgr[0, 0].tolist() == [30, 20, 10]
        mask = np.zeros(bgr.shape[:2], np.float32)
        mask[1:5, 2:6] = .75
        return mask

    def close(self):
        self.closed = True


class Parser:
    closed = False

    def parse(self, landmarks, bgr, *args, **kwargs):
        mask = np.zeros(bgr.shape[:2], np.float32)
        mask[1:4, 1 + landmarks * 3:3 + landmarks * 3] = .5
        return SimpleNamespace(skin=mask, hair=mask, face_oval=mask, lips=mask,
                               left_eye=mask, right_eye=mask)

    def close(self):
        self.closed = True


def rgb():
    return np.full((6, 8, 3), [10, 20, 30], np.uint8)


@pytest.mark.parametrize('region', ['Skin', 'Hair', 'Face', 'Lips', 'Eyes'])
def test_existing_face_regions_union_all_faces_and_preserve_pixels(region):
    image = rgb()
    mask = sem.infer_mask(image, region, Detector(), Parser())
    assert mask.dtype == np.float32 and mask.shape == (6, 8)
    assert mask[2, 1] == .5 and mask[2, 4] == .5 and mask[0, 0] == 0
    np.testing.assert_array_equal(image, rgb())


def test_person_background_are_complements_without_face_parsing():
    person = sem.infer_mask(rgb(), 'Person', Detector(), None)
    background = sem.infer_mask(rgb(), 'Background', Detector(), None)
    np.testing.assert_array_equal(person + background, np.ones((6, 8), np.float32))


def test_missing_faces_and_invalid_masks_are_not_successes():
    detector = Detector()
    detector.detect = lambda bgr: []
    with pytest.raises(ValueError, match='No faces'):
        sem.infer_mask(rgb(), 'Skin', detector, Parser())
    for mask in (np.full((6, 8), np.nan), np.ones((3, 4)), np.zeros((6, 8)),
                 np.full((6, 8), 255), np.ones((6, 8))):
        detector.segment_person = lambda bgr: mask
        with pytest.raises(ValueError):
            sem.infer_mask(rgb(), 'Person', detector, None)


@pytest.mark.parametrize('region', ['Skin', 'Person'])
def test_worker_closes_models_and_writes_single_channel_mask(tmp_path, region):
    Image.fromarray(rgb()).save(tmp_path / 'input.png')
    (tmp_path / 'settings.json').write_text(json.dumps({'region': region}))
    detector, parser = Detector(), Parser()
    sem.render_worker(tmp_path, lambda: detector, lambda: parser)
    assert detector.closed and parser.closed == (region == 'Skin')
    with Image.open(tmp_path / 'mask.png') as mask:
        assert mask.mode == 'L' and mask.size == (8, 6)


def test_worker_releases_models_on_inference_failure(tmp_path):
    Image.fromarray(rgb()).save(tmp_path / 'input.png')
    (tmp_path / 'settings.json').write_text(json.dumps({'region': 'Skin'}))
    detector, parser = Detector(), Parser()
    detector.detect = lambda bgr: []
    with pytest.raises(ValueError, match='No faces'):
        sem.render_worker(tmp_path, lambda: detector, lambda: parser)
    assert detector.closed and parser.closed
    assert not (tmp_path / 'mask.png').exists()


def fake_process(command, **kwargs):
    directory = Path(command[-1])
    with Image.open(directory / 'input.png') as image:
        assert max(image.size) <= sem.MAX_INFERENCE_SIDE
        mask = np.zeros((image.height, image.width), np.uint8)
        mask[:, :image.width // 2] = 255
    Image.fromarray(mask).save(directory / 'mask.png')
    return SimpleNamespace(returncode=0)


def test_proxy_mask_restores_native_dimensions_and_feathers(monkeypatch, tmp_path):
    monkeypatch.setattr(sem, 'MAX_INFERENCE_SIDE', 8)
    monkeypatch.setattr(sem.subprocess, 'run', fake_process)
    rgba = np.full((12, 16, 4), 255, np.uint8)
    mask = sem.automatic_selection(rgba, 'Skin', 2, str(tmp_path))
    assert mask.shape == (12, 16) and mask.dtype == np.uint8
    assert mask[:, :2].min() == 255 and mask[:, -2:].max() == 0
    assert ((mask > 0) & (mask < 255)).any()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('failure', ['timeout', 'crash', 'no_faces', 'bad_size'])
def test_failed_workers_leave_no_temporary_files(monkeypatch, tmp_path, failure):
    def run(command, **kwargs):
        directory = Path(command[-1])
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(command, 1)
        if failure == 'no_faces':
            (directory / 'error.json').write_text(json.dumps({'error': 'No faces detected.'}))
        if failure == 'bad_size':
            Image.new('L', (1, 1)).save(directory / 'mask.png')
        return SimpleNamespace(returncode=0 if failure == 'bad_size' else 1)
    monkeypatch.setattr(sem.subprocess, 'run', run)
    with pytest.raises(ValueError):
        sem.automatic_selection(np.full((6, 8, 4), 255, np.uint8), 'Skin', 0, str(tmp_path))
    assert not list(tmp_path.iterdir())


def test_transparency_and_invalid_options_fail_before_inference(monkeypatch, tmp_path):
    def run(*args, **kwargs):
        pytest.fail('Invalid requests must not launch a worker')
    monkeypatch.setattr(sem.subprocess, 'run', run)
    rgba = np.full((6, 8, 4), 255, np.uint8)
    for region, feather in [('Unknown', 0), ('Skin', -1), ('Skin', float('nan'))]:
        with pytest.raises(ValueError):
            sem.automatic_selection(rgba, region, feather, str(tmp_path))
    rgba[0, 0, 3] = 0
    with pytest.raises(ValueError, match='opaque'):
        sem.automatic_selection(rgba, 'Skin', 0, str(tmp_path))

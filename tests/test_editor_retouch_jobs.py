"""Recipe capture, color boundaries, cancellation and revision safety."""
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from retouch.editor import Document, DocumentHistory, composite
from retouch.editor.retouch_jobs import prepare_job, render_worker, run_job, attach_result


@pytest.fixture
def history():
    doc = Document(4, 3)
    doc.add_image(np.full((3, 4, 3), [10, 40, 80], np.uint8), 'Original')
    return DocumentHistory(doc, saved=True)


class Engine:
    def process(self, bgr, recipe, quality):
        assert recipe == 'natural' and quality == 'full'
        assert bgr[0, 0].tolist() == [80, 40, 10]
        return np.full_like(bgr, [150, 100, 50])


def completed(history, tmp_path, strength=100):
    job = prepare_job(history, tmp_path, 'natural', strength)
    render_worker(job.directory, Engine)
    job.status = 'completed'
    return job


def test_full_result_reproduces_worker_and_hidden_result_restores_source(history, tmp_path):
    before = composite(history.document)
    job = completed(history, tmp_path)
    lid = attach_result(history, job)
    after = composite(history.document)
    assert after[0, 0].tolist() == [50, 100, 150, 255]
    assert history.document.layer(lid).mask.min() == 255
    history.update_layer(lid, visible=False)
    np.testing.assert_array_equal(composite(history.document), before)
    history.undo()
    history.undo()
    np.testing.assert_array_equal(composite(history.document), before)
    assert not history.dirty
    with pytest.raises(ValueError):
        attach_result(history, job)


def test_half_strength_uses_layer_opacity(history, tmp_path):
    job = completed(history, tmp_path, strength=50)
    attach_result(history, job)
    assert history.document.layers[-1].opacity == 0.5
    assert composite(history.document)[0, 0].tolist() == [30, 70, 115, 255]


def test_changed_revision_rejects_result_without_overwriting_manual_edits(history, tmp_path):
    job = completed(history, tmp_path)
    history.update_layer(history.document.layers[0].id, name='Manual edit')
    revision = history.revision
    with pytest.raises(ValueError, match='changed'):
        attach_result(history, job)
    assert job.status == 'stale' and history.revision == revision
    assert len(history.document.layers) == 1
    assert history.document.layers[0].name == 'Manual edit'


def test_different_document_same_revision_rejects_result(history, tmp_path):
    job = completed(history, tmp_path)
    other = DocumentHistory(Document(4, 3), saved=True)
    with pytest.raises(ValueError, match='changed'):
        attach_result(other, job)


def test_worker_cancellation_never_publishes_result(history, tmp_path):
    job = prepare_job(history, tmp_path, 'natural', 100)
    class CancellingEngine(Engine):
        def process(self, *args, **kwargs):
            job.stop.touch()
            return super().process(*args, **kwargs)
    with pytest.raises(InterruptedError):
        render_worker(job.directory, CancellingEngine)
    assert not (job.directory / 'result.png').exists()


def test_queued_cancel_does_not_start_child(history, tmp_path, monkeypatch):
    job = prepare_job(history, tmp_path, 'natural', 100)
    job.cancel()
    def forbidden(*args, **kwargs):
        raise AssertionError('Child must not start')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    assert list(run_job(job)) == ['Recipe render cancelled before starting.']
    assert job.finished and job.status == 'cancelled'


def test_completed_but_cancelled_result_is_not_attached(history, tmp_path):
    job = completed(history, tmp_path)
    job.stop.touch()
    with pytest.raises(ValueError):
        attach_result(history, job)
    assert len(history.document.layers) == 1


def test_job_runner_observes_real_child_exit_and_result(history, tmp_path, monkeypatch):
    job = prepare_job(history, tmp_path, 'natural', 100)
    original = subprocess.Popen
    script = 'from PIL import Image; import sys; Image.new("RGBA", (4,3), (50,100,150,255)).save(sys.argv[1])'
    def spawn(command, **kwargs):
        assert '-m' in command and command[-1] == str(job.directory)
        return original([sys.executable, '-c', script, str(job.directory / 'result.png')], **kwargs)
    monkeypatch.setattr(subprocess, 'Popen', spawn)
    events = list(run_job(job))
    assert events[-1] == 'Recipe render completed.' and job.finished
    attach_result(history, job)
    assert len(history.document.layers) == 2


def test_cancellation_stops_running_child(history, tmp_path, monkeypatch):
    job = prepare_job(history, tmp_path, 'natural', 100)
    original = subprocess.Popen
    processes = []
    def spawn(command, **kwargs):
        child = original([sys.executable, '-c', 'import time; time.sleep(30)'], **kwargs)
        processes.append(child)
        return child
    monkeypatch.setattr(subprocess, 'Popen', spawn)
    iterator = run_job(job)
    next(iterator)
    job.cancel()
    list(iterator)
    assert processes[0].poll() is not None
    assert job.finished and job.status == 'cancelled'


@pytest.mark.parametrize('recipe, strength', [('missing', 100), ('natural', 0), ('natural', float('nan'))])
def test_invalid_settings_fail_before_capture(history, tmp_path, recipe, strength):
    with pytest.raises(ValueError):
        prepare_job(history, tmp_path, recipe, strength)
    assert not list(tmp_path.iterdir())


def test_transparent_composite_is_explicitly_refused(tmp_path):
    with pytest.raises(ValueError, match='opaque'):
        prepare_job(DocumentHistory(Document(4, 3)), tmp_path, 'natural', 100)


def test_malformed_result_preserves_document(history, tmp_path):
    job = completed(history, tmp_path)
    Image.new('RGB', (1, 1)).save(job.directory / 'result.png')
    with pytest.raises(ValueError, match='dimensions'):
        attach_result(history, job)
    assert not history.dirty

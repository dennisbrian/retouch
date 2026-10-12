"""GUI video capture, isolation, progress, cancellation and verified downloads."""
from pathlib import Path
import json
import sys
import zipfile

import pytest
from retouch.gui_video import VideoJobs


def source_file(tmp_path):
    path=tmp_path/'source with spaces.mp4';path.write_bytes(b'fixture')
    return path


def child_script(tmp_path,mode='success'):
    path=tmp_path/f'{mode}.py'
    path.write_text('''import sys,time,json
from pathlib import Path
out=Path(sys.argv[1]);stop=Path(sys.argv[2]);mode=sys.argv[3]
qa=Path(str(out)+'.qa');qa.mkdir()
print('rendering: 1/5 frames (0.1s)',flush=True)
if mode=='cancel':
 for i in range(100):
  if stop.exists():
   (qa/'manifest.json').write_text(json.dumps({'status':'cancelled'}))
   raise SystemExit(130)
  time.sleep(.02)
if mode=='failed':
 (qa/'manifest.json').write_text(json.dumps({'status':'failed','error':'planted decoder failure'}))
 raise SystemExit(2)
if mode!='false-complete':out.write_bytes(b'video artifact')
(qa/'manifest.json').write_text(json.dumps({'status':'completed','frames_written':5}))
(qa/'metrics.json').write_text('{}')
(qa/'contact-sheet.jpg').write_bytes(b'contact')
(qa/'null.mp4').write_bytes(b'large null video')
print('completed: 5/5 frames (0.2s)',flush=True)
''')
    return path


def patch_command(manager,owner,token,script,mode):
    job=manager._get(owner,token)
    job.command=[sys.executable,'-u',str(script),str(job.output),str(job.stop),mode]
    return job


def test_inputs_are_captured_as_args_without_shell(tmp_path):
    manager=VideoJobs(tmp_path/'work')
    path=source_file(tmp_path)
    token=manager.prepare('A',path,smooth=23,whiten=4,qa=False)
    command=manager._get('A',token).command
    assert str(path) in command and '--no-qa' in command
    assert command[command.index('--smooth')+1]=='23.0'
    with pytest.raises(ValueError,match='active'):manager.prepare('A',path)
    with pytest.raises(ValueError,match='current session'):manager.cancel('B',token)


def test_queued_cancel_never_starts_worker(tmp_path,monkeypatch):
    manager=VideoJobs(tmp_path/'work');token=manager.prepare('A',source_file(tmp_path))
    manager.cancel('A',token)
    monkeypatch.setattr('retouch.gui_video.subprocess.Popen',lambda *a,**kw:pytest.fail('queued cancellation started worker'))
    events=list(manager.run('A',token))
    assert events[-1]['done'] and 'Cancelled' in events[-1]['status'] and events[-1]['video'] is None


def test_success_publishes_only_verified_media_and_small_qa_artifacts(tmp_path):
    manager=VideoJobs(tmp_path/'work');token=manager.prepare('A',source_file(tmp_path))
    job=patch_command(manager,'A',token,child_script(tmp_path),'success')
    events=list(manager.run('A',token))
    assert any('rendering:' in e['status'] for e in events)
    last=events[-1]
    assert last['done'] and last['video']==str(job.output) and job.status=='completed'
    with zipfile.ZipFile(last['report']) as archive:
        assert 'manifest.json' in archive.namelist() and 'null.mp4' not in archive.namelist()
    manager.unload('A')
    assert not job.directory.exists() and not job.workspace.session_dir.exists()


@pytest.mark.parametrize('mode',['failed','false-complete'])
def test_failure_or_missing_media_cannot_show_completion(tmp_path,mode):
    manager=VideoJobs(tmp_path/'work');token=manager.prepare('A',source_file(tmp_path))
    patch_command(manager,'A',token,child_script(tmp_path,mode),mode)
    last=list(manager.run('A',token))[-1]
    assert last['done'] and last['video'] is None and 'failed' in last['status']


def test_running_cancellation_is_independent_of_worker_queue(tmp_path):
    manager=VideoJobs(tmp_path/'work');token=manager.prepare('A',source_file(tmp_path))
    job=patch_command(manager,'A',token,child_script(tmp_path,'cancel'),'cancel')
    iterator=manager.run('A',token);next(iterator)
    assert job.status=='running'
    assert 'Stopping' in manager.cancel('A',token)
    last=list(iterator)[-1]
    assert last['done'] and 'Cancelled' in last['status'] and not job.output.exists()


def test_unload_does_not_delete_live_worker_directory(tmp_path):
    manager=VideoJobs(tmp_path/'work');token=manager.prepare('A',source_file(tmp_path))
    job=patch_command(manager,'A',token,child_script(tmp_path,'cancel'),'cancel')
    iterator=manager.run('A',token);next(iterator)
    manager.unload('A')
    assert job.directory.exists()
    list(iterator)
    assert not job.directory.exists() and token not in manager.jobs


def test_unload_one_session_preserves_the_other(tmp_path):
    manager=VideoJobs(tmp_path/'work');source=source_file(tmp_path)
    a=manager.prepare('A',source);b=manager.prepare('B',source)
    bdir=manager._get('B',b).directory
    manager.unload('A')
    assert bdir.exists() and manager._get('B',b).status=='queued'


def test_gui_video_callbacks_have_session_injection_and_nonqueued_cancel():
    import gui
    deps={x.get('api_name'):x for x in gui.app.config['dependencies']}
    assert deps['prepare_video_job']['queue'] is False
    assert deps['cancel_video_job']['queue'] is False
    assert deps['run_video_job']['trigger_only_on_success']
    assert len(deps['prepare_video_job']['inputs'])==5
    assert gui.prepare_video_job.__annotations__['request']==gui.gr.Request
    assert any(c.get('props',{}).get('label')=='Video Retouch' for c in gui.app.config['components'])

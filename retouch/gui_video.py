"""Session-owned subprocess jobs for the GUI video tab.

The child executes the CLI on its main thread. Parent state contains only job
IDs/paths; cancellation remains available while a Gradio generator is queued.
No shell is used, and one session cannot cancel/read another session's job.
"""
from __future__ import annotations

import json
import math
import os
import queue
import secrets
import subprocess
import sys
import threading
import zipfile
from dataclasses import dataclass,field
from pathlib import Path

from .gui_workspace import SessionWorkspace,default_workspace_root


@dataclass
class VideoJob:
    """One browser session's immutable inputs and server-owned run state."""
    token:str
    owner:str
    workspace:SessionWorkspace
    directory:Path
    command:list
    stop:Path
    output:Path
    status:str='queued'
    process:object=None
    cleanup_requested:bool=False
    lock:object=field(default_factory=threading.RLock,repr=False)


class VideoJobs:
    """Own jobs by session, using compact opaque tokens in browser state."""
    def __init__(self,root=None,python=None):
        self.root=Path(root) if root else default_workspace_root()/'video'
        self.python=python
        self.jobs={}
        self.active={}
        self.lock=threading.RLock()

    @staticmethod
    def _owner(owner):
        if not owner:raise ValueError('Session unavailable. Reload the page before exporting.')
        return str(owner)

    def prepare(self,owner,source,stable=None,smooth=20,whiten=0,qa=True):
        """Reserve a job before it enters the queue, so queued jobs can cancel."""
        owner=self._owner(owner)
        source=Path(source).resolve() if source else None
        if source is None or not source.is_file():raise ValueError('Choose a video file before exporting.')
        stable=Path(stable).resolve() if stable else None
        if stable is not None and not stable.is_file():raise ValueError('The reviewed tracking file is unavailable.')
        for value in (smooth,whiten):
            if not math.isfinite(float(value)) or not 0<=float(value)<=100:raise ValueError('Video strengths must be in 0–100.')
        python=self.python or os.environ.get('RETOUCH_VIDEO_PYTHON') or (None if getattr(sys,'frozen',False) else sys.executable)
        if not python:raise ValueError('Video export needs a Python runtime with the video extra. Configure RETOUCH_VIDEO_PYTHON.')
        with self.lock:
            previous=self.jobs.get(self.active.get(owner))
            if previous is not None and previous.status in ('queued','running'):
                raise ValueError('A video job is already active in this session. Cancel it or wait for it to finish.')
            workspace=SessionWorkspace(owner,root=self.root)
            directory=workspace.request_workspace('video')
            output=directory/'retouched.mp4';stop=directory/'cancel.flag'
            command=[str(python),'-u','-m','retouch.video.export',str(source),'--out',str(output),
                     '--smooth',str(float(smooth)),'--whiten',str(float(whiten)),'--cancel-file',str(stop)]
            if stable is not None:command+=['--stable',str(stable)]
            if not qa:command+=['--no-qa']
            token=secrets.token_urlsafe(24)
            job=VideoJob(token,owner,workspace,directory,command,stop,output)
            self.jobs[token]=job;self.active[owner]=token
            return token

    def _get(self,owner,token):
        owner=self._owner(owner)
        with self.lock:job=self.jobs.get(token)
        if job is None or job.owner!=owner:raise ValueError('This video job is unavailable in the current session.')
        return job

    def cancel(self,owner,token):
        """Signal only this session's queued/running job; do not kill other jobs."""
        job=self._get(owner,token)
        with job.lock:
            if job.status not in ('queued','running'):return f'Video job is {job.status}.'
            job.stop.touch(exist_ok=True)
        return 'Stopping after the current frame. No partial delivery video will be published.'

    def unload(self,owner):
        """Stop a leaving session and defer deleting live child workspaces."""
        if not owner:return
        with self.lock:jobs=[job for job in self.jobs.values() if job.owner==str(owner)]
        for job in jobs:
            with job.lock:
                job.cleanup_requested=True
                if job.status in ('queued','running'):
                    job.stop.touch(exist_ok=True)
                if job.status!='running':
                    job.status='cancelled' if job.status=='queued' else job.status
                    job.workspace.cleanup(request_dir=job.directory)
                    self._drop(job)

    def _drop(self,job):
        """Forget a cleaned request and remove its empty owned session root."""
        with self.lock:
            self.jobs.pop(job.token,None)
            if self.active.get(job.owner)==job.token:self.active.pop(job.owner,None)
            if not any(item.owner==job.owner for item in self.jobs.values()):
                job.workspace.cleanup()

    @staticmethod
    def _event(status,video=None,report=None,contact=None,done=False):
        return {'status':status,'video':str(video) if video else None,'report':str(report) if report else None,
                'contact':str(contact) if contact else None,'done':done}

    def run(self,owner,token):
        """Stream bounded progress from the isolated exporter and verify completion."""
        job=self._get(owner,token)
        process=None
        report_path=None
        with job.lock:
            if job.status!='queued':raise ValueError('This job has already started or ended.')
            stopped=job.stop.exists()
            job.status='cancelled' if stopped else 'running'
        if stopped:
            yield self._event('Cancelled before export started.',done=True)
            return
        messages=queue.Queue(maxsize=128)
        lines=[]
        try:
            root=Path(__file__).resolve().parents[1]
            environment=os.environ.copy()
            environment['PYTHONUNBUFFERED']='1'
            process=subprocess.Popen(job.command,cwd=str(root),env=environment,stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,text=True,bufsize=1)
            with job.lock:job.process=process
            child=process
            def read_output():
                try:
                    for line in child.stdout:
                        try:messages.put(line.rstrip(),timeout=.2)
                        except queue.Full:pass  # bounded diagnostic log; exporter must never block on UI
                finally:
                    child.stdout.close()
            reader=threading.Thread(target=read_output,daemon=True);reader.start()
            yield self._event('Starting video export…')
            while process.poll() is None or not messages.empty() or reader.is_alive():
                try:line=messages.get(timeout=.25)
                except queue.Empty:continue
                lines.append(line[-2000:]);lines=lines[-40:]
                if line.startswith(('tracking:','stabilizing:','rendering:','verifying:')):
                    text='Stopping after the current frame…' if job.stop.exists() else line
                    yield self._event(text)
            code=process.wait()
            manifest_path=Path(str(job.output)+'.qa')/'manifest.json'
            manifest=json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
            if code==0 and manifest.get('status')=='completed' and job.output.is_file():
                qa_dir=manifest_path.parent
                report_path=job.directory/'qa-report.zip'
                report_error=None
                try:
                    with zipfile.ZipFile(report_path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                        for path in sorted(qa_dir.iterdir()):
                            if path.is_file() and path.suffix.lower() in ('.json','.jpg','.png'):
                                archive.write(path,path.name)
                except OSError as exc:
                    report_error=str(exc);report_path=None
                job.status='completed'
                count=manifest.get('frames_written',0)
                text=f'Completed — {Path(job.command[4]).name}: {count} frames exported. Review the video, mask edges and motion before batch use.'
                if manifest.get('encoding',{}).get('audio_mode')=='pcm16_to_alac_lossless':
                    text+=' Audio is lossless ALAC; download for review in a compatible editor/player.'
                if report_error:text+=' QA download could not be packaged: '+report_error
                contact=qa_dir/'contact-sheet.jpg'
                yield self._event(text,job.output,report_path,contact if contact.is_file() else None,True)
            elif code==130 or manifest.get('status')=='cancelled':
                job.status='cancelled'
                yield self._event('Cancelled. No partial delivery video was published.',done=True)
            else:
                job.status='failed'
                error=manifest.get('error') or next((line for line in reversed(lines) if line.strip()),f'Exporter exited with code {code}')
                detail=str(error)[:1500]
                yield self._event(detail if detail.startswith('Video export failed:') else 'Video export failed: '+detail,done=True)
        except GeneratorExit:
            self.cancel(owner,token)
            # The exporter exits between frames; close the client stream without
            # deleting files underneath its worker. A reaper completes cleanup.
            if process is not None:
                threading.Thread(target=self._reap,args=(job,process),daemon=True).start()
                process=None
            raise
        except Exception as exc:
            job.status='failed'
            if process is not None and process.poll() is None:
                job.stop.touch(exist_ok=True)
                threading.Thread(target=self._reap,args=(job,process),daemon=True).start()
                process=None
            yield self._event('Video export failed: '+str(exc)[:1500],done=True)
        finally:
            if process is not None:
                job.process=None
                if job.cleanup_requested:
                    job.workspace.cleanup(request_dir=job.directory)
                    self._drop(job)

    def _reap(self,job,process):
        """Finish a disconnected worker before cleaning its owned request files."""
        process.wait()
        with job.lock:
            job.process=None
            if job.status=='running':job.status='cancelled'
            if job.cleanup_requested:
                job.workspace.cleanup(request_dir=job.directory)
                self._drop(job)

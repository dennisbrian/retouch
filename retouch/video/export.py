"""Offline selected-face export with streaming pixels, audio passthrough and QA.

Usage: python -m retouch.video.export clip.mp4 --out retouched.mp4
       [--stable stable.json] [--smooth 20] [--whiten 0] [--cancel-file stop]

Ctrl+C/cancellation never publishes a partial video. Existing source, output,
and QA directories are never overwritten. Measurements do not certify footage.
"""
from __future__ import annotations

import argparse
import bisect
import dataclasses
import hashlib
import json
import logging
import math
import os
import time
from contextlib import ExitStack, closing
from pathlib import Path

import numpy as np

from .adapter import RenderParams, SelectedFaceRenderer
from .audio import DeliveryWriter
from .media import VideoWriter, probe, read_frames
import av
from .qa import temporal_report, verify_audio, verify_frames, _audio
from .stabilize import (REGION_NAMES, STABLE_FORMAT, STABLE_VERSION,
                        StableFrame, StableTrack, build_stable_contract, gather_evidence, stabilize)
from .tracker import track_video, build_contract


class VideoExportCancelled(Exception):
    """A requested stop, not a successful export."""


def load_stable_track(path):
    """Read S3 evidence and validate indices, geometry, cuts and visibility."""
    data = json.loads(Path(path).read_text())
    if data.get('format') != STABLE_FORMAT or data.get('version') != STABLE_VERSION:
        raise ValueError('expected a version-1 retouch_stable_track JSON')
    def integer(value, name):
        if not isinstance(value,int) or isinstance(value,bool): raise ValueError(f'{name} must be an integer')
        return value
    w,h,n = [integer(data[k],k) for k in ('width','height','frame_count')]
    fps = float(data['fps'])
    if min(w,h)<=0 or n<0 or not math.isfinite(fps) or fps<=0:
        raise ValueError('invalid stable-track dimensions, frame count or fps')
    cuts = data['cuts']
    if any(integer(c,'cut')<=0 or c>=n for c in cuts) or cuts!=sorted(set(cuts)):
        raise ValueError('cuts must be sorted unique frame indices inside the clip')
    records,seen,last_time = [],set(),None
    for item in data['frames']:
        i,shot = integer(item['frame'],'frame'),integer(item['shot'],'shot')
        t,weight = float(item['time']),float(item['weight'])
        lm = np.asarray(item['landmarks'],np.float32)
        box = tuple(integer(v,'face box') for v in item['face'])
        visibility = {key:float(value) for key,value in item['visibility'].items()}
        if i<0 or i>=n or i in seen or (records and i<=records[-1].frame):
            raise ValueError('stable frames must be sorted unique indices inside the clip')
        if not math.isfinite(t) or (last_time is not None and t<=last_time) or not math.isfinite(weight) or not 0<=weight<=1:
            raise ValueError('invalid stable timestamp or weight')
        if shot!=bisect.bisect_right(cuts,i) or item['source'] not in ('tracked','interpolated'):
            raise ValueError('stable shot/source does not match the cut contract')
        if lm.shape!=(478,3) or not np.isfinite(lm).all(): raise ValueError('expected finite 478 x 3 landmarks')
        if len(box)!=4 or min(box[2:])<=0 or min(box[:2])<0 or box[0]+box[2]>w or box[1]+box[3]>h:
            raise ValueError('stable face box is outside the clip')
        if set(visibility)!=set(REGION_NAMES) or any(not math.isfinite(v) or not 0<=v<=1 for v in visibility.values()):
            raise ValueError('expected finite per-region visibility in 0..1')
        records.append(StableFrame(i,t,shot,item['source'],weight,box,lm,visibility))
        seen.add(i);last_time=t
    return StableTrack(w,h,fps,n,cuts,records)


def _hash(path, check_cancel):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        while True:
            check_cancel()
            chunk=handle.read(4*1024*1024)
            if not chunk: break
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path,data):
    path=Path(path)
    pending=path.with_name(path.name+'.tmp')
    pending.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    pending.replace(path)


def _source_color_contract(source):
    """Fail before decode if the initial 8-bit SDR path would reinterpret color."""
    with av.open(str(source)) as container:
        context=container.streams.video[0].codec_context
        trc,primaries,space=int(context.color_trc),int(context.color_primaries),int(context.colorspace)
        bits=max((component.bits for component in context.format.components),default=8) if context.format else 8
        # FFmpeg AVColor* values: https://ffmpeg.org/doxygen/trunk/pixfmt_8h.html
        if bits>8 or trc not in (0,1,2,4,5,6,7,13) or primaries in (9,10,11,12) or primaries>=256 or space not in (0,1,2,4,5,6,7):
            raise ValueError('initial video export supports 8-bit SDR input; convert HDR/log/wide-gamut/high-bit-depth footage explicitly first')
        return {'transfer_id':trc,'primaries_id':primaries,'matrix_id':space,'component_bits':bits,
                'interpretation':'untagged_assumed_sdr' if trc in (0,2) else 'tagged_sdr'}


def export_video(source, output, *, stable_path=None, params=RenderParams(), crf=18,
                 lossless=False, qa=True, progress=None, cancelled=None, engine_factory=None):
    """Render/verify all frames before atomically publishing a new output.

    A newly-owned <output>.qa directory retains tracks, a manifest and failure
    status. Cancellation checks run between frames/passes. No full clip of
    pixels is retained. Supplied stable tracks are bound by dimensions, exact
    decoded frame count and per-tracked-frame presentation timestamps.
    """
    source,output=Path(source).resolve(),Path(output).resolve()
    if source==output or (stable_path and Path(stable_path).resolve()==output):
        raise ValueError('output must be distinct from video and input tracks')
    if not isinstance(crf,int) or not 0<=crf<=51: raise ValueError('crf must be in 0..51')
    if output.suffix.lower()!=('.mkv' if lossless else '.mp4'):
        raise ValueError('use .mp4 for H.264, or --lossless with .mkv')
    directory=Path(str(output)+'.qa')
    if output.exists() or directory.exists(): raise FileExistsError('choose a new output path; existing outputs/QA are preserved')
    def check_cancel():
        if cancelled is not None and cancelled(): raise VideoExportCancelled('video export cancelled')
    check_cancel()
    source_digest=_hash(source,check_cancel)
    info=probe(source)
    color_contract=_source_color_contract(source)
    if not lossless and (info.width%2 or info.height%2): raise ValueError('odd video dimensions require --lossless .mkv')
    _audio(source,check_cancel)  # refuse multiple audio streams before rendering
    output.parent.mkdir(parents=True,exist_ok=True)
    directory.mkdir()  # exclusively own this run's artifacts
    manifest_path=directory/'manifest.json'
    partial=directory/('render.partial.mkv' if lossless else 'render.partial.mp4')
    null=directory/('null.mkv' if lossless else 'null.mp4')
    started=time.perf_counter()
    manifest={'status':'running','source':str(source),'source_sha256':source_digest,'output':str(output),
              'params':dataclasses.asdict(params),'encoding':{'lossless':lossless,'crf':crf},
              'source_color':color_contract,'frames_written':0,'qualification':'not_certified','qa_enabled':bool(qa)}
    _write_json(manifest_path,manifest)
    def report(phase,done,total=None):
        check_cancel()
        manifest['phase']=phase
        if progress is not None: progress({'phase':phase,'frames':done,'total':total,'elapsed_s':time.perf_counter()-started})
    try:
        if stable_path:
            manifest['input_stable_sha256']=_hash(stable_path,check_cancel)
            stable=load_stable_track(stable_path)
            if _hash(stable_path,check_cancel)!=manifest['input_stable_sha256']:
                raise ValueError('stable track changed while being loaded')
        elif params.smooth==0 and params.whiten==0:
            stable=StableTrack(info.width,info.height,float(info.fps),info.frame_count,[],[])
        else:
            tracks=track_video(source,progress=lambda count,selected:report('tracking',count,info.frame_count or None))
            contract=build_contract(tracks,source.name)
            _write_json(directory/'tracks.json',contract)
            by={face.frame:face.landmarks for face in tracks.faces}
            with closing(read_frames(source)) as frames:
                evidence=gather_evidence(frames,by,progress=lambda count:report('stabilizing',count,tracks.frame_count))
            stable=stabilize(contract,evidence)
        if (stable.width,stable.height)!=(info.width,info.height): raise ValueError('stable dimensions do not match decoded video')
        if not stable.frames and (params.smooth>0 or params.whiten>0):
            raise ValueError('no selected face found; supply a reviewed stable track or use identity controls')
        by={frame.frame:frame for frame in stable.frames}
        with ExitStack() as stack:
            active=params.smooth>0 or params.whiten>0
            if active:
                if engine_factory is None:
                    from ..engine import RetouchEngine
                    engine_factory=RetouchEngine
                engine=stack.enter_context(engine_factory())
            else: engine=None
            adapter=SelectedFaceRenderer(engine,params)
            writer=stack.enter_context(DeliveryWriter(partial,source,lossless=lossless,crf=crf))
            manifest["encoding"]["audio_mode"]=writer.audio_mode
            baseline=stack.enter_context(VideoWriter(null,source,lossless=lossless,crf=crf,audio=False)) if qa else None
            frames=stack.enter_context(closing(read_frames(source)))
            for frame in frames:
                check_cancel()
                record=by.get(frame.index)
                if record is not None and abs(record.time-frame.time)>1e-4:
                    raise ValueError(f'frame {frame.index}: stable timestamp does not match source')
                result=adapter.render(frame.image,record)
                check_cancel()
                writer.write(frame,result)
                if baseline is not None: baseline.write(frame,frame.image)
                manifest['frames_written']+=1
                report('rendering',manifest['frames_written'],stable.frame_count or None)
        count=manifest['frames_written']
        if count==0 or ((stable_path or stable.frame_count) and count!=stable.frame_count):
            raise ValueError('source frame count does not match the stable track')
        if not stable.frame_count: stable=dataclasses.replace(stable,frame_count=count)
        _write_json(directory/'stable.json',build_stable_contract(stable,source.name))
        manifest['adapter']={'parsed_frames':adapter.parsed_frames,'warped_frames':adapter.warped_frames,
                             'face_render_calls':adapter.edited_frames,'untracked_frames':count-len(stable.frames)}
        report('verifying',0,count)
        manifest['audio']=verify_audio(source,partial,check_cancel=check_cancel)
        if qa:
            manifest['temporal_qa']=temporal_report(source,partial,null,stable,check_cancel=check_cancel,samples_dir=directory)
            _write_json(directory/'metrics.json',manifest['temporal_qa'])
        else:
            manifest['frame_timing']=verify_frames(source,partial,count,check_cancel=check_cancel)
            manifest['temporal_qa']={'status':'not_run','human_review_required':True}
        if _hash(source,check_cancel)!=source_digest: raise ValueError('source file changed during export')
        if stable_path and _hash(stable_path,check_cancel)!=manifest['input_stable_sha256']:
            raise ValueError('stable track changed during export')
        manifest['output_sha256']=_hash(partial,check_cancel)
        check_cancel()
        os.link(partial,output)  # atomic, exclusive; never replace an existing output
        manifest.update(status='completed',phase='completed',elapsed_s=time.perf_counter()-started)
        _write_json(manifest_path,manifest)
        if progress is not None:
            try: progress({'phase':'completed','frames':count,'total':count,'elapsed_s':manifest['elapsed_s']})
            except Exception as exc: logging.getLogger(__name__).warning('Export completed; progress callback failed: %s',exc)
        return manifest
    except BaseException as exc:
        if output.exists() and partial.exists() and os.path.samefile(output,partial): output.unlink()
        manifest.update(status='cancelled' if isinstance(exc,(VideoExportCancelled,KeyboardInterrupt)) else 'failed',
                        error=f'{type(exc).__name__}: {exc}',elapsed_s=time.perf_counter()-started)
        _write_json(manifest_path,manifest)
        raise
    finally:
        if partial.exists(): partial.unlink()


def main(argv=None):
    """Run the standalone CLI; report cancellation separately from success."""
    parser=argparse.ArgumentParser(description='Retouch one stabilized face and stream video/audio to a new output.')
    parser.add_argument('video',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stable',type=Path,help='Reviewed stable.json; otherwise track/stabilize automatically')
    parser.add_argument('--smooth',type=float,default=20)
    parser.add_argument('--whiten',type=float,default=0)
    parser.add_argument('--parse-every',type=int,default=3)
    parser.add_argument('--max-crop-dim',type=int,default=1024)
    parser.add_argument('--crf',type=int,default=18)
    parser.add_argument('--lossless',action='store_true',help='FFV1/MKV for QA; preserves pixels without codec loss')
    parser.add_argument('--no-qa',action='store_true',help='Skip same-codec null/temporal measurements; timing/audio still verified')
    parser.add_argument('--cancel-file',type=Path,help='Create this file to stop the job between frames')
    args=parser.parse_args(argv)
    def progress(event):
        if event['frames']%10==0 or event['frames']==1 or event['phase'] in ('verifying','completed'):
            print(f"{event['phase']}: {event['frames']}/{event['total'] or '?'} frames ({event['elapsed_s']:.1f}s)",flush=True)
    try:
        params=RenderParams(args.smooth,args.whiten,args.parse_every,args.max_crop_dim)
        result=export_video(args.video,args.out,stable_path=args.stable,params=params,crf=args.crf,
                            lossless=args.lossless,qa=not args.no_qa,progress=progress,
                            cancelled=lambda:args.cancel_file is not None and args.cancel_file.exists())
    except (VideoExportCancelled,KeyboardInterrupt):
        print('Cancelled; no partial video published.',flush=True)
        return 130
    except (ValueError,OSError) as exc:
        parser.exit(2,f'Video export failed: {exc}\n')
    print(f"wrote {args.out}: {result['frames_written']} frames; review {args.out}.qa/manifest.json",flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())

"""Export publication, cancellation, audio payloads, VFR and QA contracts."""
import json
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('av')

from retouch.video.adapter import RenderParams
from retouch.video.export import export_video,VideoExportCancelled,load_stable_track
from retouch.video.media import read_frames,probe
from retouch.video.stabilize import StableTrack,build_stable_contract
from tests.video.test_adapter import FakeEngine,stable_frame
from tests.video.test_media import _make_clip,CFR,VFR,_audio_packets


def track_for(source,path):
    info=probe(source);frames=list(read_frames(source))
    records=[]
    import dataclasses
    for frame in frames:
        record=stable_frame(frame.index,width=info.width,height=info.height)
        records.append(dataclasses.replace(record,time=frame.time))
    stable=StableTrack(info.width,info.height,float(info.fps),len(frames),[],records)
    path.write_text(json.dumps(build_stable_contract(stable,source.name)))
    return path


@pytest.mark.parametrize('lossless',[False,True])
def test_complete_export_keeps_all_frames_audio_and_publishes_manifest(tmp_path,lossless):
    source=_make_clip(tmp_path/'source.mp4',CFR[:6],audio_seconds=10240/48000)
    stable=track_for(source,tmp_path/'stable.json')
    output=tmp_path/('result.mkv' if lossless else 'result.mp4')
    events=[]
    manifest=export_video(source,output,stable_path=stable,params=RenderParams(smooth=20),
                          lossless=lossless,engine_factory=FakeEngine,progress=events.append)
    assert manifest['status']=='completed' and manifest['frames_written']==6
    assert len(list(read_frames(output)))==6 and probe(output).has_audio
    assert [p for p,t in _audio_packets(source)]==[p for p,t in _audio_packets(output)]
    assert manifest['audio']['status']=='packet_identical'
    assert manifest['temporal_qa']['frames_compared']==6
    assert manifest['qualification']=='not_certified'
    assert events[-1]['phase']=='completed'
    assert not (Path(str(output)+'.qa')/'render.partial.mp4').exists()
    assert (Path(str(output)+'.qa')/'contact-sheet.jpg').exists()


def test_variable_rate_identity_export_keeps_pts(tmp_path):
    source=_make_clip(tmp_path/'vfr.mp4',VFR[:8]+VFR[-4:])
    output=tmp_path/'out.mp4'
    manifest=export_video(source,output,params=RenderParams(smooth=0,whiten=0),qa=False)
    a,b=list(read_frames(source)),list(read_frames(output))
    np.testing.assert_allclose([f.time for f in a],[f.time for f in b],atol=1e-5)
    assert manifest['temporal_qa']['status']=='not_run'


@pytest.mark.parametrize('during',['rendering','verifying'])
def test_cancellation_never_publishes_partial_or_completed_status(tmp_path,during):
    source=_make_clip(tmp_path/'source.mp4',CFR[:6])
    output=tmp_path/'result.mp4';stop=[False];events=[]
    def progress(event):
        events.append(event)
        if event['phase']==during and (during!='rendering' or event['frames']==2):stop[0]=True
    with pytest.raises(VideoExportCancelled):
        export_video(source,output,params=RenderParams(smooth=0,whiten=0),progress=progress,cancelled=lambda:stop[0])
    assert not output.exists() and not any(e['phase']=='completed' for e in events)
    report=json.loads((Path(str(output)+'.qa')/'manifest.json').read_text())
    assert report['status']=='cancelled'
    assert not list(Path(str(output)+'.qa').glob('render.partial.*'))


def test_engine_failure_records_failure_without_publishing(tmp_path):
    source=_make_clip(tmp_path/'source.mp4',CFR[:4]);stable=track_for(source,tmp_path/'stable.json')
    output=tmp_path/'result.mp4'
    class Broken(FakeEngine):
        def process(self,*a,**kw):raise RuntimeError('planted render failure')
    with pytest.raises(RuntimeError,match='planted'):
        export_video(source,output,stable_path=stable,engine_factory=Broken)
    assert not output.exists()
    report=json.loads((Path(str(output)+'.qa')/'manifest.json').read_text())
    assert report['status']=='failed'


def test_existing_outputs_and_source_are_preserved(tmp_path):
    source=_make_clip(tmp_path/'source.mp4',CFR[:2]);before=source.read_bytes()
    with pytest.raises(ValueError):export_video(source,source)
    output=tmp_path/'output.mp4';output.write_bytes(b'keep me')
    with pytest.raises(FileExistsError):export_video(source,output)
    assert output.read_bytes()==b'keep me' and source.read_bytes()==before


def test_input_track_mismatch_is_not_silently_reused(tmp_path):
    source=_make_clip(tmp_path/'source.mp4',CFR[:4]);stable=track_for(source,tmp_path/'stable.json')
    data=json.loads(stable.read_text());data['frames'][2]['time']+=.005;stable.write_text(json.dumps(data))
    output=tmp_path/'result.mp4'
    with pytest.raises(ValueError,match='timestamp'):
        export_video(source,output,stable_path=stable,engine_factory=FakeEngine)
    assert not output.exists()


@pytest.mark.parametrize('bad',['nan','duplicate','visibility','shot','weight','version'])
def test_stable_track_validation(tmp_path,bad):
    source=_make_clip(tmp_path/'source.mp4',CFR[:3]);path=track_for(source,tmp_path/'stable.json')
    data=json.loads(path.read_text())
    if bad=='nan':data['frames'][0]['landmarks'][0][0]=float('nan')
    if bad=='duplicate':data['frames'].append(data['frames'][0])
    if bad=='visibility':data['frames'][0]['visibility']['nose']=2
    if bad=='shot':data['frames'][0]['shot']=4
    if bad=='weight':data['frames'][0]['weight']=-1
    if bad=='version':data['version']=99
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):load_stable_track(path)


def test_lossless_container_that_changes_aac_end_trim_is_not_published(tmp_path):
    source=_make_clip(tmp_path/'source.mp4',CFR[:6],audio_seconds=.2)
    output=tmp_path/'result.mkv'
    with pytest.raises(ValueError,match='audio'):
        export_video(source,output,params=RenderParams(smooth=0,whiten=0),lossless=True)
    assert not output.exists()
    manifest=json.loads((Path(str(output)+'.qa')/'manifest.json').read_text())
    assert manifest['status']=='failed'


def test_stop_after_engine_opens_before_first_frame_is_cancelled(tmp_path):
    source=_make_clip(tmp_path/'source.mp4',CFR[:3]);track=track_for(source,tmp_path/'stable.json')
    output=tmp_path/'result.mp4';stop=[False]
    class StopOnOpen(FakeEngine):
        def __enter__(self):stop[0]=True;return self
    with pytest.raises(VideoExportCancelled):
        export_video(source,output,stable_path=track,engine_factory=StopOnOpen,cancelled=lambda:stop[0])
    assert not output.exists()
    assert json.loads((Path(str(output)+'.qa')/'manifest.json').read_text())['status']=='cancelled'


def test_progress_callback_failure_after_publication_is_not_a_failed_export(tmp_path):
    source=_make_clip(tmp_path/'source.mp4',CFR[:3]);output=tmp_path/'result.mp4'
    def callback(event):
        if event['phase']=='completed':raise RuntimeError('broken viewer')
    result=export_video(source,output,params=RenderParams(smooth=0,whiten=0),qa=False,progress=callback)
    assert result['status']=='completed' and output.exists()


def test_cancel_after_link_unknown_outcome_reconciles_owned_output(tmp_path,monkeypatch):
    import os
    source=_make_clip(tmp_path/'source.mp4',CFR[:3]);output=tmp_path/'result.mp4'
    link=os.link
    def interrupted_link(a,b):
        link(a,b)
        raise VideoExportCancelled('stop at publication boundary')
    monkeypatch.setattr('retouch.video.export.os.link',interrupted_link)
    with pytest.raises(VideoExportCancelled):
        export_video(source,output,params=RenderParams(smooth=0,whiten=0),qa=False)
    assert not output.exists()
    assert json.loads((Path(str(output)+'.qa')/'manifest.json').read_text())['status']=='cancelled'


def test_output_created_by_someone_else_during_export_is_never_replaced(tmp_path,monkeypatch):
    import os
    source=_make_clip(tmp_path/'source.mp4',CFR[:3]);output=tmp_path/'result.mp4'
    link=os.link
    def raced_link(a,b):
        Path(b).write_bytes(b'preserve concurrent output')
        link(a,b)
    monkeypatch.setattr('retouch.video.export.os.link',raced_link)
    with pytest.raises(FileExistsError):
        export_video(source,output,params=RenderParams(smooth=0,whiten=0),qa=False)
    assert output.read_bytes()==b'preserve concurrent output'


def test_standard_aac_partial_end_trim_is_preserved_by_mp4(tmp_path):
    source=_make_clip(tmp_path/'source.mp4',CFR[:6],audio_seconds=.2)
    output=tmp_path/'result.mp4'
    manifest=export_video(source,output,params=RenderParams(smooth=0,whiten=0),qa=False)
    assert manifest['audio']['status']=='packet_identical'
    assert manifest['audio']['source']['end_pts']==manifest['audio']['output']['end_pts']


def test_reviewed_track_changed_during_job_prevents_publication(tmp_path):
    source=_make_clip(tmp_path/'source.mp4',CFR[:4]);track=track_for(source,tmp_path/'stable.json')
    output=tmp_path/'result.mp4'
    def progress(event):
        if event['phase']=='rendering' and event['frames']==1:
            track.write_text(track.read_text()+' ')
    with pytest.raises(ValueError,match='track changed'):
        export_video(source,output,stable_path=track,engine_factory=FakeEngine,progress=progress)
    assert not output.exists()


def test_tagged_hdr_input_is_rejected_before_a_job_is_created(tmp_path):
    import av
    source=tmp_path/'hdr.mp4'
    with av.open(str(source),'w') as container:
        stream=container.add_stream('libx264',rate=30);stream.width=96;stream.height=64;stream.pix_fmt='yuv420p'
        stream.codec_context.color_trc=16;stream.codec_context.color_primaries=9;stream.codec_context.colorspace=9
        frame=av.VideoFrame.from_ndarray(np.full((64,96,3),100,np.uint8),format='bgr24')
        for packet in list(stream.encode(frame))+list(stream.encode()):container.mux(packet)
    output=tmp_path/'result.mp4'
    with pytest.raises(ValueError,match='8-bit SDR'):export_video(source,output)
    assert not output.exists() and not Path(str(output)+'.qa').exists()


def make_pcm_clip(path):
    import av
    from fractions import Fraction
    rate=48000
    with av.open(str(path),'w') as container:
        video=container.add_stream('libx264',rate=30);video.width=96;video.height=64;video.pix_fmt='yuv420p'
        audio=container.add_stream('pcm_s16le',rate=rate);audio.layout='stereo'
        for i in range(6):
            frame=av.VideoFrame.from_ndarray(np.full((64,96,3),100,np.uint8),format='bgr24')
            frame.pts=i;frame.time_base=Fraction(1,30)
            for packet in video.encode(frame):container.mux(packet)
        for packet in video.encode():container.mux(packet)
        samples=np.random.default_rng(31).integers(-20000,20000,(9600,2),dtype=np.int16)
        for start in range(0,len(samples),1024):
            frame=av.AudioFrame.from_ndarray(samples[start:start+1024].reshape(1,-1),format='s16',layout='stereo')
            frame.pts=start;frame.sample_rate=rate;frame.time_base=Fraction(1,rate)
            for packet in audio.encode(frame):container.mux(packet)
        for packet in audio.encode():container.mux(packet)
    return path


def test_fujifilm_style_pcm_mov_exports_to_mp4_without_losing_audio(tmp_path):
    source=make_pcm_clip(tmp_path/'pcm.mov');output=tmp_path/'result.mp4'
    result=export_video(source,output,params=RenderParams(smooth=0,whiten=0),qa=False)
    assert result['audio']['status']=='decoded_sample_identical'
    assert result['audio']['source']['samples']==9600
    assert result['audio']['source']['sha256']==result['audio']['output']['sha256']
    assert result['encoding']['audio_mode']=='pcm16_to_alac_lossless'


def test_audio_writer_abort_does_not_flush_remaining_pcm(tmp_path,monkeypatch):
    from retouch.video.audio import DeliveryWriter
    source=make_pcm_clip(tmp_path/'pcm.mov')
    writer=DeliveryWriter(tmp_path/'partial.mp4',source)
    calls=[]
    monkeypatch.setattr(writer,'_mux_audio_until',lambda seconds:calls.append(seconds))
    writer.__exit__(RuntimeError,RuntimeError('stop'),None)
    assert calls==[]

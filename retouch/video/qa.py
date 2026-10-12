"""Streaming temporal measurements against a same-codec identity encode.

Metrics are evidence, not a release verdict. Only three decoded frames and
three small canonical crops are retained; percentile samples are scalars.
"""
from __future__ import annotations

import hashlib
from contextlib import ExitStack, closing
from itertools import zip_longest
from pathlib import Path

import cv2
import numpy as np

from .media import read_frames
import av


def _crop(image, box):
    x, y, w, h = box
    patch = image[y:y+h, x:x+w]
    if not patch.size:
        raise ValueError('QA track box is outside the source frame')
    return cv2.resize(patch, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)


def temporal_report(source, result, null, stable, *, check_cancel=lambda: None, samples_dir=None):
    """Compare synchronized decoded frames, resetting comparisons at cuts/gaps."""
    tracks = {f.frame: f for f in stable.frames}
    energy, flicker, motion = [], [], []
    null_energy, null_flicker = [], []
    previous = None
    count = 0
    offsets, max_error = {}, 0.0
    selected = sorted(set([0, stable.frame_count//4, stable.frame_count//2,
                           3*stable.frame_count//4, max(0, stable.frame_count-1)] + stable.cuts))[:8]
    samples = []
    with ExitStack() as stack:
        streams = [stack.enter_context(closing(read_frames(path))) for path in (source, result, null)]
        for src, out, baseline in zip_longest(*streams):
            check_cancel()
            if src is None or out is None or baseline is None:
                raise ValueError('source/result/null frame counts differ')
            for label, item in (('result',out), ('null',baseline)):
                offsets.setdefault(label,item.time-src.time)
                error=abs(item.time-src.time-offsets[label])
                tolerance=max(float(src.time_base),float(item.time_base))+1e-6
                if item.image.shape != src.image.shape or error>tolerance:
                    raise ValueError(f'frame {count}: decoded dimensions or relative presentation timestamp changed')
                max_error=max(max_error,error)
            record = tracks.get(src.index)
            if record is None or record.weight <= 0:
                previous = None
            else:
                a, b, c = [_crop(item.image, record.bbox) for item in (src, out, baseline)]
                effect, codec = b-a, c-a
                energy.append(float(np.abs(effect).mean()))
                null_energy.append(float(np.abs(codec).mean()))
                if previous is not None and previous[0] == record.shot:
                    _, prev_a, prev_effect, prev_codec = previous
                    flicker.append(float(np.abs(effect-prev_effect).mean()))
                    null_flicker.append(float(np.abs(codec-prev_codec).mean()))
                    motion.append(float(np.abs(a-prev_a).mean()))
                previous = record.shot, a, effect, codec
            if src.index in selected and samples_dir is not None:
                panels = []
                for label, item in [('source',src), ('retouch',out), ('codec null',baseline)]:
                    image = cv2.resize(item.image, (320, max(1, round(item.image.shape[0]*320/item.image.shape[1]))))
                    header = np.full((24,320,3), 30, np.uint8)
                    cv2.putText(header, f'{label} frame {src.index}', (5,17), cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                    panels.append(np.concatenate([header,image],axis=0))
                samples.append(np.concatenate(panels,axis=1))
            count += 1
    if count != stable.frame_count:
        raise ValueError('decoded frame count differs from stabilized track')
    if samples:
        directory = Path(samples_dir)
        directory.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(directory/'contact-sheet.jpg'), np.concatenate(samples,axis=0)):
            raise OSError('could not save temporal QA contact sheet')
    def mean(values): return float(np.mean(values)) if values else 0.0
    def p95(values): return float(np.percentile(values,95)) if values else 0.0
    source_motion = mean(motion)
    return {
        'status':'measurement_only', 'frames_compared':count, 'tracked_frames':len(energy),
        'track_coverage':len(energy)/count if count else 0.0,
        'mean_effect_energy':mean(energy), 'mean_effect_flicker':mean(flicker),
        'p95_effect_flicker':p95(flicker), 'mean_source_motion':source_motion,
        'effect_flicker_to_motion':mean(flicker)/max(source_motion,1e-6),
        'null_mean_effect_energy':mean(null_energy), 'null_mean_effect_flicker':mean(null_flicker),
        'null_p95_effect_flicker':p95(null_flicker),
        'effect_flicker_above_null':mean(flicker)-mean(null_flicker),
        'human_review_required':True, 'video_origin_shift_s':offsets.get('result',0.0),
        'max_relative_pts_error_s':max_error,
    }


def verify_frames(source, output, expected_count, *, check_cancel=lambda: None):
    """Streaming timing/count verification, also used when temporal QA is disabled."""
    count,offset,max_error = 0,None,0.0
    with closing(read_frames(source)) as a, closing(read_frames(output)) as b:
        for src, out in zip_longest(a,b):
            check_cancel()
            if src is None or out is None:
                raise ValueError('source/output frame counts differ')
            if offset is None: offset=out.time-src.time
            error=abs(out.time-src.time-offset)
            if src.image.shape != out.image.shape or error>max(float(src.time_base),float(out.time_base))+1e-6:
                raise ValueError(f'frame {count}: dimensions or relative presentation timestamp changed')
            max_error=max(max_error,error)
            count += 1
    if count != expected_count:
        raise ValueError('encoded video did not retain every source frame')
    return {"frames":count,"video_origin_shift_s":offset,"max_relative_pts_error_s":max_error}


def _audio(path, check_cancel):
    with closing(read_frames(path)) as video:
        first_frame=next(video,None)
        if first_frame is None: raise ValueError("no source video frames decoded")
        video_first=first_frame.time
    with av.open(str(path)) as container:
        if len(container.streams.audio)>1:
            raise ValueError('V1 supports one audio stream; multiple audio streams cannot be silently discarded')
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        digest, count, first, end = hashlib.sha256(), 0, None, None
        for packet in container.demux(stream):
            check_cancel()
            if packet.dts is None: continue
            digest.update(bytes(packet)); count += 1
            if packet.pts is not None:
                time = float(packet.pts*packet.time_base)
                first = time if first is None else first
                end = time + float((packet.duration or 0)*packet.time_base)
        return {'codec':stream.codec_context.name, 'sample_rate':stream.codec_context.sample_rate,
                'packets':count, 'sha256':digest.hexdigest(), 'first_pts':first, 'end_pts':end,
                'time_base':float(stream.time_base), 'video_first_pts':video_first}


def _decoded_pcm16(path,check_cancel):
    """Canonical interleaved PCM16 samples, independent of codec framing."""
    with av.open(str(path)) as container:
        stream=container.streams.audio[0]
        digest=hashlib.sha256();count=0;first=end=None
        channels=len(stream.codec_context.layout.channels)
        rate=stream.codec_context.sample_rate
        for frame in container.decode(stream):
            check_cancel()
            if frame.format.name not in ('s16','s16p') or frame.pts is None:
                raise ValueError('lossless PCM16 verification requires timestamped 16-bit decoded samples')
            samples=frame.to_ndarray()
            samples=samples.T if frame.format.is_planar else samples.reshape(-1,channels)
            digest.update(np.ascontiguousarray(samples,dtype='<i2').tobytes())
            count+=frame.samples
            stamp=float(frame.pts*frame.time_base)
            first=stamp if first is None else first
            end=stamp+frame.samples/rate
        return {'sha256':digest.hexdigest(),'samples':count,'channels':channels,'sample_rate':rate,
                'first_pts':first,'end_pts':end}


def verify_audio(source, result, *, check_cancel=lambda: None):
    original, output = _audio(source,check_cancel), _audio(result,check_cancel)
    if original is None or output is None:
        if original != output: raise ValueError('source audio was lost or unexpected audio appeared')
        return {'status':'silent_source'}
    converted=original['codec'] in ('pcm_s16le','pcm_s16be') and output['codec']=='alac'
    if converted:
        a,b=_decoded_pcm16(source,check_cancel),_decoded_pcm16(result,check_cancel)
        for key in ('sha256','samples','channels','sample_rate'):
            if a[key]!=b[key]:raise ValueError(f'PCM-to-ALAC conversion changed decoded {key}')
        tolerance=max(original['time_base'],output['time_base'],1/a['sample_rate'])+1e-6
        for key in ('first_pts','end_pts'):
            if a[key] is None or b[key] is None or abs((a[key]-original['video_first_pts'])-(b[key]-output['video_first_pts']))>tolerance:
                raise ValueError('lossless PCM audio timing changed')
        return {'status':'decoded_sample_identical','source_codec':original['codec'],'output_codec':'alac',
                'source':a,'output':b}
    for key in ('codec','sample_rate','packets','sha256'):
        if original[key] != output[key]: raise ValueError(f'audio passthrough changed {key}')
    tolerance = max(1/max(original['sample_rate'],1),original['time_base'],output['time_base']) + 1e-6
    for key in ('first_pts','end_pts'):
        if original[key] is None or output[key] is None:
            raise ValueError('audio timestamps unavailable; cannot verify A/V timing')
        elif abs((original[key]-original['video_first_pts'])-(output[key]-output['video_first_pts']))>tolerance:
            raise ValueError('audio end timing/trim changed during remux; choose MP4 for AAC source timing' if key=='end_pts' else 'audio/video offset changed during remux')
    return {'status':'packet_identical', 'source':original, 'output':output}

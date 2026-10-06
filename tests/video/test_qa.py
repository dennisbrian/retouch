"""Streaming QA agrees with the existing metric definition and isolates shots."""
from fractions import Fraction
import dataclasses
import cv2
import numpy as np
import pytest
pytest.importorskip('av')
from retouch.video.media import Frame
from retouch.video.stabilize import StableTrack
from retouch.video.qa import temporal_report
from tests.video.test_adapter import stable_frame


def test_metrics_match_existing_harness_on_contiguous_track(monkeypatch):
    from scripts.review.video_qa_report import FaceTrack,temporal_metrics
    import retouch.video.qa as qa
    n=8;rng=np.random.default_rng(12)
    source=[rng.integers(80,120,(80,100,3),dtype=np.uint8) for i in range(n)]
    result=[np.clip(x.astype(np.int16)+i,0,255).astype(np.uint8) for i,x in enumerate(source)]
    records=[stable_frame(i,width=100,height=80) for i in range(n)]
    stable=StableTrack(100,80,30,n,[],records)
    images={'src':source,'out':result,'null':source}
    def stream(path):
        for i,image in enumerate(images[path]):yield Frame(i,i,Fraction(1,30),image)
    monkeypatch.setattr(qa,'read_frames',stream)
    report=temporal_report('src','out','null',stable)
    expected=temporal_metrics(source,result,{r.frame:FaceTrack(r.frame,*r.bbox,1.0) for r in records})
    for key,value in expected.items():assert report[key]==pytest.approx(value,abs=1e-6)
    assert report['null_mean_effect_flicker']==0 and report['human_review_required']


def test_cut_boundary_does_not_count_as_flicker(monkeypatch):
    import retouch.video.qa as qa
    n=4;source=[np.full((80,100,3),100,np.uint8) for i in range(n)]
    result=[np.full((80,100,3),100 if i<2 else 120,np.uint8) for i in range(n)]
    records=[stable_frame(i,shot=int(i>=2),width=100,height=80) for i in range(n)]
    stable=StableTrack(100,80,30,n,[2],records)
    def stream(path):
        for i,image in enumerate(result if path=='out' else source):yield Frame(i,i,Fraction(1,30),image)
    monkeypatch.setattr(qa,'read_frames',stream)
    assert temporal_report('src','out','null',stable)['mean_effect_flicker']==0


def test_missing_encoded_frame_fails_qa(monkeypatch):
    import retouch.video.qa as qa
    image=np.full((80,100,3),100,np.uint8)
    stable=StableTrack(100,80,30,2,[],[])
    def stream(path):
        for i in range(1 if path=='out' else 2):yield Frame(i,i,Fraction(1,30),image)
    monkeypatch.setattr(qa,'read_frames',stream)
    with pytest.raises(ValueError,match='counts'):temporal_report('src','out','null',stable)

"""Selected-face authority, current-crop cache correctness and parsing cadence."""
from types import SimpleNamespace
import dataclasses
import cv2
import numpy as np
import pytest
from retouch.engine import RetouchEngine
from retouch.parsing import FaceParser
from retouch.video.adapter import RenderParams,SelectedFaceRenderer,engine_controls
from retouch.video.stabilize import StableFrame,REGION_NAMES
from retouch.video.tracker import landmarks_bbox
from tests.golden_face_fixture import _RAW_LANDMARKS


def stable_frame(index=0,shot=0,weight=1.0,visibility=None,width=640,height=480):
    landmarks=np.asarray(_RAW_LANDMARKS,np.float32)
    return StableFrame(index,index/30,shot,'tracked',weight,landmarks_bbox(landmarks,width,height),
                       landmarks,{name:1.0 for name in REGION_NAMES} if visibility is None else visibility)


class FakeEngine:
    _compute_face_roi_padding=staticmethod(RetouchEngine._compute_face_roi_padding)
    def __init__(self):
        self.calls=0;self.parses=0;self.fresh=[]
        self._parser=SimpleNamespace(parse=self.parse)
    def parse(self,landmarks,image,box,**kwargs):
        self.parses+=1
        return FaceParser.__new__(FaceParser)._landmark_fallback_only(landmarks,image,None,kwargs['ied'])
    def process(self,image,face_contexts,**controls):
        self.calls+=1
        assert len(face_contexts)==1
        context=face_contexts[0]
        h,w=image.shape[:2]
        assert context.frame_size==(w,h)
        x,y,bw,bh=context.face_data.bbox
        top,bottom,left,right=self._compute_face_roi_padding(bw,bh)
        crop=image[max(0,y-top):min(h,y+bh+bottom),max(0,x-left):min(w,x+bw+right)]
        self.fresh.append(context.face_image is not None)
        if context.face_image is not None: np.testing.assert_array_equal(crop,context.face_image)
        assert context.regions.skin.shape==crop.shape[:2]
        assert controls['equalize']==0 and controls['slimming']==0 and controls['hair_enhance']==0
        return np.clip(image.astype(np.int16)+30,0,255).astype(np.uint8)
    def __enter__(self):return self
    def __exit__(self,*args):pass


@pytest.mark.parametrize('max_dim',[128,1024])
def test_scope_preserves_background_features_and_another_person(max_dim):
    engine=FakeEngine();renderer=SelectedFaceRenderer(engine,RenderParams(max_crop_dim=max_dim))
    image=np.full((480,640,3),100,np.uint8)
    record=stable_frame()
    result=renderer.render(image,record)
    changed=np.any(result!=image,axis=2)
    assert changed.any()
    np.testing.assert_array_equal(result[:80],image[:80])
    np.testing.assert_array_equal(result[:,500:],image[:,500:])
    # Exact eye/lip points are protected even if the engine changes the whole crop.
    for index in (33,263,13):
        x,y=np.round(record.landmarks[index,:2]*(640,480)).astype(int)
        np.testing.assert_array_equal(result[y,x],image[y,x])


@pytest.mark.parametrize('max_dim',[128,256,1024])
def test_scaled_crops_remain_valid_cache_inputs(max_dim):
    engine=FakeEngine();renderer=SelectedFaceRenderer(engine,RenderParams(max_crop_dim=max_dim))
    image=np.full((480,640,3),100,np.uint8)
    result=renderer.render(image,stable_frame())
    assert result.shape==image.shape and result.dtype==np.uint8 and engine.calls==1


def test_cadence_cut_and_gap_reset():
    engine=FakeEngine();renderer=SelectedFaceRenderer(engine,RenderParams(parse_every=3))
    image=np.full((480,640,3),100,np.uint8)
    for i in range(4):renderer.render(image,stable_frame(i))
    assert (renderer.parsed_frames,renderer.warped_frames)==(2,2)
    renderer.render(image,stable_frame(4,shot=1))
    assert renderer.parsed_frames==3
    assert renderer.render(image,None) is image
    renderer.render(image,stable_frame(6,shot=1))
    assert renderer.parsed_frames==4


def test_covered_region_has_no_authority():
    image=np.full((480,640,3),100,np.uint8)
    visibility={name:1.0 for name in REGION_NAMES};visibility['nose']=0
    record=stable_frame(visibility=visibility)
    output=SelectedFaceRenderer(FakeEngine()).render(image,record)
    x,y=np.round(record.landmarks[4,:2]*(640,480)).astype(int)
    np.testing.assert_array_equal(output[y-2:y+3,x-2:x+3],image[y-2:y+3,x-2:x+3])
    blocked=stable_frame(visibility={name:0 for name in REGION_NAMES})
    engine=FakeEngine()
    assert SelectedFaceRenderer(engine).render(image,blocked) is image and engine.calls==0


def test_identity_untracked_and_zero_weight_skip_engine():
    image=np.zeros((480,640,3),np.uint8);engine=FakeEngine()
    renderer=SelectedFaceRenderer(engine,RenderParams(smooth=0,whiten=0))
    assert renderer.render(image,stable_frame()) is image
    assert renderer.render(image,None) is image
    assert renderer.render(image,stable_frame(weight=0)) is image
    assert engine.calls==0


@pytest.mark.parametrize('kwargs',[{'smooth':-1},{'whiten':float('nan')},{'parse_every':0},{'max_crop_dim':4096}])
def test_invalid_controls(kwargs):
    with pytest.raises(ValueError):RenderParams(**kwargs)


def test_warped_masks_do_not_claim_a_new_parser_input():
    engine=FakeEngine();renderer=SelectedFaceRenderer(engine)
    renderer.render(np.full((480,640,3),100,np.uint8),stable_frame(0))
    renderer.render(np.full((480,640,3),120,np.uint8),stable_frame(1))
    assert engine.fresh==[True,False]
    assert engine.parses==1


@pytest.mark.parametrize('bad',['weight','landmarks','visibility'])
def test_invalid_runtime_track_fails_without_rendering(bad):
    record=stable_frame()
    if bad=='weight':record=dataclasses.replace(record,weight=float('nan'))
    if bad=='landmarks':record.landmarks[1,0]=float('nan')
    if bad=='visibility':record.visibility['nose']=float('nan')
    engine=FakeEngine()
    with pytest.raises(ValueError):SelectedFaceRenderer(engine).render(np.full((480,640,3),100,np.uint8),record)
    assert engine.calls==0


def test_occlusion_does_not_become_a_parsing_keyframe():
    engine=FakeEngine();renderer=SelectedFaceRenderer(engine,RenderParams(parse_every=1))
    image=np.full((480,640,3),100,np.uint8)
    renderer.render(image,stable_frame(0))
    visible={key:1 for key in REGION_NAMES};visible['nose']=0
    renderer.render(image,stable_frame(1,visibility=visible))
    assert engine.parses==1 and renderer.warped_frames==1
    engine2=FakeEngine();renderer2=SelectedFaceRenderer(engine2)
    assert renderer2.render(image,stable_frame(0,visibility=visible)) is image
    assert engine2.parses==0


def test_live_lip_geometry_is_protected_between_parser_runs():
    engine=FakeEngine();renderer=SelectedFaceRenderer(engine,RenderParams(parse_every=10))
    image=np.full((480,640,3),100,np.uint8)
    renderer.render(image,stable_frame(0))
    record=stable_frame(1)
    from retouch.video.stabilize import REGIONS
    record.landmarks[REGIONS['lips'],1]+=.025  # mouth moves while the rigid anchors stay still
    output=renderer.render(image,record)
    x,y=np.round(record.landmarks[17,:2]*(640,480)).astype(int)
    np.testing.assert_array_equal(output[y,x],image[y,x])
    assert engine.parses==1


def test_parsing_updates_crossfade_without_claiming_a_fresh_pixel_reference():
    engine=FakeEngine();renderer=SelectedFaceRenderer(engine,RenderParams(parse_every=3))
    image=np.full((480,640,3),100,np.uint8)
    for index in range(4):renderer.render(image,stable_frame(index))
    assert engine.parses==2 and engine.fresh==[True,False,False,False]

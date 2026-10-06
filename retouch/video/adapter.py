"""Selected-face skin retouching on stabilized tracks; image-mode code is unchanged."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np

from ..detection import FaceContext, FaceData
from ..parsing import FaceRegions, LEFT_EYEBROW, RIGHT_EYEBROW
from .stabilize import REGIONS, REGION_NAMES, StableFrame
from .tracker import landmarks_bbox

_ANCHORS = [33, 263, 1, 152, 10, 234, 454]


@dataclass(frozen=True)
class RenderParams:
    """Video-only strengths, parse cadence and maximum work-crop size."""
    smooth: float = 20.0
    whiten: float = 0.0
    parse_every: int = 3
    max_crop_dim: int = 1024

    def __post_init__(self):
        for name in ('smooth', 'whiten'):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 100:
                raise ValueError(f'{name} must be in 0..100')
        if not isinstance(self.parse_every, int) or self.parse_every < 1:
            raise ValueError('parse_every must be a positive integer')
        if not isinstance(self.max_crop_dim, int) or not 128 <= self.max_crop_dim <= 2048:
            raise ValueError('max_crop_dim must be in 128..2048')


def engine_controls(params: RenderParams) -> dict:
    """Explicit neutral image controls, with only smooth/whiten enabled.

    Dataclass defaults disable image recipe effects; dependent frequency
    scales/texture opacity retain their normal defaults. No recipe effects
    are allowed to leak through the image engine's default natural recipe.
    """
    from ..engine import ProcessingContext
    from ..params import PROCESSING_PARAMS
    neutral = ProcessingContext()
    values = {spec.name: getattr(neutral, spec.name, None) for spec in PROCESSING_PARAMS}
    values.update(smooth=params.smooth, whiten=params.whiten, micro_restore=0,
                  gamut_compress=False, eye_gate=False, fast=False, quality='full',
                  ai_denoise=0, ai_sr_scale=1, nose_smooth=0)
    return values


def _compat(points, width, height):
    return SimpleNamespace(landmark=[SimpleNamespace(x=float(x / width), y=float(y / height),
                                                   z=float(z / width)) for x, y, z in points])


def _bounds(box, width, height, padding):
    x, y, w, h = box
    top, bottom, left, right = padding(w, h)
    return max(0, x - left), max(0, y - top), min(width, x + w + right), min(height, y + h + bottom)


def skin_authority(regions, points, shape, ied, visibility):
    """Face skin only, with feature and covered-region margins kept clear."""
    h, w = shape
    if regions.skin is None or regions.face_oval is None or max(visibility.values(),default=0)<=0:
        return np.zeros(shape, np.float32)
    mask = np.clip(regions.skin, 0, 1) * np.clip(regions.face_oval, 0, 1)
    margin = max(1, round(ied * 0.04))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    for name in ('left_eye', 'right_eye', 'left_eyebrow', 'right_eyebrow', 'lips', 'mouth_interior', 'hair'):
        part = getattr(regions, name, None)
        if part is not None:
            mask *= 1 - cv2.dilate((part > 0.05).astype(np.uint8), kernel).astype(np.float32)
    # Cached semantic masks cannot follow every mouth opening/blink exactly.
    # Protect live landmark geometry as well as the parsed feature support.
    for indices in (REGIONS['eye_l'],REGIONS['eye_r'],REGIONS['lips'],LEFT_EYEBROW,RIGHT_EYEBROW):
        protected=np.zeros(shape,np.uint8)
        cv2.fillConvexPoly(protected,cv2.convexHull(points[indices,:2].astype(np.int32)),1)
        mask*=1-cv2.dilate(protected,kernel).astype(np.float32)
    for name, indices in REGIONS.items():
        value = visibility.get(name, 0.0)  # unknown visibility has no authority
        if value < 1:
            covered = np.zeros(shape, np.uint8)
            cv2.fillConvexPoly(covered, cv2.convexHull(points[indices, :2].astype(np.int32)), 1)
            covered = cv2.dilate(covered, kernel).astype(np.float32)
            mask *= 1 - covered * (1 - value)
    # Feather inward; never add pixels outside the original selected support.
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), max(0.7, ied * 0.01))
    return np.minimum(mask, soft).astype(np.float32)


def _warp_regions(source,matrix,size):
    regions=FaceRegions()
    for name in FaceRegions.__slots__:
        value=getattr(source,name,None)
        if isinstance(value,np.ndarray) and value.ndim==2:
            value=cv2.warpAffine(value,matrix,size,flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        else:value=copy.deepcopy(value)
        setattr(regions,name,value)
    regions._eye_gate_cache=None
    return regions


def _blend_regions(current,previous,alpha):
    if previous is None or alpha>=1:return current
    result=copy.deepcopy(current)
    for name in FaceRegions.__slots__:
        new,old=getattr(current,name,None),getattr(previous,name,None)
        if isinstance(new,np.ndarray) and new.ndim==2 and isinstance(old,np.ndarray) and old.shape==new.shape:
            setattr(result,name,(alpha*new+(1-alpha)*old).astype(np.float32))
    return result


class SelectedFaceRenderer:
    """One stream's cached parsing state; bounded to one selected face crop.

    Mask fitting/warping stays in ROI coordinates. Every engine call receives
    exactly one FaceContext in the current crop geometry. Fresh parses record
    their input pixels; warped masks leave that parser-input claim unset. Final compositing confines even an engine-wide edit to
    the selected face's visible skin. Untracked/zero-weight frames are identity.
    """
    def __init__(self, engine, params: RenderParams = RenderParams()):
        self.engine, self.params = engine, params
        self.controls = engine_controls(params)
        self.reset()
        self.parsed_frames = 0
        self.warped_frames = 0
        self.edited_frames = 0

    def reset(self):
        """Drop a stream's mask reference at a cut or tracking loss."""
        self._reference = None
        self._shot = None
        self._last_frame = None

    def render(self, image: np.ndarray, stable: Optional[StableFrame]) -> np.ndarray:
        """Apply a bounded delta only to the selected face's visible skin."""
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2])<1:
            raise ValueError('expected uint8 H x W x 3 BGR frame')
        if stable is not None and (not np.isfinite(stable.weight) or not 0<=stable.weight<=1):
            raise ValueError('invalid stabilized frame weight')
        if stable is None or stable.weight <= 0:
            self.reset()
            return image
        if self.params.smooth == 0 and self.params.whiten == 0:
            return image
        if self.engine is None:
            raise ValueError('active video controls require a retouch engine')
        if (np.asarray(stable.landmarks).shape != (478,3)
                or not np.isfinite(stable.landmarks).all()
                or not np.isfinite(stable.weight) or not 0<=stable.weight<=1
                or set(stable.visibility)!=set(REGION_NAMES)
                or any(not np.isfinite(v) or not 0<=v<=1 for v in stable.visibility.values())):
            raise ValueError('invalid stabilized landmarks, weight or visibility')
        h, w = image.shape[:2]
        padding = self.engine._compute_face_roi_padding
        box = landmarks_bbox(stable.landmarks, w, h)
        x0, y0, x1, y1 = _bounds(box, w, h, padding)
        native = image[y0:y1, x0:x1]
        if min(native.shape[:2]) < 4:
            self.reset()
            return image
        scale = min(1.0, self.params.max_crop_dim / max(native.shape[:2]))
        work = native if scale == 1 else cv2.resize(native, (max(1, round(native.shape[1] * scale)),
                                                            max(1, round(native.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        wh, ww = work.shape[:2]
        sx, sy = ww / native.shape[1], wh / native.shape[0]
        points = stable.landmarks.astype(np.float64) * (w, h, w)
        points -= (x0, y0, 0)
        points *= (sx, sy, sx)
        normalized = points / (ww, wh, ww)
        work_box = landmarks_bbox(normalized, ww, wh)
        ix0, iy0, ix1, iy1 = _bounds(work_box, ww, wh, padding)
        crop = work[iy0:iy1, ix0:ix1]
        ch, cw = crop.shape[:2]
        if min(ch, cw) < 4:
            self.reset()
            return image
        local = points - (ix0, iy0, 0)
        ied = float(np.linalg.norm(points[33, :2] - points[263, :2]))
        if ied < 4:
            self.reset()
            return image
        landmarks = _compat(local, cw, ch)
        local_box = (work_box[0] - ix0, work_box[1] - iy0, work_box[2], work_box[3])
        continuous=(self._reference is not None and stable.shot==self._shot and self._last_frame==stable.frame-1)
        clear=stable.source=='tracked' and all(v>=1-1e-6 for v in stable.visibility.values())
        refresh=not continuous or stable.frame-self._reference['frame']>=self.params.parse_every
        matrix=None;warp_ok=False
        if continuous:
            reference=self._reference['points'][_ANCHORS,:2].astype(np.float32)
            current=local[_ANCHORS,:2].astype(np.float32)
            matrix,_=cv2.estimateAffinePartial2D(reference,current,method=cv2.LMEDS)
            if matrix is not None and np.isfinite(matrix).all():
                fit=reference@matrix[:,:2].T+matrix[:,2]
                residual=np.linalg.norm(fit-current,axis=1).max()/ied
                ratio=np.linalg.norm(matrix[0,:2])
                warp_ok=residual<=0.05 and 0.8<=ratio<=1.25
            if not warp_ok:refresh=True
        if not clear:
            if not continuous or not warp_ok:
                if not continuous:self.reset()
                else:self._last_frame=stable.frame
                return image
            refresh=False  # never learn a new skin/hair mask from an occluder
        previous=None
        if refresh:
            if continuous and warp_ok:
                previous=_warp_regions(self._reference['regions'],matrix,(cw,ch))
            regions=self.engine._parser.parse(landmarks,crop,local_box,ied=ied,mask_feather_mode='gaussian')
            self._reference={'regions':copy.deepcopy(regions),'previous':previous,'points':local.copy(),'frame':stable.frame}
            self.parsed_frames+=1
        else:
            regions=_warp_regions(self._reference['regions'],matrix,(cw,ch))
            if self._reference.get('previous') is not None:
                previous=_warp_regions(self._reference['previous'],matrix,(cw,ch))
            self.warped_frames+=1
        blend=min(1.0,(stable.frame-self._reference['frame']+1)/self.params.parse_every)
        regions=_blend_regions(regions,previous,blend)
        fresh_input=refresh and (previous is None or blend>=1)
        self._shot, self._last_frame = stable.shot, stable.frame
        authority = skin_authority(regions, local, (ch, cw), ied, stable.visibility)
        if not np.any(authority > 1e-3):
            return image
        face = FaceData(landmarks=_compat(points, ww, wh), bbox=work_box, ied=ied,
                        confidence=1.0, confidence_source='stabilized_track_presence_gate')
        context = FaceContext(face, regions, face_image=crop.copy() if fresh_input else None,
                              frame_size=(ww, wh), mask_feather_mode='gaussian')
        result = np.asarray(self.engine.process(work, face_contexts=[context], **self.controls))
        if result.shape != work.shape or result.dtype != np.uint8:
            raise ValueError('video engine output must retain uint8 crop dimensions')
        mask = np.zeros((wh, ww), np.float32)
        mask[iy0:iy1, ix0:ix1] = authority
        delta = result.astype(np.float32) - work.astype(np.float32)
        if scale < 1:
            delta = cv2.resize(delta, (native.shape[1], native.shape[0]), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (native.shape[1], native.shape[0]), interpolation=cv2.INTER_LINEAR)
        amount = np.clip(mask * stable.weight, 0, 1)[..., None]
        corrected = np.clip(native.astype(np.float32) + delta * amount + 0.5, 0, 255).astype(np.uint8)
        out = image.copy()
        out[y0:y1, x0:x1] = corrected
        self.edited_frames += 1
        return out

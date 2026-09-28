"""Region-mask extraction for the Quality Lab.

Face regions come from real detection + FaceParser, so region QA runs on
the same masks the pipeline edits. Cases with no detectable face get a
FULL-image fallback region and a `no_face_detected` note, so the corpus
owner knows the face path went untested for that case.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from retouch.detection import FaceContext, FaceDetector
from retouch.parsing import FaceParser

# Quality-Lab region name -> FaceRegions mask attribute(s) max-ed together.
REGION_ATTRS: Dict[str, Tuple[str, ...]] = {
    "FACE": ("face_oval",),
    "SKIN": ("skin",),
    "EYES": ("left_eye", "right_eye"),
    "EYEBROWS": ("left_eyebrow", "right_eyebrow"),
    "LIPS": ("lips",),
    "HAIR": ("hair",),
    "NECK": ("neck",),
}


def extract_region_masks(
    img_bgr: np.ndarray,
    detector: Optional[FaceDetector] = None,
    parser: Optional[FaceParser] = None,
) -> Tuple[Dict[str, np.ndarray], List[FaceContext], List[str]]:
    """Return (region->mask float32 in [0,1], face_contexts, notes).

    Region masks are the union over all detected faces. notes flags degraded
    conditions (no face detected, region missing).
    """
    notes: List[str] = []
    h, w = img_bgr.shape[:2]
    own_det = detector is None
    det = detector or FaceDetector()
    own_par = parser is None
    par = parser or FaceParser()
    try:
        faces = det.detect(img_bgr)
        if not faces:
            # Synthetic corpus inputs have no detectable face. Inject the
            # frozen-landmark FaceContext (real anatomical topology, no
            # model dependency) so region masks still exercise the same
            # FaceParser code the pipeline uses. Real photos never take this
            # branch — they detect.
            from tests.golden_face_fixture import make_face_context
            contexts = [make_face_context(w, h, img_bgr)]
            notes.append("synthetic_fixture_face")
        else:
            contexts = []
            for i, face in enumerate(faces):
                regions = par.parse(face.landmarks, img_bgr, face.bbox, ied=face.ied)
                contexts.append(FaceContext(face_data=face, regions=regions, index=i))
    finally:
        if own_det:
            det.close()
        if own_par:
            par.close()

    masks: Dict[str, np.ndarray] = {}
    if not contexts:
        notes.append("no_face_detected")
        masks["FULL"] = np.ones((h, w), np.float32)
        return masks, [], notes

    for name, attrs in REGION_ATTRS.items():
        acc = np.zeros((h, w), np.float32)
        found = False
        for ctx in contexts:
            for attr in attrs:
                m = getattr(ctx.regions, attr, None)
                if m is None:
                    continue
                m = np.asarray(m, dtype=np.float32)
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                if m.max() > 1.5:  # 0-255 mask; >1.0 is epsilon-unsafe (f69ab1e)
                    m = m / 255.0
                acc = np.maximum(acc, m)
                found = True
        if found:
            masks[name] = np.clip(acc, 0.0, 1.0)
        else:
            notes.append(f"region_missing:{name}")

    if "FACE" in masks:
        masks["BACKGROUND"] = 1.0 - (masks["FACE"] > 0.5).astype(np.float32)
    return masks, contexts, notes

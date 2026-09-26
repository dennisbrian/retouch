"""Match a whole set of photos to one hero frame.

Colour and exposure jump between frames shot minutes apart: the sun goes
behind a cloud, the camera's auto white balance hunts, a hall's mixed light
changes as the cosplayer moves.  On a carousel post those jumps show as the
slides are swiped.  This module measures a *hero* frame once and then gives
every other frame the global exposure and white-balance shift that brings its
subject back to the hero's, before the recipe runs, so the same recipe lands
the same way on every slide.

Design:

* The correction is a single per-channel gain in linear light (exposure plus
  a von Kries white balance).  It is the same kind of change a camera's
  exposure and white-balance settings make, so costume, wig and background
  colours keep their relationships to each other.  Whole-frame statistics
  transfer (Reinhard, ``--color-ref``) would instead drag a red-costume
  close-up toward a blue-hall wide shot's average.
* The gain is measured on the subject's own face skin (face oval minus eyes,
  brows and lips, trimmed to its middle luminance band).  The anchor is the
  same person's skin in both frames, so it is relative by construction and
  works the same at every skin tone; no absolute luminance gate is used.
* Frames without a face, or a hero without one, fall back to a trimmed
  grey-world measurement of the whole frame with tighter clamps, since
  framing changes move whole-frame averages more than lighting does.
* Clamps bound every correction (exposure and white balance), a soft
  shoulder keeps brightened highlights from clipping, and blown-out pixels
  (all three channels at the top of the range) are left untouched so a lamp
  or a white wig's clipped highlight does not pick up a tint.

Matching the hero to itself is an exact no-op.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .parsing import FACE_OVAL, LEFT_EYE, LEFT_EYEBROW, LIPS_OUTER, RIGHT_EYE, RIGHT_EYEBROW
from .utils import get_points

_logger = logging.getLogger(__name__)

# Longest side of the copy that is measured.  Statistics are means over
# thousands of pixels, so resolution beyond this changes nothing.
MEASURE_MAX_DIM = 1024

# Minimum face-skin pixels (at the measurement size) before the skin anchor is
# trusted; below this the whole-frame fallback is used.
MIN_SKIN_PIXELS = 400

# Clamps.  Skin anchor: +/-1.5 EV and each white-balance channel gain within
# [0.7, 1.43] (about half a stop per channel; covers daylight vs. a warm hall).
# Whole-frame fallback: +/-1 EV and [0.9, 1.11].
SKIN_MAX_EV = 1.5
SKIN_WB_RANGE = (0.7, 1.0 / 0.7)
SCENE_MAX_EV = 1.0
SCENE_WB_RANGE = (0.9, 1.0 / 0.9)

# Soft shoulder knee in linear light, used only when a gain brightens.
_SHOULDER_KNEE = 0.8

# Blown-pixel protection: encoded min channel from _BLOWN_LO (start) to
# _BLOWN_HI (fully protected), as a fraction of full scale.  This detects
# sensor clipping, not skin brightness, so it is not a tone gate.
_BLOWN_LO = 0.94
_BLOWN_HI = 0.99

# Rec.709 luminance weights in BGR order.
_LUMA_BGR = np.array([0.0722, 0.7152, 0.2126], dtype=np.float64)

_EPS = 1e-6


@dataclass(frozen=True)
class FrameStats:
    """What :func:`measure_frame` found in one frame.

    Colours are linear-light BGR means in ``[0, 1]``.  ``skin_bgr`` is
    ``None`` when no usable face was found.  Plain tuples keep the object tiny
    and cheap to pickle to batch workers.
    """

    scene_bgr: Tuple[float, float, float]
    skin_bgr: Optional[Tuple[float, float, float]] = None
    skin_pixels: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MatchGains:
    """Correction for one frame: overall exposure and per-channel gains."""

    basis: str                              # "skin", "scene" or "none"
    exposure_ev: float
    wb_bgr: Tuple[float, float, float]      # white-balance gains, luminance-neutral
    gains_bgr: Tuple[float, float, float]   # total linear gains (exposure x wb)

    @property
    def is_identity(self) -> bool:
        return all(abs(g - 1.0) < 1e-4 for g in self.gains_bgr)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "basis": self.basis,
            "exposure_ev": round(self.exposure_ev, 4),
            "wb_bgr": [round(g, 4) for g in self.wb_bgr],
            "gains_bgr": [round(g, 4) for g in self.gains_bgr],
        }


IDENTITY = MatchGains("none", 0.0, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _decode(x01: np.ndarray) -> np.ndarray:
    """sRGB-encoded ``[0, 1]`` to linear light."""
    x = np.clip(x01, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _encode(x: np.ndarray) -> np.ndarray:
    """Linear light to sRGB-encoded ``[0, 1]``."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def _shoulder(x: np.ndarray) -> np.ndarray:
    """Identity below the knee, exponential roll-off to 1.0 above it."""
    k = _SHOULDER_KNEE
    span = 1.0 - k
    over = np.maximum(x - k, 0.0)
    return np.where(x <= k, x, k + span * (1.0 - np.exp(-over / span)))


def _full_scale(img: np.ndarray) -> float:
    """Full-scale value of an engine image: 255 for uint8 and float [0, 255]."""
    if img.dtype == np.uint8:
        return 255.0
    if img.dtype == np.uint16:
        return 65535.0
    # Engine float inputs are [0, 255] (16-bit RAF ingest); [0, 1] floats are
    # accepted too for API callers.
    return 1.0 if float(np.max(img)) <= 1.0 + 1e-3 else 255.0


def _to_linear01(img: np.ndarray) -> np.ndarray:
    return _decode(img.astype(np.float64) / _full_scale(img))


def _luma(bgr: np.ndarray) -> np.ndarray:
    return bgr @ _LUMA_BGR


def _trimmed_mean(pixels: np.ndarray, lo: float = 5.0, hi: float = 95.0) -> Optional[np.ndarray]:
    """Mean of (N, 3) linear pixels whose luminance sits inside [lo, hi] percentiles.

    The band is relative to the pixels' own distribution: it drops specular
    highlights, nostril or eye-socket shadow and stray hair without any
    absolute brightness threshold.
    """
    if pixels.shape[0] == 0:
        return None
    y = _luma(pixels)
    a, b = np.percentile(y, [lo, hi])
    keep = (y >= a) & (y <= b)
    if not np.any(keep):
        keep = np.ones_like(y, dtype=bool)
    return pixels[keep].mean(axis=0)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def _downscale(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = MEASURE_MAX_DIM / float(max(h, w))
    if scale >= 1.0:
        return img
    return cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                      interpolation=cv2.INTER_AREA)


def face_skin_mask(landmarks: Any, h: int, w: int) -> np.ndarray:
    """Boolean skin mask for one face: eroded face oval minus eyes, brows and lips."""
    mask = np.zeros((h, w), dtype=np.uint8)
    oval = get_points(landmarks, FACE_OVAL, w, h)
    cv2.fillPoly(mask, [oval], 255)
    # Pull the edge in so hairline, jaw shadow and background never count.
    fw = max(1, int(oval[:, 0].max() - oval[:, 0].min()))
    k = max(3, (fw // 12) | 1)
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    holes = np.zeros_like(mask)
    for idx in (LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW, LIPS_OUTER):
        cv2.fillPoly(holes, [cv2.convexHull(get_points(landmarks, idx, w, h))], 255)
    holes = cv2.dilate(holes, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return (mask > 0) & (holes == 0)


def measure_frame(img: np.ndarray, detector: Any = None, faces: Any = None) -> FrameStats:
    """Measure the colour anchors of one frame.

    Args:
        img: BGR image (uint8, uint16, or float in [0, 255] / [0, 1]).
        detector: object with ``detect(img_bgr) -> List[FaceData]``; when
            ``None`` and ``faces`` is not given, only the whole-frame anchor
            is measured.
        faces: optional pre-detected faces *for the measured copy*; mostly
            for tests.

    Returns:
        :class:`FrameStats` with the whole-frame anchor and, when a face with
        enough visible skin was found, the skin anchor of the largest face.
    """
    small = _downscale(img)
    if small.dtype != np.uint8:
        det_img = np.clip(small.astype(np.float64) * (255.0 / _full_scale(small)), 0, 255).astype(np.uint8)
    else:
        det_img = small
    lin = _to_linear01(small).reshape(-1, 3)
    scene = _trimmed_mean(lin, 10.0, 90.0)
    scene_t = tuple(float(v) for v in scene) if scene is not None else (0.18, 0.18, 0.18)

    if faces is None and detector is not None:
        try:
            faces = detector.detect(det_img)
        except Exception as exc:  # detection is advisory here
            _logger.warning("set match: face detection failed (%s); using whole frame", exc)
            faces = None

    if faces:
        face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        h, w = small.shape[:2]
        try:
            skin = face_skin_mask(face.landmarks, h, w).reshape(-1)
        except (AttributeError, IndexError, TypeError) as exc:
            _logger.warning("set match: could not build skin mask (%s)", exc)
            skin = None
        if skin is not None:
            n = int(skin.sum())
            if n >= MIN_SKIN_PIXELS:
                mean = _trimmed_mean(lin[skin])
                if mean is not None and float(_luma(mean)) > _EPS:
                    return FrameStats(scene_t, tuple(float(v) for v in mean), n)
    return FrameStats(scene_t)


# ---------------------------------------------------------------------------
# Gains
# ---------------------------------------------------------------------------

def compute_gains(hero: FrameStats, frame: FrameStats, strength: float = 1.0) -> MatchGains:
    """Exposure + white-balance gains that move ``frame`` onto ``hero``.

    ``strength`` in ``[0, 1]`` interpolates geometrically from no change to
    the full (clamped) correction.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return IDENTITY
    if hero.skin_bgr is not None and frame.skin_bgr is not None:
        basis, h_rgb, f_rgb = "skin", hero.skin_bgr, frame.skin_bgr
        max_ev, (wb_lo, wb_hi) = SKIN_MAX_EV, SKIN_WB_RANGE
    else:
        basis, h_rgb, f_rgb = "scene", hero.scene_bgr, frame.scene_bgr
        max_ev, (wb_lo, wb_hi) = SCENE_MAX_EV, SCENE_WB_RANGE
    h_rgb = np.maximum(np.asarray(h_rgb, dtype=np.float64), _EPS)
    f_rgb = np.maximum(np.asarray(f_rgb, dtype=np.float64), _EPS)
    y_h, y_f = float(_luma(h_rgb)), float(_luma(f_rgb))

    ev = float(np.clip(np.log2(y_h / y_f), -max_ev, max_ev)) * strength

    # Chromaticity ratio (von Kries), clamped, then renormalised so white
    # balance alone does not change the anchor's luminance.
    wb = (h_rgb / y_h) / (f_rgb / y_f)
    wb = np.clip(wb, wb_lo, wb_hi)
    wb = wb / max(float(_luma(wb * f_rgb)) / y_f, _EPS)
    wb = wb ** strength

    gains = wb * (2.0 ** ev)
    return MatchGains(basis, ev, tuple(float(g) for g in wb), tuple(float(g) for g in gains))


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def _channel_curve(values01: np.ndarray, gain: float, shoulder: bool) -> np.ndarray:
    lin = _decode(values01) * gain
    if shoulder:
        lin = _shoulder(lin)
    return _encode(lin)


def apply_gains(img: np.ndarray, gains: MatchGains) -> np.ndarray:
    """Apply ``gains`` to a full-resolution image, keeping its dtype and range.

    Each channel goes through a 1-D curve (decode, gain, optional shoulder,
    encode), so the cost is a lookup per pixel even at 45 MP.
    """
    if gains.is_identity:
        return img
    full = _full_scale(img)
    shoulder = max(gains.gains_bgr) > 1.0 + 1e-6
    out = np.empty_like(img)
    if img.dtype == np.uint8:
        x = np.arange(256, dtype=np.float64) / 255.0
        for c in range(3):
            lut = np.clip(np.round(_channel_curve(x, gains.gains_bgr[c], shoulder) * 255.0), 0, 255).astype(np.uint8)
            out[..., c] = cv2.LUT(img[..., c], lut)
    else:
        x = np.linspace(0.0, 1.0, 4097)
        for c in range(3):
            curve = _channel_curve(x, gains.gains_bgr[c], shoulder)
            ch = np.clip(img[..., c].astype(np.float32) / full, 0.0, 1.0)
            out[..., c] = (np.interp(ch, x, curve) * full).astype(img.dtype)

    # Leave blown pixels (all channels near full scale) as they were.
    min_ch = img.min(axis=2).astype(np.float32) / full
    protect = np.clip((min_ch - _BLOWN_LO) / (_BLOWN_HI - _BLOWN_LO), 0.0, 1.0)
    sel = protect > 0
    if np.any(sel):
        p = protect[sel][:, None]
        blended = out[sel].astype(np.float32) * (1.0 - p) + img[sel].astype(np.float32) * p
        if img.dtype == np.uint8:
            blended = np.clip(np.round(blended), 0, 255)
        out[sel] = blended.astype(img.dtype)
    return out


def match_to_hero(
    img: np.ndarray,
    hero: FrameStats,
    strength: float = 1.0,
    detector: Any = None,
    faces: Any = None,
) -> Tuple[np.ndarray, MatchGains]:
    """Measure ``img``, compute the correction toward ``hero`` and apply it."""
    if strength <= 0.0:
        return img, IDENTITY
    stats = measure_frame(img, detector=detector, faces=faces)
    gains = compute_gains(hero, stats, strength)
    return apply_gains(img, gains), gains


def coerce_hero(hero: Any, detector: Any = None) -> Optional[FrameStats]:
    """Accept a :class:`FrameStats`, its ``to_dict()`` form, or a hero image."""
    if hero is None or isinstance(hero, FrameStats):
        return hero
    if isinstance(hero, dict):
        skin = hero.get("skin_bgr")
        return FrameStats(
            tuple(float(v) for v in hero["scene_bgr"]),
            tuple(float(v) for v in skin) if skin is not None else None,
            int(hero.get("skin_pixels", 0)),
        )
    if isinstance(hero, np.ndarray):
        return measure_frame(hero, detector=detector)
    raise TypeError(f"set_match_hero must be FrameStats, dict or ndarray, not {type(hero).__name__}")

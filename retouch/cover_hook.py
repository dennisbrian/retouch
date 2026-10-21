"""Cover hook score: pick the Instagram cover and slides 2-3 from one shoot.

Scores every retouched photo in a folder on six checks, ranks them, suggests a
cover plus two follow-up slides and exports the cover's 4:5 crop. It only
reads the photos; its own files go to ``<folder>/cover-hook/``.

Where the checks come from
--------------------------
A comparison of 124 Instagram photo-post covers from 12 cosplay accounts
(top third against bottom third by likes inside each account, October 2026).
The evidence is thin and the weights below are **set by hand from weak
leans, not fitted**:

* Only one difference was large: poster / promo-graphic covers were 0 of 27
  in the top third and 8 of 27 in the bottom third.
* Among real photos the leans were small: one saturated colour (14 of 21 top
  against 10 of 21 bottom), full-body or pair framing (15 against 10), outdoor
  or studio light over dim indoor light, and a crushed hard-contrast look
  leaning the other way (2 against 5).
* Eye contact, face size, centring and background clutter showed no
  difference, so none of them is scored.

The six checks
--------------
1. ``photo``    the cover is a finished photo, not a poster, collage or
                screenshot. Needs a Claude look (``--vision``).
2. ``crop``     the subject survives the 4:5 feed crop and the 3:4 grid crop.
                :mod:`retouch.social_crops` plans the crop; this module
                measures how much of each head and of the person mask is kept.
3. ``colour``   one vivid colour dominates the subject
                (:func:`retouch.image_analyzer.colour_pop_score`).
4. ``costume``  the whole costume reads: full-body or pair over a tight
                half-body. With ``--vision`` Claude names the framing;
                otherwise a proxy from the face count and face height.
5. ``light``    clean light, not a colour cast, dim or crushed
                (:class:`retouch.image_analyzer.ImageAnalyzer`). The analyzer
                cannot tell a location, so outdoor / studio over indoor is
                only scored with ``--vision``.
6. ``face``     a gate, not a score: both eyes closed, or a face much softer
                than the rest of the shoot, demote the photo
                (:mod:`retouch.face_quality`). A wink is noted, not demoted.
                It carried no engagement signal, so it can only lower a score.

A check that could not be measured (no ``--vision``, no face found) is left
out and the score is taken over the checks that were measured. It never
counts as a pass or a fail.

The Claude look
---------------
``--vision`` asks Claude one question per candidate, for the best
``--vision-top`` photos of the offline ranking only, on a 1024 px copy. By
default it runs the local Claude Code CLI headless (``claude -p``), which uses
the signed-in subscription and needs no API key. The Anthropic API is used
only when ``ANTHROPIC_API_KEY`` is set, the ``anthropic`` package is installed
and ``RETOUCH_COVER_LOOK_MODEL`` names a model. Answers are cached by file
content in ``cover-hook/.cache/look.json``.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from retouch.social_crops import FORMATS, Box, CropPlan, plan_crop, select_subject_faces

_logger = logging.getLogger(__name__)

OUTPUT_DIRNAME = "cover-hook"
SCORES_JSON = "cover_scores.json"
SCORES_CSV = "cover_scores.csv"
SCORE_VERSION = "cover-hook-v1"

# Hand-set weights (see the module docstring): the poster finding is the only
# strong one, crop safety is a hard practical limit, the rest are weak leans.
WEIGHTS: Dict[str, int] = {"photo": 30, "crop": 25, "costume": 20, "colour": 15, "light": 10}
WEIGHT_REASONS: Dict[str, str] = {
    "photo": "poster / promo covers: 0 of 27 top third, 8 of 27 bottom third (the one strong finding)",
    "crop": "covers that lose the subject in the 4:5 crop ranked lowest (6 covers, thin); the grid always crops",
    "costume": "full-body or pair 15 of 21 top against 10 of 21 bottom; half-body 4 against 8 (weak lean)",
    "colour": "saturated colour 14 of 21 top against 10 of 21 bottom (weak lean)",
    "light": "outdoor 9 against 5, dim indoor 5 against 8, hard contrast 2 against 5 (weak lean)",
    "face": "no engagement signal; quality gate only (both eyes closed x0.6, soft face x0.8)",
}

CROP_FORMATS: Tuple[Tuple[str, float], ...] = (("4:5", 0.6), ("3:4", 0.4))
# Share a crop has to keep: below the first value scores 0, above the second scores 1. Heads (face
# boxes grown to wig and ears) must stay almost whole; the person mask may lose legs or a hem.
CROP_HEAD_KEEP = (0.80, 0.97)
CROP_PERSON_KEEP = (0.60, 0.90)
CROP_SPLIT_FACES_CAP = 0.3   # faces wider than the crop: someone is cut whatever the placement

# Offline framing proxy: main face height as a share of the frame height.
FRAMING_WIDE_BELOW = 0.04
FRAMING_FULL_BELOW = 0.12
FRAMING_THREE_QUARTER_BELOW = 0.20
FRAMING_HALF_BELOW = 0.33
FRAMING_SCORES: Dict[str, float] = {
    "pair": 1.0, "full-body": 1.0, "three-quarter": 0.75, "half-body": 0.45, "close-up": 0.25, "wide": 0.5,
}
TIGHT_FRAMINGS = ("half-body", "close-up")
COSTUME_HIDDEN_FACTOR = 0.6

KIND_SCORES: Dict[str, float] = {"photo": 1.0, "collage": 0.4, "poster": 0.0, "screenshot": 0.0}
# The analyzer cannot tell a location; with a Claude look the light check is half measured, half setting.
SETTING_SCORES: Dict[str, float] = {"outdoor": 1.0, "studio": 1.0, "indoor": 0.6, "event-floor": 0.6}
LIGHT_SCORES: Dict[str, float] = {"outdoor": 1.0, "studio": 1.0, "convention": 0.6, "low_light": 0.4}
LIGHT_HARD_CONTRAST = 0.85   # ImageAnalysis.contrast_level above this, on a low-key frame, reads as crushed
LIGHT_HARD_FACTOR = 0.8

FACE_GATE_CLOSED = 0.6
FACE_GATE_SOFT = 0.8
FACE_SOFT_RATIO = 0.6        # main-face sharpness below this share of the shoot's median counts as soft

PROXY_MAX_SIDE = 1280
LOOK_MAX_SIDE = 1024
LOOK_TIMEOUT = 180
LOOK_VERSION = 1
VISION_TOP_DEFAULT = 8
SUGGESTION_POOL = 12

LOOK_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "framing", "people", "costume_readable", "setting", "note"],
    "properties": {
        "kind": {"type": "string", "enum": ["photo", "collage", "poster", "screenshot"]},
        "framing": {"type": "string", "enum": ["full-body", "three-quarter", "half-body", "close-up", "wide"]},
        "people": {"type": "integer", "minimum": 0},
        "costume_readable": {"type": "boolean"},
        "setting": {"type": "string", "enum": ["outdoor", "studio", "indoor", "event-floor"]},
        "note": {"type": "string", "maxLength": 80},
    },
}
LOOK_SYSTEM = (
    "You label one photo for a photographer choosing an Instagram cover. Describe only what is visible "
    "(framing, costume, graphics). Never identify or name a person. Reply with only the JSON object."
)
LOOK_QUESTION = (
    "kind: 'photo' for a finished photo (a small watermark or logo is still a photo); 'poster' when large "
    "title text, an event graphic or an ad dominates; 'collage' for two or more pictures side by side; "
    "'screenshot' for a screen capture. framing of the main subject, by how much of the body is inside the "
    "frame: 'full-body' (head down to the feet or ankles), 'three-quarter' (head down to the thighs or "
    "knees), 'half-body' (the frame ends at the waist, hips or chest), 'close-up' (head and shoulders "
    "only), 'wide' (the person is small in a large scene). people: number of main "
    "subjects. costume_readable: true when the costume and its key prop are clearly shown, not cut off or "
    "hidden. setting: 'outdoor', 'studio' (backdrop or built set with controlled light), 'indoor' (room, cafe, "
    "corridor) or 'event-floor' (convention hall). note: at most 8 words on what would stop a scroll."
)


class LookError(RuntimeError):
    """The Claude look could not be obtained for a photo."""


@dataclass
class CoverScore:
    """Scores and evidence for one photo. ``checks`` holds 0-1 values, or None when not measured."""

    path: str
    width: int = 0
    height: int = 0
    score: float = 0.0
    checks: Dict[str, Optional[float]] = field(default_factory=dict)
    checked_weight: int = 0
    face_gate: float = 1.0
    framing: str = "unknown"
    framing_source: str = "none"      # "vision" | "proxy" | "none"
    kind: str = "not checked"
    subject_faces: int = 0
    face_height_frac: Optional[float] = None
    face_sharpness: Optional[float] = None
    eyes_open: str = "no_face"
    crop_kept: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)
    crop_reason: Dict[str, str] = field(default_factory=dict)
    colour: Dict[str, float] = field(default_factory=dict)
    lighting: str = ""
    measured_light: Optional[float] = None
    setting: str = "not checked"
    look: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CoverSuggestion:
    cover: Optional[str] = None
    slides: List[str] = field(default_factory=list)
    reasons: Dict[str, str] = field(default_factory=dict)
    cover_crop: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Measuring one photo (no network)
# ---------------------------------------------------------------------------

def _proxy(img: np.ndarray, max_side: int) -> np.ndarray:
    """uint8 BGR copy of ``img`` with the long side at most ``max_side``."""
    if img.dtype != np.uint8:
        scale = 257.0 if img.dtype == np.uint16 else 1.0
        img = np.clip(np.rint(img.astype(np.float32) / scale), 0, 255).astype(np.uint8)
    h, w = img.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return img


def _person_mask(img: np.ndarray, detector) -> Optional[np.ndarray]:
    """Person mask for ``img`` or None (segmenter missing, failed, or its all-ones fallback)."""
    if detector is None or not hasattr(detector, "segment_person"):
        return None
    try:
        mask = detector.segment_person(img)
    except Exception as exc:
        _logger.debug("cover hook: person segmentation failed: %s", exc)
        return None
    if mask is None or mask.shape[:2] != img.shape[:2] or float(np.min(mask)) >= 0.5:
        return None
    return np.asarray(mask, dtype=np.float32)


def _head_boxes(faces: Sequence[Box], width: int, height: int) -> List[Tuple[int, int, int, int]]:
    """Face boxes grown to the whole head (wig, ears), as clipped x0, y0, x1, y1."""
    out = []
    for x, y, w, h in faces:
        out.append((max(0, int(x - 0.25 * w)), max(0, int(y - 0.6 * h)),
                    min(width, int(x + 1.25 * w)), min(height, int(y + 1.25 * h))))
    return out


def crop_keep(plan: CropPlan, faces: Sequence[Box], mask: Optional[np.ndarray],
              width: int, height: int) -> Tuple[Optional[float], Optional[float]]:
    """``(head share, person share)`` that ``plan`` keeps; None where there is nothing to measure."""
    x0, y0, x1, y1 = plan.x, plan.y, plan.x + plan.w, plan.y + plan.h
    person_kept: Optional[float] = None
    if mask is not None:
        person = mask > 0.5
        total = int(person.sum())
        if total >= 0.005 * person.size:
            person_kept = float(person[y0:y1, x0:x1].sum()) / total
    head_kept: Optional[float] = None
    heads = _head_boxes(faces, width, height)
    total = sum(max(0, bx1 - bx0) * max(0, by1 - by0) for bx0, by0, bx1, by1 in heads)
    if total > 0:
        inside = sum(max(0, min(bx1, x1) - max(bx0, x0)) * max(0, min(by1, y1) - max(by0, y0))
                     for bx0, by0, bx1, by1 in heads)
        head_kept = inside / total
    return head_kept, person_kept


def _ramp(value: float, low: float, high: float) -> float:
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def framing_from_faces(subject_faces: Sequence[Box], height: int) -> Tuple[str, Optional[float]]:
    """Offline framing proxy: ``(framing, main face height / frame height)``."""
    if not subject_faces:
        return "unknown", None
    frac = max(b[3] for b in subject_faces) / float(max(height, 1))
    if len(subject_faces) >= 2:
        return "pair", frac
    if frac < FRAMING_WIDE_BELOW:
        return "wide", frac
    if frac < FRAMING_FULL_BELOW:
        return "full-body", frac
    if frac < FRAMING_THREE_QUARTER_BELOW:
        return "three-quarter", frac
    if frac < FRAMING_HALF_BELOW:
        return "half-body", frac
    return "close-up", frac


def _eye_state(evidence: Any) -> str:
    """``closed`` / ``wink`` / ``open`` / ``uncertain`` from one face's quality evidence.

    :mod:`retouch.face_quality` reports ``"no"`` when either eye is shut. A wink is a pose, not a
    blink, so only both eyes below its closed band count as closed here.
    """
    if evidence.eyes_open == "yes":
        return "open"
    if evidence.eyes_open != "no":
        return "uncertain"
    details = evidence.measurement_details or {}
    left, right = details.get("left_eye_aperture"), details.get("right_eye_aperture")
    band = details.get("eyes_closed_below")
    if left is None or right is None or band is None:
        return "closed"
    return "closed" if max(left, right) < band else "wink"


def measure_photo(path: Union[str, Path], detector=None, *, image: Optional[np.ndarray] = None) -> CoverScore:
    """Measure the offline checks for one photo. Never raises; a failure is recorded in ``error``."""
    from retouch.face_quality import FaceQualityAnalyzer
    from retouch.image_analyzer import ImageAnalyzer, colour_pop_score

    result = CoverScore(path=str(path))
    try:
        if image is None:
            from retouch.io import imread_exif

            image = imread_exif(path)
        if image is None:
            raise ValueError("could not read image")
        result.height, result.width = image.shape[:2]
        img = _proxy(image, PROXY_MAX_SIDE)
        h, w = img.shape[:2]

        detected = list(detector.detect(img) or []) if detector is not None and getattr(detector, "available", True) else []
        boxes: List[Box] = [tuple(int(round(v)) for v in f.bbox) for f in detected]
        subject = select_subject_faces(boxes)
        # The mask is used without a face too: a profile or back view still has a person to keep in the crop.
        mask = _person_mask(img, detector) if getattr(detector, "available", False) else None
        result.subject_faces = len(subject)

        # Check 2: crop safety.
        from retouch.social_crops import find_subject_top

        subject_top = find_subject_top(mask, boxes) if mask is not None else None
        parts: List[Tuple[float, float]] = []
        for key, weight in CROP_FORMATS:
            plan = plan_crop(w, h, boxes, FORMATS[key], subject_top=subject_top)
            head, person = crop_keep(plan, subject, mask, w, h)
            result.crop_reason[key] = plan.reason
            result.crop_kept[key] = {"head": None if head is None else round(head, 4),
                                     "person": None if person is None else round(person, 4)}
            if head is None and person is None:
                continue
            part = min(1.0 if head is None else _ramp(head, *CROP_HEAD_KEEP),
                       1.0 if person is None else _ramp(person, *CROP_PERSON_KEEP))
            if plan.reason == "faces_wider_than_crop":
                part = min(part, CROP_SPLIT_FACES_CAP)
            parts.append((part, weight))
        result.checks["crop"] = (round(sum(p * wt for p, wt in parts) / sum(wt for _, wt in parts), 4)
                                 if parts else None)
        if not parts:
            result.notes.append("crop not checked: no face or person found")

        # Check 3: colour pop, on the subject when there is a mask.
        result.colour = colour_pop_score(img, mask)
        result.checks["colour"] = result.colour["score"]

        # Check 4: costume framing (offline proxy; a Claude look replaces it).
        framing, frac = framing_from_faces(subject, h)
        result.face_height_frac = None if frac is None else round(frac, 4)
        if framing != "unknown":
            result.framing, result.framing_source = framing, "proxy"
            result.checks["costume"] = FRAMING_SCORES[framing]
        else:
            result.checks["costume"] = None

        # Check 5: light.
        analysis = ImageAnalyzer().analyze(img, face_bboxes=boxes or None)
        result.lighting = analysis.lighting_type
        light = LIGHT_SCORES.get(analysis.lighting_type, 0.6)
        if analysis.key == "low" and analysis.contrast_level > LIGHT_HARD_CONTRAST:
            light *= LIGHT_HARD_FACTOR
            result.notes.append("dark, hard-contrast frame")
        result.measured_light = round(light, 4)
        result.checks["light"] = result.measured_light

        # Check 1 needs a look.
        result.checks["photo"] = None

        # Check 6: face gate evidence (the soft-face part is relative to the shoot; see score_shoot).
        if detected:
            result.eyes_open = "uncertain"
            try:
                evidence = FaceQualityAnalyzer().analyze(img, detected)
            except Exception as exc:  # the gate is optional evidence; the other checks stand
                _logger.debug("cover hook: face quality failed for %s: %s", Path(path).name, exc)
                evidence = []
            if len(evidence) == len(boxes):
                main = max(range(len(boxes)), key=lambda i: boxes[i][2] * boxes[i][3])
                result.face_sharpness = evidence[main].face_sharpness
                states = [_eye_state(e) for e, b in zip(evidence, boxes) if b in subject]
                for state in ("closed", "wink", "open"):
                    if state in states:
                        result.eyes_open = state
                        break
    except Exception as exc:  # one bad file must not stop the shoot
        _logger.warning("cover hook: %s: %s", Path(path).name, exc)
        result.error = str(exc)
    return result


# ---------------------------------------------------------------------------
# The Claude look (checks 1 and 4)
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_look(look: Any) -> bool:
    return (isinstance(look, dict) and look.get("kind") in KIND_SCORES
            and look.get("framing") in LOOK_SCHEMA["properties"]["framing"]["enum"])


def _parse_json(text: str) -> Dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise LookError(f"no JSON answer: {text[:200]}")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise LookError(f"unreadable JSON answer: {exc}") from None


def _look_via_cli(image_path: Path) -> Dict[str, Any]:
    """Ask the local Claude Code CLI (headless) about one image file."""
    exe = shutil.which("claude")
    if exe is None:
        raise LookError("Claude Code (the claude command) is not installed here")
    # Inside a Claude Code session the parent's variables would redirect or bill the call.
    env = {k: v for k, v in os.environ.items()
           if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN") and not k.startswith("CLAUDECODE")}
    prompt = f"Read this image: {image_path}\n\n{LOOK_QUESTION}\nReply with only the JSON object described by the schema."
    cmd = [exe, "-p", prompt, "--output-format", "json", "--json-schema", json.dumps(LOOK_SCHEMA),
           "--model", os.environ.get("RETOUCH_COVER_LOOK_MODEL", "haiku"), "--system-prompt", LOOK_SYSTEM,
           "--tools", "Read", "--allowedTools", "Read", "--add-dir", str(image_path.parent),
           "--no-session-persistence"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=LOOK_TIMEOUT, env=env,
                              cwd=str(image_path.parent))
    except subprocess.TimeoutExpired:
        raise LookError(f"Claude Code took more than {LOOK_TIMEOUT} s") from None
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        detail = (proc.stderr or proc.stdout).strip()[-300:]
        raise LookError(f"Claude Code failed (exit {proc.returncode}): {detail or 'no output'}") from None
    if out.get("is_error"):
        raise LookError(f"Claude Code: {str(out.get('result', 'error'))[:300]}")
    if isinstance(out.get("structured_output"), dict):
        return out["structured_output"]
    return _parse_json(str(out.get("result", "")))


def _look_via_api(image_path: Path, model: str) -> Dict[str, Any]:
    """Ask the Anthropic API about one image (billed per use)."""
    import anthropic

    data = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
    try:
        message = anthropic.Anthropic().messages.create(
            model=model, max_tokens=300, system=LOOK_SYSTEM,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
                {"type": "text", "text": LOOK_QUESTION + "\nReply with only a JSON object with the keys "
                                         "kind, framing, people, costume_readable, setting, note."},
            ]}],
        )
    except Exception as exc:
        raise LookError(f"Anthropic API: {exc}") from None
    return _parse_json("".join(getattr(block, "text", "") for block in message.content))


def look_backend() -> str:
    """``"api"`` only with a key, the SDK and a model name; otherwise ``"claude-code"``."""
    if not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("RETOUCH_COVER_LOOK_MODEL")):
        return "claude-code"
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return "claude-code"
    return "api"


class LookCache:
    """Claude's answers, keyed by the photo's content hash, in ``<cache_dir>/look.json``."""

    def __init__(self, cache_dir: Union[str, Path]):
        self.dir = Path(cache_dir)
        self.path = self.dir / "look.json"
        self.looks: Dict[str, Dict[str, Any]] = {}
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if saved.get("version") == LOOK_VERSION:
                self.looks = {k: v for k, v in saved.get("looks", {}).items() if _valid_look(v)}
        except (OSError, ValueError, AttributeError):
            pass

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": LOOK_VERSION, "looks": self.looks}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        tmp.replace(self.path)


def claude_look(path: Union[str, Path], cache: LookCache, *, refresh: bool = False,
                ask: Optional[Callable[[Path], Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Claude's label for one photo (cached). Raises :class:`LookError` when it cannot be had."""
    from retouch.io import imread_exif

    path = Path(path)
    key = _sha256(path)
    if not refresh and key in cache.looks:
        return cache.looks[key]
    image = imread_exif(path)
    if image is None:
        raise LookError("could not read image")
    cache.dir.mkdir(parents=True, exist_ok=True)
    small = cache.dir / f"{key[:16]}.jpg"
    if not cv2.imwrite(str(small), _proxy(image, LOOK_MAX_SIDE), [cv2.IMWRITE_JPEG_QUALITY, 85]):
        raise LookError(f"could not write {small}")
    if ask is None:
        if look_backend() == "api":
            look = _look_via_api(small, os.environ["RETOUCH_COVER_LOOK_MODEL"])
        else:
            look = _look_via_cli(small)
    else:
        look = ask(small)
    if not _valid_look(look):
        raise LookError(f"unexpected answer: {str(look)[:200]}")
    cache.looks[key] = look
    cache.save()
    return look


def apply_look(result: CoverScore, look: Mapping[str, Any]) -> None:
    """Fill checks 1 and 4 of ``result`` from Claude's label."""
    result.look = dict(look)
    result.kind = str(look["kind"])
    result.checks["photo"] = KIND_SCORES[result.kind]
    framing = str(look["framing"])
    if framing != "wide" and int(look.get("people") or 0) >= 2:
        framing = "pair"
    costume = FRAMING_SCORES[framing]
    if look.get("costume_readable") is False:
        costume *= COSTUME_HIDDEN_FACTOR
        result.notes.append("costume or prop partly hidden")
    result.framing, result.framing_source = framing, "vision"
    result.checks["costume"] = round(costume, 4)
    setting = SETTING_SCORES.get(str(look.get("setting")))
    if setting is not None and result.measured_light is not None:
        result.setting = str(look["setting"])
        result.checks["light"] = round(0.5 * result.measured_light + 0.5 * setting, 4)


# ---------------------------------------------------------------------------
# Scoring and ranking a shoot
# ---------------------------------------------------------------------------

def finalize(result: CoverScore, median_sharpness: Optional[float]) -> CoverScore:
    """Turn the measured checks into the 0-100 score (weights over measured checks, times the face gate)."""
    gate = 1.0
    if result.eyes_open == "closed":
        gate *= FACE_GATE_CLOSED
        result.notes.append("eyes closed")
    elif result.eyes_open == "wink":
        result.notes.append("one eye closed (wink); not demoted")
    if (median_sharpness and result.face_sharpness is not None
            and result.face_sharpness < FACE_SOFT_RATIO * median_sharpness):
        gate *= FACE_GATE_SOFT
        result.notes.append("face softer than the rest of the shoot")
    measured = [(WEIGHTS[name], value) for name, value in result.checks.items()
                if name in WEIGHTS and value is not None]
    result.checked_weight = sum(weight for weight, _ in measured)
    result.face_gate = round(gate, 3)
    result.checks["face"] = result.face_gate if result.eyes_open != "no_face" else None
    if result.error or not measured:
        result.score = 0.0
    else:
        result.score = round(100.0 * gate * sum(weight * value for weight, value in measured)
                             / result.checked_weight, 1)
    return result


def _rank(results: Sequence[CoverScore]) -> List[CoverScore]:
    return sorted(results, key=lambda r: (-r.score, r.path))


def score_shoot(paths: Sequence[Union[str, Path]], *, detector=None, vision: bool = False,
                vision_top: int = VISION_TOP_DEFAULT, cache_dir: Optional[Union[str, Path]] = None,
                refresh: bool = False, ask: Optional[Callable[[Path], Dict[str, Any]]] = None,
                progress: Optional[Callable[[int, int, str], None]] = None) -> List[CoverScore]:
    """Score every photo and return them best first.

    With ``vision`` the best ``vision_top`` photos of the offline ranking get one Claude look each
    (cached under ``cache_dir``); the others keep their offline score.
    """
    results: List[CoverScore] = []
    for i, path in enumerate(paths):
        results.append(measure_photo(path, detector))
        if progress is not None:
            progress(i + 1, len(paths), Path(path).name)
    sharpness = [r.face_sharpness for r in results if r.face_sharpness is not None]
    median = float(np.median(sharpness)) if len(sharpness) >= 3 else None

    if vision:
        if cache_dir is None:
            raise ValueError("vision needs a cache_dir")
        offline = _rank([finalize(_copy(r), median) for r in results if not r.error])
        chosen = {r.path for r in offline[:max(1, vision_top)]}
        cache = LookCache(cache_dir)
        for result in results:
            if result.path not in chosen:
                continue
            try:
                apply_look(result, claude_look(result.path, cache, refresh=refresh, ask=ask))
            except LookError as exc:
                _logger.warning("cover hook: no Claude look for %s: %s", Path(result.path).name, exc)
                result.notes.append(f"Claude look failed: {exc}")
    return _rank([finalize(r, median) for r in results])


def _copy(result: CoverScore) -> CoverScore:
    return CoverScore(**{**asdict(result), "notes": list(result.notes)})


def _is_tight(result: CoverScore) -> bool:
    return result.framing in TIGHT_FRAMINGS


def suggest(ranked: Sequence[CoverScore], *, pool: int = SUGGESTION_POOL,
            same_shot: Optional[Callable[[str, str], bool]] = None) -> CoverSuggestion:
    """Cover plus slides 2 and 3 from a ranked list.

    The cover is the best-scoring photo (looked-at photos only, when any were looked at, and never a
    poster). Slide 2 is the best photo with the other kind of framing (a tight shot after a full-body
    cover, and the reverse); slide 3 is the best one left. A slide is never a near-duplicate of a
    photo already chosen (``same_shot``, by default :mod:`retouch.duplicates`).
    """
    usable = [r for r in ranked if not r.error and r.score > 0 and r.kind not in ("poster", "screenshot")]
    looked = [r for r in usable if r.look is not None]
    candidates = (looked or usable)[:pool]
    out = CoverSuggestion()
    if not candidates:
        return out
    if same_shot is None:
        same_shot = _duplicate_checker()
    cover = candidates[0]
    out.cover = cover.path
    out.reasons[cover.path] = f"cover: highest score ({cover.score:.0f}), {cover.framing}"
    chosen = [cover]

    def fresh(candidate: CoverScore) -> bool:
        return all(not same_shot(candidate.path, other.path) for other in chosen)

    rest = [r for r in candidates[1:]]
    contrast = next((r for r in rest if _is_tight(r) != _is_tight(cover) and fresh(r)), None)
    second = contrast or next((r for r in rest if fresh(r)), None)
    if second is not None:
        chosen.append(second)
        why = ("tight shot after a wider cover" if _is_tight(second) and not _is_tight(cover)
               else "wider shot after a tight cover" if contrast is not None else "next best, different pose")
        out.reasons[second.path] = f"slide 2: {why} ({second.score:.0f}, {second.framing})"
        third = next((r for r in rest if r is not second and fresh(r)), None)
        if third is not None:
            chosen.append(third)
            out.reasons[third.path] = f"slide 3: next best, different pose ({third.score:.0f}, {third.framing})"
    out.slides = [r.path for r in chosen[1:]]
    return out


def _duplicate_checker() -> Callable[[str, str], bool]:
    """``same_shot(a, b)`` backed by :mod:`retouch.duplicates` (signatures cached; errors mean 'different')."""
    from retouch.duplicates import compare_frames, frame_signature

    signatures: Dict[str, Any] = {}

    def signature(path: str):
        if path not in signatures:
            try:
                signatures[path] = frame_signature(path)
            except Exception as exc:
                _logger.debug("cover hook: no duplicate signature for %s: %s", path, exc)
                signatures[path] = None
        return signatures[path]

    def same_shot(a: str, b: str) -> bool:
        sa, sb = signature(a), signature(b)
        if sa is None or sb is None:
            return False
        try:
            return bool(compare_frames(sa, sb).duplicate)
        except Exception as exc:
            _logger.debug("cover hook: duplicate check failed: %s", exc)
            return False

    return same_shot


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def list_photos(folder: Union[str, Path], *, recursive: bool = False) -> List[Path]:
    """Photos in ``folder``, skipping this tool's folder, crops, watermarked copies and review folders."""
    from retouch.duplicates import EXTRAS_DIRNAME, REPORT_DIRNAME, list_images

    skip = (OUTPUT_DIRNAME, REPORT_DIRNAME, EXTRAS_DIRNAME, "social", "picks", "rejected", "watermarked")
    return list_images(folder, recursive=recursive, skip_dirs=skip)


def export_cover_crop(cover: Union[str, Path], out_dir: Union[str, Path], detector=None) -> Optional[Path]:
    """Write the cover's 4:5 and 3:4 crops with the social-crop exporter; returns the 4:5 path or None.

    The 3:4 (1080x1440) crop sits next to it: Instagram shows 3:4 feed posts natively.
    """
    from retouch.social_crops import export_social_crops

    found = None
    for res in export_social_crops(cover, out_dir, [FORMATS["4:5"], FORMATS["3:4"]],
                                   detector=detector, force=True):
        if res.status != "done":
            _logger.warning("cover hook: %s crop failed: %s", res.fmt_key, res.error)
        elif res.fmt_key == "4:5":
            found = res.path
    return found


def write_scores(ranked: Sequence[CoverScore], suggestion: CoverSuggestion, out_dir: Union[str, Path],
                 root: Union[str, Path], *, vision: bool) -> Path:
    """Write ``cover_scores.json`` and ``cover_scores.csv``; returns the JSON path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(root).expanduser().resolve()

    def rel(path: Optional[str]) -> Optional[str]:
        if path is None:
            return None
        try:
            return Path(path).resolve().relative_to(root).as_posix()
        except ValueError:
            return str(path)

    payload = {
        "schema": "retouch.cover_hook",
        "version": SCORE_VERSION,
        "vision": vision,
        "weights": WEIGHTS,
        "weight_reasons": WEIGHT_REASONS,
        "suggestion": {
            "cover": rel(suggestion.cover),
            "slides": [rel(p) for p in suggestion.slides],
            "reasons": {rel(k): v for k, v in suggestion.reasons.items()},
            "cover_crop": rel(suggestion.cover_crop),
        },
        "photos": [{**r.to_dict(), "path": rel(r.path), "rank": i + 1} for i, r in enumerate(ranked)],
    }
    target = out_dir / SCORES_JSON
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / SCORES_CSV).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "file", "score", "photo", "crop", "costume", "colour", "light", "face_gate",
                         "framing", "framing_source", "kind", "eyes_open", "lighting", "notes"])
        for i, r in enumerate(ranked):
            writer.writerow([i + 1, rel(r.path), r.score] + [
                "" if r.checks.get(name) is None else r.checks[name]
                for name in ("photo", "crop", "costume", "colour", "light")
            ] + [r.face_gate, r.framing, r.framing_source, r.kind, r.eyes_open, r.lighting,
                 "; ".join(r.notes + ([r.error] if r.error else []))])
    return target


def update_manifest(manifest_path: Union[str, Path], ranked: Sequence[CoverScore]) -> int:
    """Record scores on assets a shoot-review manifest already has; returns how many were updated.

    Only an asset whose path and content hash both match a scored photo is touched, and only its
    ``culling_evidence["cover_hook"]`` entry. Nothing is created and no decision changes. A later
    shoot scan rebuilds ``culling_evidence`` and drops the entry.
    """
    from retouch.shoot_review import ShootReviewManifest

    target = Path(manifest_path).expanduser()
    if not target.is_file():
        return 0
    manifest = ShootReviewManifest.load(target)
    root = Path(manifest.project_root)
    by_path = {str(Path(r.path).resolve()): (i + 1, r) for i, r in enumerate(ranked) if not r.error}
    updated = 0
    for asset in manifest.assets.values():
        found = by_path.get(str((root / asset.relative_path).resolve()))
        if found is None:
            continue
        rank, result = found
        if _sha256(Path(result.path)) != asset.source_sha256:
            continue
        asset.culling_evidence["cover_hook"] = {
            "version": SCORE_VERSION, "score": result.score, "rank": rank, "framing": result.framing,
            "checks": dict(result.checks), "notes": list(result.notes),
        }
        updated += 1
    if updated:
        manifest.save(target)
    return updated


def format_table(ranked: Sequence[CoverScore], root: Union[str, Path], limit: int = 15) -> str:
    """Plain-text ranking for the terminal."""
    root = Path(root).expanduser().resolve()

    def cell(value: Optional[float]) -> str:
        return "  - " if value is None else f"{value:4.2f}"

    lines = [f"{'#':>2}  {'score':>5}  photo crop  cost colr lght gate  framing        file"]
    for i, r in enumerate(ranked[:limit]):
        try:
            name = Path(r.path).resolve().relative_to(root).as_posix()
        except ValueError:
            name = Path(r.path).name
        c = r.checks
        lines.append(
            f"{i + 1:>2}  {r.score:5.1f}  {cell(c.get('photo'))} {cell(c.get('crop'))} {cell(c.get('costume'))} "
            f"{cell(c.get('colour'))} {cell(c.get('light'))} {r.face_gate:4.2f}  {r.framing:<13}  {name}"
            + (f"  [{'; '.join(r.notes + ([r.error] if r.error else []))}]" if r.notes or r.error else "")
        )
    if len(ranked) > limit:
        lines.append(f"    ... {len(ranked) - limit} more in {SCORES_CSV}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m retouch.cover_hook",
        description=(
            "Rank a shoot's retouched photos as Instagram covers, suggest a cover and slides 2-3, "
            "and export the cover's 4:5 crop. Reads the photos only; writes to FOLDER/cover-hook/."
        ),
    )
    parser.add_argument("input", type=Path, nargs="+", help="Folder of retouched photos, or photo files")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help=f"Output folder (default: FOLDER/{OUTPUT_DIRNAME})")
    parser.add_argument("-r", "--recursive", action="store_true", help="Include subfolders")
    parser.add_argument("--vision", action="store_true",
                        help="One Claude look per top candidate for the photo-not-poster and costume checks "
                             "(local `claude -p`; cached). Without it those checks run offline or are skipped")
    parser.add_argument("--vision-top", type=int, default=VISION_TOP_DEFAULT,
                        help=f"How many top candidates Claude looks at (default: {VISION_TOP_DEFAULT})")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached Claude looks")
    parser.add_argument("--no-crop", action="store_true", help="Do not export the cover's 4:5 crop")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Shoot-review manifest to record scores in (only assets it already lists)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``python -m retouch.cover_hook``. Returns an exit code."""
    args = _build_parser().parse_args(argv)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    inputs = [p.expanduser() for p in args.input]
    missing = [p for p in inputs if not p.exists()]
    if missing:
        print(f"error: not found: {missing[0]}", file=sys.stderr)
        return 1
    files: List[Path] = []
    for item in inputs:
        files.extend(list_photos(item, recursive=args.recursive) if item.is_dir() else [item.resolve()])
    if not files:
        print("error: no photos found", file=sys.stderr)
        return 1
    root = inputs[0] if inputs[0].is_dir() else inputs[0].parent
    out_dir = (args.output.expanduser() if args.output else root / OUTPUT_DIRNAME)

    from retouch.detection import FaceDetector

    detector = FaceDetector(max_faces=10, allow_unavailable=True)
    try:
        if not getattr(detector, "available", True):
            print(f"warning: face detection unavailable ({getattr(detector, 'unavailable_reason', 'unknown')}); "
                  "crop, costume and face checks are skipped", file=sys.stderr)
        total = len(files)

        def progress(done: int, count: int, name: str) -> None:
            if done == count or done % 10 == 0:
                print(f"  measured {done}/{count}", flush=True)

        print(f"Scoring {total} photos" + (f", Claude looks at the top {args.vision_top}" if args.vision else ""))
        ranked = score_shoot(files, detector=detector, vision=args.vision, vision_top=args.vision_top,
                             cache_dir=out_dir / ".cache", refresh=args.refresh, progress=progress)
        suggestion = suggest(ranked)
        if suggestion.cover and not args.no_crop:
            crop = export_cover_crop(suggestion.cover, out_dir, detector)
            suggestion.cover_crop = None if crop is None else str(crop)
    finally:
        if hasattr(detector, "close"):
            detector.close()

    scores = write_scores(ranked, suggestion, out_dir, root, vision=args.vision)
    print(format_table(ranked, root))
    if not args.vision:
        print("Photo-not-poster was not checked and costume framing is a face-size estimate; add --vision.")
    for path in [suggestion.cover] + suggestion.slides:
        if path:
            print(f"  {suggestion.reasons[path]}: {Path(path).name}")
    if suggestion.cover_crop:
        print(f"Cover 4:5 crop → {suggestion.cover_crop}")
    print(f"Scores → {scores}")
    if args.manifest is not None:
        print(f"Manifest: {update_manifest(args.manifest, ranked)} assets updated in {args.manifest}")
    failed = sum(1 for r in ranked if r.error)
    if failed:
        print(f"{failed} photos could not be read", file=sys.stderr)
    return 0 if failed < len(ranked) else 1


if __name__ == "__main__":
    sys.exit(main())

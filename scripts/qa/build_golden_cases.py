"""Build the tests/golden/ hard-case inputs from public test photos.

The golden QA harness (tests/test_golden_qa.py) needs small, committable
images of the situations that broke retouching on real shoots: a background
face that is not a person, a hair clip next to the face, a red prop touching
the skin, darker and lighter exposures, a painted face. Alex's shoot photos
can't go in a public repo, so each case is either a public photo as is
("real") or a public photo with the hard part planted on it ("simulated").

Base photos come from MediaPipe's public test-asset bucket (the same Google
storage host CI already downloads the models from), pinned by sha256:

- portrait.jpg: official White House portrait (US government work, public domain)
- mozart_square.jpg: 18th-century oil portrait (public domain)
- pose.jpg, cats_and_dogs.jpg: MediaPipe test data

Usage (writes tests/golden/inputs/*.jpg, tests/golden/masks/*.png and
updates tests/golden/cases.json; references are made separately by
``scripts/qa/golden_qa.py --approve``):

    .venv/bin/python scripts/qa/build_golden_cases.py [--cache DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GOLDEN = ROOT / "tests" / "golden"
BASE_URL = "https://storage.googleapis.com/mediapipe-assets/"
BASES = {
    "portrait.jpg": "a6f11efaa834706db23f275b6115058fa87fc7f14362681e6abe14e82749de3e",
    "mozart_square.jpg": "4feb4dadc5d6f853ade57b8c9d4c9a1f5ececd6469616c8e505f9a14823392b6",
    "pose.jpg": "c8a830ed683c0276d713dd5aeda28f415f10cd6291972084a40d0d8b934ed62b",
    "cats_and_dogs.jpg": "a2eaa7ad3a1aae4e623dd362a5f737e8a88d122597ecd1a02b3e1444db56df9c",
}
LONG_SIDE = 512


def _fetch(name: str, cache: Path) -> np.ndarray:
    path = cache / name
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(BASE_URL + name, timeout=60) as resp:
            path.write_bytes(resp.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != BASES[name]:
        raise SystemExit(f"{name}: sha256 {digest} != pinned {BASES[name]}")
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _fit(img: np.ndarray, long_side: int = LONG_SIDE) -> np.ndarray:
    """Resize so the long side is ``long_side`` and both sides divide by 8."""
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    nh, nw = max(8, int(round(h * s / 8)) * 8), max(8, int(round(w * s / 8)) * 8)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def _linear_gain(img: np.ndarray, gain: float) -> np.ndarray:
    """Exposure change in linear light (simulated darker/brighter capture)."""
    f = img.astype(np.float32) / 255.0
    lin = np.where(f <= 0.04045, f / 12.92, ((f + 0.055) / 1.055) ** 2.4)
    lin = np.clip(lin * gain, 0.0, 1.0)
    out = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _face_box(img: np.ndarray):
    from retouch.detection import FaceDetector

    faces = FaceDetector().detect(img)
    if not faces:
        raise SystemExit("builder: no face found on a base photo")
    return max(faces, key=lambda f: f.bbox[2] * f.bbox[3]).bbox


def _plant(img: np.ndarray, mask: np.ndarray, color_bgr, shade: float = 0.25):
    """Composite a flat-ish plastic object with a soft highlight and edge AA."""
    m = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 0.8)
    h, w = mask.shape
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    # Lit from above: top of the object brighter, bottom darker.
    tone = 1.0 + shade * (0.5 - yy)
    obj = np.clip(np.asarray(color_bgr, np.float32)[None, None, :] * tone[..., None], 0, 255)
    rng = np.random.default_rng(3)
    obj = obj + rng.normal(0, 2.0, obj.shape).astype(np.float32)
    out = img.astype(np.float32) * (1 - m[..., None]) + obj * m[..., None]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _hair_clip(img: np.ndarray, box):
    """A pink plastic barrette at the temple, half on hair, half on skin."""
    x, y, w, h = box
    mask = np.zeros(img.shape[:2], np.uint8)
    cx, cy = int(x + 0.86 * w), int(y + 0.10 * h)
    ax, ay = max(6, int(0.17 * w)), max(3, int(0.045 * h))
    cv2.ellipse(mask, (cx, cy), (ax, ay), -30, 0, 360, 1, -1)
    out = _plant(img, mask, (170, 90, 235))  # pink (BGR)
    # Two small rhinestones on it: bright, near-white.
    for t in (-0.4, 0.4):
        px = int(cx + t * ax * np.cos(np.deg2rad(-30)))
        py = int(cy + t * ax * np.sin(np.deg2rad(-30)))
        cv2.circle(out, (px, py), max(1, ay // 3), (245, 245, 250), -1)
    return out, mask


def _red_prop(img: np.ndarray, box):
    """A red fabric rose held against the jaw, plus faint red cheek blotches."""
    x, y, w, h = box
    mask = np.zeros(img.shape[:2], np.uint8)
    cx, cy = int(x + 0.18 * w), int(y + 0.92 * h)
    r = max(6, int(0.16 * w))
    cv2.circle(mask, (cx, cy), r, 1, -1)
    out = _plant(img, mask, (40, 30, 200), shade=0.5)
    # Petal folds: darker arcs inside the rose.
    for k in range(3):
        cv2.ellipse(out, (cx, cy), (int(r * (0.3 + 0.2 * k)),) * 2, 30 * k, 0, 220,
                    (25, 20, 140), 1)
    # Planted blotches on the opposite cheek (the op should even these).
    f = out.astype(np.float32)
    yy, xx = np.mgrid[: img.shape[0], : img.shape[1]].astype(np.float32)
    for dx, dy in ((0.70, 0.62), (0.62, 0.70), (0.76, 0.72)):
        bx, by = x + dx * w, y + dy * h
        g = np.exp(-(((xx - bx) ** 2 + (yy - by) ** 2) / (2 * (0.035 * w) ** 2)))
        f[..., 2] += 22 * g
        f[..., 1] -= 8 * g
    return np.clip(f, 0, 255).astype(np.uint8), mask


def _glasses(img: np.ndarray, box):
    """Thin dark glasses frames resting on the face (an object ON the skin)."""
    x, y, w, h = box
    mask = np.zeros(img.shape[:2], np.uint8)
    t = max(2, int(0.025 * w))
    ey = int(y + 0.40 * h)
    for ex in (0.30, 0.70):
        cv2.ellipse(mask, (int(x + ex * w), ey), (int(0.15 * w), int(0.075 * h)), 0, 0, 360, 1, t)
    cv2.line(mask, (int(x + 0.45 * w), ey), (int(x + 0.55 * w), ey), 1, t)
    # Temple arms running back to the ears, past the face outline.
    cv2.line(mask, (int(x + 0.15 * w), ey), (int(x - 0.08 * w), int(ey - 0.02 * h)), 1, t)
    cv2.line(mask, (int(x + 0.85 * w), ey), (int(x + 1.08 * w), int(ey - 0.02 * h)), 1, t)
    return _plant(img, mask, (35, 30, 30), shade=0.1), mask


def _poster(img: np.ndarray, mozart: np.ndarray, box):
    """A framed portrait poster on the wall behind the subject."""
    x, y, w, h = box
    side = int(0.85 * w)
    crop = mozart[int(0.02 * mozart.shape[0]): int(0.62 * mozart.shape[0]),
                  int(0.18 * mozart.shape[1]): int(0.78 * mozart.shape[1])]
    crop = cv2.resize(crop, (side, int(side * crop.shape[0] / crop.shape[1])), interpolation=cv2.INTER_AREA)
    out = img.copy()
    px, py = 8, 8
    ph, pw = min(crop.shape[0], out.shape[0] // 2), min(crop.shape[1], max(1, x - 16))
    out[py - 4: py + ph + 4, px - 4: px + pw + 4] = (40, 120, 170)  # gilt frame
    out[py: py + ph, px: px + pw] = crop[:ph, :pw]
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[py - 4: py + ph + 4, px - 4: px + pw + 4] = 1
    return out, mask


def build(cache: Path) -> dict:
    # Head-and-shoulders crop so the face is ~200 px wide at 512 px.
    src = _fetch("portrait.jpg", cache)
    sh, sw = src.shape[:2]
    portrait = _fit(src[: int(0.52 * sh), int(0.06 * sw): int(0.94 * sw)])
    mozart = _fit(_fetch("mozart_square.jpg", cache))
    pose = _fit(_fetch("pose.jpg", cache))
    # 768 px: the size at which FaceMesh finds both dog faces.
    pets = _fit(_fetch("cats_and_dogs.jpg", cache), 768)
    pbox = _face_box(portrait)
    mbox = _face_box(mozart)

    cases = {}

    def add(name, img, persons, source, why, mask=None, kind="real"):
        (GOLDEN / "inputs").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(GOLDEN / "inputs" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        entry = {"persons": persons, "kind": kind, "source": source, "why": why}
        if mask is not None:
            (GOLDEN / "masks").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(GOLDEN / "masks" / f"{name}.png"), (mask > 0).astype(np.uint8) * 255)
            entry["objects_mask"] = f"masks/{name}.png"
        cases[name] = entry

    add("portrait_studio", portrait, 1, "portrait.jpg", "Baseline studio portrait; red-striped flag and red curtains in the frame.")
    img, m = _hair_clip(portrait, pbox)
    add("hair_clip_temple", img, 1, "portrait.jpg", "Pink plastic clip at the temple, half on hair and half on skin (the albedo_even hair-clip frame).", m, "simulated")
    img, m = _red_prop(portrait, pbox)
    add("red_prop_jaw", img, 1, "portrait.jpg", "Red rose against the jaw plus red cheek blotches (the redness_even leak case).", m, "simulated")
    img, m = _glasses(portrait, pbox)
    add("glasses_frames", img, 1, "portrait.jpg", "Dark frames resting on the skin: a non-skin object inside the skin area.", m, "simulated")
    cases["glasses_frames"]["object_gap"] = (
        "Dark, neutral frames have the colour of shadowed skin, so the skin mask "
        "keeps the rims that rest on the face; 2-14% of the frame changes under "
        "most recipes. Cutting dark neutral pixels also cut real creases and beard "
        "shadow on darker skin, so it is left as a known gap.")
    img, m = _poster(portrait, mozart, pbox)
    add("background_poster_face", img, 1, "portrait.jpg + mozart_square.jpg", "Framed portrait poster on the wall behind the subject (background face, not a person).", m, "simulated")
    add("pet_false_face", pets, 0, "cats_and_dogs.jpg", "Cats and dogs, no person: FaceMesh finds the dogs' faces and the selfie and pose models call them people.")
    cases["pet_false_face"]["known_gap"] = (
        "No model available offline separates a dog's face from a person's: the selfie "
        "segmenter, the multiclass face-skin class and the pose landmarker all accept the "
        "poodle. Expected to fail until a real person/animal classifier is added.")
    add("darker_exposure", _linear_gain(portrait, 0.45), 1, "portrait.jpg", "Portrait at 0.45x linear exposure (tone invariance, darker capture).", kind="simulated")
    add("brighter_exposure", _linear_gain(portrait, 1.5), 1, "portrait.jpg", "Portrait at 1.5x linear exposure (lighter capture, near-clipped highlights).", kind="simulated")
    add("painted_pale_face", mozart, 1, "mozart_square.jpg", "Painted, powder-pale face with a white wig and a blue ribbon.")
    img, m = _hair_clip(mozart, mbox)
    add("white_wig_clip", img, 1, "mozart_square.jpg", "Pink clip on a white wig beside a pale face.", m, "simulated")
    add("small_face_full_body", pose, 1, "pose.jpg", "Full-body shot, small face, darker skin, bright beach.")
    two = np.hstack([portrait, cv2.flip(_linear_gain(portrait, 0.6), 1)])
    add("two_people", _fit(two), 2, "portrait.jpg", "Two subjects side by side (one darker): both must be processed.", kind="simulated")

    (GOLDEN / "cases.json").write_text(json.dumps(cases, indent=2, sort_keys=True) + "\n")
    return cases


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "retouch" / "golden-bases")
    args = ap.parse_args()
    out = build(args.cache)
    print(f"wrote {len(out)} cases to {GOLDEN}")

"""Bounded, opt-in snapshots of individual effects, separate from export pixels."""
from pathlib import Path

import cv2
import numpy as np

EFFECT_LABELS = {
    "body_skin_even": "Body Skin Evening",
    "neck_tone_match": "Neck Tone Match",
    "costume_clarity": "Costume Clarity",
    "hair_remove_flyaways": "Stray Hair Cleanup",
}
MAX_DIM = 800


def _strength(ctx, name):
    value = float(getattr(ctx, name, 0) or 0)
    if name == "hair_remove_flyaways":
        value = max(value, float(getattr(ctx, "flyaway_cleanup", 0) or 0))
    return value


def initialize(ctx, enabled):
    ctx._collect_effect_previews = bool(enabled)
    ctx._effect_previews = {}
    if enabled:
        for name in EFFECT_LABELS:
            active = _strength(ctx, name) > 0
            ctx._effect_previews[name] = {
                "status": "Effect not run. Render with a detected face to inspect it." if active else "Off — raise this effect's slider and render again.",
            }


def skipped(ctx, name, message):
    if (getattr(ctx, "_collect_effect_previews", False)
            and _strength(ctx, name) > 0):
        ctx._effect_previews[name] = {"status": message}


def capture(ctx, name, before, after, empty_message, image_scale=1.0):
    """Store stage-local before/after and actual changed-pixel coverage.

    Images are BGR in [0,image_scale]. Coverage is measured before resizing: small
    changes cannot disappear through averaging or a final 8-bit export.
    """
    if not getattr(ctx, "_collect_effect_previews", False):
        return
    if after is before:
        skipped(ctx, name, empty_message)
        return
    changed = np.zeros(before.shape[:2], dtype=bool)
    for y in range(0, before.shape[0], 256):
        changed[y:y + 256] = np.max(
            np.abs(after[y:y + 256] - before[y:y + 256]), axis=2,
        ) > 1e-5 * image_scale
    if not changed.any():
        skipped(ctx, name, "No visible correction needed on the selected region.")
        return
    h, w = before.shape[:2]
    scale = min(1.0, MAX_DIM / max(h, w))
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    def small(image):
        if scale < 1:
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        return np.clip(image * (255.0 / image_scale) + 0.5, 0, 255).astype(np.uint8)
    b, a = small(before), small(after)
    mask = changed.astype(np.float32)
    if scale < 1:
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)
    overlay = a.astype(np.float32)
    alpha = (mask * 0.45)[..., None]
    overlay = overlay * (1 - alpha) + np.array([40, 210, 255], np.float32) * alpha
    share = float(changed.mean()) * 100
    ctx._effect_previews[name] = {
        "before": b, "after": a,
        "overlay": np.clip(overlay + 0.5, 0, 255).astype(np.uint8),
        "status": f"Changed {share:.2f}% of pixels at this effect's stage. Gold marks affected pixels. Before/after isolates this effect, before later grading; preview limited to 800 px.",
    }


def save_previews(previews, directory):
    """Persist bounded images in a caller-owned GUI request workspace."""
    directory = Path(directory)
    saved = {}
    for name in EFFECT_LABELS:
        item = previews.get(name, {})
        record = {"status": item.get("status", "No effect preview available. Render again.")}
        for view in ("before", "after", "overlay"):
            if view in item:
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"{name}-{view}.png"
                if not cv2.imwrite(str(path), item[view]):
                    raise OSError(f"Cannot save effect preview: {path}")
                record[view] = str(path)
        saved[name] = record
    return saved

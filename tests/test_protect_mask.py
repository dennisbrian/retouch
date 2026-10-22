"""Protect / Restore brush: one user protection mask for every retouch op.

Unit tests for ``retouch/protect_mask.py`` plus engine tests on the public
golden portrait (tests/golden/inputs, real photo; the planted marks and the
darker exposure are simulated). A planted red-brown spot that Spot Healing
and legacy Blemish Removal take out must survive when it is protected, on
the native, draft-proxy and fast-preview paths, single- and multi-face
(worker pool), while the colour grade still reaches it.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from retouch.protect_mask import (
    coerce_mask,
    decode_mask,
    encode_mask,
    fit_to_source,
    frame_mask,
    keep_protected,
    merge_masks,
    summary,
    warp_mask,
)

GOLDEN = Path(__file__).parent / "golden" / "inputs"


# ---------------------------------------------------------------------------
# Unit
# ---------------------------------------------------------------------------


def test_coerce_accepts_common_forms_and_empty_is_none():
    m = np.zeros((20, 30), np.uint8)
    assert coerce_mask(m) is None
    m[5:10, 5:10] = 255
    out = coerce_mask(m)
    assert out.dtype == np.float32 and out.max() == 1.0
    rgba = np.zeros((20, 30, 4), np.uint8)
    rgba[5:10, 5:10, 3] = 255
    assert np.array_equal(coerce_mask(rgba) > 0.5, out > 0.5)
    assert np.array_equal(coerce_mask(m.astype(bool)), out)
    # float mask one ulp above 1.0 stays a 0-1 mask (epsilon-safe test)
    f = out * np.float32(1.0000001)
    assert np.allclose(coerce_mask(f), out)


def test_encode_decode_roundtrip_and_gui_state():
    m = np.zeros((40, 50), np.float32)
    m[10:20, 15:30] = 1.0
    uri = encode_mask(m)
    assert uri.startswith("data:image/png;base64,")
    assert np.array_equal(coerce_mask(uri), m)
    assert np.array_equal(coerce_mask({"mask": uri, "source": "x.jpg"}), m)
    assert decode_mask(uri).shape == (40, 50)
    assert encode_mask(np.zeros((4, 4))) is None


def test_fit_to_source_resizes_same_framing_and_refuses_other():
    m = np.zeros((100, 150), np.uint8)
    m[40:60, 60:90] = 255
    out = fit_to_source(m, (400, 600))
    assert out.shape == (400, 600)
    assert out[200, 300] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="framing"):
        fit_to_source(m, (400, 400))


def test_merge_is_union():
    a = np.zeros((10, 10), np.float32); a[0:3] = 1
    b = np.zeros((10, 10), np.float32); b[7:] = 1
    u = merge_masks(a, None, b)
    assert u.sum() == 60
    assert merge_masks(None, np.zeros((3, 3))) is None


def test_warp_follows_displacement():
    m = np.zeros((50, 50), np.float32)
    m[20:30, 20:30] = 1
    disp = np.zeros((50, 50, 2), np.float32)
    disp[..., 0] = 5.0  # content moved 5 px right
    w = warp_mask(m, disp)
    assert w[25, 27] == pytest.approx(1.0) and w[25, 22] == pytest.approx(0.0)


def _textured(h=120, w=120, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(140 + rng.normal(0, 8, (h, w, 3)), 0, 255).astype(np.float32)


def test_keep_protected_restores_detail_carries_broad_tone_outside_untouched():
    before = _textured()
    before[55:65, 55:65] -= 50  # a "mole"
    after = cv2.GaussianBlur(before, (0, 0), 3) + 20.0  # smoothing + whitening
    protect = np.zeros(before.shape[:2], np.float32)
    protect[50:70, 50:70] = 1
    out = keep_protected(before, after, protect, tone_sigma_px=6.0)
    outside = protect == 0
    assert np.array_equal(out[outside], after[outside])
    inside = out[52:68, 52:68] - before[52:68, 52:68]
    # broad tone (+20) carried, detail (texture and mole contrast) kept
    assert abs(float(inside.mean()) - 20.0) < 3.0
    mole = out[57:63, 57:63].mean() - out[52:55, 52:68].mean()
    orig = before[57:63, 57:63].mean() - before[52:55, 52:68].mean()
    assert mole == pytest.approx(orig, abs=4.0)


def test_keep_protected_none_is_identity_and_uint8_stays_uint8():
    before = _textured().astype(np.uint8)
    after = (before // 2).astype(np.uint8)
    assert keep_protected(before, after, None, 5.0) is after
    p = np.zeros(before.shape[:2], np.float32); p[10:20, 10:20] = 1
    out = keep_protected(before, after, p, 5.0, feather_px=1.0)
    assert out.dtype == np.uint8


def test_frame_mask_resamples_per_frame_and_caches():
    class Ctx:
        pass
    ctx = Ctx()
    src = np.zeros((400, 600), np.float32); src[100:200, 100:200] = 1
    ctx._protect_source = src
    small = frame_mask(ctx, (200, 300))
    assert small.shape == (200, 300) and small[75, 75] == pytest.approx(1.0)
    assert frame_mask(ctx, (200, 300)) is small
    assert summary(src)["painted"] and not summary(None)["painted"]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    from retouch.engine import RetouchEngine
    return RetouchEngine()


def _load(name, scale=1.0):
    img = cv2.imread(str(GOLDEN / f"{name}.jpg"))
    assert img is not None
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return img


def _plant(engine, img, face_index=0):
    """Plant a small red-brown spot (simulated) on a cheek; return geometry."""
    faces = sorted(engine._detector.detect(img), key=lambda f: f.bbox[0])
    face = faces[face_index]
    lm = face.landmarks.landmark
    h, w = img.shape[:2]
    cx, cy = int(lm[205].x * w), int(lm[205].y * h)
    r = max(2, int(0.018 * face.bbox[2]))
    yy, xx = np.mgrid[:h, :w]
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (0.7 * r) ** 2))[..., None]
    tint = np.array([0.62, 0.55, 0.82], np.float32)
    planted = np.clip(img.astype(np.float32) * (1 - g + g * tint), 0, 255).astype(np.uint8)
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (cx, cy), 2 * r, 255, -1)
    return planted, mask, (cx, cy, r)


def _spot_contrast(img, geom):
    cx, cy, r = geom
    lab = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2Lab).astype(np.float32)
    h, w = lab.shape[:2]
    core = lab[cy - r // 2:cy + r // 2 + 1, cx - r // 2:cx + r // 2 + 1].reshape(-1, 3).mean(0)
    yy, xx = np.ogrid[:h, :w]
    d = np.hypot(xx - cx, yy - cy)
    ring = lab[(d > 2.2 * r) & (d < 3.5 * r)].mean(0)
    diff = core - ring
    return float(abs(diff[0]) + np.hypot(diff[1], diff[2]))


_HEAL = dict(recipe="natural", spot_heal=100, blemish=100)


@pytest.mark.parametrize(
    "name,scale,kwargs",
    [
        ("portrait_studio", 1.0, {}),                                   # native
        ("portrait_studio", 2.4, {"fast": True}),                       # 800 px fast preview
        ("portrait_studio", 5.5, {"quality": "draft"}),                 # 2048 px draft proxy
        ("portrait_studio", 5.5, {"quality": "full"}),                  # proxy detect, native faces
        ("darker_exposure", 1.0, {}),                                   # 0.45x exposure (simulated)
    ],
    ids=["native", "fast-preview", "draft-proxy", "full-proxy", "darker-skin"],
)
def test_protected_spot_survives_heal(engine, name, scale, kwargs):
    img = _load(name, scale)
    planted, mask, geom = _plant(engine, img)
    c0 = _spot_contrast(planted, geom)
    healed = engine.process(planted.copy(), **_HEAL, **kwargs)
    kept = engine.process(planted.copy(), protect_mask=mask, **_HEAL, **kwargs)
    assert _spot_contrast(healed, geom) < 0.25 * c0, "planted spot should be healed without a mask"
    assert _spot_contrast(kept, geom) > 0.8 * c0, "protected spot must survive"
    notes = kept.runtime_diagnostics["face_notes"][0]["notes"]
    assert any(n["op"] == "protect" for n in notes)
    assert kept.runtime_diagnostics["protection"]["painted"]


def test_mask_painted_on_smaller_copy_is_resized(engine):
    img = _load("portrait_studio", 2.0)
    planted, mask, geom = _plant(engine, img)
    small = cv2.resize(mask, (mask.shape[1] // 2, mask.shape[0] // 2), interpolation=cv2.INTER_AREA)
    kept = engine.process(planted.copy(), protect_mask=small, **_HEAL)
    assert _spot_contrast(kept, geom) > 0.8 * _spot_contrast(planted, geom)


def test_no_mask_or_empty_mask_is_bit_identical(engine):
    img = _load("portrait_studio")
    a = np.asarray(engine.process(img.copy(), **_HEAL))
    b = np.asarray(engine.process(img.copy(), protect_mask=np.zeros(img.shape[:2], np.uint8), **_HEAL))
    assert np.array_equal(a, b)


def test_effect_stays_local_and_grade_still_applies(engine):
    img = _load("portrait_studio")
    planted, mask, geom = _plant(engine, img)
    cx, cy, r = geom
    kw = dict(_HEAL, brightness=40)
    free = np.asarray(engine.process(planted.copy(), **kw)).astype(np.int16)
    kept = np.asarray(engine.process(planted.copy(), protect_mask=mask, **kw)).astype(np.int16)
    far = cv2.dilate(mask, np.ones((12 * r + 1, 12 * r + 1), np.uint8)) == 0
    # Away from the mask only rounding-level drift: the kept spot stays in
    # the face-wide statistics later skin ops take (measured: a few hundred
    # pixels at 1 level, a handful at 2-5; the exact handful differs between
    # OpenCV builds, macOS arm64 reached 5). A leak of the restore would move
    # whole areas by tens of levels.
    far_diff = np.abs(free - kept).max(axis=2)[far]
    assert np.percentile(far_diff, 99.5) <= 1
    assert (far_diff > 2).mean() < 1e-4 and far_diff.max() <= 8
    # Global brightness reaches the protected spot like the skin around it.
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    d = np.hypot(xx - cx, yy - cy)
    core, ring = d < r * 0.6, (d > 2.5 * r) & (d < 3.5 * r)
    lum = lambda x: cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2Lab)[..., 0].astype(np.float32)
    base_l, kept_l = lum(planted), lum(kept)
    lift_core = float((kept_l - base_l)[core].mean())
    lift_ring = float((kept_l - base_l)[ring].mean())
    assert lift_core > 0.5 * lift_ring > 0


def test_multi_face_pool_honours_mask(engine):
    """Two faces go through the FaceProcessorPool, which gets ROI crops only."""
    img = _load("two_people", 3.0)
    planted, mask, geom = _plant(engine, img, face_index=0)
    healed = engine.process(planted.copy(), **_HEAL)
    kept = engine.process(planted.copy(), protect_mask=mask, **_HEAL)
    assert healed.face_count >= 2
    c0 = _spot_contrast(planted, geom)
    assert _spot_contrast(healed, geom) < 0.5 * c0
    assert _spot_contrast(kept, geom) > 0.8 * c0
    protected = [
        e for e in kept.runtime_diagnostics["face_notes"]
        if any(n["op"] == "protect" for n in e["notes"])
    ]
    assert len(protected) == 1


def test_aspect_mismatch_is_refused(engine):
    img = _load("portrait_studio")
    with pytest.raises(ValueError, match="framing"):
        engine.process(img, protect_mask=np.full((100, 100), 255, np.uint8), recipe="natural")


def test_cli_loader_file_folder_and_data_uri(tmp_path):
    from cli import _with_protect_mask

    m = np.zeros((30, 40), np.uint8); m[5:10, 5:10] = 255
    cv2.imwrite(str(tmp_path / "IMG_1.png"), m)
    out = _with_protect_mask({"protect_mask": str(tmp_path / "IMG_1.png"), "recipe": "natural"}, "x/IMG_9.jpg")
    assert out["protect_mask"].shape == (30, 40) and out["recipe"] == "natural"
    out = _with_protect_mask({"protect_mask": str(tmp_path)}, "a/IMG_1.jpg")
    assert out["protect_mask"].shape == (30, 40)
    assert "protect_mask" not in _with_protect_mask({"protect_mask": str(tmp_path)}, "a/IMG_2.jpg")
    uri = encode_mask(m)
    assert _with_protect_mask({"protect_mask": uri}, "a.jpg")["protect_mask"] == uri


def test_edit_report_records_protection_and_skips():
    from retouch.edit_report import build_edit_report

    diag = {
        "protection": {"painted": True, "coverage_pct": 0.4, "frame": [10, 10]},
        "face_notes": [{"face": 0, "anchor_bbox_norm": [0.1, 0.1, 0.2, 0.2], "recipe": "natural",
                        "notes": [{"op": "eye_edits", "action": "skipped", "reason": "left-hand eye: eye closed"}]}],
    }
    report = build_edit_report({"smooth": 30}, face_count=1, diagnostics=diag)
    assert report["protection"]["painted"] and "grade" in report["protection"]["note"]
    assert report["skipped"]["caption"] == ["Face 0: eye edits skipped, left-hand eye: eye closed"]
    assert "protection" not in build_edit_report({"smooth": 30}, face_count=1)

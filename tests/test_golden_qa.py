"""Golden hard-case QA: faces, skin leaks, objects in the face, references.

Cases and their stories are in tests/golden/README.md; the metrics and
thresholds are in tests/golden_qa.py. CI runs a sample that covers each
check on the cases built for it; ``scripts/qa/golden_qa.py`` runs every
curated recipe on every case, and ``RETOUCH_GOLDEN_FULL=1`` widens the
per-op matrix here to every skin op on every face case.

Refresh references (after looking at the renders) with:
    .venv/bin/python scripts/qa/golden_qa.py --approve \\
        --recipes natural,cosplay_clear_v1,studio_porcelain_clear_v1
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from tests import golden_qa as gq

CASES = gq.load_cases()
FULL = os.environ.get("RETOUCH_GOLDEN_FULL", "").strip() not in {"", "0"}

# Ops the curated recipes lean on most, plus the ones the golden cases were
# built for (albedo_even: hair clip; redness_even: red prop; skin_warmth:
# white wig).
_CI_OPS = (
    "smooth", "whiten", "equalize", "blemish", "albedo_even", "redness_even",
    "skin_hue_unify", "skin_chroma_even", "shine_removal", "dodge_burn",
    "shadow_lift", "face_polish", "powder_finish", "skin_warmth",
)
_CI_OP_CASES = ("hair_clip_temple", "red_prop_jaw", "white_wig_clip")

if FULL:
    OP_CASES = tuple(n for n, c in sorted(CASES.items()) if c.persons > 0)
    OPS = tuple(gq.skin_op_names())
else:
    OP_CASES = _CI_OP_CASES
    OPS = tuple(op for op in _CI_OPS if op in gq.skin_op_names())


def _render(engine, img, recipe, **kw):
    np.random.seed(0)  # grain/sparkle use the global RNG
    return engine.process(img, recipe=recipe, **kw)


@pytest.fixture(scope="module")
def engine():
    from retouch.engine import RetouchEngine

    return RetouchEngine()


@pytest.fixture(scope="module")
def natural(engine):
    """Detected-face natural renders, one per case, shared by the tests."""
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = _render(engine, CASES[name].image, "natural")
        return cache[name]

    return get


@pytest.fixture(scope="module")
def op_base(engine, natural):
    """Per case: natural with every skin op off, on the detected faces."""
    cache = {}

    def get(name):
        if name not in cache:
            first = natural(name)
            cache[name] = _render(
                engine, CASES[name].image, "natural", face_contexts=first.face_contexts,
                **gq.skin_ops_off(), **gq.NO_GRAIN,
            )
        return cache[name]

    return get


def test_cases_load():
    assert len(CASES) >= 10
    for case in CASES.values():
        assert case.image is not None, case.name
        h, w = case.image.shape[:2]
        assert max(h, w) <= 1024 and h % 8 == 0 and w % 8 == 0, case.name
        if case.objects_mask:
            assert case.objects is not None and case.objects.shape == (h, w), case.name


@pytest.mark.parametrize("name", sorted(CASES))
def test_faces_retouched_equal_people(natural, name, request):
    case = CASES[name]
    if case.known_gap:
        request.applymarker(pytest.mark.xfail(reason=case.known_gap, strict=True))
    result = natural(name)
    assert int(result.face_count) == case.persons, (
        f"{name}: retouched {result.face_count} faces, photo has {case.persons} people ({case.why})"
    )


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("name", OP_CASES)
def test_skin_op_stays_on_skin(engine, natural, op_base, name, op):
    case = CASES[name]
    first = natural(name)
    assert first.face_contexts, f"{name}: no face to test on"
    kw = {**gq.skin_ops_off(), **gq.NO_GRAIN, op: gq.probe_value(op)}
    on = _render(engine, case.image, "natural", face_contexts=first.face_contexts, **kw)
    off = op_base(name)

    problems = []
    leak, px = gq.skin_leak(on, off, first, include_hair=op in gq.EDITS_HAIR)
    if op not in gq.SPREADS_TO_BODY_SKIN and leak > gq.SKIN_LEAK_MAX:
        problems.append(f"changed {px} px outside the skin ({leak:.1%} of skin area)")
    hair = gq.hair_change(on, off, first)
    if op not in gq.EDITS_HAIR and hair is not None and hair > gq.HAIR_CHANGE_MAX:
        problems.append(f"changed {hair:.1%} of the hair")
    if case.objects is not None:
        obj = gq.object_change(on, off, case.objects, first)
        if obj is not None and obj > gq.OBJECT_CHANGE_MAX:
            if case.object_gap and not problems:
                pytest.xfail(case.object_gap)
            problems.append(f"changed {obj:.1%} of the planted object")
    assert not problems, f"{op}={kw[op]} on {name}: " + "; ".join(problems)


# natural on every case; the two heavier looks on a spread of skin tones
# and exposures (the full sweep script checks all of them).
_REF_SAMPLE = ("portrait_studio", "darker_exposure", "painted_pale_face", "two_people")
REF_PAIRS = [(n, r) for r in gq.REFERENCE_RECIPES for n in sorted(CASES)
             if FULL or r == "natural" or n in _REF_SAMPLE]


@pytest.mark.parametrize("name,recipe", REF_PAIRS)
def test_matches_approved_reference(engine, natural, name, recipe):
    ref_path = gq.reference_path(name, recipe)
    if not ref_path.exists():
        pytest.fail(f"missing approved reference {ref_path.name}; run scripts/qa/golden_qa.py --approve")
    out = natural(name) if recipe == "natural" else _render(engine, CASES[name].image, recipe)
    ref = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    score = gq.ssim(gq.to_u8(out), ref)
    assert score >= gq.SSIM_MIN, f"{name} / {recipe}: SSIM {score:.4f} < {gq.SSIM_MIN} against the approved output"


def test_metrics_catch_planted_leak():
    """The leak and object metrics must fire on a known leak."""
    case = CASES["hair_clip_temple"]
    img = case.image
    obj = case.objects

    class _R:
        skin_mask = np.zeros(img.shape[:2], np.float32)
        face_contexts = []

    r = _R()
    r.skin_mask[-40:, :40] = 1.0  # far from the clip
    off = img.copy()
    on = img.copy()
    on[obj] = 255 - on[obj]
    leak, px = gq.skin_leak(on, off, r)
    assert px == int(obj.sum()) and leak > gq.SKIN_LEAK_MAX
    assert gq.ssim(img, img) == pytest.approx(1.0)
    assert gq.ssim(img, on) < 0.999

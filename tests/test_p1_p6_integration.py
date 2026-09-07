"""End-to-end routing checks for the accepted P1-P6 production leaves.

These tests deliberately use the real Engine/ColorGrader routing methods with
small deterministic canvases.  They do not enable any recipe or invoke face
detection; P4 and P5 remain explicit caller operations.
"""

import numpy as np
import pytest

from retouch.engine import _CoreResult, ProcessingContext, RetouchEngine, build_context
from retouch.grading import ColorGrader
from retouch.params import RECIPES, resolve_recipe
from retouch.self_blend import ANALYTICAL_SELF_BLEND_MODES, apply_self_blend


def _engine_stub():
    engine = RetouchEngine.__new__(RetouchEngine)
    engine._grader = ColorGrader()
    return engine


def _stage(engine, image, context):
    h, w = image.shape[:2]
    zeros = np.zeros((h, w), dtype=np.float32)
    return engine._stage_grade(image, context, zeros, zeros, None)


def test_p4_engine_operation_preserves_bound_and_exposes_diagnostics():
    image = np.full((32, 32, 3), (80, 100, 180), dtype=np.uint8)
    reference = np.full_like(image, (110, 120, 165))
    support = np.ones((32, 32), dtype=np.float32)

    result = RetouchEngine.apply_bounded_makeup_attenuation(
        image, support, reference, support,
        category="blush", strength=1.0, max_delta=5.0,
    )

    assert result.applied and not result.abstained
    assert result.reason == "applied"
    assert result.applied_delta_max <= 5.0 + 1e-6
    assert result.changed_pixels > 0
    assert result.eligible_pixels == image.shape[0] * image.shape[1]

    abstained = RetouchEngine.apply_bounded_makeup_attenuation(
        image, support, reference, support,
        category="lipstick", strength=1.0,
    )
    assert abstained.abstained and not abstained.applied
    assert abstained.reason == "protected_or_unsupported_category"
    assert np.array_equal(abstained.image, image)


@pytest.mark.parametrize("mode", ANALYTICAL_SELF_BLEND_MODES)
def test_p5_all_modes_route_through_engine_global_grading(mode):
    image = np.array(
        [[[0, 32, 64], [96, 128, 160]], [[192, 224, 240], [250, 127, 1]]],
        dtype=np.uint8,
    )
    engine = _engine_stub()
    context = ProcessingContext(
        self_blend_mode=mode,
        self_blend_amount=1.0,
        self_blend_domain="encoded",
    )

    routed = _stage(engine, image, context)
    expected = apply_self_blend(image, mode, amount=1.0, domain="encoded")
    np.testing.assert_array_equal(routed, expected)


@pytest.mark.parametrize("mode", ANALYTICAL_SELF_BLEND_MODES)
def test_p5_all_modes_route_through_color_grader_api(mode):
    image = np.array([[[17, 93, 201], [244, 128, 5]]], dtype=np.uint8)
    grader = ColorGrader()
    routed = grader.grade(
        image,
        {"gamut_compress": True},
        intensity=1.0,
        skip_glows=True,
        skip_post_effects=True,
        self_blend_mode=mode,
        self_blend_amount=1.0,
        self_blend_domain="encoded",
    )
    expected = apply_self_blend(image, mode, amount=1.0, domain="encoded")
    np.testing.assert_array_equal(routed, expected)


def test_p5_engine_opacity_zero_is_exact_identity():
    image = np.array([[[17, 93, 201], [244, 128, 5]]], dtype=np.uint8)
    engine = _engine_stub()
    routed = _stage(
        engine,
        image,
        ProcessingContext(
            self_blend_mode="soft_light",
            self_blend_amount=0.0,
            self_blend_domain="encoded",
        ),
    )
    np.testing.assert_array_equal(routed, image)


def test_p5_engine_domain_selection_routes_encoded_and_linear_distinctly():
    image = np.array([[[32, 96, 192]]], dtype=np.uint8)
    engine = _engine_stub()
    encoded = _stage(
        engine,
        image,
        ProcessingContext(self_blend_mode="screen", self_blend_amount=1.0, self_blend_domain="encoded"),
    )
    linear = _stage(
        engine,
        image,
        ProcessingContext(self_blend_mode="screen", self_blend_amount=1.0, self_blend_domain="linear"),
    )
    np.testing.assert_array_equal(encoded, apply_self_blend(image, "screen", domain="encoded"))
    np.testing.assert_array_equal(linear, apply_self_blend(image, "screen", domain="linear"))
    assert not np.array_equal(encoded, linear)


def test_p5_no_face_fallback_routes_selected_operator():
    image = np.array([[[32, 96, 192], [64, 128, 224]]], dtype=np.uint8)
    engine = _engine_stub()
    context = ProcessingContext(
        self_blend_mode="multiply",
        self_blend_amount=0.5,
        self_blend_domain="encoded",
    )

    routed = engine._no_face_fallback(image, context)
    expected = apply_self_blend(image, "multiply", amount=0.5, domain="encoded")
    np.testing.assert_array_equal(routed, expected)


def test_p5_large_no_face_path_does_not_drop_selected_operator():
    class NoFaceDetector:
        def detect(self, _image):
            return []

        def segment_person(self, image):
            return np.zeros(image.shape[:2], dtype=np.float32)

    image = np.array([[[32, 96, 192], [64, 128, 224]], [[17, 93, 201], [244, 128, 5]]], dtype=np.uint8)
    engine = _engine_stub()
    engine._detector = NoFaceDetector()
    context = ProcessingContext(
        self_blend_mode="screen",
        self_blend_amount=0.5,
        self_blend_domain="linear",
    )

    core = engine._process_native_faces(
        image, image[::2, ::2], 0.5, context, None, {},
    )
    expected = apply_self_blend(image, "screen", amount=0.5, domain="linear")
    assert core.no_face is True
    np.testing.assert_array_equal(core.result, expected)


def test_p5_public_process_arguments_reach_the_operation_route():
    image = np.array([[[32, 96, 192], [64, 128, 224]]], dtype=np.uint8)
    engine = _engine_stub()

    def fake_core(input_image, context, _style_ref, _timings):
        h, w = input_image.shape[:2]
        result = engine._no_face_fallback(input_image, context, None)
        zeros = np.zeros((h, w), dtype=np.float32)
        return _CoreResult(
            result=result,
            acc_skin=zeros,
            acc_skin_hair=zeros,
            acc_lips=zeros,
            acc_sharpen=zeros,
            faces=[],
            person_mask=None,
            no_face=True,
            face_contexts=None,
            qa=[],
            qa_evidence={},
        )

    engine._process_with_proxy = fake_core
    routed = engine.process(
        image,
        recipe="natural",
        self_blend_mode="multiply",
        self_blend_amount=0.5,
        self_blend_domain="encoded",
    )
    expected = apply_self_blend(image, "multiply", amount=0.5, domain="encoded")
    np.testing.assert_array_equal(np.asarray(routed), expected)
    assert routed.params.self_blend_mode == "multiply"
    assert routed.params.self_blend_amount == 0.5


def test_existing_recipe_contexts_do_not_enable_new_opt_ins():
    for recipe_name in RECIPES:
        context = build_context(recipe_name, resolve_recipe(recipe_name), {})
        assert context.clarity_noise_aware is False
        assert context.self_blend_mode is None
        assert context.self_blend_amount is None
        assert context.self_blend_domain == "encoded"


def test_default_global_grade_has_no_self_blend_delta():
    image = np.array([[[12, 64, 128], [200, 220, 240]]], dtype=np.uint8)
    engine = _engine_stub()
    default = _stage(engine, image, ProcessingContext())
    explicit_none = _stage(
        engine,
        image,
        ProcessingContext(self_blend_mode=None, clarity_noise_aware=False),
    )
    np.testing.assert_array_equal(default, explicit_none)

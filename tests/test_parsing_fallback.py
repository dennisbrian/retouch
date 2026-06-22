"""Tests for retouch/parsing.py — FaceParser landmark fallback and helpers."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from retouch.parsing import FaceParser, FaceRegions
from retouch.detection import _Landmark, _LandmarkCompat
from retouch.utils import get_points


class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def _make_landmarks(num=478):
    lm_list = [_Landmark(0.5, 0.5, 0.0) for _ in range(num)]
    np.random.seed(42)
    for i in range(num):
        lm_list[i] = _Landmark(
            float(np.random.uniform(0.1, 0.9)),
            float(np.random.uniform(0.1, 0.9)),
            0.0
        )
    return _LandmarkCompat(lm_list)


@pytest.fixture
def parser():
    return FaceParser()


@pytest.fixture
def img():
    return np.full((200, 200, 3), 128, dtype=np.uint8)


class TestLandmarkFallbackOnly:
    def test_returns_face_regions(self, parser, img):
        landmarks = _make_landmarks()
        result = parser._landmark_fallback_only(landmarks, img, None, 50.0)
        assert isinstance(result, FaceRegions)

    def test_skin_mask_populated(self, parser, img):
        landmarks = _make_landmarks()
        result = parser._landmark_fallback_only(landmarks, img, None, 50.0)
        assert result.skin is not None
        assert result.skin.shape == img.shape[:2]
        assert result.skin.dtype == np.float32
        assert result.skin.max() <= 1.0

    def test_face_oval_populated(self, parser, img):
        landmarks = _make_landmarks()
        result = parser._landmark_fallback_only(landmarks, img, None, 50.0)
        assert result.face_oval is not None
        assert result.face_oval.shape == img.shape[:2]

    def test_with_person_mask(self, parser, img):
        landmarks = _make_landmarks()
        person_mask = np.ones((200, 200), dtype=np.float32)
        person_mask[100:, :] = 0.0
        result = parser._landmark_fallback_only(landmarks, img, person_mask, 50.0)
        assert result.skin is not None

    def test_person_mask_uint8(self, parser, img):
        landmarks = _make_landmarks()
        person_mask = np.ones((200, 200), dtype=np.uint8) * 255
        result = parser._landmark_fallback_only(landmarks, img, person_mask, 50.0)
        assert result.skin is not None

    def test_person_mask_3d(self, parser, img):
        landmarks = _make_landmarks()
        person_mask = np.ones((200, 200, 1), dtype=np.float32)
        result = parser._landmark_fallback_only(landmarks, img, person_mask, 50.0)
        assert result.skin is not None

    def test_all_subregions_populated(self, parser, img):
        landmarks = _make_landmarks()
        result = parser._landmark_fallback_only(landmarks, img, np.ones((200, 200), dtype=np.float32), 50.0)
        assert result.left_eye is not None
        assert result.right_eye is not None
        assert result.lips is not None
        assert result.left_iris is not None
        assert result.right_iris is not None
        assert result.nose is not None
        assert result.left_under_eye is not None
        assert result.right_under_eye is not None
        assert result.forehead is not None

    def test_skin_excludes_eyes_and_lips(self, parser, img):
        landmarks = _make_landmarks()
        result = parser._landmark_fallback_only(landmarks, img, None, 50.0)
        assert result.skin is not None
        if result.left_eye is not None:
            overlap = (result.skin * result.left_eye).max()
            assert overlap < 0.5


class TestAddLandmarkSubregions:
    def test_iris_fallback_on_index_error(self, parser, img):
        regions = FaceRegions()
        regions.skin = np.ones((200, 200), dtype=np.float32)
        bad_landmarks = _LandmarkCompat([_Landmark(0.5, 0.5, 0.0) for _ in range(466)])
        parser._add_landmark_subregions(regions, bad_landmarks, 200, 200, 50.0, 5)
        assert regions.left_iris is not None
        assert regions.right_iris is not None
        assert regions.left_iris.shape == (200, 200)

    def test_nose_bridge_populated(self, parser, img):
        regions = FaceRegions()
        regions.skin = np.ones((200, 200), dtype=np.float32)
        landmarks = _make_landmarks()
        parser._add_landmark_subregions(regions, landmarks, 200, 200, 50.0, 5)
        assert regions.nose_bridge is not None
        assert regions.forehead_center is not None
        assert regions.cheek_highlights_l is not None
        assert regions.cheek_highlights_r is not None
        assert regions.jawline_contour is not None


class TestInternalHelpers:
    def test_mask_returns_float32(self, parser, img):
        landmarks = _make_landmarks()
        from retouch.parsing import FACE_OVAL
        result = parser._mask(landmarks, FACE_OVAL, 200, 200, 5)
        assert result.dtype == np.float32
        assert result.shape == (200, 200)

    def test_circle_mask(self, parser, img):
        landmarks = _make_landmarks()
        result = parser._circle_mask(landmarks, 151, 20, 200, 200, 5)
        assert result.dtype == np.float32
        assert result.shape == (200, 200)
        assert result.max() <= 1.0

    def test_iris_mask(self, parser, img):
        landmarks = _make_landmarks()
        result = parser._iris_mask(landmarks, [468, 469, 470, 471, 472], 200, 200, 50.0)
        assert result.dtype == np.float32
        assert result.shape == (200, 200)

    def test_forehead_mask(self, parser, img):
        landmarks = _make_landmarks()
        result = parser._forehead_mask(landmarks, 200, 200, 5)
        assert result.dtype == np.float32
        assert result.shape == (200, 200)


# ---------------------------------------------------------------------------
# FaceParser.__init__  — ONNX session loading
# ---------------------------------------------------------------------------


def _make_fake_logits(skin_value=10.0, other_value=-10.0, shape=(1, 19, 512, 512)):
    """Build a fake BiSeNet logits tensor.

    Default config: class 1 (skin) wins argmax everywhere, so the parsed
    BiSeNet skin mask is non-empty (>= 0.01 max) and the ONNX path is taken.
    """
    logits = np.full(shape, other_value, dtype=np.float32)
    logits[:, 1, :, :] = skin_value
    return logits


class TestFaceParserInit:
    """Tests for FaceParser.__init__ — ONNX session creation."""

    def test_model_path_attribute_set(self):
        parser = FaceParser()
        assert hasattr(parser, "_model_path")
        assert parser._model_path.endswith("resnet18.onnx")

    def test_session_attribute_initialized(self):
        parser = FaceParser()
        # The session is either a real ort.InferenceSession or None
        # (when the model file is missing or ONNX init failed). Either
        # way the attribute must exist.
        assert hasattr(parser, "_sess")

    def test_session_is_none_when_model_missing(self):
        with patch("os.path.exists", return_value=False):
            parser = FaceParser()
        assert parser._sess is None

    def test_session_is_none_when_onnx_init_fails(self):
        with patch("os.path.exists", return_value=True), \
             patch(
                 "retouch.parsing.ort.InferenceSession",
                 side_effect=Exception("init failed"),
             ):
            parser = FaceParser()
        # Both the preferred-providers attempt and the fallback must fail,
        # leaving _sess as None.
        assert parser._sess is None

    def test_session_set_when_onnx_init_succeeds(self):
        with patch("os.path.exists", return_value=True), \
             patch("retouch.parsing.ort.InferenceSession") as mock_sess_cls:
            mock_sess_cls.return_value = MagicMock(name="fake_ort_session")
            parser = FaceParser()
        assert parser._sess is not None


# ---------------------------------------------------------------------------
# FaceParser.parse()  — ONNX path with a mock session
# ---------------------------------------------------------------------------


def _make_random_landmarks(num=478, seed=7):
    rng = np.random.default_rng(seed)
    lm_list = []
    for i in range(num):
        lm_list.append(_Landmark(
            x=float(rng.uniform(0.2, 0.8)),
            y=float(rng.uniform(0.2, 0.8)),
            z=0.0,
        ))
    return _LandmarkCompat(lm_list)


class TestFaceParserParseOnnx:
    """Tests for FaceParser.parse() when an ONNX session is available."""

    def test_returns_face_regions(self):
        mock_sess = MagicMock()
        mock_sess.run.return_value = [_make_fake_logits()]

        parser = FaceParser()
        parser._sess = mock_sess

        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        bbox = (50, 50, 100, 100)
        landmarks = _make_random_landmarks()

        result = parser.parse(landmarks, img, bbox, None, 50.0)
        assert isinstance(result, FaceRegions)

    def test_calls_session_run_with_correct_input_shape(self):
        mock_sess = MagicMock()
        mock_sess.run.return_value = [_make_fake_logits()]

        parser = FaceParser()
        parser._sess = mock_sess

        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        bbox = (50, 50, 100, 100)
        landmarks = _make_random_landmarks()

        parser.parse(landmarks, img, bbox, None, 50.0)

        assert mock_sess.run.called
        # run() signature: run(None, {input_name: input_array})
        call_args = mock_sess.run.call_args
        # First positional arg: output_names (None = all)
        # Second positional/keyword arg: input feed dict
        feed = call_args.kwargs.get(None) or call_args[0][1]
        # The 'input' key holds the preprocessed crop
        assert "input" in feed
        crop = feed["input"]
        # BiSeNet expects (1, 3, 512, 512) float32
        assert crop.shape == (1, 3, 512, 512)
        assert crop.dtype == np.float32

    def test_skin_mask_populated_from_onnx(self):
        mock_sess = MagicMock()
        mock_sess.run.return_value = [_make_fake_logits()]

        parser = FaceParser()
        parser._sess = mock_sess

        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        bbox = (50, 50, 100, 100)
        landmarks = _make_random_landmarks()

        result = parser.parse(landmarks, img, bbox, None, 50.0)
        assert result.skin is not None
        assert result.skin.shape == (200, 200)
        assert result.skin.dtype == np.float32
        # When skin class wins argmax everywhere, the skin mask max must
        # be high (close to 1.0) after feathering.
        assert result.skin.max() > 0.5

    def test_face_oval_populated_from_onnx(self):
        mock_sess = MagicMock()
        mock_sess.run.return_value = [_make_fake_logits()]

        parser = FaceParser()
        parser._sess = mock_sess

        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        bbox = (50, 50, 100, 100)
        landmarks = _make_random_landmarks()

        result = parser.parse(landmarks, img, bbox, None, 50.0)
        assert result.face_oval is not None
        assert result.face_oval.shape == (200, 200)
        # Class 1 (skin) is part of face_oval; expect high coverage.
        assert result.face_oval.max() > 0.5

    def test_landmark_subregions_still_populated(self):
        # When ONNX succeeds, _add_landmark_subregions is still called
        # so iris / nose / forehead / cheek / etc. are populated from
        # landmark geometry.
        mock_sess = MagicMock()
        mock_sess.run.return_value = [_make_fake_logits()]

        parser = FaceParser()
        parser._sess = mock_sess

        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        bbox = (50, 50, 100, 100)
        landmarks = _make_random_landmarks()

        result = parser.parse(landmarks, img, bbox, None, 50.0)
        for attr in ["left_iris", "right_iris", "nose", "forehead",
                     "left_cheek", "right_cheek", "left_under_eye",
                     "right_under_eye"]:
            assert getattr(result, attr) is not None, f"{attr} should be set"

    def test_hair_and_neck_populated_from_onnx(self):
        mock_sess = MagicMock()
        # Assign a different BiSeNet class to win argmax in three
        # disjoint spatial regions of the (512, 512) prediction map.
        logits = np.full((1, 19, 512, 512), -10.0, dtype=np.float32)
        # Skin everywhere except:
        logits[:, 1, :, :] = 5.0
        # top row wins class 17 (hair)
        logits[:, 17, 0:32, :] = 10.0
        # bottom row wins class 14 (neck)
        logits[:, 14, 480:512, :] = 10.0
        mock_sess.run.return_value = [logits]

        parser = FaceParser()
        parser._sess = mock_sess

        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        bbox = (50, 50, 100, 100)
        landmarks = _make_random_landmarks()

        result = parser.parse(landmarks, img, bbox, None, 50.0)
        assert result.neck is not None
        assert result.hair is not None
        # After feathering the class 14 / 17 regions should still have
        # some nonzero pixels.
        assert result.neck.max() > 0.0
        assert result.hair.max() > 0.0

    def test_skin_excludes_eyes_and_lips(self):
        # When ONNX populates both skin and eyes/lips classes, the
        # cleaning step must subtract the eye/lip masks from skin.
        logits = np.full((1, 19, 512, 512), -10.0, dtype=np.float32)
        logits[:, 1, :, :] = 10.0   # skin
        logits[:, 4, :, :] = 5.0    # left eye
        logits[:, 5, :, :] = 5.0    # right eye
        logits[:, 12, :, :] = 5.0   # lips
        mock_sess = MagicMock()
        mock_sess.run.return_value = [logits]

        parser = FaceParser()
        parser._sess = mock_sess

        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        bbox = (50, 50, 100, 100)
        landmarks = _make_random_landmarks()

        result = parser.parse(landmarks, img, bbox, None, 50.0)
        assert result.skin is not None
        assert result.left_eye is not None
        assert result.lips is not None
        # Skin should not overlap eyes (after subtraction, the eye
        # region should have near-zero skin coverage)
        overlap = (result.skin * result.left_eye).max()
        assert overlap < 0.5

    def test_preprocessing_uses_imagenet_normalization(self):
        # Verify the input tensor follows ImageNet mean/std normalization.
        # After normalization, a uniform 128 image has well-defined values.
        mock_sess = MagicMock()
        mock_sess.run.return_value = [_make_fake_logits()]

        parser = FaceParser()
        parser._sess = mock_sess

        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        bbox = (50, 50, 100, 100)
        landmarks = _make_random_landmarks()

        parser.parse(landmarks, img, bbox, None, 50.0)

        feed = mock_sess.run.call_args.kwargs.get(None) or mock_sess.run.call_args[0][1]
        crop = feed["input"]
        # 128/255 ≈ 0.5020. ImageNet mean[0] = 0.485.
        # Expected (R): (0.5020 - 0.485) / 0.229 ≈ 0.074
        r_mean = float(crop[0, 0].mean())
        # (G): (0.5020 - 0.456) / 0.224 ≈ 0.205
        g_mean = float(crop[0, 1].mean())
        # (B): (0.5020 - 0.406) / 0.225 ≈ 0.427
        b_mean = float(crop[0, 2].mean())
        assert r_mean == pytest.approx(0.074, abs=1e-3)
        assert g_mean == pytest.approx(0.205, abs=1e-3)
        assert b_mean == pytest.approx(0.427, abs=1e-3)


# ---------------------------------------------------------------------------
# FaceParser.parse()  — fallback behaviour when ONNX is broken
# ---------------------------------------------------------------------------


class TestFaceParserParseFallback:
    """Tests for FaceParser.parse() error handling and fallback."""

    def test_falls_back_when_session_run_raises(self):
        mock_sess = MagicMock()
        mock_sess.run.side_effect = RuntimeError("ONNX inference exploded")

        parser = FaceParser()
        parser._sess = mock_sess

        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        bbox = (50, 50, 100, 100)
        landmarks = _make_random_landmarks()

        # Must not raise — instead falls back to landmark-based parsing
        result = parser.parse(landmarks, img, bbox, None, 50.0)
        assert isinstance(result, FaceRegions)
        assert result.skin is not None

    def test_falls_back_when_session_is_none(self):
        parser = FaceParser()
        parser._sess = None

        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        bbox = (50, 50, 100, 100)
        landmarks = _make_random_landmarks()

        result = parser.parse(landmarks, img, bbox, None, 50.0)
        assert isinstance(result, FaceRegions)
        assert result.skin is not None

    def test_falls_back_when_face_bbox_is_none(self):
        mock_sess = MagicMock()
        mock_sess.run.return_value = [_make_fake_logits()]

        parser = FaceParser()
        parser._sess = mock_sess

        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        landmarks = _make_random_landmarks()

        # face_bbox=None should skip the ONNX path entirely
        result = parser.parse(landmarks, img, None, None, 50.0)
        assert isinstance(result, FaceRegions)
        # The mock session.run should never be invoked
        assert not mock_sess.run.called

    def test_falls_back_when_face_bbox_too_small(self):
        # Crop width/height must be >= 4 to attempt ONNX inference.
        mock_sess = MagicMock()
        mock_sess.run.return_value = [_make_fake_logits()]

        parser = FaceParser()
        parser._sess = mock_sess

        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        # Tiny bbox that won't produce a valid crop
        landmarks = _make_random_landmarks()

        result = parser.parse(landmarks, img, (10, 10, 1, 1), None, 50.0)
        assert isinstance(result, FaceRegions)
        assert result.skin is not None
        assert not mock_sess.run.called

    def test_falls_back_when_onnx_skin_too_sparse(self):
        # ONNX runs successfully but produces an empty skin mask.
        # In that case the parse() method must fall back to landmarks.
        mock_sess = MagicMock()
        # All zeros -> argmax = 0 for every pixel, no class 1, skin is empty
        empty_logits = np.zeros((1, 19, 512, 512), dtype=np.float32)
        mock_sess.run.return_value = [empty_logits]

        parser = FaceParser()
        parser._sess = mock_sess

        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        bbox = (50, 50, 100, 100)
        landmarks = _make_random_landmarks()

        result = parser.parse(landmarks, img, bbox, None, 50.0)
        assert isinstance(result, FaceRegions)
        # Skin max must be > 0.01 (the threshold) — i.e. the landmark
        # fallback produced a valid mask.
        assert result.skin is not None
        assert result.skin.max() >= 0.01

    def test_parse_does_not_propagate_session_run_exception(self):
        mock_sess = MagicMock()
        mock_sess.run.side_effect = MemoryError("OOM in ONNX")

        parser = FaceParser()
        parser._sess = mock_sess

        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        bbox = (50, 50, 100, 100)
        landmarks = _make_random_landmarks()

        # Even catastrophic errors should be swallowed and the parser
        # should fall back to landmarks.
        result = parser.parse(landmarks, img, bbox, None, 50.0)
        assert isinstance(result, FaceRegions)

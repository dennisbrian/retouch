"""Tests for batch ONNX parsing in retouch/parsing.py."""

import numpy as np
import pytest
from retouch.parsing import FaceParser, FaceRegions

class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

class MockLandmarksList:
    def __init__(self):
        # We need landmarks up to index 477 (for left/right iris)
        self.landmark = [MockLandmark(0.5, 0.5, 0.0) for _ in range(478)]


def _assert_region_attr_equal(a, b, attr):
    """Compare one FaceRegions attribute: float32 mask, or the
    parse_confidence dict (observational scalars, not array-like)."""
    if a is None:
        assert b is None
        return
    assert b is not None
    if attr == "parse_confidence":
        assert a.keys() == b.keys()
        for k in a:
            assert a[k] == pytest.approx(b[k], abs=1e-5)
    else:
        assert np.allclose(a, b, atol=1e-5)

def test_parse_batch_equivalence():
    parser = FaceParser()
    
    # 1. Verify N=1 batch output matches sequential parse()
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    landmarks = MockLandmarksList()
    bbox = (20, 20, 160, 160)
    person_mask = np.ones((200, 200), dtype=np.uint8) * 255
    ied = 50.0
    
    # Sequential parse
    res_seq = parser.parse(landmarks, img, bbox, person_mask, ied)
    
    # Batch parse with N=1
    res_batch = parser.parse_batch([img], [landmarks], [bbox], [person_mask], [ied])
    
    assert len(res_batch) == 1
    
    # Compare masks (and the parse_confidence scalar summary)
    for attr in FaceRegions.__slots__:
        _assert_region_attr_equal(getattr(res_seq, attr), getattr(res_batch[0], attr), attr)

def test_parse_batch_multiple_crops():
    parser = FaceParser()
    
    # 2. Verify N=2 identical crops yield identical masks
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    landmarks = MockLandmarksList()
    bbox = (20, 20, 160, 160)
    person_mask = np.ones((200, 200), dtype=np.uint8) * 255
    ied = 50.0
    
    res_batch = parser.parse_batch(
        [img, img], 
        [landmarks, landmarks], 
        [bbox, bbox], 
        [person_mask, person_mask], 
        [ied, ied]
    )
    
    assert len(res_batch) == 2
    for attr in FaceRegions.__slots__:
        _assert_region_attr_equal(getattr(res_batch[0], attr), getattr(res_batch[1], attr), attr)

def test_parse_batch_order_invariance():
    parser = FaceParser()
    
    # Create two different crops/landmarks
    img_a = np.zeros((200, 200, 3), dtype=np.uint8)
    landmarks_a = MockLandmarksList()
    landmarks_a.landmark[10] = MockLandmark(0.3, 0.4)
    bbox_a = (20, 20, 160, 160)
    ied_a = 50.0
    
    img_b = np.ones((200, 200, 3), dtype=np.uint8) * 100
    landmarks_b = MockLandmarksList()
    landmarks_b.landmark[10] = MockLandmark(0.7, 0.8)
    bbox_b = (10, 10, 180, 180)
    ied_b = 60.0
    
    # Parse [A, B]
    res_ab = parser.parse_batch(
        [img_a, img_b],
        [landmarks_a, landmarks_b],
        [bbox_a, bbox_b],
        [None, None],
        [ied_a, ied_b]
    )
    
    # Parse [B, A]
    res_ba = parser.parse_batch(
        [img_b, img_a],
        [landmarks_b, landmarks_a],
        [bbox_b, bbox_a],
        [None, None],
        [ied_b, ied_a]
    )
    
    assert len(res_ab) == 2
    assert len(res_ba) == 2
    
    # Verify A in [A, B] matches A in [B, A]
    for attr in FaceRegions.__slots__:
        _assert_region_attr_equal(getattr(res_ab[0], attr), getattr(res_ba[1], attr), attr)

    # Verify B in [A, B] matches B in [B, A]
    for attr in FaceRegions.__slots__:
        _assert_region_attr_equal(getattr(res_ab[1], attr), getattr(res_ba[0], attr), attr)


class TestParseConfidence:
    """Priority-3 evidence: parse_confidence is observational-only metadata,
    never used to gate or scale any op's strength."""

    def test_bisenet_confidence_by_region_margin_and_labels(self):
        from retouch.parsing import _bisenet_confidence_by_region

        # 2 classes over a 4x4 grid: top-left quadrant is class 1 (skin) with
        # a wide logit gap (confident); rest is class 4 (right_eye) with a
        # narrow gap (uncertain). Uses only 19 "channels" as BiSeNet expects,
        # but only classes 1 and 4 carry a strong logit.
        logits = np.full((19, 4, 4), -10.0, dtype=np.float32)
        logits[1, :2, :2] = 10.0   # confident skin quadrant
        logits[4, :2, 2:] = 0.6    # low-margin right_eye region
        logits[0, :2, 2:] = 0.5    # runner-up close behind, on purpose
        logits[4, 2:, :] = 10.0    # confident right_eye elsewhere
        pred = np.argmax(logits, axis=0).astype(np.uint8)

        out = _bisenet_confidence_by_region(logits, pred)
        assert set(out.keys()) <= {"skin", "right_eye"}
        assert 0.0 <= out["right_eye"] <= 1.0
        assert 0.0 <= out["skin"] <= 1.0
        # The confident skin quadrant must show a larger margin than the
        # deliberately-close right_eye quadrant.
        assert out["skin"] > out["right_eye"]

    def test_bisenet_confidence_pools_multi_label_regions(self):
        """lips = labels 12 + 13 (upper/lower lip). The pooled mean must
        land strictly between the two sub-labels' individual margins, not
        equal whichever one happens to be smaller (a min()-over-means bug)."""
        from retouch.parsing import _bisenet_confidence_by_region

        logits = np.full((19, 2, 8), -10.0, dtype=np.float32)
        # 4 pixels label 12 (high margin), 4 pixels label 13 (low margin).
        logits[12, :, :4] = 10.0
        logits[13, :, 4:] = 0.6
        logits[0, :, 4:] = 0.5  # close runner-up for the label-13 half
        pred = np.argmax(logits, axis=0).astype(np.uint8)

        out = _bisenet_confidence_by_region(logits, pred)
        # Compute each sub-label's own margin independently for the bound check.
        shifted = logits - logits.max(axis=0, keepdims=True)
        sm = np.exp(shifted) / np.exp(shifted).sum(axis=0, keepdims=True)
        top2 = np.partition(sm, -2, axis=0)[-2:]
        margin = top2[1] - top2[0]
        m12 = float(margin[pred == 12].mean())
        m13 = float(margin[pred == 13].mean())
        assert min(m12, m13) < out["lips"] < max(m12, m13)

    def test_bisenet_confidence_no_selected_label_is_absent(self):
        from retouch.parsing import _bisenet_confidence_by_region

        logits = np.full((19, 4, 4), 0.0, dtype=np.float32)
        logits[7] = 10.0  # class 7 has no entry in _CONFIDENCE_LABEL_TO_REGION
        pred = np.argmax(logits, axis=0).astype(np.uint8)
        out = _bisenet_confidence_by_region(logits, pred)
        assert out == {}

    def test_landmark_fallback_sets_empty_dict_not_none(self):
        """Landmark-only fallback has no BiSeNet confidence to report — the
        sentinel must be an explicit {} (checked, nothing available), never
        None (which reads as 'not populated')."""
        parser = FaceParser()
        landmarks = MockLandmarksList()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        regions = parser._landmark_fallback_only(landmarks, img, None, 40.0)
        assert regions.parse_confidence == {}

    def test_healthy_bisenet_path_reports_real_confidence(self):
        """On a real (non-degenerate) crop with BiSeNet available,
        parse_confidence must be non-empty with values in [0, 1]."""
        parser = FaceParser()
        if parser._sess is None:
            pytest.skip("BiSeNet ONNX model unavailable in this environment")
        rng = np.random.RandomState(0)
        img = rng.randint(60, 200, (200, 200, 3), dtype=np.uint8)
        landmarks = MockLandmarksList()
        bbox = (20, 20, 160, 160)
        regions = parser.parse(landmarks, img, bbox, None, 50.0)
        assert isinstance(regions.parse_confidence, dict)
        for v in regions.parse_confidence.values():
            assert 0.0 <= v <= 1.0

    def test_parse_confidence_not_read_by_any_op(self):
        """Guard against scope creep: no retouch module may branch on
        parse_confidence to change edit strength (observation-only per
        priority 3 — soft probabilities are not calibrated confidence)."""
        import pathlib
        import re

        retouch_dir = pathlib.Path(__file__).resolve().parent.parent / "retouch"
        hits = []
        for path in retouch_dir.glob("*.py"):
            if path.name == "parsing.py":
                continue
            text = path.read_text()
            if re.search(r"\bparse_confidence\b", text):
                hits.append(path.name)
        assert hits == [], (
            f"parse_confidence must stay observation-only, but is referenced "
            f"outside parsing.py in: {hits}"
        )

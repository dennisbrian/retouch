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
    """Compare one FaceRegions attribute: float32 mask, or a dict of
    observational scalars (parse_confidence, parse_boundary_confidence —
    not array-like)."""
    if a is None:
        assert b is None
        return
    assert b is not None
    if isinstance(a, dict):
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
        parse_confidence or parse_boundary_confidence to change edit
        strength (observation-only per priority 3 — soft probabilities are
        not calibrated confidence)."""
        import pathlib
        import re

        retouch_dir = pathlib.Path(__file__).resolve().parent.parent / "retouch"
        hits = []
        for path in retouch_dir.glob("*.py"):
            if path.name == "parsing.py":
                continue
            text = path.read_text()
            if re.search(r"\bparse(_boundary)?_confidence\b", text):
                hits.append(path.name)
        assert hits == [], (
            f"parse_confidence/parse_boundary_confidence must stay "
            f"observation-only, but referenced outside parsing.py in: {hits}"
        )


class TestParseBoundaryConfidence:
    """FA-01 diagnostic: boundary-ring margin, distinct from the whole-region
    average parse_confidence already reports (docs/plans/RESEARCH_FACE_RETOUCH_ALGORITHMS_2026_09_05.md
    §2.1). Same observation-only contract — covered by
    test_parse_confidence_not_read_by_any_op above."""

    def test_boundary_confidence_lower_than_interior_average(self):
        """A region confident in its interior but uncertain right at its
        true edge should show a lower boundary margin than its own
        whole-region average — that gap is the point of computing it
        separately. Needs a genuine internal region edge (a second label
        winning part of the crop): a single label filling the whole image
        has no edge for dilate/erode to find."""
        from retouch.parsing import (
            _bisenet_boundary_confidence_by_region,
            _bisenet_confidence_by_region,
        )

        size = 40
        logits = np.full((19, size, size), -10.0, dtype=np.float32)
        # Most of the crop: confident skin.
        logits[1, :, :] = 10.0
        # Near one edge, skin's margin over right_eyebrow collapses...
        logits[1, size - 6:, :] = 0.6
        logits[2, size - 6:, :] = 0.5
        # ...and the last couple of rows actually flip to right_eyebrow, so
        # skin has a true internal boundary inside the crop.
        logits[1, size - 2:, :] = -10.0
        logits[2, size - 2:, :] = 10.0
        pred = np.argmax(logits, axis=0).astype(np.uint8)

        whole = _bisenet_confidence_by_region(logits, pred)
        boundary = _bisenet_boundary_confidence_by_region(logits, pred, ring_px=2)

        assert "skin" in whole and "skin" in boundary
        assert boundary["skin"] < whole["skin"]

    def test_boundary_confidence_pools_multi_label_before_ringing(self):
        """lips = 12 | 13. The seam between the two sub-labels must not be
        treated as the lips region's outer boundary. Lips sits as a
        horizontal band between two "background" strips (label 7) so it has
        a genuine top/bottom edge distinct from the internal 12/13 seam."""
        from retouch.parsing import _bisenet_boundary_confidence_by_region

        h, w = 20, 20
        logits = np.full((19, h, w), -10.0, dtype=np.float32)
        logits[7, :6, :] = 10.0
        logits[7, 14:, :] = 10.0
        logits[12, 6:14, :10] = 10.0  # lips left half (label 12)
        logits[13, 6:14, 10:] = 10.0  # lips right half (label 13)
        # Low margin only at the true top edge of the lips region.
        logits[12, 6, :10] = 0.6
        logits[7, 6, :10] = 0.5
        pred = np.argmax(logits, axis=0).astype(np.uint8)

        out = _bisenet_boundary_confidence_by_region(logits, pred, ring_px=1)
        assert "lips" in out
        # If the 12/13 seam leaked into the ring, the pooled boundary mean
        # would sit near the confident seam value (~1.0), not be pulled
        # down by the genuinely low-margin top edge.
        assert out["lips"] < 0.9

    def test_boundary_confidence_omits_regions_too_thin_for_the_ring(self):
        """A region entirely consumed by the ring (thin eyebrow/eye slivers
        at model resolution) must be omitted, not silently fall back to the
        whole-region average."""
        from retouch.parsing import _bisenet_boundary_confidence_by_region

        logits = np.full((19, 20, 20), -10.0, dtype=np.float32)
        # A 1px-wide right_eyebrow line has no interior to erode away.
        logits[2, 5, :] = 10.0
        logits[1, :, :] = np.where(logits[1, :, :] > -10.0, logits[1, :, :], 5.0)
        pred = np.argmax(logits, axis=0).astype(np.uint8)

        out = _bisenet_boundary_confidence_by_region(logits, pred, ring_px=2)
        assert "right_eyebrow" not in out

    def test_landmark_fallback_sets_empty_boundary_dict(self):
        parser = FaceParser()
        landmarks = MockLandmarksList()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        regions = parser._landmark_fallback_only(landmarks, img, None, 40.0)
        assert regions.parse_boundary_confidence == {}

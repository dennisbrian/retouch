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
    
    # Compare masks
    for attr in FaceRegions.__slots__:
        m_seq = getattr(res_seq, attr)
        m_batch = getattr(res_batch[0], attr)
        if m_seq is None:
            assert m_batch is None
        else:
            assert m_batch is not None
            assert np.allclose(m_seq, m_batch, atol=1e-5)

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
        m0 = getattr(res_batch[0], attr)
        m1 = getattr(res_batch[1], attr)
        if m0 is None:
            assert m1 is None
        else:
            assert m1 is not None
            assert np.allclose(m0, m1, atol=1e-5)

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
        ma_in_ab = getattr(res_ab[0], attr)
        ma_in_ba = getattr(res_ba[1], attr)
        if ma_in_ab is None:
            assert ma_in_ba is None
        else:
            assert ma_in_ba is not None
            assert np.allclose(ma_in_ab, ma_in_ba, atol=1e-5)
            
    # Verify B in [A, B] matches B in [B, A]
    for attr in FaceRegions.__slots__:
        mb_in_ab = getattr(res_ab[1], attr)
        mb_in_ba = getattr(res_ba[0], attr)
        if mb_in_ab is None:
            assert mb_in_ba is None
        else:
            assert mb_in_ba is not None
            assert np.allclose(mb_in_ab, mb_in_ba, atol=1e-5)

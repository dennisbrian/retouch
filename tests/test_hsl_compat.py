import numpy as np
import pytest
from retouch.grading import ColorGrader

def test_hsl_dual_schema_compatibility():
    # Create a random colorful image
    np.random.seed(42)
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    
    # Flat schema (astia.json style)
    flat_adj = {
        "hue": {"red": 10, "orange": 5, "green": -15},
        "saturation": {"red": -20, "orange": -10, "green": 15},
        "luminance": {"red": 5, "orange": 8, "green": -5}
    }
    
    # Nested schema (classic_chrome.json style)
    nested_adj = {
        "red": {"hue_shift": 10, "sat_shift": -20, "lum_shift": 5},
        "orange": {"hue_shift": 5, "sat_shift": -10, "lum_shift": 8},
        "green": {"hue_shift": -15, "sat_shift": 15, "lum_shift": -5}
    }
    
    grader = ColorGrader()
    
    # Apply using flat adjustments
    out_flat = grader._apply_hsl_adjustments(img, flat_adj)
    
    # Apply using nested adjustments
    out_nested = grader._apply_hsl_adjustments(img, nested_adj)
    
    # Verify outputs are identical
    np.testing.assert_array_equal(out_flat, out_nested)

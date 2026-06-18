"""
Pro Max Face Retouch Engine
===========================
Professional-grade automated face retouching pipeline.

Pipeline:
    MediaPipe Face Mesh → Face Region Parsing → 3-Level Frequency Separation
    → Skin Smoothing → Blemish Removal → Under-Eye Repair → Eye Enhancement
    → Lip Enhancement → Teeth Whitening → Skin Equalization → Color Grading

Usage:
    from retouch import retouch
    result = retouch(img_bgr, preset='natural', smooth=60)

    # Or use the engine directly for more control:
    from retouch import RetouchEngine
    engine = RetouchEngine()
    result = engine.process(img_bgr, smooth=60, whiten=30)
"""

from .engine import RetouchEngine, retouch

__version__ = "2.0.0"
__all__ = ["RetouchEngine", "retouch", "__version__"]

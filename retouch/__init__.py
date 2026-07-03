"""Professional-grade automated face retouching pipeline.

Operates entirely in float32 with explicit colorspace boundaries (LAB for
skin tone, BGR for compositing). Preserves high-frequency skin detail via
3-level frequency separation — never blurs skin directly.

Pipeline:
    MediaPipe Face Mesh → BiSeNet Region Parsing → 3-Level Frequency
    Separation → Skin Smoothing (detail-preserving) → Blemish Removal →
    Under-Eye Repair → Eye Enhancement → Lip Enhancement → Teeth Whitening
    → Skin Equalization → Color Grading

Usage:
    from retouch import retouch, RetouchEngine
    result = retouch(img_bgr, preset='natural', smooth=60)
    engine = RetouchEngine()
    result = engine.process(img_bgr, smooth=60, whiten=30)
"""

from .engine import RetouchEngine, retouch

__version__ = "2.0.0"
__all__ = ["RetouchEngine", "retouch", "__version__"]

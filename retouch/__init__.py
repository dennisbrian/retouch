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

from typing import Any


__version__ = "2.0.0"
__all__ = ["RetouchEngine", "retouch", "__version__"]


def __getattr__(name: str) -> Any:
    """Load the engine only when a caller actually requests it.

    Runtime diagnostics must remain usable in a partially installed
    environment.  Importing the package used to eagerly import OpenCV,
    MediaPipe, ONNX Runtime, and every pipeline stage, which turned a missing
    optional binary into an unreportable import traceback.  The public API is
    unchanged for callers importing ``RetouchEngine`` or ``retouch``.
    """
    if name in {"RetouchEngine", "retouch"}:
        from .engine import RetouchEngine, retouch

        globals().update(RetouchEngine=RetouchEngine, retouch=retouch)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

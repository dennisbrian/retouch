"""16-bit float internal pipeline precision helpers.

Provides a small set of utilities for routing image data through a
float32 [0, 1] internal pipeline. The motivation: cumulative rounding
errors compound when many color operations are chained in uint8 (e.g.
``cv2.cvtColor`` -> curve -> ``cv2.cvtColor`` -> split-tone -> ...).
Doing the math in float32 between BGR<->LAB/HSV conversions preserves
the working precision and matches the working depth of commercial color
tools (DaVinci Resolve, Photoshop 32-bit mode, etc.).

Public API:
    to_float(img)             uint8 -> float32 [0, 1]
    to_uint8(img)             float32 [0, 1] -> uint8
    ensure_float(img)         pass-through if float, else convert
    PrecisionContext(...)     context manager exposing ``process(op, img)``
"""

from __future__ import annotations

from types import TracebackType
from typing import Callable, Optional, Type

import numpy as np


_FLOAT_MAX: float = 1.0
_FLOAT_MIN: float = 0.0
_UINT8_MAX: float = 255.0


def to_float(img: np.ndarray) -> np.ndarray:
    """Convert a uint8 BGR image to float32 in [0, 1].

    Float inputs are returned unchanged (pass-through). Values outside
    the unit interval are clipped to keep downstream math well-defined.

    Args:
        img: (H, W, C) uint8 image, or an existing float array.

    Returns:
        (H, W, C) float32 image with values in [0, 1].
    """
    if img.dtype == np.float32:
        return np.clip(img, _FLOAT_MIN, _FLOAT_MAX).astype(np.float32, copy=False)
    if img.dtype == np.float64:
        return np.clip(img, _FLOAT_MIN, _FLOAT_MAX).astype(np.float32)
    f = img.astype(np.float32) / _UINT8_MAX
    return np.clip(f, _FLOAT_MIN, _FLOAT_MAX)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert a float32 image in [0, 1] back to uint8.

    Uses ``np.round`` + ``clip`` (not ``astype`` truncation) to give
    a proper round-to-nearest conversion; this is the correct inverse
    of the implicit floor used by integer display paths.

    Args:
        img: (H, W, C) float32 image with values nominally in [0, 1].

    Returns:
        (H, W, C) uint8 image with values in [0, 255].
    """
    f = np.clip(img, _FLOAT_MIN, _FLOAT_MAX).astype(np.float32, copy=False)
    return np.clip(np.round(f * _UINT8_MAX), 0, 255).astype(np.uint8)


def ensure_float(img: np.ndarray) -> np.ndarray:
    """Return ``img`` as float32 [0, 1], converting from uint8 if needed.

    Args:
        img: (H, W, C) uint8 or float image.

    Returns:
        (H, W, C) float32 image in [0, 1].
    """
    if img.dtype == np.float32:
        return np.clip(img, _FLOAT_MIN, _FLOAT_MAX).astype(np.float32, copy=False)
    return to_float(img)


class PrecisionContext:
    """Context manager for running image ops in float32 precision.

    ``PrecisionContext`` does not change global state; it provides a
    documented entry point and a ``process`` helper that wraps an
    operation in float32:

        with PrecisionContext(bit_depth="16") as pc:
            out_u8 = pc.process(img_u8, my_float_op)

    Supported ``bit_depth`` values:
        - ``"16"`` (default): internal pipeline runs in float32 [0, 1].
        - ``"8"``: identity (no conversion). Provided for symmetry so
          call sites can branch on a single configuration value.

    Args:
        bit_depth: Internal precision mode. Only ``"16"`` performs a
            float conversion; ``"8"`` is a no-op for compatibility.
    """

    _SUPPORTED = ("8", "16")

    def __init__(self, bit_depth: str = "16") -> None:
        if bit_depth not in self._SUPPORTED:
            raise ValueError(
                f"PrecisionContext: bit_depth must be one of {self._SUPPORTED}, got {bit_depth!r}"
            )
        self.bit_depth: str = bit_depth
        self._active: bool = False

    def __enter__(self) -> "PrecisionContext":
        self._active = True
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        self._active = False
        return False

    @property
    def is_float(self) -> bool:
        """True when the context is running in float32 mode (``bit_depth='16'``)."""
        return self.bit_depth == "16"

    def process(
        self,
        img: np.ndarray,
        op: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Run ``op`` in float32 precision and return uint8 output.

        Args:
            img: (H, W, C) uint8 or float32 image.
            op: Callable taking a float32 [0, 1] image and returning a
                float32 [0, 1] image.

        Returns:
            (H, W, C) uint8 image.
        """
        if not self.is_float:
            return to_uint8(op(to_float(img)))
        if not self._active:
            raise RuntimeError(
                "PrecisionContext.process called outside of a `with` block"
            )
        out_f = op(to_float(img))
        return to_uint8(out_f)

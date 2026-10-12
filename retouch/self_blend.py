"""Analytical duplicate-layer blend operators.

This module is deliberately independent of the Retouch pipeline.  It models
an opaque image blended with an identical duplicate, ``f(x) = B(x, x)``.
The equations are the P5 research operators; this module makes no claim of
Photoshop pixel parity.
"""

from __future__ import annotations

from typing import Union

import numpy as np


ANALYTICAL_SELF_BLEND_MODES = (
    "multiply", "screen", "overlay", "hard_light", "soft_light",
    "color_dodge", "color_burn", "exclusion", "linear_light", "vivid_light",
)
__all__ = [
    "ANALYTICAL_SELF_BLEND_MODES",
    "self_blend_transfer",
    "apply_self_blend",
]
ArrayLike = Union[float, np.ndarray]


def _unit(value: ArrayLike) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    if not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
        raise ValueError("Analytical self-blend values must be finite and in [0, 1]")
    return x


def _mode(mode: str) -> str:
    if mode not in ANALYTICAL_SELF_BLEND_MODES:
        raise ValueError(f"Unknown analytical self-blend mode: {mode!r}")
    return mode


def self_blend_transfer(value: ArrayLike, mode: str) -> np.ndarray:
    """Evaluate ``B(x, x)`` in normalized component space.

    The returned array is float64 (including for scalar input), in ``[0, 1]``.
    Color Dodge/Burn use endpoint precedence from the P5 definition, so no
    singular division is evaluated at zero or one.
    """
    x = _unit(value)
    mode = _mode(mode)
    if mode == "multiply":
        y = x * x
    elif mode == "screen":
        y = 2 * x - x * x
    elif mode in ("overlay", "hard_light"):
        y = np.empty_like(x)
        low = x <= 0.5
        y[low] = 2 * x[low] * x[low]
        y[~low] = 1 - 2 * (1 - x[~low]) ** 2
    elif mode == "soft_light":
        y = np.empty_like(x)
        low = x <= 0.5
        y[low] = x[low] - (1 - 2 * x[low]) * x[low] * (1 - x[low])
        high = ~low
        y[high] = x[high] + (2 * x[high] - 1) * (np.sqrt(x[high]) - x[high])
    elif mode == "color_dodge":
        y = np.ones_like(x)
        zero = x == 0
        y[zero] = 0
        interior = (x > 0) & (x < 0.5)
        np.divide(x, 1 - x, out=y, where=interior)
    elif mode == "color_burn":
        y = np.zeros_like(x)
        interior = x > 0.5
        np.divide(1 - x, x, out=y, where=interior)
        y[interior] = 1 - y[interior]
        y[x == 1] = 1
    elif mode == "exclusion":
        y = 2 * x * (1 - x)
    elif mode == "linear_light":
        y = np.clip(3 * x - 1, 0, 1)
    else:  # vivid_light
        y = np.empty_like(x)
        low = x <= 0.5
        y[low] = 0
        low_mid = low & (x > 1 / 3)
        np.divide(3 * x - 1, 2 * x, out=y, where=low_mid)
        high = ~low
        y[high] = 1
        high_mid = high & (x < 2 / 3)
        np.divide(x, 2 * (1 - x), out=y, where=high_mid)
    return np.clip(y, 0, 1)


def _srgb_decode(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _srgb_encode(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def apply_self_blend(
    image: np.ndarray,
    mode: str,
    amount: float = 1.0,
    *,
    domain: str = "encoded",
) -> np.ndarray:
    """Apply an analytical self-blend with ``amount`` acting as layer opacity.

    ``image`` may be float32/float64 BGR/RGB in ``[0,1]`` or uint8/uint16.
    Integer output uses explicit nearest-half-up quantization. ``domain='linear'``
    decodes/encodes sRGB for the mathematical operation; it is a hypothesis,
    not an assertion about Photoshop's working-space behavior.
    """
    mode = _mode(mode)
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        raise ValueError("amount must be finite and in [0, 1]") from None
    if not np.isfinite(amount) or not 0 <= amount <= 1:
        raise ValueError("amount must be finite and in [0, 1]")
    if domain not in ("encoded", "linear"):
        raise ValueError("domain must be 'encoded' or 'linear'")
    source = np.asarray(image)
    if source.ndim < 2 or source.shape[-1] not in (1, 3):
        raise ValueError("image must have shape (..., 1) or (..., 3)")
    integer = np.issubdtype(source.dtype, np.integer)
    if integer:
        if source.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
            raise TypeError("Only uint8 and uint16 integer images are supported")
        peak = float(np.iinfo(source.dtype).max)
        x = source.astype(np.float64) / peak
    else:
        if not np.issubdtype(source.dtype, np.floating):
            raise TypeError("image must be float, uint8 or uint16")
        x = _unit(source)
    # Opacity zero is an exact identity in every selected domain. Returning
    # before decode/encode also avoids a needless transfer-function round
    # trip changing a float value by a few ulps or a quantized endpoint by one
    # DN.
    if amount == 0.0:
        return source.copy()
    work = _srgb_decode(x) if domain == "linear" else x
    transformed = self_blend_transfer(work, mode)
    result = (1.0 - amount) * work + amount * transformed
    if domain == "linear":
        result = _srgb_encode(result)
    result = np.clip(result, 0, 1)
    if not integer:
        return result.astype(source.dtype, copy=False)
    peak = float(np.iinfo(source.dtype).max)
    scaled = result * peak
    return np.floor(np.nextafter(scaled + 0.5, np.inf)).astype(source.dtype)

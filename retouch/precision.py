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
    to_uint8_dithered(img)    float32 [0, 1] -> uint8, final-delivery only
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
_DITHER_TILE_SIZE: int = 64


_BLUE_NOISE_RANK_TILE: Optional[np.ndarray] = None


def _void_and_cluster_rank_tile() -> np.ndarray:
    """Build a deterministic 64x64 void-and-cluster threshold matrix.

    The construction follows Ulichney's void-and-cluster method: relax a
    half-density binary pattern by replacing its densest cluster with its
    largest void, then rank removal and addition order from that stable
    pattern.  Every threshold level therefore inherits a well-dispersed dot
    distribution, unlike an ordered (Bayer) dither matrix.

    The tile is generated lazily because only final 8-bit delivery uses it.
    It has no external asset or random runtime dependency: the seed and the
    bounded relaxation count are fixed.
    """
    size = _DITHER_TILE_SIZE
    count = size * size
    sigma = np.float32(1.9)
    rng = np.random.default_rng(20_260_717)

    y, x = np.indices((size, size))
    dy = np.minimum(y, size - y)
    dx = np.minimum(x, size - x)
    kernel = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).astype(
        np.float32
    )

    # A deterministic half-density seed, then a periodic void-and-cluster
    # relaxation.  Updating the density field avoids thousands of FFTs.
    state = np.zeros((size, size), dtype=np.uint8)
    state.flat[rng.permutation(count)[: count // 2]] = 1
    density = np.fft.ifft2(
        np.fft.fft2(state) * np.fft.fft2(kernel)
    ).real.astype(np.float32)
    for _ in range(3_000):
        cluster = np.where(state, density, -np.inf)
        void = np.where(state, np.inf, density)
        cy, cx = np.unravel_index(np.argmax(cluster), state.shape)
        vy, vx = np.unravel_index(np.argmin(void), state.shape)
        if density[cy, cx] <= density[vy, vx]:
            break
        state[cy, cx] = 0
        density -= np.roll(kernel, (cy, cx), axis=(0, 1))
        state[vy, vx] = 1
        density += np.roll(kernel, (vy, vx), axis=(0, 1))

    mid_state = state.copy()
    ranks = np.empty((size, size), dtype=np.uint16)

    # Removing densest dots first ranks the lower half of the threshold tile.
    for rank in range(count // 2 - 1, -1, -1):
        cluster = np.where(state, density, -np.inf)
        cy, cx = np.unravel_index(np.argmax(cluster), state.shape)
        ranks[cy, cx] = rank
        state[cy, cx] = 0
        density -= np.roll(kernel, (cy, cx), axis=(0, 1))

    # Filling largest voids ranks the upper half from the same midpoint.
    state = mid_state
    density = np.fft.ifft2(
        np.fft.fft2(state) * np.fft.fft2(kernel)
    ).real.astype(np.float32)
    for rank in range(count // 2, count):
        void = np.where(state, np.inf, density)
        vy, vx = np.unravel_index(np.argmin(void), state.shape)
        ranks[vy, vx] = rank
        state[vy, vx] = 1
        density += np.roll(kernel, (vy, vx), axis=(0, 1))

    return ranks


def _blue_noise_rank_tile() -> np.ndarray:
    """Return the process-local deterministic void-and-cluster tile."""
    global _BLUE_NOISE_RANK_TILE
    if _BLUE_NOISE_RANK_TILE is None:
        _BLUE_NOISE_RANK_TILE = _void_and_cluster_rank_tile()
    return _BLUE_NOISE_RANK_TILE


def blue_noise_lsb_noise(
    shape: tuple[int, int],
    *,
    origin: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Return deterministic, zero-mean, sub-LSB dither in uint8 code units.

    The returned 2D float32 array is tiled in absolute pixel coordinates,
    which lets future final-export callers retain a consistent pattern when
    processing crops or tiles.  Its values are strictly inside ``[-0.5, 0.5]``
    and have an exact zero mean over each 64x64 tile.

    Args:
        shape: Requested ``(height, width)``.
        origin: Absolute ``(y, x)`` offset for tiled export/crop alignment.
    """
    height, width = shape
    if height < 0 or width < 0:
        raise ValueError(f"dither shape must be non-negative, got {shape!r}")

    y0, x0 = origin
    y = (np.arange(height, dtype=np.int64) + y0) % _DITHER_TILE_SIZE
    x = (np.arange(width, dtype=np.int64) + x0) % _DITHER_TILE_SIZE
    ranks = _blue_noise_rank_tile()[np.ix_(y, x)].astype(np.float32)
    return (
        (ranks + np.float32(0.5)) / np.float32(_DITHER_TILE_SIZE**2)
        - np.float32(0.5)
    )


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


def to_uint8_dithered(
    img: np.ndarray,
    *,
    origin: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Quantize a final float image with deterministic sub-LSB display dither.

    This is deliberately separate from :func:`to_uint8`: most current calls
    are intermediate compatibility boundaries, where injecting noise would
    accumulate across stages.  Use this helper exactly once at the selected
    float-to-8-bit delivery boundary.  All color channels share one threshold
    value per pixel so the dither does not introduce chromatic speckle.

    Existing uint8 input is returned unchanged.  This makes the helper safe
    for delivery code that accepts either a completed uint8 image or a float
    image, while preserving an already-quantized asset exactly.

    Args:
        img: A 2D or 3D float image nominally in [0, 1], or uint8 delivery
            image.
        origin: Absolute ``(y, x)`` offset for tiled export/crop alignment.

    Returns:
        uint8 image in [0, 255].
    """
    if img.dtype == np.uint8:
        return img.copy()
    if img.ndim not in (2, 3):
        raise ValueError(
            "to_uint8_dithered expects a 2D grayscale or 3D color image, "
            f"got shape {img.shape!r}"
        )

    f = np.clip(img, _FLOAT_MIN, _FLOAT_MAX).astype(np.float32, copy=False)
    noise = blue_noise_lsb_noise(f.shape[:2], origin=origin)
    if f.ndim == 3:
        noise = noise[..., np.newaxis]
    codes = f * _UINT8_MAX + noise
    return np.clip(np.round(codes), 0, 255).astype(np.uint8)


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


def check_float_pipeline(img: np.ndarray) -> bool:
    """Return True if image is float32 (pipeline running at float precision).

    Args:
        img: Input image array.

    Returns:
        True if dtype is float32.
    """
    return img.dtype == np.float32


def assert_no_uint8_intermediate(
    before: np.ndarray,
    after: np.ndarray,
    msg: str = "",
) -> None:
    """Assert that processing did not introduce uint8 quantization steps.

    Args:
        before: Image before operation (float32).
        after: Image after operation (float32).
        msg: Optional assertion message.

    Raises:
        AssertionError: If either input is uint8.
    """
    if before.dtype == np.uint8 or after.dtype == np.uint8:
        raise AssertionError(f"uint8 intermediate detected: {msg}")

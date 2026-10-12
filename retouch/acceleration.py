"""Hardware Acceleration Abstraction Layer (CUDA, MPS, CPU Fallback).

Provides high-throughput vectorized operations for batch processing, image filtering,
and color space transformations. Automatically falls back to CPU NumPy when GPU acceleration
is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Check PyTorch / CUDA / Metal (MPS) availability
_HAS_TORCH = False
_TORCH_DEVICE: Optional[str] = None

try:
    import torch

    _HAS_TORCH = True
    if torch.cuda.is_available():
        _TORCH_DEVICE = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _TORCH_DEVICE = "mps"
    else:
        _TORCH_DEVICE = "cpu"
except ImportError:
    torch = None  # type: ignore[assignment]


def get_acceleration_backend() -> str:
    """Return active hardware acceleration backend ('cuda', 'mps', or 'cpu')."""
    if _HAS_TORCH and _TORCH_DEVICE:
        return _TORCH_DEVICE
    return "cpu"


def is_gpu_available() -> bool:
    """Return True if CUDA or Metal (MPS) GPU acceleration is available."""
    return get_acceleration_backend() in ("cuda", "mps")


def accelerated_gaussian_blur(
    img: np.ndarray,
    ksize: int,
    sigma: float,
) -> np.ndarray:
    """Perform Gaussian blur with GPU acceleration if available, falling back to CPU.

    Args:
        img: (H, W) or (H, W, C) float32 or uint8 array.
        ksize: Kernel diameter (must be odd).
        sigma: Gaussian standard deviation.

    Returns:
        Blurred array matching input shape and dtype.
    """
    import cv2

    if not is_gpu_available() or img.size > 8000 * 8000:
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)

    try:
        # PyTorch GPU path
        is_float = img.dtype == np.float32
        img_t = torch.from_numpy(img).to(_TORCH_DEVICE)
        
        # Prepare 2D kernel
        k = cv2.getGaussianKernel(ksize, sigma, ktype=cv2.CV_32F)
        kernel2d = np.outer(k, k).astype(np.float32)
        kernel_t = torch.from_numpy(kernel2d).to(_TORCH_DEVICE)

        if img.ndim == 2:
            inp = img_t.unsqueeze(0).unsqueeze(0).float()
            weight = kernel_t.unsqueeze(0).unsqueeze(0)
            padding = ksize // 2
            out_t = torch.nn.functional.conv2d(inp, weight, padding=padding)
            out = out_t.squeeze(0).squeeze(0).cpu().numpy()
        elif img.ndim == 3:
            C = img.shape[2]
            inp = img_t.permute(2, 0, 1).unsqueeze(0).float()
            weight = kernel_t.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
            padding = ksize // 2
            out_t = torch.nn.functional.conv2d(inp, weight, padding=padding, groups=C)
            out = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        else:
            return cv2.GaussianBlur(img, (ksize, ksize), sigma)

        if not is_float:
            return np.clip(out, 0, 255).astype(np.uint8)
        return out.astype(np.float32)
    except Exception as exc:
        logger.debug("GPU blur fallback to CPU: %s", exc)
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)

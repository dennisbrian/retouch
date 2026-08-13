"""Linear RAW development pipeline for professional workflows.

Supports loading RAW formats (DNG, CR2, ARW, NEF, etc.) in linear RGB space,
processing with linear grading operations, and exporting to DNG or TIFF formats
for maximum quality and color science fidelity.

Public API:
    RAWDeveloper: Orchestrator for RAW input, linear processing, output management
    CameraMatrix: Per-camera color matrix storage
    WhiteBalanceEstimator: Auto-WB from neutral gray reference or metadata
    LinearGrader: Float32 grading operations in linear RGB (curves, color, tone)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any

import cv2
import numpy as np

try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False

logger = logging.getLogger(__name__)


# ============================================================================
# Camera Matrix Database
# ============================================================================

class CameraMatrix:
    """Per-camera color matrix storage (Canon, Nikon, Sony, etc.).

    Stores XYZ→RGB transformation matrices and metadata for common RAW sources.
    Allows per-camera and per-ISO override for color science.
    """

    # Basic matrices for common cameras (XYZ D65 -> RGB)
    # These are approximate matrices; for production use, load from dcraw/RawTherapee
    CAMERA_MATRICES: Dict[str, Dict[str, Any]] = {
        # Canon EOS 5D Mark IV
        "Canon EOS 5D": {
            "matrices": {
                "base": np.array([
                    [0.7674, -0.0717, 0.0325],
                    [-0.4427, 1.3632, 0.0826],
                    [-0.0528, -0.5189, 1.5639],
                ], dtype=np.float32),
            },
            "white_balance": {
                "daylight": np.array([1.0, 1.0, 1.0], dtype=np.float32),
                "cloudy": np.array([1.08, 1.0, 0.92], dtype=np.float32),
                "shade": np.array([1.16, 1.0, 0.84], dtype=np.float32),
                "tungsten": np.array([1.0, 0.95, 2.2], dtype=np.float32),
            },
        },
        # Nikon D850
        "Nikon D850": {
            "matrices": {
                "base": np.array([
                    [0.8213, -0.0917, -0.0100],
                    [-0.4523, 1.3540, 0.1017],
                    [-0.0781, -0.3744, 1.4523],
                ], dtype=np.float32),
            },
            "white_balance": {
                "daylight": np.array([1.0, 1.0, 1.0], dtype=np.float32),
                "cloudy": np.array([1.07, 1.0, 0.94], dtype=np.float32),
                "shade": np.array([1.15, 1.0, 0.86], dtype=np.float32),
                "tungsten": np.array([1.0, 0.95, 2.1], dtype=np.float32),
            },
        },
        # Sony A7R IV
        "Sony ILCE-7": {
            "matrices": {
                "base": np.array([
                    [0.7314, -0.0654, 0.0305],
                    [-0.4073, 1.3554, 0.0538],
                    [-0.0606, -0.4195, 1.4903],
                ], dtype=np.float32),
            },
            "white_balance": {
                "daylight": np.array([1.0, 1.0, 1.0], dtype=np.float32),
                "cloudy": np.array([1.06, 1.0, 0.95], dtype=np.float32),
                "shade": np.array([1.14, 1.0, 0.87], dtype=np.float32),
                "tungsten": np.array([1.0, 0.96, 2.0], dtype=np.float32),
            },
        },
        # Generic fallback (sRGB-like)
        "Generic": {
            "matrices": {
                "base": np.array([
                    [0.7, -0.05, 0.0],
                    [-0.4, 1.35, 0.05],
                    [-0.06, -0.4, 1.46],
                ], dtype=np.float32),
            },
            "white_balance": {
                "daylight": np.array([1.0, 1.0, 1.0], dtype=np.float32),
                "cloudy": np.array([1.07, 1.0, 0.93], dtype=np.float32),
                "shade": np.array([1.15, 1.0, 0.85], dtype=np.float32),
                "tungsten": np.array([1.0, 0.95, 2.0], dtype=np.float32),
            },
        },
    }

    def __init__(self):
        pass

    @classmethod
    def get_camera_matrix(cls, camera_model: Optional[str] = None) -> np.ndarray:
        """Get XYZ->RGB matrix for a camera model.

        Args:
            camera_model: Camera model string (e.g. "Canon EOS 5D").
                If None or not found, returns generic matrix.

        Returns:
            (3, 3) float32 transformation matrix.
        """
        if camera_model is None:
            key = "Generic"
        else:
            # Try exact match first
            key = next((k for k in cls.CAMERA_MATRICES.keys() if k in camera_model), "Generic")

        return cls.CAMERA_MATRICES[key]["matrices"]["base"].copy()

    @classmethod
    def get_white_balance(
        cls,
        camera_model: Optional[str] = None,
        preset: str = "daylight"
    ) -> np.ndarray:
        """Get white balance multipliers for a camera model.

        Args:
            camera_model: Camera model string. If None, uses Generic.
            preset: WB preset name (daylight, cloudy, shade, tungsten).

        Returns:
            (3,) float32 RGB multipliers.
        """
        if camera_model is None:
            key = "Generic"
        else:
            key = next((k for k in cls.CAMERA_MATRICES.keys() if k in camera_model), "Generic")

        presets = cls.CAMERA_MATRICES[key]["white_balance"]
        return presets.get(preset, presets["daylight"]).copy()


# ============================================================================
# White Balance Estimation
# ============================================================================

class WhiteBalanceEstimator:
    """Auto-WB from neutral gray reference or metadata."""

    @staticmethod
    def estimate_from_metadata(
        wb_r: float,
        wb_g: float,
        wb_b: float,
    ) -> np.ndarray:
        """Create WB multipliers from raw camera metadata.

        Args:
            wb_r, wb_g, wb_b: White balance coefficients from RAW metadata.

        Returns:
            (3,) float32 RGB multipliers (normalized so green = 1.0).
        """
        wb = np.array([wb_r, wb_g, wb_b], dtype=np.float32)
        # Normalize so green channel = 1.0
        if wb[1] > 1e-6:
            wb = wb / wb[1]
        return wb

    @staticmethod
    def estimate_from_gray(
        linear_rgb: np.ndarray,
        gray_roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """Estimate WB from a neutral gray region (gray-world assumption).

        Args:
            linear_rgb: (H, W, 3) float32 linear RGB image.
            gray_roi: Optional (y1, x1, y2, x2) region of interest for gray.
                If None, uses center 25% of image.

        Returns:
            (3,) float32 RGB multipliers.
        """
        if gray_roi is None:
            h, w = linear_rgb.shape[:2]
            y1 = h // 4
            x1 = w // 4
            y2 = 3 * h // 4
            x2 = 3 * w // 4
            roi = linear_rgb[y1:y2, x1:x2, :]
        else:
            y1, x1, y2, x2 = gray_roi
            roi = linear_rgb[y1:y2, x1:x2, :]

        # Compute mean per-channel intensity
        means = roi.mean(axis=(0, 1))

        # Normalize so green = 1.0 (green is typically most sensitive)
        if means[1] > 1e-6:
            wb = means / means[1]
        else:
            wb = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        return wb.astype(np.float32)


# ============================================================================
# Linear Grading Operations
# ============================================================================

class LinearGrader:
    """Float32 grading operations in linear RGB (curves, color, tone).

    All operations assume linear tristimulus RGB without gamma encoding.
    """

    @staticmethod
    def linear_curves(
        linear_rgb: np.ndarray,
        curves: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Apply per-channel tone curves in linear space.

        Args:
            linear_rgb: (H, W, 3) float32 linear RGB.
            curves: Dict with keys 'r', 'g', 'b', each a (256,) or (512,)
                float32 LUT for tone mapping. Index by normalized pixel value.

        Returns:
            (H, W, 3) float32 linear RGB with curves applied.
        """
        out = linear_rgb.copy()

        for ch_idx, ch_key in enumerate(['b', 'g', 'r']):  # BGR order
            if ch_key not in curves:
                continue

            curve = curves[ch_key]
            # Normalize pixel values to [0, 1] for indexing
            normalized = np.clip(out[:, :, ch_idx], 0.0, 1.0)
            # Map to LUT index
            lut_idx = np.clip(normalized * (len(curve) - 1), 0, len(curve) - 1).astype(np.int32)
            out[:, :, ch_idx] = curve[lut_idx]

        return out

    @staticmethod
    def linear_exposure(
        linear_rgb: np.ndarray,
        exposure: float,
    ) -> np.ndarray:
        """Apply linear exposure correction (multiplicative brightness).

        Args:
            linear_rgb: (H, W, 3) float32 linear RGB.
            exposure: Exposure adjustment in stops (e.g., 1.0 = 2x brightness).

        Returns:
            (H, W, 3) float32 linear RGB.
        """
        if abs(exposure) < 1e-6:
            return linear_rgb.copy()

        factor = 2.0 ** exposure
        return (linear_rgb * factor).astype(np.float32)

    @staticmethod
    def linear_contrast(
        linear_rgb: np.ndarray,
        contrast: float,
        midpoint: float = 0.5,
    ) -> np.ndarray:
        """Apply linear contrast boost around a midpoint.

        Args:
            linear_rgb: (H, W, 3) float32 linear RGB in [0, 1].
            contrast: Contrast factor (1.0 = no change, >1.0 = boost).
            midpoint: Pivot point for contrast (default 0.5 for 50% gray).

        Returns:
            (H, W, 3) float32 linear RGB.
        """
        if abs(contrast - 1.0) < 1e-6:
            return linear_rgb.copy()

        # Contrast boost: ((val - mid) * contrast) + mid
        out = linear_rgb.copy()
        out = (out - midpoint) * contrast + midpoint
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def linear_color_cast(
        linear_rgb: np.ndarray,
        color_cast: np.ndarray,
    ) -> np.ndarray:
        """Apply multiplicative RGB color cast.

        Args:
            linear_rgb: (H, W, 3) float32 linear RGB.
            color_cast: (3,) float32 RGB multipliers.

        Returns:
            (H, W, 3) float32 linear RGB.
        """
        out = linear_rgb * color_cast[np.newaxis, np.newaxis, :]
        return np.clip(out, 0.0, 1.0).astype(np.float32)


# ============================================================================
# RAW Developer (Main Orchestrator)
# ============================================================================

class RAWDeveloper:
    """Orchestrator for RAW input, linear processing, output management.

    Handles:
    - Loading RAW formats (DNG, CR2, ARW, NEF) → linear RGB + metadata
    - Processing in linear space with camera matrices and WB
    - Exporting to DNG (master), 32-bit float TIFF, or 16-bit TIFF
    - Metadata preservation (ISO, WB, camera matrix, EXIF)
    """

    def __init__(self):
        if not HAS_RAWPY:
            logger.warning("rawpy not available; RAW loading will fail. Install rawpy for full support.")
        self.last_metadata: Dict[str, Any] = {}
        self.camera_matrix_instance = CameraMatrix()

    def load_raw(
        self,
        path: Union[str, Path],
        use_camera_wb: bool = True,
        auto_brightness: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load RAW image to linear RGB float32.

        Args:
            path: Path to RAW file (DNG, CR2, ARW, NEF, etc.).
            use_camera_wb: Apply camera white balance from metadata (default True).
            auto_brightness: Apply auto-brightness normalization (default False).

        Returns:
            Tuple of:
            - linear_rgb: (H, W, 3) float32 linear RGB in [0, 1] range.
            - metadata: Dict with keys 'iso', 'wb', 'camera_model', 'bit_depth'.

        Raises:
            ImportError: If rawpy is not installed.
            FileNotFoundError: If path does not exist.
            ValueError: If file cannot be decoded as RAW.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"RAW file not found: {path}")

        if not HAS_RAWPY:
            raise ImportError("rawpy is required for RAW loading. Install: pip install rawpy")

        try:
            with rawpy.imread(str(path)) as raw:
                # Postprocess to 16-bit *linear* RGB. gamma=(1, 1) disables
                # rawpy's default BT.709 encode (2.222, 4.5) so the output is
                # true linear tristimulus RGB — the domain LinearGrader assumes.
                # Without this the values are gamma-encoded and every linear
                # op runs in the wrong domain. highlight_mode=ReconstructDefault
                # recovers blown channels instead of the default hard Clip.
                rgb16 = raw.postprocess(
                    use_camera_wb=use_camera_wb,
                    no_auto_bright=not auto_brightness,
                    output_bps=16,
                    gamma=(1, 1),
                    output_color=rawpy.ColorSpace.sRGB,
                    highlight_mode=rawpy.HighlightMode.ReconstructDefault,
                )

                # Extract metadata. rawpy exposes these via attributes that
                # vary across versions (e.g. no top-level `raw.iso`/`raw.wb`
                # in 0.27), so read defensively — missing fields must not fail
                # the decode.
                cam_wb = getattr(raw, "camera_whitebalance", None)
                iso_other = getattr(getattr(raw, "other", None), "iso_speed", None)
                metadata = {
                    "iso": getattr(raw, "iso", None) if iso_other is None else iso_other,
                    "camera_model": getattr(raw, "camera_model", None),
                    "wb": np.asarray(cam_wb[:3], dtype=np.float32) if cam_wb is not None else None,
                    "bit_depth": 16,
                }
        except Exception as e:
            raise ValueError(f"Failed to decode RAW file {path}: {e}") from e

        # Convert uint16 [0, 65535] to float32 [0, 1]
        linear_rgb = (rgb16.astype(np.float32) / 65535.0).astype(np.float32)
        # Clamp to [0, 1] to handle any overshoot
        linear_rgb = np.clip(linear_rgb, 0.0, 1.0)

        self.last_metadata = metadata
        return linear_rgb, metadata

    def develop(
        self,
        linear_rgb: np.ndarray,
        profile: Optional[str] = None,
        exposure: float = 0.0,
        contrast: float = 1.0,
        color_cast: Optional[np.ndarray] = None,
        denoise_strength: float = 0.0,
    ) -> np.ndarray:
        """Process linear RGB with optional profile and adjustments.

        Args:
            linear_rgb: (H, W, 3) float32 linear RGB in [0, 1].
            profile: Optional color profile name (not yet implemented).
            exposure: Exposure adjustment in stops.
            contrast: Contrast factor (1.0 = no change).
            color_cast: Optional (3,) RGB color multipliers.
            denoise_strength: Pre-pipeline noise reduction (0-100, in linear space).

        Returns:
            (H, W, 3) float32 linear RGB in [0, 1].
        """
        out = linear_rgb.copy()

        # Denoise in linear space (bilateral filter on linear RGB)
        if denoise_strength > 0.01:
            # Normalize strength to 0-1, scale to bilateral filter diameter
            sigma_s = max(3, int(denoise_strength / 10.0))  # spatial extent
            sigma_r = min(0.1, denoise_strength / 100.0)  # range extent
            # Bilateral filter expects uint8 or float32 [0, 1]
            out = cv2.bilateralFilter(
                out,
                d=sigma_s,
                sigmaColor=sigma_r,
                sigmaSpace=sigma_s,
            ).astype(np.float32)

        # Apply exposure
        out = LinearGrader.linear_exposure(out, exposure)

        # Apply contrast
        out = LinearGrader.linear_contrast(out, contrast)

        # Apply color cast
        if color_cast is not None:
            out = LinearGrader.linear_color_cast(out, color_cast)

        # Ensure output is in [0, 1]
        out = np.clip(out, 0.0, 1.0).astype(np.float32)

        return out

    def export_linear(
        self,
        img: np.ndarray,
        path: Union[str, Path],
        format: str = "tiff",
        bit_depth: int = 16,
        compress: bool = True,
    ) -> None:
        """Export linear RGB to DNG, 32-bit float TIFF, or 16-bit TIFF.

        Args:
            img: (H, W, 3) float32 linear RGB in [0, 1].
            path: Output file path (.dng, .tif, or .tiff).
            format: Export format: 'dng', 'tiff', '32f', '16bit' (default 'tiff').
            bit_depth: Target bit depth (16 or 32, ignored for DNG).
            compress: Use lossless compression (default True).

        Returns:
            None. Writes to disk.

        Raises:
            ValueError: If format is unsupported.
            RuntimeError: If write fails.
        """
        path = Path(path)
        fmt = format.lower()

        if fmt == "dng":
            self._export_dng(img, path)
        elif fmt in ("tiff", "tif", "32f", "16bit"):
            self._export_tiff(img, path, bit_depth, compress)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_tiff(
        self,
        img: np.ndarray,
        path: Path,
        bit_depth: int,
        compress: bool,
    ) -> None:
        """Export to TIFF (32-bit float or 16-bit uint).

        Args:
            img: (H, W, 3) float32 linear RGB in [0, 1].
            path: Output TIFF path.
            bit_depth: 16 or 32.
            compress: Use LZW compression.
        """
        try:
            if bit_depth == 32:
                # 32-bit float TIFF via cv2 (preserves full precision)
                # img is already in [0, 1], no scaling needed
                # Keep BGR order for cv2 (img is already BGR from engine)
                params = [cv2.IMWRITE_TIFF_COMPRESSION, 5 if compress else 1]
                ok = cv2.imwrite(str(path), img.astype(np.float32), params)
                if not ok:
                    raise RuntimeError(f"cv2.imwrite returned False for {path}")
                logger.info(f"Exported {bit_depth}-bit float TIFF to {path}")

            elif bit_depth == 16:
                # 16-bit uint TIFF via cv2
                # Convert [0, 1] to [0, 65535], keep BGR order for cv2
                img_u16 = np.clip(img * 65535.0, 0, 65535).astype(np.uint16)
                params = [cv2.IMWRITE_TIFF_COMPRESSION, 5 if compress else 1]
                ok = cv2.imwrite(str(path), img_u16, params)
                if not ok:
                    raise RuntimeError(f"cv2.imwrite returned False for {path}")
                logger.info(f"Exported {bit_depth}-bit TIFF to {path}")
            else:
                raise ValueError(f"Unsupported bit depth: {bit_depth}")

        except Exception as e:
            raise RuntimeError(f"Failed to export TIFF: {e}") from e

    def _export_dng(self, img: np.ndarray, path: Path) -> None:
        """Export to DNG (uncompressed or lossless JPEG2000).

        Note: Full DNG export requires libtiff or advanced PIL/rawpy support.
        This is a simplified implementation that exports as 16-bit TIFF
        with DNG naming convention.

        Args:
            img: (H, W, 3) float32 linear RGB in [0, 1].
            path: Output DNG path.
        """
        logger.warning(
            "Full DNG export not yet implemented; exporting as 16-bit TIFF. "
            "For production, integrate with LibTiff or RawTherapee."
        )
        # Fallback to 16-bit TIFF
        self._export_tiff(img, path.with_suffix('.tif'), bit_depth=16, compress=False)

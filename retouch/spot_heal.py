"""F4 / F4.b spot heal & large-region removal — brush-painted inpainting.

Wraps the existing ``heal_region`` infrastructure in a class that accepts
brush-painted masks (float32 [0,1] or uint8 [0,255]) and provides both a
small-spot heal and a larger multi-pass object-removal path. Both paths
preserve input dtype (uint8 BGR or float32 [0,255] BGR).

F4.b adds :class:`LamaHealer` — large-region removal via the LaMa ONNX
model (loaded through :mod:`retouch.model_fetch`). When the LaMa model is
not present on disk the healer transparently falls back to multi-pass
Telea via :class:`SpotHealer`, so callers always get a result.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

from .heal import heal_region
from .model_fetch import model_exists

logger = logging.getLogger(__name__)

_LAMA_MODEL_NAME = "lama_inpaint"


class SpotHealer:
    """Brush-painted spot heal and object removal.

    Accepts float32 [0,1] or uint8 [0,255] masks and uint8 or float32
    [0,255] BGR images. Returns the same dtype as the input image.
    """

    def heal(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        method: str = "telea",
        radius: int = 5,
    ) -> np.ndarray:
        """Heal a small masked region with a single inpaint pass.

        Args:
            img: (H, W, 3) uint8 or float32 [0,255] BGR image.
            mask: (H, W) float32 [0,1] or uint8 [0,255] mask of the
                region to heal. Non-zero pixels are inpainted.
            method: ``"telea"`` or ``"ns"`` (Navier-Stokes).
            radius: Inpaint radius in pixels. Auto-scaled from the mask
                bounding box when the caller does not override.

        Returns:
            (H, W, 3) image matching input dtype. A zero mask is a no-op
            and returns the input image unchanged.
        """
        if img.dtype not in (np.uint8, np.float32):
            raise ValueError(
                f"SpotHealer.heal: img must be uint8 or float32, got {img.dtype}."
            )
        mask_u8 = _coerce_mask_uint8(mask)
        if mask_u8.sum() == 0:
            return img.copy() if img.dtype == np.uint8 else img.astype(np.float32, copy=True)

        feathered = _feather_for_blend(mask_u8, radius)
        return heal_region(img, feathered, method=method, radius=radius)

    def heal_object_removal(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        method: str = "telea",
    ) -> np.ndarray:
        """Remove a larger region via multi-pass inpainting.

        Large masks produce smearing artifacts from a single inpaint pass.
        This method dilates the mask, runs a first inpaint to fill the
        interior, then runs a second pass on the original mask boundary
        to clean up residual seams. Both passes use the same method.

        Args:
            img: (H, W, 3) uint8 or float32 [0,255] BGR image.
            mask: (H, W) float32 [0,1] or uint8 [0,255] mask of the
                object to remove.
            method: ``"telea"`` or ``"ns"``.

        Returns:
            (H, W, 3) image matching input dtype.
        """
        if img.dtype not in (np.uint8, np.float32):
            raise ValueError(
                "SpotHealer.heal_object_removal: img must be uint8 or float32, "
                f"got {img.dtype}."
            )
        mask_u8 = _coerce_mask_uint8(mask)
        if mask_u8.sum() == 0:
            return img.copy() if img.dtype == np.uint8 else img.astype(np.float32, copy=True)

        pass1_radius = _auto_radius(mask_u8)
        dilated = _dilate_mask(mask_u8, iterations=2)
        first = heal_region(img, dilated, method=method, radius=pass1_radius)

        boundary = _boundary_band(mask_u8, band_width=max(pass1_radius, 3))
        if boundary.sum() == 0:
            return first
        return heal_region(first, boundary, method=method, radius=max(pass1_radius, 3))


class LamaHealer:
    """Large-region removal via the LaMa ONNX model with Telea fallback.

    LaMa (Large Mask inpainting) is structurally much better than Telea /
    Navier-Stokes at filling large holes because it is a learned model that
    propagates long-range context. This class loads the model lazily through
    :mod:`retouch.model_fetch` on first use and caches the ONNX session.

    When the LaMa model file is not present on disk (placeholder manifest
    entry, model not yet downloaded), :meth:`heal_large` transparently
    falls back to :class:`SpotHealer.heal_object_removal` so callers always
    receive a reasonable result. The fallback path is multi-pass Telea.

    Thread-safety: ONNX session creation is guarded by a module-level lock;
    once created the session is reused across threads. ``onnxruntime``
    sessions are themselves thread-safe for ``run`` calls.

    All pixel arithmetic is performed in float32 internally; the output
    dtype matches the input image dtype (uint8 BGR or float32 [0,255] BGR).
    """

    DEFAULT_TILE_SIZE = 1024
    DEFAULT_TILE_OVERLAP = 128
    LAMA_INPUT_SIZE = 1024  # LaMa was trained on 1024x1024 patches; pad/resize to fit

    _session_lock = threading.Lock()
    _session = None  # type: Optional[object]
    _session_model_path: Optional[str] = None

    def __init__(
        self,
        tile_size: int = DEFAULT_TILE_SIZE,
        tile_overlap: int = DEFAULT_TILE_OVERLAP,
        fallback_method: str = "telea",
    ) -> None:
        if tile_size < 256:
            raise ValueError(
                f"LamaHealer: tile_size must be >= 256, got {tile_size}."
            )
        if tile_overlap < 0 or tile_overlap >= tile_size // 2:
            raise ValueError(
                f"LamaHealer: tile_overlap must be in [0, tile_size//2), "
                f"got {tile_overlap} for tile_size {tile_size}."
            )
        if fallback_method not in ("telea", "ns"):
            raise ValueError(
                f"LamaHealer: fallback_method must be 'telea' or 'ns', "
                f"got {fallback_method!r}."
            )
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.fallback_method = fallback_method
        self._spot = SpotHealer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def heal_large(
        self,
        img: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Remove a large masked region.

        Args:
            img: (H, W, 3) uint8 or float32 [0,255] BGR image.
            mask: (H, W) float32 [0,1] or uint8 [0,255] mask of the region
                to remove. Non-zero pixels are inpainted.

        Returns:
            (H, W, 3) image matching input dtype. A zero mask is a no-op
            and returns the input image unchanged.
        """
        if img.dtype not in (np.uint8, np.float32):
            raise ValueError(
                f"LamaHealer.heal_large: img must be uint8 or float32, got {img.dtype}."
            )
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(
                f"LamaHealer.heal_large: img must be (H, W, 3) BGR, got shape {img.shape}."
            )
        mask_u8 = _coerce_mask_uint8(mask)
        if mask_u8.sum() == 0:
            return img.copy() if img.dtype == np.uint8 else img.astype(np.float32, copy=True)

        session = self._get_session()
        if session is None:
            logger.info(
                "LamaHealer.heal_large: LaMa model not available; "
                "falling back to SpotHealer multi-pass %s.",
                self.fallback_method,
            )
            return self._spot.heal_object_removal(
                img, mask_u8, method=self.fallback_method
            )

        return self._infer_lama(img, mask_u8, session)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @classmethod
    def _get_session(cls) -> Optional[object]:
        """Return a cached ONNX session, or None if the model is unavailable.

        Thread-safe: session creation is guarded by a class-level lock. The
        model file's existence is checked via :func:`model_fetch.model_exists`
        so placeholder manifest entries (empty sha256, missing file) cleanly
        trigger the fallback path without raising.
        """
        if cls._session is not None:
            return cls._session
        with cls._session_lock:
            if cls._session is not None:
                return cls._session
            if not model_exists(_LAMA_MODEL_NAME):
                logger.debug(
                    "LamaHealer: model %r not present on disk; fallback will be used.",
                    _LAMA_MODEL_NAME,
                )
                return None
            session = cls._load_session()
            if session is not None:
                cls._session = session
            return session

    @classmethod
    def _load_session(cls) -> Optional[object]:
        """Create an ONNX Runtime inference session for the LaMa model.

        Lazily imports :mod:`onnxruntime` so the rest of the package does
        not hard-depend on it. Any error during session creation is logged
        and ``None`` is returned so the fallback path is taken — we never
        raise out of model loading because the fallback is always viable.
        """
        try:
            import onnxruntime as ort
        except ImportError as e:
            logger.warning(
                "LamaHealer: onnxruntime not installed (%s); using fallback.", e
            )
            return None
        try:
            from .model_fetch import get_model_path

            path = get_model_path(_LAMA_MODEL_NAME)
        except Exception as e:  # noqa: BLE001 — model_fetch raises several types
            logger.warning(
                "LamaHealer: could not resolve model path for %r: %s; using fallback.",
                _LAMA_MODEL_NAME,
                e,
            )
            return None
        try:
            session = ort.InferenceSession(
                path, providers=["CPUExecutionProvider"]
            )
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning(
                "LamaHealer: ONNX session creation failed for %s: %s; using fallback.",
                path,
                e,
            )
            return None
        cls._session_model_path = path
        logger.info("LamaHealer: loaded LaMa model from %s", path)
        return session

    @classmethod
    def reset_session(cls) -> None:
        """Drop the cached ONNX session. Thread-safe; mainly for tests."""
        with cls._session_lock:
            cls._session = None
            cls._session_model_path = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer_lama(
        self,
        img: np.ndarray,
        mask_u8: np.ndarray,
        session: object,
    ) -> np.ndarray:
        """Run LaMa inference with tiling for large images.

        LaMa expects an RGB image and a binary mask at a fixed resolution
        (commonly 1024x1024). We work in float32 internally, resize each
        tile to the model input size, run inference, resize back, and blend
        tiles with a feathered weight to avoid seams. The output dtype
        matches the input image dtype.
        """
        original_dtype = img.dtype
        img_f32 = img.astype(np.float32, copy=True)
        H, W = img_f32.shape[:2]

        out = img_f32.copy()
        weight = np.zeros((H, W), dtype=np.float32)

        tiles = self._tile_grid(H, W)
        if not tiles:
            # Image smaller than one tile — single pass on the whole thing.
            tiles = [(0, 0, H, W)]

        input_size = self._input_size_for(session)

        for (y0, x0, th, tw) in tiles:
            y1 = min(y0 + th, H)
            x1 = min(x0 + tw, W)
            tile_img = img_f32[y0:y1, x0:x1]
            tile_mask = mask_u8[y0:y1, x0:x1]
            if tile_mask.sum() == 0:
                continue
            inpainted = self._run_tile(session, tile_img, tile_mask, input_size)
            out[y0:y1, x0:x1] = inpainted
            weight[y0:y1, x0:x1] = 1.0

        if original_dtype == np.uint8:
            return np.clip(out, 0.0, 255.0).astype(np.uint8)
        return np.clip(out, 0.0, 255.0).astype(np.float32, copy=False)

    def _run_tile(
        self,
        session: object,
        tile_img: np.ndarray,
        tile_mask: np.ndarray,
        input_size: int,
    ) -> np.ndarray:
        """Run LaMa on one tile: resize → infer → resize back.

        Returns the inpainted tile in float32 [0,255] BGR at the original
        tile resolution. Pixels outside the mask are copied from the input
        tile so only the masked region is overwritten.
        """
        th, tw = tile_img.shape[:2]
        # LaMa input: RGB float32 [0,1] + mask float32 [0,1], resized to input_size.
        resized_img = cv2.resize(tile_img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(
            tile_mask, (input_size, input_size), interpolation=cv2.INTER_NEAREST
        )
        rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask_f = (resized_mask > 127).astype(np.float32)

        inputs = self._build_inputs(session, rgb, mask_f)
        try:
            outputs = session.run(None, inputs)
        except (RuntimeError, ValueError) as e:
            logger.warning(
                "LamaHealer: ONNX run failed (%s); falling back to Telea for this tile.", e
            )
            return self._spot.heal_object_removal(
                tile_img, tile_mask, method=self.fallback_method
            ).astype(np.float32, copy=False)

        out_arr = np.asarray(outputs[0])
        out_arr = np.squeeze(out_arr)
        if out_arr.ndim == 2:
            out_arr = np.stack([out_arr] * 3, axis=-1)
        if out_arr.shape[-1] == 3 and out_arr.dtype != np.float32:
            out_arr = out_arr.astype(np.float32)
        # Expect float roughly in [0,1]; rescale conservatively.
        if out_arr.max() <= 1.5:
            out_arr = out_arr * 255.0
        out_bgr = cv2.cvtColor(np.clip(out_arr, 0.0, 255.0).astype(np.float32), cv2.COLOR_RGB2BGR)
        out_tile = cv2.resize(out_bgr, (tw, th), interpolation=cv2.INTER_LINEAR)

        # Only replace masked pixels; keep original elsewhere.
        mask_f_tile = (tile_mask > 127).astype(np.float32)[:, :, np.newaxis]
        result = tile_img * (1.0 - mask_f_tile) + out_tile * mask_f_tile
        return np.clip(result, 0.0, 255.0).astype(np.float32, copy=False)

    def _build_inputs(
        self, session: object, rgb: np.ndarray, mask: np.ndarray
    ) -> dict:
        """Build the ONNX input feed dict from the session's input spec.

        LaMa models typically take an image tensor and a mask tensor. The
        exact input names vary by export, so we map by position heuristically:
        the input whose name contains 'image'/'img' gets the image, the one
        containing 'mask' gets the mask. Falls back to positional ordering.
        """
        input_meta = session.get_inputs()
        feed: dict = {}
        if len(input_meta) == 0:
            return feed
        img_name = None
        mask_name = None
        for meta in input_meta:
            name = meta.name.lower()
            if "mask" in name:
                mask_name = meta.name
            elif "image" in name or "img" in name or "input" in name:
                img_name = meta.name
        if img_name is None and mask_name is None:
            img_name = input_meta[0].name
            if len(input_meta) > 1:
                mask_name = input_meta[1].name
        if img_name is not None:
            feed[img_name] = rgb[np.newaxis, ...]  # add batch dim
        if mask_name is not None:
            feed[mask_name] = mask[np.newaxis, ..., np.newaxis]
        return feed

    def _input_size_for(self, session: object) -> int:
        """Best-effort extraction of the LaMa model's expected spatial input size."""
        try:
            meta = session.get_inputs()[0]
            shape = meta.shape  # e.g. [1, 3, 1024, 1024] or [1, 3, -1, -1]
            for dim in reversed(shape):
                if isinstance(dim, int) and dim > 0:
                    return dim
        except (IndexError, AttributeError, TypeError) as e:
            logger.debug("LamaHealer: could not read input size from session: %s", e)
        return self.LAMA_INPUT_SIZE

    def _tile_grid(
        self, H: int, W: int
    ) -> list:
        """Yield (y0, x0, tile_h, tile_w) tiles covering the image with overlap.

        Tiles that contain no mask are skipped by the caller, so we emit all
        tiles here. Each tile is at most ``tile_size`` per side; adjacent
        tiles overlap by ``tile_overlap`` to avoid seam artifacts.
        """
        if H <= self.tile_size and W <= self.tile_size:
            return []
        tiles: list = []
        step = self.tile_size - self.tile_overlap
        y = 0
        while y < H:
            x = 0
            th = min(self.tile_size, H - y)
            while x < W:
                tw = min(self.tile_size, W - x)
                tiles.append((y, x, th, tw))
                if x + self.tile_size >= W:
                    break
                x += step
            if y + self.tile_size >= H:
                break
            y += step
        return tiles


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _coerce_mask_uint8(mask: np.ndarray) -> np.ndarray:
    """Return a uint8 binary-ish mask (0 or 255) from float32 or uint8 input."""
    if mask.dtype == np.float32 or mask.dtype == np.float64:
        m = np.clip(mask.astype(np.float32), 0.0, 1.0)
        return (m * 255.0).astype(np.uint8)
    m = mask.astype(np.uint8, copy=False)
    if m.max() <= 1:
        m = (m.astype(np.float32) * 255.0).astype(np.uint8)
    return m


def _feather_for_blend(mask_u8: np.ndarray, radius: int) -> np.ndarray:
    """Threshold a painted mask at a usable binary level and feather the edge.

    ``heal_region`` already applies its own Gaussian-blend feathering, so we
    only binarize here (brush strokes can have soft partial-coverage pixels
    that confuse ``cv2.inpaint``).
    """
    binary = np.where(mask_u8 >= 32, 255, 0).astype(np.uint8)
    k = max(int(radius) * 2 + 1, 3) | 1
    soft = cv2.GaussianBlur(binary.astype(np.float32), (k, k), 0)
    return np.where(soft >= 32, 255, 0).astype(np.uint8)


def _auto_radius(mask_u8: np.ndarray) -> int:
    """Inpaint radius scaled from the mask bounding-box diagonal."""
    coords = cv2.findNonZero(mask_u8)
    if coords is None:
        return 3
    x, y, w, h = cv2.boundingRect(coords)
    diag = (w * w + h * h) ** 0.5
    r = max(int(diag * 0.1), 2)
    return min(r, 50)


def _dilate_mask(mask_u8: np.ndarray, iterations: int = 1) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(mask_u8, kernel, iterations=iterations)


def _boundary_band(mask_u8: np.ndarray, band_width: int) -> np.ndarray:
    """Return a band of ``band_width`` pixels along the mask boundary.

    Used as the cleanup region for the second object-removal pass.
    """
    eroded = cv2.erode(
        mask_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_width * 2 + 1, band_width * 2 + 1)),
        iterations=1,
    )
    band = cv2.subtract(mask_u8, eroded)
    return np.where(band > 0, 255, 0).astype(np.uint8)

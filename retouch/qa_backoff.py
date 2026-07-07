"""A5 — "No plastic skin" guarantee (QA auto-back-off).

When the QA detectors flag plastic skin on a processed image, this module
computes a conservative set of parameter adjustments that reduce the
skin-smoothing aggressiveness so the engine can re-process and recover
texture. Back-off is only triggered on an actual *flag*, never on a mere
warning, and is bounded so the engine can never oscillate or push params
into a degenerate state.

Public API:
    QABackoff.check_and_backoff(img, ctx, qa_warnings) -> Optional[Dict[str, float]]
    QABackoff.backoff_strategy(warning_type, current_params) -> Dict[str, float]
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .qa_detectors import QAWarning

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Back-off configuration
# ---------------------------------------------------------------------------
#
# All multipliers are < 1.0 (they *reduce* the offending parameter). The
# floor values prevent the engine from collapsing a parameter to zero on a
# single flag — a complete removal of smoothing reads as a different class
# of artifact (raw unretouched skin) and is not the goal of A5. The goal is
# to recover *enough* high-frequency texture to clear the plastic-skin flag.
SMOOTH_BACKOFF_FACTOR: float = 0.6       # cut smoothing to 60% of current
SMOOTH_BACKOFF_FLOOR: float = 0.0        # absolute floor (0 = disabled)
WHITEN_BACKOFF_FACTOR: float = 0.8       # whiten/foundation flattens texture too
EQUALIZE_BACKOFF_FACTOR: float = 0.7     # CLAHE equalize can over-compress skin
BLEMISH_BACKOFF_FACTOR: float = 0.8      # blemish inpainting smears HF
SKIN_QUANTIZE_BACKOFF_FACTOR: float = 0.5  # quantization is a direct texture killer

# Max number of back-off iterations before we give up and ship the least-bad
# result. Two iterations is enough to clear a real flag at conservative
# factors; more risks thrashing.
MAX_BACKOFF_ITERATIONS: int = 2

# Parameters that influence plastic-skin perception. Ordered by impact.
_PLASTIC_SKIN_PARAMS: List[str] = [
    "smooth",
    "equalize",
    "whiten",
    "blemish",
    "skin_quantize",
    "micro_dodge_burn",
    "nose_smooth",
]


class QABackoff:
    """Conservative parameter back-off driven by QA flags.

    The class is intentionally stateless beyond its configuration so it can
    be reused across images and threads. The engine owns the iteration
    loop: it calls :meth:`check_and_backoff`, applies the returned
    adjustments, re-runs the pipeline, and re-checks QA until the flag
    clears or :data:`MAX_BACKOFF_ITERATIONS` is reached.
    """

    def __init__(
        self,
        max_iterations: int = MAX_BACKOFF_ITERATIONS,
        smooth_factor: float = SMOOTH_BACKOFF_FACTOR,
        smooth_floor: float = SMOOTH_BACKOFF_FLOOR,
        whiten_factor: float = WHITEN_BACKOFF_FACTOR,
        equalize_factor: float = EQUALIZE_BACKOFF_FACTOR,
        blemish_factor: float = BLEMISH_BACKOFF_FACTOR,
        skin_quantize_factor: float = SKIN_QUANTIZE_BACKOFF_FACTOR,
    ) -> None:
        self.max_iterations = max_iterations
        self._smooth_factor = smooth_factor
        self._smooth_floor = smooth_floor
        self._whiten_factor = whiten_factor
        self._equalize_factor = equalize_factor
        self._blemish_factor = blemish_factor
        self._skin_quantize_factor = skin_quantize_factor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_and_backoff(
        self,
        img: np.ndarray,
        ctx: Any,
        qa_warnings: List[QAWarning],
    ) -> Optional[Dict[str, float]]:
        """Inspect QA warnings and return adjusted params if back-off is needed.

        Args:
            img: The processed output image (BGR uint8 or float32 [0,255]).
                Inspected only to confirm shape/validity; the actual
                plastic-skin measurement comes from the QA warnings already
                computed by the engine.
            ctx: The :class:`ProcessingContext` used for the current pass.
                Read for the *current* parameter values that back-off is
                computed against.
            qa_warnings: List of :class:`QAWarning` from the just-completed
                pass. Only ``flagged=True`` warnings trigger back-off.

        Returns:
            ``None`` if no flagged warning requires back-off (the image
            ships as-is). Otherwise a dict of ``{param_name: new_value}``
            suitable for applying to a fresh :class:`ProcessingContext`
            before re-processing. The dict is the *complete* set of
            adjustments for this iteration, not a delta.
        """
        if img is None or not isinstance(img, np.ndarray):
            logger.debug("QABackoff: invalid image, skipping back-off")
            return None

        if not qa_warnings:
            return None

        # Only flagged warnings warrant back-off — a warning that is below
        # threshold is informational, not actionable.
        flagged = [w for w in qa_warnings if getattr(w, "flagged", False)]
        if not flagged:
            return None

        # Build a {warning_type: current_params} snapshot. We only back off
        # on warning types we know how to handle.
        current_params = self._snapshot_params(ctx)

        adjustments: Dict[str, float] = {}
        for warning in flagged:
            strategy = self.backoff_strategy(warning.detector, current_params)
            for key, new_val in strategy.items():
                # On multiple flagged warnings touching the same param, keep
                # the more conservative (smaller) value.
                if key not in adjustments or new_val < adjustments[key]:
                    adjustments[key] = new_val
                    current_params[key] = new_val  # so later strategies compound

        if not adjustments:
            return None

        logger.info(
            "QABackoff: backing off %d param(s) due to flagged QA warnings: %s",
            len(adjustments),
            ", ".join(f"{k}={v:.2f}" for k, v in adjustments.items()),
        )
        return adjustments

    def backoff_strategy(
        self,
        warning_type: str,
        current_params: Dict[str, float],
    ) -> Dict[str, float]:
        """Return the param adjustments for a given QA warning type.

        This is a pure function of ``(warning_type, current_params)`` — it
        does not inspect the image. The engine uses it both directly and
        via :meth:`check_and_backoff`.

        Args:
            warning_type: The ``detector`` field of a :class:`QAWarning`
                (e.g. ``"plastic_skin"``, ``"halo"``).
            current_params: Snapshot of the current relevant parameter
                values (``smooth``, ``whiten``, ``equalize``, ...). Values
                may be ``None`` (e.g. ``nose_smooth``); those are skipped.

        Returns:
            Dict of ``{param_name: new_value}`` to apply. Empty dict if
            ``warning_type`` is not handled.
        """
        if warning_type == "plastic_skin":
            return self._plastic_skin_strategy(current_params)
        # Future warning types (halo, banding, ...) can be added here. A5
        # scope is plastic-skin only; others return no adjustments so the
        # engine ships the original result rather than guessing.
        return {}

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _plastic_skin_strategy(
        self,
        current: Dict[str, float],
    ) -> Dict[str, float]:
        """Reduce the params that compress skin high-frequency texture.

        The hierarchy (most → least impactful) is:
          1. ``smooth`` — the primary texture-erasing parameter.
          2. ``skin_quantize`` — direct posterization of skin tone.
          3. ``equalize`` — CLAHE on skin can flatten micro-contrast.
          4. ``whiten`` — foundation/brightening lifts shadows and hides pores.
          5. ``blemish`` — inpainting smears HF in blemish regions.
          6. ``nose_smooth`` — independent nose smoothing, if set.

        Each is reduced by its factor but never below the configured floor.
        A param that is already at 0 (or unset) is left alone — backing
        off a no-op is wasteful and would mask a real flag.
        """
        out: Dict[str, float] = {}

        def _reduce(key: str, factor: float, floor: float = 0.0) -> None:
            val = current.get(key)
            if val is None:
                return
            try:
                v = float(val)
            except (TypeError, ValueError):
                return
            if v <= 0.0:
                return
            new_v = max(v * factor, floor)
            # Round to 2 decimals so downstream float comparisons are stable.
            out[key] = round(new_v, 2)

        _reduce("smooth", self._smooth_factor, self._smooth_floor)
        _reduce("skin_quantize", self._skin_quantize_factor)
        _reduce("equalize", self._equalize_factor)
        _reduce("whiten", self._whiten_factor)
        _reduce("blemish", self._blemish_factor)
        # nose_smooth is Optional[float] — only back off if explicitly set
        # and > 0, otherwise the engine uses the global smooth value which
        # is already covered above.
        _reduce("nose_smooth", self._smooth_factor, self._smooth_floor)

        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _snapshot_params(ctx: Any) -> Dict[str, float]:
        """Extract a float snapshot of the back-off-relevant params.

        Missing/unset params are present in the dict as ``None`` so
        :meth:`backoff_strategy` can detect and skip them. This keeps the
        strategy function pure and testable without needing the full
        :class:`ProcessingContext`.
        """
        snap: Dict[str, float] = {}
        for key in _PLASTIC_SKIN_PARAMS:
            snap[key] = getattr(ctx, key, None)
        return snap

    @staticmethod
    def apply_adjustments(ctx: Any, adjustments: Dict[str, float]) -> None:
        """Apply back-off adjustments to a ProcessingContext in place.

        Separated from :meth:`check_and_backoff` so the engine can log the
        decision before mutating the context, and so tests can verify the
        strategy without constructing a full context.

        Args:
            ctx: A :class:`ProcessingContext` (mutated in place).
            adjustments: The dict returned by :meth:`check_and_backoff`.
        """
        for key, value in adjustments.items():
            if hasattr(ctx, key):
                setattr(ctx, key, value)
            else:
                logger.warning("QABackoff: param %r not found on context, skipping", key)

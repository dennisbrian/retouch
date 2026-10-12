"""F10 — Smart Default (one-click, explainable) + ``--smart`` batch.

Wraps F9 (:mod:`retouch.image_analyzer`) into a one-click smart processing
mode. ``SmartProcessor.analyze_and_suggest()`` runs the F9 analysis on the
image, picks the best recipe + per-parameter overrides, and returns a
human-readable explanation of *why* each parameter was chosen — so the
suggestion is auditable rather than a black box.

``SmartProcessor.process_smart()`` is the one-call entry point: analyze +
apply the suggestion through the regular :class:`RetouchEngine` pipeline.

Design notes
------------
* The analyzer (F9) is pure-vectorized NumPy on a downsampled proxy, so
  analysis cost is in the millisecond range regardless of input resolution.
* Suggestions are *target-based* (move the image toward a neutral target),
  not delta-based — the same philosophy F9 follows. A bright image is not
  brightened further.
* Explanations are short, plain-language strings suitable for direct GUI
  display (e.g. ``"noise detected → ai_denoise=30"``).
* float32 internal throughout; uint8 inputs converted at the boundary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .image_analyzer import ImageAnalysis, ImageAnalyzer, SkinCondition

logger = logging.getLogger(__name__)

_EPS = 1e-6


# ---------------------------------------------------------------------------
# SmartSuggestion dataclass
# ---------------------------------------------------------------------------


@dataclass
class SmartSuggestion:
    """The output of a smart-analysis pass.

    Attributes:
        recipe: Suggested recipe name (a key in ``retouch.recipes.RECIPES``).
        params: GUI-scale parameter overrides — only the keys that should
            change from the recipe default are present (matching
            :meth:`ImageAnalyzer.suggest_params`).
        analysis: The underlying :class:`ImageAnalysis` from F9.
        explanations: Plain-language strings explaining each non-default
            parameter choice, e.g.
            ``"noise detected (σ=0.62) → ai_denoise=37"``.
    """

    recipe: str
    params: Dict[str, Any] = field(default_factory=dict)
    analysis: Optional[ImageAnalysis] = None
    explanations: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SmartProcessor
# ---------------------------------------------------------------------------


class SmartProcessor:
    """One-click smart processing: analyze → suggest → (optionally) process.

    Construct with an optional ``RetouchEngine`` instance. When none is
    supplied, ``process_smart`` lazily constructs one (and the caller owns
    its lifecycle). When the caller already has an engine (e.g. the GUI's
    shared instance), pass it in to avoid re-loading the BiSeNet model.
    """

    def __init__(
        self,
        engine: Optional[Any] = None,
        analyzer: Optional[ImageAnalyzer] = None,
        downsample_dim: int = 512,
    ) -> None:
        self._engine = engine
        self._analyzer = analyzer or ImageAnalyzer(downsample_dim=downsample_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_and_suggest(
        self,
        img: np.ndarray,
        face_bboxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
        skin_mask: Optional[np.ndarray] = None,
        person_mask: Optional[np.ndarray] = None,
    ) -> SmartSuggestion:
        """Analyze ``img`` and return a :class:`SmartSuggestion`.

        Args:
            img: (H, W, 3) BGR uint8 or float32 [0, 255].
            face_bboxes: Optional list of (x, y, w, h) face boxes — enables
                skin-region analysis when paired with ``skin_mask``.
            skin_mask: Optional (H, W) float mask [0, 1] (or [0, 255]).
            person_mask: Optional (H, W) float mask (informational).

        Returns:
            SmartSuggestion with recipe, params, analysis, explanations.
        """
        analysis = self._analyzer.analyze(
            img,
            face_bboxes=face_bboxes,
            skin_mask=skin_mask,
            person_mask=person_mask,
        )
        params = self._analyzer.suggest_params(analysis)
        recipe = self._analyzer.suggest_recipe(analysis)
        explanations = self._explain(analysis, params, recipe)
        return SmartSuggestion(
            recipe=recipe,
            params=params,
            analysis=analysis,
            explanations=explanations,
        )

    def process_smart(
        self,
        img: np.ndarray,
        face_bboxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
        skin_mask: Optional[np.ndarray] = None,
        person_mask: Optional[np.ndarray] = None,
        extra_engine_kwargs: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Analyze + process in one call.

        Runs :meth:`analyze_and_suggest`, then applies the suggested recipe
        and parameter overrides through :class:`RetouchEngine.process`.

        Args:
            img: (H, W, 3) BGR uint8 or float32 [0, 255].
            face_bboxes, skin_mask, person_mask: Forwarded to
                :meth:`analyze_and_suggest`.
            extra_engine_kwargs: Additional kwargs merged on top of the
                suggestion (caller overrides win). Useful for ``fast``,
                ``quality``, ``debug_dir``, etc.

        Returns:
            The processed BGR uint8 image (a ``ProcessingResult`` ndarray
            when the engine returns one).
        """
        suggestion = self.analyze_and_suggest(
            img,
            face_bboxes=face_bboxes,
            skin_mask=skin_mask,
            person_mask=person_mask,
        )
        engine = self._get_engine()
        kwargs: Dict[str, Any] = {"recipe": suggestion.recipe}
        # Translate GUI-scale suggestion params → engine kwargs.
        kwargs.update(self._gui_params_to_engine_kwargs(suggestion.params))
        if extra_engine_kwargs:
            kwargs.update(extra_engine_kwargs)
        return engine.process(img, **kwargs)

    # ------------------------------------------------------------------
    # Internal: explanation generation
    # ------------------------------------------------------------------

    def _explain(
        self,
        analysis: ImageAnalysis,
        params: Dict[str, Any],
        recipe: str,
    ) -> List[str]:
        """Build a list of plain-language explanations for each chosen param.

        Each explanation is a short string of the form
        ``"<reason> → <param>=<value>"``. The order matches the order in
        which params were emitted by F9's ``suggest_params`` so the GUI can
        display them top-to-bottom without re-sorting.
        """
        explanations: List[str] = []
        skin = analysis.skin_condition

        # Recipe choice — always explain the lighting/wb context.
        explanations.append(self._explain_recipe(analysis, recipe))

        # Brightness
        if "brightness" in params:
            direction = "lift" if params["brightness"] > 0 else "reduce"
            explanations.append(
                f"mean luminance {analysis.mean_luminance:.0f} (target 128) "
                f"→ brightness={params['brightness']} ({direction})"
            )

        # Blacks / whites / contrast (dynamic range)
        p = analysis.l_percentiles
        if "blacks" in params and len(p) >= 7:
            explanations.append(
                f"crushed blacks (p1={p[0]:.0f}) → blacks={params['blacks']}"
            )
        if "whites" in params and len(p) >= 7:
            explanations.append(
                f"blown highlights (p99={p[6]:.0f}) → whites={params['whites']}"
            )
        if "contrast" in params:
            dr = analysis.dynamic_range
            direction = "boost" if params["contrast"] > 0 else "reduce"
            explanations.append(
                f"dynamic range {dr:.0f} → contrast={params['contrast']} ({direction})"
            )

        # White balance
        wb = analysis.white_balance_estimate
        kelvin = float(wb.get("kelvin", 6500.0))
        tint = float(wb.get("tint", 0.0))
        if "white_balance_kelvin" in params:
            cast_dir = "warm" if kelvin < 6500.0 else "cool"
            explanations.append(
                f"{cast_dir} cast (kelvin≈{kelvin:.0f}) → "
                f"white_balance_kelvin={params['white_balance_kelvin']}"
            )
        if "white_balance_tint" in params:
            tint_dir = "green" if tint > 0 else "magenta"
            explanations.append(
                f"{tint_dir} tint (tint≈{tint:.1f}) → "
                f"white_balance_tint={params['white_balance_tint']}"
            )

        # Noise
        if "ai_denoise" in params:
            explanations.append(
                f"noise detected (level={analysis.noise_level:.2f}) → "
                f"ai_denoise={params['ai_denoise']}"
            )

        # Sharpness
        if "sharpen" in params:
            explanations.append(
                f"soft image (sharpness={analysis.sharpness_estimate:.2f}) → "
                f"sharpen={params['sharpen']}"
            )

        # Skin condition
        if skin.present:
            if "whiten" in params:
                explanations.append(
                    f"skin underexposed (skin L={skin.l_mean:.0f}, target 175) → "
                    f"whiten={params['whiten']}"
                )
            if "redness_even" in params:
                explanations.append(
                    f"skin redness (a*={skin.a_mean:.0f}, expected 140) → "
                    f"redness_even={params['redness_even']}"
                )
            if "equalize" in params:
                explanations.append(
                    f"uneven skin chroma (σ_C={skin.c_std:.1f}) → "
                    f"equalize={params['equalize']}"
                )

        # Lighting-type specific
        if "shadows" in params:
            explanations.append(
                f"low-light scene → shadows={params['shadows']}"
            )
        if "clarity" in params and "shadows" in params:
            explanations.append(
                f"low-light scene → clarity={params['clarity']}"
            )
        elif "vibrance" in params:
            explanations.append(
                f"flat outdoor lighting → vibrance={params['vibrance']}"
            )

        return explanations

    def _explain_recipe(self, analysis: ImageAnalysis, recipe: str) -> str:
        """One-line explanation of why this recipe was chosen."""
        parts: List[str] = []
        parts.append(f"lighting={analysis.lighting_type}")
        parts.append(f"key={analysis.key}")
        if analysis.noise_level > 0.5:
            parts.append(f"high noise ({analysis.noise_level:.2f})")
        wb = analysis.white_balance_estimate
        cast = float(wb.get("cast_strength", 0.0))
        if cast > 0.12:
            kelvin = float(wb.get("kelvin", 6500.0))
            parts.append(f"WB cast (kelvin≈{kelvin:.0f})")
        skin = analysis.skin_condition
        if skin.present:
            parts.append("face detected")
        reason = ", ".join(parts)
        return f"recipe='{recipe}' ({reason})"

    # ------------------------------------------------------------------
    # Internal: engine plumbing
    # ------------------------------------------------------------------

    def _get_engine(self) -> Any:
        if self._engine is None:
            from . import RetouchEngine
            self._engine = RetouchEngine()
        return self._engine

    @staticmethod
    def _gui_params_to_engine_kwargs(
        gui_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Translate GUI-scale suggestion params → engine kwargs.

        ``ImageAnalyzer.suggest_params`` emits GUI-scale values (matching
        the slider ranges). The engine consumes the same names but with
        per-spec conversions applied. We reuse
        :func:`retouch.params.gui_values_to_engine_kwargs` so the
        conversion is consistent with the GUI path.
        """
        from .params import gui_values_to_engine_kwargs
        return gui_values_to_engine_kwargs(gui_params)


__all__ = [
    "SmartProcessor",
    "SmartSuggestion",
]

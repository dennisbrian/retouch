"""Confidence-aware decisions for automatic retouch stages.

Automatic stages use this contract instead of silently forcing an edit. The
module is processor-agnostic so a stage can record the same evidence whether
it is a classical detector or an optional model. Applying a skipped decision
returns the original array unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np


VALID_ACTIONS = {"apply", "dampen", "skip", "review"}
SAFE_AUTO_POLICY_VERSION = "safe-auto-v2"
MEASURED_CONFIDENCE_SOURCES = {"retinaface"}


def confidence_evidence(value: Any, source: Any) -> dict[str, Any]:
    """Normalize detector confidence without treating compatibility values as evidence.

    MediaPipe landmark APIs used by this project do not always expose a face
    presence score. Their compatibility value of ``1.0`` must therefore never
    enter the automatic-apply band.
    """

    source_name = str(source or "unknown")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    if not np.isfinite(numeric):
        numeric = 0.0
    numeric = float(np.clip(numeric, 0.0, 1.0))
    measured = source_name in MEASURED_CONFIDENCE_SOURCES
    return {
        "reported_confidence": numeric,
        "confidence_source": source_name,
        "confidence_measured": measured,
        # Unmeasured presence remains review-only. It is not zero because the
        # landmarks may still support an explicit user-requested edit.
        "safe_auto_confidence": numeric if measured else 0.50,
        "policy_version": SAFE_AUTO_POLICY_VERSION,
    }


@dataclass(frozen=True)
class SafeAutoDecision:
    stage: str
    action: str
    confidence: float
    reason: str
    evidence: Mapping[str, Any]
    strength_scale: float

    def __post_init__(self) -> None:
        if self.action not in VALID_ACTIONS:
            raise ValueError(f"unknown Safe Auto action: {self.action!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not 0.0 <= self.strength_scale <= 1.0:
            raise ValueError("strength_scale must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "action": self.action,
            "confidence": self.confidence,
            "reason": self.reason,
            "evidence": dict(self.evidence),
            "strength_scale": self.strength_scale,
        }


def record_decision(target: Any, decision: SafeAutoDecision) -> None:
    """Append a serialisable decision to a mutable processing context.

    The helper deliberately accepts ``Any`` because worker processes use a
    lightweight context mapping while the engine uses ``ProcessingContext``.
    Manual Advanced Retouch edits do not call this helper; their explicit edit
    records remain separate from Safe Auto evidence.
    """
    decisions = getattr(target, "_safe_auto_decisions", None)
    if decisions is None and isinstance(target, dict):
        decisions = target.setdefault("_safe_auto_decisions", [])
    if decisions is None:
        decisions = []
        try:
            setattr(target, "_safe_auto_decisions", decisions)
        except Exception:
            return
    decisions.append(decision.to_dict())


def decide(
    stage: str,
    *,
    confidence: float,
    evidence: Optional[Mapping[str, Any]] = None,
    reason: str = "",
    apply_at: float = 0.85,
    dampen_at: float = 0.65,
    review_at: float = 0.40,
) -> SafeAutoDecision:
    """Choose an action from explicit confidence and evidence.

    Thresholds are intentionally supplied by the caller because blemishes,
    geometry, eyes, and model-backed stages have different risk profiles.
    Missing/invalid confidence fails closed to ``review``.
    """
    if not stage:
        raise ValueError("stage is required")
    if not (0.0 <= review_at <= dampen_at <= apply_at <= 1.0):
        raise ValueError("thresholds must satisfy review <= dampen <= apply")
    value = float(confidence)
    facts = dict(evidence or {})
    if not np.isfinite(value):
        value = 0.0
    value = float(np.clip(value, 0.0, 1.0))
    if value >= apply_at:
        action, scale, default_reason = "apply", 1.0, "evidence supports automatic edit"
    elif value >= dampen_at:
        action, scale, default_reason = "dampen", 0.5, "evidence is usable but uncertain"
    elif value >= review_at:
        action, scale, default_reason = "review", 0.0, "evidence requires human review"
    else:
        action, scale, default_reason = "skip", 0.0, "evidence is too uncertain"
    return SafeAutoDecision(
        stage=stage,
        action=action,
        confidence=value,
        reason=reason or default_reason,
        evidence=facts,
        strength_scale=scale,
    )


def decide_mask_stage(
    stage: str,
    *,
    mask_coverage: float,
    landmark_stability: Optional[float] = None,
    occluded: bool = False,
    model_confidence: Optional[float] = None,
) -> SafeAutoDecision:
    """Build a conservative decision from common portrait-stage evidence."""
    coverage = float(np.clip(mask_coverage, 0.0, 1.0))
    stability = 1.0 if landmark_stability is None else float(np.clip(landmark_stability, 0.0, 1.0))
    model = 1.0 if model_confidence is None else float(np.clip(model_confidence, 0.0, 1.0))
    confidence = min(coverage, stability, model)
    if occluded:
        confidence = min(confidence, 0.35)
    reason = "occluded region" if occluded else "mask/landmark/model evidence"
    return decide(
        stage,
        confidence=confidence,
        evidence={
            "mask_coverage": coverage,
            "landmark_stability": stability,
            "model_confidence": model,
            "occluded": bool(occluded),
        },
        reason=reason,
    )


def apply_decision(
    original: np.ndarray,
    candidate: np.ndarray,
    decision: SafeAutoDecision,
) -> np.ndarray:
    """Apply a decision while guaranteeing a byte-identical skip path."""
    if original.shape != candidate.shape:
        raise ValueError("original and candidate must have the same shape")
    if decision.action in {"skip", "review"}:
        return original.copy()
    if decision.action == "apply":
        return candidate.copy()
    alpha = decision.strength_scale
    if np.issubdtype(original.dtype, np.integer):
        blended = np.rint(
            original.astype(np.float32) * (1.0 - alpha)
            + candidate.astype(np.float32) * alpha
        )
        return np.clip(blended, np.iinfo(original.dtype).min, np.iinfo(original.dtype).max).astype(original.dtype)
    return (original.astype(np.float32) * (1.0 - alpha) + candidate.astype(np.float32) * alpha).astype(original.dtype)

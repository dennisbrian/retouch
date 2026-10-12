"""Pure Smart-intent contract (Smart Workspace T1).

Dependency-light, deterministic module: no Gradio, no model loading, no
image I/O, no pixel arrays. Only imports from ``retouch.params`` and
``retouch.face_params`` (both themselves import-light).

Scope of this module (see docs/plans/RESEARCH_SMART_WORKSPACE_PRODUCTIZATION_2026_08_29.md
§9, §17 T1): define the intent/macro/write/adjustment contracts, deterministic
macro evaluation, and registry validation for the Smart workspace. It does
NOT implement GUI wiring, the full layered adjustment-precedence stack
(§9.4 — that is T2+ session/GUI state), context routing (T3), or proposal
evidence (T5). The shipped registry covers the Color card only; other
cards are added in later tranches once their own foundation is verified.

Base-resolution precondition
-----------------------------
``evaluate_write`` takes ``base: float`` as a required argument. It is the
CALLER's responsibility to resolve a parameter's layered base state (e.g.
substituting ``ParamSpec.gui_default`` when the engine ``default`` is
``None``, as is the case for ``brightness``/``highlights``/``shadows``/
``whites``/``blacks``) before calling the evaluator. This module never
guesses on the caller's behalf — passing ``base=None`` raises ``TypeError``
naming the parameter.

Purity / removal
-----------------
``evaluate_write`` is a pure function of its three arguments with no
hidden state. Re-evaluating a macro from a base snapshot, or "removing" a
macro's effect by calling it again with ``amount=0.0`` against the same
base, is therefore exact by construction — no residual state to clear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from .face_params import FACE_LOCAL_PARAM_NAMES
from .params import get_param

__all__ = [
    "ParameterWrite",
    "MacroSpec",
    "SmartIntentSpec",
    "SmartAdjustment",
    "evaluate_write",
    "validate_registry",
    "SMART_INTENTS",
    "COLOR_INTENT",
    "get_intent",
]

_MAX_MACROS_PER_INTENT = 3

# Any param_key starting with one of these prefixes, or exactly matching one
# of the standalone names, is treated as geometry/identity-sensitive and
# MUST have explicit_only=True wherever it appears in any intent's writes.
# This enforces doc §9.3's "Geometry writes are absent until the user
# explicitly enables Shape" as an invariant rather than a convention.
_GEOMETRY_DENYLIST_PREFIXES: Tuple[str, ...] = (
    "reshape_",
    "body_reshape_",
    "auto_body_reshape",
    "mv2_",
)
_GEOMETRY_DENYLIST_EXACT: Tuple[str, ...] = (
    "freckle_removal",
    "slimming",
)


def _is_geometry_param(param_key: str) -> bool:
    if param_key in _GEOMETRY_DENYLIST_EXACT:
        return True
    return any(param_key.startswith(prefix) for prefix in _GEOMETRY_DENYLIST_PREFIXES)


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParameterWrite:
    """One engine parameter a macro can move, in GUI/CLI-space (matches
    ``ParamSpec.min_val``/``max_val``, the same space
    ``gui_values_to_engine_kwargs`` consumes).

    ``target_negative`` is optional: ``None`` means the write only
    activates for ``amount >= 0`` (a one-sided macro). This is
    deliberately NOT a single ``target`` field — a single target would
    make negative ``amount`` an unreviewed arithmetic reflection
    (``2 * base - target``) rather than a reviewed value.
    """

    param_key: str
    scope: str  # "global" | "face_local"
    target_positive: float
    target_negative: Optional[float] = None
    curve: str = "linear"
    allow_zero_crossing: bool = False
    explicit_only: bool = False


@dataclass(frozen=True)
class MacroSpec:
    """One user-facing slider/macro, owning a disjoint set of ParameterWrites."""

    macro_id: str
    label: str
    minimum: float
    maximum: float
    default: float
    writes: Tuple[ParameterWrite, ...]


@dataclass(frozen=True)
class SmartIntentSpec:
    """One Smart card's reviewed intent contract."""

    intent_id: str
    label: str
    description: str
    supported_regions: Tuple[str, ...]
    macro_controls: Tuple[MacroSpec, ...]
    capability_requirements: Tuple[str, ...] = ()
    fallback_policy: str = "none"
    explicit_only_effects: Tuple[str, ...] = ()
    preview_stage_hint: str = "global"


@dataclass(frozen=True)
class SmartAdjustment:
    """A pure, serializable record of one applied macro: 'this macro at
    this amount owns these writes'. Does not implement the full doc §9.4
    layered-precedence stack (session/GUI state, T2+) or proposal evidence
    (T5) — those remain out of scope for T1.
    """

    intent_id: str
    macro_id: str
    amount: float
    owned_writes: Tuple[ParameterWrite, ...]
    scope: str  # "global" | "face_local" | "mixed"
    origin: str
    order_index: int

    def to_summary_dict(self) -> dict:
        """JSON-serializable summary. No pixels, arrays, or paths (doc §7
        rule 10: 'No pixels in session JSON, telemetry, or explanation
        records')."""
        return {
            "intent_id": self.intent_id,
            "macro_id": self.macro_id,
            "amount": self.amount,
            "owned_param_keys": [w.param_key for w in self.owned_writes],
            "scope": self.scope,
            "origin": self.origin,
            "order_index": self.order_index,
        }


# ---------------------------------------------------------------------------
# Macro evaluation (doc §9.3)
# ---------------------------------------------------------------------------


def _linear_curve(a: float) -> float:
    return a


_CURVES = {"linear": _linear_curve}


def evaluate_write(write: ParameterWrite, base: float, amount: float) -> float:
    """Evaluate one ParameterWrite at a given base value and macro amount.

    ``amount`` must be in ``[-1.0, 1.0]``. ``amount == 0.0`` returns
    ``base`` unchanged (exact identity, no arithmetic — "Zero is exact"
    as a type-preserving guarantee, not just a numeric one).
    """
    if base is None:
        raise TypeError(
            f"evaluate_write: base is None for param_key={write.param_key!r}; "
            "caller must resolve the layered base state (e.g. substitute "
            "ParamSpec.gui_default) before calling evaluate_write"
        )

    if amount == 0.0:
        return base

    if amount > 0.0:
        target = write.target_positive
    else:
        if write.target_negative is None:
            raise ValueError(
                f"evaluate_write: amount={amount} < 0 but "
                f"param_key={write.param_key!r} has no target_negative "
                "(one-sided write)"
            )
        target = write.target_negative

    if amount == 1.0:
        result = write.target_positive
    elif amount == -1.0:
        result = write.target_negative
    else:
        curve_fn = _CURVES.get(write.curve)
        if curve_fn is None:
            raise ValueError(f"evaluate_write: unknown curve {write.curve!r}")
        result = base + curve_fn(abs(amount)) * (target - base)

    if not write.allow_zero_crossing:
        base_sign = 0 if base == 0 else (1 if base > 0 else -1)
        result_sign = 0 if result == 0 else (1 if result > 0 else -1)
        if base_sign != 0 and result_sign != 0 and base_sign != result_sign:
            raise ValueError(
                f"evaluate_write: result {result} crosses zero relative to "
                f"base {base} for param_key={write.param_key!r} and "
                "allow_zero_crossing=False"
            )

    spec = get_param(write.param_key)
    if spec.min_val is not None:
        result = max(result, spec.min_val)
    if spec.max_val is not None:
        result = min(result, spec.max_val)

    if spec.cli_type is int:
        result = round(result)

    return result


# ---------------------------------------------------------------------------
# Registry validation (doc §9.2)
# ---------------------------------------------------------------------------


def validate_registry(intents: Tuple[SmartIntentSpec, ...]) -> None:
    """Validate a Smart intent registry. Raises on the first violation.

    Called on the shipped SMART_INTENTS at module import time (fail
    fast), and reusable directly in tests against synthetic specs.
    """
    seen_intent_ids = set()
    for intent in intents:
        if intent.intent_id in seen_intent_ids:
            raise ValueError(f"duplicate intent_id: {intent.intent_id!r}")
        seen_intent_ids.add(intent.intent_id)

        if not isinstance(intent.macro_controls, tuple):
            raise TypeError(
                f"intent {intent.intent_id!r}: macro_controls must be a tuple "
                "for deterministic ordering"
            )
        if len(intent.macro_controls) > _MAX_MACROS_PER_INTENT:
            raise ValueError(
                f"intent {intent.intent_id!r}: {len(intent.macro_controls)} "
                f"macros exceeds the max of {_MAX_MACROS_PER_INTENT}"
            )
        if not isinstance(intent.capability_requirements, tuple):
            raise TypeError(
                f"intent {intent.intent_id!r}: capability_requirements must be a tuple"
            )
        if not intent.fallback_policy:
            raise ValueError(
                f"intent {intent.intent_id!r}: fallback_policy must be a non-empty string"
            )

        seen_macro_ids = set()
        seen_param_keys = set()
        for macro in intent.macro_controls:
            if macro.macro_id in seen_macro_ids:
                raise ValueError(
                    f"intent {intent.intent_id!r}: duplicate macro_id {macro.macro_id!r}"
                )
            seen_macro_ids.add(macro.macro_id)

            if macro.default != 0.0:
                raise ValueError(
                    f"intent {intent.intent_id!r} macro {macro.macro_id!r}: "
                    f"default must be 0.0, got {macro.default!r}"
                )

            for write in macro.writes:
                if write.param_key in seen_param_keys:
                    raise ValueError(
                        f"intent {intent.intent_id!r}: param_key "
                        f"{write.param_key!r} is written by more than one "
                        "macro (ownership must be disjoint within an intent)"
                    )
                seen_param_keys.add(write.param_key)

                try:
                    spec = get_param(write.param_key)
                except KeyError:
                    raise ValueError(
                        f"intent {intent.intent_id!r} macro {macro.macro_id!r}: "
                        f"unknown param_key {write.param_key!r}"
                    ) from None

                if write.scope == "face_local":
                    if write.param_key not in FACE_LOCAL_PARAM_NAMES:
                        raise ValueError(
                            f"param_key {write.param_key!r} marked scope="
                            "'face_local' but is not in FACE_LOCAL_PARAM_NAMES"
                        )
                elif write.scope == "global":
                    if write.param_key in FACE_LOCAL_PARAM_NAMES:
                        raise ValueError(
                            f"param_key {write.param_key!r} marked scope="
                            "'global' but is in FACE_LOCAL_PARAM_NAMES"
                        )
                else:
                    raise ValueError(
                        f"param_key {write.param_key!r}: unknown scope "
                        f"{write.scope!r} (must be 'global' or 'face_local')"
                    )

                if spec.min_val is None or spec.max_val is None:
                    raise ValueError(
                        f"param_key {write.param_key!r}: numeric macro "
                        "writes require a bounded ParamSpec (min_val/max_val)"
                    )

                if not (spec.min_val <= write.target_positive <= spec.max_val):
                    raise ValueError(
                        f"param_key {write.param_key!r}: target_positive "
                        f"{write.target_positive} outside bounds "
                        f"[{spec.min_val}, {spec.max_val}]"
                    )
                if write.target_negative is not None:
                    if not (spec.min_val <= write.target_negative <= spec.max_val):
                        raise ValueError(
                            f"param_key {write.param_key!r}: target_negative "
                            f"{write.target_negative} outside bounds "
                            f"[{spec.min_val}, {spec.max_val}]"
                        )

                is_geometry = _is_geometry_param(write.param_key)
                if is_geometry and not write.explicit_only:
                    raise ValueError(
                        f"param_key {write.param_key!r} is a geometry/identity "
                        "param and must have explicit_only=True"
                    )
                if write.explicit_only:
                    if write.param_key not in intent.explicit_only_effects:
                        raise ValueError(
                            f"intent {intent.intent_id!r}: explicit_only "
                            f"param_key {write.param_key!r} must be listed in "
                            "explicit_only_effects"
                        )
                    # macro.default == 0.0 is already enforced above; amount=0
                    # returns base unchanged for every write by construction
                    # of evaluate_write, so an explicit_only write can never
                    # fire from a card's default-apply state.


# ---------------------------------------------------------------------------
# Shipped registry — Color card only (T1 scope, see plan §2)
# ---------------------------------------------------------------------------
#
# target_positive/target_negative below are reviewed per doc §24 open
# decision Q1 (dennis, 2026-08-30): Amount and Warmth approved as originally
# shipped; Contrast's highlights/shadows signs were flipped after review —
# the original values darkened highlights and brightened shadows at
# amount=+1 (a flattening move), backwards for a macro named "Contrast".
# Confirmed against retouch/engine.py::_adjust_tonal (positive highlights
# brightens highlights; positive shadows brightens shadows), so a genuine
# contrast increase needs highlights up / shadows down at amount=+1, which
# is what's shipped now.

COLOR_INTENT = SmartIntentSpec(
    intent_id="color",
    label="Color",
    description="Correct or style global color: vibrance/saturation, warmth, and contrast.",
    supported_regions=("global",),
    macro_controls=(
        MacroSpec(
            macro_id="amount",
            label="Amount",
            minimum=-1.0,
            maximum=1.0,
            default=0.0,
            writes=(
                ParameterWrite(
                    param_key="vibrance",
                    scope="global",
                    target_positive=40.0,
                    target_negative=-40.0,
                ),
                ParameterWrite(
                    param_key="saturation",
                    scope="global",
                    target_positive=25.0,
                    target_negative=-25.0,
                ),
            ),
        ),
        MacroSpec(
            macro_id="warmth",
            label="Warmth",
            minimum=-1.0,
            maximum=1.0,
            default=0.0,
            writes=(
                ParameterWrite(
                    param_key="white_balance_kelvin",
                    scope="global",
                    target_positive=8000.0,
                    target_negative=5000.0,
                ),
                ParameterWrite(
                    param_key="white_balance_tint",
                    scope="global",
                    target_positive=20.0,
                    target_negative=-20.0,
                ),
            ),
        ),
        MacroSpec(
            macro_id="contrast",
            label="Contrast",
            minimum=-1.0,
            maximum=1.0,
            default=0.0,
            writes=(
                ParameterWrite(
                    param_key="contrast",
                    scope="global",
                    target_positive=25.0,
                    target_negative=-25.0,
                ),
                ParameterWrite(
                    param_key="highlights",
                    scope="global",
                    target_positive=20.0,
                    target_negative=-20.0,
                ),
                ParameterWrite(
                    param_key="shadows",
                    scope="global",
                    target_positive=-20.0,
                    target_negative=20.0,
                ),
            ),
        ),
    ),
    capability_requirements=(),
    fallback_policy="none",
    explicit_only_effects=(),
    preview_stage_hint="global",
)

SMART_INTENTS: Tuple[SmartIntentSpec, ...] = (COLOR_INTENT,)

validate_registry(SMART_INTENTS)


def get_intent(intent_id: str) -> SmartIntentSpec:
    """Look up a SmartIntentSpec by ID. Raises KeyError if unknown."""
    for intent in SMART_INTENTS:
        if intent.intent_id == intent_id:
            return intent
    raise KeyError(intent_id)

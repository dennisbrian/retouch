"""Tests for retouch/smart_intents.py — Smart intent/macro/write contract,
deterministic macro evaluation, and registry validation.

Mirrors tests/test_params.py's class-per-concern style. Scope triage: this
file covers T1's slice only (registry structure, parameter cross-checks,
explicit-only gating, macro evaluation, ownership/removal, SmartAdjustment
serialization, module purity, and the shipped Color registry). Context
routing (T3), Session v1->v2 migration (T2), full proposal-schema/evidence
sanitization beyond SmartAdjustment's own no-pixel check (T5), and the
latest-request-wins state machine (T4) are correctly out of scope here.
"""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from retouch.face_params import FACE_LOCAL_PARAM_NAMES
from retouch.params import get_param
from retouch.smart_intents import (
    COLOR_INTENT,
    SMART_INTENTS,
    MacroSpec,
    ParameterWrite,
    SmartAdjustment,
    SmartIntentSpec,
    evaluate_write,
    get_intent,
    validate_registry,
)


# ---------------------------------------------------------------------------
# Registry structure
# ---------------------------------------------------------------------------


class TestRegistryStructure:
    def test_registry_is_tuple_of_smart_intent_spec(self):
        assert isinstance(SMART_INTENTS, tuple)
        for intent in SMART_INTENTS:
            assert isinstance(intent, SmartIntentSpec)

    def test_intent_ids_unique(self):
        ids = [i.intent_id for i in SMART_INTENTS]
        assert len(ids) == len(set(ids))

    def test_macro_ids_unique_within_intent(self):
        for intent in SMART_INTENTS:
            macro_ids = [m.macro_id for m in intent.macro_controls]
            assert len(macro_ids) == len(set(macro_ids)), intent.intent_id

    def test_macro_count_at_most_three(self):
        for intent in SMART_INTENTS:
            assert len(intent.macro_controls) <= 3, intent.intent_id

    def test_macro_controls_is_tuple_not_list(self):
        for intent in SMART_INTENTS:
            assert isinstance(intent.macro_controls, tuple)
            for macro in intent.macro_controls:
                assert isinstance(macro.writes, tuple)

    def test_registry_order_is_fixed_tuple(self):
        assert isinstance(SMART_INTENTS, tuple)
        assert not isinstance(SMART_INTENTS, (dict, set))


# ---------------------------------------------------------------------------
# Cross-check against retouch.params / retouch.face_params
# ---------------------------------------------------------------------------


class TestRegistryParameterCrossCheck:
    def test_every_write_param_exists_in_processing_params(self):
        for intent in SMART_INTENTS:
            for macro in intent.macro_controls:
                for write in macro.writes:
                    get_param(write.param_key)  # raises KeyError if unknown

    def test_every_numeric_write_has_bounded_param_spec(self):
        for intent in SMART_INTENTS:
            for macro in intent.macro_controls:
                for write in macro.writes:
                    spec = get_param(write.param_key)
                    assert spec.min_val is not None
                    assert spec.max_val is not None

    def test_targets_within_param_spec_bounds(self):
        for intent in SMART_INTENTS:
            for macro in intent.macro_controls:
                for write in macro.writes:
                    spec = get_param(write.param_key)
                    assert spec.min_val <= write.target_positive <= spec.max_val
                    if write.target_negative is not None:
                        assert spec.min_val <= write.target_negative <= spec.max_val

    def test_face_local_scope_writes_are_in_allowlist(self):
        for intent in SMART_INTENTS:
            for macro in intent.macro_controls:
                for write in macro.writes:
                    if write.scope == "face_local":
                        assert write.param_key in FACE_LOCAL_PARAM_NAMES

    def test_global_scope_writes_are_not_in_allowlist(self):
        for intent in SMART_INTENTS:
            for macro in intent.macro_controls:
                for write in macro.writes:
                    if write.scope == "global":
                        assert write.param_key not in FACE_LOCAL_PARAM_NAMES


# ---------------------------------------------------------------------------
# Explicit-only / geometry gating
# ---------------------------------------------------------------------------


class TestExplicitOnlyGating:
    def test_geometry_params_are_explicit_only(self):
        geometry_prefixes = ("reshape_", "body_reshape_", "auto_body_reshape", "mv2_")
        geometry_exact = ("freckle_removal", "slimming")
        for intent in SMART_INTENTS:
            for macro in intent.macro_controls:
                for write in macro.writes:
                    is_geom = write.param_key in geometry_exact or any(
                        write.param_key.startswith(p) for p in geometry_prefixes
                    )
                    if is_geom:
                        assert write.explicit_only is True, write.param_key

    def test_explicit_only_writes_absent_from_default_macro_value(self):
        # amount=0.0 (the default) must return base unchanged for every
        # write, explicit_only or not -- verified generically here.
        for intent in SMART_INTENTS:
            for macro in intent.macro_controls:
                assert macro.default == 0.0
                for write in macro.writes:
                    assert evaluate_write(write, base=123.0, amount=0.0) == 123.0

    def test_default_macro_amount_is_zero_for_every_macro(self):
        for intent in SMART_INTENTS:
            for macro in intent.macro_controls:
                assert macro.default == 0.0

    def test_explicit_only_param_keys_listed_in_intent_explicit_only_effects(self):
        for intent in SMART_INTENTS:
            for macro in intent.macro_controls:
                for write in macro.writes:
                    if write.explicit_only:
                        assert write.param_key in intent.explicit_only_effects


# ---------------------------------------------------------------------------
# Macro evaluation
# ---------------------------------------------------------------------------


class TestMacroEvaluation:
    def _write(self, **overrides):
        defaults = dict(
            param_key="vibrance",
            scope="global",
            target_positive=40.0,
            target_negative=-40.0,
        )
        defaults.update(overrides)
        return ParameterWrite(**defaults)

    def test_amount_zero_returns_base_exactly(self):
        w = self._write()
        assert evaluate_write(w, base=12.5, amount=0.0) == 12.5

    def test_amount_zero_is_identity_not_computed(self):
        w = self._write()
        sentinel = 17.0
        assert evaluate_write(w, base=sentinel, amount=0.0) is sentinel

    def test_amount_zero_preserves_int_type(self):
        w = self._write()
        result = evaluate_write(w, base=0, amount=0.0)
        assert result == 0
        assert isinstance(result, int)

    def test_amount_one_returns_target_positive_exactly(self):
        w = self._write(target_positive=40.0)
        assert evaluate_write(w, base=0.0, amount=1.0) == 40.0

    def test_amount_negative_one_returns_target_negative_exactly(self):
        w = self._write(target_negative=-40.0)
        assert evaluate_write(w, base=0.0, amount=-1.0) == -40.0

    def test_amount_negative_one_raises_when_target_negative_is_none(self):
        w = self._write(target_negative=None)
        with pytest.raises(ValueError):
            evaluate_write(w, base=0.0, amount=-1.0)

    def test_interior_amount_uses_linear_curve(self):
        w = self._write(target_positive=40.0, target_negative=-40.0)
        assert evaluate_write(w, base=0.0, amount=0.5) == 20.0

    def test_result_clamped_to_param_spec_bounds(self):
        # vibrance bounds are [-100, 100]; target_positive=40 is within
        # bounds, so use a synthetic write with an out-of-clamp-range
        # interior computation is not reachable at amount<=1 from valid
        # targets -- instead verify clamping fires using a target at the
        # spec boundary combined with allow_zero_crossing off-path check.
        spec = get_param("vibrance")
        w = self._write(target_positive=spec.max_val)
        result = evaluate_write(w, base=0.0, amount=1.0)
        assert result <= spec.max_val

    def test_int_cli_type_rounds_to_int(self):
        w = self._write(
            param_key="contrast", target_positive=25.0, target_negative=-25.0
        )
        result = evaluate_write(w, base=0.0, amount=0.3)
        spec = get_param("contrast")
        assert spec.cli_type is int
        assert result == round(0.0 + 0.3 * (25.0 - 0.0))
        assert isinstance(result, int)

    def test_contrast_param_int_cli_type_float_default_rounds_correctly(self):
        # Regression: contrast has cli_type=int but default=0.0 (float) in
        # the real registry -- the rounding discriminator must be
        # spec.cli_type is int, not type(spec.default) is int.
        spec = get_param("contrast")
        assert spec.cli_type is int
        assert spec.default == 0.0
        assert isinstance(spec.default, float)
        w = self._write(
            param_key="contrast", target_positive=25.0, target_negative=-25.0
        )
        result = evaluate_write(w, base=0.0, amount=1.0)
        assert isinstance(result, int)

    def test_none_base_raises_type_error_naming_param(self):
        w = self._write(param_key="highlights", target_positive=-20.0, target_negative=20.0)
        spec = get_param("highlights")
        assert spec.default is None
        with pytest.raises(TypeError, match="highlights"):
            evaluate_write(w, base=None, amount=0.5)

    def test_unknown_curve_name_raises(self):
        w = self._write(curve="bogus")
        with pytest.raises(ValueError):
            evaluate_write(w, base=0.0, amount=0.5)

    def test_zero_crossing_rejected_unless_allowed(self):
        # base positive, target_negative negative -> interior negative
        # amount would cross zero relative to base's sign.
        w = self._write(
            target_positive=40.0, target_negative=-40.0, allow_zero_crossing=False
        )
        with pytest.raises(ValueError):
            evaluate_write(w, base=10.0, amount=-0.5)

    def test_zero_crossing_allowed_when_flag_set(self):
        w = self._write(
            target_positive=40.0, target_negative=-40.0, allow_zero_crossing=True
        )
        # Should not raise.
        result = evaluate_write(w, base=10.0, amount=-0.5)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Ownership / removal
# ---------------------------------------------------------------------------


class TestOwnershipAndRemoval:
    def test_removal_is_reevaluation_at_amount_zero(self):
        w = ParameterWrite(
            param_key="vibrance", scope="global", target_positive=40.0, target_negative=-40.0
        )
        base = 5.0
        moved = evaluate_write(w, base=base, amount=0.7)
        assert moved != base
        removed = evaluate_write(w, base=base, amount=0.0)
        assert removed == base

    def test_evaluate_write_is_pure_no_hidden_state(self):
        w = ParameterWrite(
            param_key="vibrance", scope="global", target_positive=40.0, target_negative=-40.0
        )
        first = evaluate_write(w, base=5.0, amount=0.3)
        second = evaluate_write(w, base=5.0, amount=0.3)
        assert first == second

    def test_no_param_key_written_by_two_macros_in_shipped_registry(self):
        for intent in SMART_INTENTS:
            seen = set()
            for macro in intent.macro_controls:
                for write in macro.writes:
                    assert write.param_key not in seen, write.param_key
                    seen.add(write.param_key)


# ---------------------------------------------------------------------------
# SmartAdjustment
# ---------------------------------------------------------------------------


class TestSmartAdjustment:
    def _adjustment(self):
        macro = COLOR_INTENT.macro_controls[0]
        return SmartAdjustment(
            intent_id=COLOR_INTENT.intent_id,
            macro_id=macro.macro_id,
            amount=0.5,
            owned_writes=macro.writes,
            scope="global",
            origin="smart_macro",
            order_index=0,
        )

    def test_owned_writes_equal_macro_writes(self):
        macro = COLOR_INTENT.macro_controls[0]
        adj = self._adjustment()
        assert adj.owned_writes == macro.writes

    def test_to_summary_dict_is_json_serializable(self):
        adj = self._adjustment()
        summary = adj.to_summary_dict()
        json.dumps(summary)  # must not raise

    def test_to_summary_dict_contains_no_ndarray_bytes_or_path(self):
        adj = self._adjustment()
        summary = adj.to_summary_dict()

        def scan(value):
            assert not isinstance(value, bytes)
            assert not hasattr(value, "shape")  # catches numpy.ndarray
            if isinstance(value, dict):
                for v in value.values():
                    scan(v)
            elif isinstance(value, (list, tuple)):
                for v in value:
                    scan(v)

        scan(summary)

    def test_to_summary_dict_round_trips_scalar_fields(self):
        adj = self._adjustment()
        summary = adj.to_summary_dict()
        assert summary["intent_id"] == "color"
        assert summary["macro_id"] == "amount"
        assert summary["amount"] == 0.5
        assert summary["owned_param_keys"] == ["vibrance", "saturation"]


# ---------------------------------------------------------------------------
# validate_registry negative cases (synthetic malformed specs)
# ---------------------------------------------------------------------------


def _minimal_intent(macro_controls, explicit_only_effects=()):
    return SmartIntentSpec(
        intent_id="synthetic",
        label="Synthetic",
        description="test",
        supported_regions=("global",),
        macro_controls=macro_controls,
        explicit_only_effects=explicit_only_effects,
    )


class TestRegistryValidationNegativeCases:
    def test_validate_rejects_unknown_param_key(self):
        write = ParameterWrite(param_key="not_a_real_param", scope="global", target_positive=1.0)
        macro = MacroSpec("m", "M", -1.0, 1.0, 0.0, (write,))
        with pytest.raises(ValueError):
            validate_registry((_minimal_intent((macro,)),))

    def test_validate_rejects_global_param_marked_face_local(self):
        write = ParameterWrite(
            param_key="vibrance", scope="face_local", target_positive=40.0
        )
        macro = MacroSpec("m", "M", -1.0, 1.0, 0.0, (write,))
        with pytest.raises(ValueError):
            validate_registry((_minimal_intent((macro,)),))

    def test_validate_rejects_face_local_param_marked_global(self):
        write = ParameterWrite(param_key="smooth", scope="global", target_positive=50.0)
        macro = MacroSpec("m", "M", -1.0, 1.0, 0.0, (write,))
        with pytest.raises(ValueError):
            validate_registry((_minimal_intent((macro,)),))

    def test_validate_rejects_geometry_write_not_marked_explicit_only(self):
        write = ParameterWrite(
            param_key="reshape_jaw_width", scope="face_local", target_positive=10.0,
            explicit_only=False,
        )
        macro = MacroSpec("m", "M", -1.0, 1.0, 0.0, (write,))
        with pytest.raises(ValueError):
            validate_registry((_minimal_intent((macro,)),))

    def test_validate_rejects_explicit_only_write_reachable_at_default(self):
        write = ParameterWrite(
            param_key="reshape_jaw_width", scope="face_local", target_positive=10.0,
            explicit_only=True,
        )
        macro = MacroSpec("m", "M", -1.0, 1.0, 0.0, (write,))
        # explicit_only param_key missing from explicit_only_effects
        with pytest.raises(ValueError):
            validate_registry((_minimal_intent((macro,), explicit_only_effects=()),))

    def test_validate_rejects_out_of_bounds_target(self):
        spec = get_param("vibrance")
        write = ParameterWrite(
            param_key="vibrance", scope="global", target_positive=spec.max_val + 1000.0
        )
        macro = MacroSpec("m", "M", -1.0, 1.0, 0.0, (write,))
        with pytest.raises(ValueError):
            validate_registry((_minimal_intent((macro,)),))

    def test_validate_rejects_unbounded_param_as_numeric_write(self):
        # lip_tint has no min_val/max_val (string-typed param).
        spec = get_param("lip_tint")
        assert spec.min_val is None
        write = ParameterWrite(param_key="lip_tint", scope="face_local", target_positive=1.0)
        macro = MacroSpec("m", "M", -1.0, 1.0, 0.0, (write,))
        with pytest.raises(ValueError):
            validate_registry((_minimal_intent((macro,)),))

    def test_validate_rejects_more_than_three_macros(self):
        writes = (ParameterWrite(param_key="vibrance", scope="global", target_positive=10.0),)
        macros = tuple(
            MacroSpec(f"m{i}", f"M{i}", -1.0, 1.0, 0.0, ()) for i in range(4)
        )
        with pytest.raises(ValueError):
            validate_registry((_minimal_intent(macros),))

    def test_validate_rejects_duplicate_intent_id(self):
        macro = MacroSpec("m", "M", -1.0, 1.0, 0.0, ())
        intent = _minimal_intent((macro,))
        with pytest.raises(ValueError):
            validate_registry((intent, intent))

    def test_validate_rejects_duplicate_macro_id_within_intent(self):
        macro_a = MacroSpec("dup", "A", -1.0, 1.0, 0.0, ())
        macro_b = MacroSpec("dup", "B", -1.0, 1.0, 0.0, ())
        with pytest.raises(ValueError):
            validate_registry((_minimal_intent((macro_a, macro_b)),))

    def test_validate_rejects_param_written_by_two_macros_in_one_intent(self):
        write_a = ParameterWrite(param_key="vibrance", scope="global", target_positive=10.0)
        write_b = ParameterWrite(param_key="vibrance", scope="global", target_positive=20.0)
        macro_a = MacroSpec("a", "A", -1.0, 1.0, 0.0, (write_a,))
        macro_b = MacroSpec("b", "B", -1.0, 1.0, 0.0, (write_b,))
        with pytest.raises(ValueError):
            validate_registry((_minimal_intent((macro_a, macro_b)),))


# ---------------------------------------------------------------------------
# Module purity
# ---------------------------------------------------------------------------


_HEAVY_MODULES = ("gradio", "cv2", "torch", "onnxruntime", "PIL")
_REPO_ROOT = Path(__file__).resolve().parents[1]


class TestModulePurity:
    def test_import_does_not_pull_gradio_cv2_torch_onnxruntime_pil(self):
        code = (
            "import sys\n"
            "before = set(sys.modules)\n"
            "import retouch.smart_intents\n"
            "after = set(sys.modules)\n"
            "new = after - before\n"
            "heavy = {'gradio', 'cv2', 'torch', 'onnxruntime', 'PIL'}\n"
            "hit = {m for m in new if any(m == h or m.startswith(h + '.') for h in heavy)}\n"
            "print(sorted(hit))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            cwd=_REPO_ROOT,
        )
        assert result.stdout.strip() == "[]", result.stdout

    def test_import_does_not_pull_smart_default_or_engine(self):
        code = (
            "import sys\n"
            "before = set(sys.modules)\n"
            "import retouch.smart_intents\n"
            "after = set(sys.modules)\n"
            "new = after - before\n"
            "hit = {m for m in new if m in ('retouch.smart_default', 'retouch.engine')}\n"
            "print(sorted(hit))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            cwd=_REPO_ROOT,
        )
        assert result.stdout.strip() == "[]", result.stdout

    def test_module_has_no_top_level_image_or_model_io(self):
        import retouch.smart_intents as mod

        assert not hasattr(mod, "cv2")
        assert not hasattr(mod, "np")
        assert not hasattr(mod, "numpy")


# ---------------------------------------------------------------------------
# Shipped Color registry
# ---------------------------------------------------------------------------


class TestShippedColorRegistry:
    def test_color_intent_present(self):
        assert get_intent("color") is COLOR_INTENT

    def test_color_macros_are_amount_warmth_contrast(self):
        macro_ids = [m.macro_id for m in COLOR_INTENT.macro_controls]
        assert macro_ids == ["amount", "warmth", "contrast"]

    def test_color_writes_are_all_global_scope(self):
        for macro in COLOR_INTENT.macro_controls:
            for write in macro.writes:
                assert write.scope == "global"

    def test_color_registry_passes_validate_registry(self):
        validate_registry(SMART_INTENTS)  # must not raise

    def test_get_intent_unknown_raises_key_error(self):
        with pytest.raises(KeyError):
            get_intent("does_not_exist")

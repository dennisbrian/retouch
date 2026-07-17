"""CLI/engine contracts for the opt-in identity-mark policy."""

from __future__ import annotations

import pytest

from retouch.engine import build_context, resolve_recipe
from retouch.params import get_param, recipe_to_params


def test_mark_policy_is_a_registry_parameter_with_cli_choices():
    spec = get_param("mark_policy")
    assert spec.cli_flag == "mark-policy"
    assert tuple(spec.choices or ()) == ("legacy", "protect_identity", "preserve_all")
    assert recipe_to_params("natural")["mark_policy"] == "legacy"


def test_context_resolves_named_mark_policy_without_changing_legacy():
    recipe = resolve_recipe("natural")
    legacy = build_context("natural", recipe, {})
    protected = build_context("natural", recipe, {"mark_policy": "protect_identity"})

    assert legacy.mark_policy is None
    assert protected.mark_policy is not None
    assert protected.mark_policy["mole"]["action"] == "preserve"


def test_context_rejects_unknown_mark_policy():
    with pytest.raises(ValueError, match="Unknown mark policy"):
        build_context("natural", resolve_recipe("natural"), {"mark_policy": "unsafe"})

"""Keep public parameter references aligned with the ParamSpec registry."""

from pathlib import Path

from retouch.params import PROCESSING_PARAMS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PARAMETER_DOCS = (
    PROJECT_ROOT / "docs/architecture/API.md",
    PROJECT_ROOT / "docs/guides/RECIPE_GUIDE.md",
)


def test_parameter_registry_is_covered_by_public_references():
    """Every registered control remains discoverable in both public references."""
    expected = {f"`{spec.name}`" for spec in PROCESSING_PARAMS}

    for path in PUBLIC_PARAMETER_DOCS:
        content = path.read_text(encoding="utf-8")
        missing = sorted(name for name in expected if name not in content)
        assert not missing, f"{path.relative_to(PROJECT_ROOT)} is missing: {missing}"

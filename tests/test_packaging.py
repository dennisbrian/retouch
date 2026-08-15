"""Static checks for the portable frozen-app packaging source of truth."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_desktop_spec_is_portable_and_includes_package_data():
    spec = (ROOT / "scripts" / "build" / "retouch_app.spec").read_text(encoding="utf-8")
    assert "SPECPATH" in spec
    assert "/Applications/htdocs" not in spec
    for package_name in ("retouch", "models", "presets"):
        assert f'ROOT / "{package_name}"' in spec


def test_build_wrapper_uses_spec_instead_of_divergent_data_flags():
    script = (ROOT / "scripts" / "build" / "build_app.sh").read_text(encoding="utf-8")
    assert 'SPEC_PATH="$SCRIPT_DIR/retouch_app.spec"' in script
    assert '"${PYINSTALLER_CMD[@]}" --clean "$SPEC_PATH"' in script
    assert "--add-data" not in script


def test_lut_assets_are_declared_for_wheels_and_frozen_apps():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    spec = (ROOT / "scripts" / "build" / "retouch_app.spec").read_text(encoding="utf-8")

    assert '"luts"' in pyproject
    assert 'luts = ["*.cube", "ACQUISITION.md"]' in pyproject
    assert '(str(ROOT / "luts"), "luts")' in spec
    assert (ROOT / "luts" / "__init__.py").is_file()
    assert (ROOT / "luts" / "kodak.cube").is_file()
    assert (ROOT / "luts" / "fuji.cube").is_file()

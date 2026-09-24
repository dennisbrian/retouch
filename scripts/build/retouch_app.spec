# -*- mode: python ; coding: utf-8 -*-
"""Portable PyInstaller source of truth for the desktop application.

Keep repository-relative paths here.  The build wrapper invokes this spec from
any working directory, and the same data list is used for local and release
builds so presets cannot disappear from a frozen app.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules


ROOT = Path(SPECPATH).resolve().parents[1]
APP_NAME = "Pro Max Retouch Studio"

# desktop.py imports ``gui`` and ``webview`` with importlib so the launcher
# stays headless-safe; PyInstaller cannot see those imports, so every
# runtime dependency has to be declared here or the frozen app exits with
# "No module named 'gui'" at launch.
hiddenimports = [
    "gui",
    "gui_advanced",
    "gui_batch",
    "gui_shoot",
    "gui_smart",
    "webview",
    *collect_submodules("retouch"),
]
datas = [
    (str(ROOT / "retouch"), "retouch"),
    (str(ROOT / "models"), "models"),
    (str(ROOT / "presets"), "presets"),
    (str(ROOT / "luts"), "luts"),
]
binaries = []
# Packages that load data files (templates, graphs, JSON schemas) at runtime.
for package in ("gradio", "gradio_client", "safehttpx", "groovy", "mediapipe", "webview"):
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    except Exception:  # optional package missing from this environment
        continue
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden


a = Analysis(
    [str(ROOT / "desktop.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
    # Gradio reads its own .py sources (component templates) at runtime.
    module_collection_mode={"gradio": "py"},
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME,
)
app = BUNDLE(
    coll,
    name=f"{APP_NAME}.app",
    icon=None,
    bundle_identifier=None,
)

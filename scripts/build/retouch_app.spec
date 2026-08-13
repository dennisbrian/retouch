# -*- mode: python ; coding: utf-8 -*-
"""Portable PyInstaller source of truth for the desktop application.

Keep repository-relative paths here.  The build wrapper invokes this spec from
any working directory, and the same data list is used for local and release
builds so presets cannot disappear from a frozen app.
"""

from pathlib import Path


ROOT = Path(SPECPATH).resolve().parents[1]
APP_NAME = "Pro Max Retouch Studio"


a = Analysis(
    [str(ROOT / "desktop.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        (str(ROOT / "retouch"), "retouch"),
        (str(ROOT / "models"), "models"),
        (str(ROOT / "presets"), "presets"),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
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

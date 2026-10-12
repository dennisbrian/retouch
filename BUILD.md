# BUILD — packaging Pro Max Retouch Studio for distribution

Audience: maintainer producing shippable builds. For development setup see `docs/CONTRIBUTING.md`.

## Prerequisites (macOS)

- Python 3.11 with the locked desktop environment (`./setup`, or `uv sync --locked --extra desktop`)
- PyInstaller (declared by the desktop extra)
- For signed/notarized builds:
  - Apple Developer Program membership
  - **Developer ID Application** certificate in Keychain
  - notarytool keychain profile:
    ```bash
    xcrun notarytool store-credentials "promax-notary" \
        --apple-id you@example.com --team-id TEAMID
    ```

## Build

Unsigned dev build (Gatekeeper will block on other machines):

```bash
bash scripts/build/build_app.sh
```

Signed + notarized release build:

```bash
export CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"
export NOTARY_PROFILE="promax-notary"
bash scripts/build/build_app.sh
```

Output: `dist/Pro Max Retouch Studio.app` (+ notarized `.dmg` for a release build).
The portable source of truth is `scripts/build/retouch_app.spec`; it includes
the `retouch`, `models`, and `presets` package data directories.

App-only signed build without DMG/notarization:

```bash
SKIP_DMG=1 CODESIGN_IDENTITY="..." bash scripts/build/build_app.sh
```

## Verifying a build

```bash
codesign --verify --deep --strict dist/"Pro Max Retouch Studio".app
spctl --assess --type open --context context:primary-signature -vv dist/"Pro Max Retouch Studio".app
xcrun stapler validate dist/"Pro Max Retouch Studio".dmg
spctl --assess --type open -vv dist/"Pro Max Retouch Studio".dmg
```

`RELEASE_BUILD=1` makes signing, notarization, stapling, and both `spctl`
checks mandatory. The macOS release workflow also uploads and attests the
archived `.app` bundle and DMG; configure its Developer ID and App Store
Connect secrets before dispatching it.

Fresh-machine smoke test (the P4 acceptance gate): copy the DMG to a Mac
without dev tools → install → first launch downloads models with progress →
process a photo. Gatekeeper must not block.

## Windows and Linux

PyInstaller does not cross-compile: each platform's app is built on that
platform from the same spec. The **Desktop builds** workflow
(`.github/workflows/build-desktop.yml`, run it from the Actions tab) builds
unsigned apps on macOS, Windows and Linux, launches each one until its local
server answers, and uploads the three builds as artifacts.

To build by hand on Windows or Linux:

```bash
uv sync --locked --extra desktop
# Linux only: pywebview needs a GUI backend. Qt installs from wheels.
uv pip install PyQt6 PyQt6-WebEngine qtpy
uv run --no-sync pyinstaller --clean --noconfirm scripts/build/retouch_app.spec
```

The result is `dist/Pro Max Retouch Studio/` (run the `.exe` on Windows).
Linux targets also need the system OpenGL/EGL and xcb libraries listed in the
workflow. The Linux build was verified on 2026-09-23 (window opens, a photo
renders end to end); the Windows build has only the workflow's launch check so
far, so test it on real hardware before shipping it. The ONNX runtime provider
selection (`build_ort_providers`, DirectML) is platform-aware. If pywebview
misbehaves on a platform, ship "open in browser" mode (`gui.py`) as the
fallback.

### Frozen-build pitfalls

- `desktop.py` loads `gui` and `webview` with `importlib`, so PyInstaller cannot
  see them. The spec lists them (and collects Gradio, MediaPipe and pywebview
  data) explicitly. A build of about 30 MB means dependencies are missing, and
  the app exits with `No module named 'gui'`.
- The engine's face worker pool uses the `spawn` start method. `desktop.py`
  calls `multiprocessing.freeze_support()`; without it every worker boots
  another copy of the app and renders never finish.

## Version + update check

Single source of truth: `retouch/__init__.py::__version__`. The GUI does a
non-blocking check against GitHub releases on launch and shows a toast if a
newer tag exists. Releases must be tagged `v<major>.<minor>.<patch>[-codename]`
for the check to parse them.

## Diagnostics

GUI and desktop-app startup attach a rotating `retouch.log` (1 MB x 3) in
`~/Library/Logs/ProMaxRetouch` on macOS, `%LOCALAPPDATA%\ProMaxRetouch\Logs` on
Windows, and `~/.cache/promaxretouch/logs` on Linux. Users can copy a version/environment bundle from the
🩺 Diagnostics accordion in the GUI — ask for it on any bug report.

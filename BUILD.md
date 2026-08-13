# BUILD — packaging Pro Max Retouch Studio for distribution

Audience: maintainer producing shippable builds. For development setup see `docs/CONTRIBUTING.md`.

## Prerequisites (macOS)

- Python 3.9+ with the locked desktop environment (`uv sync --locked --extra desktop`)
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

## Windows

The same PyInstaller entry (`desktop.py`) builds on Windows; the ONNX
runtime provider selection (`build_ort_providers`, DirectML) is already
platform-aware. Not yet validated on real Windows hardware — timeboxed
validation pending. If PyWebView misbehaves there, ship "open in browser"
mode (`gui.py`, which is platform-independent) as the fallback.

## Version + update check

Single source of truth: `retouch/__init__.py::__version__`. The GUI does a
non-blocking check against GitHub releases on launch and shows a toast if a
newer tag exists. Releases must be tagged `v<major>.<minor>.<patch>[-codename]`
for the check to parse them.

## Diagnostics

GUI startup attaches a rotating log at `~/Library/Logs/ProMaxRetouch/retouch.log`
(1 MB x 3). Users can copy a version/environment bundle from the
🩺 Diagnostics accordion in the GUI — ask for it on any bug report.

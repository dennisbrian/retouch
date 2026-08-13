#!/bin/bash
# Standalone PyInstaller build script for Pro Max Retouch Studio
#
# Signing/notarization (macOS): set these env vars before running to produce
# a Gatekeeper-passing build. Without them the build is unsigned (dev only).
#   CODESIGN_IDENTITY   e.g. "Developer ID Application: Your Name (TEAMID)"
#   NOTARY_PROFILE      keychain profile for notarytool (xcrun notarytool store-credentials)
# Optional:
#   SKIP_DMG=1          skip the DMG packaging + notarization step (app only)

set -e

APP_NAME="Pro Max Retouch Studio"

echo "=== 1. Installing Desktop Dependencies ==="
if command -v uv >/dev/null 2>&1 && [ -f uv.lock ]; then
    echo "Using locked uv environment"
    uv sync --locked --extra desktop
    PYINSTALLER_CMD=(uv run --locked pyinstaller)
else
    echo "uv/uv.lock unavailable; using requirements/desktop.txt fallback"
    python3 -m pip install -r requirements/desktop.txt
    PYINSTALLER_CMD=(python3 -m PyInstaller)
fi

echo "=== 2. Compiling Standalone Desktop Application ==="
# On macOS, --windowed creates a double-clickable .app bundle
# We copy package files and the models/ directory into the application bundle
"${PYINSTALLER_CMD[@]}" --windowed \
            --name "$APP_NAME" \
            --add-data "retouch:retouch" \
            --add-data "models:models" \
            desktop.py

APP_PATH="dist/$APP_NAME.app"

if [ -n "$CODESIGN_IDENTITY" ]; then
    echo "=== 3. Code Signing ==="
    # Deep sign, runtime hardening required for notarization
    codesign --deep --force --options runtime \
             --sign "$CODESIGN_IDENTITY" \
             --timestamp \
             "$APP_PATH"
    codesign --verify --deep --strict --verbose=2 "$APP_PATH"
else
    echo "=== 3. Code Signing SKIPPED (set CODESIGN_IDENTITY to enable) ==="
fi

if [ -n "$CODESIGN_IDENTITY" ] && [ -n "$NOTARY_PROFILE" ] && [ "$SKIP_DMG" != "1" ]; then
    echo "=== 4. DMG Packaging + Notarization ==="
    DMG_PATH="dist/$APP_NAME.dmg"
    [ -f "$DMG_PATH" ] && rm "$DMG_PATH"
    hdiutil create -volname "$APP_NAME" -srcfolder "$APP_PATH" -ov -format UDZO "$DMG_PATH"
    codesign --sign "$CODESIGN_IDENTITY" --timestamp "$DMG_PATH"

    xcrun notarytool submit "$DMG_PATH" --keychain-profile "$NOTARY_PROFILE" --wait
    xcrun stapler staple "$DMG_PATH"
    echo "Notarized DMG: $DMG_PATH"
else
    echo "=== 4. DMG/Notarization SKIPPED (needs CODESIGN_IDENTITY + NOTARY_PROFILE) ==="
fi

echo "=== Build Complete ==="
echo "App: $APP_PATH"

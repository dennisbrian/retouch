#!/bin/bash
# Standalone PyInstaller build script for Pro Max Retouch Studio
#
# Signing/notarization (macOS): set these env vars before running to produce
# a Gatekeeper-passing build. Without them the build is unsigned (dev only).
#   CODESIGN_IDENTITY   e.g. "Developer ID Application: Your Name (TEAMID)"
#   NOTARY_PROFILE      keychain profile for notarytool (xcrun notarytool store-credentials)
#   APPLE_ID / APPLE_TEAM_ID / APPLE_APP_PASSWORD  App Store Connect credentials
#     alternative to NOTARY_PROFILE (use an app-specific password)
# Optional:
#   SKIP_DMG=1          skip the DMG packaging + notarization step (app only)

set -e

APP_NAME="Pro Max Retouch Studio"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SPEC_PATH="$SCRIPT_DIR/retouch_app.spec"

if [ ! -f "$SPEC_PATH" ]; then
    echo "Missing portable PyInstaller spec: $SPEC_PATH" >&2
    exit 1
fi

cd "$REPO_ROOT"

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
# The spec is the single packaging source of truth. It includes retouch,
# manifests/models, and presets using paths relative to the repository.
"${PYINSTALLER_CMD[@]}" --clean "$SPEC_PATH"

APP_PATH="dist/$APP_NAME.app"

if [ -n "${CODESIGN_IDENTITY:-}" ]; then
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

NOTARY_MODE=""
if [ -n "${NOTARY_PROFILE:-}" ]; then
    NOTARY_MODE="keychain-profile"
elif [ -n "${APPLE_ID:-}" ] && [ -n "${APPLE_TEAM_ID:-}" ] && [ -n "${APPLE_APP_PASSWORD:-}" ]; then
    NOTARY_MODE="apple-id"
fi

if [ "${RELEASE_BUILD:-0}" = "1" ] && { [ -z "${CODESIGN_IDENTITY:-}" ] || [ -z "$NOTARY_MODE" ] || [ "${SKIP_DMG:-0}" = "1" ]; }; then
    echo "RELEASE_BUILD=1 requires signing, notarization credentials, and a DMG" >&2
    exit 1
fi

if [ -n "${CODESIGN_IDENTITY:-}" ] && [ -n "$NOTARY_MODE" ] && [ "${SKIP_DMG:-0}" != "1" ]; then
    echo "=== 4. DMG Packaging + Notarization ==="
    DMG_PATH="dist/$APP_NAME.dmg"
    [ -f "$DMG_PATH" ] && rm "$DMG_PATH"
    hdiutil create -volname "$APP_NAME" -srcfolder "$APP_PATH" -ov -format UDZO "$DMG_PATH"
    codesign --sign "$CODESIGN_IDENTITY" --timestamp "$DMG_PATH"

    if [ "$NOTARY_MODE" = "keychain-profile" ]; then
        xcrun notarytool submit "$DMG_PATH" --keychain-profile "$NOTARY_PROFILE" --wait
    else
        xcrun notarytool submit "$DMG_PATH" \
            --apple-id "$APPLE_ID" \
            --team-id "$APPLE_TEAM_ID" \
            --password "$APPLE_APP_PASSWORD" \
            --wait
    fi
    xcrun stapler staple "$DMG_PATH"
    # These are release gates: a successful notarization submission alone is
    # insufficient if the ticket cannot be stapled or Gatekeeper rejects it.
    xcrun stapler validate "$DMG_PATH"
    spctl --assess --type open --context context:primary-signature -vv "$APP_PATH"
    spctl --assess --type open -vv "$DMG_PATH"
    echo "Notarized DMG: $DMG_PATH"
else
    echo "=== 4. DMG/Notarization SKIPPED (needs signing + notarization credentials) ==="
fi

echo "=== Build Complete ==="
echo "App: $APP_PATH"

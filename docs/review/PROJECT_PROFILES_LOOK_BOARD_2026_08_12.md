# Project Profiles and Look Board — 2026-08-12

Implemented `retouch/project_profiles.py` and the Shoot Intelligence GUI
controls.

- Subject-linked profiles store recipe/style preferences and protected-mark
  metadata without storing face pixels.
- Look Boards store multiple weighted references, labels, notes, and an
  optional linked profile ID.
- Profiles and boards use atomic JSON persistence through the shared Retouch
  cache resolver (`RETOUCH_CACHE_DIR`, then XDG cache), isolating tests and
  environments without a separate home-directory store.
- Duplicate Look Board references are replaced explicitly rather than silently
  accumulating stale copies.
- GUI actions expose profile and board creation while the original images
  remain untouched. **Apply Look Board to Process** extracts each available
  reference independently, combines scalar engine parameters by the saved
  weight, and feeds the normal editable look-parameter state; it never
  replaces faces or mutates source captures.

The future style learner may consume these metadata records only after consent
and retention policy are defined. It must remain inspectable and must not
generate a replacement face.

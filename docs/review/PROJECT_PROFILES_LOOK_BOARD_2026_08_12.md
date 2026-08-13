# Project Profiles and Look Board — 2026-08-12

Implemented `retouch/project_profiles.py` and the Shoot Intelligence GUI
controls.

- Subject-linked profiles store recipe/style preferences and protected-mark
  metadata without storing face pixels.
- Look Boards store multiple weighted references, labels, notes, and an
  optional linked profile ID.
- Profiles and boards use atomic JSON persistence under the user's Retouch
  project state directory.
- Duplicate Look Board references are replaced explicitly rather than silently
  accumulating stale copies.
- GUI actions expose profile and board creation while the original images
  remain untouched.

The future style learner may consume these metadata records only after consent
and retention policy are defined. It must remain inspectable and must not
generate a replacement face.

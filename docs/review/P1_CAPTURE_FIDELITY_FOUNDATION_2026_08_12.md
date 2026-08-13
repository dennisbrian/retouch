# P1/P1.1 Capture Fidelity Foundation — 2026-08-12

Implemented foundation:

- `retouch/capture_fidelity.py` reads camera, lens, focal length, aperture,
  ISO, shutter, and orientation from EXIF without guessing missing values.
- Calibration profiles are keyed by camera and select the nearest ISO/shutter
  settings without silently matching another camera.
- RAW-mosaic calibration supports dark-frame subtraction, flat-field
  normalization, and explicit hot-pixel replacement before demosaicing.
- Lensfun availability is reported explicitly. No creative RGB displacement or
  vignette operation is treated as optical correction.

Remaining P1 work is intentionally separate: execute Lensfun corrections with
the local database, add verified camera profiles, and build the optional
ColorChecker/gray-card wizard. Those require hardware/database fixtures and
must retain a no-op default.

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
- `--optical-correction` exposes requested Lensfun work and its fallback reason
  in the CLI; the GUI has a matching **Apply Lensfun corrections** control and
  includes the per-image result in its completion status.
- The installed Lensfun adapter is used only for verified uint8 input. A
  uint16/float capture is explicitly left unchanged and reports a
  precision-preserving skip; it is never converted to uint8 and cast back.

Remaining P1 work is intentionally separate: add camera/lens fixtures against
a real Lensfun database (including distortion, TCA, and vignetting), provide a
precision-preserving non-8-bit correction backend, add verified camera
profiles, and build the optional ColorChecker/gray-card wizard. Those require
hardware/database fixtures and must retain a no-op default.

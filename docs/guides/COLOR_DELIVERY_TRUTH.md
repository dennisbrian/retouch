# Color, Precision, and Delivery Truth

Retouch now makes the color and precision boundary explicit. The GUI and API
should describe the actual pixel contract, not the container format alone.

## Color contract

The processing path is:

```text
source pixels + source profile
    -> ColorContext decode
    -> working-space sRGB
    -> creative processing
    -> export profile and metadata policy
```

`retouch.io` returns a `ColorContext` alongside context-aware image reads:

- an embedded ICC source is converted to the engine's sRGB working space;
- an untagged non-RAW source is treated as assumed-sRGB and recorded as such;
- RAW input is recorded as raw-srgb until a RAW-specific scene-referred
  contract is introduced.

The default export is the processed working-space sRGB image. The GUI's
**Preserve source ICC profile on export** option is the explicit opt-in for
converting the processed pixels back to the source profile. Re-embedding an
ICC profile without that conversion is not a valid color-managed export.

The context and its profile hashes are included in the pixel-free render
manifest. An old C2PA APP11 payload remains a copied, unverified passthrough;
it is not treated as a newly signed derived-work claim.

## Precision contract

The current engine's final processing boundary is still uint8. A source may
be 16-bit or floating point, but the result reports the effective processed
precision and any downgrade through `ProcessingResult.precision` and the
render manifest.

Consequently, **PNG-16 is currently a 16-bit container carrying processed
data whose effective levels may be 8-bit**. It must not be advertised as
true 16-bit processing until float/high-bit output is preserved end to end.

## Preview and inspection

`Render Preview` is a fast, non-delivery render. `Export Full Quality` captures
its own settings snapshot and uses the full-quality path. A preview can be
inspected at Fit, 100%, Face, or ROI through `retouch.gui_inspection`; the
crop is native-resolution when the source/render evidence is available and is
revision-gated so an old preview cannot be presented as current. Inspection
output has downloading disabled.

## Render evidence

`retouch.render_manifest.RenderManifest` records a sanitized, pixel-free
snapshot of the render: source/output identity, dimensions and dtype, color
context, effective precision, cache status, face evidence, timings, QA,
backend/provider, settings hash, and stale status. This is an operational
render manifest, separate from Certification Evidence v2 and its human-review
gates.

## Current release boundary

The runtime doctor and manifest may report `global_only` when the installed
detector backend is unavailable. A completed render in that state is not
face-aware certification evidence. Certification still requires actual
per-image detector evidence and the separate corpus, automatic-quality, and
human-review gates.

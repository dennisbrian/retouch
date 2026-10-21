# Compositor port: format and renderer notes (milestones 1–2)

Date: 2026-10-07. Companion to [COMPOSITOR_PYTHON_PORT.md](COMPOSITOR_PYTHON_PORT.md).
Upstream pinned at `11d8d7a`; provenance and license in
[third_party/compositor/PROVENANCE.md](../../third_party/compositor/PROVENANCE.md).

## Published docs versus source

Where the two disagree the port follows the source (`ProjectStore.swift`,
`DocumentLimits.swift`):

| Topic | `docs/project-format.md` | Source at 11d8d7a | Port |
| --- | --- | --- | --- |
| Total pixels | "100 million total source pixels" | `documentPixelBudget` = min(800 MP, max(200 MP, RAM / 16)), counted separately for images and masks | 200 MP each for images and masks: the floor every upstream build accepts |
| Single surface (export) | not stated | `maxSurfacePixels` = 200 MP | 200 MP |
| Side, layers, manifest, asset size | 30,000 px, 10,000 layers, 4 MiB, 512 MiB | same | same |
| Transform keys | example shows all six | synthesized `Codable`: all six are required | all six required |
| Unknown JSON keys | not stated | ignored by `JSONDecoder` (silently lost on save) | refused with `UnsupportedFeature`, so nothing is lost |
| Asset bit depth | "8-bit PNGs" | rejects depth > 8; masks must be 8-bit gray, no alpha | requires exactly 8-bit; masks 8-bit gray, no alpha |

The handoff writer (`retouch/compositor.py`) keeps its own tighter
three-layer budget (100 M RGB samples); it is a bridge policy, not the format
limit.

## Declared subset (read and write)

Read: format versions 1–11, with upstream's version gating (opacity and
blend mode from 3, masks from 4, guides from 8, text colour/font runs from
10/11). Written: version 11.

Supported: ungrouped pixel layers and blank layers, bottom-to-top order,
visibility, opacity, any of the 24 blend-mode names (stored), resizing with
fractional origins rendered at subpixel accuracy (layers may hang off the
canvas), flips, an 8-bit raster
mask with enabled flag (any mask size; 1×1 uniform masks broadcast), document
resolution, active layer, guides (kept verbatim), text and shape metadata
(kept verbatim; the PNG stays authoritative and the metadata is dropped when
the pixels are replaced, as upstream does on a destructive edit).

Refused with `UnsupportedFeature` (valid upstream, not modelled yet):
folders (`isGroup`, `parentID`), clipping masks (`maskSourceID`), adjustment
layers, unlinked folder masks, fractional layer sizes, unknown
keys. Unlinked image-layer masks render: `maskPlacement` is a full
`LayerTransform`; the mask is resampled into the layer's own pixel grid
(`pixelToDocument(placement) * pixelToDocument(layer).inverted()`) over a
dominant-border-gray background, then rides the layer's transform like a
linked one (upstream `LayerMask.clipImage`). `maskLinked` rides along
round-trip; it governs interactive moves upstream, not rendering. Blend modes render for
the full `RENDER_BLEND_MODES` set (all 24 separable and component modes);
hidden modes outside it are kept and skipped. The handoff's
Difference inspection layer can now be revealed.

Refused with `ProjectError` (invalid or over a limit), before any pixels are
decoded: wrong format or colour space, unsupported version, bad UUIDs or
duplicate ids, image/mask files not named `<id>.png` / `<id>.mask.png`, paths
that leave the package or are symlinks, missing or non-PNG assets, assets over
the side, pixel or byte limits, masks with colour or alpha, non-8-bit PNGs.

## Retouch recipe extension (retouch-recipes.json)

The manifest stays at native format version 11; recipe provenance is a
Retouch-specific sidecar inside the transactional package, added by `6a366ff`:

- `retouch-recipes.json`: `{version: 1, documentID, layers: {<layer-id>: {
  settings, source_sha256, result_sha256}}}`. `settings` is
  `retouch/editor/recipe_state.py`'s validated state: `{version: 1, recipe
  (str ≤200), overrides, source_revision}`. `overrides` keys are engine
  ParamSpec names (`smooth`, `whiten`, `iris_pop`, `catchlight`, `blush`,
  `nose_tip_blush`, `face_shadow_fill`, `texture_opacity` — the latter set
  added by `034ab60`); each value is validated against the engine's
  ParamSpec min/max, not a fixed 0–100.
- Per editable recipe layer: a `<layer-id>.recipe-source.png` RGBA asset
  holding the immutable original composite, counted in the image budget.
- Read path: header/schema/checksum validation; `pixel_digest` (sha256 of
  raw RGBA bytes) must match both the source asset and the current layer
  pixels, otherwise the package is refused ("recipe source/result changed
  outside Retouch"). Unknown layer ids, foreign document ids and non-hex
  checksums are refused. Absent file (and non-symlink) loads as a plain
  raster document.
- Size cap: the JSON must fit `MAX_MANIFEST_BYTES` (4 MiB) or save fails.
- Fast save (`project_store`) remembers the source PNG per array so unchanged
  recipe sources copy rather than re-encode; the saved-array cache lock is
  re-entrant since `568cd2c`.
- Other apps may discard the extension; Retouch refuses the mismatch if a
  retained sidecar disagrees with edited rasters.

Not stored: clone anchors (`aa1c80f`), local-retouch drafts, wand/selection
state — all session-only, never persisted.

## Colour and precision

- The 2026-10-08 finishing slice adds Multiply, Screen, Overlay and Difference;
  a later slice extends to all 24 stored modes (20 separable plus the Hue,
  Saturation, Color and Luminosity component modes, which use the W3C/PDF-32000
  non-separable formulas behind Core Image's blend filters).
  Formulas follow [W3C Compositing and Blending Level 1](https://www.w3.org/TR/compositing-1/#blending):
  blend straight RGB only in overlapping source/backdrop coverage, then apply
  source-over with layer opacity and enabled mask. No backdrop means source
  colour is retained, even for Multiply. Overlay switches on backdrop colour.
  These operate in gamma-encoded sRGB. Existing Normal arithmetic is retained.
  Independent pixel/alpha checks, band invariance, save/reopen and undo are
  tested. Native Compositor parity for the newer modes remains unmeasured.
- Resizing stores target width/height separately from unchanged source pixels.
  Nearest uses nearest-neighbour, Smooth uses bilinear, High quality uses Lanczos
  RGBA interpolation through premultiplied alpha. Masks span the rendered
  rectangle, then flip with pixels. The side limit is 30,000 pixels and summed
  rendered image areas must stay within 200 MP, checked before PNG decode or
  render allocation. These bound extra scaled buffers, not total process RSS.
  Original source dimensions remain authoritative for pixel replacement.
- Working space: gamma-encoded sRGB, as upstream. No linearisation; the
  existing Retouch canvases are 8-bit working sRGB already.
- Arrays are RGB/RGBA. Retouch's BGR stays at the engine boundary.
- Layers hold straight alpha. The compositor accumulates premultiplied
  float32 and returns straight-alpha uint8 on a transparent canvas, like
  `ImageExporter.render` (Core Graphics, `premultipliedLast`).
- Source-over per layer: coverage = alpha × mask × opacity.
- Rounding: once, at the end. Core Graphics rounds the 8-bit premultiplied
  canvas after every layer, so stacks of translucent layers can differ from
  the app by about one level per layer; opaque, fully revealed and fully
  hidden pixels match exactly. Upstream's own pixel tests allow ±0.02.
- 1:1 layers keep their pixels without resampling; sampling controls scaled
  pixels and resized masks. Native-app scaled-render parity is not yet measured.
- Saving never claims more than 8 bits: PNG assets are 8-bit, and float
  buffers do not recover precision.

## Evidence (2026-10-07, cloud sandbox, Python 3.11)

Numerical checks (`tests/test_editor_composite.py`,
`tests/test_editor_project_store.py`): the upstream pixel expectations for
Normal and opacity (0.8 / 0.6 / 0.4 gray), mask coverage `[255, 0, 128, 255]`,
mask × 50% opacity, disabled masks, flipped masks; random-image analytic
source-over within 1 level across band boundaries; round trips and every
rejection above.

Real photos (`scripts/dev/compositor_reference.py`, Alex's DSCF3503 and
DSCF3518 at 4160×6240, `cosplay_clear_v1`, rendered with `--max-dim 2048`):

| Check | DSCF3503 | DSCF3518 |
| --- | --- | --- |
| Handoff reopened and flattened == render | exact | exact |
| Result layer hidden == original | exact | exact |
| Result layer at 50% vs analytic blend | ≤ 1 level | ≤ 1 level |
| Save + reload, every layer | identical | identical |
| Load / flatten / save+reload | 7.2 s / 2.4 s / 37 s | 7.1 s / 2.6 s / 35 s |
| Peak RSS (whole script, 3 full-size layers) | 1.9 GB | 1.9 GB |

These compare the port against itself and against the engine output; they
are not a comparison with the Mac app's own render. Parity with Compositor's
renderer needs the app (or Xcode) on a Mac: open the same package there,
export a PNG and compare. Saving is slow because three 26 MP PNGs are
compressed at Pillow's default level; that is a milestone 6 item.

## UI runtime spike

PySide6-Essentials, in throwaway virtual environments outside the project:

- Resolves together with the project's locked runtime on Python 3.9
  (PySide6 6.10.3) and 3.11 (6.11.2) without moving any pin (numpy 1.26.4,
  opencv-contrib 4.11.0.86, mediapipe 0.10.5, protobuf 3.20.3 unchanged).
- About 233 MB installed (Essentials only, no WebEngine).
- Widgets could not be started in the cloud sandbox (`libEGL.so.1` missing),
  so no window was shown; that needs a desktop.
- License: LGPLv3 (Qt for Python), usable in a closed or paid app if it is
  dynamically linked and replaceable. Note that the Linux desktop build already
  installs **PyQt6** for pywebview, which is GPLv3 or commercial; that is a
  licensing question for a paid build independent of this port.

Recommendation, not yet a decision: start milestone 3 as a web canvas inside
the existing pywebview/Gradio shell (no new dependency, works on every
supported Python), and revisit PySide6 only if canvas painting at native
resolution proves too slow there. Either way the editor core stays UI-free.

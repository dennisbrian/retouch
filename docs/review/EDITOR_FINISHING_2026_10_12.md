# Editor finishing and retained-work review — 2026-10-12

Base: `main` fast-forwarded to `1bb2a6ff4c982c30545e1cc17a0dbc864715916e`.
The user authorized cleanup/acceptance work and the full remaining editor
backlog categories. Another Claude agent is working on a different task;
this extension owns the editor package, focused tests and related handoff notes.

## Implemented scope

- Non-destructive clockwise center rotation, source/mask alignment, save/reopen,
  bounded transformed surfaces and rotation-aware fit-to-canvas.
- Rectangle, ellipse and polygon selections in canvas coordinates; feather,
  replace/add/subtract/intersect, invert/clear and selection-to-layer-mask.
  Selections remain transient until applied to a persistent mask.
- Pass-through folders, inherited opacity/visibility, transformed folder masks,
  parent reassignment, adjacent-sibling grouping and hierarchy-aware layer listing.
  Duplicate copies descendants and remaps internal clipping links. Removing a
  folder removes descendants; ungroup refuses to lose mask/opacity/visibility effects.
- Native `maskSourceID` links, cycle/dangling-reference rejection before asset decode,
  adjacent sibling clipping stacks that retain base alpha, and nonadjacent live
  alpha dependencies independent of source visibility.
- Exposure, Levels and Invert adjustment layers with editable controls, masks,
  opacity, blending and undo. Exposure uses the upstream linear-light transfer;
  Levels applies color-channel ranges then combined RGB. Backdrop alpha is preserved.
- Folder and adjustment mask brushing in local, unrotated orientation.
- A durable pinned source archive and corrected current handoff text.

The manifest stays at native format version 11; these use existing native fields.
Unsupported adjustment kinds, fractional layer sizes and unlinked folder masks
are rejected explicitly. Fractional origins are supported and rendered at
subpixel accuracy; unlinked image-layer masks (`maskPlacement`) render and
round-trip. All seven upstream layer effect kinds (stroke, drop shadow,
colour overlay, inner shadow, outer glow, inner glow) render with upstream's
compose order, margins and sigma conventions, and shadow offsets shift
bilinearly like upstream's `effects_shift`. This is the requested
category backlog, not every feature of the native app.

## Evidence

Focused suites cover document/history changes, composition, storage, UI callbacks,
resize, recovery, subprocess recipe jobs and the existing bridge. New independent
contracts cover analytic folder opacity, clipping alpha, rotation direction,
transparent color edges, transformed selections, grading transfers, graph rejection,
transactional failures and native save/reopen. See `tests/test_editor_finishing.py`.

Browser interaction on an isolated local editor at `http://127.0.0.1:7865` used
an actual photo project: opened the native-format folder, changed layer rotation
from 30 to 45 degrees, created a canvas rectangle and applied it as a mask,
added a 0.5-stop Exposure layer, saved and reopened the six-layer project,
and exported a 640 × 640 PNG. The standalone preview used the compatibility
shim already present in `gui.py`; no dependency or launcher changes were made.
The normal application shell was then exercised for a real pointer-drawn mask stroke; packaged desktop remains a separate acceptance check.

Local artifacts are under `test_output/editor_acceptance_2026_10_12/` (ignored):

- `native-parity.comp`, `python-reference.png`, `native-export.png`,
  `native-parity-initial.json`, `native-comparison.png`.
- `finishing-native.comp`, `finishing-python.png`, `finishing-native-export.png`,
  `finishing-native-parity.json`, `finishing-comparison.png`,
  `finishing-rotation-crop.png`.
- `browser-finished.comp` and per-photo profile folders with `report.json` and preview.

The installed native Compositor opened both generated fixtures and exported PNGs
through its own UI. The original fixture covers Normal/Multiply/Screen/Overlay/
Difference, resize/flips, masks, partial alpha and negative/clipped placement.
Its maximum opaque RGB difference is 2 levels; alpha differs by at most 2 levels.
Mean premultiplied RGB error is 0.101 levels, maximum 2.30.

The finishing fixture covers folder masking, rotation, a clipped Screen layer,
and a masked Exposure adjustment over a photo. Alpha is exact. Mean premultiplied
RGB error is 0.052 levels, with maximum 45 on the aliased rotated-mask boundary;
244 opaque pixels exceed 5 levels. The side-by-side and amplified crop were
visually inspected. Straight RGB errors at alpha 1 can be large from native
premultiplied quantization, so alpha-weighted errors are reported as well.
These measurements establish compatible loading and meaningful visual comparison,
not pixel-exact renderer parity. The rotated-mask resampling follow-up is now
implemented (see "Rotated-mask and resampling parity" below).

## Rotated-mask and resampling parity

Core Graphics resamples a layer's mask as a **separate coverage pass**, not as a
premultiplied twin of the pixels: native draws the layer image with the mask
applied as a clip (`context.clip(to:mask:)`), each with its own filter. The CPU
renderer now mirrors that split in `_rotate`, and follows the native policy in
`_resample`/`_downsample`:

- Large reductions prefilter through a chain of exactly-2x halvings made with
  high-quality (Lanczos) resampling, premultiplied and ringing-clamped, so only
  the last at-most-2x step is left — matching `DownsampleCache`.
- Upright 1:1 draws copy across (nearest); a shrink takes the low-quality path;
  an enlarge and every rotated draw keep the layer's own sampling.
- Core Graphics' "low" filter is a **4-tap cubic, not bilinear**. Its impulse
  response is `15/63/192/240` out of 512 (a cubic with a ~= 0.5-class kernel),
  versus bilinear's `0/64/192/256`. Probed directly from the API with a
  one-pixel impulse; the final resample step now uses `cv2.INTER_CUBIC`.

Ground truth is captured from the real API: a 7x5 gray mask rotated 33 degrees
into a 13x11 canvas via `clip(to:mask:)` (`.low`/`.medium`/`.high` give the same
grid for a grayscale clip). Regenerate with `swift scripts/dev/cg_rotated_mask_fixture.swift`.
`test_rotated_mask_matches_core_graphics_reference` renders the same case and
asserts agreement within a documented tolerance: mean error ~3 levels, 95th
percentile under 40. The residual is Core Graphics' own subpixel/edge convention,
which an affine warp cannot reproduce byte-for-byte; the test pins it so a
regression cannot hide.

### Adjustment-kernel parity

`Retouch`'s Exposure / Levels / Invert records are now pinned against Compositor's
real grader kernel, not a transcription. `scripts/dev/cg_adjustment_fixture.swift`
links the actual `Rendering/LevelsPixels.c` (pinned 11d8d7a) and runs it over a
256-step RGB ramp plus low-alpha edge colours for seven settings: three Exposure
curves (including `-2` stops with offset and a `0.5` gamma), three Levels
variants (per-channel gamma, non-default black/white and output range, combined
RGB pass), and Invert. The captured premultiplied bytes live in
`tests/fixtures/cg_adjustment_reference.txt`; `test_adjustment_pixels_match_native_kernel_fixture`
drives the same ramp through the Python compositor and compares.

Native contract confirmed from the kernel: un-premultiply `p*255/alpha`, index a
256-entry curve table with linear interpolation, then `clamp(round(result*alpha))`
back into premultiplied bytes; alpha is untouched and `alpha == 0` pixels are
skipped. Python evaluates the same transfer continuously on float straight-alpha.
On the opaque ramp the two agree within **one output level** for every setting
(the residual is the table's own 1/255 interpolation, which `cv2`-free float
evaluation does not round the same way) — a corrupted record (e.g. swapping red
and blue gammas) moves an opaque pixel by 64 levels, so the fixture is not
vacuous. Translucent pixels are compared as premultiplied bytes, the native
storage form, and also land within one level.

## Native-resolution Mac profiling

Each separate Python process used a pass-through folder, one native photo layer,
a quarter-size rotated/masked overlay, and an Exposure adjustment. Results are
wall time on this Mac, including local disk I/O. They are editor workloads,
not Retouch recipe or face-quality certification.

| Photo | Canvas | Preview | Composite | First save | Metadata re-save | Reopen | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DSCF7142 | 4160 × 6240 | 0.193 s | 1.997 s | 2.535 s | 0.030 s | 0.934 s | 953.7 MiB |
| DSCF4503 | 4160 × 6240 | 0.169 s | 2.122 s | 2.355 s | 0.022 s | 1.066 s | 955.3 MiB |
| p7_visual/source | 3879 × 5850 | 0.151 s | 1.873 s | 2.314 s | 0.012 s | 0.937 s | 761.8 MiB |

All three reopened composites have exactly the original composite SHA-256.
Package sizes are 38.1, 29.3 and 26.9 MiB respectively. Larger/more numerous
native layers and arbitrary rotated layers can use substantially more memory.

### Large-stack / maximum-size characterization

Measured with `scripts/dev/editor_scale_benchmark.py`; each case runs in its
own process with synthetic uint8 layers, `ru_maxrss` in MiB. `MAX_SIDE` is
30,000 px and `MAX_SURFACE_PIXELS` is 200 MP.

| Case | Surface / layers | Composite | Peak RSS | Result |
| --- | --- | ---: | ---: | --- |
| `canvas_8000` | 64 MP / 1 | 1.50 s | 899.5 MiB | completed |
| `canvas_12000` | 144 MP / 1 | 3.39 s | 1,619.1 MiB | completed |
| `canvas_14142` | 200 MP / 1 | 4.80 s | 1,757.5 MiB | completed at pixel limit |
| `canvas_200mp_4` | 200 MP / 4 × 50 MP | 14.98 s | 1,741.4 MiB | completed at pixel limit |
| `stack_16` | 12 MP / 16 layers, 192 MP layer budget | 3.56 s | 987.3 MiB | completed |
| `stack_64` | 12 MP / 64 × 0.75 MP, 48 MP layer budget | 11.26 s | 408.4 MiB | completed |
| `stack_over_budget` | 12 MP / 64 × 12 MP, 768 MP layer budget | — | 2,362.0 MiB | refused: `document exceeds the 200 MP layer budget` |
| `save_12000` | 144 MP / 1 | 3.23 s | 2,948.9 MiB peak incl. save | save 3.38 s; reload 1.68 s |

Conclusion: banded composite keeps canvas working memory bounded enough to
finish the 200 MP limit on this 18 GB Mac, but maximum-size save peaks near
3 GB and deep stacks cost time more than canvas size (64 small layers at 12 MP
out-pace 4 x 50 MP by composite count, not area). The 200 MP layer-budget guard
rejects an oversized stack before composite. These are synthetic CPU editor
limits, not a shipping OOM guarantee; packaged desktop recovery and Windows
acceptance remain separate gates.

### Dirty preview resampling

`preview_document` now memoizes each scaled layer/mask by frozen source-buffer
identity, source shape, and preview size. `DocumentHistory` snapshots share
unchanged read-only buffers, so ordinary metadata edits reuse their preview
pixels; replacing a layer buffer invalidates only that layer. Cache entries hold
weak source references and are capped at 64, avoiding a second full-resolution
undo-buffer budget. `test_preview_cache_reuses_unchanged_layers_and_invalidates_replaced_pixels`
compares real composite pixels across cold/warm/replaced renders and counts
the actual resize calls: two layer resizes plus one mask resize on cold render,
zero on warm render, and one additional layer resize after replacement.

The 16-bit TIFF delivery path now carries the same document resolution as PNG:
tags 282/283 store X/Y pixels-per-inch and tag 296 declares inches. The export
test verifies 16-bit samples, ICC profile, alpha, and all three resolution tags
from the real encoded TIFF.

## Repository housekeeping disposition

No unrelated files were staged, committed, reverted or deleted.

| Retained item | Current evidence | Disposition |
| --- | --- | --- |
| Historical feature-count report | Existing untracked file, ends at pinned October 7 revision | Preserved and linked from the docs index; remains a historical audit |
| `stash@{0}` | September 24 saved docs; all 10 added index lines are already present, but only 1 of 6 added CLAUDE lines matches literally | Preserve; do not blindly pop or drop |
| Agent worktree at `de346b1` | Clean, with a unique EXR fix; `git cherry main` reports it as unapplied | Preserve for its owner; integration is a separate code review |
| `retouch-s1` at `09002e0` | Clean detached worktree; no commits outside main | Preserve while another agent is active; retirement can be done after ownership is settled |
| EXP-0001 A/B/C, EXP-0002 A, EXP-0003 A | Experiment worktrees at exact frozen baseline tips | Preserve as experiment evidence; do not treat them as disposable cache |
| Temporary Compositor reference | Original temp tree contains directories but no usable source/Git objects; original archive is absent | Replaced with durable `third_party/compositor/source-11d8d7a.tar.gz` and recorded checksum |

## Remaining gates

- More native adjustment fixtures: **closed** by the real-kernel ramp fixture above.
- Representative larger stacks and maximum-size/RSS characterization: **closed**
  for the CPU editor limits above; packaged desktop crash recovery remains.
- Packaged desktop crash recovery and platform acceptance, including Windows.
- High-bit/color-managed editor export: **closed** by `f45176d`; dirty-preview
  resampling cache now **closed** for the interactive preview path. Optional
  native acceleration remains separate.
- Component blend modes (Hue/Saturation/Color/Luminosity) and the stroke,
  drop-shadow, colour-overlay layer effects: implemented and verified against
  the W3C/PDF-32000 formulas and the pinned upstream Swift source, plus band
  invariance fuzzing. Native Compositor parity for these additions is
  unmeasured; the existing five-mode contact-sheet fixture predates them.
- Inner shadow, outer glow and inner glow (`7868c9f`): the remaining effect
  kinds render with upstream's compose order (shadow → outer glow → outside
  stroke → layer → overlay → inner glow → inner shadow → inside stroke),
  margin math (`ceil(max(...))+2`, outer-glow `size*3` plus the port's own
  band-margin terms for inner effects) and `sigma = size/2` glows. Verified
  by an upstream line-by-line math audit (sigmas, coverage formulas, masks,
  defaults, validation ranges) and a 300-doc randomized fuzz; the fuzz
  also caught and fixed whole-pixel shadow-offset snapping (now bilinear
  `_shift_float`, matching `effects_shift`) and band-margin misses.
- Unlinked image-layer masks (`da25dca`): `maskPlacement` renders per the
  export path's `LayerMask.clipImage` — the mask is fitted to its placement
  size, warped into the layer's pixel grid through
  `pixelToDocument(placement) * pixelToDocument(layer).inverted()` over its
  dominant border gray, then rides the layer's transform. `maskLinked`
  round-trips (it governs interactive moves upstream, not rendering).
  Folder placements stay refused, matching the export path. Fuzzing here
  also fixed a pre-existing `_rotate` bug: rotated coverage warps with
  `BORDER_REPLICATE` now (a full-coverage mask used to clip the layer's
  alpha at the rotation bounding-box corners).

The saved stash and unique EXR worktree are retained work, not cleanup failures.
They require their owners' context before retirement or integration.

## Follow-up after the concurrent commits

While this verification was running, another agent committed the finishing work
as `d2aae51` and added a folder-mask propagation test as `23dcf12`; the local
checkout moved to that committed main. The original untracked feature-count
report was included in that concurrent commit. The retained stash and worktrees
were not removed. The follow-up brush correction is a separate scoped change.

Actual pointer interaction exposed a Gradio palette problem: selecting black
still produced a white stroke, and the saved mask stayed entirely at 255.
The editor now has an explicit Reveal/Hide action, with colored strokes showing
coverage. Applying a stroke uses its alpha support and the selected action,
independent of palette RGB. Color-based programmatic callers keep their existing
behaviour. Tests cover white, black and colored strokes under both actions,
undo, invalid-action rejection and the actual UI callback.

The normal full Retouch shell on a separate loopback port opened the 400 × 600
photo project, drew a Hide stroke with the pointer, applied it and saved. The
saved mask ranges from 0 to 255, with 6,161 pixels below 255; exactly 6,161 composite
pixels changed. Both layers' source arrays remain exact. Evidence:
`brush-hide-fixed.comp`, `brush-hide-check.json`, `brush-hide-fixed.png` and
`brush-photo-comparison.png` under the same ignored acceptance directory.

The full shell also reports a pre-existing `denoise_nafnet` model checksum mismatch:
expected `4bab67c8618c722f57c90c0112f4ac57cfeea6dd83cbf5a7e8fbac5c0f995e65`,
actual `a2b1aa5a5c9d901779948faad92930e13a25b314c89b12cbee4d1e1d9f6cbbf7`.
A manifest-pinned release copy was downloaded and verified in temporary storage.
By the time replacement was considered, the local file already matched the
expected hash (another process repaired it); this agent did not replace it.
`retouch.model_fetch.verify_model('denoise_nafnet')` now succeeds for the local
file, and its CPU inference returns finite output at shape `[1,3,32,32]`.
This clears the observed integrity warning; it does not certify real-photo denoising.
Evidence: `model-repair/report.json`. No dependencies or engine files were changed.

`scripts/dev/local-gate` passed: 423 tests, governance 0 errors/0 warnings.
Its full Quality Lab threshold step remains skipped by the fast gate, as the
script reports; this is not a real-photo recipe-quality certification.

Final focused editor/bridge suite before concurrent renderer edits: 256 passed.
Governance: 0 errors/0 warnings; `git diff --check` clean. The real pointer mask
can be undone and saved: its mask returns to absent and the reopened composite
is exactly the pre-stroke composite (`brush-undo-fixed.comp`, with results added
to `brush-hide-check.json`).

At the end of this pass, another agent had uncommitted downsampling/resampling
changes in `retouch/editor/composite.py`. The 256-test result and native comparison
measurements above predate those edits and must not be treated as certification
of that concurrent renderer revision. This agent's final follow-up files are
`retouch/editor/ui.py`, `tests/test_editor_ui.py`, this report and the Compositor
plan. Renderer ownership/validation is being coordinated with the user.

Before the scoped brush commit, `tests/test_editor_ui.py` passed all 37 tests
with the current shared workspace, and governance again reported zero errors
and warnings. Concurrent renderer/resampling test edits are excluded from this
commit; the broader native-parity qualification remains their owner's work.

## Changelog after the 2026-10-12 review (ff3516a..e765e06)

Editor-side commits landed since this review's base; each entry verified
against its commit diff.

- **JPEG export and Select→Modify** (`b8f0d8f`): `export_jpeg` in
  `retouch/editor/composite.py` flattens over an opaque sRGB background in
  gamma-encoded space (upstream `ImageExporter.swift:125-133` parity), embeds
  the sRGB ICC profile, honours quality (0..1) with 4:4:4 subsampling and
  carries document resolution. `retouch/editor/selections.py` gains Expand,
  Contract, Border (round-disk kernels, capped at radius 16 per cv2 pass,
  canvas edges erode like an unbounded canvas) and Move (nearest-neighbour
  warp, zeros fill). Amounts validate 1–500 px.
- **Core Graphics float fixture** (`f751da1`):
  `scripts/dev/cg_float_adjustment_fixture.swift` plus a 5,400-line pinned
  reference; `tests/test_editor_finishing.py` pins float adjustments to the
  native fixture. Test-only; no editor behavior change.
- **Canvas resize and crop** (`0d83b38`): `DocumentHistory.canvas_resize`
  (3×3 numpad anchor, optional explicit `content_offset`, guides and mask
  placements shift with content; out-of-canvas layers kept and shifted) and
  `crop_document` (crop rectangle as one undo step via `canvas_resize`).
  `selection_bbox` added to selections for crop-from-selection.
- **True 16-bit export + EXR read guard** (`fb02002`): `composite_16`
  quantizes the float band renderer directly to uint16 (no 8-bit detour);
  `export_png(bit_depth=16)` uses it. `retouch/io.py` `read_image_16bit`
  gains `allow_exr=False` — EXR decode is opt-in with a hard clamp above
  1.0, refusing by default. `_embed_png_metadata` gains a `dpi` pHYs chunk
  splice (cv2 PNG encoder has no DPI params).
- **Magic wand + Tier-1 hotkeys** (`a3e5c69`): `wand_selection` (box-average
  seed sample with C-style round-half-up, per-channel tolerance, 4-connected
  `FLOODFILL_MASK_ONLY` or global match). `EDITOR_HOTKEYS_JS` wired via
  `gui.py` `Blocks(js=...)`: cmd/ctrl+Z undo, cmd/ctrl+shift+Z / cmd/ctrl+Y
  redo, cmd/ctrl+D clear selection, skipping text inputs.
- **Spot Healing and Clone Stamp on layer pixels** (`316531f`):
  `retouch/editor/healing.py` (Poisson-style `_solve`/`_relax` content-aware
  fill, `spot_heal`, `clone_stamp`); `paint_pixels` applies brushed coverage
  (flip-aware, softness via Gaussian falloff) directly to the selected
  raster layer as one undo step. `replace_pixels` gains a label override.
- **Colour brush on layer pixels** (`c0299d1`): see the Colour brush section
  of COMPOSITOR_PYTHON_PORT.md (already documented there).
- **Grading controls, region selections, local retouch; recipe layers;
  local clone; clone anchors** (`9edef8c`, `6a366ff`, `034ab60`, `aa1c80f`):
  documented in COMPOSITOR_PYTHON_PORT.md's acceptance sections; editor-relevant
  summary: adjustment picker Curves/Grain/Noise, semantic region selections,
  Heal/Dodge/Burn/Clone overlay layers, re-editable recipe layers persisted as
  a `retouch-recipes.json` extension, and a preview-based click-to-place clone
  picker (`clone_picker.py`).
- **Fixes**: `568cd2c` makes the saved-array cache lock re-entrant
  (`threading.RLock`), fixing a deadlock when a weakref callback fires under
  a held lock; `ebde387` adds `from __future__ import annotations` to
  `clone_picker.py` for Python 3.9; `9e0d65b` builds the Core Image blend
  fixture only on macOS; `638f14a` widens the 12 MP hue/sat timing margin on
  CI.


- **Texture-aware healing, colour range, brush stroke parity, copy/cut
  selection** (`c4d785f`): Heal method **Texture-aware** reuses the
  deterministic PatchMatch primitive (seed 0, patch 7, 3 passes, 64
  candidates) with an optional donor selection and 16-256 search distance;
  budget guards (4,096 painted px, 512x512 search window) run before the
  composite is built, and a degenerate donor set abstains into the
  donor-constrained Telea fallback with explicit layer naming.
  `color_range_selection` in `retouch/editor/selections.py` ports
  `WandPixels.c color_range_mask` (per-channel RGB box distance, include
  union minus exclude, fuzziness 0-200, transparent never matches).
  `stroke_coverage` fixes a native-oracle divergence: separate canvas layers
  are separate native strokes that accumulate by screen `1-(1-a1)(1-a2)`
  (two 50% strokes stack toward 75%), replacing `np.maximum`; within one
  canvas layer the browser flatten still caps. `DocumentHistory.
  copy_selection_as_layer` ports Cmd-J/Cmd-X (feathered-edge partial alpha;
  cut clears to transparent holes and drops recipe provenance; one undo
  step; selection dropped). Also in this commit: `export_jpeg` flattens in
  512-row chunks (peak was ~1.9 GB above the composite at 80 MP; output
  bit-identical); deeply nested JSON in a manifest, recipe sidecar, recovery
  entry or save-over target fails as a clean ValueError instead of a raw
  RecursionError (worst case crashed `list_recoveries` at editor startup);
  8-bit PNG delivery encodes via cv2 `IMWRITE_PNG_COMPRESSION 3` (12 MP
  2.6 s to 1.3 s, pixels identical, ~0.2% larger files); local heal
  auto-radius clamped to 8 with padding <= 64; `validate_adjustment`'s
  error string derives from `KINDS`. Acceptance sections added in
  COMPOSITOR_PYTHON_PORT.md; suites: `test_editor_texture_heal.py` (6),
  `test_editor_color_range_selection.py` (13),
  `test_editor_copy_selection.py` (11), layer-paint parity tests updated.

Engine-only commits (`50a6521` Costume Shine, `191c1e2` Teeth Brightening)
touch `retouch/` and the main GUI, not the editor package; see their own
doc entries in README/RECIPE_GUIDE. The retained EXR worktree noted in this
review was integrated as `ff3516a` (EXR delivery contract) plus the
`fb02002` read guard.

## Notes for the owner

- The "Remaining gates" bullet "High-bit/color-managed editor export: closed
  by f45176d" can be extended: `fb02002` supersedes the 257-lift widening
  with true float→uint16 rendering for 16-bit PNG/TIFF export.
- The retained-work table row about the EXR worktree (`de346b1`) is now
  stale — cherry-picked as `ff3516a` per its own commit message.

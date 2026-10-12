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
Unsupported adjustment kinds, layer effects, unlinked masks and fractional
placement are rejected explicitly. This is the requested category backlog,
not every feature of the native app.

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
The normal application shell and packaged desktop remain separate acceptance checks.

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
not pixel-exact renderer parity. Matching Core Graphics rotated-mask resampling
remains an explicit follow-up; the CPU renderer uses OpenCV's bilinear affine
sampling (nearest when requested), with premultiplied alpha to prevent color halos.

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

- Exact rotated-mask/native resampling agreement and more native adjustment fixtures.
- Representative larger stacks and maximum-size/history/RSS characterization.
- Real pointer-drawn mask acceptance in the normal full application shell.
- Packaged desktop crash recovery and platform acceptance, including Windows.
- High-bit/color-managed editor export, dirty-tile caching and optional acceleration.

The saved stash and unique EXR worktree are retained work, not cleanup failures.
They require their owners' context before retirement or integration.

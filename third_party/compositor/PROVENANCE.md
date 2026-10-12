# Compositor provenance

The editor core in `retouch/editor/` is a Python port of parts of
**Compositor**, a macOS image editor by Wonder Assembly LLC, released under the
MIT License. The full upstream license is in [LICENSE](LICENSE) next to this
file and must ship with any build that includes `retouch/editor/`.

- Upstream: https://github.com/robbietilton/Compositor
- Pinned commit: `11d8d7a50992b24fd9a760a1c13b1c01b70aaf30`
- Durable source reference: [source-11d8d7a.tar.gz](source-11d8d7a.tar.gz),
  retrieved on 2026-10-12 from GitHub's codeload endpoint for the exact pinned
  commit. SHA-256:
  `b0312e408f8da9047ba8d28f0e22904bb6f3116e3ca0db5558a8c2a717b3b261`.
  This archive uses GitHub's commit-prefixed tree layout; its checksum differs
  from the earlier `git archive` artifact. The original temporary checkout and
  archive were absent when rechecked. To reproduce the retrieval:

  ```bash
  curl -fL https://codeload.github.com/robbietilton/Compositor/tar.gz/11d8d7a50992b24fd9a760a1c13b1c01b70aaf30 -o source-11d8d7a.tar.gz
  ```

The archive is retained for source reference and includes the upstream license.
It is not imported at runtime. The Python modules re-implement behaviour read
from the source below; translated rendering/adjustment modules repeat the notice.

| Python module | Upstream source it follows |
| --- | --- |
| `retouch/editor/document.py` | `Compositor/IO/ProjectStore.swift` (`ProjectManifest`, `ProjectLayerRecord`), `Compositor/Document/DocumentLimits.swift`, `Compositor/Document/LayerTransform.swift`, `docs/writing-comp-files.md` (blend-mode names) |
| `retouch/editor/project_store.py` | `Compositor/IO/ProjectStore.swift` (`load`, `validate`, `checkSize`, `checkFile`), `Compositor/Document/LayerMask.swift` (`isValid`), `docs/project-format.md` |
| `retouch/editor/composite.py` | `Compositor/IO/ImageExporter.swift` (`render`), `Compositor/Rendering/LayerRenderer.swift` (`draw`) |
| `tests/test_editor_composite.py`, `tests/test_editor_project_store.py` | Expected values translated from `CompositorTests/LayerAppearanceTests.swift`, `LayerMaskTests.swift` and `ProjectTests.swift` |

The finishing extension also follows `Document/LayerTransform.swift` for
clockwise center rotation, `Document/LayerGroups.swift` and
`Document/LayerMask.swift` for pass-through opacity and folder masks,
`Document/LiveLayerMask.swift` and `Rendering/LiveMaskRenderer.swift` for
live-alpha links and clipping stacks, and `Document/ImageAdjustments.swift`,
`Document/Levels.swift` and `Document/LayerAdjustment.swift` for the supported
grading subset in `retouch/editor/adjustments.py`. Selection geometry and UI
controls are original Python implementations. Native resampling differs from
the CPU reference; see the current editor acceptance report.

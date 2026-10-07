# Compositor provenance

The editor core in `retouch/editor/` is a Python port of parts of
**Compositor**, a macOS image editor by Wonder Assembly LLC, released under the
MIT License. The full upstream license is in [LICENSE](LICENSE) next to this
file and must ship with any build that includes `retouch/editor/`.

- Upstream: https://github.com/robbietilton/Compositor
- Pinned commit: `11d8d7a50992b24fd9a760a1c13b1c01b70aaf30`
- Source archive of that commit (`git archive`, 2,763,442 bytes,
  SHA-256 `5182b56203d8a729ad3d7817eb14607c448145e7c5c9d8e83d03623b668eb8c4`):
  kept in the project's shared files as
  `retouch/compositor-port/compositor-source-11d8d7a.tar.gz`. To rebuild it:

  ```bash
  git clone https://github.com/robbietilton/Compositor.git /tmp/compositor-reference
  git -C /tmp/compositor-reference archive --format=tar.gz \
      --prefix=Compositor-11d8d7a/ -o compositor-source-11d8d7a.tar.gz \
      11d8d7a50992b24fd9a760a1c13b1c01b70aaf30
  ```

No upstream file is vendored. The Python modules re-implement behaviour read
from the upstream sources below; each module's docstring repeats the notice.

| Python module | Upstream source it follows |
| --- | --- |
| `retouch/editor/document.py` | `Compositor/IO/ProjectStore.swift` (`ProjectManifest`, `ProjectLayerRecord`), `Compositor/Document/DocumentLimits.swift`, `Compositor/Document/LayerTransform.swift`, `docs/writing-comp-files.md` (blend-mode names) |
| `retouch/editor/project_store.py` | `Compositor/IO/ProjectStore.swift` (`load`, `validate`, `checkSize`, `checkFile`), `Compositor/Document/LayerMask.swift` (`isValid`), `docs/project-format.md` |
| `retouch/editor/composite.py` | `Compositor/IO/ImageExporter.swift` (`render`), `Compositor/Rendering/LayerRenderer.swift` (`draw`) |
| `tests/test_editor_composite.py`, `tests/test_editor_project_store.py` | Expected values translated from `CompositorTests/LayerAppearanceTests.swift`, `LayerMaskTests.swift` and `ProjectTests.swift` |

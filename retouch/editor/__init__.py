"""Layer-based editor core, ported in stages from Compositor (MIT).

Milestones 1-3 of ``docs/plans/COMPOSITOR_PYTHON_PORT.md``: a document model,
a validated ``.comp`` reader/writer for a declared subset of the format, a
NumPy reference compositor for Normal blending, undo history, the mask brush
and the editor session behind the Layer Editor tab. The only UI is the page in
``static/``, served by ``web.py``; the other modules never import it.

Upstream: https://github.com/robbietilton/Compositor at
11d8d7a50992b24fd9a760a1c13b1c01b70aaf30, Copyright (c) 2026 Wonder Assembly
LLC, MIT License (full text in ``third_party/compositor/LICENSE``; per-module
origins in ``third_party/compositor/PROVENANCE.md``).
"""
from .composite import composite, export_png
from .commands import DocumentHistory
from .document import Document, Layer, UnsupportedFeature
from .project_store import ProjectError, load_project, save_project
from .session import EditorError, EditorSession

__all__ = [
    'Document',
    'DocumentHistory',
    'EditorError',
    'EditorSession',
    'Layer',
    'ProjectError',
    'UnsupportedFeature',
    'composite',
    'export_png',
    'load_project',
    'save_project',
]

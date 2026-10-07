"""Layer-based editor core, ported in stages from Compositor (MIT).

Milestones 1-2 of ``docs/plans/COMPOSITOR_PYTHON_PORT.md``: a document model,
a validated ``.comp`` reader/writer for a declared subset of the format, and a
NumPy reference compositor for Normal blending. No UI lives here.

Upstream: https://github.com/robbietilton/Compositor at
11d8d7a50992b24fd9a760a1c13b1c01b70aaf30, Copyright (c) 2026 Wonder Assembly
LLC, MIT License (full text in ``third_party/compositor/LICENSE``; per-module
origins in ``third_party/compositor/PROVENANCE.md``).
"""
from .composite import composite, export_png
from .document import Document, Layer, UnsupportedFeature
from .project_store import ProjectError, load_project, save_project

__all__ = [
    'Document',
    'Layer',
    'ProjectError',
    'UnsupportedFeature',
    'composite',
    'export_png',
    'load_project',
    'save_project',
]

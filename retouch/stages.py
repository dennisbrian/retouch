"""Stage-registry architecture for the Retouch Engine pipeline (P3).

Formalizes the previously-hardcoded stage sequence in ``engine.py`` into
an ordered registry of ``Stage`` objects, each with:

    - ``name``: human-readable identifier
    - ``phase``: pipeline phase ("detection", "face", "global", "grade", "finish")
    - ``enabled(ctx)``: gate predicate (whether the stage should run)
    - ``run(state)``: mutate ``PipelineState`` and return it

The registry is built in ``RetouchEngine.__init__`` and ``process()``
becomes a fold over stages. This enables:

    - F2: per-stage opacity/bypass (trivial on the registry)
    - F11: QA probes between stages
    - T4: plugin API hook points
    - Timing collection in the runner (deletes ~40 lines of boilerplate)

Golden-output gated: every migration commit MUST keep
``tests/test_golden_pipeline.py`` byte-identical.

Public API:
    PipelineState — mutable dataclass passed through the stage fold
    Stage — protocol/ABC for pipeline stages
    StageRegistry — ordered collection of stages with fold/runner
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np


@dataclass
class PipelineState:
    """Mutable state passed through the stage fold.

    Created at pipeline entry, mutated by each stage's ``run()`` method,
    and consumed at pipeline exit. All stages share this single state
    object — no hidden coupling through method parameters.

    Attributes:
        img: Current image (float32 [0,1] in global phases, uint8 or
            float32 [0,255] in face/detection phases).
        ctx: ProcessingContext with all user parameters.
        timings: Dict of stage_name -> elapsed_ms, filled by the runner.
        person_mask: Optional person segmentation mask.
        acc_skin: Accumulated skin mask from face processing.
        acc_skin_hair: Accumulated skin+hair mask.
        acc_lips: Accumulated lips mask.
        acc_sharpen: Accumulated sharpen mask.
        acc_hair_only: Hair-only mask for background matte operations.
        faces: List of detected FaceData objects.
        face_contexts: List of FaceContext objects (cached).
        qa: List of QAWarning objects (filled in QA stage).
        no_face: Whether no faces were detected.
        style_ref: Optional style reference image.
        h_img: Image height (cached for convenience).
        w_img: Image width (cached for convenience).
        bypass: Set of stage names to skip (F2 stage mixer).
        opacity: Dict of stage_name -> opacity [0,1] (F2 stage mixer,
            not yet wired — stages that support it blend their output
            with the pre-stage image).
    """

    img: np.ndarray
    ctx: Any  # ProcessingContext (avoid circular import)
    timings: Dict[str, float] = field(default_factory=dict)
    person_mask: Optional[np.ndarray] = None
    acc_skin: Optional[np.ndarray] = None
    acc_skin_hair: Optional[np.ndarray] = None
    acc_lips: Optional[np.ndarray] = None
    acc_sharpen: Optional[np.ndarray] = None
    acc_hair_only: Optional[np.ndarray] = None
    faces: List[Any] = field(default_factory=list)
    face_contexts: Optional[List[Any]] = None
    qa: List[Any] = field(default_factory=list)
    no_face: bool = False
    style_ref: Optional[np.ndarray] = None
    h_img: int = 0
    w_img: int = 0
    bypass: set = field(default_factory=set)
    opacity: Dict[str, float] = field(default_factory=dict)


@runtime_checkable
class Stage(Protocol):
    """Protocol for pipeline stages.

    Each stage has a name, a phase, an enabled gate, and a run method.
    Stages are registered in a ``StageRegistry`` and executed in order
    by ``StageRegistry.run()``.

    The ``run`` method mutates ``PipelineState`` in-place and returns it.
    Stages should be idempotent where possible (running twice produces
    the same result as running once) — this makes F2's opacity blending
    straightforward.
    """

    name: str
    phase: str  # "detection" | "face" | "global" | "grade" | "finish"

    def enabled(self, state: PipelineState) -> bool: ...

    def run(self, state: PipelineState) -> PipelineState: ...


class BaseStage:
    """Base class for stages with sensible defaults.

    Subclasses should override ``name``, ``phase``, and ``run``.
    The ``enabled`` method defaults to True (always run).
    """

    name: str = "base"
    phase: str = "global"

    def enabled(self, state: PipelineState) -> bool:
        return True

    def run(self, state: PipelineState) -> PipelineState:
        raise NotImplementedError


class StageRegistry:
    """Ordered collection of pipeline stages with a fold runner.

    Stages are added in phase order: detection -> face -> global -> grade -> finish.
    The runner executes each enabled stage, collecting timings, and skipping
    stages in the ``bypass`` set.

    Usage::

        registry = StageRegistry()
        registry.add(MyStage())
        registry.add(AnotherStage())
        state = PipelineState(img=img, ctx=ctx)
        state = registry.run(state)
    """

    def __init__(self) -> None:
        self._stages: List[Stage] = []
        self._by_name: Dict[str, Stage] = {}

    def add(self, stage: Stage) -> "StageRegistry":
        """Add a stage to the registry. Returns self for chaining."""
        if stage.name in self._by_name:
            raise ValueError(f"Stage '{stage.name}' already registered")
        self._stages.append(stage)
        self._by_name[stage.name] = stage
        return self

    def remove(self, name: str) -> Optional[Stage]:
        """Remove a stage by name. Returns the removed stage or None."""
        stage = self._by_name.pop(name, None)
        if stage is not None:
            self._stages = [s for s in self._stages if s.name != name]
        return stage

    def get(self, name: str) -> Optional[Stage]:
        """Get a stage by name."""
        return self._by_name.get(name)

    def names(self) -> List[str]:
        """Return the ordered list of stage names."""
        return [s.name for s in self._stages]

    def phases(self) -> List[str]:
        """Return the ordered list of phase names."""
        return [s.phase for s in self._stages]

    def by_phase(self, phase: str) -> List[Stage]:
        """Return all stages in a given phase."""
        return [s for s in self._stages if s.phase == phase]

    def run(self, state: PipelineState) -> PipelineState:
        """Execute all enabled, non-bypassed stages in order.

        Collects timings into ``state.timings`` under each stage's name.
        Stages in ``state.bypass`` are skipped entirely.
        """
        import time

        for stage in self._stages:
            if stage.name in state.bypass:
                state.timings[stage.name] = 0.0
                continue
            if not stage.enabled(state):
                state.timings[stage.name] = 0.0
                continue

            t0 = time.perf_counter()
            try:
                state = stage.run(state)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(
                    f"Stage '{stage.name}' failed: {e}", exc_info=True
                )
                raise
            elapsed_ms = (time.perf_counter() - t0) * 1000
            state.timings[stage.name] = elapsed_ms

        return state

    def run_phase(self, phase: str, state: PipelineState) -> PipelineState:
        """Execute only stages in a given phase."""
        import time

        for stage in self._stages:
            if stage.phase != phase:
                continue
            if stage.name in state.bypass:
                state.timings[stage.name] = 0.0
                continue
            if not stage.enabled(state):
                state.timings[stage.name] = 0.0
                continue

            t0 = time.perf_counter()
            state = stage.run(state)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            state.timings[stage.name] = elapsed_ms

        return state

    def __len__(self) -> int:
        return len(self._stages)

    def __iter__(self):
        return iter(self._stages)

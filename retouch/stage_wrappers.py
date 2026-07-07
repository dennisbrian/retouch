"""Stage wrapper classes for the P3 stage-registry migration.

Each wrapper delegates to the existing ``RetouchEngine._stage_*`` method,
preserving byte-identical output. The wrappers are registered in a
``StageRegistry`` and executed via ``StageRegistry.run()``.

This is the incremental migration path described in
``PLAN_TIERP_PERF_ARCH_SHIP.md`` §P3 step 2:
"Migrate mechanically, phase by phase, starting with Phase 3 (the existing
_stage_* methods are already stage-shaped — wrapping them is renaming,
not rewriting)."

Golden-output gated: ``tests/test_golden_pipeline.py`` must remain
byte-identical across every migration commit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .stages import BaseStage, PipelineState

if TYPE_CHECKING:
    from .engine import RetouchEngine


class _EngineStage(BaseStage):
    """Base class for stages that delegate to a RetouchEngine method.

    Stores a reference to the engine and the method name to call.
    Subclasses override ``_call`` to invoke the right method with the
    right arguments extracted from ``PipelineState``.
    """

    def __init__(self, engine: "RetouchEngine") -> None:
        self._engine = engine

    def _call(self, state: PipelineState) -> np.ndarray:
        raise NotImplementedError

    def run(self, state: PipelineState) -> PipelineState:
        state.img = self._call(state)
        return state


class SubjectSeparationStage(_EngineStage):
    """Stage 3: Subject-background separation."""

    name = "subject_separation"
    phase = "global"

    def enabled(self, state: PipelineState) -> bool:
        return state.ctx.subject_separation > 0

    def _call(self, state: PipelineState) -> np.ndarray:
        return self._engine._stage_subject_separation(
            state.img, state.person_mask, state.ctx
        )


class BackgroundHarmonizeStage(_EngineStage):
    """Stage 3.1: C5 — skin-anchored background color harmonization."""

    name = "background_harmonize"
    phase = "global"

    def enabled(self, state: PipelineState) -> bool:
        return state.ctx.background_harmonize > 0

    def _call(self, state: PipelineState) -> np.ndarray:
        return self._engine._stage_harmonize(
            state.img, state.ctx, state.acc_skin, state.person_mask
        )


class BodySkinStage(_EngineStage):
    """Stage 3.5: Body skin retouch."""

    name = "body_skin"
    phase = "global"

    def enabled(self, state: PipelineState) -> bool:
        return True

    def _call(self, state: PipelineState) -> np.ndarray:
        return self._engine._stage_body_skin(
            state.img, state.ctx, state.person_mask,
            state.acc_skin, state.acc_skin_hair, state.acc_lips,
            state.faces, state.h_img, state.w_img,
        )


class GlobalStage(_EngineStage):
    """Stage 4: Global tonal adjustments."""

    name = "global"
    phase = "global"

    def enabled(self, state: PipelineState) -> bool:
        return True

    def _call(self, state: PipelineState) -> np.ndarray:
        return self._engine._stage_global(state.img, state.ctx)


class GradeStage(_EngineStage):
    """Stage 5: Color grading."""

    name = "grading"
    phase = "grade"

    def enabled(self, state: PipelineState) -> bool:
        return True

    def _call(self, state: PipelineState) -> np.ndarray:
        return self._engine._stage_grade(
            state.img, state.ctx, state.acc_skin, state.acc_lips,
            state.person_mask, style_ref=state.style_ref, faces=state.faces,
        )


class FinishStage(_EngineStage):
    """Stage 6: Selective sharpening + impact finish."""

    name = "finish"
    phase = "finish"

    def enabled(self, state: PipelineState) -> bool:
        return True

    def _call(self, state: PipelineState) -> np.ndarray:
        return self._engine._stage_finish(
            state.img, state.ctx, state.acc_sharpen,
            faces=state.faces, person_mask=state.person_mask,
        )


def build_global_registry(engine: "RetouchEngine") -> "StageRegistry":
    """Build the stage registry for the global phases (stages 3-6).

    This is the P3 migration entry point. The registry is built once
    in ``RetouchEngine.__init__`` and ``_run_global_phases`` can optionally
    use it via ``registry.run(state)`` instead of the hardcoded stage calls.

    Returns:
        A ``StageRegistry`` with stages 3-6 registered in order.
    """
    from .stages import StageRegistry

    registry = StageRegistry()
    registry.add(SubjectSeparationStage(engine))
    registry.add(BackgroundHarmonizeStage(engine))
    registry.add(BodySkinStage(engine))
    registry.add(GlobalStage(engine))
    registry.add(GradeStage(engine))
    registry.add(FinishStage(engine))
    return registry

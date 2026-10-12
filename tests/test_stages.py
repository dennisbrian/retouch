"""Tests for the stage-registry infrastructure (P3).

Verifies that StageRegistry correctly:
  - Maintains ordered stage lists
  - Skips bypassed stages
  - Skips disabled stages
  - Collects timings
  - Handles phase-specific execution
  - Rejects duplicate stage names
"""

from __future__ import annotations

import numpy as np
import pytest

from retouch.stages import PipelineState, BaseStage, StageRegistry


class _NoopStage(BaseStage):
    """Stage that does nothing (for testing)."""
    def __init__(self, name: str, phase: str = "global"):
        self.name = name
        self.phase = phase

    def run(self, state: PipelineState) -> PipelineState:
        return state


class _IncrementStage(BaseStage):
    """Stage that increments a counter in state.timings (for ordering tests)."""
    def __init__(self, name: str, phase: str = "global"):
        self.name = name
        self.phase = phase
        self.run_count = 0

    def run(self, state: PipelineState) -> PipelineState:
        self.run_count += 1
        return state


class _DisabledStage(BaseStage):
    """Stage that is always disabled."""
    name = "disabled"
    phase = "global"

    def enabled(self, state: PipelineState) -> bool:
        return False

    def run(self, state: PipelineState) -> PipelineState:
        raise RuntimeError("Should not run")


class _FailingStage(BaseStage):
    """Stage that raises an exception."""
    name = "failing"
    phase = "global"

    def run(self, state: PipelineState) -> PipelineState:
        raise ValueError("Intentional failure")


@pytest.fixture
def state():
    img = np.zeros((64, 64, 3), dtype=np.float32)
    return PipelineState(img=img, ctx=None)


@pytest.fixture
def registry():
    return StageRegistry()


class TestStageRegistryBasics:
    def test_empty_registry_runs_without_error(self, registry, state):
        result = registry.run(state)
        assert result is state

    def test_add_stage(self, registry):
        stage = _NoopStage("a")
        registry.add(stage)
        assert len(registry) == 1
        assert registry.names() == ["a"]

    def test_add_multiple_stages_preserves_order(self, registry):
        registry.add(_NoopStage("a"))
        registry.add(_NoopStage("b"))
        registry.add(_NoopStage("c"))
        assert registry.names() == ["a", "b", "c"]

    def test_add_duplicate_name_raises(self, registry):
        registry.add(_NoopStage("a"))
        with pytest.raises(ValueError, match="already registered"):
            registry.add(_NoopStage("a"))

    def test_remove_stage(self, registry):
        registry.add(_NoopStage("a"))
        registry.add(_NoopStage("b"))
        removed = registry.remove("a")
        assert removed is not None
        assert removed.name == "a"
        assert registry.names() == ["b"]

    def test_remove_nonexistent_returns_none(self, registry):
        assert registry.remove("nonexistent") is None

    def test_get_stage_by_name(self, registry):
        stage = _NoopStage("a")
        registry.add(stage)
        assert registry.get("a") is stage
        assert registry.get("nonexistent") is None


class TestStageRegistryRun:
    def test_run_executes_all_enabled_stages(self, registry, state):
        s1 = _IncrementStage("s1")
        s2 = _IncrementStage("s2")
        registry.add(s1).add(s2)
        registry.run(state)
        assert s1.run_count == 1
        assert s2.run_count == 1

    def test_run_skips_bypassed_stages(self, registry, state):
        s1 = _IncrementStage("s1")
        s2 = _IncrementStage("s2")
        registry.add(s1).add(s2)
        state.bypass.add("s1")
        registry.run(state)
        assert s1.run_count == 0
        assert s2.run_count == 1

    def test_run_skips_disabled_stages(self, registry, state):
        s1 = _IncrementStage("s1")
        s2 = _DisabledStage()
        s3 = _IncrementStage("s3")
        registry.add(s1).add(s2).add(s3)
        registry.run(state)
        assert s1.run_count == 1
        assert s3.run_count == 1

    def test_run_collects_timings(self, registry, state):
        registry.add(_NoopStage("s1")).add(_NoopStage("s2"))
        registry.run(state)
        assert "s1" in state.timings
        assert "s2" in state.timings
        assert state.timings["s1"] >= 0.0
        assert state.timings["s2"] >= 0.0

    def test_run_bypassed_stage_has_zero_timing(self, registry, state):
        registry.add(_NoopStage("s1"))
        state.bypass.add("s1")
        registry.run(state)
        assert state.timings["s1"] == 0.0

    def test_run_propagates_exceptions(self, registry, state):
        registry.add(_FailingStage())
        with pytest.raises(ValueError, match="Intentional failure"):
            registry.run(state)


class TestStageRegistryPhases:
    def test_by_phase_returns_correct_stages(self, registry):
        registry.add(_NoopStage("a", "detection"))
        registry.add(_NoopStage("b", "global"))
        registry.add(_NoopStage("c", "global"))
        registry.add(_NoopStage("d", "finish"))
        global_stages = registry.by_phase("global")
        assert len(global_stages) == 2
        assert [s.name for s in global_stages] == ["b", "c"]

    def test_run_phase_executes_only_that_phase(self, registry, state):
        s1 = _IncrementStage("a", "detection")
        s2 = _IncrementStage("b", "global")
        s3 = _IncrementStage("c", "finish")
        registry.add(s1).add(s2).add(s3)
        registry.run_phase("global", state)
        assert s1.run_count == 0
        assert s2.run_count == 1
        assert s3.run_count == 0

    def test_phases_returns_ordered_list(self, registry):
        registry.add(_NoopStage("a", "detection"))
        registry.add(_NoopStage("b", "global"))
        registry.add(_NoopStage("c", "finish"))
        assert registry.phases() == ["detection", "global", "finish"]


class TestPipelineState:
    def test_default_state(self):
        img = np.zeros((32, 32, 3), dtype=np.float32)
        state = PipelineState(img=img, ctx=None)
        assert state.timings == {}
        assert state.faces == []
        assert state.qa == []
        assert state.no_face is False
        assert state.bypass == set()
        assert state.opacity == {}

    def test_state_is_mutable(self):
        img = np.zeros((32, 32, 3), dtype=np.float32)
        state = PipelineState(img=img, ctx=None)
        state.timings["test"] = 42.0
        state.bypass.add("skip_me")
        assert state.timings["test"] == 42.0
        assert "skip_me" in state.bypass

    def test_state_img_can_be_modified(self):
        img = np.zeros((32, 32, 3), dtype=np.float32)
        state = PipelineState(img=img, ctx=None)
        state.img = np.ones((32, 32, 3), dtype=np.float32)
        assert state.img[0, 0, 0] == 1.0

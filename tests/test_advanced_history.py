"""Focused tests for the compact Advanced Retouch history state."""

from __future__ import annotations

import copy
import json

import pytest

from retouch.advanced_history import (
    AdvancedHistory,
    STATUS_APPLIED,
    STATUS_REJECTED,
    STATUS_SKIP,
    history_from_state,
    history_state,
)


class ArrayStub:
    """Small array protocol test double; it is intentionally not NumPy."""

    def __init__(self, size: int = 4096, shape=(6240, 4160, 3)):
        self.shape = shape
        self.dtype = "uint8"
        self._raw = bytes(index % 251 for index in range(size))

    def tobytes(self):
        return self._raw


def test_empty_undo_redo_are_explicit_skip_noop_results():
    history = AdvancedHistory()

    undo = history.undo()
    redo = history.redo()

    assert undo.status == STATUS_SKIP
    assert undo.outcome == "noop"
    assert undo.is_noop is True
    assert undo.changed is False
    assert undo.message == "Nothing to undo."
    assert redo.status == STATUS_SKIP
    assert redo.outcome == "noop"
    assert history.cursor == 0
    assert history.current_edit() is None


def test_append_undo_redo_and_new_edit_drop_redo_branch():
    history = AdvancedHistory(checkpoint_interval=20)
    assert history.append({"name": "one"}).status == STATUS_APPLIED
    assert history.append({"name": "two"}).status == STATUS_APPLIED
    assert history.append({"name": "three"}).status == STATUS_APPLIED
    assert history.current_edit() == {"name": "three"}

    assert history.undo().status == STATUS_APPLIED
    assert history.current_edit() == {"name": "two"}
    assert history.redo().status == STATUS_APPLIED
    assert history.current_edit() == {"name": "three"}

    history.undo()
    appended = history.append({"name": "replacement"})
    assert appended.status == STATUS_APPLIED
    assert history.can_redo is False
    assert history.current_edit() == {"name": "replacement"}
    assert [record.edit for record in history.entries] == [
        {"name": "one"},
        {"name": "two"},
        {"name": "replacement"},
    ]


def test_checkpoints_are_occasional_and_selected_from_the_past():
    history = AdvancedHistory(checkpoint_interval=2, max_checkpoints=8)
    history.append("one", preview={"preview": 1}, checkpoint=True)
    history.append("two", preview={"preview": 2})
    history.append("three", preview={"preview": 3})
    history.append("four", preview={"preview": 4})

    assert [item.cursor for item in history.checkpoints] == [1, 2, 4]
    selected = history.select_checkpoint(3)
    assert selected is not None
    assert selected.cursor == 2
    assert selected.preview == {"preview": 2}
    assert history.select_checkpoint(0) is None

    # Returned checkpoints are detached from the history object.
    selected.metadata["mutated"] = True
    assert "mutated" not in history.checkpoints[1].metadata


def test_replay_plan_preserves_edit_order_with_stub_apply_function():
    history = AdvancedHistory(checkpoint_interval=2)
    history.append({"step": "a"}, preview="frame-a", checkpoint=True)
    history.append({"step": "b"}, preview="frame-b")
    history.append({"step": "c"}, preview="frame-c")
    history.append({"step": "d"}, preview="frame-d")

    plan = history.replay_plan(4)
    assert plan.from_cursor == 4
    assert plan.to_cursor == 4
    assert plan.edit_ids == ()

    plan = history.replay_plan(3)
    assert plan.from_cursor == 2
    assert plan.edit_ids == ("edit-3",)
    assert plan.edits == ({"step": "c"},)

    applied = plan.replay("checkpoint-state", lambda value, edit: value + edit["step"])
    assert applied == "checkpoint-statec"


def test_replay_plan_can_start_at_source_without_checkpoint():
    history = AdvancedHistory(checkpoint_interval=100)
    history.append("a")
    history.append("b")
    plan = history.replay_plan()

    assert plan.checkpoint is None
    assert plan.from_cursor == 0
    assert plan.complete_from_source is True
    assert plan.edits == ("a", "b")


def test_memory_budget_trims_old_edits_but_keeps_state_bounded():
    history = AdvancedHistory(
        max_entries=100,
        memory_budget_bytes=1500,
        checkpoint_interval=1000,
    )
    for index in range(40):
        outcome = history.append({"index": index, "note": "x" * 80})
        assert outcome.status == STATUS_APPLIED

    assert history.memory_size_bytes <= history.memory_budget_bytes
    assert history.size < 40
    assert history.history_truncated is True
    assert history.base_cursor > 0
    assert history.cursor == history.end_cursor
    assert history.undo().status == STATUS_APPLIED

    while history.can_undo:
        history.undo()
    boundary = history.undo()
    assert boundary.status == STATUS_SKIP
    assert boundary.outcome == "noop"


def test_edit_that_cannot_fit_is_rejected_without_losing_existing_history():
    history = AdvancedHistory(memory_budget_bytes=1200, checkpoint_interval=1000)
    history.append({"name": "kept"})
    before = history.to_state()

    result = history.append({"huge": "x" * 5000})

    assert result.status == STATUS_REJECTED
    assert history.to_state() == before
    assert history.current_edit() == {"name": "kept"}


def test_preview_budget_does_not_retain_source_sized_array():
    history = AdvancedHistory(
        memory_budget_bytes=2000,
        preview_max_bytes=64,
        checkpoint_interval=1,
    )
    outcome = history.append(
        {"operation": "heal", "mask": ArrayStub(32)},
        preview=ArrayStub(10000),
    )
    assert outcome.status == STATUS_APPLIED

    state = history.to_state()
    encoded = json.dumps(state, allow_nan=False)
    assert len(encoded.encode("utf-8")) <= history.memory_budget_bytes
    assert state["entries"][0]["edit"]["mask"]["__type__"] == "array-descriptor"
    assert state["checkpoints"][0]["preview"]["__type__"] == "preview-omitted"
    assert "data" not in state["checkpoints"][0]["preview"]


def test_json_and_deepcopy_state_helpers_round_trip_without_live_objects():
    history = AdvancedHistory(checkpoint_interval=1)
    history.append({"operation": "paint", "value": 0.5}, preview={"w": 32, "h": 32})

    state = history_state(history)
    assert json.loads(json.dumps(state)) == state
    detached = copy.deepcopy(state)
    detached["entries"][0]["edit"]["value"] = 0.9
    assert history.current_edit()["value"] == 0.5

    restored = history_from_state(state)
    assert restored.to_state() == state
    assert copy.deepcopy(restored).to_state() == state
    assert restored.to_json(indent=None)


def test_add_checkpoint_empty_preview_is_explicit_skip():
    history = AdvancedHistory()
    result = history.add_checkpoint(None)
    assert result.status == STATUS_SKIP
    assert result.outcome == "noop"
    assert history.checkpoint_count == 0


def test_checkpoint_and_replay_cursor_validation():
    history = AdvancedHistory()
    history.append("one")

    with pytest.raises(ValueError):
        history.select_checkpoint(2)
    with pytest.raises(ValueError):
        history.add_checkpoint("preview", cursor=2)
    with pytest.raises(ValueError):
        history.replay_plan(checkpoint_cursor=0)

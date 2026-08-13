"""Project profile and Look Board persistence tests."""

from pathlib import Path

import pytest

from retouch.project_profiles import (
    LookBoard,
    LookReference,
    ProjectProfile,
    ProjectProfileStore,
)


def test_look_board_replaces_duplicate_reference_and_round_trips():
    board = LookBoard("board-1", project_id="project-1")
    board.add_reference(LookReference("a.jpg", "warm", 0.8))
    board.add_reference(LookReference("a.jpg", "updated", 0.5, "keep skin neutral"))
    board.add_reference(LookReference("b.jpg", "contrast", 1.0))

    restored = LookBoard.from_json(board.to_json())

    assert [item.path for item in restored.references] == ["a.jpg", "b.jpg"]
    assert restored.references[0].label == "updated"
    assert restored.references[0].notes == "keep skin neutral"


def test_look_reference_weight_is_bounded():
    with pytest.raises(ValueError):
        LookReference("a.jpg", "bad", 1.2)


def test_project_profile_store_persists_profiles_and_boards(tmp_path: Path):
    store = ProjectProfileStore(tmp_path / "projects.json")
    profile = ProjectProfile(
        profile_id="dennis-subject-1",
        subject_key="subject-1",
        display_name="Subject One",
        recipe="natural",
        preferred_params={"smooth": 18},
        protected_marks=[{"kind": "mole", "label": "left cheek", "action": "preserve"}],
    )
    board = LookBoard("portrait-board", project_id=profile.profile_id)
    board.add_reference(LookReference("look-a.jpg", "soft daylight"))
    store.upsert_profile(profile)
    store.upsert_board(board)

    restored = ProjectProfileStore(tmp_path / "projects.json")

    assert restored.profiles[profile.profile_id].preferred_params == {"smooth": 18}
    assert restored.profiles[profile.profile_id].protected_marks[0]["action"] == "preserve"
    assert restored.boards[board.board_id].project_id == profile.profile_id


def test_project_profile_store_handles_corrupt_state(tmp_path: Path):
    path = tmp_path / "projects.json"
    path.write_text("not json", encoding="utf-8")

    store = ProjectProfileStore(path)

    assert store.list_profiles() == []
    assert store.list_boards() == []


def test_project_profile_store_uses_shared_cache_override(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("RETOUCH_CACHE_DIR", str(tmp_path / "cache"))

    store = ProjectProfileStore()

    assert store.path == tmp_path / "cache" / "projects.json"

"""Shoot Intelligence foundation tests."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from retouch.shoot_intelligence import (
    ProjectGraph,
    ProjectNode,
    ShootAsset,
    group_bursts,
    inspect_asset,
    BurstGroup,
    CAPTURE_ONLY_POLICY,
    FACE_AWARE_POLICY,
    rank_burst_candidates,
)


def test_project_graph_blocks_until_dependencies_succeed():
    graph = ProjectGraph([
        ProjectNode("ingest", "ingest"),
        ProjectNode("cull", "cull", dependencies=["ingest"]),
    ])

    graph.refresh_ready()
    assert [node.node_id for node in graph.ready_nodes()] == ["ingest"]
    assert graph.nodes["cull"].status == "blocked"

    graph.set_status("ingest", "succeeded", outputs=["manifest.json"])
    assert [node.node_id for node in graph.ready_nodes()] == ["cull"]


def test_project_graph_rejects_cycles_and_missing_dependencies():
    try:
        ProjectGraph([ProjectNode("a", "a", dependencies=["missing"])])
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("missing dependency was accepted")

    graph = ProjectGraph([ProjectNode("a", "a")])
    graph.nodes["a"].dependencies.append("a")
    try:
        graph.validate()
    except ValueError as exc:
        assert "cycle" in str(exc)
    else:
        raise AssertionError("cycle was accepted")


def test_project_graph_round_trip():
    graph = ProjectGraph([ProjectNode("ingest", "ingest", status="succeeded")])

    restored = ProjectGraph.from_json(graph.to_json())

    assert restored.nodes["ingest"].status == "succeeded"


def _write_capture(path: Path, value: int, second: int):
    image = Image.new("RGB", (32, 24), (value, value, value))
    exif = image.getexif()
    exif[306] = f"2026:08:12 10:00:{second:02d}"
    exif[271] = "Acme"
    exif[272] = "Camera"
    exif[42036] = "Prime"
    image.save(path, exif=exif.tobytes())


def test_inspect_asset_and_group_bursts_are_explainable(tmp_path: Path):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    distant = tmp_path / "distant.jpg"
    _write_capture(first, 100, 1)
    _write_capture(second, 101, 2)
    _write_capture(distant, 220, 20)

    assets = [inspect_asset(path) for path in (first, second, distant)]
    groups = group_bursts(assets)

    assert len(groups) == 1
    assert groups[0].asset_paths == (str(first.resolve()), str(second.resolve()))
    assert groups[0].review_required is True
    assert groups[0].evidence["count"] == 2
    assert "capture timestamps are close" in groups[0].reasons


def test_burst_group_does_not_use_missing_timestamps():
    assets = [
        ShootAsset("a.jpg", None, "Acme", "Prime", 50.0, 32, 24, "a" * 16),
        ShootAsset("b.jpg", None, "Acme", "Prime", 50.0, 32, 24, "a" * 16),
    ]

    assert group_bursts(assets) == []


def test_burst_candidate_ranking_is_explainable_and_non_destructive(tmp_path: Path):
    sharp = np.zeros((64, 64, 3), dtype=np.uint8)
    sharp[::2, :, :] = 255
    soft = cv2.GaussianBlur(sharp, (0, 0), 4.0)
    sharp_path = tmp_path / "sharp.jpg"
    soft_path = tmp_path / "soft.jpg"
    assert cv2.imwrite(str(sharp_path), cv2.cvtColor(sharp, cv2.COLOR_RGB2BGR))
    assert cv2.imwrite(str(soft_path), cv2.cvtColor(soft, cv2.COLOR_RGB2BGR))
    group = BurstGroup(
        group_id="burst-0001",
        asset_paths=(str(sharp_path), str(soft_path)),
        confidence=0.9,
        reasons=("test",),
        evidence={"count": 2},
    )

    ranked = rank_burst_candidates(group)

    assert [candidate.path for candidate in ranked] == [str(sharp_path), str(soft_path)]
    assert ranked[0].rank == 1
    assert ranked[0].review_required is True
    assert "scoring_policy" in ranked[0].evidence
    assert sharp_path.is_file() and soft_path.is_file()


def _burst(tmp_path: Path, names, sharp_name=None):
    """Write frames that are identical except that ``sharp_name`` is crisper."""
    base = np.zeros((64, 64, 3), dtype=np.uint8)
    base[::2, :, :] = 255
    paths = []
    for name in names:
        frame = base if name == sharp_name else cv2.GaussianBlur(base, (0, 0), 4.0)
        path = tmp_path / f"{name}.jpg"
        assert cv2.imwrite(str(path), frame)
        paths.append(str(path))
    group = BurstGroup("burst-0001", tuple(paths), 0.9, ("test",), {"count": len(paths)})
    return group, dict(zip(names, paths))


def _face(focus, eyes_open="yes", apertures=(0.30, 0.30), coverage=0.2):
    return {
        "coverage": coverage,
        "left_eye_sharpness": focus,
        "right_eye_sharpness": focus * 0.8,
        "face_sharpness": focus * 0.5,
        "eyes_open": eyes_open,
        "measurement_details": {
            "left_eye_aperture": apertures[0], "right_eye_aperture": apertures[1],
        },
    }


def test_sharp_face_outranks_sharp_background(tmp_path: Path):
    group, paths = _burst(tmp_path, ["background_sharp", "face_sharp"], sharp_name="background_sharp")
    faces = {paths["background_sharp"]: [_face(0.2)], paths["face_sharp"]: [_face(1.0)]}

    capture_only = rank_burst_candidates(group)
    ranked = rank_burst_candidates(group, faces)

    assert capture_only[0].path == paths["background_sharp"]
    assert capture_only[0].evidence["scoring_policy"] == CAPTURE_ONLY_POLICY
    assert ranked[0].path == paths["face_sharp"]
    assert ranked[0].evidence["scoring_policy"] == FACE_AWARE_POLICY
    assert ranked[0].evidence["face_focus_relative"] == 1.0


def test_closed_eyes_are_flagged_and_ranked_below_open_eyes_but_kept(tmp_path: Path):
    group, paths = _burst(tmp_path, ["blink", "open"])
    faces = {
        paths["blink"]: [_face(1.0, "no", (0.05, 0.04))],
        paths["open"]: [_face(0.8, "yes", (0.30, 0.29))],
    }

    ranked = rank_burst_candidates(group, faces)

    assert [candidate.path for candidate in ranked] == [paths["open"], paths["blink"]]
    assert ranked[1].evidence["flags"] == ["eyes_closed"]
    assert ranked[0].evidence["flags"] == []
    assert all(candidate.review_required for candidate in ranked)


def test_blink_is_judged_against_the_same_persons_widest_eyes(tmp_path: Path):
    group, paths = _burst(tmp_path, ["wide", "half", "wide2"])
    faces = {
        paths["wide"]: [_face(0.9, "yes", (0.40, 0.41))],
        # 0.20 reads as open on its own but is half this person's open eyes.
        paths["half"]: [_face(1.0, "yes", (0.20, 0.21))],
        paths["wide2"]: [_face(0.9, "yes", (0.39, 0.40))],
    }

    ranked = {candidate.path: candidate for candidate in rank_burst_candidates(group, faces)}

    assert ranked[paths["half"]].evidence["flags"] == ["eyes_closed"]
    assert ranked[paths["half"]].evidence["eye_aperture_relative"] == pytest.approx(0.5)
    assert ranked[paths["half"]].rank == 3


def test_narrow_eyed_subject_is_not_flagged_in_every_frame(tmp_path: Path):
    group, paths = _burst(tmp_path, ["a", "b", "c"])
    faces = {
        paths["a"]: [_face(1.0, "uncertain", (0.19, 0.19))],
        paths["b"]: [_face(0.9, "yes", (0.21, 0.22))],
        paths["c"]: [_face(0.8, "yes", (0.20, 0.20))],
    }

    ranked = rank_burst_candidates(group, faces)

    assert all("eyes_closed" not in candidate.evidence["flags"] for candidate in ranked)


def test_background_persons_closed_eyes_do_not_flag_the_frame(tmp_path: Path):
    group, paths = _burst(tmp_path, ["a", "b"])
    faces = {
        paths["a"]: [_face(1.0), _face(0.1, "no", (0.05, 0.05), coverage=0.01)],
        paths["b"]: [_face(0.5)],
    }

    ranked = rank_burst_candidates(group, faces)

    assert ranked[0].path == paths["a"]
    assert ranked[0].evidence["flags"] == []
    assert ranked[0].evidence["faces_detected"] == 2


def test_group_shot_flags_any_subject_with_closed_eyes(tmp_path: Path):
    group, paths = _burst(tmp_path, ["a", "b"])
    faces = {
        paths["a"]: [_face(1.0), _face(0.9, "no", (0.05, 0.05), coverage=0.15)],
        paths["b"]: [_face(0.8), _face(0.8, coverage=0.15)],
    }

    ranked = rank_burst_candidates(group, faces)

    assert ranked[0].path == paths["b"]
    assert "eyes_closed" in ranked[1].evidence["flags"]


def test_frame_without_a_face_is_flagged_in_a_face_aware_burst(tmp_path: Path):
    group, paths = _burst(tmp_path, ["face", "empty"], sharp_name="empty")
    faces = {paths["face"]: [_face(1.0)], paths["empty"]: []}

    ranked = rank_burst_candidates(group, faces)

    assert ranked[0].path == paths["face"]
    assert ranked[1].evidence["flags"] == ["no_face_detected"]

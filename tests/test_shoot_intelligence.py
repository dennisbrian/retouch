"""Shoot Intelligence foundation tests."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from retouch.shoot_intelligence import (
    ProjectGraph,
    ProjectNode,
    ShootAsset,
    group_bursts,
    inspect_asset,
    BurstGroup,
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

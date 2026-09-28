"""R&D Director contract tests; no production state is written."""
import json
from pathlib import Path

import pytest

from scripts.dev.rd_director import core


@pytest.fixture
def state(tmp_path):
    return core.load(tmp_path / "rd.json")


def test_add_and_lifecycle(state):
    opp = core.add_opportunity(state, "wig edge refinement", "RESEARCH",
                               "wig edges lost in low light")
    assert opp["id"] == "OPP-001" and opp["stage"] == "IDEA"
    core.advance(state, "OPP-001", "EVIDENCE", "3 research notes found")
    with pytest.raises(ValueError, match="requires evidence"):
        core.advance(state, "OPP-001", "IMPLEMENTATION", "skip")
    state["opportunities"][0]["evidence"].append({"source": "quality-lab", "detail": "x"})
    core.advance(state, "OPP-001", "IMPLEMENTATION", "owner approved")
    assert state["opportunities"][0]["history"][-1]["stage"] == "IMPLEMENTATION"


def test_duplicate_open_title_rejected(state):
    core.add_opportunity(state, "same idea", "FEATURE", "p")
    with pytest.raises(ValueError, match="same title"):
        core.add_opportunity(state, "same idea", "FEATURE", "p2")
    core.advance(state, "OPP-001", "REJECTED", "bad idea")
    core.add_opportunity(state, "same idea", "FEATURE", "p3")  # allowed after rejection


def test_negative_result_memory(state):
    core.add_negative_result(state, "global sharpening after retouch",
                             "improves hair, exaggerates skin texture", "rejected",
                             retry_unless="local-mask architecture available")
    hits = core.check_negative(state, "try global sharpening again")
    assert len(hits) == 1 and hits[0]["id"] == "NEG-001"
    assert not core.check_negative(state, "unrelated wig work")


def test_hypothesis_registry(state):
    hyp = core.add_hypothesis(state, "edge-confidence mask reduces wig loss",
                              ">20% fewer failures, <2% degradation elsewhere")
    assert hyp["id"] == "HYP-001" and hyp["status"] == "PROPOSED"
    core.update_hypothesis(state, "HYP-001", status="EXPERIMENTING", experiment="EXP-041")
    core.update_hypothesis(state, "HYP-001", experiment="EXP-042")
    hyp = state["hypotheses"][0]
    assert [e["id"] for e in hyp["experiments"]] == ["EXP-041", "EXP-042"]
    with pytest.raises(ValueError, match="unknown hypothesis"):
        core.update_hypothesis(state, "HYP-999", status="CONFIRMED")


def test_portfolio_balance_and_warning(state):
    for _ in range(4):
        core.log_work(state, "FEATURE", "slider")
    core.log_work(state, "QUALITY", "fix")
    port = core.portfolio(state)
    assert port["actual"]["FEATURE"] == 0.8
    assert any("FEATURE" in w for w in port["warnings"])
    # share-based form
    s2 = core.load(Path("/nonexistent/rd.json"))
    core.log_work(s2, "QUALITY", "a", share=0.35)
    core.log_work(s2, "FEATURE", "b", share=0.25)
    assert not core.portfolio(s2)["warnings"]
    core.log_work(s2, "QUALITY", "c", share=0.40)
    assert any("QUALITY" in w for w in core.portfolio(s2)["warnings"])


def test_brief_is_advisory_and_counts(state):
    core.add_opportunity(state, "a", "RESEARCH", "p")
    core.add_opportunity(state, "b", "QUALITY", "p", evidence=[{"source": "x", "detail": "y"}])
    b = core.brief(state)
    assert b["open_opportunities"] == 2
    assert b["by_stage"]["IDEA"] == 2
    assert b["top"][0]["title"] == "b"  # more evidence ranks first
    assert "advisory" in b["authority"]


def test_state_roundtrip(tmp_path):
    path = tmp_path / "rd.json"
    state = core.load(path)
    core.add_opportunity(state, "a", "PERFORMANCE", "p")
    core.save(state, path)
    loaded = core.load(path)
    assert loaded["opportunities"][0]["id"] == "OPP-001"
    assert json.loads(path.read_text())["schema"] == 1

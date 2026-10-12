"""Tests for the Phase 1.c Fuji-sims documentation.

These tests verify that the user-facing guide
(``docs/FUJI_SIMS_GUIDE.md``) and the validation report
(``docs/PHASE_1C_VALIDATION.md``) exist, are non-empty, and
reference the three official sims (``Classic Chrome``, ``Astia``,
``Provia``) and the key Fuji research points from
``docs/FUJI_COLOR_RESEARCH.md``.

Per the task brief, the user has deferred ``pytest`` execution —
this file is committed but is intended to be runnable when the
test suite is next invoked. The tests are pure file/IO checks,
no model code is exercised, so they are safe to ship and easy
to read.
"""
from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DOCS_DIR = _REPO_ROOT / "docs"
GUIDE_PATH = _DOCS_DIR / "FUJI_SIMS_GUIDE.md"
VALIDATION_PATH = _DOCS_DIR / "PHASE_1C_VALIDATION.md"
RESEARCH_PATH = _DOCS_DIR / "FUJI_COLOR_RESEARCH.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _lower(text: str) -> str:
    return text.lower()


# ---------------------------------------------------------------------------
# Existence
# ---------------------------------------------------------------------------

class TestDocsExist:
    """Both docs must exist on disk and be non-empty."""

    def test_guide_path_exists(self):
        assert GUIDE_PATH.exists(), f"Missing user guide: {GUIDE_PATH}"

    def test_validation_path_exists(self):
        assert VALIDATION_PATH.exists(), f"Missing validation doc: {VALIDATION_PATH}"

    def test_guide_is_non_empty(self):
        assert GUIDE_PATH.stat().st_size > 0

    def test_validation_is_non_empty(self):
        assert VALIDATION_PATH.stat().st_size > 0

    def test_research_doc_still_exists(self):
        """The Phase 1.a research doc is a sibling and must still be present."""
        assert RESEARCH_PATH.exists(), f"Missing research doc: {RESEARCH_PATH}"


# ---------------------------------------------------------------------------
# User-facing guide coverage
# ---------------------------------------------------------------------------

class TestGuideCoversAllSims:
    """The user guide must cover all 3 official Fuji sims."""

    @pytest.fixture(scope="class")
    def guide_text(self) -> str:
        return _read(GUIDE_PATH)

    @pytest.mark.parametrize(
        "needle",
        [
            "Classic Chrome",
            "Astia",
            "Provia",
        ],
    )
    def test_guide_mentions_sim(self, guide_text: str, needle: str):
        assert needle in guide_text, (
            f"User guide is missing a mention of {needle!r}"
        )

    def test_guide_has_overview_section(self, guide_text: str):
        assert "## Overview" in guide_text, "User guide missing 'Overview' section"

    def test_guide_has_comparison_table(self, guide_text: str):
        """Quick-reference comparison table is a required user-guide section."""
        lower = _lower(guide_text)
        # Match 'comparison' as a section header, not just a body word.
        assert "## comparison table" in lower or "### comparison" in lower, (
            "User guide missing 'Comparison Table' section"
        )

    def test_guide_has_how_to_use_section(self, guide_text: str):
        lower = _lower(guide_text)
        assert "## how to use" in lower, "User guide missing 'How to Use' section"

    def test_guide_has_limitations_section(self, guide_text: str):
        lower = _lower(guide_text)
        assert "## limitations" in lower, "User guide missing 'Limitations' section"


# ---------------------------------------------------------------------------
# Per-sim content
# ---------------------------------------------------------------------------

class TestGuidePerSimContent:
    """The user guide must give each sim a dedicated section."""

    @pytest.fixture(scope="class")
    def guide_text(self) -> str:
        return _read(GUIDE_PATH)

    def test_classic_chrome_section(self, guide_text: str):
        # Section header is `## Classic Chrome`
        assert "## Classic Chrome" in guide_text, (
            "Missing '## Classic Chrome' section in user guide"
        )
        # Key characteristic: low saturation
        lower = _lower(guide_text)
        assert "low saturation" in lower, (
            "Classic Chrome section should mention 'low saturation'"
        )
        # Key characteristic: lifted blacks
        assert "lifted black" in lower or "lifted shadow" in lower, (
            "Classic Chrome section should mention 'lifted blacks' or similar"
        )

    def test_astia_section(self, guide_text: str):
        assert "## Astia" in guide_text, "Missing '## Astia' section in user guide"
        lower = _lower(guide_text)
        # Key characteristic: skin protection / portrait sim
        assert "skin" in lower, "Astia section should mention 'skin'"
        assert "portrait" in lower, "Astia section should mention 'portrait'"

    def test_provia_section(self, guide_text: str):
        assert "## Provia" in guide_text, "Missing '## Provia' section in user guide"
        lower = _lower(guide_text)
        # Key characteristic: neutral / accurate
        assert "neutral" in lower or "accurate" in lower, (
            "Provia section should describe neutral/accurate character"
        )


# ---------------------------------------------------------------------------
# Validation doc coverage
# ---------------------------------------------------------------------------

class TestValidationCoversAllSims:
    """The validation doc must cover all 3 official Fuji sims."""

    @pytest.fixture(scope="class")
    def validation_text(self) -> str:
        return _read(VALIDATION_PATH)

    @pytest.mark.parametrize(
        "needle",
        [
            "Classic Chrome",
            "Astia",
            "Provia",
        ],
    )
    def test_validation_mentions_sim(self, validation_text: str, needle: str):
        assert needle in validation_text, (
            f"Validation doc is missing a mention of {needle!r}"
        )

    def test_validation_has_per_sim_observations(self, validation_text: str):
        """Each sim should have its own 'observations' subsection."""
        lower = _lower(validation_text)
        for sim in ("classic chrome", "astia", "provia"):
            # Looking for `### 2.X <Sim>` or `## <Sim>` style sections.
            assert f"### 2.{sim}" in lower or f"## {sim}" in lower, (
                f"Validation doc missing dedicated {sim!r} section"
            )

    def test_validation_has_tl_dr(self, validation_text: str):
        lower = _lower(validation_text)
        assert "## tl;dr" in lower or "## summary" in lower, (
            "Validation doc should have a TL;DR / Summary section"
        )

    def test_validation_has_performance_section(self, validation_text: str):
        lower = _lower(validation_text)
        assert "performance" in lower, (
            "Validation doc should have a 'Performance' section"
        )

    def test_validation_has_limitations_section(self, validation_text: str):
        lower = _lower(validation_text)
        assert "## known limitations" in lower or "## limitations" in lower, (
            "Validation doc should have a 'Known Limitations' section"
        )

    def test_validation_has_recommendations_section(self, validation_text: str):
        lower = _lower(validation_text)
        assert "recommendation" in lower, (
            "Validation doc should have a 'Recommendations' section"
        )


# ---------------------------------------------------------------------------
# Cross-references to the Fuji research doc
# ---------------------------------------------------------------------------

class TestDocsReferenceResearch:
    """Both docs should reference the key Fuji research points."""

    @pytest.fixture(scope="class")
    def guide_text(self) -> str:
        return _read(GUIDE_PATH)

    @pytest.fixture(scope="class")
    def validation_text(self) -> str:
        return _read(VALIDATION_PATH)

    @pytest.mark.parametrize(
        "research_keyword",
        [
            # The H&D / film response curve is the central research finding.
            "H&D",
            # X-Trans is the structural limit on fidelity.
            "X-Trans",
            # Color matrix / warm skin bias is a key research point.
            "color matrix",
        ],
    )
    def test_guide_references_research(
        self, guide_text: str, research_keyword: str
    ):
        # Case-insensitive substring search.
        assert research_keyword.lower() in guide_text.lower(), (
            f"User guide is missing the research keyword {research_keyword!r}"
        )

    @pytest.mark.parametrize(
        "research_keyword",
        [
            "H&D",
            "X-Trans",
            "color matrix",
        ],
    )
    def test_validation_references_research(
        self, validation_text: str, research_keyword: str
    ):
        assert research_keyword.lower() in validation_text.lower(), (
            f"Validation doc is missing the research keyword {research_keyword!r}"
        )

    def test_validation_acknowledges_demo_luts(self, validation_text: str):
        """The honest 'demo LUT, not real film stock' caveat must be present."""
        lower = validation_text.lower()
        # Either of these phrasings is acceptable; both make the same point.
        assert (
            "demo lut" in lower
            or "not real film stock" in lower
            or "17" in lower and "lut" in lower
        ), (
            "Validation doc must explicitly acknowledge the demo-LUT limitation"
        )


# ---------------------------------------------------------------------------
# Honest-limitations enforcement
# ---------------------------------------------------------------------------

class TestDocsAreHonest:
    """The docs should not overstate fidelity; they must call out the limits."""

    @pytest.fixture(scope="class")
    def guide_text(self) -> str:
        return _read(GUIDE_PATH)

    @pytest.fixture(scope="class")
    def validation_text(self) -> str:
        return _read(VALIDATION_PATH)

    def test_guide_does_not_claim_100_percent_match(self, guide_text: str):
        """A '100%' or 'exact' claim would be a red flag; the brief is 90–95%."""
        lower = guide_text.lower()
        assert "100%" not in lower, (
            "User guide must not claim a 100% match — honest limit is 90–95%"
        )

    def test_validation_does_not_claim_100_percent_match(self, validation_text: str):
        lower = validation_text.lower()
        assert "100%" not in lower, (
            "Validation doc must not claim a 100% match — honest limit is 90–95%"
        )

    def test_guide_mentions_90_95_target(self, guide_text: str):
        """The honest 90–95% match target should be present in the user guide."""
        # Acceptable phrasings: '90-95', '90–95', '90 to 95'.
        lower = guide_text.lower()
        assert (
            "90-95" in lower
            or "90–95" in lower
            or "90 to 95" in lower
        ), (
            "User guide should mention the 90–95% match target"
        )

    def test_validation_mentions_90_95_target(self, validation_text: str):
        lower = validation_text.lower()
        assert (
            "90-95" in lower
            or "90–95" in lower
            or "90 to 95" in lower
        ), (
            "Validation doc should mention the 90–95% match target"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

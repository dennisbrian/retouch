"""Unit tests for the Quality Lab (scripts/dev/quality_lab).

No engine renders here — metrics/triage/selection are exercised on
synthetic arrays and fabricated report dicts. End-to-end rendering is
covered by manually running scripts/dev/quality-lab (too slow for CI).
"""
from __future__ import annotations

import copy

import cv2
import numpy as np
import pytest

from scripts.dev.quality_lab import metrics
from scripts.dev.quality_lab.core import (IMPACT_TAG_MAP, THRESHOLDS,
                                          select_cases)
from scripts.dev.quality_lab.perf import compare_perf
from scripts.dev.quality_lab.triage import triage_case, triage_report


def _img(seed=0, w=64, h=64, base=120):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), base, np.float32)
    img += rng.normal(0, 8, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_ssim_identical_is_one(self):
        img = _img()
        assert metrics.ssim(img, img.copy()) == pytest.approx(1.0, abs=1e-3)

    def test_ssim_drops_on_blur(self):
        img = _img()
        blurred = cv2.GaussianBlur(img, (9, 9), 0)
        assert metrics.ssim(img, blurred) < 0.95

    def test_delta_e_identical_is_zero(self):
        img = _img()
        assert metrics.delta_e_mean(img, img.copy()) == pytest.approx(0.0, abs=1e-6)

    def test_texture_retention_blur_below_one(self):
        img = _img()
        blurred = cv2.GaussianBlur(img, (9, 9), 0)
        assert metrics.texture_retention(img, blurred) < 0.8

    def test_clip_fraction_counts_white(self):
        img = np.full((10, 10, 3), 255, np.uint8)
        assert metrics.clip_fraction(img, "highlight") == pytest.approx(1.0)
        assert metrics.clip_fraction(img, "shadow") == pytest.approx(0.0)

    def test_masked_stats_ignore_outside(self):
        img = np.zeros((10, 10, 3), np.uint8)
        img[:, :5] = 200
        mask = np.zeros((10, 10), np.float32)
        mask[:, :5] = 1.0
        stats = metrics.channel_stats(img, mask)
        # OpenCV Lab L is nonlinear in gray level: 200-gray -> L≈206
        assert stats["L"] == pytest.approx(206, abs=3)


# ---------------------------------------------------------------------------
# impact map / case selection
# ---------------------------------------------------------------------------

class TestSelectCases:
    MANIFEST = {"cases": [
        {"id": "a", "tags": ["face", "skin"], "input": "x/a.png"},
        {"id": "b", "tags": ["eyes", "face"], "input": "x/b.png"},
        {"id": "c", "tags": ["wig", "hair"], "input": "x/c.png"},
    ]}

    def test_docs_only_selects_nothing(self):
        cases, tags = select_cases(self.MANIFEST, changed_files=["docs/INDEX.md"])
        assert cases == [] and tags == ["none"]

    def test_eye_module_selects_eye_cases(self):
        cases, tags = select_cases(self.MANIFEST, changed_files=["retouch/eye_visibility.py"])
        assert {c["id"] for c in cases} == {"a", "b"}

    def test_engine_selects_full(self):
        cases, tags = select_cases(self.MANIFEST, changed_files=["retouch/engine.py"])
        assert len(cases) == 3 and tags == ["full"]

    def test_unknown_engine_module_moderate_sweep(self):
        cases, tags = select_cases(self.MANIFEST, changed_files=["retouch/some_new_op.py"])
        assert {c["id"] for c in cases} == {"a", "b"}

    def test_explicit_tags_win(self):
        cases, tags = select_cases(self.MANIFEST, tags=["wig"])
        assert [c["id"] for c in cases] == ["c"]

    def test_most_specific_prefix_wins(self):
        # retouch/eye appears before retouch/ in the map
        eye_prefixes = [p for p, _ in IMPACT_TAG_MAP if p.startswith("retouch/eye")]
        assert eye_prefixes and eye_prefixes[0] == "retouch/eye"


# ---------------------------------------------------------------------------
# triage
# ---------------------------------------------------------------------------

def _entry(l_drift=1.0, chroma=1.0, tex=0.95, edge=0.95,
           clip_in=0.01, clip_out=0.01):
    return {"time_s": 3.0, "peak_ram_mb": 100.0, "output": "/dev/null",
            "metrics": {
                "global": {"luminance_drift": l_drift,
                           "highlight_clip_in": clip_in, "highlight_clip_out": clip_out,
                           "shadow_clip_in": clip_in, "shadow_clip_out": clip_out,
                           "texture_retention": tex, "edge_retention": edge},
                "regions": {"SKIN": {"luminance_drift": l_drift, "chroma_drift": chroma,
                                     "coverage": 0.1,
                                     "texture_retention": tex, "edge_retention": edge}},
            }}


class TestTriage:
    def test_clean_case_passes(self):
        row = triage_case("a", "natural", _entry(), None, THRESHOLDS)
        assert row["verdict"] == "PASS"

    def test_render_error_is_regression(self):
        row = triage_case("a", "natural", {"error": "boom"}, None, THRESHOLDS)
        assert row["verdict"] == "REGRESSION"

    def test_texture_loss_flags(self):
        entry = _entry(tex=0.40)
        row = triage_case("a", "natural", entry, None, THRESHOLDS)
        assert row["verdict"] == "REGRESSION"
        assert any(f["check"] == "region_texture_retention" for f in row["findings"])

    def test_highlight_clip_explosion_flags(self):
        entry = _entry(clip_in=0.01, clip_out=0.05)  # 5x
        row = triage_case("a", "natural", entry, None, THRESHOLDS)
        assert row["verdict"] == "REGRESSION"
        assert any(f["check"] == "highlight_clip_delta" for f in row["findings"])

    def test_report_counts_and_human_attention(self):
        candidate = {"provenance": {"commit": "abc"}, "cases": {
            "ok": {"recipes": {"natural": _entry()}},
            "bad": {"recipes": {"natural": _entry(tex=0.3)}},
        }}
        result = triage_report(candidate, None, THRESHOLDS)
        assert result["counts"] == {"PASS": 1, "REVIEW": 0, "REGRESSION": 1}
        assert result["overall"] == "REGRESSION"
        assert [r["case"] for r in result["human_attention"]] == ["bad"]


# ---------------------------------------------------------------------------
# perf
# ---------------------------------------------------------------------------

class TestPerf:
    def _report(self, time_s, ram=100.0):
        return {"cases": {"a": {"recipes": {"natural": {
            "time_s": time_s, "peak_ram_mb": ram}}}}}

    def test_faster_than_floor_skips(self):
        result = compare_perf(self._report(0.05), self._report(0.10), THRESHOLDS)
        assert result["overall"] == "PASS"
        assert result["rows"][0].get("skipped") == "baseline_too_fast"

    def test_slow_candidate_flags(self):
        result = compare_perf(self._report(10.0), self._report(4.0), THRESHOLDS)
        assert result["overall"] == "REGRESSION"

    def test_ram_explosion_flags(self):
        result = compare_perf(self._report(4.0, ram=250.0), self._report(4.0), THRESHOLDS)
        assert result["overall"] == "REGRESSION"


# ---------------------------------------------------------------------------
# threshold policy integrity
# ---------------------------------------------------------------------------

class TestThresholdPolicy:
    def test_policy_shape(self):
        for key, spec in THRESHOLDS.items():
            assert spec["direction"] in ("low", "high"), key
            assert "warn" in spec and "regress" in spec, key
            if spec["direction"] == "high":
                assert spec["warn"] <= spec["regress"], key
            else:
                assert spec["warn"] >= spec["regress"], key

    def test_committed_file_matches_builtin(self):
        """thresholds.json exists to be *edited by reviewed PRs*, not to drift
        accidentally. If it diverges from the built-in policy the divergence
        must be deliberate (check-thresholds catches loosening in CI)."""
        import json
        from scripts.dev.quality_lab.core import THRESHOLDS_PATH, load_thresholds
        if not THRESHOLDS_PATH.exists():
            pytest.skip("thresholds.json not generated yet")
        active = load_thresholds()
        for key, spec in THRESHOLDS.items():
            assert key in active
            assert active[key]["warn"] == pytest.approx(spec["warn"])
            assert active[key]["regress"] == pytest.approx(spec["regress"])

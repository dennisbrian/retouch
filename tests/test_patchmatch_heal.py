"""Focused regression tests for classical PatchMatch healing and ROI blending."""

import cv2
import numpy as np
import pytest

from retouch.heal import heal_region
from retouch.patchmatch import patchmatch_fill, seamless_blend_roi


def _periodic_texture(height: int = 112, width: int = 112) -> np.ndarray:
    """A repeatable fabric-like texture with known long-range exemplars."""
    yy, xx = np.indices((height, width))
    base = ((xx * 37 + yy * 19) % 31).astype(np.float32)
    wave = 32.0 * np.sin(2.0 * np.pi * xx / 14.0) + 20.0 * np.cos(2.0 * np.pi * yy / 12.0)
    texture = np.clip(118.0 + base + wave, 0, 255).astype(np.uint8)
    return np.dstack((texture, np.roll(texture, 3, axis=1), np.roll(texture, 5, axis=0)))


def _gradient_error(actual: np.ndarray, expected: np.ndarray, mask: np.ndarray) -> float:
    actual_gray = cv2.cvtColor(actual, cv2.COLOR_BGR2GRAY).astype(np.float32)
    expected_gray = cv2.cvtColor(expected, cv2.COLOR_BGR2GRAY).astype(np.float32)
    actual_grad = cv2.Sobel(actual_gray, cv2.CV_32F, 1, 1, ksize=3)
    expected_grad = cv2.Sobel(expected_gray, cv2.CV_32F, 1, 1, ksize=3)
    return float(np.mean(np.abs(actual_grad[mask > 0] - expected_grad[mask > 0])))


class TestPatchmatchFill:
    def test_empty_mask_is_exact_noop(self) -> None:
        image = _periodic_texture(48, 48)
        empty = np.zeros(image.shape[:2], dtype=np.uint8)
        result, source_map = patchmatch_fill(image, empty, return_source_map=True)
        assert np.array_equal(result, image)
        assert np.all(source_map == -1)

    def test_is_deterministic_and_honours_source_mask(self) -> None:
        image = _periodic_texture()
        hole = np.zeros(image.shape[:2], dtype=np.uint8)
        hole[42:70, 42:70] = 255
        source = np.zeros_like(hole)
        source[:, :34] = 255

        first, first_map = patchmatch_fill(
            image, hole, source_mask=source, seed=73, return_source_map=True
        )
        second, second_map = patchmatch_fill(
            image, hole, source_mask=source, seed=73, return_source_map=True
        )

        assert np.array_equal(first, second)
        assert np.array_equal(first_map, second_map)
        mapped = first_map[hole > 0]
        assert np.all(mapped[:, 0] >= 0)
        assert np.all(source[mapped[:, 0], mapped[:, 1]] > 0)
        # A valid source centre must keep its full 7x7 patch inside the mask.
        for sy, sx in mapped[:: max(1, len(mapped) // 32)]:
            assert np.all(source[sy - 3 : sy + 4, sx - 3 : sx + 4] > 0)

    def test_repeated_texture_gradient_is_materially_better_than_telea(self) -> None:
        reference = _periodic_texture()
        hole = np.zeros(reference.shape[:2], dtype=np.uint8)
        hole[36:76, 36:76] = 255
        damaged = reference.copy()
        damaged[hole > 0] = 0

        patchmatch = patchmatch_fill(
            damaged, hole, patch_size=7, iterations=3, seed=11, max_candidates=384
        )
        telea = heal_region(damaged, hole, method="telea")
        patch_error = _gradient_error(patchmatch, reference, hole)
        telea_error = _gradient_error(telea, reference, hole)

        assert patch_error < telea_error * 0.75, (patch_error, telea_error)

    def test_float32_contract_and_tiny_source_fallback(self) -> None:
        image = _periodic_texture(48, 48).astype(np.float32)
        hole = np.zeros(image.shape[:2], dtype=np.float32)
        hole[18:30, 18:30] = 1.0
        tiny_source = np.zeros(image.shape[:2], dtype=np.uint8)
        tiny_source[2, 2] = 255

        result, source_map = patchmatch_fill(
            image, hole, source_mask=tiny_source, return_source_map=True
        )
        assert result.dtype == np.float32
        assert result.shape == image.shape
        assert np.all(source_map[hole > 0] == -1)


_SKIN_BGR = np.array([150, 170, 210], dtype=np.float32)
_PROTECTED_BGR = np.array([30, 30, 200], dtype=np.float32)


def _protected_neighbour_scene(dtype):
    """Skin with a repair hole abutting a strongly coloured protected region.

    The permitted donor mask is a 2x2 skin speck, so no 7x7 PatchMatch
    source patch fits and the degenerate-source fallback is exercised.
    """
    image = np.empty((64, 64, 3), dtype=np.float32)
    image[:] = _SKIN_BGR
    image[:, 34:] = _PROTECTED_BGR
    hole = np.zeros((64, 64), dtype=np.uint8)
    hole[22:42, 26:34] = 255
    donor = np.zeros_like(hole)
    donor[2:4, 2:4] = 255
    return image.astype(dtype), hole, donor


class TestDegenerateSourceFallbackHonoursDonors:
    """P2 (RESEARCH_RETOUCH_PROTECTION_LIFECYCLE_2026_09_22 §4): the fallback
    used to call unconstrained Telea, pulling protected pixels into the fill."""

    @staticmethod
    def _recoloured(image: np.ndarray, hole: np.ndarray, donor: np.ndarray) -> np.ndarray:
        # Repaint every forbidden pixel (neither donor nor hole) a different
        # colour. A donor-constrained fill cannot depend on them.
        other = image.copy()
        forbidden = (donor == 0) & (hole == 0)
        other[forbidden] = np.array([200, 40, 20], dtype=np.float32).astype(image.dtype)
        return other

    @pytest.mark.parametrize("dtype", [np.uint8, np.float32])
    def test_fallback_never_reads_forbidden_pixels(self, dtype) -> None:
        image, hole, donor = _protected_neighbour_scene(dtype)
        result = patchmatch_fill(image, hole, source_mask=donor)
        recoloured = patchmatch_fill(self._recoloured(image, hole, donor), hole, source_mask=donor)
        # Invariance: forbidden content must not influence the repair.
        assert np.array_equal(result[hole > 0], recoloured[hole > 0])
        # And no repaired pixel is pulled toward the protected red (the old
        # unconstrained Telea fallback put ~1/3 of them closer to it).
        repaired = result[hole > 0].astype(np.float32)
        d_skin = np.linalg.norm(repaired - _SKIN_BGR, axis=1)
        d_prot = np.linalg.norm(repaired - _PROTECTED_BGR, axis=1)
        assert not np.any(d_prot < d_skin)

    def test_heal_region_fallback_never_reads_forbidden_pixels(self) -> None:
        image, hole, donor = _protected_neighbour_scene(np.uint8)
        result = heal_region(image, hole, method="patchmatch", source_mask=donor, seamless=False)
        recoloured = heal_region(
            self._recoloured(image, hole, donor), hole, method="patchmatch",
            source_mask=donor, seamless=False,
        )
        assert np.array_equal(result[hole > 0], recoloured[hole > 0])

    def test_fallback_reports_executed_backend_and_reason(self) -> None:
        image, hole, donor = _protected_neighbour_scene(np.uint8)
        report: dict = {}
        heal_region(
            image, hole, method="patchmatch", source_mask=donor, seamless=False, report=report
        )
        assert report["requested"] == "patchmatch"
        assert report["executed"] == "telea_donor_constrained"
        assert report["reason"] == "no_full_source_patch"
        assert report["valid_source_centres"] == 0
        assert report["permitted_donor_pixels"] == 4
        assert report["fallback_pixels"] == int(np.count_nonzero(hole))
        assert report["source_map_available"] is False

    @pytest.mark.parametrize("seamless", [False, True])
    def test_abstains_when_no_permitted_donor_exists(self, seamless) -> None:
        image, hole, _ = _protected_neighbour_scene(np.uint8)
        image[hole > 0] = (0, 255, 0)  # a visible defect that must stay put
        donor = hole.copy()  # every permitted pixel is inside the hole
        report: dict = {}
        result = heal_region(
            image, hole, method="patchmatch", source_mask=donor,
            seamless=seamless, report=report,
        )
        assert np.array_equal(result, image)
        assert report["executed"] == "abstain"
        assert report["reason"] == "no_permitted_donor"

    def test_patchmatch_success_reports_patchmatch(self) -> None:
        image = _periodic_texture(56, 56)
        hole = np.zeros(image.shape[:2], dtype=np.uint8)
        hole[20:36, 20:36] = 255
        report: dict = {}
        patchmatch_fill(image, hole, seed=5, report=report)
        assert report["executed"] == "patchmatch"
        assert report["reason"] is None
        assert report["fallback_pixels"] == 0
        assert report["source_map_available"] is True
        assert report["valid_source_centres"] > 0

    @pytest.mark.parametrize("dtype", [np.uint8, np.float32])
    def test_unconstrained_fallback_matches_legacy_telea(self, dtype) -> None:
        # source_mask=None keeps the legacy fallback byte-identical: every
        # non-hole pixel is a permitted donor.
        image = _periodic_texture(12, 12).astype(dtype)
        hole = np.zeros(image.shape[:2], dtype=np.uint8)
        hole[3:9, 3:9] = 255
        result = patchmatch_fill(image, hole)
        if dtype == np.uint8:
            expected = cv2.inpaint(image, hole, 3, cv2.INPAINT_TELEA)
        else:
            expected = np.stack(
                [cv2.inpaint(image[..., c], hole, 3, cv2.INPAINT_TELEA) for c in range(3)],
                axis=-1,
            )
        assert np.array_equal(result, expected)


class TestSeamlessBlendRoi:
    def test_reduces_deliberate_boundary_step_without_touching_outside(self) -> None:
        base = np.full((72, 72, 3), 90, dtype=np.uint8)
        filled = base.copy()
        mask = np.zeros(base.shape[:2], dtype=np.uint8)
        mask[20:52, 20:52] = 255
        filled[mask > 0] = 180

        result = seamless_blend_roi(base, filled, mask, feather_radius=4)
        boundary_inside = np.zeros_like(mask, dtype=bool)
        boundary_inside[20:22, 20:52] = True
        raw_step = np.mean(np.abs(filled[boundary_inside].astype(np.float32) - 90.0))
        blended_step = np.mean(np.abs(result[boundary_inside].astype(np.float32) - 90.0))

        assert blended_step < raw_step
        assert np.array_equal(result[mask == 0], base[mask == 0])

    def test_float32_contract(self) -> None:
        base = np.full((40, 40, 3), 110.0, dtype=np.float32)
        filled = base.copy()
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[10:30, 10:30] = 255
        filled[mask > 0] = 145.0
        result = seamless_blend_roi(base, filled, mask)
        assert result.dtype == np.float32
        assert result.shape == base.shape


class TestHealRegionPatchmatchWire:
    def test_patchmatch_wire_and_invalid_method(self) -> None:
        image = _periodic_texture(56, 56)
        hole = np.zeros(image.shape[:2], dtype=np.uint8)
        hole[20:36, 20:36] = 255
        damaged = image.copy()
        damaged[hole > 0] = 0
        result = heal_region(damaged, hole, method="patchmatch", seed=5)
        assert result.dtype == np.uint8
        assert result.shape == image.shape
        with pytest.raises(ValueError, match="method"):
            heal_region(damaged, hole, method="invalid")

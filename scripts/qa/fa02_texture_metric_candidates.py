"""RESEARCH-ONLY candidate eligibility metrics for FA-02 micro-texture
restoration. Not wired into production. Nothing here may be imported by
retouch/*.py.

This module exists to test one hypothesis: that the shipped production
metric (``retouch.fa02_texture_eligibility.highpass_std`` at a FIXED 2px
Gaussian sigma, registered here as ``legacy_highpass_std_px2`` so it cannot
be confused with a tuned successor) is confounded by face capture scale and
sensor noise rather than measuring recoverable skin/hair microstructure.

Every candidate below is a pure function of (canvas, mask, face_width_px).
None of them may be imported into a production module, and none may be
treated as a replacement gate without a larger, deliberately varied corpus
than the 9-11 faces this round of research used (see
docs/plans/RESEARCH_FA02_METRIC_VALIDITY_2026_09_06.md).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from retouch.fa02_texture_eligibility import (  # noqa: E402
    HIGHPASS_SIGMA_PX,
    estimate_noise_sigma,
    highpass_std as _production_highpass_std,
)

__all__ = [
    "legacy_highpass_std_px2",
    "face_scaled_highpass_std",
    "noise_floor_subtracted_detail",
    "METRIC_REGISTRY",
]


def legacy_highpass_std_px2(
    canvas: np.ndarray, mask: Optional[np.ndarray], face_width_px: float,
) -> Optional[float]:
    """The FROZEN production baseline, unchanged, imported not reimplemented.

    Named explicitly so no experimental variant below can be confused with
    it. This function is a thin, non-diverging wrapper: it calls the exact
    shipped ``retouch.fa02_texture_eligibility.highpass_std`` at its fixed
    ``HIGHPASS_SIGMA_PX = 2.0``, ignoring ``face_width_px`` entirely -- which
    IS the property under test (requirement: "fixed pixel-scale high-pass").
    """
    return _production_highpass_std(canvas, mask=mask, sigma=HIGHPASS_SIGMA_PX)


def face_scaled_highpass_std(
    canvas: np.ndarray, mask: Optional[np.ndarray], face_width_px: float,
    reference_face_width_px: float = 500.0,
) -> Optional[float]:
    """Candidate A: tie the high-pass sigma to face scale instead of a fixed
    pixel count.

    Reuses the FA-02 frozen convention already in
    ``fa02_texture_eligibility.scaled_sigmas``: sigma 2.0 is defined at the
    reference face width 500 px (inter-eye distance 200), so
    ``sigma = 2.0 * face_width_px / 500`` recovers exactly 2.0 at that
    reference and scales linearly elsewhere. This directly tests whether
    tying the measurement scale to the face (rather than the sensor pixel
    grid) removes the capture-scale confound.
    """
    if not np.isfinite(face_width_px) or face_width_px <= 0:
        return None
    sigma = HIGHPASS_SIGMA_PX * float(face_width_px) / reference_face_width_px
    sigma = max(sigma, 0.3)  # avoid a degenerate near-zero kernel
    return _production_highpass_std(canvas, mask=mask, sigma=sigma)


def noise_floor_subtracted_detail(
    canvas: np.ndarray, mask: Optional[np.ndarray], face_width_px: float,
) -> Optional[float]:
    """Candidate B: signal-vs-noise decomposition.

    sqrt(max(0, hp_var - k * noise_var)) at the face-scaled sigma from
    Candidate A. Subtracts an estimate of the noise floor's own contribution
    to the high-pass variance before taking the square root, so a face whose
    highpass_std is dominated by flat-block noise (rather than structure)
    should score near zero instead of scoring high.

    k=1.0: the flat-block noise estimate and the high-pass residual are both
    "deviation from local smoothness" measures over comparable supports, so
    a 1:1 subtraction is the natural first attempt, NOT a fitted constant.
    This is explicitly unvalidated -- see the module docstring.
    """
    if mask is None:
        return None
    binary_mask = (np.asarray(mask) > 0.5).astype(np.uint8)
    hp = face_scaled_highpass_std(canvas, binary_mask, face_width_px)
    noise = estimate_noise_sigma(canvas, mask=binary_mask)
    if hp is None or noise is None:
        return None
    hp_var = hp ** 2
    noise_var = noise ** 2
    return float(np.sqrt(max(0.0, hp_var - noise_var)))


METRIC_REGISTRY = {
    "legacy_highpass_std_px2": legacy_highpass_std_px2,
    "face_scaled_highpass_std": face_scaled_highpass_std,
    "noise_floor_subtracted_detail": noise_floor_subtracted_detail,
}

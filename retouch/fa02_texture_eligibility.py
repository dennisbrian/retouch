"""FA-02 production eligibility / abstention gate for micro-texture restoration.

This is the PRODUCTION-facing counterpart to the offline research gate in
``scripts/qa/fa02_scoring_contract.py::evaluate_eligibility``. It is a
*separate*, deliberately re-implemented module, not an import of that script:

* the research gate consumes an owner-approved **annotation record** (manual
  or ``owner_approved`` supports, populated ground-truth pore/hair categories).
  A live single-image render has no human in the loop, so that input does not
  exist here and cannot be faked;
* the research gate runs on saved native ``X``/``S`` arrays with externally
  supplied ``allow``/``corrected``/``protected`` masks. Inside
  ``perf_optimizations._process_face_core`` the equivalent quantities are the
  live pre-smooth canvas, the post-smooth canvas and ``regions.skin`` — the
  same *concepts*, different shapes and lifetimes.

What IS carried over verbatim is the measured concept set and the metric
definitions, so the two gates stay comparable: face scale, texture amplitude
(``highpass_std`` at sigma 2.0), a noise/compression level, and support
validity.

**Nothing in this module is calibrated.** Every threshold below is provisional
or an outright placeholder, and each carries its provenance in
``THRESHOLD_PROVENANCE``. Read those notes before trusting any number here.
The gate is conservative by construction: any input that cannot be measured
produces ``eligible=False``, never a pass.

The functions here are pure — no I/O, no logging, no global state. Logging of
the returned decision is the caller's job (see
``perf_optimizations._process_face_core``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

__all__ = [
    "HIGHPASS_SIGMA_PX",
    "MIN_HIGHPASS_STD",
    "MIN_FACE_WIDTH_PX",
    "MAX_NOISE_SIGMA",
    "MIN_SUPPORT_PX",
    "THRESHOLD_PROVENANCE",
    "highpass_std",
    "estimate_noise_sigma",
    "evaluate_face_eligibility",
]


# ---------------------------------------------------------------------------
# Thresholds. Every one of these is UNCALIBRATED. See THRESHOLD_PROVENANCE.
# ---------------------------------------------------------------------------

#: Gaussian sigma for the high-pass texture measure, in pixels.
#: MEASURED (borrowed): the exact sigma of the ff47-event visual measurement
#: (commit 427ae49, "highpass std over a 2px Gaussian recorded per tile") that
#: produced the only two texture anchors this project owns. Changing this makes
#: MIN_HIGHPASS_STD meaningless, because the anchors are in *this* unit.
#: Note ``retouch/image_analyzer.py``'s Laplacian variance is a DIFFERENT
#: quantity; its numbers are not interchangeable with these.
HIGHPASS_SIGMA_PX = 2.0

#: Minimum texture amplitude (std of luma minus its sigma-2 Gaussian low-pass)
#: over the candidate restoration support.
#:
#: PROVISIONAL, unreplicated: the only committed patch with a genuine fine-hair
#: label (DSCF1058) measures 3.12 and FAILS this threshold, while a patch whose
#: own annotation calls its signal sensor/ISO noise (DSCF1606) measures 3.74 and
#: PASSES it. This number is not yet trustworthy and must not be treated as
#: calibrated. See docs/plans/EXPERIMENT_FA02_TEXTURE_REPRESENTATIONS_2026_09_06.md
#: and the FA-02 scoring lock's own eligibility.minimum_highpass_std note.
#:
#: Compare the eye-gate precedent (CLAUDE.md, 2026-08-31): a threshold set from
#: a single mislabelled anchor missed 84% of true positives until it was
#: re-derived from study data. Assume the same failure mode is live here.
MIN_HIGHPASS_STD = 3.5

#: Minimum face width in pixels (``ied * 2.5``, the pipeline's own convention).
#:
#: PLACEHOLDER, no measurement or calibration exists. Chosen only so the gate
#: has a functioning face-scale input. Rationale for the magnitude only, NOT a
#: validated boundary: the FA-02 frozen sigma stack is specified at inter-eye
#: distance 200 (face width 500) with a 0.6 px sigma floor, and 250 is the
#: half-reference point at which the SECOND band (sigma_1) lifts off that floor.
#:
#: IMPORTANT and measured this session -- this floor does NOT make every arm
#: usable. At face width exactly 250 the sigmas are [0.6, 0.6, 1.2, 2.4, 4.8]:
#: sigma_0 and sigma_1 are BOTH clamped to the floor, so band 0 is identically
#: zero and the A2 ``dog`` arm is a guaranteed no-op AT THIS GATE'S OWN FLOOR.
#: Band 0 only becomes nonzero above face width 250, and only equals the frozen
#: ``G.6 - G1.2`` at face width >= 500 (at fw=300 it is 0.6-vs-0.72, a
#: compressed ratio, not the frozen band). A3 ``multiscale`` is unaffected: its
#: band 1 carries weight 0.5 and resolves from fw=250 upward.
#:
#: Consequence: ``dog`` cannot be meaningfully evaluated on faces below ~500 px
#: face width regardless of corpus size or annotation quality. See
#: :func:`collapsed_bands`. Never checked against a real corpus; do not trust.
MIN_FACE_WIDTH_PX = 250.0

#: Maximum tolerated noise sigma (8-bit levels) over flat sub-regions of the
#: restoration support.
#:
#: PLACEHOLDER, no measurement or calibration exists, chosen only to have a
#: functioning gate — must not be trusted. There is NO validated noise or
#: compression measurement anywhere in this project: the FA-02 scoring lock's
#: own ``noise_amplification`` term is recorded as blocked on all 10 real
#: patches, so no threshold was ever derived from data. The value below is a
#: round number placed above typical low-ISO sensor noise and below visibly
#: noisy captures, on no evidence at all. Recalibrating this requires a real
#: study, not a tweak.
MAX_NOISE_SIGMA = 6.0

#: Minimum number of support pixels for the measurement to mean anything.
#:
#: PLACEHOLDER. Mirrors the 100 px floor the eye-visibility gate uses for its
#: own "is this mask big enough to measure" question (``retouch/eye_visibility.py``,
#: CLAUDE.md 2026-08-26). Borrowed for consistency, not measured for this gate.
MIN_SUPPORT_PX = 100


#: Machine-readable provenance for every threshold, mirroring the documentation
#: discipline of ``scripts/qa/fa02_scoring_contract.py``'s ``SCORING_RULE``.
#: Emitted in the eligibility decision so a log line carries its own caveats.
THRESHOLD_PROVENANCE: Dict[str, Dict[str, Any]] = {
    "highpass_sigma_px": {
        "value": HIGHPASS_SIGMA_PX,
        "provenance": "MEASURED (borrowed from commit 427ae49 ff47-event measurement)",
    },
    "minimum_highpass_std": {
        "value": MIN_HIGHPASS_STD,
        "provenance": "PROVISIONAL, unreplicated, contradicted by its own anchors",
        "note": (
            "DSCF1058 (only genuine fine-hair label) measures 3.12 and FAILS; "
            "DSCF1606 (annotated as sensor/ISO noise) measures 3.74 and PASSES. "
            "Not calibrated."
        ),
    },
    "minimum_face_width_px": {
        "value": MIN_FACE_WIDTH_PX,
        "provenance": "PLACEHOLDER, never measured",
        "note": "Half the FA-02 frozen reference face width (500 px at IED 200).",
    },
    "maximum_noise_sigma": {
        "value": MAX_NOISE_SIGMA,
        "provenance": "PLACEHOLDER, no measurement or calibration exists",
        "note": (
            "The FA-02 scoring lock's noise_amplification term is blocked on all "
            "10 real patches; no noise threshold was ever derived from data."
        ),
    },
    "minimum_support_px": {
        "value": MIN_SUPPORT_PX,
        "provenance": "PLACEHOLDER, borrowed from eye_visibility's 100 px floor",
    },
}


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def _luma(image: np.ndarray) -> np.ndarray:
    """Rec.601 luma of a BGR array, or the array itself when single-channel.

    Matches ``scripts/qa/fa02_texture_representation_experiment.py::luma`` so
    the production measurement and the research measurement are the same number
    on the same pixels.
    """
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3 and array.shape[2] >= 3:
        return (
            array[..., 0] * 0.114
            + array[..., 1] * 0.587
            + array[..., 2] * 0.299
        )
    return array


def _gaussian(plane: np.ndarray, sigma: float, truncate: float = 3.0) -> np.ndarray:
    """Explicit-radius Gaussian with REFLECT_101 padding.

    Mirrors the frozen FA-02 convention (``ceil(3 sigma)`` radius, REFLECT_101)
    rather than letting OpenCV derive a kernel from the sigma, so the production
    high-pass is bit-comparable with the offline harness's.
    """
    radius = int(np.ceil(float(sigma) * truncate))
    ksize = 2 * radius + 1
    return cv2.GaussianBlur(
        plane, (ksize, ksize), float(sigma), borderType=cv2.BORDER_REFLECT_101,
    )


def highpass_std(
    image_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    sigma: Optional[float] = None,
) -> Optional[float]:
    """Texture amplitude: std of luma minus its Gaussian low-pass.

    This is the FA-02 scoring lock's ``highpass_std``, re-implemented for the
    live render path. It must be measured on the PRE-smooth canvas (the
    production analogue of the harness's native ``X``), never on the smoothed
    result — the smoothed canvas is exactly the thing whose texture was removed.

    Tone-invariance: this is an amplitude measure of deviation from the local
    mean, not an absolute intensity threshold, so it does not scale with the
    subject's skin tone (CLAUDE.md, Tone-Invariance & Fairness).

    Args:
        image_bgr: ``(H, W, 3)`` or ``(H, W)`` array in [0, 255].
        mask: Optional ``(H, W)`` support; only pixels ``> 0`` are measured.
        sigma: Gaussian sigma; defaults to :data:`HIGHPASS_SIGMA_PX`.

    Returns:
        The standard deviation as a float, or ``None`` when the support selects
        no pixels (unmeasurable — the caller must abstain, never assume a pass).
    """
    if sigma is None:
        sigma = HIGHPASS_SIGMA_PX
    plane = _luma(image_bgr)
    residual = plane - _gaussian(plane, sigma)
    if mask is not None:
        selected = residual[np.asarray(mask) > 0]
        if selected.size == 0:
            return None
        return float(np.std(selected.astype(np.float64)))
    if residual.size == 0:
        return None
    return float(np.std(residual.astype(np.float64)))


def estimate_noise_sigma(
    image_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    block: int = 8,
    quiet_fraction: float = 0.25,
) -> Optional[float]:
    """Estimate noise/compression level as the median std of the flattest blocks.

    PLACEHOLDER MEASUREMENT. The *concept* is borrowed from
    ``retouch/image_analyzer.py::_estimate_noise`` (split into blocks, keep the
    quietest quarter by Laplacian energy, take a robust centre of their stds),
    but that implementation is a Python double loop over every 8x8 block, which
    violates CLAUDE.md's no-Python-pixel-loops rule in a per-face hot path and
    would also drag ``image_analyzer`` into this module's import graph. This is
    a vectorised reimplementation kept local so this module stays a numpy/cv2
    leaf.

    It conflates sensor noise, film grain, JPEG ringing and real fine texture —
    a flat-block std cannot separate them. That is acknowledged, not solved:
    there is no validated noise measurement in this project to defer to (the
    FA-02 scoring lock's ``noise_amplification`` term is blocked on every real
    patch it was run against). Treat the number as an ordering hint at best.

    Args:
        image_bgr: ``(H, W, 3)`` or ``(H, W)`` array in [0, 255].
        mask: Optional ``(H, W)`` support; a block is only considered when it is
            fully inside the support, so a block straddling the mask boundary
            cannot report the boundary step as "noise".
        block: Block edge length in pixels.
        quiet_fraction: Fraction of blocks (by lowest Laplacian energy) kept.

    Returns:
        Estimated sigma in 8-bit levels, or ``None`` when the support contains
        no fully-covered block (unmeasurable — the caller must abstain).
    """
    plane = _luma(image_bgr)
    h, w = plane.shape[:2]
    if h < block * 2 or w < block * 2:
        return None

    n_y = h // block
    n_x = w // block
    if n_y < 1 or n_x < 1:
        return None

    # Vectorised block reduction: crop to a whole number of blocks, then reshape
    # so each block becomes one row. No Python loop over blocks.
    cropped = plane[: n_y * block, : n_x * block]
    tiles = cropped.reshape(n_y, block, n_x, block).transpose(0, 2, 1, 3)
    tiles = tiles.reshape(n_y * n_x, block * block)

    if mask is not None:
        mask_f = np.asarray(mask, dtype=np.float32)
        if mask_f.shape[:2] != plane.shape[:2]:
            return None
        m_cropped = mask_f[: n_y * block, : n_x * block]
        m_tiles = m_cropped.reshape(n_y, block, n_x, block).transpose(0, 2, 1, 3)
        m_tiles = m_tiles.reshape(n_y * n_x, block * block)
        # Fully-covered blocks only. A partially covered block would measure the
        # mask boundary (a hard skin/hair step) rather than the noise floor.
        keep = np.all(m_tiles > 0.5, axis=1)
        tiles = tiles[keep]
        if tiles.shape[0] == 0:
            return None

    stds = tiles.std(axis=1)
    # "Flatness" proxy: the block's own std ordering is the same ordering a
    # Laplacian-energy ranking would produce on flat vs. textured blocks, and it
    # needs no second filter pass. Keep the quietest quarter.
    n_keep = max(1, int(round(stds.shape[0] * float(quiet_fraction))))
    quietest = np.partition(stds, n_keep - 1)[:n_keep]
    return float(np.median(quietest.astype(np.float64)))


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def evaluate_face_eligibility(
    pre_smooth_canvas: np.ndarray,
    support_mask: Optional[np.ndarray],
    face_width_px: float,
    protected_mask: Optional[np.ndarray] = None,
    *,
    min_highpass_std: Optional[float] = None,
    min_face_width_px: Optional[float] = None,
    max_noise_sigma: Optional[float] = None,
    min_support_px: Optional[int] = None,
) -> Dict[str, Any]:
    """Decide whether FA-02 micro-texture restoration may run on this face.

    Four gates, ALL of which must pass. Any input that cannot be measured
    abstains; the gate never defaults to eligible.

    1. **Face scale** — ``face_width_px`` (the pipeline's ``ied * 2.5``) at or
       above ``min_face_width_px``. Below that, the frozen FA-02 sigma stack
       collapses onto its 0.6 px floor and there is no band left to restore.
    2. **Sharpness / detail observability** — ``highpass_std`` on the pre-smooth
       canvas inside the effective support, at or above ``min_highpass_std``.
       Below it there is no texture information present to restore, so
       restoring is amplifying noise.
    3. **Noise / compression** — the flat-block noise proxy at or below
       ``max_noise_sigma``. Restoration re-injects whatever is in the residual;
       on a noisy or heavily compressed source that residual IS the noise.
    4. **Support validity** — the mask restoration would write through exists,
       is non-degenerate, and retains at least ``min_support_px`` pixels after
       the active mark-policy protection is removed.

    On support validity and ``mark_policy``: ``protected_mask`` is the FA-01
    preserve mask (``None`` when no policy is active, which is the default).
    An ABSENT policy is not invalid support — it simply means nothing is
    protected. Only a policy that protects effectively the whole support
    invalidates it.

    Args:
        pre_smooth_canvas: ``(H, W, 3)`` float32 [0, 255] canvas BEFORE
            frequency-separation smoothing (production analogue of native ``X``).
        support_mask: ``(H, W)`` float32 [0, 1] restoration support (skin), or
            ``None``.
        face_width_px: Face width in pixels (``shifted_face.ied * 2.5``).
        protected_mask: Optional ``(H, W)`` float32 [0, 1] mark-policy preserve
            mask. ``None`` means no policy is active.
        min_highpass_std: Override; ``None`` reads :data:`MIN_HIGHPASS_STD`.
        min_face_width_px: Override; ``None`` reads :data:`MIN_FACE_WIDTH_PX`.
        max_noise_sigma: Override; ``None`` reads :data:`MAX_NOISE_SIGMA`.
        min_support_px: Override; ``None`` reads :data:`MIN_SUPPORT_PX`.

    Note:
        The threshold overrides resolve from the module constants *inside* the
        body, not as keyword defaults. Keyword defaults would bind at import
        time, which would silently make the module constants unpatchable and
        the "thresholds are configurable" contract above untrue.

    Returns:
        ``{"eligible": bool, "reason": str, "measured": {...},
        "thresholds": {...}, "effective_support": ndarray | None}``.
        ``reason`` is ``"eligible"`` on a pass, otherwise a semicolon-joined
        list of every failing gate (all gates are evaluated, so a log line
        shows every problem, not just the first).
        ``effective_support`` is the support the caller must write through — the
        input support minus any protected pixels — or ``None`` when the gate
        abstained on support validity.
    """
    # Resolved here, not as keyword defaults: a keyword default binds the value
    # at import time, so patching the module constant would have no effect and
    # the documented "configurable thresholds" contract would be a lie.
    if min_highpass_std is None:
        min_highpass_std = MIN_HIGHPASS_STD
    if min_face_width_px is None:
        min_face_width_px = MIN_FACE_WIDTH_PX
    if max_noise_sigma is None:
        max_noise_sigma = MAX_NOISE_SIGMA
    if min_support_px is None:
        min_support_px = MIN_SUPPORT_PX

    reasons = []
    measured: Dict[str, Any] = {}

    # ---- Gate 1: face scale -------------------------------------------------
    try:
        width = float(face_width_px)
    except (TypeError, ValueError):
        width = float("nan")
    measured["face_width_px"] = None if not np.isfinite(width) else width
    if not np.isfinite(width):
        reasons.append("face scale not measured (face_width_px is not finite)")
    elif width < min_face_width_px:
        reasons.append(
            "insufficient face scale: face_width_px={0:.4g} < {1:.4g} "
            "(PLACEHOLDER floor)".format(width, min_face_width_px)
        )

    # ---- Gate 4 (first, it produces the support the others measure over) ----
    effective_support: Optional[np.ndarray] = None
    support_ok = False
    if support_mask is None:
        reasons.append("support invalid: no restoration support mask supplied")
    else:
        support = np.asarray(support_mask, dtype=np.float32)
        if support.ndim != 2 or support.shape[:2] != np.asarray(
            pre_smooth_canvas
        ).shape[:2]:
            reasons.append(
                "support invalid: mask shape {0} does not match canvas {1}".format(
                    support.shape, np.asarray(pre_smooth_canvas).shape[:2],
                )
            )
        elif not np.all(np.isfinite(support)):
            reasons.append("support invalid: mask contains non-finite values")
        else:
            # Same 0-255-vs-0-1 detection the rest of the render path uses.
            # Threshold 1.5, not 1.0: feathered float masks overshoot 1.0 by an
            # ulp (CLAUDE.md, 2026-09-02 composite-mask epsilon bug).
            if support.max() > 1.5:
                support = support / 255.0
            support = np.clip(support, 0.0, 1.0)

            candidate = support
            if protected_mask is not None:
                protect = np.asarray(protected_mask, dtype=np.float32)
                if protect.shape[:2] == support.shape[:2]:
                    if protect.max() > 1.5:
                        protect = protect / 255.0
                    protect = np.clip(protect, 0.0, 1.0)
                    # Mark-policy preserved pixels are removed from the support:
                    # restoration is a re-injection op over the region, so a
                    # protected mark must not be written through.
                    candidate = np.clip(support - protect, 0.0, 1.0)
                else:
                    reasons.append(
                        "support invalid: protected mask shape {0} does not match "
                        "support {1}".format(protect.shape, support.shape)
                    )
                    candidate = None

            if candidate is not None:
                selected_px = int(np.count_nonzero(candidate > 0.5))
                measured["support_px"] = selected_px
                measured["support_px_before_protection"] = int(
                    np.count_nonzero(support > 0.5)
                )
                if selected_px < min_support_px:
                    reasons.append(
                        "support invalid: {0} usable px after mark-policy protection "
                        "< {1} (PLACEHOLDER floor)".format(selected_px, min_support_px)
                    )
                else:
                    support_ok = True
                    effective_support = candidate

    if "support_px" not in measured:
        measured["support_px"] = None

    # ---- Gates 2 and 3: measured over the effective support -----------------
    # Both are unmeasurable without a valid support, and an unmeasurable gate
    # abstains. They are still recorded as None so the log line is complete.
    texture = None
    noise = None
    if support_ok and effective_support is not None:
        binary_support = (effective_support > 0.5).astype(np.uint8)
        texture = highpass_std(pre_smooth_canvas, mask=binary_support)
        noise = estimate_noise_sigma(pre_smooth_canvas, mask=binary_support)

        if texture is None or not np.isfinite(texture):
            reasons.append(
                "source texture not measured (highpass_std returned no value over "
                "the effective support)"
            )
        elif float(texture) < min_highpass_std:
            reasons.append(
                "insufficient source texture: highpass_std={0:.4g} < {1:.4g} "
                "(PROVISIONAL floor, contradicted by its own anchors)".format(
                    float(texture), min_highpass_std,
                )
            )

        if noise is None or not np.isfinite(noise):
            reasons.append(
                "noise/compression not measured (no fully-covered flat block inside "
                "the effective support)"
            )
        elif float(noise) > max_noise_sigma:
            reasons.append(
                "excessive noise/compression: noise_sigma={0:.4g} > {1:.4g} "
                "(PLACEHOLDER ceiling)".format(float(noise), max_noise_sigma)
            )
    else:
        reasons.append(
            "sharpness and noise not measurable: no valid support to measure over"
        )

    measured["highpass_std"] = (
        None if texture is None or not np.isfinite(texture) else float(texture)
    )
    measured["noise_sigma"] = (
        None if noise is None or not np.isfinite(noise) else float(noise)
    )
    measured["mark_policy_active"] = protected_mask is not None

    eligible = not reasons
    return {
        "eligible": bool(eligible),
        "reason": "eligible" if eligible else "; ".join(reasons),
        "measured": measured,
        "thresholds": {
            "minimum_highpass_std": min_highpass_std,
            "minimum_face_width_px": min_face_width_px,
            "maximum_noise_sigma": max_noise_sigma,
            "minimum_support_px": min_support_px,
            "highpass_sigma_px": HIGHPASS_SIGMA_PX,
        },
        "threshold_status": (
            "UNCALIBRATED: minimum_highpass_std is PROVISIONAL and contradicted by "
            "its own anchors; face-scale, noise and support-size floors are "
            "PLACEHOLDERS with no measurement behind them. See "
            "THRESHOLD_PROVENANCE and "
            "docs/plans/EXPERIMENT_FA02_TEXTURE_REPRESENTATIONS_2026_09_06.md."
        ),
        "effective_support": effective_support,
    }


# ---------------------------------------------------------------------------
# Frozen candidate detail signals (A1 / A2 / A3 only)
# ---------------------------------------------------------------------------

#: Gaussian sigmas at the FA-02 reference inter-eye distance of 200 px
#: (face width 500 px). FROZEN — see the "Frozen arms" section of
#: docs/plans/EXPERIMENT_FA02_TEXTURE_REPRESENTATIONS_2026_09_06.md.
SIGMAS_AT_IED_200: Tuple[float, ...] = (0.6, 1.2, 2.4, 4.8, 9.6)

#: Floor below which a scaled sigma is not allowed to fall. Collapsed bands are
#: reported, never upsampled into invented detail.
MINIMUM_SIGMA_PX = 0.6

#: A3's frozen band weights: band 0 at 1.0, band 1 at 0.5, the rest zero.
MULTISCALE_WEIGHTS: Tuple[float, ...] = (1.0, 0.5, 0.0, 0.0)


def collapsed_bands(face_width_px: float) -> Tuple[int, ...]:
    """Indices of band differences that have collapsed to identically zero.

    A band ``i`` is the difference ``G(sigma_i) - G(sigma_i+1)``. When a face is
    small enough that both sigmas clamp to :data:`MINIMUM_SIGMA_PX`, that
    difference is exactly zero and the band carries no signal at all.

    The frozen FA-02 spec is explicit that collapsed bands are *logged, not
    upsampled into invented detail*. This is a real operating condition, not an
    error: e.g. at face width 132.9 px (the golden fixture) sigmas 0 and 1 both
    clamp to 0.6, so the A2 DoG arm -- which uses band 0 alone -- produces an
    identically zero detail signal and the render is a no-op. A2 and A3 are
    NOT interchangeable at small face scales for this reason.
    """
    sigmas = scaled_sigmas(face_width_px)
    return tuple(
        i for i in range(len(sigmas) - 1) if sigmas[i] == sigmas[i + 1]
    )


def scaled_sigmas(face_width_px: float) -> Tuple[float, ...]:
    """Scale the frozen sigma stack to this face's size.

    The harness scales by ``inter_eye_distance_px / 200``. The render path
    carries ``face_width = ied * 2.5`` instead, so the equivalent scale is
    ``face_width_px / 500``. Reimplemented locally rather than imported from
    ``scripts/qa/fa02_texture_representation_experiment.py::geometry`` — a
    production module must not depend on a QA script.
    """
    scale = float(face_width_px) / 500.0
    return tuple(max(MINIMUM_SIGMA_PX, s * scale) for s in SIGMAS_AT_IED_200)


def extract_detail(
    residual: np.ndarray,
    mode: str,
    face_width_px: float,
) -> np.ndarray:
    """Frozen FA-02 detail signal for one candidate arm.

    ``residual`` is ``D = X - S`` (pre-smooth minus post-smooth canvas), the
    same quantity every frozen arm operates on. All arms are signed,
    same-coordinate differences with no coefficient denoising.

    Frozen formulas (EXPERIMENT_FA02_TEXTURE_REPRESENTATIONS_2026_09_06.md):

    * ``dog`` (A2):        ``G.6(D) - G1.2(D)``
    * ``multiscale`` (A3): ``[G.6 - G1.2](D) + .5 [G1.2 - G2.4](D)``,
      remaining band weights zero.

    A4 (orientation-aware) and A5 (existing frequency bands) are deliberately
    NOT implemented here and must not be added without an explicit owner
    decision — they are reserved research arms, not production candidates.

    Args:
        residual: ``(H, W, 3)`` float32 signed residual.
        mode: ``"dog"`` or ``"multiscale"``.
        face_width_px: Face width used to scale the frozen sigma stack.

    Returns:
        ``(H, W, 3)`` float32 signed detail signal.

    Raises:
        ValueError: If ``mode`` is not a supported frozen arm.
    """
    sigmas = scaled_sigmas(face_width_px)
    residual = np.asarray(residual, dtype=np.float32)

    if mode == "dog":
        indices = (0,)
    elif mode == "multiscale":
        indices = tuple(i for i, w in enumerate(MULTISCALE_WEIGHTS) if w != 0.0)
    else:
        raise ValueError(
            "extract_detail supports only the frozen 'dog' and 'multiscale' arms; "
            "got {0!r}. A4/A5 are reserved and must not be dispatched.".format(mode)
        )

    # One blur per distinct level, shared between adjacent band differences.
    levels = {}
    for i in sorted(set(list(indices) + [i + 1 for i in indices])):
        levels[i] = _gaussian(residual, sigmas[i])

    out = np.zeros_like(residual)
    for i in indices:
        weight = 1.0 if mode == "dog" else float(MULTISCALE_WEIGHTS[i])
        out += weight * (levels[i] - levels[i + 1])
    return out

"""Colour fringe removal along bright edges (``purple_fringing``, GUI "Defringe").

Lenses bend blue, green and red light by slightly different amounts, so a
dark edge against something very bright (a wig or a black glove against a
blown window, a costume edge against a softbox, a bokeh highlight) picks up a
thin purple, blue, cyan or green band that isn't in the scene. Alex's white
wig photos against a blown window show it on gloves, shoulders and the edges
of black vinyl, and the bunny photos show it round the softbox bokeh.

Why it replaced the first version (``utils.remove_purple_fringing``): that
one only touched pixels whose own 3x3 luminance gradient passed a fixed level
of 35 and whose colour was magenta-purple (a* > 8, b* < -5). A real fringe is
a band several pixels wide that sits mostly *beside* the edge, where the
pixel's own gradient is small, and the blue, cyan and green fringes failed
the colour gate outright, so on Alex's wig photos it removed little. It also
greyed any purple object's edge that passed both gates.

This version:

* finds edges by local contrast, not a fixed gradient: within a band whose
  width scales with the image, the brightest pixel must sit near the photo's
  own highlight level (its 99.5th luminance percentile, so a dim photo works
  the same) and the band must span a large share of that level;
* takes purple, blue, cyan and green hues (reds, oranges, yellows and skin
  are outside the hue gate, so lips, skin, blush and gold trim never move);
* replaces the fringe's colour (a*/b*, lightness kept) with the colour of
  the nearby pixels *outside* the fringe band, found by a normalised blur.
  A real purple or green object is the same colour beyond the band, so its
  colour comes back unchanged; a fringe on a black or white object is
  replaced by that object's own near-neutral colour;
* where nothing outside the band is close enough to borrow from (a thin
  strand or strap that lies wholly inside the band), it falls back to
  taking the colour toward neutral, like a camera raw defringe.

Pixels outside the band are returned bit-identical. No skin or face level is
involved: the gates are relative to the photo's own highlights, so the op
behaves the same whatever the subject's skin tone (``tests/test_defringe.py``).
"""

from __future__ import annotations

import cv2
import numpy as np

# Band half-width as a fraction of the image's long side (14 px at 6240 px).
_BAND_FRAC = 0.0022
_BAND_MIN_PX = 2
_BAND_MAX_PX = 24
# The colour reference is borrowed from this many band widths around a pixel.
_SUPPORT_MULT = 3.0
# Highlight level: this percentile of the photo's L*.
_HI_PCT = 99.5
# The brightest pixel in the band must reach this share of the highlight level.
_BRIGHT_LO = 0.80
_BRIGHT_HI = 0.92
# Brightest minus darkest in the band, as a share of the highlight level.
_CONTRAST_LO = 0.25
_CONTRAST_HI = 0.40
# Fringe hues in CIELab degrees [0, 360): green (125) through cyan and blue
# to purple (330). Ramps on both sides keep yellow-green and pink/magenta
# objects mostly out; red, orange, yellow and skin (about 10-100) never pass.
_HUE_IN_LO = 110.0
_HUE_FULL_LO = 125.0
_HUE_FULL_HI = 330.0
_HUE_IN_HI = 345.0
# Chroma ramp (a*/b* units): below this a pixel is already neutral.
_CHROMA_LO = 3.0
_CHROMA_HI = 8.0
# Band pixels this far from the bright side (in band widths) count as
# support again: the fringe itself is the part nearest the bright side.
_CORE_LO = 0.6
_CORE_HI = 1.0
# Lightness bins for the reference colour (centres span 0..highlight level).
_L_BINS = 9
# A lightness bin counts as present near a pixel when its support covers at
# least this share of the support window; with no bin present the pixel
# falls back toward neutral.
_BIN_MIN = 0.02


def _ramp(x, lo: float, hi: float):
    return np.clip((x - lo) * np.float32(1.0 / max(hi - lo, 1e-6)), 0.0, 1.0)


def _hue_gate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    hue = np.degrees(np.arctan2(b, a)) % 360.0
    rise = _ramp(hue, _HUE_IN_LO, _HUE_FULL_LO)
    fall = 1.0 - _ramp(hue, _HUE_FULL_HI, _HUE_IN_HI)
    return np.minimum(rise, fall).astype(np.float32)


def band_radius(shape: tuple) -> int:
    """Fringe band half-width in pixels for an image of this shape."""
    r = int(round(_BAND_FRAC * max(shape[0], shape[1])))
    return int(np.clip(r, _BAND_MIN_PX, _BAND_MAX_PX))


def fringe_weight(img_bgr: np.ndarray) -> np.ndarray:
    """Per-pixel fringe weight in [0, 1] (edge band x hue x chroma).

    ``img_bgr`` is uint8 or float BGR in [0, 1] (float) / [0, 255] (uint8).
    Exposed for tests and the QA overlay.
    """
    weight, _, _, _ = _analyse(_to_unit_float(img_bgr))
    return weight


def _to_unit_float(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) * np.float32(1.0 / 255.0)
    return np.clip(img.astype(np.float32, copy=False), 0.0, 1.0)


def _analyse(f: np.ndarray):
    """Return (weight, lab, r, hi). Hue and chroma are only evaluated on the
    edge band, which is a small share of the frame."""
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    h, w = L.shape
    r = band_radius((h, w))
    weight = np.zeros((h, w), np.float32)

    step = max(1, int(np.sqrt(h * w / 250_000)))
    hi = float(np.percentile(L[::step, ::step], _HI_PCT))
    if hi <= 1.0:
        return weight, lab, r, hi

    # Band extrema on an 8-bit copy of L* (fast morphology; the gates are
    # soft ramps, so 0.4 L* quantisation doesn't matter).
    L8 = cv2.convertScaleAbs(L, alpha=2.55)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
    lmax = cv2.dilate(L8, k)
    lmin = cv2.erode(L8, k)
    hi8 = hi * 2.55
    cand = (lmax >= np.uint8(min(255, int(_BRIGHT_LO * hi8)))) & (
        cv2.subtract(lmax, lmin) >= np.uint8(min(255, int(_CONTRAST_LO * hi8)))
    )
    idx = np.flatnonzero(cand)
    if idx.size == 0:
        return weight, lab, r, hi
    mx = lmax.ravel()[idx].astype(np.float32) / hi8
    mn = lmin.ravel()[idx].astype(np.float32) / hi8
    edge = _ramp(mx, _BRIGHT_LO, _BRIGHT_HI) * _ramp(mx - mn, _CONTRAST_LO, _CONTRAST_HI)
    ab = lab[:, :, 1:].reshape(-1, 2)[idx]
    a = ab[:, 0]
    b = ab[:, 1]
    chroma = np.hypot(a, b)
    weight.ravel()[idx] = edge * _hue_gate(a, b) * _ramp(chroma, _CHROMA_LO, _CHROMA_HI)
    return weight, lab, r, hi


def _reference_colour(lab: np.ndarray, support: np.ndarray, r: int, hi: float):
    """Lightness-aware normalised blur of a*/b* over ``support``.

    Returns (ref_a, ref_b, trust, ds) on a grid ``ds`` times coarser than
    the image; trust is 0 where no support pixel lies within the window.
    """
    h, w = support.shape
    ds = max(1, r // 3)
    sh, sw_ = max(1, h // ds), max(1, w // ds)
    def down(x):
        return cv2.resize(x, (sw_, sh), interpolation=cv2.INTER_AREA)

    L, a, b = cv2.split(lab)
    Lt = down(L)                             # target lightness (all pixels)
    ss = down(support)                       # support density
    inv = 1.0 / np.maximum(ss, 1e-6)
    Ls = down(cv2.multiply(L, support)) * inv  # support-only means
    as_ = down(cv2.multiply(a, support)) * inv
    bs = down(cv2.multiply(b, support)) * inv
    ksz = int(2 * round(_SUPPORT_MULT * r / ds) + 1)
    # Per lightness bin (triangular membership): the support's mean L, a*, b*
    # and density near each grid cell.
    centres = np.linspace(0.0, 1.0, _L_BINS) * hi
    width = centres[1] - centres[0]
    n = len(centres)
    bl = np.zeros((n,) + Lt.shape, np.float32)
    ba = np.zeros_like(bl)
    bb = np.zeros_like(bl)
    bw = np.zeros_like(bl)
    for j, c in enumerate(centres):
        ws = ss * np.clip(1.0 - np.abs(Ls - c) / width, 0.0, 1.0)
        dw = cv2.blur(ws, (ksz, ksz))
        inv_w = 1.0 / np.maximum(dw, 1e-6)
        bl[j] = cv2.blur(Ls * ws, (ksz, ksz)) * inv_w
        ba[j] = cv2.blur(as_ * ws, (ksz, ksz)) * inv_w
        bb[j] = cv2.blur(bs * ws, (ksz, ksz)) * inv_w
        bw[j] = dw
    ok = bw > _BIN_MIN
    # Nearest supported bin at or below / at or above the pixel's own
    # lightness; the reference colour is interpolated between them by
    # lightness, so a real colour edge (light blue against black, navy
    # against a window) predicts its own transition colours and stays.
    below = np.where(ok & (bl <= Lt[None]), np.arange(n)[:, None, None], -1).max(0)
    above = np.where(ok & (bl > Lt[None]), np.arange(n)[:, None, None], n).min(0)
    has_b = below >= 0
    has_a = above < n
    jb = np.clip(below, 0, n - 1)[None]
    ja = np.clip(above, 0, n - 1)[None]
    pick = lambda arr, j: np.take_along_axis(arr, j, 0)[0]
    lb, la_ = pick(bl, jb), pick(bl, ja)
    t = np.clip((Lt - lb) / np.maximum(la_ - lb, 1e-3), 0.0, 1.0)
    t = np.where(has_b & has_a, t, np.where(has_a, 1.0, 0.0)).astype(np.float32)
    ref_a = (1 - t) * pick(ba, jb) + t * pick(ba, ja)
    ref_b = (1 - t) * pick(bb, jb) + t * pick(bb, ja)
    trust = (has_b | has_a).astype(np.float32)
    return ref_a, ref_b, trust, ds


def _sample(grid: np.ndarray, ys: np.ndarray, xs: np.ndarray, ds: int) -> np.ndarray:
    """Bilinear sample of a coarse grid at full-resolution pixel coords."""
    gh, gw = grid.shape
    gy = np.clip((ys + 0.5) / ds - 0.5, 0, gh - 1)
    gx = np.clip((xs + 0.5) / ds - 0.5, 0, gw - 1)
    y0 = np.floor(gy).astype(np.int64)
    x0 = np.floor(gx).astype(np.int64)
    y1 = np.minimum(y0 + 1, gh - 1)
    x1 = np.minimum(x0 + 1, gw - 1)
    fy = (gy - y0).astype(np.float32)
    fx = (gx - x0).astype(np.float32)
    top = grid[y0, x0] * (1 - fx) + grid[y0, x1] * fx
    bot = grid[y1, x0] * (1 - fx) + grid[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def remove_fringes(img_bgr: np.ndarray, strength: float) -> np.ndarray:
    """Remove purple/blue/cyan/green fringes along bright high-contrast edges.

    Args:
        img_bgr: (H, W, 3) BGR, float32 in [0, 1] or uint8.
        strength: 0-100. 0 returns the input unchanged.

    Returns:
        Same dtype and range as the input. Pixels the op doesn't touch are
        returned bit-identical.
    """
    s = float(np.clip(strength, 0.0, 100.0)) / 100.0
    if s <= 0.0 or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        return img_bgr

    f = _to_unit_float(img_bgr)
    weight, lab, r, hi = _analyse(f)
    touched = weight > 0.01
    if not touched.any():
        return img_bgr

    # Colour reference: normalised blur of a*/b* over nearby pixels that are
    # not fringe candidates, borrowed from pixels of similar lightness (so the
    # dark side of an edge borrows from the dark object, the bright side from
    # the bright one). Candidates are dilated a little so the soft tail of a
    # fringe isn't borrowed back. The reference colour is smooth, so it is
    # computed at reduced resolution.
    cand = cv2.dilate((weight > 0.05).astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    # Fringe sits right against the bright side; band pixels further in
    # (the middle of a thin glove or strap) carry the object's own colour,
    # so they count as partial support. Without this a thin navy glove
    # against a window borrowed the wall's colour.
    L8 = cv2.convertScaleAbs(lab[:, :, 0], alpha=2.55)
    bright = (L8 >= np.uint8(min(255, int(_BRIGHT_LO * hi * 2.55)))).astype(np.uint8)
    dist = cv2.distanceTransform(1 - bright, cv2.DIST_L2, 3)
    deep = _ramp(dist, _CORE_LO * r, _CORE_HI * r)
    support = (1 - cand).astype(np.float32) + cand.astype(np.float32) * deep
    ref_a, ref_b, trust, ds = _reference_colour(lab, support, r, hi)
    ys, xs = np.nonzero(touched)
    # Too little support nearby: fall back toward neutral.
    t = _sample(trust, ys, xs, ds)
    ref = np.stack([_sample(ref_a, ys, xs, ds) * t,
                    _sample(ref_b, ys, xs, ds) * t], axis=1)
    k = (s * weight[ys, xs])[:, None]
    ab = lab[ys, xs, 1:]
    px = lab[ys, xs].copy()
    px[:, 1:] = ab + k * (ref - ab)
    out = cv2.cvtColor(px[:, None, :], cv2.COLOR_LAB2BGR)[:, 0, :]
    out = np.clip(out, 0.0, 1.0)

    res = img_bgr.copy()
    if img_bgr.dtype == np.uint8:
        res[ys, xs] = np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)
    else:
        res[ys, xs] = out.astype(res.dtype)
    return res

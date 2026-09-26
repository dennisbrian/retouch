"""Watermark / credit overlay for finished photos.

Reposts strip names; a small credit on the image survives them. This module
stamps a text credit (``"© Alex Studio"``, ``"@cosplayer · ph: @me"``) and/or
a logo onto *copies* of retouched photos, so the clean masters stay untouched:

* **Face-aware placement.** ``position="auto"`` (the default) tries the
  bottom-right, bottom-left, top-right, top-left and bottom-centre spots in
  that order and takes the first one that stays clear of every face (with a
  margin for wigs and headpieces), so a credit never lands on the cosplayer.
* **Readable anywhere.** Text colour follows the background under the mark
  (white on dark, black on light) and a soft shadow keeps it legible on busy
  convention backdrops. A logo keeps its own colours and transparency.
* **Scales with the photo.** Size and margin are fractions of the short side,
  so a 1080 px crop and a 26 MP master get the same-looking mark.
* **Safe to ship.** The built-in font is Pillow's embedded Aileron Regular
  (dotcolon.net, CC0 public domain). It covers Latin letters, digits, ``©``
  and ``@`` but not accented or CJK characters; pass ``font=`` a .ttf/.otf you
  own for those (a warning names the missing characters).

Copies are written as sRGB JPEG under ``<output>/watermarked/``. Camera EXIF
(GPS, serial numbers) is not carried over; the credit is also written into
the EXIF Copyright field (``©`` as ``(C)``).

Usage::

    python -m retouch.watermark out/ --text "© Alex Studio {year}"
    python cli.py shoot/ -o out/ --watermark "© Alex Studio"   # during a batch
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

_logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]  # (x, y, w, h) in image pixels

POSITIONS = ("auto", "bottom-right", "bottom-left", "top-right", "top-left", "bottom-center")
# Order "auto" tries spots in: the usual credit corner first.
AUTO_ORDER = ("bottom-right", "bottom-left", "top-right", "top-left", "bottom-center")
# Face boxes stop at the hairline and jaw; grow them by this many face sizes
# before testing overlap so wigs, ears and headpieces count as "face".
FACE_PAD = 0.5
# Output folder for watermarked copies (skipped by social crops / review).
WATERMARK_DIRNAME = "watermarked"

DEFAULT_OPACITY = 0.7
DEFAULT_SIZE = 0.03    # font size as a fraction of the image's short side
DEFAULT_MARGIN = 0.03  # gap to the edge as a fraction of the short side
_MIN_FONT_PX = 12


@dataclasses.dataclass
class WatermarkSpec:
    """What to stamp and how. ``text`` and/or ``logo`` must be set."""

    text: str = ""
    logo: Optional[Union[str, Path]] = None
    position: str = "auto"
    opacity: float = DEFAULT_OPACITY
    size: float = DEFAULT_SIZE
    margin: float = DEFAULT_MARGIN
    font: Optional[Union[str, Path]] = None
    color: str = "auto"  # auto | white | black

    def __post_init__(self) -> None:
        self.text = (self.text or "").strip()
        if self.position not in POSITIONS:
            raise ValueError(
                f"position must be one of {', '.join(POSITIONS)}, got {self.position!r}"
            )
        if self.color not in ("auto", "white", "black"):
            raise ValueError(f"color must be auto, white or black, got {self.color!r}")
        if not 0.0 <= float(self.opacity) <= 1.0:
            raise ValueError(f"opacity must be 0-1, got {self.opacity!r}")
        if not 0.005 <= float(self.size) <= 0.2:
            raise ValueError(f"size must be 0.005-0.2 of the short side, got {self.size!r}")
        if not 0.0 <= float(self.margin) <= 0.2:
            raise ValueError(f"margin must be 0-0.2 of the short side, got {self.margin!r}")
        if self.logo is not None and not Path(self.logo).is_file():
            raise ValueError(f"logo file not found: {self.logo}")
        if self.font is not None and not Path(self.font).is_file():
            raise ValueError(f"font file not found: {self.font}")

    @property
    def enabled(self) -> bool:
        return bool(self.text) or self.logo is not None

    def resolved_text(self) -> str:
        """Text with ``{year}`` filled in."""
        return self.text.replace("{year}", str(datetime.date.today().year))


@dataclasses.dataclass
class MarkLayer:
    """Premultiplication-free RGBA mark: colour (H,W,3 float 0-1 RGB), alpha (H,W)."""

    rgb: np.ndarray
    alpha: np.ndarray
    text_alpha: np.ndarray  # alpha of the text only (recoloured per image)
    shadow: np.ndarray      # soft shadow alpha, same size

    @property
    def size(self) -> Tuple[int, int]:
        h, w = self.alpha.shape
        return w, h


# --------------------------------------------------------------------- fonts

def load_font(size_px: int, font_path: Optional[Union[str, Path]] = None):
    """A FreeType font at ``size_px``; the CC0 built-in one when no path is given."""
    from PIL import ImageFont

    size_px = max(_MIN_FONT_PX, int(size_px))
    if font_path is not None:
        return ImageFont.truetype(str(font_path), size_px)
    try:
        return ImageFont.load_default(size=size_px)
    except TypeError:  # Pillow < 10.1: bitmap fallback, fixed size
        _logger.warning("watermark: Pillow >= 10.1 needed for a scalable built-in font")
        return ImageFont.load_default()


def missing_glyphs(text: str, font) -> List[str]:
    """Characters ``font`` draws as the empty 'missing glyph' box."""
    try:
        notdef = font.getmask("￿")
        ref = (notdef.size, bytes(notdef))
    except Exception:
        return []
    missing = []
    for ch in dict.fromkeys(text):
        if ch.isspace():
            continue
        m = font.getmask(ch)
        if (m.size, bytes(m)) == ref:
            missing.append(ch)
    return missing


# ------------------------------------------------------------------ building

def build_mark(spec: WatermarkSpec, short_side: int) -> Optional[MarkLayer]:
    """Render the logo + text for an image whose short side is ``short_side``."""
    from PIL import Image, ImageDraw

    if not spec.enabled:
        return None
    font_px = max(_MIN_FONT_PX, int(round(spec.size * short_side)))
    text = spec.resolved_text()

    parts = []  # (rgba uint8 array, is_text)
    if spec.logo is not None:
        logo = Image.open(spec.logo).convert("RGBA")
        target_h = max(8, int(round(font_px * (1.6 if text else 2.5))))
        scale = target_h / max(1, logo.height)
        logo = logo.resize(
            (max(1, int(round(logo.width * scale))), target_h), Image.LANCZOS,
        )
        parts.append((np.asarray(logo), False))
    if text:
        font = load_font(font_px, spec.font)
        missing = missing_glyphs(text, font)
        if missing:
            _logger.warning(
                "watermark: the %s font has no glyph for %s; pass a .ttf/.otf "
                "that covers them",
                "chosen" if spec.font else "built-in", " ".join(missing),
            )
        left, top, right, bottom = font.getbbox(text)
        pad = max(2, font_px // 8)
        w, h = right - left + 2 * pad, bottom - top + 2 * pad
        canvas = Image.new("L", (w, h), 0)
        ImageDraw.Draw(canvas).text((pad - left, pad - top), text, font=font, fill=255)
        a = np.asarray(canvas)
        rgba = np.zeros((h, w, 4), np.uint8)
        rgba[..., :3] = 255
        rgba[..., 3] = a
        parts.append((rgba, True))

    gap = int(round(font_px * 0.5)) if len(parts) == 2 else 0
    total_w = sum(p.shape[1] for p, _ in parts) + gap
    total_h = max(p.shape[0] for p, _ in parts)
    # Room around the mark for its shadow.
    blur = max(1.0, font_px * 0.08)
    border = int(np.ceil(blur * 3)) + 1
    W, H = total_w + 2 * border, total_h + 2 * border
    rgb = np.ones((H, W, 3), np.float32)
    alpha = np.zeros((H, W), np.float32)
    text_alpha = np.zeros((H, W), np.float32)
    x = border
    for arr, is_text in parts:
        ph, pw = arr.shape[:2]
        y = border + (total_h - ph) // 2
        a = arr[..., 3].astype(np.float32) / 255.0
        alpha[y:y + ph, x:x + pw] = np.maximum(alpha[y:y + ph, x:x + pw], a)
        if is_text:
            text_alpha[y:y + ph, x:x + pw] = a
        else:
            rgb[y:y + ph, x:x + pw] = arr[..., :3].astype(np.float32) / 255.0
        x += pw + gap
    shift = max(1, int(round(font_px * 0.04)))
    shadow = np.zeros_like(alpha)
    shadow[shift:, shift:] = alpha[:-shift, :-shift]
    shadow = cv2.GaussianBlur(shadow, (0, 0), blur) * 0.55
    return MarkLayer(rgb=rgb, alpha=alpha, text_alpha=text_alpha, shadow=shadow)


# ----------------------------------------------------------------- placement

def _spot(position: str, img_w: int, img_h: int, mark_w: int, mark_h: int,
          margin_px: int) -> Tuple[int, int]:
    """Top-left corner of the mark for a named spot, clamped inside the image."""
    if position.endswith("left"):
        x = margin_px
    elif position.endswith("right"):
        x = img_w - margin_px - mark_w
    else:
        x = (img_w - mark_w) // 2
    y = margin_px if position.startswith("top") else img_h - margin_px - mark_h
    x = int(min(max(0, x), max(0, img_w - mark_w)))
    y = int(min(max(0, y), max(0, img_h - mark_h)))
    return x, y


def _overlap(a: Box, b: Box) -> int:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    w = min(ax + aw, bx + bw) - max(ax, bx)
    h = min(ay + ah, by + bh) - max(ay, by)
    return max(0, w) * max(0, h)


def _padded(box: Box, pad: float = FACE_PAD) -> Box:
    x, y, w, h = box
    px, py = w * pad, h * pad
    return int(x - px), int(y - py), int(w + 2 * px), int(h + 2 * py)


def choose_position(
    img_w: int, img_h: int, mark_w: int, mark_h: int, margin_px: int,
    faces: Sequence[Box] = (), position: str = "auto",
) -> Tuple[str, int, int]:
    """(spot name, x, y) for the mark. ``auto`` avoids the faces."""
    if position != "auto":
        return (position, *_spot(position, img_w, img_h, mark_w, mark_h, margin_px))
    padded = [_padded(b) for b in faces]
    best = None
    for name in AUTO_ORDER:
        x, y = _spot(name, img_w, img_h, mark_w, mark_h, margin_px)
        hit = sum(_overlap((x, y, mark_w, mark_h), f) for f in padded)
        if hit == 0:
            return name, x, y
        if best is None or hit < best[0]:
            best = (hit, name, x, y)
    return best[1], best[2], best[3]


# ---------------------------------------------------------------- compositing

def _value_scale(img: np.ndarray) -> float:
    if img.dtype == np.uint8:
        return 255.0
    if img.dtype == np.uint16:
        return 65535.0
    return 255.0 if float(np.max(img)) > 1.5 else 1.0


def apply_watermark(
    img: np.ndarray,
    spec: WatermarkSpec,
    faces: Sequence[Box] = (),
    mark: Optional[MarkLayer] = None,
) -> Tuple[np.ndarray, Optional[str]]:
    """Stamp ``spec`` onto a copy of BGR ``img``; returns (image, spot used).

    ``faces`` are (x, y, w, h) boxes in image pixels used by ``position="auto"``.
    The result keeps ``img``'s dtype and value range. A disabled spec or an
    image too small for the mark returns the input unchanged and ``None``.
    """
    if not spec.enabled or img is None or img.ndim < 2:
        return img, None
    h, w = img.shape[:2]
    if mark is None:
        mark = build_mark(spec, min(h, w))
    if mark is None:
        return img, None
    mw, mh = mark.size
    if mw > w or mh > h:
        _logger.warning("watermark: %dx%d mark does not fit a %dx%d image; skipped",
                        mw, mh, w, h)
        return img, None
    margin_px = int(round(spec.margin * min(h, w)))
    spot, x, y = choose_position(w, h, mw, mh, margin_px, faces, spec.position)

    scale = _value_scale(img)
    out = img.copy()
    region = out[y:y + mh, x:x + mw].astype(np.float32) / scale
    gray = region if region.ndim == 2 else region[..., :3]
    if gray.ndim == 3:
        # BGR luma of what sits under the mark decides white or black text.
        lum = float(np.median(gray[..., 0] * 0.114 + gray[..., 1] * 0.587 + gray[..., 2] * 0.299))
    else:
        lum = float(np.median(gray))
    colour = spec.color
    if colour == "auto":
        colour = "black" if lum > 0.62 else "white"
    ink = 1.0 if colour == "white" else 0.0
    shade = 1.0 - ink

    rgb = mark.rgb.copy()
    t = mark.text_alpha > 0
    rgb[t] = ink
    bgr = rgb[..., ::-1]
    op = float(spec.opacity)
    a = (mark.alpha * op)[..., None]
    s = (mark.shadow * op * (1.0 - mark.alpha))[..., None]
    if region.ndim == 2:
        bgr = bgr.mean(axis=2)
        a, s = a[..., 0], s[..., 0]
        comp = region * (1 - s) + shade * s
        comp = comp * (1 - a) + bgr * a
    else:
        base = region[..., :3]
        comp3 = base * (1 - s) + shade * s
        comp3 = comp3 * (1 - a) + bgr * a
        comp = region.copy()
        comp[..., :3] = comp3
    comp = np.clip(comp * scale, 0, scale)
    if np.issubdtype(out.dtype, np.integer):
        comp = np.rint(comp)
    out[y:y + mh, x:x + mw] = comp.astype(out.dtype)
    return out, spot


# ------------------------------------------------------------------- files

@dataclasses.dataclass
class WatermarkResult:
    path: Path
    status: str  # done | skipped | failed
    spot: Optional[str] = None
    error: Optional[str] = None


def watermark_path(out_dir: Union[str, Path], image_path: Union[str, Path]) -> Path:
    return Path(out_dir) / f"{Path(image_path).stem}.jpg"


def credit_exif(text: str) -> Optional[bytes]:
    """EXIF carrying only a Copyright field.

    EXIF text is ASCII, so ``©`` is written as ``(C)``; a credit with other
    non-ASCII characters gets no EXIF field rather than a mangled one.
    """
    text = (text or "").replace("\u00a9", "(C)").strip()
    if not text or not text.isascii():
        return None
    from PIL import Image

    exif = Image.Exif()
    exif[0x8298] = text  # Copyright
    return exif.tobytes()


def _make_detector():
    from retouch.detection import FaceDetector

    return FaceDetector(max_faces=10, allow_unavailable=True)


def detect_faces(img: np.ndarray, detector=None) -> List[Box]:
    """Face boxes for placement; empty when detection is off or fails."""
    if detector is None:
        return []
    from retouch.social_crops import detect_faces as _detect

    try:
        return _detect(img, detector)
    except Exception as exc:
        _logger.debug("watermark: face detection failed: %s", exc)
        return []


def export_watermarked(
    image_path: Union[str, Path],
    out_dir: Union[str, Path],
    spec: WatermarkSpec,
    *,
    detector=None,
    quality: int = 92,
    force: bool = False,
) -> WatermarkResult:
    """Write a watermarked JPEG copy of ``image_path`` into ``out_dir``."""
    from retouch.io import imread_exif_with_context, write_image_with_color_context

    image_path = Path(image_path)
    dest = watermark_path(out_dir, image_path)
    if dest.exists() and not force:
        return WatermarkResult(dest, "skipped")
    try:
        img, color_context = imread_exif_with_context(image_path)
        if img is None:
            raise ValueError("could not read image")
        if img.dtype != np.uint8:
            img = np.clip(np.rint(img.astype(np.float32) / _value_scale(img) * 255.0),
                          0, 255).astype(np.uint8)
        faces = detect_faces(img, detector) if spec.position == "auto" else []
        out, spot = apply_watermark(img, spec, faces)
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_image_with_color_context(
            str(dest), out, color_context, bit_depth=8, quality=quality,
            exif=credit_exif(spec.resolved_text()),
        )
        return WatermarkResult(dest, "done", spot)
    except Exception as exc:
        _logger.warning("watermark: %s: %s", image_path.name, exc)
        return WatermarkResult(dest, "failed", error=str(exc))


def export_folder(
    paths: Sequence[Union[str, Path]],
    out_dir: Union[str, Path],
    spec: WatermarkSpec,
    *,
    quality: int = 92,
    force: bool = False,
    detector=None,
    progress=None,
) -> dict:
    """Watermark every path; creates (and closes) a face detector for ``auto``."""
    own_detector = detector is None and spec.position == "auto"
    if own_detector:
        try:
            detector = _make_detector()
        except Exception as exc:
            _logger.warning("watermark: face detection unavailable (%s); "
                            "using the bottom-right corner", exc)
            detector = None
    summary = {"done": 0, "skipped": 0, "failed": 0, "results": []}
    try:
        for i, path in enumerate(paths):
            r = export_watermarked(path, out_dir, spec, detector=detector,
                                   quality=quality, force=force)
            summary[r.status] += 1
            summary["results"].append(r)
            if progress is not None:
                progress(i + 1, len(paths), Path(path).name)
    finally:
        if own_detector and detector is not None and hasattr(detector, "close"):
            detector.close()
    return summary


def format_summary(summary: dict, out_dir: Union[str, Path]) -> str:
    return (
        f"Watermark: {summary['done']} written, {summary['skipped']} skipped, "
        f"{summary['failed']} failed → {out_dir}"
    )


def add_cli_args(parser: argparse.ArgumentParser, prefix: str = "") -> None:
    """Style options shared by this module's CLI and ``cli.py`` (``--watermark-*``)."""
    p = prefix
    parser.add_argument(f"--{p}logo", default=None, metavar="PNG",
                        help="Logo image (PNG with transparency) placed before the text")
    parser.add_argument(f"--{p}position", choices=POSITIONS, default="auto",
                        help="Where the mark goes; auto = first corner clear of faces "
                             "(default: auto)")
    parser.add_argument(f"--{p}opacity", type=int, default=int(DEFAULT_OPACITY * 100),
                        metavar="0-100", help="Mark opacity (default: 70)")
    parser.add_argument(f"--{p}size", type=float, default=DEFAULT_SIZE * 100,
                        metavar="PCT",
                        help="Text size as %% of the photo's short side (default: 3)")
    parser.add_argument(f"--{p}font", default=None, metavar="TTF",
                        help="Font file for the text (default: built-in Aileron; "
                             "use your own for accented or CJK names)")
    parser.add_argument(f"--{p}color", choices=["auto", "white", "black"], default="auto",
                        help="Text colour; auto follows the background (default)")


def spec_from_args(text: str, args: argparse.Namespace, prefix: str = "") -> WatermarkSpec:
    """Build a spec from :func:`add_cli_args` options (raises ValueError)."""
    g = lambda name: getattr(args, (prefix + name).replace("-", "_"))  # noqa: E731
    opacity = g("opacity")
    if not 0 <= opacity <= 100:
        raise ValueError(f"opacity must be 0-100, got {opacity}")
    return WatermarkSpec(
        text=text or "", logo=g("logo"), position=g("position"),
        opacity=opacity / 100.0, size=g("size") / 100.0,
        font=g("font"), color=g("color"),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m retouch.watermark",
        description="Stamp a credit and/or logo onto copies of retouched photos.",
    )
    parser.add_argument("input", help="Retouched image or folder")
    parser.add_argument("-o", "--output",
                        help=f"Folder for the copies (default: <input>/{WATERMARK_DIRNAME})")
    parser.add_argument("--text", default="",
                        help='Credit text, e.g. "© Alex Studio {year}" ({year} = this year)')
    add_cli_args(parser)
    parser.add_argument("--quality", type=int, default=92, help="JPEG quality (default: 92)")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing copies")
    parser.add_argument("-r", "--recursive", action="store_true", help="Include subfolders")
    args = parser.parse_args(argv)

    try:
        spec = spec_from_args(args.text, args)
    except ValueError as exc:
        parser.error(str(exc))
    if not spec.enabled:
        parser.error("give --text and/or --logo")

    from retouch.social_crops import find_crop_sources

    src = Path(args.input)
    if not src.exists():
        parser.error(f"not found: {src}")
    paths = find_crop_sources(src, recursive=args.recursive)
    if not paths:
        print(f"No images found in {src}")
        return 1
    base = src if src.is_dir() else src.parent
    out_dir = Path(args.output) if args.output else base / WATERMARK_DIRNAME
    summary = export_folder(
        paths, out_dir, spec, quality=args.quality, force=args.force,
        progress=lambda i, n, name: print(f"[{i}/{n}] {name}"),
    )
    for r in summary["results"]:
        if r.status == "failed":
            print(f"  ✖ {r.path.name}: {r.error}")
    print(format_summary(summary, out_dir))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())

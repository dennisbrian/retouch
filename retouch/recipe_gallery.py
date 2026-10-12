"""Recipe gallery: preview every recipe on the user's own photo.

The GUI's "Recipe Gallery" panel and the ``python3 -m retouch.recipe_gallery``
command both use this module. It shrinks one photo to a small preview size,
runs face detection once, then renders each recipe on that preview so the
user can compare looks side by side instead of guessing from recipe names.

Public API:
    gallery_groups()                          -> List[str]
    gallery_recipe_names(group)               -> List[str]
    prepare_gallery_source(img, max_dim)      -> np.ndarray (uint8 BGR)
    RecipeGalleryRenderer                     — renders + caches thumbnails
    before_after(before_rgb, after_rgb)       -> np.ndarray (uint8 RGB)
    contact_sheet(tiles, columns)             -> np.ndarray (uint8 RGB)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .recipe_cookbook import list_categories, list_recipes
from .recipes import (
    CONDITIONAL_RECIPE_NAMES,
    CURATED_RECIPE_NAMES,
    RECOMMENDED_RECIPE_NAMES,
)

_logger = logging.getLogger(__name__)

# Long edge of the preview each recipe renders at. Small enough that the 65
# curated recipes finish in about a minute on a laptop CPU, big enough to judge
# skin, eyes and colour.
GALLERY_MAX_DIM = 512

# Thumbnails kept in memory across gallery runs (about 0.6 MB each at 512 px).
GALLERY_CACHE_SIZE = 200

GROUP_RECOMMENDED = "Recommended"
GROUP_SCENE = "Scene / creative"
GROUP_ALL = "All curated"


def gallery_groups() -> List[str]:
    """Return the group choices the gallery offers, broadest last."""
    return [GROUP_RECOMMENDED, GROUP_SCENE] + [
        c.capitalize() for c in list_categories()
    ] + [GROUP_ALL]


def gallery_recipe_names(group: str = GROUP_RECOMMENDED) -> List[str]:
    """Return curated recipe names for ``group``, in the recipe list's order.

    Only curated recipes are offered, the same set the recipe picker shows.
    Cookbook categories (portrait, cosplay, ...) are filtered to that set.
    """
    key = (group or GROUP_RECOMMENDED).strip()
    if key == GROUP_RECOMMENDED:
        return list(RECOMMENDED_RECIPE_NAMES)
    if key == GROUP_SCENE:
        return list(CONDITIONAL_RECIPE_NAMES)
    if key == GROUP_ALL:
        return list(CURATED_RECIPE_NAMES)
    category = key.lower()
    if category not in list_categories():
        raise ValueError(f"Unknown gallery group: {group!r}")
    in_category = {info.name for info in list_recipes(category)}
    return [name for name in CURATED_RECIPE_NAMES if name in in_category]


def prepare_gallery_source(img_bgr: np.ndarray, max_dim: int = GALLERY_MAX_DIM) -> np.ndarray:
    """Return a uint8 BGR copy of ``img_bgr`` with its long edge at most ``max_dim``."""
    if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] < 3:
        raise ValueError("Recipe gallery needs a colour (BGR) image")
    img = img_bgr[:, :, :3]
    if img.dtype == np.uint16:
        img = (img.astype(np.float32) / 257.0).round()
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating) and float(img.max(initial=0.0)) <= 1.5:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    h, w = img.shape[:2]
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    if scale < 1.0:
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)


def _to_rgb_uint8(img_bgr: np.ndarray) -> np.ndarray:
    arr = np.asarray(img_bgr)[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def _failed_tile(shape: Tuple[int, ...]) -> np.ndarray:
    tile = np.full((shape[0], shape[1], 3), 60, dtype=np.uint8)
    cv2.putText(tile, "preview failed", (10, shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (230, 230, 230), 1, cv2.LINE_AA)
    return tile


@dataclass
class GalleryTile:
    """One rendered recipe preview (RGB) and how long it took."""

    recipe: str
    image_rgb: np.ndarray
    seconds: float
    ok: bool = True


class RecipeGalleryRenderer:
    """Render recipe thumbnails for one photo, caching results between runs.

    Face detection runs once per photo: the first recipe's face contexts are
    reused for every later recipe, the same way the editor's preview cache
    reuses them when the user switches recipes.
    """

    def __init__(self, max_dim: int = GALLERY_MAX_DIM, cache_size: int = GALLERY_CACHE_SIZE):
        self.max_dim = int(max_dim)
        self.cache_size = int(cache_size)
        self._tiles: "OrderedDict[Tuple[str, str], np.ndarray]" = OrderedDict()
        self._contexts: Dict[str, list] = {}
        self._lock = threading.Lock()

    @staticmethod
    def source_key(src_bgr: np.ndarray) -> str:
        digest = hashlib.sha1(np.ascontiguousarray(src_bgr).tobytes())
        digest.update(repr(src_bgr.shape).encode())
        return digest.hexdigest()

    def cached(self, key: str, recipe: str) -> Optional[np.ndarray]:
        with self._lock:
            tile = self._tiles.get((key, recipe))
            if tile is not None:
                self._tiles.move_to_end((key, recipe))
            return tile

    def _store(self, key: str, recipe: str, tile: np.ndarray) -> None:
        with self._lock:
            self._tiles[(key, recipe)] = tile
            self._tiles.move_to_end((key, recipe))
            while len(self._tiles) > self.cache_size:
                self._tiles.popitem(last=False)

    def render(
        self,
        img_bgr: np.ndarray,
        recipes: Sequence[str],
        engine,
        progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[np.ndarray, List[GalleryTile]]:
        """Render ``recipes`` on a preview of ``img_bgr``.

        Returns ``(original_rgb, tiles)`` where ``original_rgb`` is the
        untouched preview. ``progress(done, total, recipe)`` is called before
        each recipe. A recipe that raises gets a grey "preview failed" tile
        and is logged, so one bad recipe never hides the rest.
        """
        src = prepare_gallery_source(img_bgr, self.max_dim)
        key = self.source_key(src)
        tiles: List[GalleryTile] = []
        total = len(recipes)
        for idx, name in enumerate(recipes):
            if progress is not None:
                progress(idx, total, name)
            hit = self.cached(key, name)
            if hit is not None:
                tiles.append(GalleryTile(name, hit, 0.0))
                continue
            start = time.perf_counter()
            kwargs = {"recipe": name}
            contexts = self._contexts.get(key)
            if contexts:
                kwargs["face_contexts"] = list(contexts)
            try:
                result = engine.process(src.copy(), **kwargs)
            except Exception as exc:  # one broken recipe must not stop the gallery
                _logger.warning("Recipe gallery: %s failed: %s", name, exc, exc_info=True)
                tiles.append(GalleryTile(name, _failed_tile(src.shape), time.perf_counter() - start, ok=False))
                continue
            if key not in self._contexts:
                self._contexts[key] = list(getattr(result, "face_contexts", None) or [])
            rgb = _to_rgb_uint8(result)
            if rgb.shape[:2] != src.shape[:2]:
                rgb = cv2.resize(rgb, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_AREA)
            self._store(key, name, rgb)
            tiles.append(GalleryTile(name, rgb, time.perf_counter() - start))
        if progress is not None:
            progress(total, total, "")
        return _to_rgb_uint8(src), tiles


def before_after(before_rgb: np.ndarray, after_rgb: np.ndarray, separator: int = 4) -> np.ndarray:
    """Return ``before | separator | after`` as one RGB image."""
    if after_rgb.shape[:2] != before_rgb.shape[:2]:
        after_rgb = cv2.resize(after_rgb, (before_rgb.shape[1], before_rgb.shape[0]),
                               interpolation=cv2.INTER_AREA)
    sep = np.full((before_rgb.shape[0], separator, 3), 200, dtype=np.uint8)
    return np.concatenate([before_rgb, sep, after_rgb], axis=1)


def contact_sheet(
    tiles: Sequence[Tuple[np.ndarray, str]],
    columns: int = 6,
    tile_height: int = 320,
    gap: int = 6,
) -> np.ndarray:
    """Lay out ``(rgb, caption)`` tiles in a grid with captions under each."""
    if not tiles:
        raise ValueError("contact_sheet needs at least one tile")
    columns = max(1, min(int(columns), len(tiles)))
    caption_h = 26
    scaled = []
    for rgb, caption in tiles:
        h, w = rgb.shape[:2]
        tw = max(1, round(w * tile_height / h))
        scaled.append((cv2.resize(rgb, (tw, tile_height), interpolation=cv2.INTER_AREA), caption))
    cell_w = max(img.shape[1] for img, _ in scaled)
    rows = (len(scaled) + columns - 1) // columns
    sheet_h = rows * (tile_height + caption_h + gap) + gap
    sheet_w = columns * (cell_w + gap) + gap
    sheet = np.full((sheet_h, sheet_w, 3), 24, dtype=np.uint8)
    for i, (img, caption) in enumerate(scaled):
        r, c = divmod(i, columns)
        y = gap + r * (tile_height + caption_h + gap)
        x = gap + c * (cell_w + gap) + (cell_w - img.shape[1]) // 2
        sheet[y:y + tile_height, x:x + img.shape[1]] = img
        font_scale = 0.45
        text = caption
        while cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0] > cell_w and len(text) > 4:
            text = text[:-2]
        cv2.putText(sheet, text, (gap + c * (cell_w + gap) + 2, y + tile_height + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (235, 235, 235), 1, cv2.LINE_AA)
    return sheet


def _group_from_cli(value: str) -> str:
    lookup = {g.lower(): g for g in gallery_groups()}
    lookup.update({"all": GROUP_ALL, "recommended": GROUP_RECOMMENDED,
                   "scene": GROUP_SCENE, "creative-scene": GROUP_SCENE})
    try:
        return lookup[value.strip().lower()]
    except KeyError:
        raise argparse.ArgumentTypeError(
            f"unknown group {value!r}; choose from: recommended, scene, all, "
            + ", ".join(list_categories())
        ) from None


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Render a recipe contact sheet for one photo (``python3 -m retouch.recipe_gallery``)."""
    parser = argparse.ArgumentParser(
        prog="python3 -m retouch.recipe_gallery",
        description="Preview recipes on one photo and save them as a contact sheet.",
    )
    parser.add_argument("photo", help="photo to preview recipes on")
    parser.add_argument("-o", "--output", default="recipe_gallery.jpg",
                        help="contact sheet path (default: recipe_gallery.jpg)")
    parser.add_argument("--group", type=_group_from_cli, default=GROUP_RECOMMENDED,
                        help="recommended (default), scene, all, or a category: "
                             + ", ".join(list_categories()))
    parser.add_argument("--recipes", default="",
                        help="comma-separated recipe names; overrides --group")
    parser.add_argument("--size", type=int, default=GALLERY_MAX_DIM,
                        help=f"preview long edge in pixels (default {GALLERY_MAX_DIM})")
    parser.add_argument("--columns", type=int, default=6, help="tiles per row (default 6)")
    args = parser.parse_args(argv)

    from .engine import RetouchEngine
    from .io import imread_exif

    try:
        img = imread_exif(args.photo)
    except Exception as exc:
        print(f"Could not read {args.photo}: {exc}", file=sys.stderr)
        return 1
    names = [n.strip() for n in args.recipes.split(",") if n.strip()] or gallery_recipe_names(args.group)

    def report(done: int, total: int, name: str) -> None:
        if name:
            print(f"  [{done + 1}/{total}] {name}", flush=True)

    renderer = RecipeGalleryRenderer(max_dim=args.size)
    engine = RetouchEngine()
    try:
        start = time.perf_counter()
        original, tiles = renderer.render(img, names, engine, progress=report)
    finally:
        close = getattr(engine, "close", None)
        if callable(close):
            close()
    sheet = contact_sheet([(original, "original")] + [(t.image_rgb, t.recipe) for t in tiles],
                          columns=args.columns)
    if not cv2.imwrite(args.output, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)):
        print(f"Could not write {args.output}", file=sys.stderr)
        return 1
    failed = [t.recipe for t in tiles if not t.ok]
    print(f"Saved {len(tiles)} recipe previews to {args.output} in {time.perf_counter() - start:.1f}s")
    if failed:
        print("Failed: " + ", ".join(failed), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

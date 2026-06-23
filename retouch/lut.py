"""3D colour lookup tables — Adobe Cube loader and trilinear interpolation.

Provides :class:`CubeLUT` (a BGR float32 3D LUT) and pure-function
:func:`trilinear_sample` for vectorized 3D LUT application. Includes an
Adobe ``.cube`` parser and a canonical ``luts/`` directory resolver that
sits next to :mod:`retouch.presets`.

The :class:`LUTRegistry` provides a hot-loadable cache: it invalidates
entries on file-mtime change so adding or editing a ``.cube`` file in
:func:`luts_dir` is reflected on the next :meth:`LUTRegistry.get` call.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np


__all__ = [
    "CubeLUT",
    "load_cube",
    "load_3dl",
    "trilinear_sample",
    "luts_dir",
    "list_available_luts",
    "LUTRegistry",
    "get_registry",
    "watch_luts_dir",
]


ArrayLike = Union[int, np.integer, np.ndarray]


def _build_identity_lut(size: int) -> np.ndarray:
    if size < 2:
        raise ValueError(f"LUT size must be >= 2; got {size}")
    lut = np.empty((size, size, size, 3), dtype=np.float32)
    axis = np.arange(size, dtype=np.float32) / float(size - 1)
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
    lut[..., 0] = b
    lut[..., 1] = g
    lut[..., 2] = r
    return lut


class CubeLUT:
    """3D colour lookup table stored as ``(N, N, N, 3)`` BGR float32 in [0, 1]."""

    def __init__(self, source: ArrayLike) -> None:
        if isinstance(source, np.ndarray):
            arr = source
            if arr.ndim != 4 or arr.shape[3] != 3:
                raise ValueError(
                    f"array must be shape (N, N, N, 3); got {arr.shape}"
                )
            if arr.shape[0] != arr.shape[1] or arr.shape[1] != arr.shape[2]:
                raise ValueError(f"array must be cubic; got {arr.shape}")
            if arr.shape[0] < 2:
                raise ValueError(f"LUT size must be >= 2; got {arr.shape[0]}")
            self._array = arr.astype(np.float32, copy=True)
        elif isinstance(source, (int, np.integer)):
            self._array = _build_identity_lut(int(source))
        else:
            raise TypeError(
                f"source must be int or np.ndarray; got {type(source).__name__}"
            )

    @property
    def size(self) -> int:
        return int(self._array.shape[0])

    @property
    def array(self) -> np.ndarray:
        return self._array

    def apply(self, img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError(
                f"img_bgr must be (H, W, 3); got {img_bgr.shape}"
            )
        if img_bgr.dtype == np.uint8:
            f = img_bgr.astype(np.float32) * (1.0 / 255.0)
        else:
            f = img_bgr.astype(np.float32)
            if f.size and float(f.max()) > 1.0:
                f = f * (1.0 / 255.0)
        out = trilinear_sample(self._array, f)
        return np.clip(out * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)


def trilinear_sample(lut: np.ndarray, bgr: np.ndarray) -> np.ndarray:
    """Vectorized trilinear lookup of a 3D LUT.

    Args:
        lut: ``(N, N, N, 3)`` BGR float32 array.
        bgr: ``(H, W, 3)`` BGR float32 array with values in [0, 1].

    Returns:
        ``(H, W, 3)`` BGR float32 array with values in [0, 1].
    """
    if lut.ndim != 4 or lut.shape[3] != 3:
        raise ValueError(f"lut must be (N, N, N, 3); got {lut.shape}")
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"bgr must be (H, W, 3); got {bgr.shape}")
    if lut.shape[0] != lut.shape[1] or lut.shape[1] != lut.shape[2]:
        raise ValueError(f"lut must be cubic; got {lut.shape}")
    n = int(lut.shape[0])
    if n < 2:
        raise ValueError(f"lut size must be >= 2; got {n}")
    lut_f = lut if lut.dtype == np.float32 else lut.astype(np.float32)
    bgr_f = bgr if bgr.dtype == np.float32 else bgr.astype(np.float32)
    max_idx = float(n - 1)
    coords = np.clip(bgr_f, 0.0, 1.0) * max_idx
    b0 = np.floor(coords[..., 0]).astype(np.int64)
    g0 = np.floor(coords[..., 1]).astype(np.int64)
    r0 = np.floor(coords[..., 2]).astype(np.int64)
    b1 = np.minimum(b0 + 1, n - 1)
    g1 = np.minimum(g0 + 1, n - 1)
    r1 = np.minimum(r0 + 1, n - 1)
    fb = (coords[..., 0] - b0)[..., np.newaxis]
    fg = (coords[..., 1] - g0)[..., np.newaxis]
    fr = (coords[..., 2] - r0)[..., np.newaxis]
    ob = 1.0 - fb
    og = 1.0 - fg
    orr = 1.0 - fr
    c000 = lut_f[b0, g0, r0]
    c001 = lut_f[b0, g0, r1]
    c010 = lut_f[b0, g1, r0]
    c011 = lut_f[b0, g1, r1]
    c100 = lut_f[b1, g0, r0]
    c101 = lut_f[b1, g0, r1]
    c110 = lut_f[b1, g1, r0]
    c111 = lut_f[b1, g1, r1]
    out = (
        ob * og * orr * c000
        + ob * og * fr * c001
        + ob * fg * orr * c010
        + ob * fg * fr * c011
        + fb * og * orr * c100
        + fb * og * fr * c101
        + fb * fg * orr * c110
        + fb * fg * fr * c111
    )
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def load_cube(path: Union[str, Path]) -> CubeLUT:
    """Parse an Adobe ``.cube`` file into a :class:`CubeLUT`."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f".cube file not found: {p}")
    size: Optional[int] = None
    triplets: List[float] = []
    with open(p, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            head = line[:32].upper()
            if head.startswith("TITLE"):
                continue
            if head.startswith("DOMAIN_MIN") or head.startswith("DOMAIN_MAX"):
                continue
            if head.startswith("LUT_1D_SIZE"):
                raise ValueError(
                    f"1D LUT not supported in {p.name}; expected LUT_3D_SIZE"
                )
            if head.startswith("LUT_3D_SIZE"):
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Malformed LUT_3D_SIZE line: {line!r}")
                try:
                    parsed = int(parts[1])
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid LUT_3D_SIZE value: {parts[1]!r}"
                    ) from exc
                if parsed < 2:
                    raise ValueError(f"LUT_3D_SIZE must be >= 2; got {parsed}")
                size = parsed
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"Malformed data line in {p.name}: {line!r}"
                )
            try:
                r_v, g_v, b_v = (float(x) for x in parts)
            except ValueError as exc:
                raise ValueError(
                    f"Malformed triplet in {p.name}: {line!r}"
                ) from exc
            triplets.extend((r_v, g_v, b_v))
    if size is None:
        raise ValueError(f"Missing LUT_3D_SIZE declaration in {p.name}")
    expected = size * size * size * 3
    actual = len(triplets)
    if actual != expected:
        raise ValueError(
            f"Expected {expected} floats ({size}^3 * 3) in {p.name}; got {actual}"
        )
    rgb = np.array(triplets, dtype=np.float32).reshape(size, size, size, 3)
    bgr = rgb[..., ::-1].copy()
    return CubeLUT(bgr)


def load_3dl(path: Union[str, Path]) -> CubeLUT:
    """Parse an Iridas ``.3dl`` file. Not implemented in this spike."""
    raise NotImplementedError("3dl loader not yet implemented; use load_cube")


def luts_dir() -> Path:
    """Return the canonical ``luts/`` directory, creating it if missing."""
    p = Path(__file__).resolve().parent.parent / "luts"
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_available_luts() -> List[str]:
    """Return the sorted stems of every ``.cube`` in :func:`luts_dir`."""
    return sorted(p.stem for p in luts_dir().glob("*.cube"))


_REGISTERED_MTIME: float = float("inf")


class LUTRegistry:
    """Hot-loadable registry of 3D LUTs.

    Caches loaded LUTs by stem, with file-mtime invalidation. Call
    :meth:`reload` to force a re-scan, or rely on automatic invalidation
    when a file's mtime changes since last load. Pre-loaded LUTs added
    via :meth:`register` are stored with a sentinel mtime and are not
    subject to disk invalidation until :meth:`reload` is called.

    Thread safety: This class is NOT thread-safe. External locking is
    required if it is accessed concurrently from multiple threads. The
    companion helper :func:`watch_luts_dir` runs in a background thread
    and invokes a user callback for each detected change.
    """

    def __init__(self, luts_dir_path: Optional[Path] = None) -> None:
        self._dir: Path = (
            Path(luts_dir_path) if luts_dir_path is not None else luts_dir()
        )
        self._cache: Dict[str, Tuple[CubeLUT, float]] = {}
        self._known_mtimes: Optional[Dict[str, float]] = None

    @property
    def luts_dir(self) -> Path:
        return self._dir

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def _resolve_stem(self, name: str) -> str:
        p = Path(name)
        if p.suffix.lower() == ".cube":
            return p.stem
        if "/" in name or p.is_absolute():
            return p.stem
        return name

    def _disk_path(self, stem: str) -> Path:
        return self._dir / f"{stem}.cube"

    @staticmethod
    def _read_mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except FileNotFoundError:
            return -1.0
        except OSError:
            return -1.0

    def get(self, name: str) -> CubeLUT:
        """Get a LUT by stem or relative path. Auto-reloads if mtime changed.

        ``name`` may be a bare stem (e.g. ``"kodak"``), a stem with the
        ``.cube`` suffix (e.g. ``"kodak.cube"``), or a path with directory
        separators — the path is resolved relative to this registry's
        :attr:`luts_dir`.
        """
        stem = self._resolve_stem(name)
        disk_path = self._disk_path(stem)
        cached = self._cache.get(stem)
        if cached is not None:
            lut, cached_mtime = cached
            if cached_mtime == _REGISTERED_MTIME:
                return lut
            if disk_path.exists() and self._read_mtime(disk_path) == cached_mtime:
                return lut
        if not disk_path.exists():
            if cached is not None:
                return cached[0]
            raise FileNotFoundError(
                f"LUT {stem!r} not found in {self._dir}"
            )
        current_mtime = self._read_mtime(disk_path)
        lut = load_cube(disk_path)
        self._cache[stem] = (lut, current_mtime)
        return lut

    def list_available(self) -> List[str]:
        """Return the sorted list of available LUT stems. Re-scans dir."""
        if not self._dir.exists():
            return []
        return sorted(p.stem for p in self._dir.glob("*.cube"))

    def reload(self) -> None:
        """Clear the cache and reset the change-tracking baseline."""
        self._cache.clear()
        self._known_mtimes = None

    def register(self, lut: CubeLUT, name: Optional[str] = None) -> str:
        """Register a pre-loaded LUT in the registry. Returns the stem.

        Registered entries are stored with a sentinel mtime and are
        returned by :meth:`get` without consulting the disk. They are
        cleared on :meth:`reload`.
        """
        if name is None:
            name = f"lut_{len(self._cache)}"
        stem = self._resolve_stem(name)
        self._cache[stem] = (lut, _REGISTERED_MTIME)
        return stem

    def poll_changes(self) -> List[str]:
        """Return a sorted list of changed stems (added, removed, or modified).

        The first call establishes the baseline and returns an empty list.
        Subsequent calls return stems whose presence or mtime has changed
        since the previous call, and evict them from the cache.
        """
        current: Dict[str, float] = {}
        if self._dir.exists():
            for p in self._dir.glob("*.cube"):
                mtime = self._read_mtime(p)
                if mtime >= 0.0:
                    current[p.stem] = mtime
        if self._known_mtimes is None:
            self._known_mtimes = current
            return []
        prev = self._known_mtimes
        added = set(current) - set(prev)
        removed = set(prev) - set(current)
        modified = {
            stem for stem in current
            if stem in prev and current[stem] != prev[stem]
        }
        changed = sorted(added | removed | modified)
        for stem in changed:
            self._cache.pop(stem, None)
        self._known_mtimes = current
        return changed


_DEFAULT_REGISTRY = LUTRegistry()


def get_registry() -> LUTRegistry:
    """Return the default :class:`LUTRegistry` (singleton)."""
    return _DEFAULT_REGISTRY


def watch_luts_dir(
    callback: Callable[[str], None],
    interval: float = 1.0,
) -> threading.Thread:
    """Watch the default luts dir and call ``callback(stem)`` on changes.

    Polls the directory every ``interval`` seconds (default 1.0) and
    invokes ``callback`` once per changed stem (added, removed, or
    mtime-modified). The watcher runs in a daemon thread that exits
    when the main program exits; the returned thread handle is provided
    so callers can ``join()`` or stop it if needed.

    The callback is invoked from a background thread and must be
    thread-safe with respect to any other code accessing the registry.
    """
    if interval <= 0.0:
        raise ValueError(f"interval must be > 0; got {interval}")
    registry = get_registry()

    def _loop() -> None:
        while True:
            try:
                for stem in registry.poll_changes():
                    callback(stem)
            except (OSError, ValueError) as exc:
                print(f"lut watcher error: {exc}")
            time.sleep(interval)

    t = threading.Thread(target=_loop, daemon=True, name="lut-watcher")
    t.start()
    return t

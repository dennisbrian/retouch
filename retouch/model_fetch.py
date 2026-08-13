"""Manifest-driven model resolution and verified user-cache downloads.

Models may be bundled with a source checkout or wheel, but downloaded assets
are always written to the user cache so frozen/read-only installations remain
usable. Downloadable entries must declare a SHA-256 digest and byte size;
missing or mismatched integrity metadata fails closed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import urllib.error
import urllib.request
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

from .utils import get_cache_dir, offline_mode_enabled

logger = logging.getLogger(__name__)

_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
_MANIFEST_PATH = os.path.join(_MODELS_DIR, "manifest.json")
_PLACEHOLDER_URL_HOST = "github.com/owner/retouch-models"
_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_manifest_lock = threading.Lock()
_download_locks: Dict[str, threading.Lock] = {}
_download_locks_guard = threading.Lock()
_CHUNK_SIZE = 1 << 16


class ModelFetchError(Exception):
    """Base error for manifest, integrity, and download failures."""


class ModelNotFoundError(ModelFetchError):
    pass


class ModelNameError(ModelFetchError):
    pass


class ChecksumMismatchError(ModelFetchError):
    pass


class ModelSizeMismatchError(ModelFetchError):
    pass


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ModelNameError(
            f"Invalid model name {name!r}: must match {_NAME_RE.pattern} "
            "(path-traversal guard)"
        )


def _get_download_lock(name: str) -> threading.Lock:
    with _download_locks_guard:
        lock = _download_locks.get(name)
        if lock is None:
            lock = threading.Lock()
            _download_locks[name] = lock
        return lock


def _resource_manifest_text() -> Optional[str]:
    try:
        return resources.files("models").joinpath("manifest.json").read_text(encoding="utf-8")
    except (AttributeError, FileNotFoundError, ModuleNotFoundError, OSError):
        return None


def load_manifest() -> Dict[str, Any]:
    """Load the manifest from the checkout, then from installed package data."""
    with _manifest_lock:
        try:
            with open(_MANIFEST_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            text = _resource_manifest_text()
            if text is None:
                raise ModelFetchError(f"Manifest not found at {_MANIFEST_PATH}")
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                raise ModelFetchError(f"Manifest JSON invalid: {e}") from e
        except json.JSONDecodeError as e:
            raise ModelFetchError(f"Manifest JSON invalid: {e}") from e
        except OSError as e:
            raise ModelFetchError(f"Could not read manifest: {e}") from e


def _manifest_entry(name: str) -> Dict[str, Any]:
    _validate_name(name)
    models = load_manifest().get("models", {})
    entry = models.get(name)
    if entry is None:
        raise ModelNotFoundError(
            f"Model {name!r} not in manifest. Known: {sorted(models.keys())}"
        )
    return entry


def _filename(entry: Dict[str, Any], name: str) -> str:
    filename = entry.get("filename", "")
    if not filename or "/" in filename or "\\" in filename or filename != os.path.basename(filename):
        raise ModelNameError(
            f"Manifest filename for {name!r} is not a bare basename: {filename!r}"
        )
    return filename


def _bundled_path(name: str, entry: Optional[Dict[str, Any]] = None) -> Path:
    entry = entry or _manifest_entry(name)
    return Path(_MODELS_DIR) / _filename(entry, name)


def _cache_models_dir() -> Path:
    return get_cache_dir() / "models"


def _cache_path(name: str, entry: Optional[Dict[str, Any]] = None) -> Path:
    entry = entry or _manifest_entry(name)
    return _cache_models_dir() / _filename(entry, name)


def _candidate_paths(name: str, entry: Optional[Dict[str, Any]] = None) -> Iterable[Path]:
    entry = entry or _manifest_entry(name)
    yield _bundled_path(name, entry)
    cache_path = _cache_path(name, entry)
    if cache_path != _bundled_path(name, entry):
        yield cache_path


def _local_path(name: str) -> str:
    """Compatibility helper returning the bundled/source path."""
    return str(_bundled_path(name))


def _integrity_metadata(name: str, entry: Dict[str, Any]) -> tuple[str, int]:
    expected_sha = str(entry.get("sha256", "") or "").strip()
    expected_size = int(entry.get("size_bytes", 0) or 0)
    if not _SHA256_RE.fullmatch(expected_sha):
        raise ModelFetchError(
            f"Manifest entry for {name!r} must declare a 64-character sha256"
        )
    if expected_size <= 0:
        raise ModelFetchError(
            f"Manifest entry for {name!r} must declare a positive size_bytes"
        )
    return expected_sha.lower(), expected_size


def _verify_path(name: str, path: Path, entry: Dict[str, Any]) -> bool:
    expected_sha, expected_size = _integrity_metadata(name, entry)
    if not path.is_file():
        return False
    try:
        actual_size = path.stat().st_size
    except OSError as e:
        raise ModelFetchError(f"Could not stat {path}: {e}") from e
    if actual_size != expected_size:
        raise ModelSizeMismatchError(
            f"Size mismatch for {name!r}: expected {expected_size}, got {actual_size}"
        )
    try:
        actual_sha = _sha256_of_file(str(path))
    except OSError as e:
        raise ModelFetchError(f"Could not hash {path}: {e}") from e
    if actual_sha.lower() != expected_sha:
        raise ChecksumMismatchError(
            f"Checksum mismatch for {name!r}: expected {expected_sha}, got {actual_sha}"
        )
    return True


def model_exists(name: str) -> bool:
    """Return true only for a local model that passes all integrity checks."""
    try:
        entry = _manifest_entry(name)
        for path in _candidate_paths(name, entry):
            if path.is_file():
                try:
                    if _verify_path(name, path, entry):
                        return True
                except ModelFetchError as e:
                    logger.warning("Invalid local model %r at %s: %s", name, path, e)
        return False
    except ModelFetchError:
        return False


def model_status(name: str) -> Dict[str, Any]:
    """Return side-effect-free capability and integrity status."""
    entry = dict(_manifest_entry(name))
    status: Dict[str, Any] = dict(entry)
    status["available"] = False
    status["path"] = None
    status["integrity_error"] = None
    integrity_valid = True
    try:
        _integrity_metadata(name, entry)
    except ModelFetchError as e:
        integrity_valid = False
        status["integrity_error"] = str(e)
    for path in _candidate_paths(name, entry):
        if not path.is_file():
            continue
        try:
            if _verify_path(name, path, entry):
                status["available"] = True
                status["path"] = str(path)
                break
        except ModelFetchError as e:
            status["integrity_error"] = str(e)
    url = str(entry.get("url", "") or "")
    status["downloadable"] = bool(url) and integrity_valid and _PLACEHOLDER_URL_HOST not in url
    return status


def verify_model(name: str) -> bool:
    """Verify a bundled or cached model; reject incomplete metadata."""
    entry = _manifest_entry(name)
    first_error: Optional[ModelFetchError] = None
    for path in _candidate_paths(name, entry):
        if path.is_file():
            try:
                if _verify_path(name, path, entry):
                    return True
            except ModelFetchError as exc:
                first_error = first_error or exc
    if first_error is not None:
        raise first_error
    return False


def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _cleanup(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("Could not remove incomplete model file %s", path)


def download_model(
    name: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> str:
    """Download and atomically install a verified model in the user cache."""
    entry = _manifest_entry(name)
    url = str(entry.get("url", "") or "")
    expected_sha, expected_size = _integrity_metadata(name, entry)
    if not url:
        raise ModelFetchError(f"Manifest entry for {name!r} has no url")
    if _PLACEHOLDER_URL_HOST in url:
        raise ModelFetchError(f"Manifest entry for {name!r} points at a placeholder URL")
    if offline_mode_enabled():
        raise ModelFetchError(
            f"Offline mode is enabled; model {name!r} is not available locally"
        )

    dest = _cache_path(name, entry)
    dest.parent.mkdir(parents=True, exist_ok=True)

    for existing in (dest, _bundled_path(name, entry)):
        if existing.is_file():
            try:
                if _verify_path(name, existing, entry):
                    return str(existing)
            except ModelFetchError as e:
                logger.warning("%s: %s", existing, e)

    lock = _get_download_lock(name)
    with lock:
        if dest.is_file():
            try:
                if _verify_path(name, dest, entry):
                    return str(dest)
            except ModelFetchError as e:
                logger.warning("Replacing invalid cached model %s: %s", dest, e)
        tmp = dest.with_name(dest.name + ".part")
        _cleanup(tmp)
        try:
            _stream_download(url, str(tmp), expected_size, progress_callback)
            _verify_path(name, tmp, entry)
            os.replace(str(tmp), str(dest))
        except ModelFetchError:
            _cleanup(tmp)
            raise
        except (urllib.error.URLError, OSError) as e:
            _cleanup(tmp)
            raise ModelFetchError(f"Download failed for {name!r} from {url}: {e}") from e

    logger.info("download_model(%r): stored verified model at %s", name, dest)
    return str(dest)


def _stream_download(
    url: str,
    dest: str,
    expected_size: int,
    progress_callback: Optional[Callable[[int, int], None]],
) -> None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "retouch-model-fetch/1"})
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        raise ModelFetchError(f"HTTP {e.code} fetching {url}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise ModelFetchError(f"URL error fetching {url}: {e.reason}") from e

    content_length = resp.headers.get("Content-Length")
    total = int(content_length) if content_length and content_length.isdigit() else expected_size
    received = 0
    with open(dest, "wb") as fh:
        while True:
            try:
                chunk = resp.read(_CHUNK_SIZE)
            except OSError as e:
                raise ModelFetchError(f"Read error during download from {url}: {e}") from e
            if not chunk:
                break
            fh.write(chunk)
            received += len(chunk)
            if progress_callback is not None:
                try:
                    progress_callback(received, total)
                except Exception as e:  # noqa: BLE001
                    logger.warning("progress_callback raised %s: %s", type(e).__name__, e)

    if received != expected_size:
        raise ModelSizeMismatchError(
            f"Downloaded size mismatch for {url}: expected {expected_size}, got {received}"
        )


def get_model_path(name: str) -> str:
    """Return a verified bundled/cache path, downloading only when required."""
    _validate_name(name)
    entry = _manifest_entry(name)
    first_error: Optional[ModelFetchError] = None
    for path in _candidate_paths(name, entry):
        if path.is_file():
            try:
                if _verify_path(name, path, entry):
                    return str(path)
            except ModelFetchError as exc:
                first_error = first_error or exc
    if first_error is not None and not str(entry.get("url", "") or ""):
        raise first_error
    return download_model(name)


def list_models() -> Dict[str, Dict[str, Any]]:
    return dict(load_manifest().get("models", {}))

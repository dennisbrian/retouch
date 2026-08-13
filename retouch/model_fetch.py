"""Model-fetch infrastructure — manifest-driven download-on-first-use.

Reads `models/manifest.json`, resolves local model paths, downloads missing
models with checksum verification. Thread-safe for Gradio use.

Public API:
    load_manifest() -> Dict[str, Any]
    get_model_path(name) -> str
    download_model(name, progress_callback=None) -> str
    verify_model(name) -> bool
    model_exists(name) -> bool
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
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
_MANIFEST_PATH = os.path.join(_MODELS_DIR, "manifest.json")

# Template host left over from the repo skeleton; no models are published
# there. Manifest entries still carrying it cannot be downloaded.
_PLACEHOLDER_URL_HOST = "github.com/owner/retouch-models"

_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")

_manifest_lock = threading.Lock()
_download_locks: Dict[str, threading.Lock] = {}
_download_locks_guard = threading.Lock()

_CHUNK_SIZE = 1 << 16


class ModelFetchError(Exception):
    pass


class ModelNotFoundError(ModelFetchError):
    pass


class ModelNameError(ModelFetchError):
    pass


class ChecksumMismatchError(ModelFetchError):
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


def load_manifest() -> Dict[str, Any]:
    with _manifest_lock:
        try:
            with open(_MANIFEST_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError as e:
            raise ModelFetchError(f"Manifest not found at {_MANIFEST_PATH}") from e
        except json.JSONDecodeError as e:
            raise ModelFetchError(f"Manifest JSON invalid: {e}") from e
        except OSError as e:
            raise ModelFetchError(f"Could not read manifest: {e}") from e


def _manifest_entry(name: str) -> Dict[str, Any]:
    _validate_name(name)
    manifest = load_manifest()
    models = manifest.get("models", {})
    entry = models.get(name)
    if entry is None:
        raise ModelNotFoundError(
            f"Model {name!r} not in manifest. Known: {sorted(models.keys())}"
        )
    return entry


def _local_path(name: str) -> str:
    entry = _manifest_entry(name)
    filename = entry.get("filename", "")
    if not filename or "/" in filename or "\\" in filename or filename != os.path.basename(filename):
        raise ModelNameError(
            f"Manifest filename for {name!r} is not a bare basename: {filename!r}"
        )
    return os.path.join(_MODELS_DIR, filename)


def model_exists(name: str) -> bool:
    try:
        path = _local_path(name)
    except ModelFetchError:
        return False
    return os.path.isfile(path) and os.path.getsize(path) > 0


def model_status(name: str) -> Dict[str, Any]:
    """Return a truthful, side-effect-free status for a manifest entry.

    ``available`` means a usable local file is present; it does not imply that
    an optional model is downloadable.  This distinction keeps UI and
    diagnostics honest when a feature intentionally falls back.
    """
    entry = dict(_manifest_entry(name))
    entry["available"] = model_exists(name)
    entry["downloadable"] = bool(entry.get("url")) and _PLACEHOLDER_URL_HOST not in str(entry.get("url", ""))
    return entry


def verify_model(name: str) -> bool:
    entry = _manifest_entry(name)
    expected = entry.get("sha256", "")
    if not expected:
        logger.warning(
            "verify_model(%r): manifest has empty sha256 (placeholder entry); "
            "skipping checksum verification.",
            name,
        )
        return True
    path = _local_path(name)
    if not os.path.isfile(path):
        return False
    try:
        actual = _sha256_of_file(path)
    except OSError as e:
        raise ModelFetchError(f"Could not hash {path}: {e}") from e
    if actual.lower() != expected.lower():
        raise ChecksumMismatchError(
            f"Checksum mismatch for {name!r}: expected {expected}, got {actual}"
        )
    return True


def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def download_model(
    name: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> str:
    entry = _manifest_entry(name)
    url = entry.get("url", "")
    if not url:
        raise ModelFetchError(f"Manifest entry for {name!r} has no url")
    # A legacy manifest must never be allowed to turn a placeholder host into
    # a confusing network failure. Current optional entries use an empty URL
    # and are explicitly unavailable until a verified release is supplied.
    if _PLACEHOLDER_URL_HOST in url:
        raise ModelFetchError(
            f"Manifest entry for {name!r} still points at the placeholder URL "
            f"({url}). No release host is configured, so this model cannot be "
            f"fetched automatically — install it manually into models/ or "
            f"update models/manifest.json with a real url, sha256 and size_bytes."
        )

    dest = _local_path(name)
    expected_size = int(entry.get("size_bytes", 0) or 0)
    expected_sha = entry.get("sha256", "")

    if os.path.isfile(dest) and expected_sha:
        try:
            if verify_model(name):
                logger.info("download_model(%r): already present and verified.", name)
                return dest
            logger.warning(
                "download_model(%r): present but checksum failed; re-downloading.", name
            )
        except ChecksumMismatchError as e:
            logger.warning("download_model(%r): %s; re-downloading.", name, e)

    lock = _get_download_lock(name)
    with lock:
        if os.path.isfile(dest) and expected_sha and verify_model(name):
            return dest

        tmp = dest + ".part"
        try:
            _stream_download(url, tmp, expected_size, progress_callback)
        except (urllib.error.URLError, OSError) as e:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            raise ModelFetchError(f"Download failed for {name!r} from {url}: {e}") from e

        if expected_sha:
            try:
                actual = _sha256_of_file(tmp)
            except OSError as e:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                raise ModelFetchError(f"Could not hash downloaded {tmp}: {e}") from e
            if actual.lower() != expected_sha.lower():
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                raise ChecksumMismatchError(
                    f"Checksum mismatch for {name!r}: expected {expected_sha}, "
                    f"got {actual}"
                )

        try:
            os.replace(tmp, dest)
        except OSError as e:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            raise ModelFetchError(f"Could not move {tmp} -> {dest}: {e}") from e

    logger.info("download_model(%r): stored at %s", name, dest)
    return dest


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
                except Exception as e:
                    logger.warning("progress_callback raised %s: %s", type(e).__name__, e)

    if total and received != total and total > 0:
        logger.warning(
            "Downloaded %d bytes from %s but Content-Length/expected was %d",
            received, url, total,
        )


def get_model_path(name: str) -> str:
    _validate_name(name)
    path = _local_path(name)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    return download_model(name)


def list_models() -> Dict[str, Dict[str, Any]]:
    manifest = load_manifest()
    return dict(manifest.get("models", {}))

"""Non-blocking update check against GitHub releases.

Compares ``retouch.__version__`` with the latest release tag on the
project's GitHub repo. Never downloads anything; the caller decides how
to surface "update available" (GUI toast, CLI note).

Usage:
    from retouch.update_check import check_for_update
    info = check_for_update(timeout=3.0)   # None if offline / up to date
    if info is not None:
        print(info.latest_version, info.url)
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_RELEASES_API = "https://api.github.com/repos/dennisbrian/retouch/releases/latest"
_USER_AGENT = "retouch-update-check"


@dataclass(frozen=True)
class UpdateInfo:
    """Result of a successful check that found a newer release."""

    latest_version: str
    url: str


def _parse_version(tag: str) -> Optional[Tuple[int, ...]]:
    """Parse 'v2.1.0-codename' / '2.1.0' into a comparable int tuple.

    Returns None for tags with no numeric dotted prefix.
    """
    m = re.match(r"^v?(\d+(?:\.\d+)*)", tag.strip())
    if not m:
        return None
    return tuple(int(p) for p in m.group(1).split("."))


def check_for_update(timeout: float = 3.0) -> Optional[UpdateInfo]:
    """Return UpdateInfo if GitHub's latest release is newer, else None.

    All network/parse failures are logged and swallowed — an update check
    must never break app startup.
    """
    from . import __version__

    current = _parse_version(__version__)
    if current is None:
        logger.warning("update_check: unparseable local version %r", __version__)
        return None

    try:
        req = urllib.request.Request(
            _RELEASES_API,
            headers={"User-Agent": _USER_AGENT, "Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        tag = str(payload.get("tag_name", ""))
        url = str(payload.get("html_url", ""))
    except Exception as e:  # noqa: BLE001 — offline, rate-limited, DNS, etc.
        logger.info("update_check: check failed (%s); treating as up to date.", e)
        return None

    latest = _parse_version(tag)
    if latest is None:
        logger.info("update_check: unparseable release tag %r", tag)
        return None

    # Pad to equal length so (2,0) < (2,0,1) compares correctly.
    n = max(len(current), len(latest))
    cur_p = current + (0,) * (n - len(current))
    lat_p = latest + (0,) * (n - len(latest))
    if lat_p > cur_p:
        logger.info("update_check: update available: %s -> %s", __version__, tag)
        return UpdateInfo(latest_version=tag, url=url)
    return None

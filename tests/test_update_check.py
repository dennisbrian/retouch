"""Tests for retouch/update_check.py — all network access mocked."""

from __future__ import annotations

import io
import json
import urllib.error
from unittest import mock

from retouch.update_check import UpdateInfo, _parse_version, check_for_update


class TestParseVersion:
    def test_plain(self):
        assert _parse_version("2.0.0") == (2, 0, 0)

    def test_v_prefix_and_codename(self):
        assert _parse_version("v2.1.0-fuji") == (2, 1, 0)

    def test_short(self):
        assert _parse_version("2") == (2,)

    def test_garbage(self):
        assert _parse_version("not-a-version") is None

    def test_empty(self):
        assert _parse_version("") is None


def _mock_response(tag: str, url: str = "https://example.com/rel"):
    body = json.dumps({"tag_name": tag, "html_url": url}).encode()
    resp = mock.MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = mock.MagicMock(return_value=False)
    return resp


class TestCheckForUpdate:
    def test_newer_release_returned(self):
        with mock.patch("urllib.request.urlopen", return_value=_mock_response("v99.0.0-x")):
            info = check_for_update()
        assert info is not None
        assert info.latest_version == "v99.0.0-x"
        assert info.url == "https://example.com/rel"

    def test_same_release_returns_none(self):
        from retouch import __version__
        with mock.patch("urllib.request.urlopen", return_value=_mock_response(f"v{__version__}")):
            assert check_for_update() is None

    def test_older_release_returns_none(self):
        with mock.patch("urllib.request.urlopen", return_value=_mock_response("v0.0.1")):
            assert check_for_update() is None

    def test_network_failure_returns_none(self):
        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
            assert check_for_update() is None

    def test_malformed_json_returns_none(self):
        resp = mock.MagicMock()
        resp.read.return_value = b"not json{"
        resp.__enter__ = lambda s: s
        resp.__exit__ = mock.MagicMock(return_value=False)
        with mock.patch("urllib.request.urlopen", return_value=resp):
            assert check_for_update() is None

    def test_unparseable_tag_returns_none(self):
        with mock.patch("urllib.request.urlopen", return_value=_mock_response("latest")):
            assert check_for_update() is None

    def test_shorter_local_version_padded(self):
        # local 2.0.0 vs remote 2.0.0.1-style tag
        with mock.patch("urllib.request.urlopen", return_value=_mock_response("v2.0.0.1")):
            info = check_for_update()
        assert info is not None

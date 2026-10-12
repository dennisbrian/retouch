"""Tests for the dependency-light Retouch package import boundary."""

from __future__ import annotations

import subprocess
import sys


def test_package_import_does_not_eagerly_load_engine() -> None:
    code = (
        "import retouch; "
        "assert retouch.__version__ == '2.0.0'; "
        "assert 'RetouchEngine' not in vars(retouch)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

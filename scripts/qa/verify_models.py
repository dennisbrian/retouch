#!/usr/bin/env python3
"""Verify downloaded model files against the checked-in manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("models/manifest.json"))
    parser.add_argument("--root", type=Path, default=Path("models"))
    parser.add_argument("--model", action="append", required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    models = manifest.get("models", {})
    for name in args.model:
        if name not in models:
            raise SystemExit(f"model {name!r} is missing from {args.manifest}")
        entry = models[name]
        digest = str(entry.get("sha256", ""))
        expected_size = int(entry.get("size_bytes", 0) or 0)
        if len(digest) != 64 or expected_size <= 0:
            raise SystemExit(f"model {name!r} has incomplete integrity metadata")
        path = args.root / str(entry["filename"])
        if not path.is_file():
            raise SystemExit(f"model {name!r} is missing at {path}")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise SystemExit(
                f"model {name!r} size mismatch: expected {expected_size}, got {actual_size}"
            )
        actual_digest = sha256_file(path)
        if actual_digest.lower() != digest.lower():
            raise SystemExit(
                f"model {name!r} sha256 mismatch: expected {digest}, got {actual_digest}"
            )
        print(f"verified {name}: {actual_size} bytes sha256={actual_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

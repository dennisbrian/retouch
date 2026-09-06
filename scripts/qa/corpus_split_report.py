"""Validate a retouch.corpus_manifest v3 manifest and print its split report.

    python3 scripts/qa/corpus_split_report.py path/to/manifest.json

Prints per-split asset/subject counts and tag/stratum coverage, and the full
list of validation errors if the manifest is not valid. Exits nonzero on
validation failure so this can be used as a CI/pre-promotion gate -- e.g.
before assigning any subject to the "locked_test" split.

This never assigns anything to a split; it only reports on a manifest a
human already wrote and asserts its own structural invariants (split
vocabulary, mandatory person_id, no person_id crossing splits, etc). Do not
add auto-population logic here -- see retouch/corpus_manifest.py's module
docstring on why "locked_test" population is deliberately not automatable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from retouch.corpus_manifest import ALLOWED_SPLITS, corpus_split_report, validate_corpus_manifest


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <manifest.json>", file=sys.stderr)
        return 2

    manifest_path = Path(argv[1]).expanduser().resolve()
    if not manifest_path.is_file():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = validate_corpus_manifest(manifest, root=manifest_path.parent)

    print(f"manifest: {manifest_path}")
    print(f"schema_version: {report['schema_version']}")
    print(f"valid: {report['valid']}")
    print(f"asset_count: {report['asset_count']}")
    print()

    split_report = corpus_split_report(report)
    print("split_counts:")
    for split in ALLOWED_SPLITS:
        assets = split_report["split_counts"].get(split, 0)
        persons = split_report["person_counts_by_split"].get(split, 0)
        print(f"  {split:12s} assets={assets:4d}  distinct_subjects={persons:4d}")
    print()

    print("coverage_by_split:")
    for split in ALLOWED_SPLITS:
        coverage = split_report["coverage_by_split"].get(split, {})
        tags = coverage.get("tags", {})
        strata = coverage.get("strata", {})
        print(f"  {split}:")
        if not tags and not strata:
            print("    (no assets)")
            continue
        for tag, asset_ids in sorted(tags.items()):
            print(f"    tag {tag}: {len(asset_ids)} asset(s)")
        for dimension, values in sorted(strata.items()):
            for value, asset_ids in sorted(values.items()):
                print(f"    stratum {dimension}={value}: {len(asset_ids)} asset(s)")
    print()

    if not report["valid"]:
        print(f"errors ({len(report['errors'])}):", file=sys.stderr)
        for issue in report["errors"]:
            print(f"  {issue}", file=sys.stderr)
        return 1

    if report["warnings"]:
        print(f"warnings ({len(report['warnings'])}):")
        for issue in report["warnings"]:
            print(f"  {issue}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

"""Merge scattered per-asset corpus_manifest.json files into one reviewed
inventory and report where the vocabulary has no coverage at all.

    .venv/bin/python scripts/qa/corpus_inventory.py test_output/fa02_pilot_*/corpus_manifest.json

This is step 1 of docs/plans/PLAN_RETOUCH_GAP_RESEARCH_2026_09_15.md: "a
reviewed development inventory ... source hashes, reviewed subject/shoot
grouping, per-region labels, missing-strata table, disagreements; distinguish
available photos from usable evidence." It does not run step 2 or later, does
not assign anything to calibration/locked_test, and does not invent subject or
shoot identity -- it only aggregates what each input manifest already declares
under retouch.corpus_manifest v3, then validates the merge.

Each input manifest here declares one asset with its own consent_reference
(a different shoot). Consent is asset-level in the merge -- do not pick one
shoot's consent_reference to stand for the merged file. Paths are re-rooted to
be relative to --root (default: the common parent of the input files) so the
merged manifest is portable without switching to absolute paths.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from retouch.corpus_manifest import (
    ALLOWED_SPLITS,
    DEFAULT_STRATA_VOCABULARY,
    DEFAULT_TAG_VOCABULARY,
    corpus_split_report,
    validate_corpus_manifest,
)


def _load(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_manifests(manifest_paths: List[Path], *, merged_root: Path) -> Dict[str, Any]:
    """Combine single- or multi-asset v3 manifests into one, path-rewritten to merged_root.

    Each source asset keeps its own consent_reference (pushed from the source
    manifest's top-level field onto the asset if the asset did not already
    carry one) rather than collapsing distinct shoots' consent under one value.
    """
    assets: List[Dict[str, Any]] = []
    sources_seen: List[str] = []
    for manifest_path in manifest_paths:
        raw = _load(manifest_path)
        source_consent = raw.get("consent_reference")
        sources_seen.append(str(manifest_path))
        for asset in raw.get("assets", []):
            asset = dict(asset)
            asset_path = manifest_path.parent / str(asset.get("path", ""))
            asset["path"] = asset_path.resolve().relative_to(merged_root.resolve()).as_posix()
            if "consent_reference" not in asset and source_consent:
                asset["consent_reference"] = source_consent
            assets.append(asset)

    return {
        "schema_version": 3,
        "consent_reference": "see per-asset consent_reference -- this merge spans multiple shoots",
        "assets": assets,
    }, sources_seen


def missing_strata_table(coverage: Mapping[str, Any]) -> Dict[str, Any]:
    """For every vocabulary dimension/value, report zero-coverage entries.

    This is the table the plan asks for: not just what's present, but which
    entire dimensions (e.g. occlusion, texture, compression) have no asset at
    all, versus dimensions that are populated but missing specific values.
    """
    observed_strata = coverage.get("strata", {})
    unset_dimensions: List[str] = []
    missing_values: Dict[str, List[str]] = {}
    for dimension, allowed_values in sorted(DEFAULT_STRATA_VOCABULARY.items()):
        observed_values = observed_strata.get(dimension)
        if not observed_values:
            unset_dimensions.append(dimension)
            continue
        gaps = [value for value in allowed_values if value not in observed_values]
        if gaps:
            missing_values[dimension] = gaps

    observed_tags = coverage.get("tags", {})
    unused_tags = [tag for tag in DEFAULT_TAG_VOCABULARY if tag not in observed_tags]

    return {
        "dimensions_with_zero_coverage": unset_dimensions,
        "dimensions_missing_some_values": missing_values,
        "tags_never_used": unused_tags,
    }


def candidate_groupings(assets: List[Mapping[str, Any]]) -> Dict[str, List[str]]:
    """Report `source` as an unconfirmed shoot-grouping candidate.

    No asset here sets group_id. `source` free text (e.g. "ff47 event") is the
    only shoot-ish signal on disk, but promoting it to group_id would be
    inferring shoot identity without owner confirmation -- report it as a
    candidate for review instead, per corpus_manifest.py's no-inference rule.
    """
    groups: Dict[str, List[str]] = {}
    for asset in assets:
        key = str(asset.get("source") or "(no source recorded)")
        groups.setdefault(key, []).append(str(asset.get("asset_id")))
    return {key: sorted(ids) for key, ids in sorted(groups.items())}


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", help="corpus_manifest.json files to merge")
    parser.add_argument("--root", default=None, help="root for re-rooted paths (default: common parent)")
    parser.add_argument("--out", default=None, help="write the merged manifest JSON here (not committed)")
    args = parser.parse_args(argv[1:])

    manifest_paths = [Path(p).expanduser().resolve() for p in args.manifests]
    missing = [p for p in manifest_paths if not p.is_file()]
    if missing:
        for p in missing:
            print(f"MISS manifest not found: {p}", file=sys.stderr)
        return 2

    merged_root = Path(args.root).expanduser().resolve() if args.root else ROOT / "test_output"
    merged_manifest, sources_seen = merge_manifests(manifest_paths, merged_root=merged_root)

    report = validate_corpus_manifest(merged_manifest, root=merged_root)

    print(f"merged from {len(sources_seen)} source manifest(s):")
    for source in sources_seen:
        print(f"  {source}")
    print()
    print(f"valid: {report['valid']}")
    print(f"asset_count: {report['asset_count']}")

    person_ids = sorted({pid for asset in report["assets"] for pid in asset.get("person_ids", [])})
    print(f"distinct_person_ids: {len(person_ids)} -> {person_ids}")
    print("(asset_count and distinct_person_ids are different denominators -- "
          "frames are not independent people)")
    print()

    if not report["valid"]:
        print(f"errors ({len(report['errors'])}):", file=sys.stderr)
        for issue in report["errors"]:
            print(f"  {issue}", file=sys.stderr)
        print(file=sys.stderr)
        print("This is an availability/provenance report, not a usable merged "
              "inventory, until these errors are resolved.", file=sys.stderr)

    split_report = corpus_split_report(report)
    print("split_counts (all assets are expected to be 'dev' pre-review):")
    for split in ALLOWED_SPLITS:
        assets_n = split_report["split_counts"].get(split, 0)
        persons_n = split_report["person_counts_by_split"].get(split, 0)
        print(f"  {split:12s} assets={assets_n:4d}  distinct_subjects={persons_n:4d}")
    print()

    gaps = missing_strata_table(report["coverage"])
    print("missing-strata table:")
    print(f"  dimensions with ZERO coverage (no asset sets this at all):")
    for dimension in gaps["dimensions_with_zero_coverage"]:
        print(f"    - {dimension}")
    print(f"  dimensions missing specific values:")
    for dimension, values in gaps["dimensions_missing_some_values"].items():
        print(f"    - {dimension}: missing {values}")
    print(f"  tags never used: {gaps['tags_never_used']}")
    print()

    print("candidate shoot groupings from the 'source' field (UNCONFIRMED -- "
          "review before treating as group_id):")
    for source, asset_ids in candidate_groupings(report["assets"]).items():
        print(f"  {source}: {asset_ids}")
    print()

    print("known schema gap: per-region defect/mark labels have no home in v3 "
          "(identity_marks is free-text, not region geometry) -- out of scope "
          "for this inventory, flagged for an owner decision, not a new schema here.")

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report["canonical_manifest"], indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote merged manifest: {out_path}")

    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

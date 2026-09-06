"""Read-only owner-review contact sheet over a corpus_manifest v3 registry.

    python3 scripts/qa/fa02_portrait_review.py manifest.json --html out/review.html
    python3 scripts/qa/fa02_portrait_review.py manifest.json --split dev --tag marks

Prints the candidate assets in a v3 manifest (optionally filtered by split,
tag or stratum) and, with ``--html``, writes a single contact-sheet page of
native-resolution ``<img>`` references so a human can actually look at the
candidates before deciding anything.

**This tool decides nothing.** It does not assign identities, does not create
or modify a manifest, an annotation record or an image, does not run a
detector, and does not rank or score anything. Its entire job is "show me
these candidates so I can go populate the manifest/annotation files by hand"
-- the same read-only stance as ``scripts/qa/corpus_split_report.py``, which
is deliberately not allowed to auto-populate splits either. The only file it
ever writes is the HTML page the caller explicitly names.

It works on an incrementally-built manifest: calibration and locked_test may
be empty (v3 does not require them to be populated) and the tool still runs.

Exit codes match corpus_split_report.py: 0 ok, 1 manifest invalid, 2 usage.
"""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from retouch.corpus_manifest import (  # noqa: E402
    ALLOWED_SPLITS,
    corpus_split_report,
    validate_corpus_manifest,
)


def select_assets(report, *, splits=None, tags=None, strata=None):
    """Filter validated corpus assets. Pure selection -- no mutation, no scoring."""
    selected = []
    for asset in report.get("assets", []):
        if splits and asset.get("split") not in splits:
            continue
        if tags and not set(tags) <= set(asset.get("tags") or []):
            continue
        if strata:
            asset_strata = asset.get("strata") or {}
            if any(asset_strata.get(dimension) != value for dimension, value in strata.items()):
                continue
        selected.append(asset)
    return selected


def build_contact_sheet(assets, html_path, manifest_root, *, title="FA-02 candidate review"):
    """Write one HTML page of native <img> references. Writes nothing else.

    Images are referenced in place by relative path; nothing is copied,
    resized, cropped or re-encoded. Native pixel dimensions are preserved so a
    reviewer looks at real pixels, matching the FA-02 runner's own
    ``native_review.html`` stance ("scroll, do not fit-to-page").
    """
    html_path = Path(html_path)
    root = Path(manifest_root).resolve()
    lines = [
        '<!doctype html><meta charset="utf-8">',
        f"<title>{html.escape(title)}</title>",
        f"<h1>{html.escape(title)}</h1>",
        "<p>Read-only candidate review. No identity is assigned here, no manifest or "
        "annotation file is written, no detector is run, and no ranking is implied. "
        "Images are shown at native pixel dimensions -- scroll, do not fit-to-page. "
        "After reviewing, populate the corpus manifest and the FA-02 support "
        "annotation records by hand.</p>",
        f"<p>{len(assets)} candidate asset(s).</p>",
    ]
    for asset in assets:
        asset_id = html.escape(str(asset.get("asset_id")))
        split = html.escape(str(asset.get("split")))
        people = html.escape(", ".join(asset.get("person_ids") or []) or "(none declared)")
        tags = html.escape(", ".join(asset.get("tags") or []) or "(none)")
        strata = html.escape(
            ", ".join(f"{k}={v}" for k, v in sorted((asset.get("strata") or {}).items())) or "(none)"
        )
        metadata = asset.get("metadata") or {}
        width, height = metadata.get("width"), metadata.get("height")
        dimensions = f"{width}x{height}" if width and height else "unknown (probe unavailable)"
        try:
            source = (root / asset["path"]).resolve()
            relative = Path(
                os.path.relpath(str(source), str(html_path.resolve().parent))
            ).as_posix()
        except (KeyError, ValueError, OSError):
            relative = str(asset.get("path", ""))
        lines.append(
            f"<h2>{asset_id} <small>[{split}]</small></h2>"
            f"<p>person_ids: {people}<br>tags: {tags}<br>strata: {strata}<br>"
            f"native: {html.escape(dimensions)}<br>path: {html.escape(str(asset.get('path')))}</p>"
            f'<img src="{html.escape(relative)}" alt="{asset_id}">'
        )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text("\n".join(lines), encoding="utf-8")
    return html_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("manifest", type=Path, help="corpus_manifest v3 JSON")
    parser.add_argument("--split", action="append", choices=list(ALLOWED_SPLITS), default=[])
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument(
        "--stratum",
        action="append",
        default=[],
        metavar="DIMENSION=VALUE",
        help="Filter by an exact stratum value (repeatable)",
    )
    parser.add_argument("--html", type=Path, help="Write a contact-sheet HTML page here")
    args = parser.parse_args(argv)

    manifest_path = args.manifest.expanduser().resolve()
    if not manifest_path.is_file():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    strata = {}
    for entry in args.stratum:
        if "=" not in entry:
            parser.error("--stratum expects DIMENSION=VALUE")
        dimension, _, value = entry.partition("=")
        strata[dimension.strip()] = value.strip()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = validate_corpus_manifest(manifest, root=manifest_path.parent)

    print(f"manifest: {manifest_path}")
    print(f"valid:    {report['valid']}")
    split_report = corpus_split_report(report)
    for split in ALLOWED_SPLITS:
        assets = split_report["split_counts"].get(split, 0)
        persons = split_report["person_counts_by_split"].get(split, 0)
        print(f"  {split:12s} assets={assets:4d}  distinct_subjects={persons:4d}")

    if not report["valid"]:
        print(f"\nerrors ({len(report['errors'])}):", file=sys.stderr)
        for issue in report["errors"]:
            print(f"  {issue}", file=sys.stderr)
        return 1

    selected = select_assets(
        report, splits=args.split or None, tags=args.tag or None, strata=strata or None
    )
    print(f"\ncandidates ({len(selected)}):")
    for asset in selected:
        people = ", ".join(asset.get("person_ids") or []) or "(none)"
        tags = ", ".join(asset.get("tags") or []) or "(none)"
        metadata = asset.get("metadata") or {}
        dimensions = (
            f"{metadata.get('width')}x{metadata.get('height')}"
            if metadata.get("width") and metadata.get("height")
            else "native size unknown"
        )
        print(
            f"  {asset.get('asset_id')}  split={asset.get('split')}  "
            f"person_ids=[{people}]  {dimensions}  tags=[{tags}]  path={asset.get('path')}"
        )
    if not selected:
        print("  (no assets match the filter; an incrementally-built manifest may have empty splits)")

    if args.html:
        written = build_contact_sheet(selected, args.html, manifest_path.parent)
        print(f"\ncontact sheet: {written}")
        print("Read-only: no manifest, annotation or image file was created or modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

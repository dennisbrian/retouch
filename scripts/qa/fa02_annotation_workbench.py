"""Read-only native-resolution *looking* tool for FA-02 texture annotation.

    python3 scripts/qa/fa02_annotation_workbench.py CORPUS.json --html out/work.html
    python3 scripts/qa/fa02_annotation_workbench.py CORPUS.json \
        --annotation test_output/fa02_pilot_DSCF2650/annotation_cheek_patch_01.json \
        --html test_output/fa02_pilot_DSCF2650/workbench.html

``fa02_portrait_review.py`` answers "which assets are candidates" with a
contact sheet of whole frames. That is the wrong magnification for the job
this file exists for: deciding whether a 24x24 pixel patch of cheek contains
resolvable pores or JPEG blocking is a 1:1 question, and a browser fitting a
6240x4160 frame into a page has already destroyed the evidence.

This tool renders, per asset, a **windowed view of the native file at 1:1**:
the annotated crop plus a configurable margin of surrounding context, with the
existing annotated regions drawn as outlined overlays so a reviewer can see at
a glance which rectangles are already claimed and which skin is still blank.

**It labels nothing.** There is no detector, no filter, no threshold, no
heuristic and no image analysis of any kind in this file -- it emits an HTML
document containing an ``<img>`` tag pointing at the untouched native file and
some CSS. Every rectangle it can ever draw was typed by a human into an
annotation JSON first. The windowing is done with CSS ``overflow:hidden`` and
a negative offset, so no pixel is ever cropped, resized, re-encoded or written
to disk: the evidence a reviewer looks at is byte-identical to the checksummed
capture, and there is no derived-image provenance question to answer later.

The only file it writes is the HTML page the caller explicitly names, matching
``fa02_portrait_review.py``'s contract.

Workflow it is meant to sit inside:

1. Run this to look at a candidate patch at 1:1.
2. Read coordinates off the on-page rulers / the browser's own inspector.
3. Type them, by hand, into the annotation JSON's ``pore_regions`` /
   ``fine_hair_regions`` / ... lists (copy
   ``scripts/qa/fa02_support_annotation_template.json``).
4. Validate with ``--annotation`` here (it re-validates through
   ``fa02_portrait_manifest.validate_support_annotation`` and refuses to draw
   an invalid record), then with ``fa02_portrait_manifest.py`` for the full
   gate pass.

Exit codes match corpus_split_report.py: 0 ok, 1 invalid input, 2 usage.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fa02_portrait_manifest import (  # noqa: E402
    TEXTURE_REGION_CATEGORIES,
    validate_support_annotation,
)
from retouch.corpus_manifest import validate_corpus_manifest  # noqa: E402

# Distinct outline colours per category so a reviewer can tell claims apart
# without reading labels. Purely presentational; carries no semantics.
CATEGORY_COLOURS = {
    "pore_regions": "#00c2ff",
    "fine_hair_regions": "#7cff00",
    "makeup_edge_regions": "#ff9d00",
    "protected_identity_marks": "#ff0055",
    "corrected_defect_regions": "#b34cff",
    "noise_regions": "#ffe600",
    "uncertainty_regions": "#888888",
    "review_regions": "#ffffff",
}

DEFAULT_CONTEXT_MARGIN = 256


def collect_regions(annotation):
    """Flatten every region list into ``(category, id, xywh, tags)`` records.

    Crop-local coordinates, exactly as stored. No transform is applied here;
    the HTML shifts them into window coordinates at render time so the numbers
    a reviewer sees in this file's output are the numbers in the JSON.
    """
    records = []
    for category in ("review_regions",) + tuple(TEXTURE_REGION_CATEGORIES):
        for region in annotation.get(category) or []:
            records.append(
                {
                    "category": category,
                    "id": region["id"],
                    "xywh": list(region["xywh"]),
                    "tags": list(region.get("tags") or []),
                    "notes": region.get("notes"),
                }
            )
    return records


def _window(annotation, margin):
    """Compute the native-coordinate viewport: the crop plus context margin.

    Clamped to the declared native frame so the window never asks the browser
    for pixels outside the image (which would render as blank padding and
    silently shift every overlay by the clamp amount).
    """
    native_w, native_h = annotation["native_dimensions"]
    crop_x, crop_y, crop_w, crop_h = annotation["native_roi_xywh"]
    left = max(0, crop_x - margin)
    top = max(0, crop_y - margin)
    right = min(native_w, crop_x + crop_w + margin)
    bottom = min(native_h, crop_y + crop_h + margin)
    return {
        "left": left,
        "top": top,
        "width": right - left,
        "height": bottom - top,
        "crop_x": crop_x,
        "crop_y": crop_y,
        "crop_w": crop_w,
        "crop_h": crop_h,
    }


def _panel_html(annotation, image_href, *, margin, title, image_present):
    """One asset panel: 1:1 CSS-windowed native image + typed-region overlays.

    The window is a fixed-size ``overflow:hidden`` box; the ``<img>`` inside it
    is positioned at ``-left/-top`` so the browser paints the requested native
    rectangle at 1:1 with no scaling attribute anywhere. Overlay rectangles are
    absolutely positioned in the same coordinate space.
    """
    win = _window(annotation, margin)
    regions = collect_regions(annotation)
    escape = html.escape

    parts = [f"<h2>{escape(title)}</h2>"]
    parts.append(
        "<p class='meta'>native {nw}x{nh} &middot; crop [{cx},{cy},{cw},{ch}] &middot; "
        "window [{wl},{wt},{ww},{wh}] (crop + {m}px context) &middot; "
        "person_id {pid} &middot; reviewer {rev}</p>".format(
            nw=annotation["native_dimensions"][0],
            nh=annotation["native_dimensions"][1],
            cx=win["crop_x"], cy=win["crop_y"], cw=win["crop_w"], ch=win["crop_h"],
            wl=win["left"], wt=win["top"], ww=win["width"], wh=win["height"],
            m=margin,
            pid=escape(str(annotation.get("person_id"))),
            rev=escape(str(annotation.get("reviewer") or "(none)")),
        )
    )

    if not image_present:
        # Honesty over a rendered-looking placeholder: say the pixels are not
        # here rather than emit a broken <img> that reads like a failed load.
        parts.append(
            "<p class='warn'>Native capture not present at "
            f"<code>{escape(image_href)}</code>. The window geometry and the "
            "typed regions below are still shown, but there are no pixels to "
            "look at in this environment. Run this where the native file "
            "lives; nothing here fabricates image content.</p>"
        )
    else:
        overlays = []
        for record in regions:
            x, y, w, h = record["xywh"]
            # Region coordinates are crop-local; shift into window space.
            wx = win["crop_x"] + x - win["left"]
            wy = win["crop_y"] + y - win["top"]
            colour = CATEGORY_COLOURS.get(record["category"], "#ffffff")
            overlays.append(
                f"<div class='roi' style='left:{wx}px;top:{wy}px;width:{w}px;"
                f"height:{h}px;border-color:{colour}'>"
                f"<span style='background:{colour}'>{escape(record['id'])}</span></div>"
            )
        # The crop boundary itself, so a reviewer sees where the scored canvas
        # ends and the context margin begins.
        overlays.append(
            "<div class='cropline' style='left:{x}px;top:{y}px;width:{w}px;"
            "height:{h}px'></div>".format(
                x=win["crop_x"] - win["left"], y=win["crop_y"] - win["top"],
                w=win["crop_w"], h=win["crop_h"],
            )
        )
        parts.append(
            "<div class='window' style='width:{ww}px;height:{wh}px'>"
            "<img src='{src}' alt='native capture, unmodified' "
            "style='left:{ml}px;top:{mt}px'>{ov}</div>".format(
                ww=win["width"], wh=win["height"], src=escape(image_href),
                ml=-win["left"], mt=-win["top"], ov="".join(overlays),
            )
        )

    if regions:
        rows = "".join(
            "<tr><td style='color:{c}'>&#9632;</td><td>{cat}</td><td><code>{rid}</code></td>"
            "<td><code>{xywh}</code></td><td>{tags}</td><td>{notes}</td></tr>".format(
                c=CATEGORY_COLOURS.get(r["category"], "#fff"),
                cat=escape(r["category"]), rid=escape(r["id"]),
                xywh=escape(json.dumps(r["xywh"])),
                tags=escape(", ".join(r["tags"]) or "-"),
                notes=escape(r.get("notes") or "-"),
            )
            for r in regions
        )
        parts.append(
            "<table><tr><th></th><th>category</th><th>id</th><th>xywh (crop-local)</th>"
            f"<th>tags</th><th>notes</th></tr>{rows}</table>"
        )
    else:
        parts.append(
            "<p class='warn'>No regions declared in this annotation. Nothing in "
            "this tool will create any -- open the crop above at 1:1, decide "
            "what you can actually resolve, and type the coordinates into the "
            "annotation JSON by hand.</p>"
        )

    missing = [c for c in TEXTURE_REGION_CATEGORIES if not annotation.get(c)]
    if missing:
        parts.append(
            "<p class='meta'>unpopulated categories: <code>"
            + escape(", ".join(missing))
            + "</code></p>"
        )
    return "\n".join(parts)


_STYLE = """
body{font:13px/1.5 -apple-system,Helvetica,Arial,sans-serif;margin:24px;background:#111;color:#ddd}
h1{font-size:18px}h2{font-size:15px;margin-top:32px}
.meta{color:#999;font-size:12px}
.warn{color:#ffb347;border-left:3px solid #ffb347;padding-left:10px}
.window{position:relative;overflow:hidden;border:1px solid #444;margin:8px 0}
.window img{position:absolute;image-rendering:pixelated}
.roi{position:absolute;border:1px solid;box-sizing:border-box}
.roi span{position:absolute;top:-16px;left:0;font-size:10px;color:#000;padding:0 3px;white-space:nowrap}
.cropline{position:absolute;border:1px dashed #fff;box-sizing:border-box;opacity:.6}
table{border-collapse:collapse;margin:8px 0;font-size:12px}
th,td{border:1px solid #333;padding:2px 8px;text-align:left}
code{color:#8cf}
"""


def build_workbench(panels, html_path, *, title="FA-02 annotation workbench"):
    """Write the single HTML page. Writes nothing else, ever."""
    html_path = Path(html_path)
    document = [
        '<!doctype html><meta charset="utf-8">',
        f"<title>{html.escape(title)}</title>",
        f"<style>{_STYLE}</style>",
        f"<h1>{html.escape(title)}</h1>",
        "<p class='warn'>READ-ONLY LOOKING TOOL. Nothing here is auto-labelled. "
        "No detector, filter or threshold is run; no image is cropped, resized "
        "or re-encoded (the native file is windowed with CSS and shown at 1:1). "
        "Every rectangle drawn was typed by a human into an annotation JSON. "
        "After looking, edit the annotation file by hand and re-validate with "
        "<code>scripts/qa/fa02_portrait_manifest.py</code>.</p>",
        f"<p class='meta'>{len(panels)} panel(s).</p>",
    ]
    document.extend(panels)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text("\n".join(document), encoding="utf-8")
    return html_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("manifest", type=Path, help="corpus_manifest v3 JSON")
    parser.add_argument(
        "--annotation",
        action="append",
        default=[],
        type=Path,
        help="FA-02 support annotation JSON to window and overlay (repeatable). "
             "Without any, every asset is listed with its native geometry only.",
    )
    parser.add_argument(
        "--context-margin",
        type=int,
        default=DEFAULT_CONTEXT_MARGIN,
        help="Native pixels of surrounding context around the crop (default 256)",
    )
    parser.add_argument("--html", type=Path, help="Write the workbench page here")
    args = parser.parse_args(argv)

    if args.context_margin < 0:
        parser.error("--context-margin must be >= 0")

    manifest_path = args.manifest.expanduser().resolve()
    if not manifest_path.is_file():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = validate_corpus_manifest(manifest, root=manifest_path.parent)
    print(f"manifest: {manifest_path}")
    print(f"valid:    {report['valid']}")
    if not report["valid"]:
        for issue in report["errors"]:
            print(f"  {issue}", file=sys.stderr)
        return 1

    assets = {asset["asset_id"]: asset for asset in report.get("assets", [])}

    panels = []
    for annotation_path in args.annotation:
        annotation_path = annotation_path.expanduser().resolve()
        try:
            annotation = validate_support_annotation(
                json.loads(annotation_path.read_text(encoding="utf-8"))
            )
        except (ValueError, OSError) as error:
            # Fail closed: refuse to draw an unvalidated record. Drawing one
            # would put un-checked coordinates in front of a reviewer who then
            # believes they were verified.
            print(f"annotation invalid ({annotation_path}): {error}", file=sys.stderr)
            return 1
        asset = assets.get(annotation["asset_id"])
        if asset is None:
            print(
                f"annotation asset_id {annotation['asset_id']!r} is not in this "
                f"manifest ({annotation_path})",
                file=sys.stderr,
            )
            return 1

        native = (manifest_path.parent / annotation["native_source_reference"]).resolve()
        present = native.is_file()
        if args.html:
            href = Path(
                os.path.relpath(str(native), str(args.html.expanduser().resolve().parent))
            ).as_posix()
        else:
            href = annotation["native_source_reference"]
        panels.append(
            _panel_html(
                annotation,
                href,
                margin=args.context_margin,
                title=f"{annotation['asset_id']} -- {annotation_path.name}",
                image_present=present,
            )
        )
        counts = {c: len(annotation.get(c) or []) for c in TEXTURE_REGION_CATEGORIES}
        print(
            f"  {annotation['asset_id']}: review_regions="
            f"{len(annotation['review_regions'])} "
            + " ".join(f"{k}={v}" for k, v in counts.items())
            + ("" if present else "  [native file NOT present locally]")
        )

    if not args.annotation:
        # No annotation supplied: list what could be annotated, with the
        # native geometry a reviewer needs to pick a crop. Still no labelling.
        print(f"\nassets available to annotate ({len(assets)}):")
        for asset_id, asset in sorted(assets.items()):
            metadata = asset.get("metadata") or {}
            dimensions = (
                f"{metadata.get('width')}x{metadata.get('height')}"
                if metadata.get("width") and metadata.get("height")
                else "native size unknown"
            )
            print(
                f"  {asset_id}  split={asset.get('split')}  {dimensions}  "
                f"person_ids={asset.get('person_ids')}  path={asset.get('path')}"
            )
        print(
            "\nSupply --annotation PATH to window a specific patch at 1:1. "
            "Copy scripts/qa/fa02_support_annotation_template.json to start one."
        )

    if args.html:
        written = build_workbench(panels, args.html)
        print(f"\nworkbench: {written}")
        print(
            "Read-only: no manifest, annotation or image file was created or "
            "modified, and no region was auto-labelled."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

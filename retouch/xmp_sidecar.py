"""XMP sidecars: carry ratings, picks and flags into Lightroom and Capture One.

Culling decisions made in retouch (the batch review page, the Shoot
Intelligence review manifest) and the batch's automatic QA flags are written as
standard XMP metadata so they show up in other photo editors:

* ``xmp:Rating`` -- 0 to 5 stars; ``-1`` means rejected (Adobe Bridge and
  Capture One read it as a reject; Lightroom keeps pick flags in its catalog
  only, so a reject also gets the Red label and a keyword there).
* ``xmp:Label`` -- the colour label: ``Green`` = pick, ``Red`` = reject,
  ``Yellow`` = held or flagged by QA. The names match the default label sets of
  Lightroom, Bridge and Capture One.
* ``dc:subject`` -- keywords, all starting with ``retouch-`` (for example
  ``retouch-pick``, ``retouch-qa-banding``, ``retouch-closed-eyes``) plus the
  reviewer's own labels from the Shoot manifest.

Where the metadata goes:

* next to each **source** as an Adobe-style ``<stem>.xmp`` sidecar
  (``DSCF1234.RAF`` -> ``DSCF1234.xmp``). Lightroom, Capture One, Bridge and
  Photo Mechanic read these for RAW files; Lightroom ignores sidecars beside
  JPEG originals.
* inside each retouched **JPEG output** as an embedded XMP packet, because
  that is where Lightroom looks for a JPEG's metadata. Non-JPEG outputs get a
  ``<stem>.xmp`` sidecar instead. Sources are never modified.

Sidecars that already exist (written by Lightroom, Capture One or a camera) are
merged, never replaced: everything else in them is kept, the reviewer's own
rating and label are never overwritten, and only keywords starting with
``retouch-`` are replaced. Retouch records the values it wrote in
``retouch:Managed``, so a later run updates a value only while it is still the
one retouch wrote (a label changed in Lightroom afterwards is kept); before the first
change to a sidecar retouch did not create, a one-time ``.xmp.orig`` backup is
saved beside it.

Command line::

    python -m retouch.xmp_sidecar review ROOT [DECISIONS.json]
    python -m retouch.xmp_sidecar shoot MANIFEST.json
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import shutil
import struct
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from urllib.parse import quote, unquote

logger = logging.getLogger(__name__)

NS_X = "adobe:ns:meta/"
NS_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
NS_XMP = "http://ns.adobe.com/xap/1.0/"
NS_DC = "http://purl.org/dc/elements/1.1/"
NS_RETOUCH = "https://github.com/dennisbrian/retouch/ns/xmp/1.0/"

# Well-known prefixes so a merged sidecar keeps readable names. Prefixes found
# in the file being merged are registered on top of these.
_KNOWN_PREFIXES = {
    "x": NS_X,
    "rdf": NS_RDF,
    "xmp": NS_XMP,
    "dc": NS_DC,
    "retouch": NS_RETOUCH,
    "xmpMM": "http://ns.adobe.com/xap/1.0/mm/",
    "stEvt": "http://ns.adobe.com/xap/1.0/sType/ResourceEvent#",
    "stRef": "http://ns.adobe.com/xap/1.0/sType/ResourceRef#",
    "photoshop": "http://ns.adobe.com/photoshop/1.0/",
    "crs": "http://ns.adobe.com/camera-raw-settings/1.0/",
    "crd": "http://ns.adobe.com/camera-raw-defaults/1.0/",
    "lr": "http://ns.adobe.com/lightroom/1.0/",
    "tiff": "http://ns.adobe.com/tiff/1.0/",
    "exif": "http://ns.adobe.com/exif/1.0/",
    "exifEX": "http://cipa.jp/exif/1.0/",
    "aux": "http://ns.adobe.com/exif/1.0/aux/",
    "xmpRights": "http://ns.adobe.com/xap/1.0/rights/",
    "Iptc4xmpCore": "http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/",
}
for _prefix, _uri in _KNOWN_PREFIXES.items():
    ET.register_namespace(_prefix, _uri)

KEYWORD_PREFIX = "retouch-"
LABEL_PICK = "Green"
LABEL_REJECT = "Red"
LABEL_HOLD = "Yellow"
LABEL_FLAGGED = "Yellow"
RATING_REJECT = -1

_MANAGED_ATTR = f"{{{NS_RETOUCH}}}Managed"
_RATING = f"{{{NS_XMP}}}Rating"
_LABEL = f"{{{NS_XMP}}}Label"
_SUBJECT = f"{{{NS_DC}}}subject"
_DESCRIPTION = f"{{{NS_RDF}}}Description"
_BAG = f"{{{NS_RDF}}}Bag"
_LI = f"{{{NS_RDF}}}li"
_ABOUT = f"{{{NS_RDF}}}about"

_JPEG_XMP_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"
_JPEG_SEGMENT_MAX = 65533  # payload bytes allowed after the 2-byte length
_KEYWORD_SAFE_RE = re.compile(r"[^a-z0-9]+")

PathLike = Union[str, Path]


class XmpError(ValueError):
    """A sidecar or embedded packet exists but is not readable XMP."""


@dataclass
class XmpFields:
    """The metadata retouch writes for one photo.

    Attributes:
        rating: ``0``-``5`` stars, ``-1`` for rejected, ``None`` to leave the
            rating alone.
        label: Colour label name (``Green``, ``Red``, ``Yellow`` ...), or
            ``None`` to leave it alone.
        keywords: Keywords to add. Keywords starting with ``retouch-`` that
            are not listed here are removed from the file, so a flipped
            decision does not leave the old one behind.
    """

    rating: Optional[int] = None
    label: Optional[str] = None
    keywords: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.rating is not None:
            self.rating = int(self.rating)
            if self.rating < -1 or self.rating > 5:
                raise ValueError("rating must be -1 (rejected) or 0-5")
        seen: List[str] = []
        for word in self.keywords:
            word = str(word).strip()
            if word and word not in seen:
                seen.append(word)
        self.keywords = seen

    def is_empty(self) -> bool:
        return self.rating is None and self.label is None and not self.keywords


def keyword(*parts: str) -> str:
    """Return a ``retouch-`` keyword, e.g. ``keyword("qa", "Plastic Skin")``."""
    slug = "-".join(_KEYWORD_SAFE_RE.sub("-", str(p).lower()).strip("-") for p in parts if str(p).strip())
    return KEYWORD_PREFIX + slug


# ---------------------------------------------------------------------------
# Packet building and merging
# ---------------------------------------------------------------------------

def _new_root() -> ET.Element:
    root = ET.Element(f"{{{NS_X}}}xmpmeta", {f"{{{NS_X}}}xmptk": "retouch"})
    rdf = ET.SubElement(root, f"{{{NS_RDF}}}RDF")
    ET.SubElement(rdf, _DESCRIPTION, {_ABOUT: ""})
    return root


def _register_prefixes(text: str) -> None:
    """Register every prefix declared in *text* so re-serialising keeps it."""
    try:
        for _event, (prefix, uri) in ET.iterparse(io.StringIO(text), events=("start-ns",)):
            if not prefix or re.match(r"ns\d+$", prefix):
                continue
            try:
                ET.register_namespace(prefix, uri)
            except ValueError:
                pass
    except ET.ParseError:
        pass


def _parse(text: str) -> ET.Element:
    _register_prefixes(text)
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise XmpError(f"not valid XMP: {exc}") from exc
    if root.tag == f"{{{NS_RDF}}}RDF":  # packet without the x:xmpmeta wrapper
        wrapper = ET.Element(f"{{{NS_X}}}xmpmeta", {f"{{{NS_X}}}xmptk": "retouch"})
        wrapper.append(root)
        root = wrapper
    if root.find(f"{{{NS_RDF}}}RDF") is None:
        raise XmpError("no rdf:RDF element")
    return root


def _descriptions(root: ET.Element) -> List[ET.Element]:
    rdf = root.find(f"{{{NS_RDF}}}RDF")
    descs = rdf.findall(_DESCRIPTION)
    if not descs:
        descs = [ET.SubElement(rdf, _DESCRIPTION, {_ABOUT: ""})]
    return descs


def _get_simple(descs: Sequence[ET.Element], name: str) -> Tuple[Optional[str], Optional[ET.Element]]:
    """Value of a simple property written as an attribute or a child element."""
    for desc in descs:
        if name in desc.attrib:
            return desc.attrib[name], desc
        child = desc.find(name)
        if child is not None:
            return (child.text or "").strip(), desc
    return None, None


def _set_simple(descs: Sequence[ET.Element], name: str, value: Optional[str]) -> None:
    for desc in descs:
        desc.attrib.pop(name, None)
        for child in desc.findall(name):
            desc.remove(child)
    if value is not None:
        descs[0].set(name, value)


def _get_keywords(descs: Sequence[ET.Element]) -> List[str]:
    words: List[str] = []
    for desc in descs:
        subject = desc.find(_SUBJECT)
        if subject is None:
            continue
        for li in subject.iter(_LI):
            if li.text and li.text.strip():
                words.append(li.text.strip())
    return words


def _set_keywords(descs: Sequence[ET.Element], words: Sequence[str]) -> None:
    home = descs[0]
    for desc in descs:
        subject = desc.find(_SUBJECT)
        if subject is not None:
            home = desc
            desc.remove(subject)
            break
    if not words:
        return
    subject = ET.SubElement(home, _SUBJECT)
    bag = ET.SubElement(subject, _BAG)
    for word in words:
        ET.SubElement(bag, _LI).text = word


def _parse_managed(value: Optional[str]) -> Dict[str, Optional[str]]:
    """``"Rating=-1 Label=Green Keywords"`` -> ``{"Rating": "-1", ...}``."""
    out: Dict[str, Optional[str]] = {}
    for token in (value or "").split():
        name, sep, text = token.partition("=")
        out[name] = unquote(text) if sep else None
    return out


def _format_managed(managed: Mapping[str, Optional[str]]) -> Optional[str]:
    tokens = [name if value is None else f"{name}={quote(value, safe='-')}"
              for name, value in sorted(managed.items())]
    return " ".join(tokens) or None


def merge_fields(root: ET.Element, fields: XmpFields) -> bool:
    """Merge *fields* into a parsed packet in place; return True if it changed.

    Rating and label are written only when the file has none, or when it
    still holds the value retouch wrote last time (recorded in
    ``retouch:Managed``). A value a person set or changed in another editor is
    left alone. ``None`` for a field retouch still owns removes it.
    """
    descs = _descriptions(root)
    managed_value, _ = _get_simple(descs, _MANAGED_ATTR)
    managed = _parse_managed(managed_value)
    before = ET.tostring(root)

    for name, short, wanted in ((_RATING, "Rating", fields.rating), (_LABEL, "Label", fields.label)):
        current, _ = _get_simple(descs, name)
        current = None if current in (None, "") else current
        if short == "Rating" and current is not None and current.strip() in ("0", "0.0"):
            current = None  # 0 = unrated in Adobe apps; nobody decided anything
        wanted_text = None if wanted is None else str(wanted)
        ours = short in managed and managed[short] == current
        if ours or current is None:
            _set_simple(descs, name, wanted_text)
            if wanted_text is None:
                managed.pop(short, None)
            else:
                managed[short] = wanted_text
        else:  # someone else's value: keep it and stop claiming the field
            managed.pop(short, None)

    existing = _get_keywords(descs)
    kept = [w for w in existing if not w.startswith(KEYWORD_PREFIX)]
    ours_words = [w for w in fields.keywords if w.startswith(KEYWORD_PREFIX)]
    extra = [w for w in fields.keywords if not w.startswith(KEYWORD_PREFIX) and w not in kept]
    words = kept + extra + ours_words
    if words != existing:
        _set_keywords(descs, words)
    if ours_words:
        managed["Keywords"] = None
    else:
        managed.pop("Keywords", None)

    _set_simple(descs, _MANAGED_ATTR, _format_managed(managed))
    return ET.tostring(root) != before


def _serialise(root: ET.Element) -> str:
    try:
        ET.indent(root, space=" ")  # Python 3.9+
    except AttributeError:  # pragma: no cover
        pass
    return ET.tostring(root, encoding="unicode") + "\n"


def build_packet(fields: XmpFields, existing: Optional[str] = None) -> Tuple[str, bool]:
    """Return ``(packet_text, changed)`` for *fields* merged into *existing*."""
    root = _parse(existing) if existing else _new_root()
    changed = merge_fields(root, fields)
    return _serialise(root), changed or existing is None


def read_fields(text: str) -> Dict[str, Any]:
    """Read rating, label, keywords and the managed list from a packet."""
    descs = _descriptions(_parse(text))
    rating, _ = _get_simple(descs, _RATING)
    label, _ = _get_simple(descs, _LABEL)
    managed, _ = _get_simple(descs, _MANAGED_ATTR)
    try:
        rating_value: Optional[int] = int(float(rating)) if rating not in (None, "") else None
    except ValueError:
        rating_value = None
    return {
        "rating": rating_value,
        "label": label or None,
        "keywords": _get_keywords(descs),
        "managed": sorted(_parse_managed(managed)),
    }


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def sidecar_path(image: PathLike) -> Path:
    """Adobe-style sidecar path: ``DSCF1234.RAF`` -> ``DSCF1234.xmp``."""
    return Path(image).with_suffix(".xmp")


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def write_sidecar(image: PathLike, fields: XmpFields) -> Optional[Path]:
    """Create or merge the ``.xmp`` sidecar for *image*.

    Returns the sidecar path when it was written, ``None`` when nothing
    changed. Raises :class:`XmpError` when an existing sidecar is not
    readable XMP (it is left untouched).
    """
    target = sidecar_path(image)
    existing: Optional[str] = None
    if target.exists():
        existing = target.read_text(encoding="utf-8-sig", errors="strict")
    elif fields.is_empty():
        return None
    text, changed = build_packet(fields, existing)
    if not changed:
        return None
    if existing is not None and not read_fields(existing)["managed"]:
        backup = target.with_name(target.name + ".orig")
        if not backup.exists():
            shutil.copy2(target, backup)
    _atomic_write(target, text.encode("utf-8"))
    return target


def _jpeg_segments(data: bytes) -> Iterable[Tuple[int, int, int]]:
    """Yield ``(marker, start, end)`` for header segments up to SOS."""
    if data[:2] != b"\xff\xd8":
        raise XmpError("not a JPEG file")
    pos = 2
    while pos + 4 <= len(data):
        if data[pos] != 0xFF:
            raise XmpError("corrupt JPEG header")
        marker = data[pos + 1]
        if marker == 0xFF:  # fill byte
            pos += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        length = struct.unpack(">H", data[pos + 2:pos + 4])[0]
        end = pos + 2 + length
        yield marker, pos, end
        if marker == 0xDA:  # start of scan: header ends
            return
        pos = end


def embed_in_jpeg(path: PathLike, fields: XmpFields) -> bool:
    """Merge *fields* into the XMP packet embedded in a JPEG (APP1).

    Only for files retouch wrote itself (batch outputs); never call this on a
    source. Returns True when the file was rewritten.
    """
    path = Path(path)
    data = path.read_bytes()
    found: Optional[Tuple[int, int]] = None
    insert_at = 2
    existing: Optional[str] = None
    in_app_block = True
    for marker, start, end in _jpeg_segments(data):
        if marker == 0xE1 and data[start + 4:start + 4 + len(_JPEG_XMP_HEADER)] == _JPEG_XMP_HEADER:
            found = (start, end)
            raw = data[start + 4 + len(_JPEG_XMP_HEADER):end]
            existing = raw.decode("utf-8", errors="replace").strip("\x00 \n")
            existing = re.sub(r"<\?xpacket[^>]*\?>", "", existing).strip()
            break
        if in_app_block and 0xE0 <= marker <= 0xEF:  # new XMP goes after JFIF/EXIF
            insert_at = end
        else:
            in_app_block = False
    if found is None and fields.is_empty():
        return False
    text, changed = build_packet(fields, existing or None)
    if not changed:
        return False
    packet = ('<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
              + text + '<?xpacket end="w"?>').encode("utf-8")
    payload = _JPEG_XMP_HEADER + packet
    if len(payload) + 2 > _JPEG_SEGMENT_MAX + 2:
        raise XmpError("XMP packet too large for one JPEG segment")
    segment = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    if found is not None:
        out = data[:found[0]] + segment + data[found[1]:]
    else:
        out = data[:insert_at] + segment + data[insert_at:]
    _atomic_write(path, out)
    return True


def read_jpeg_fields(path: PathLike) -> Optional[Dict[str, Any]]:
    """Fields from a JPEG's embedded XMP packet, or ``None`` when it has none."""
    data = Path(path).read_bytes()
    for marker, start, end in _jpeg_segments(data):
        if marker == 0xE1 and data[start + 4:start + 4 + len(_JPEG_XMP_HEADER)] == _JPEG_XMP_HEADER:
            raw = data[start + 4 + len(_JPEG_XMP_HEADER):end].decode("utf-8", errors="replace")
            raw = re.sub(r"<\?xpacket[^>]*\?>", "", raw).strip("\x00 \n")
            return read_fields(raw)
    return None


def write_output_metadata(output: PathLike, fields: XmpFields) -> Optional[Path]:
    """Embed into a JPEG output, or write a sidecar for any other format."""
    output = Path(output)
    if output.suffix.lower() in (".jpg", ".jpeg"):
        return output if embed_in_jpeg(output, fields) else None
    return write_sidecar(output, fields)


# ---------------------------------------------------------------------------
# Mapping retouch decisions to fields
# ---------------------------------------------------------------------------

def fields_for_review(record: Any, decision: Optional[str] = None) -> XmpFields:
    """Fields for one batch review record plus an optional pick/reject.

    Automatic QA flags become keywords and the Yellow label; a pick is the
    Green label, a reject the Red label and ``Rating -1``. No stars are
    invented: the batch never rates photos by itself.
    """
    words: List[str] = []
    label: Optional[str] = None
    rating: Optional[int] = None
    status = getattr(record, "status", "done")
    flagged = [q for q in (getattr(record, "qa", None) or []) if q.get("flagged")]
    for entry in flagged:
        words.append(keyword("qa", str(entry.get("detector") or "flag")))
    if status == "qa_fail":
        words.append(keyword("qa-fail"))
    elif status == "failed":
        words.append(keyword("failed"))
    if flagged or status == "qa_fail":
        label = LABEL_FLAGGED
    recipe = getattr(record, "recipe", None)
    if recipe and status in ("done", "qa_fail"):
        words.append(keyword("recipe", recipe))
    if decision == "pick":
        label = LABEL_PICK
        words.append(keyword("pick"))
    elif decision == "reject":
        label = LABEL_REJECT
        rating = RATING_REJECT
        words.append(keyword("reject"))
    return XmpFields(rating=rating, label=label, keywords=words)


def fields_for_shoot_asset(asset: Any) -> XmpFields:
    """Fields for one Shoot Intelligence review asset.

    A saved human review sets the colour label (select Green, reject Red,
    hold Yellow) and its 0-5 rating becomes stars; a reject with no rating is
    ``-1``. The automatic burst pick (``retouch-burst-best``) and closed or
    blinking eyes (``retouch-closed-eyes``) become keywords only, never labels
    or stars, so they stay suggestions.
    """
    words: List[str] = []
    label: Optional[str] = None
    rating = getattr(asset, "rating", None)
    decision = getattr(asset, "decision", "hold")
    # Only a saved human review counts: a re-scan marks unchanged assets
    # "human" even when nobody has reviewed them yet.
    human = (getattr(asset, "decision_origin", "automatic") == "human"
             and bool(getattr(asset, "override_history", None)))
    if human:
        label = {"select": LABEL_PICK, "reject": LABEL_REJECT, "hold": LABEL_HOLD}.get(decision)
        words.append(keyword({"select": "pick", "reject": "reject", "hold": "hold"}.get(decision, decision)))
        if decision == "reject" and rating is None:
            rating = RATING_REJECT
    else:
        rating = None
    candidate = (getattr(asset, "culling_evidence", None) or {}).get("candidate") or {}
    flags = (candidate.get("evidence") or {}).get("flags") or []
    if getattr(asset, "burst_ids", None) or (getattr(asset, "culling_evidence", None) or {}).get("burst_groups"):
        words.append(keyword("burst"))
        if candidate.get("rank") == 1:
            words.append(keyword("burst-best"))
    faces = getattr(asset, "faces", None) or []
    if "eyes_closed" in flags or any(getattr(face, "eyes_open", "uncertain") == "no" for face in faces):
        words.append(keyword("closed-eyes"))
    words.extend(str(item) for item in (getattr(asset, "labels", None) or []))
    return XmpFields(rating=rating, label=label, keywords=words)


# ---------------------------------------------------------------------------
# Bulk writers
# ---------------------------------------------------------------------------

def _empty_counts() -> Dict[str, int]:
    return {"sidecars": 0, "embedded": 0, "unchanged": 0, "missing": 0, "errors": 0}


def _write_one(counts: Dict[str, int], kind: str, func, path: Path, fields: XmpFields) -> None:
    try:
        written = func(path, fields)
    except (OSError, XmpError, ValueError) as exc:
        logger.warning("XMP not written for %s: %s", path, exc)
        counts["errors"] += 1
        return
    if written:
        counts[kind] += 1
    else:
        counts["unchanged"] += 1


def write_review_xmp(root: PathLike, decisions: Optional[Union[PathLike, Mapping[str, str]]] = None, *,
                     sources: bool = True, outputs: bool = True) -> Dict[str, int]:
    """Write XMP for every record of a batch review root.

    Args:
        root: Batch review root (the batch output folder).
        decisions: Optional exported ``decisions.json`` (or a mapping) from
            ``review.html``; picks and rejects are added on top of QA flags.
        sources: Write ``<stem>.xmp`` beside each source image.
        outputs: Embed into each JPEG output (sidecar for other formats).

    Returns:
        Counts: ``sidecars``, ``embedded``, ``unchanged``, ``missing``,
        ``errors``.
    """
    from .review_page import _load_decisions, load_review_records, record_key

    root = Path(root).expanduser().resolve()
    wanted = _load_decisions(root, decisions) if decisions is not None else {}
    counts = _empty_counts()
    for record in load_review_records(root):
        fields = fields_for_review(record, wanted.get(record_key(record.source)))
        source = Path(record.source)
        if sources:
            if source.is_file():
                _write_one(counts, "sidecars", write_sidecar, source, fields)
            else:
                counts["missing"] += 1
        if outputs and record.output:
            output = Path(record.output)
            if output.is_file():
                kind = "embedded" if output.suffix.lower() in (".jpg", ".jpeg") else "sidecars"
                _write_one(counts, kind, write_output_metadata, output, fields)
            else:
                counts["missing"] += 1
    return counts


def write_shoot_xmp(manifest: Union[PathLike, Any]) -> Dict[str, int]:
    """Write a ``<stem>.xmp`` sidecar beside every source in a Shoot manifest."""
    if not hasattr(manifest, "assets"):
        from .shoot_review import ShootReviewManifest
        manifest = ShootReviewManifest.load(manifest)
    root = Path(manifest.project_root)
    counts = _empty_counts()
    for asset in manifest.assets.values():
        source = root / asset.relative_path
        if not getattr(asset, "present", True) or not source.is_file():
            counts["missing"] += 1
            continue
        _write_one(counts, "sidecars", write_sidecar, source, fields_for_shoot_asset(asset))
    return counts


def format_counts(counts: Mapping[str, int]) -> str:
    """One-line human summary of a bulk write."""
    parts = [f"{counts.get('sidecars', 0)} sidecar(s) written"]
    if counts.get("embedded"):
        parts.append(f"{counts['embedded']} JPEG output(s) tagged")
    if counts.get("unchanged"):
        parts.append(f"{counts['unchanged']} already up to date")
    if counts.get("missing"):
        parts.append(f"{counts['missing']} file(s) missing")
    if counts.get("errors"):
        parts.append(f"{counts['errors']} error(s), see log")
    return "XMP: " + ", ".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m retouch.xmp_sidecar",
        description="Write ratings, picks and flags as XMP for Lightroom, Capture One and Bridge.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    review = sub.add_parser("review", help="A batch output folder (QA flags, plus review.html decisions)")
    review.add_argument("root", help="Batch output folder (the one holding review.html)")
    review.add_argument("decisions", nargs="?", default=None,
                        help="decisions.json exported from review.html (optional)")
    review.add_argument("--no-sources", action="store_true", help="Skip sidecars beside the source photos")
    review.add_argument("--no-outputs", action="store_true", help="Skip the retouched outputs")
    shoot = sub.add_parser("shoot", help="A Shoot Intelligence review manifest (ratings, decisions)")
    shoot.add_argument("manifest", help="Path to .retouch-shoot-review.json")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    try:
        if args.command == "review":
            counts = write_review_xmp(args.root, args.decisions,
                                      sources=not args.no_sources, outputs=not args.no_outputs)
        else:
            counts = write_shoot_xmp(args.manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(format_counts(counts))
    return 1 if counts.get("errors") else 0


if __name__ == "__main__":
    sys.exit(main())

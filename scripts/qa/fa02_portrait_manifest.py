"""Bridge: build FA-02 portrait cases FROM a corpus_manifest v3 asset record.

Two provenance systems exist independently in this repository and, before this
module, never spoke to each other:

* ``retouch/corpus_manifest.py`` (schema v3) governs *who* is in the corpus and
  *which split* they belong to. It enforces mandatory ``person_ids``, the
  ``dev``/``calibration``/``locked_test`` vocabulary and, critically,
  ``person_crosses_split`` -- one real individual can never appear in two
  splits.
* ``scripts/qa/fa02_texture_representation_experiment.py`` (its own bespoke
  ``schema_version=1``) governs *what an FA-02 case is*: canvases, external
  supports, crop geometry, checksums, purpose-gated splits.

The FA-02 harness's ``validate_manifest`` already asserts a portrait contract,
but only *internally*: a case that claims ``person_ids=["x"], split="dev"`` is
self-consistent no matter what the project-wide registry says about ``x``.
Nothing stopped a ``locked_test`` corpus subject from being silently reused in
a ``development_pilot`` case, because the two systems never cross-checked.

This module is that cross-check. It does three things and deliberately nothing
else:

1. :func:`build_portrait_case` constructs an FA-02 case dict from a *validated*
   corpus_manifest v3 asset plus an accepted-support annotation record, so the
   case's subject/split provenance is proven against the registry rather than
   asserted from thin air.
2. :func:`validate_support_annotation` validates the new annotation/support
   record -- the acceptance provenance for masks/ROIs/reviewer that
   corpus_manifest v3 has no concept of. Since 2026-09-06 it also carries the
   semantic texture categories in :data:`TEXTURE_REGION_CATEGORIES` (pore,
   fine-hair, makeup-edge, protected, corrected, noise, uncertainty), which
   are the human ground truth FA-02 needs and does not yet have. They are all
   optional, so annotations written before they existed still validate.
3. :func:`check_promotion_gates` re-checks both systems against each other for
   a whole prospective run, and :func:`main` exposes it as a CLI that exits
   nonzero on any violation (matching ``scripts/qa/corpus_split_report.py``).

Error-handling convention follows the file being extended: the harness's
``validate_manifest`` raises ``ValueError`` on the first structural problem, so
the builder and the annotation validator raise ``ValueError`` too. Only the
gate *CLI* accumulates a list, so an owner sees every violation in one pass.

**Nothing here infers anything from an image.** No detector, no landmark, no
segmentation is consulted; ``detector_derived`` is a vocabulary value a human
records about where a mask came from, never something this module computes.
Masks/ROIs must already exist and be externally accepted; this module only
carries and checks their stated provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fa02_texture_representation_experiment as exp  # noqa: E402
from retouch.corpus_manifest import (  # noqa: E402
    ALLOWED_SPLITS,
    validate_corpus_manifest,
)

# Where a mask/annotation came from. This is a *record of human provenance*,
# not a computation. "detector_derived" means a human ran something and then
# reviewed/accepted the result -- this module never runs a detector, and a
# detector_derived annotation is still only usable once a human accepts it
# (see ACCEPTED_ANNOTATION_SOURCES below).
ANNOTATION_SOURCES = ("manual", "synthetic", "detector_derived", "owner_approved")

# Only these two count as an accepted support for a real portrait run. A raw
# detector output or a synthetic stand-in is recordable but never promotable:
# the FA-02 harness requires support_status == "externally_accepted", and a
# bare detector mask is by definition not externally accepted.
ACCEPTED_ANNOTATION_SOURCES = ("manual", "owner_approved")

# Splits an FA-02 development/tuning run may draw from. Mirrors the harness's
# own development_pilot gate (dev/calibration only); locked_test is refused by
# default for anything that behaves like development or tuning.
DEVELOPMENT_SPLITS = ("dev", "calibration")

PURPOSE_SPLITS = {
    "development_pilot": DEVELOPMENT_SPLITS,
    "locked_comparison": ("locked_test",),
}

_SAFE_ID_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
)

ANNOTATION_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Semantic texture-annotation categories (added 2026-09-06)
# ---------------------------------------------------------------------------
# The original schema had exactly one region list, ``review_regions``, whose
# only descriptor was a free-text ``tags`` list. Free text is not a label: a
# reviewer writing ``tags: ["pore_review"]`` is saying "someone should look
# here", NOT "this rectangle contains pores I have identified". Every one of
# the four committed real pilot patches uses exactly that advisory form, which
# is why the FA-02 research question is still unanswered -- there is no
# ground truth anywhere in the corpus.
#
# These named lists are the ground-truth carrier. A rectangle appearing in
# ``pore_regions`` is a human assertion that pores are visible there. Nothing
# in this repository may write one: they exist only because a person typed
# coordinates after looking at native pixels (see
# ``scripts/qa/fa02_annotation_workbench.py``, which shows and never labels).
#
# ALL of them are optional and default to ``[]``. That is deliberate and
# load-bearing: the four committed annotations (and the four committed FA-02
# manifests that embed them) declare none of these, and must keep validating
# byte-for-byte unchanged. This schema EXTENDS, it does not replace.
TEXTURE_REGION_CATEGORIES = (
    # Visible skin pores the reviewer can actually resolve in the native crop.
    "pore_regions",
    # Vellus/fine facial hair -- the other signal FA-02 is meant to preserve.
    "fine_hair_regions",
    # Boundaries of applied makeup (liner, contour, lip edge). Distinct from a
    # protected mark: a makeup edge is a nuisance edge a representation may
    # over-sharpen, not necessarily something to hold untouched.
    "makeup_edge_regions",
    # Marks a human decided are identity and must never be altered.
    "protected_identity_marks",
    # Regions where a defect was already repaired before S was captured.
    "corrected_defect_regions",
    # Sensor noise / JPEG blocking heavy enough to be the dominant signal.
    "noise_regions",
    # The reviewer looked and could NOT decide. Explicitly recorded rather
    # than silently omitted, because an unlabeled region and an unlabelABLE
    # region are different evidence and only one of them is a coverage gap.
    "uncertainty_regions",
)

# Categories that carry positive texture ground truth. Used by the readiness
# report to decide whether human-reviewed texture labels exist at all --
# uncertainty_regions deliberately does NOT count (recording that you could
# not tell is honest, but it is not a label).
GROUND_TRUTH_CATEGORIES = (
    "pore_regions",
    "fine_hair_regions",
)

# Nuisance/control categories: they bound where a representation must behave,
# rather than asserting recoverable texture.
CONTROL_CATEGORIES = (
    "makeup_edge_regions",
    "protected_identity_marks",
    "corrected_defect_regions",
    "noise_regions",
)

# Pairs of categories that must not geometrically overlap, and why.
#
#   protected x corrected -- the FA-02 harness ALREADY refuses overlapping
#       corrected/protected *masks* (``validate_manifest``: "Corrected and
#       protected supports must be disjoint"). A region-level annotation that
#       claims both about the same pixels contradicts the mask-level rule this
#       annotation is the provenance record for. Extend, do not weaken.
#   protected x noise -- the owner named this one. "This is an identity mark
#       I am asserting must survive" and "this is compression garbage" are
#       incompatible claims about the same pixels; one of them is wrong, and
#       silently keeping both would let a scoring pass count the same
#       rectangle as both signal and nuisance.
#
# Everything NOT listed here may legitimately overlap and is left alone:
# pores and fine hair coexist on any real cheek; a makeup edge sits on top of
# a noisy region; uncertainty overlaps anything by construction (that is what
# being unsure means). Over-constraining would force reviewers to lie.
MUTUALLY_EXCLUSIVE_CATEGORIES = (
    ("protected_identity_marks", "corrected_defect_regions"),
    ("protected_identity_marks", "noise_regions"),
)


def _text(value, field):
    """Require a non-empty string; raise like the harness does."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Annotation field {field} must be a non-empty string")
    return value.strip()


def _safe_id(value, field):
    text = _text(value, field)
    if any(character not in _SAFE_ID_CHARS for character in text):
        raise ValueError(f"{field} must be a filesystem-safe token")
    return text


def _roi(value, field, *, bounds=None):
    """Validate an ``[x, y, width, height]`` rectangle.

    Mirrors the FA-02 harness's own ROI check (``min(x, y) < 0 or
    min(w, h) <= 0 or x+w > W or y+h > H``) rather than inventing a looser
    rule. ``bounds`` is ``(width, height)`` of the coordinate space the
    rectangle lives in; for an annotation's native ROI that is the *declared
    native image size*, not the saved crop's array shape -- those are two
    different rectangles in two different coordinate spaces and are validated
    separately on purpose.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{field} must be [x, y, width, height]")
    for component in value:
        if isinstance(component, bool) or not isinstance(component, int):
            raise ValueError(f"{field} must contain integer pixel coordinates")
    x, y, width, height = value
    if min(x, y) < 0 or min(width, height) <= 0:
        raise ValueError(f"{field} must be a positive in-frame rectangle")
    if bounds is not None:
        limit_w, limit_h = bounds
        if x + width > limit_w or y + height > limit_h:
            raise ValueError(f"{field} is out of bounds for the declared native image")
    return [int(x), int(y), int(width), int(height)]


def _positive_int(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _positive_number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a positive number")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{field} must be a positive finite number")
    return float(value)


def _rectangles_overlap(first, second):
    """True when two ``[x, y, w, h]`` rectangles share interior area.

    Strict area overlap, not edge contact: two rectangles that merely abut
    (``[0,0,10,10]`` and ``[10,0,10,10]``) share a boundary line of zero area
    and are NOT in conflict. A reviewer tiling a cheek into adjacent patches
    must not be punished for it.
    """
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _region_list(value, field):
    """Validate one list of crop-local annotated regions.

    Same shape as ``review_regions`` on purpose -- ``{id, xywh, tags}`` -- so a
    reviewer types the same thing regardless of category, and so the existing
    :func:`_roi` bounds checker is the single implementation of "is this a
    rectangle". ``notes`` is accepted as optional per-region free text because
    a category label without a reason ages badly.

    Bounds against the saved crop are NOT checked here: the crop's array shape
    is not known at annotation-validation time (the annotation is a standalone
    file). :func:`check_promotion_gates` does that check once the case's NPZ is
    on hand, exactly as it already does for ``review_regions``.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list when present")
    normalized = []
    for index, region in enumerate(value):
        if not isinstance(region, dict):
            raise ValueError(f"{field}[{index}] must be an object")
        tags = region.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError(f"{field}[{index}].tags must be a list of strings")
        entry = {
            "id": _safe_id(region.get("id"), f"{field}[{index}].id"),
            "xywh": _roi(region.get("xywh"), f"{field}[{index}].xywh"),
            "tags": sorted(set(tags)),
        }
        note = region.get("notes")
        if note is not None:
            entry["notes"] = _text(note, f"{field}[{index}].notes")
        normalized.append(entry)
    return normalized


def _validate_texture_regions(annotation, result):
    """Validate, normalize and cross-check the semantic category lists.

    Enforces, in order:

    1. Each present list is a well-formed list of in-frame rectangles
       (:func:`_region_list` -> :func:`_roi`; the same rule the harness uses).
    2. Region IDs are unique across ALL categories, not merely within one.
       The readiness report counts these IDs and an owner will reference them
       by name, so a duplicate is an ambiguous reference, not a harmless one.
    3. No geometric overlap between the pairs in
       :data:`MUTUALLY_EXCLUSIVE_CATEGORIES`.
    4. Provenance: if ANY category is non-empty, the annotation as a whole must
       carry an accepted source and a named reviewer.

    On (4) -- the provenance question the task asks to decide explicitly: this
    is **annotation-level, gated on non-emptiness**, reusing the existing
    ``annotation_source`` + ``reviewer`` fields rather than adding per-region
    provenance. Rationale: a per-region provenance field would have to be
    optional to keep the four committed annotations valid, and an optional
    provenance field is not a gate. Annotation-level is already mandatory,
    already audited by ``check_promotion_gates``
    (``owner_approved_without_reviewer``), and matches the physical reality --
    one person sat down with one patch and typed all of its regions in one
    sitting. A reviewer who needs to mix provenance within a patch should
    write two annotation records, which the format already supports.
    """
    present = {}
    for category in TEXTURE_REGION_CATEGORIES:
        regions = _region_list(annotation.get(category), category)
        result[category] = regions
        if regions:
            present[category] = regions

    # (2) IDs unique across every category, including legacy review_regions --
    # they all land in the same reporting namespace.
    seen = {}
    all_lists = list(present.items()) + [("review_regions", result.get("review_regions") or [])]
    for category, regions in all_lists:
        for region in regions:
            previous = seen.get(region["id"])
            if previous is not None:
                raise ValueError(
                    f"Region id {region['id']!r} is used in both {previous} and "
                    f"{category}; region ids must be unique across all categories"
                )
            seen[region["id"]] = category

    # (3) Semantically contradictory overlaps.
    for left, right in MUTUALLY_EXCLUSIVE_CATEGORIES:
        for first in present.get(left, []):
            for second in present.get(right, []):
                if _rectangles_overlap(first["xywh"], second["xywh"]):
                    raise ValueError(
                        f"Region {first['id']!r} ({left}) overlaps {second['id']!r} "
                        f"({right}); these categories are mutually exclusive"
                    )

    # (4) Provenance for the new claims.
    if present:
        categories = ", ".join(sorted(present))
        if annotation.get("annotation_source") not in ACCEPTED_ANNOTATION_SOURCES:
            raise ValueError(
                f"Texture regions ({categories}) require an accepted "
                f"annotation_source ({', '.join(ACCEPTED_ANNOTATION_SOURCES)}); "
                "a detector-derived or synthetic label is not human ground truth"
            )
        if not result.get("reviewer"):
            raise ValueError(
                f"Texture regions ({categories}) require a named reviewer; "
                "an unattributable label is not ground truth"
            )
    return result


def validate_support_annotation(annotation):
    """Validate one accepted-support annotation record; return it normalized.

    This is the format that carries what corpus_manifest v3 has no concept of:
    which *rectangle* of a native asset an FA-02 patch covers, where its
    allow/corrected/protected supports came from, who accepted them, and what
    the reviewer was unsure about. It describes *acceptance provenance* for
    those masks; it does not redefine the masks, which live in the case's NPZ
    exactly as the harness already requires.

    Fail-closed: raises ``ValueError`` on the first structural problem, the
    same convention as ``fa02_texture_representation_experiment.validate_manifest``.
    """
    if not isinstance(annotation, dict):
        raise ValueError("Support annotation must be a JSON object")
    if annotation.get("schema_version") != ANNOTATION_SCHEMA_VERSION:
        raise ValueError("Expected FA-02 support annotation schema_version=1")

    result = {"schema_version": ANNOTATION_SCHEMA_VERSION}
    result["asset_id"] = _text(annotation.get("asset_id"), "asset_id")
    result["person_id"] = _text(annotation.get("person_id"), "person_id")

    source = annotation.get("annotation_source")
    if source not in ANNOTATION_SOURCES:
        raise ValueError(
            "annotation_source must be one of " + ", ".join(ANNOTATION_SOURCES)
        )
    result["annotation_source"] = source

    # A human identifier is what makes "owner_approved" mean anything. Without
    # it the value is an unattributable claim, so refuse it outright.
    reviewer = annotation.get("reviewer")
    if source == "owner_approved":
        result["reviewer"] = _text(reviewer, "reviewer")
    elif reviewer is not None:
        result["reviewer"] = _text(reviewer, "reviewer")

    native = annotation.get("native_dimensions")
    if not isinstance(native, (list, tuple)) or len(native) != 2:
        raise ValueError("native_dimensions must be [width, height] in native pixels")
    native_w = _positive_int(native[0], "native_dimensions[0]")
    native_h = _positive_int(native[1], "native_dimensions[1]")
    result["native_dimensions"] = [native_w, native_h]

    result["native_roi_xywh"] = _roi(
        annotation.get("native_roi_xywh"), "native_roi_xywh", bounds=(native_w, native_h)
    )

    # The native capture this patch was cut from, stated by the annotator and
    # relative to the eventual FA-02 manifest directory. Required: without it
    # the builder would have to guess (e.g. from the corpus asset's own path),
    # and a silent guess is exactly the provenance hole this bridge closes.
    result["native_source_reference"] = _text(
        annotation.get("native_source_reference"), "native_source_reference"
    )

    # Mask references, not mask contents: this record says where the accepted
    # supports came from and who accepted them. The arrays themselves are
    # validated by the harness against the saved NPZ.
    supports = annotation.get("supports")
    if not isinstance(supports, dict):
        raise ValueError("supports must be an object with allow/corrected/protected entries")
    normalized_supports = {}
    for key in exp.MASKS:
        entry = supports.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"supports.{key} must be an object")
        normalized = {
            "reference": _text(entry.get("reference"), f"supports.{key}.reference"),
            "provenance": entry.get("provenance"),
        }
        if normalized["provenance"] not in ANNOTATION_SOURCES:
            raise ValueError(
                f"supports.{key}.provenance must be one of " + ", ".join(ANNOTATION_SOURCES)
            )
        # "corrected" describes a known repaired defect: the harness treats it
        # as forbidden territory, so the record must say what was corrected and
        # on whose authority, not merely that something was.
        if key == "corrected":
            normalized["description"] = _text(
                entry.get("description"), "supports.corrected.description"
            )
            normalized["human_reference"] = _text(
                entry.get("human_reference"), "supports.corrected.human_reference"
            )
        if key == "protected":
            # Marks/makeup a human decided must not be altered. A reference,
            # deliberately never a literal detector output.
            normalized["description"] = _text(
                entry.get("description"), "supports.protected.description"
            )
        elif key == "allow" and entry.get("description") is not None:
            # Optional for allow, but do not silently DROP it when supplied:
            # all four committed real annotations carry one, and this module's
            # whole purpose is lossless provenance carriage. A normalizer that
            # rebuilds the record key-by-key discards anything it forgets.
            normalized["description"] = _text(
                entry.get("description"), "supports.allow.description"
            )
        normalized_supports[key] = normalized
    result["supports"] = normalized_supports

    result["support_acceptance_reference"] = _text(
        annotation.get("support_acceptance_reference"), "support_acceptance_reference"
    )

    # Optional: small rectangles a reviewer flagged for pore/fine-hair looking.
    # Coordinates are local to the saved crop, matching the harness's own ROI
    # convention, and are bounds-checked against the crop later by the harness.
    review_regions = annotation.get("review_regions", [])
    if not isinstance(review_regions, list):
        raise ValueError("review_regions must be a list when present")
    normalized_regions = []
    for index, region in enumerate(review_regions):
        if not isinstance(region, dict):
            raise ValueError(f"review_regions[{index}] must be an object")
        tags = region.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError(f"review_regions[{index}].tags must be a list of strings")
        normalized_regions.append(
            {
                "id": _safe_id(region.get("id"), f"review_regions[{index}].id"),
                "xywh": _roi(region.get("xywh"), f"review_regions[{index}].xywh"),
                "tags": sorted(set(tags)),
            }
        )
    result["review_regions"] = normalized_regions

    for optional in ("notes", "uncertainty"):
        value = annotation.get(optional)
        if value is not None:
            result[optional] = _text(value, optional)

    # Semantic texture categories. Added after the four committed real pilot
    # patches shipped with review_regions only; all of these default to [] so
    # those files -- and the FA-02 manifests that embed them -- keep validating
    # unchanged. Must run AFTER review_regions/reviewer are normalized: the ID
    # uniqueness check spans both namespaces and the provenance gate reads
    # result["reviewer"].
    _validate_texture_regions(annotation, result)
    return result


def load_support_annotation(path):
    """Read and validate an annotation JSON file."""
    path = Path(path)
    return validate_support_annotation(json.loads(path.read_text()))


def _corpus_asset(corpus_report, asset_id):
    for asset in corpus_report.get("assets", []):
        if asset.get("asset_id") == asset_id:
            return asset
    raise ValueError(f"asset_id {asset_id!r} is not in the corpus manifest")


def _person_splits(corpus_report):
    """Map every person_id in the registry to the splits it appears in."""
    mapping = {}
    for asset in corpus_report.get("assets", []):
        split = asset.get("split")
        for person_id in asset.get("person_ids", []):
            mapping.setdefault(person_id, set()).add(split)
    return mapping


def load_corpus_report(manifest_path, **validator_kwargs):
    """Validate a corpus_manifest v3 file, or raise.

    Fail-closed by design: an FA-02 portrait case may not be built from a
    registry that does not itself validate, because every downstream guarantee
    (person_crosses_split above all) rests on that validation having passed.
    """
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    kwargs = {"root": manifest_path.parent}
    kwargs.update(validator_kwargs)
    report = validate_corpus_manifest(manifest, **kwargs)
    if not report.get("valid"):
        codes = sorted({issue.get("code") for issue in report.get("errors", [])})
        raise ValueError(
            "Corpus manifest validation failed; refusing to source an FA-02 case: "
            + ", ".join(str(code) for code in codes)
        )
    return report


def build_portrait_case(
    corpus_report,
    annotation,
    *,
    case_id,
    arrays,
    arrays_base,
    corpus_manifest_path,
    session_id,
    owner_mapping_reference,
    smoothing_provenance,
    color_profile,
    inter_eye_distance_px,
    face_width_px,
    native_source_reference=None,
    purpose="development_pilot",
    previously_inspected=False,
    allow_locked_test=False,
):
    """Build one FA-02 ``schema_version=1`` portrait case from registry + annotation.

    ``corpus_report`` is the *validated* report from :func:`load_corpus_report`.
    ``annotation`` is a validated (or raw, it is re-validated here) support
    annotation. ``arrays`` is the case's NPZ path relative to ``arrays_base``,
    which is the directory the eventual FA-02 manifest will live in -- the
    harness resolves ``case["arrays"]`` against the manifest's own parent.

    The returned dict is a plain FA-02 case. It is deliberately *not* validated
    in isolation here: :func:`build_portrait_manifest` assembles cases and then
    hands the whole manifest to the harness's own ``validate_manifest``, so the
    existing purpose gates, checksum checks and crop/array-shape agreement are
    inherited rather than reimplemented (and cannot drift from them).

    Every failure raises ``ValueError``. There is no warn-and-continue path.
    """
    annotation = validate_support_annotation(annotation)

    if purpose not in PURPOSE_SPLITS:
        raise ValueError(
            "Portrait purpose must be one of " + ", ".join(sorted(PURPOSE_SPLITS))
        )

    asset = _corpus_asset(corpus_report, annotation["asset_id"])
    split = asset.get("split")
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Corpus asset {asset['asset_id']} has no valid v3 split")

    person_ids = list(asset.get("person_ids") or [])
    if not person_ids:
        raise ValueError(
            f"Corpus asset {asset['asset_id']} declares no person_ids; "
            "an FA-02 portrait case requires subject provenance"
        )
    if annotation["person_id"] not in person_ids:
        raise ValueError(
            "Annotation person_id is not among the corpus asset's person_ids"
        )

    # A development/tuning run must never draw from locked_test. The harness's
    # own development_pilot gate says the same thing about the *case's* split;
    # this says it about the *registry's* split for the same asset, which is a
    # separate assertion and the whole point of the bridge.
    allowed = PURPOSE_SPLITS[purpose]
    if split not in allowed:
        raise ValueError(
            f"Corpus split {split!r} is not permitted for purpose {purpose!r} "
            f"(allowed: {', '.join(allowed)})"
        )
    if split == "locked_test" and not allow_locked_test:
        raise ValueError(
            "Refusing locked_test material by default; pass allow_locked_test=True "
            "only for an owner-approved final comparison"
        )

    # The same subject must not already be spread across splits registry-side.
    # validate_corpus_manifest already rejects that globally, but re-assert it
    # for the specific subject being pulled so the failure names the person.
    person_splits = _person_splits(corpus_report)
    for person_id in person_ids:
        splits = person_splits.get(person_id, set())
        if len(splits) > 1:
            raise ValueError(
                f"person_id {person_id!r} appears in multiple corpus splits: "
                + ", ".join(sorted(str(value) for value in splits))
            )

    if annotation["annotation_source"] not in ACCEPTED_ANNOTATION_SOURCES:
        raise ValueError(
            "Support masks must carry an accepted provenance "
            f"({', '.join(ACCEPTED_ANNOTATION_SOURCES)}); "
            f"got {annotation['annotation_source']!r}. A raw detector or synthetic "
            "support is recordable but never externally accepted."
        )
    for key, entry in annotation["supports"].items():
        if entry["provenance"] not in ACCEPTED_ANNOTATION_SOURCES:
            raise ValueError(
                f"supports.{key}.provenance {entry['provenance']!r} is not an "
                "accepted support provenance"
            )

    arrays_base = Path(arrays_base)
    arrays_path = (arrays_base / arrays).resolve()
    if not arrays_path.is_file():
        raise ValueError(f"Case arrays file not found: {arrays_path}")
    saved = exp.load_arrays(arrays_path)
    if "X" not in saved:
        raise ValueError("Case arrays must contain an X canvas")
    height, width = saved["X"].shape[:2]

    roi_x, roi_y, roi_w, roi_h = annotation["native_roi_xywh"]
    if [roi_w, roi_h] != [int(width), int(height)]:
        raise ValueError(
            "Native ROI dimensions do not match the saved canvas shape "
            f"({roi_w}x{roi_h} vs {width}x{height}); a resize or registration "
            "mismatch is not permitted"
        )

    native_w, native_h = annotation["native_dimensions"]
    if roi_x + roi_w > native_w or roi_y + roi_h > native_h:
        raise ValueError("Native ROI is out of bounds for the declared native dimensions")

    # The native capture the X/S pair came from, stated relative to the FA-02
    # manifest directory (the harness resolves and checksums it there). The
    # annotation always carries this; an explicit caller argument may override
    # it when the FA-02 manifest lives in a different directory from the corpus
    # root. There is deliberately NO guess-from-the-corpus-path fallback: a
    # silently wrong source reference would checksum a file nobody reviewed.
    source_reference = _text(
        native_source_reference or annotation["native_source_reference"],
        "native_source_reference",
    )
    source_path = (arrays_base / source_reference).resolve()
    if not source_path.is_file():
        raise ValueError(f"Native source reference not found: {source_path}")

    case = {
        "id": _safe_id(case_id, "case_id"),
        "kind": "portrait",
        "split": split,
        "person_ids": sorted(person_ids),
        "session_id": _text(session_id, "session_id"),
        "arrays": str(arrays),
        "arrays_sha256": exp.digest(arrays_path),
        "inter_eye_distance_px": _positive_number(
            inter_eye_distance_px, "inter_eye_distance_px"
        ),
        "face_width_px": _positive_number(face_width_px, "face_width_px"),
        # crop_xywh is in NATIVE coordinates and its width/height must equal the
        # saved array shape -- the harness asserts exactly this. The annotation's
        # native_roi_xywh is the same rectangle stated independently by a human;
        # the equality above is what proves the two agree.
        "crop_xywh": [int(roi_x), int(roi_y), int(width), int(height)],
        "native_dimensions": [int(native_w), int(native_h)],
        "owner_mapping_reference": _text(owner_mapping_reference, "owner_mapping_reference"),
        "support_acceptance_reference": annotation["support_acceptance_reference"],
        "support_status": "externally_accepted",
        "masks_provenance": annotation["annotation_source"],
        "native_source_reference": source_reference,
        "source_sha256": exp.digest(source_path),
        "smoothing_provenance": _text(smoothing_provenance, "smoothing_provenance"),
        "color_profile": _text(color_profile, "color_profile"),
        "resampled": False,
        "previously_inspected": bool(previously_inspected),
        "rois": list(annotation["review_regions"]),
        "pores": [],
        "profiles": [],
        # Traceability back to the registry this case was sourced from, so an
        # auditor can re-derive the split/subject claim from primary evidence.
        "corpus_source": {
            "manifest_path": str(Path(corpus_manifest_path)),
            "asset_id": asset["asset_id"],
            "asset_sha256": asset.get("sha256"),
            "asset_split": split,
            "asset_person_ids": sorted(person_ids),
            "annotation_person_id": annotation["person_id"],
        },
        "annotation": annotation,
    }
    if annotation.get("reviewer"):
        case["reviewer"] = annotation["reviewer"]
    return case


def build_portrait_manifest(
    cases,
    *,
    manifest_dir,
    purpose="development_pilot",
    corpus_note="",
    parameter_selection="a priori research settings; not fitted to these cases",
    extra=None,
):
    """Assemble portrait cases into a validated FA-02 manifest.

    The final step is the harness's own ``validate_manifest``, so every
    pre-existing structural rule (unique filesystem-safe IDs, checksum match,
    crop-vs-array agreement, disjoint corrected/protected supports, subject and
    session split separation, purpose gating, ROI bounds) applies unchanged and
    cannot silently diverge from the analytic-control path.
    """
    if not cases:
        raise ValueError("A portrait manifest needs at least one case")
    if purpose not in PURPOSE_SPLITS:
        raise ValueError("Unknown portrait manifest purpose")
    if any(case.get("kind") != "portrait" for case in cases):
        raise ValueError(
            "This builder produces portrait-only manifests; analytic controls "
            "are locked to purpose=harness_validation by the harness"
        )
    manifest = {
        "schema_version": 1,
        "purpose": purpose,
        "corpus": corpus_note or "portrait cases sourced from a corpus_manifest v3 registry",
        "parameter_selection": parameter_selection,
        "cases": list(cases),
    }
    if extra:
        manifest.update(extra)
    exp.validate_manifest(manifest, Path(manifest_dir))
    return manifest


# --------------------------------------------------------------------------
# Promotion gates
# --------------------------------------------------------------------------


def check_promotion_gates(
    corpus_manifest_path,
    fa02_manifest_path,
    *,
    annotations=None,
    corpus_validator_kwargs=None,
):
    """Cross-check an FA-02 portrait manifest against the corpus v3 registry.

    Returns a JSON-safe report ``{"valid": bool, "violations": [...], ...}``.
    Unlike the builder (which raises on the first problem), this collects every
    violation so an owner sees the whole picture in one pass -- the same reason
    ``corpus_split_report.py`` prints all errors before exiting nonzero.

    Gates, in order:

    1. Corpus manifest itself validates (no person crosses splits, etc).
    2. Each FA-02 case's split equals the corpus split of its sourced asset,
       AND every one of the case's person_ids has that same split registry-side.
    3. No locked_test subject/asset is reachable from a development purpose.
    4. Every case has support_status == "externally_accepted" with a real
       acceptance reference and an accepted masks provenance.
    5. Source/S canvas correspondence: the declared source_sha256 matches the
       digest of the referenced native file (an assertion with provenance --
       it proves the named file is unchanged, not that S was derived from it).
    6. No resize/registration mismatch: crop width/height equal the saved array
       shape, and the annotation's native ROI agrees with both.
    7. Nothing in a case can hand different supports/gain/settings to different
       arms: no per-arm override key, exactly one arrays entry whose hash
       matches the file on disk, the shared support trio present, and all six
       ``exp.ARMS`` distinct. See :func:`check_arm_support_uniformity`.

    8. The annotation's declared native dimensions agree with the corpus
       asset's probed image dimensions *when both are known*. A reviewer who
       declares a ROI inside a frame that does not exist is caught here. This
       is skipped when the probe is unavailable (no Pillow) or when the native
       capture is legitimately a different file from the registered asset --
       in which case the owner must supply the correspondence, since no
       structural check can establish it.
    """
    violations = []

    def fail(code, **details):
        record = {"code": code}
        record.update(details)
        violations.append(record)

    corpus_manifest_path = Path(corpus_manifest_path).expanduser().resolve()
    fa02_manifest_path = Path(fa02_manifest_path).expanduser().resolve()

    # Gate 1 -----------------------------------------------------------------
    corpus_report = None
    try:
        corpus_report = load_corpus_report(
            corpus_manifest_path, **(corpus_validator_kwargs or {})
        )
    except (ValueError, OSError) as error:
        fail("corpus_manifest_invalid", detail=str(error))

    try:
        fa02_manifest = json.loads(fa02_manifest_path.read_text())
    except (ValueError, OSError) as error:
        fail("fa02_manifest_unreadable", detail=str(error))
        return {
            "valid": False,
            "violations": violations,
            "corpus_manifest": str(corpus_manifest_path),
            "fa02_manifest": str(fa02_manifest_path),
            "case_count": 0,
            "arms": list(exp.ARMS),
        }

    base = fa02_manifest_path.parent
    purpose = fa02_manifest.get("purpose")

    # The harness's own contract must hold before any cross-check means
    # anything: checksums, disjoint supports, crop/array agreement, its
    # internal purpose gates.
    try:
        exp.validate_manifest(fa02_manifest, base)
    except (ValueError, KeyError, OSError) as error:
        fail("fa02_manifest_invalid", detail=str(error))

    annotation_by_case = {}
    for case_id, annotation in (annotations or {}).items():
        try:
            annotation_by_case[case_id] = validate_support_annotation(annotation)
        except ValueError as error:
            fail("annotation_invalid", case_id=case_id, detail=str(error))

    cases = fa02_manifest.get("cases") or []
    person_splits = _person_splits(corpus_report) if corpus_report else {}

    for case in cases:
        if not isinstance(case, dict):
            fail("case_not_object")
            continue
        case_id = case.get("id")
        if case.get("kind") != "portrait":
            fail("non_portrait_case_in_gate", case_id=case_id, kind=case.get("kind"))
            continue

        case_split = case.get("split")
        source = case.get("corpus_source") or {}
        asset_id = source.get("asset_id")

        # Gate 2 -------------------------------------------------------------
        if not asset_id:
            fail("missing_corpus_source", case_id=case_id)
        elif corpus_report is not None:
            try:
                asset = _corpus_asset(corpus_report, asset_id)
            except ValueError as error:
                fail("corpus_asset_missing", case_id=case_id, detail=str(error))
                asset = None
            if asset is not None:
                if asset.get("split") != case_split:
                    fail(
                        "split_disagrees_with_corpus",
                        case_id=case_id,
                        case_split=case_split,
                        corpus_split=asset.get("split"),
                        asset_id=asset_id,
                    )
                if source.get("asset_sha256") and source["asset_sha256"] != asset.get("sha256"):
                    fail("corpus_asset_hash_changed", case_id=case_id, asset_id=asset_id)
                asset_people = set(asset.get("person_ids") or [])
                case_people = set(case.get("person_ids") or [])
                if not case_people:
                    fail("person_id_missing", case_id=case_id)
                if case_people - asset_people:
                    fail(
                        "person_not_in_corpus_asset",
                        case_id=case_id,
                        asset_id=asset_id,
                        unexpected=sorted(case_people - asset_people),
                    )
                for person_id in sorted(case_people):
                    registry_splits = person_splits.get(person_id, set())
                    if not registry_splits:
                        fail("person_not_in_corpus", case_id=case_id, person_id=person_id)
                    elif len(registry_splits) > 1:
                        fail(
                            "person_crosses_split",
                            case_id=case_id,
                            person_id=person_id,
                            splits=sorted(registry_splits),
                        )
                    elif case_split not in registry_splits:
                        fail(
                            "case_split_contradicts_person_split",
                            case_id=case_id,
                            person_id=person_id,
                            case_split=case_split,
                            corpus_split=sorted(registry_splits)[0],
                        )

                # Gate 3 ---------------------------------------------------
                if purpose == "development_pilot" and asset.get("split") == "locked_test":
                    fail("locked_test_used_for_development", case_id=case_id, asset_id=asset_id)

        if purpose == "development_pilot" and case_split not in DEVELOPMENT_SPLITS:
            fail("development_purpose_non_development_split", case_id=case_id, split=case_split)
        if purpose == "locked_comparison" and case_split != "locked_test":
            fail("locked_comparison_non_locked_split", case_id=case_id, split=case_split)

        # Gate 4 -------------------------------------------------------------
        if case.get("support_status") != "externally_accepted":
            fail("support_not_externally_accepted", case_id=case_id)
        if not case.get("support_acceptance_reference"):
            fail("missing_support_acceptance_reference", case_id=case_id)
        provenance = case.get("masks_provenance")
        if provenance not in ANNOTATION_SOURCES:
            fail("missing_masks_provenance", case_id=case_id, value=provenance)
        elif provenance not in ACCEPTED_ANNOTATION_SOURCES:
            fail("unaccepted_masks_provenance", case_id=case_id, value=provenance)

        # A supplied external record wins over the copy frozen into the case at
        # build time. NOTE for the workflow: after adding texture labels to a
        # standalone annotation_*.json, rebuild the case through
        # build_portrait_case so the embedded copy carries them too -- the two
        # can otherwise fork silently. That is deliberately not gated here
        # (a legitimate build-order artifact is not a provenance violation);
        # the readiness report surfaces it as a non-blocking observation.
        annotation = annotation_by_case.get(case_id) or case.get("annotation")
        if annotation is None:
            fail("missing_annotation_record", case_id=case_id)
        else:
            try:
                annotation = validate_support_annotation(annotation)
            except ValueError as error:
                fail("annotation_invalid", case_id=case_id, detail=str(error))
                annotation = None
        if annotation is not None:
            if annotation["annotation_source"] == "owner_approved" and not annotation.get("reviewer"):
                fail("owner_approved_without_reviewer", case_id=case_id)
            if annotation["asset_id"] != asset_id:
                fail("annotation_asset_mismatch", case_id=case_id, asset_id=asset_id)
            if annotation["person_id"] not in (case.get("person_ids") or []):
                fail("annotation_person_mismatch", case_id=case_id)

        # Gate 5 -------------------------------------------------------------
        reference = case.get("native_source_reference")
        declared_hash = case.get("source_sha256")
        if not reference or not declared_hash:
            fail("missing_native_source_reference", case_id=case_id)
        else:
            source_path = (base / reference).resolve()
            if not source_path.is_file():
                fail("native_source_missing", case_id=case_id, path=str(source_path))
            elif exp.digest(source_path) != declared_hash:
                fail("native_source_hash_mismatch", case_id=case_id)
        if case.get("resampled") is not False:
            fail("resampled_pixels_not_permitted", case_id=case_id)

        # Gate 6 -------------------------------------------------------------
        arrays = case.get("arrays")
        if not arrays:
            fail("missing_arrays", case_id=case_id)
            continue
        arrays_path = (base / arrays).resolve()
        if not arrays_path.is_file():
            fail("arrays_missing", case_id=case_id, path=str(arrays_path))
            continue
        if exp.digest(arrays_path) != case.get("arrays_sha256"):
            fail("arrays_hash_mismatch", case_id=case_id)
            continue
        saved = exp.load_arrays(arrays_path)
        if "X" not in saved or "S" not in saved:
            fail("arrays_missing_canvases", case_id=case_id)
            continue
        height, width = saved["X"].shape[:2]
        if saved["S"].shape != saved["X"].shape:
            fail("source_smoothed_shape_mismatch", case_id=case_id)
        crop = case.get("crop_xywh")
        if not isinstance(crop, list) or len(crop) != 4:
            fail("malformed_crop", case_id=case_id, value=str(crop))
        elif [crop[2], crop[3]] != [int(width), int(height)]:
            fail(
                "crop_array_shape_mismatch",
                case_id=case_id,
                crop=[crop[2], crop[3]],
                array=[int(width), int(height)],
            )
        native = case.get("native_dimensions")
        if isinstance(native, list) and len(native) == 2 and isinstance(crop, list) and len(crop) == 4:
            if crop[0] + crop[2] > native[0] or crop[1] + crop[3] > native[1]:
                fail("crop_outside_native_frame", case_id=case_id)

        # Gate 8 -------------------------------------------------------------
        # Only meaningful when the native capture IS the registered asset and
        # the probe actually ran; otherwise the owner must supply the
        # correspondence and no structural check can substitute for it.
        if corpus_report is not None and asset_id and isinstance(native, list) and len(native) == 2:
            try:
                probed = _corpus_asset(corpus_report, asset_id).get("metadata") or {}
            except ValueError:
                probed = {}
            probe_w, probe_h = probed.get("width"), probed.get("height")
            same_file = reference and Path(reference).name == Path(
                _corpus_asset(corpus_report, asset_id).get("path", "")
            ).name
            if same_file and probe_w and probe_h and [probe_w, probe_h] != list(native):
                fail(
                    "native_dimensions_disagree_with_corpus_probe",
                    case_id=case_id,
                    declared=list(native),
                    probed=[probe_w, probe_h],
                )
        if annotation is not None:
            if annotation["native_roi_xywh"][2:] != [int(width), int(height)]:
                fail(
                    "annotation_roi_array_shape_mismatch",
                    case_id=case_id,
                    roi=annotation["native_roi_xywh"][2:],
                    array=[int(width), int(height)],
                )
            if isinstance(crop, list) and len(crop) == 4 and annotation["native_roi_xywh"] != crop:
                fail("annotation_roi_crop_mismatch", case_id=case_id)
            for region in annotation["review_regions"]:
                x, y, w, h = region["xywh"]
                if x + w > width or y + h > height:
                    fail("review_region_outside_crop", case_id=case_id, region_id=region["id"])
            # The semantic categories live only in the annotation (deliberately
            # NOT folded into case["rois"]: validate_manifest does not enforce
            # ROI id uniqueness, so merging would let cross-category collisions
            # through). Bounds are therefore checked here, where the crop's
            # real array shape is finally known.
            for category in TEXTURE_REGION_CATEGORIES:
                for region in annotation.get(category) or []:
                    x, y, w, h = region["xywh"]
                    if x + w > width or y + h > height:
                        fail(
                            "texture_region_outside_crop",
                            case_id=case_id,
                            category=category,
                            region_id=region["id"],
                        )

    # Gate 7 -----------------------------------------------------------------
    # The runner computes eligibility once per case and passes the SAME map and
    # the SAME frozen config to every arm (see run()/worker()). Rather than
    # assume that, recompute the shared support map and confirm every arm in
    # exp.ARMS consumes exactly it, with one arrays entry per case.
    arm_propagation = check_arm_support_uniformity(fa02_manifest, base)
    if not arm_propagation["uniform"]:
        for detail in arm_propagation["violations"]:
            fail("arm_support_not_uniform", **detail)

    return {
        "valid": not violations,
        "violations": violations,
        "corpus_manifest": str(corpus_manifest_path),
        "fa02_manifest": str(fa02_manifest_path),
        "purpose": purpose,
        "case_count": len(cases),
        "arms": list(exp.ARMS),
        "arm_support_uniformity": arm_propagation,
    }


# Case keys that would let a hand-edited manifest hand one arm something the
# others do not get. The runner has no such feature -- which is exactly why an
# unnoticed key like this would be silently ignored rather than rejected.
_PER_ARM_OVERRIDE_KEYS = ("arm_overrides", "per_arm", "arms", "overrides_by_arm")


def check_arm_support_uniformity(fa02_manifest, base, config=None, config_sha256=None):
    """Check that no case can hand different supports/config to different arms.

    The runner computes geometry and eligibility ONCE per case and launches
    every arm from that single record with one frozen config lock, so
    uniformity holds by construction. The point of a gate is to make that an
    explicitly falsifiable check rather than an implicit consequence of how
    ``run()`` happens to be written, so this looks for the things that would
    actually break it:

    * a per-arm override key on the case (``arm_overrides``, a key named after
      one of ``exp.ARMS``, ...). The runner would silently ignore such a key,
      so a manifest carrying one is misleading evidence and is rejected.
    * more than one canvas source: exactly one ``arrays`` + one
      ``arrays_sha256``, and the recorded hash must match the file on disk.
    * a config that is not the single frozen one, when ``config_sha256`` from a
      lock file is supplied to compare against.
    * all six arms present and distinct in ``exp.ARMS``.

    The shared eligibility map is recomputed exactly as ``worker()`` does and
    reported as evidence (eligible pixel count + signature). It is evidence,
    not a per-arm comparison -- there is only one map to compare.

    Read-only; it does not run any arm's transform.
    """
    config = config or exp.DEFAULT_CONFIG
    violations = []
    per_case = []

    if len(set(exp.ARMS)) != len(exp.ARMS):
        violations.append({"case_id": None, "detail": "duplicate arm names in exp.ARMS"})
    if config_sha256 is not None and exp.canonical_hash(config) != config_sha256:
        violations.append(
            {"case_id": None, "detail": "config does not match the frozen lock hash"}
        )

    for case in fa02_manifest.get("cases") or []:
        case_id = case.get("id")

        # A per-arm override would be silently ignored by the runner. Refuse it
        # rather than let a manifest imply an asymmetry that never happened.
        for key in case:
            if key in _PER_ARM_OVERRIDE_KEYS or key in exp.ARMS:
                violations.append(
                    {"case_id": case_id, "detail": f"per-arm override key {key!r} is not permitted"}
                )

        if not case.get("arrays") or not case.get("arrays_sha256"):
            violations.append({"case_id": case_id, "detail": "case needs exactly one arrays entry"})
            continue
        arrays_path = (Path(base) / case["arrays"]).resolve()
        if not arrays_path.is_file():
            violations.append({"case_id": case_id, "detail": "arrays missing"})
            continue
        if exp.digest(arrays_path) != case["arrays_sha256"]:
            violations.append(
                {"case_id": case_id, "detail": "arrays hash differs from the shared canvas on disk"}
            )
            continue

        arr = exp.load_arrays(arrays_path)
        missing = [key for key in exp.MASKS if key not in arr]
        if missing:
            violations.append(
                {"case_id": case_id, "detail": f"shared supports missing: {', '.join(missing)}"}
            )
            continue
        geo = exp.geometry(case, config)
        geo["face_width_px"] = case["face_width_px"]
        eligibility = exp.map_external_supports(arr, geo, config)
        signature = exp.canonical_hash(
            {
                "arrays_sha256": case.get("arrays_sha256"),
                "eligibility_sha256": hashlib.sha256(
                    np.ascontiguousarray(eligibility, dtype=np.float32).tobytes()
                ).hexdigest(),
                "geometry": {
                    key: (list(value) if isinstance(value, list) else value)
                    for key, value in sorted(geo.items())
                },
                "gain": config["gain"],
                "config_sha256": exp.canonical_hash(config),
            }
        )
        per_case.append(
            {
                "case_id": case_id,
                "arms": list(exp.ARMS),
                "shared_support_signature": signature,
                "eligible_pixels": int(np.count_nonzero(eligibility)),
            }
        )
    return {"uniform": not violations, "violations": violations, "cases": per_case}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Cross-check FA-02 portrait cases against a corpus_manifest v3 registry."
    )
    parser.add_argument("corpus_manifest", type=Path, help="corpus_manifest v3 JSON")
    parser.add_argument("fa02_manifest", type=Path, help="FA-02 schema_version=1 portrait manifest")
    parser.add_argument(
        "--annotation",
        action="append",
        default=[],
        metavar="CASE_ID=PATH",
        help="Optional external annotation record for a case (repeatable)",
    )
    parser.add_argument("--json", type=Path, help="Write the full gate report as JSON")
    args = parser.parse_args(argv)

    annotations = {}
    for entry in args.annotation:
        if "=" not in entry:
            parser.error("--annotation expects CASE_ID=PATH")
        case_id, _, path = entry.partition("=")
        annotations[case_id] = json.loads(Path(path).read_text())

    report = check_promotion_gates(
        args.corpus_manifest, args.fa02_manifest, annotations=annotations
    )
    print(f"corpus_manifest: {report['corpus_manifest']}")
    print(f"fa02_manifest:   {report['fa02_manifest']}")
    print(f"purpose:         {report.get('purpose')}")
    print(f"cases:           {report['case_count']}")
    print(f"arms:            {len(report['arms'])} ({', '.join(report['arms'])})")
    print(f"valid:           {report['valid']}")
    if args.json:
        exp.write_json(args.json, report)
    if not report["valid"]:
        print(f"\nviolations ({len(report['violations'])}):", file=sys.stderr)
        for violation in report["violations"]:
            print(f"  {violation}", file=sys.stderr)
        return 1
    print("\nAll promotion gates passed. This certifies structure and provenance")
    print("references only -- it cannot prove an owner has never seen a subject.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

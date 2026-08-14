"""Deterministic validation for consented certification corpora.

The certification corpus is deliberately represented by explicit metadata.  A
filename is only a path to an asset; it is never a label.  Each asset must
carry a content hash, split, controlled tags/strata, and an explicit label
validation record (or be accepted by the caller's validation callback).

The module has no mandatory image dependency at import time.  When Pillow is
available, :func:`validate_corpus_manifest` probes decodability and dimensions;
callers can provide ``image_probe`` for a different decoder or for a test
fixture.  All returned reports contain only JSON-compatible values and are
sorted before being returned so they can be hashed or written as evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


CORPUS_MANIFEST_SCHEMA_VERSION = 2
ALLOWED_SPLITS = ("holdout", "pilot")

# These values are intentionally product vocabulary rather than filename
# aliases.  A project may provide a narrower vocabulary in ``vocabulary`` or
# through the validator arguments, but values outside the selected vocabulary
# are always rejected.
DEFAULT_STRATA_VOCABULARY: Dict[str, Tuple[str, ...]] = {
    "skin_tone": ("very_light", "light", "medium", "tan", "deep", "very_deep", "varied"),
    "lighting": (
        "daylight",
        "studio",
        "overcast",
        "flash",
        "mixed",
        "backlit",
        "high_key",
        "low_key",
    ),
    "face_scale": ("close", "medium", "small"),
    "pose": ("frontal", "three_quarter", "profile", "mixed"),
    "occlusion": ("none", "glasses", "facial_hair", "wig", "hands", "props", "mixed"),
    "texture": ("clean", "textured", "marks", "mixed"),
    "compression": ("clean", "compressed", "noisy", "mixed"),
    "exposure": ("normal", "high_key", "low_key", "mixed"),
    "faces": ("single", "multiple"),
}

# The four axes below are required for every asset by default.  The remaining
# dimensions are available for explicit coverage requirements and may be
# required by a project policy without changing this module.
DEFAULT_REQUIRED_STRATA = ("skin_tone", "lighting", "face_scale", "pose")

DEFAULT_TAG_VOCABULARY = (
    "glasses",
    "facial_hair",
    "wig",
    "hands",
    "props",
    "marks",
    "makeup",
    "texture",
    "compression",
    "noise",
    "high_key",
    "low_key",
    "multiple_faces",
    "group_portrait",
    "multi_image_shoot",
)

GROUP_TAGS = frozenset({"group_portrait", "multi_image_shoot", "multiple_faces"})
GROUP_STRATA = {
    ("faces", "multiple"),
}

ImageProbe = Callable[[Path], Mapping[str, Any]]
AssetValidator = Callable[[Mapping[str, Any]], Any]


class CorpusManifestError(ValueError):
    """Raised by :func:`require_valid_corpus_manifest` for invalid evidence."""

    def __init__(self, message: str, report: Mapping[str, Any]):
        super().__init__(message)
        self.report = dict(report)


def _normalise_label(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip().lower().replace("-", "_").replace(" ", "_")
    return text or None


def _normalise_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _normalise_vocabulary(
    vocabulary: Any,
    *,
    allowed_tags: Optional[Iterable[str]],
    allowed_strata: Optional[Mapping[str, Iterable[str]]],
) -> Tuple[Tuple[str, ...], Dict[str, Tuple[str, ...]], List[Dict[str, Any]]]:
    """Return sorted vocabularies and structural vocabulary errors."""
    errors: List[Dict[str, Any]] = []
    raw_tags: Any = DEFAULT_TAG_VOCABULARY
    raw_strata: Any = DEFAULT_STRATA_VOCABULARY

    if vocabulary is not None:
        if not isinstance(vocabulary, Mapping):
            errors.append({"code": "vocabulary_not_mapping", "field": "vocabulary"})
        else:
            if "tags" in vocabulary:
                raw_tags = vocabulary.get("tags")
            if "strata" in vocabulary:
                raw_strata = vocabulary.get("strata")

    if allowed_tags is not None:
        raw_tags = allowed_tags
    if allowed_strata is not None:
        raw_strata = allowed_strata

    tags: List[str] = []
    if isinstance(raw_tags, (str, bytes)) or not isinstance(raw_tags, Iterable):
        errors.append({"code": "tag_vocabulary_invalid", "field": "vocabulary.tags"})
    else:
        for value in raw_tags:
            label = _normalise_label(value)
            if label is None:
                errors.append({"code": "tag_vocabulary_invalid", "field": "vocabulary.tags"})
                continue
            tags.append(label)

    strata: Dict[str, Tuple[str, ...]] = {}
    if not isinstance(raw_strata, Mapping):
        errors.append({"code": "strata_vocabulary_invalid", "field": "vocabulary.strata"})
    else:
        for raw_dimension, raw_values in raw_strata.items():
            dimension = _normalise_label(raw_dimension)
            if dimension is None or isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Iterable):
                errors.append({"code": "strata_vocabulary_invalid", "field": "vocabulary.strata"})
                continue
            values: List[str] = []
            for value in raw_values:
                label = _normalise_label(value)
                if label is None:
                    errors.append({"code": "strata_vocabulary_invalid", "field": f"vocabulary.strata.{dimension or '?'}"})
                    continue
                values.append(label)
            if dimension is not None:
                strata[dimension] = tuple(sorted(set(values)))

    return tuple(sorted(set(tags))), dict(sorted(strata.items())), errors


def _issue(
    code: str,
    *,
    asset_id: Optional[str] = None,
    field: Optional[str] = None,
    value: Optional[Any] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"code": code}
    if asset_id:
        result["asset_id"] = asset_id
    if field:
        result["field"] = field
    if value is not None:
        # Error values are intentionally reduced to JSON-safe scalars.  Do not
        # leak arbitrary objects or callback exception text into a manifest.
        if isinstance(value, (str, int, float, bool)):
            result["value"] = value
        else:
            result["value"] = type(value).__name__
    return result


def _sort_issues(issues: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        dict(item)
        for item in sorted(
            (dict(issue) for issue in issues),
            key=lambda item: (
                str(item.get("code", "")),
                str(item.get("asset_id", "")),
                str(item.get("field", "")),
                json.dumps(item.get("value"), sort_keys=True, default=str),
            ),
        )
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(raw_path: str, root: Optional[Path]) -> Tuple[Optional[Path], str, Optional[str]]:
    source = Path(raw_path).expanduser()
    if root is None:
        resolved = source.resolve()
        canonical = source.as_posix()
        return resolved, canonical, None

    resolved_root = root.expanduser().resolve()
    candidate = source if source.is_absolute() else resolved_root / source
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError:
        return None, source.as_posix(), "asset_path_outside_root"
    return resolved, relative.as_posix(), None


def _default_image_probe(path: Path) -> Dict[str, Any]:
    """Probe an image lazily; missing Pillow is a non-fatal unavailable probe."""
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return {
            "probe_status": "unavailable",
            "decodable": None,
            "reason": "pillow_unavailable",
        }

    try:
        # verify() detects truncated/corrupt data without retaining a decoder
        # object; reopen for dimensions because verify() invalidates the image.
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return {
                "probe_status": "ok",
                "decodable": True,
                "width": int(image.width),
                "height": int(image.height),
                "format": str(image.format or "").upper() or None,
                "mode": str(image.mode or "") or None,
            }
    except Exception:
        return {
            "probe_status": "failed",
            "decodable": False,
            "reason": "image_decode_failed",
        }


def _json_probe_result(raw: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    if not isinstance(raw, Mapping):
        return {"probe_status": "unavailable", "decodable": None, "reason": "invalid_probe_result"}, "image_probe_invalid"

    result: Dict[str, Any] = {}
    status = _normalise_label(raw.get("probe_status")) or "ok"
    result["probe_status"] = status

    decodable = raw.get("decodable")
    if decodable is not None and not isinstance(decodable, bool):
        return result, "image_probe_invalid"
    result["decodable"] = decodable

    for key in ("width", "height"):
        value = raw.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            return result, "image_probe_invalid"
        result[key] = value

    for key in ("format", "mode", "reason"):
        value = raw.get(key)
        if value is not None and not isinstance(value, str):
            return result, "image_probe_invalid"
        if value:
            result[key] = value
    return result, None


def _metadata_for_asset(
    path: Path,
    declared: Any,
    image_probe: Optional[ImageProbe],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []
    try:
        raw_probe = image_probe(path) if image_probe is not None else _default_image_probe(path)
    except Exception:
        raw_probe = {"probe_status": "unavailable", "decodable": None, "reason": "image_probe_failed"}
    observed, probe_error = _json_probe_result(raw_probe)
    if probe_error:
        issues.append(_issue(probe_error, field="metadata"))

    if isinstance(declared, Mapping):
        declared_safe: Dict[str, Any] = {}
        for key in ("decodable", "width", "height", "format", "mode"):
            if key in declared:
                value = declared.get(key)
                if key == "decodable" and value is not None and not isinstance(value, bool):
                    issues.append(_issue("declared_metadata_invalid", field=f"metadata.{key}"))
                elif key in {"width", "height"} and value is not None and (
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                ):
                    issues.append(_issue("declared_metadata_invalid", field=f"metadata.{key}"))
                elif key in {"format", "mode"} and value is not None and not isinstance(value, str):
                    issues.append(_issue("declared_metadata_invalid", field=f"metadata.{key}"))
                else:
                    declared_safe[key] = value
        if declared_safe:
            observed["declared"] = declared_safe
        if observed.get("decodable") is not None and declared_safe.get("decodable") is not None:
            if observed["decodable"] != declared_safe["decodable"]:
                issues.append(_issue("declared_metadata_mismatch", field="metadata.decodable"))
        for key in ("width", "height"):
            if observed.get(key) is not None and declared_safe.get(key) is not None:
                if observed[key] != declared_safe[key]:
                    issues.append(_issue("declared_metadata_mismatch", field=f"metadata.{key}"))

    return observed, issues


def _normalise_validation_fields(
    raw: Any,
    *,
    callback: Optional[AssetValidator],
    candidate: Mapping[str, Any],
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []
    callback_result: Any = None
    callback_used = callback is not None
    if callback is not None:
        try:
            callback_result = callback(candidate)
        except Exception:
            issues.append(_issue("label_validator_failed", field="label_validation"))
        else:
            if callback_result is False or (
                isinstance(callback_result, Mapping) and callback_result.get("valid") is False
            ):
                issues.append(_issue("label_validation_rejected", field="label_validation"))
            elif callback_result not in (True, None) and not isinstance(callback_result, Mapping):
                issues.append(_issue("label_validator_invalid_result", field="label_validation"))

    source = raw
    if source is None and isinstance(callback_result, Mapping):
        source = callback_result
    if source is None and callback_used and not issues:
        return {"status": "validated", "method": "callback"}, issues
    if not isinstance(source, Mapping):
        if not issues:
            issues.append(_issue("label_validation_required", field="label_validation"))
        return None, issues

    status = _normalise_label(source.get("status"))
    validated = source.get("validated") is True or (
        callback_used and source is callback_result and source.get("valid") is True
    )
    if status != "validated" and not validated:
        issues.append(_issue("label_validation_required", field="label_validation.status"))
    method = _normalise_text(source.get("method") or source.get("source") or source.get("validator"))
    if method is None:
        issues.append(_issue("label_validation_method_required", field="label_validation.method"))
    if issues:
        return None, issues

    result: Dict[str, Any] = {"status": "validated", "method": method}
    reference = _normalise_text(source.get("reference"))
    if reference is not None:
        result["reference"] = reference
    return result, issues


def _coverage_for_assets(assets: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    tags: Dict[str, List[str]] = {}
    strata: Dict[str, Dict[str, List[str]]] = {}
    for asset in assets:
        asset_id = str(asset.get("asset_id"))
        for tag in asset.get("tags", []):
            tags.setdefault(str(tag), []).append(asset_id)
        for dimension, value in (asset.get("strata") or {}).items():
            strata.setdefault(str(dimension), {}).setdefault(str(value), []).append(asset_id)
    return {
        "tags": {key: sorted(values) for key, values in sorted(tags.items())},
        "strata": {
            dimension: {value: sorted(ids) for value, ids in sorted(values.items())}
            for dimension, values in sorted(strata.items())
        },
    }


def _coverage_requirements(
    requirements: Any,
    coverage: Mapping[str, Any],
    allowed_tags: Sequence[str],
    allowed_strata: Mapping[str, Sequence[str]],
) -> List[Dict[str, Any]]:
    if requirements is None:
        return []
    if not isinstance(requirements, Mapping):
        return [_issue("coverage_requirements_invalid", field="coverage_requirements")]

    issues: List[Dict[str, Any]] = []
    required_tags = requirements.get("tags", [])
    if isinstance(required_tags, (str, bytes)) or not isinstance(required_tags, Iterable):
        issues.append(_issue("coverage_requirements_invalid", field="coverage_requirements.tags"))
    else:
        for raw_tag in required_tags:
            tag = _normalise_label(raw_tag)
            if tag is None or tag not in allowed_tags:
                issues.append(_issue("unknown_coverage_tag", field="coverage_requirements.tags", value=raw_tag))
            elif tag not in (coverage.get("tags") or {}):
                issues.append(_issue("coverage_missing", field=f"tags.{tag}"))

    required_strata = requirements.get("strata", {})
    if not isinstance(required_strata, Mapping):
        issues.append(_issue("coverage_requirements_invalid", field="coverage_requirements.strata"))
    else:
        for raw_dimension, raw_values in required_strata.items():
            dimension = _normalise_label(raw_dimension)
            if dimension is None or dimension not in allowed_strata:
                issues.append(_issue("unknown_coverage_stratum", field="coverage_requirements.strata", value=raw_dimension))
                continue
            if isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Iterable):
                issues.append(_issue("coverage_requirements_invalid", field=f"coverage_requirements.strata.{dimension}"))
                continue
            observed = (coverage.get("strata") or {}).get(dimension, {})
            for raw_value in raw_values:
                value = _normalise_label(raw_value)
                if value is None or value not in allowed_strata[dimension]:
                    issues.append(_issue("unknown_coverage_stratum", field=f"coverage_requirements.strata.{dimension}", value=raw_value))
                elif value not in observed:
                    issues.append(_issue("coverage_missing", field=f"strata.{dimension}.{value}"))
    return issues


def _canonical_coverage_requirements(requirements: Any) -> Dict[str, Any]:
    """Reduce optional coverage policy to a stable, JSON-safe shape."""
    result: Dict[str, Any] = {"tags": [], "strata": {}}
    if not isinstance(requirements, Mapping):
        return result

    raw_tags = requirements.get("tags", [])
    if not isinstance(raw_tags, (str, bytes)) and isinstance(raw_tags, Iterable):
        result["tags"] = sorted({label for label in (_normalise_label(value) for value in raw_tags) if label})

    raw_strata = requirements.get("strata", {})
    if isinstance(raw_strata, Mapping):
        normalized_strata: Dict[str, List[str]] = {}
        for raw_dimension, raw_values in raw_strata.items():
            dimension = _normalise_label(raw_dimension)
            if dimension is None or isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Iterable):
                continue
            values = sorted({label for label in (_normalise_label(value) for value in raw_values) if label})
            normalized_strata[dimension] = values
        result["strata"] = {dimension: normalized_strata[dimension] for dimension in sorted(normalized_strata)}
    return result


def validate_corpus_manifest(
    manifest: Mapping[str, Any],
    *,
    root: Optional[os.PathLike[str] | str] = None,
    allowed_tags: Optional[Iterable[str]] = None,
    allowed_strata: Optional[Mapping[str, Iterable[str]]] = None,
    required_strata: Optional[Iterable[str]] = None,
    asset_validator: Optional[AssetValidator] = None,
    image_probe: Optional[ImageProbe] = None,
) -> Dict[str, Any]:
    """Validate and normalize a v2 corpus manifest.

    ``root`` is the directory against which relative asset paths are resolved.
    If omitted, relative paths are resolved from the current working directory.
    ``asset_validator`` receives a JSON-safe candidate containing the explicit
    labels and may return ``True`` or ``{"valid": True, ...}``; without it,
    every asset needs ``label_validation`` (or ``validation``) with a validated
    status and a method.  ``image_probe`` can return ``decodable``, ``width``,
    ``height``, ``format`` and ``mode``.  Pillow is used lazily otherwise.

    The returned report has a stable shape even for malformed input.  Its
    ``canonical_manifest`` is suitable for deterministic JSON serialization;
    ``valid`` is true only when all errors are absent.
    """
    errors: List[Dict[str, Any]] = []
    if not isinstance(manifest, Mapping):
        return {
            "schema_version": CORPUS_MANIFEST_SCHEMA_VERSION,
            "valid": False,
            "errors": [_issue("manifest_not_mapping")],
            "warnings": [],
            "canonical_manifest": None,
        }

    schema_version = manifest.get("schema_version")
    if schema_version != CORPUS_MANIFEST_SCHEMA_VERSION:
        errors.append(_issue("schema_version_invalid", field="schema_version", value=schema_version))

    consent_reference = _normalise_text(manifest.get("consent_reference"))
    if consent_reference is None:
        errors.append(_issue("consent_reference_required", field="consent_reference"))

    tags, strata_vocabulary, vocabulary_errors = _normalise_vocabulary(
        manifest.get("vocabulary"),
        allowed_tags=allowed_tags,
        allowed_strata=allowed_strata,
    )
    errors.extend(vocabulary_errors)

    requested_required = required_strata if required_strata is not None else manifest.get("required_strata", DEFAULT_REQUIRED_STRATA)
    if isinstance(requested_required, (str, bytes)) or not isinstance(requested_required, Iterable):
        errors.append(_issue("required_strata_invalid", field="required_strata"))
        required_dimensions: Tuple[str, ...] = tuple(DEFAULT_REQUIRED_STRATA)
    else:
        required_dimensions = tuple(sorted({_normalise_label(value) for value in requested_required if _normalise_label(value)}))
    for dimension in required_dimensions:
        if dimension not in strata_vocabulary:
            errors.append(_issue("required_stratum_not_in_vocabulary", field="required_strata", value=dimension))

    raw_assets = manifest.get("assets")
    if isinstance(raw_assets, (str, bytes)) or not isinstance(raw_assets, Sequence):
        errors.append(_issue("assets_required", field="assets"))
        raw_assets = []
    if not raw_assets:
        errors.append(_issue("assets_required", field="assets"))

    root_path = Path(root).expanduser() if root is not None else None
    canonical_assets: List[Dict[str, Any]] = []
    seen_ids: Dict[str, str] = {}
    seen_hashes: Dict[str, str] = {}
    seen_actual_hashes: Dict[str, str] = {}
    seen_paths: Dict[str, str] = {}
    group_splits: Dict[str, set[str]] = {}

    for raw_asset in raw_assets:
        if not isinstance(raw_asset, Mapping):
            errors.append(_issue("asset_not_mapping", field="assets"))
            continue

        raw_id = _normalise_text(raw_asset.get("asset_id"))
        asset_id = raw_id or ""
        if raw_id is None:
            errors.append(_issue("asset_id_required", field="asset_id"))
        elif raw_id in seen_ids:
            errors.append(_issue("duplicate_asset_id", asset_id=raw_id, field="asset_id"))
        else:
            seen_ids[raw_id] = raw_id

        raw_path = raw_asset.get("path")
        path_text = _normalise_text(raw_path)
        resolved_path: Optional[Path] = None
        canonical_path = path_text or ""
        if path_text is None:
            errors.append(_issue("asset_path_required", asset_id=asset_id or None, field="path"))
        elif "\x00" in path_text:
            errors.append(_issue("asset_path_invalid", asset_id=asset_id or None, field="path"))
        else:
            resolved_path, canonical_path, path_error = _resolve_path(path_text, root_path)
            if path_error:
                errors.append(_issue(path_error, asset_id=asset_id or None, field="path"))
            elif resolved_path is None or not resolved_path.is_file():
                errors.append(_issue("asset_file_missing", asset_id=asset_id or None, field="path"))
            elif canonical_path in seen_paths:
                errors.append(_issue("duplicate_asset_path", asset_id=asset_id or None, field="path"))
            else:
                seen_paths[canonical_path] = asset_id

        raw_hash = _normalise_label(raw_asset.get("sha256"))
        if raw_hash is None or len(raw_hash) != 64 or any(character not in "0123456789abcdef" for character in raw_hash):
            errors.append(_issue("sha256_required", asset_id=asset_id or None, field="sha256"))
            raw_hash = ""
        elif raw_hash in seen_hashes:
            errors.append(_issue("duplicate_content_hash", asset_id=asset_id or None, field="sha256"))
        else:
            seen_hashes[raw_hash] = asset_id

        actual_hash: Optional[str] = None
        if resolved_path is not None and resolved_path.is_file():
            try:
                actual_hash = _sha256_file(resolved_path)
            except (OSError, ValueError):
                errors.append(_issue("asset_hash_unreadable", asset_id=asset_id or None, field="sha256"))
        if raw_hash and actual_hash is not None and raw_hash != actual_hash:
            errors.append(_issue("sha256_mismatch", asset_id=asset_id or None, field="sha256"))
        if actual_hash is not None:
            if actual_hash in seen_actual_hashes:
                errors.append(_issue("duplicate_content_hash", asset_id=asset_id or None, field="sha256"))
            else:
                seen_actual_hashes[actual_hash] = asset_id

        split = _normalise_label(raw_asset.get("split"))
        if split not in ALLOWED_SPLITS:
            errors.append(_issue("split_invalid", asset_id=asset_id or None, field="split", value=raw_asset.get("split")))
            split = ""

        raw_tags = raw_asset.get("tags")
        asset_tags: List[str] = []
        if isinstance(raw_tags, (str, bytes)) or not isinstance(raw_tags, Sequence):
            errors.append(_issue("tags_required", asset_id=asset_id or None, field="tags"))
        else:
            for raw_tag in raw_tags:
                tag = _normalise_label(raw_tag)
                if tag is None:
                    errors.append(_issue("tag_invalid", asset_id=asset_id or None, field="tags"))
                elif tag not in tags:
                    errors.append(_issue("unknown_tag", asset_id=asset_id or None, field="tags", value=raw_tag))
                elif tag in asset_tags:
                    errors.append(_issue("duplicate_tag", asset_id=asset_id or None, field="tags", value=tag))
                else:
                    asset_tags.append(tag)
        asset_tags = sorted(asset_tags)

        raw_strata = raw_asset.get("strata")
        asset_strata: Dict[str, str] = {}
        if not isinstance(raw_strata, Mapping):
            errors.append(_issue("strata_required", asset_id=asset_id or None, field="strata"))
        else:
            for raw_dimension, raw_value in raw_strata.items():
                dimension = _normalise_label(raw_dimension)
                value = _normalise_label(raw_value)
                if dimension is None or dimension not in strata_vocabulary:
                    errors.append(_issue("unknown_stratum", asset_id=asset_id or None, field="strata", value=raw_dimension))
                elif value is None or value not in strata_vocabulary[dimension]:
                    errors.append(_issue("unknown_stratum_value", asset_id=asset_id or None, field=f"strata.{dimension}", value=raw_value))
                elif dimension in asset_strata:
                    errors.append(_issue("duplicate_stratum", asset_id=asset_id or None, field=f"strata.{dimension}"))
                else:
                    asset_strata[dimension] = value
            for dimension in required_dimensions:
                if dimension not in asset_strata:
                    errors.append(_issue("required_stratum_missing", asset_id=asset_id or None, field=f"strata.{dimension}"))

        group_applicable = raw_asset.get("group_applicable")
        if group_applicable is not None and not isinstance(group_applicable, bool):
            errors.append(_issue("group_applicable_invalid", asset_id=asset_id or None, field="group_applicable"))
            group_applicable = False
        group_id = _normalise_text(raw_asset.get("group_id"))
        group_by_tag = bool(set(asset_tags) & GROUP_TAGS)
        group_by_stratum = any((dimension, value) in GROUP_STRATA for dimension, value in asset_strata.items())
        group_required = bool(group_applicable) or group_by_tag or group_by_stratum
        if group_required and group_id is None:
            errors.append(_issue("group_id_required", asset_id=asset_id or None, field="group_id"))
        if group_id is not None and split in ALLOWED_SPLITS:
            group_splits.setdefault(group_id, set()).add(split)

        declared_validation = raw_asset.get("label_validation")
        if declared_validation is None:
            declared_validation = raw_asset.get("validation")

        metadata, metadata_errors = _metadata_for_asset(
            resolved_path,
            raw_asset.get("metadata"),
            image_probe,
        ) if resolved_path is not None and resolved_path.is_file() else ({"probe_status": "unavailable", "decodable": None, "reason": "asset_unavailable"}, [])
        for issue in metadata_errors:
            issue["asset_id"] = asset_id or issue.get("asset_id")
        errors.extend(metadata_errors)
        if metadata.get("decodable") is False:
            errors.append(_issue("asset_not_decodable", asset_id=asset_id or None, field="metadata.decodable"))

        candidate: Dict[str, Any] = {
            "asset_id": asset_id,
            "path": canonical_path,
            "sha256": raw_hash,
            "split": split,
            "tags": asset_tags,
            "strata": dict(sorted(asset_strata.items())),
            "metadata": metadata,
        }
        if group_id is not None:
            candidate["group_id"] = group_id
        if group_applicable is not None:
            candidate["group_applicable"] = bool(group_applicable)

        validation, validation_errors = _normalise_validation_fields(
            declared_validation,
            callback=asset_validator,
            candidate=candidate,
        )
        for issue in validation_errors:
            issue["asset_id"] = asset_id or issue.get("asset_id")
        errors.extend(validation_errors)
        if validation is not None:
            candidate["label_validation"] = validation
        canonical_assets.append(candidate)

    for group_id, splits in sorted(group_splits.items()):
        if len(splits) > 1:
            errors.append(_issue("group_crosses_split", field="group_id", value=group_id))

    canonical_assets.sort(key=lambda item: (str(item.get("asset_id", "")), str(item.get("path", ""))))
    split_counts = {split: sum(1 for asset in canonical_assets if asset.get("split") == split) for split in ALLOWED_SPLITS}
    if split_counts["pilot"] == 0:
        errors.append(_issue("pilot_split_required", field="assets.split"))
    if split_counts["holdout"] == 0:
        errors.append(_issue("holdout_split_required", field="assets.split"))

    coverage = _coverage_for_assets(canonical_assets)
    errors.extend(_coverage_requirements(manifest.get("coverage_requirements"), coverage, tags, strata_vocabulary))

    canonical_manifest: Dict[str, Any] = {
        "schema_version": CORPUS_MANIFEST_SCHEMA_VERSION,
        "consent_reference": consent_reference,
        "vocabulary": {
            "tags": list(tags),
            "strata": {dimension: list(values) for dimension, values in sorted(strata_vocabulary.items())},
        },
        "required_strata": list(required_dimensions),
        "assets": canonical_assets,
    }
    if "coverage_requirements" in manifest:
        canonical_manifest["coverage_requirements"] = _canonical_coverage_requirements(
            manifest.get("coverage_requirements")
        )

    # Metadata probes can be unavailable on a minimal installation.  This is
    # useful diagnostic evidence, but does not turn a real path/hash/label
    # manifest into a false failure merely because Pillow is absent.
    warnings = [
        _issue("image_probe_unavailable", asset_id=str(asset.get("asset_id")), field="metadata")
        for asset in canonical_assets
        if asset.get("metadata", {}).get("probe_status") == "unavailable"
    ]
    report: Dict[str, Any] = {
        "schema_version": CORPUS_MANIFEST_SCHEMA_VERSION,
        "valid": not errors,
        "consent_reference": consent_reference,
        "asset_count": len(canonical_assets),
        "split_counts": split_counts,
        "coverage": coverage,
        "assets": canonical_assets,
        "errors": _sort_issues(errors),
        "warnings": _sort_issues(warnings),
        "canonical_manifest": canonical_manifest,
    }
    return report


def validate_corpus_coverage(
    manifest_or_report: Mapping[str, Any],
    *,
    required_tags: Iterable[str] = (),
    required_strata: Optional[Mapping[str, Iterable[str]]] = None,
) -> Dict[str, Any]:
    """Check explicit tag/stratum coverage without looking at filenames."""
    if "coverage" in manifest_or_report and isinstance(manifest_or_report.get("coverage"), Mapping):
        coverage = manifest_or_report.get("coverage")
    else:
        assets = manifest_or_report.get("assets", [])
        coverage = _coverage_for_assets(assets if isinstance(assets, Sequence) else [])

    errors: List[Dict[str, Any]] = []
    for raw_tag in required_tags:
        tag = _normalise_label(raw_tag)
        if tag is None or tag not in (coverage.get("tags") or {}):
            errors.append(_issue("coverage_missing", field=f"tags.{tag or raw_tag}"))
    for raw_dimension, values in (required_strata or {}).items():
        dimension = _normalise_label(raw_dimension)
        observed = (coverage.get("strata") or {}).get(dimension or "", {})
        if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
            errors.append(_issue("coverage_requirements_invalid", field=f"strata.{dimension or raw_dimension}"))
            continue
        for raw_value in values:
            value = _normalise_label(raw_value)
            if value is None or value not in observed:
                errors.append(_issue("coverage_missing", field=f"strata.{dimension or raw_dimension}.{value or raw_value}"))
    return {
        "valid": not errors,
        "coverage": coverage,
        "errors": _sort_issues(errors),
    }


def require_valid_corpus_manifest(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Return a report or raise :class:`CorpusManifestError` on failure."""
    report = validate_corpus_manifest(*args, **kwargs)
    if not report.get("valid"):
        raise CorpusManifestError("corpus manifest validation failed", report)
    return report


def write_corpus_manifest(
    path: os.PathLike[str] | str,
    manifest: Mapping[str, Any],
    **validator_kwargs: Any,
) -> Dict[str, Any]:
    """Validate and atomically write the canonical JSON manifest."""
    report = require_valid_corpus_manifest(manifest, **validator_kwargs)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    payload = json.dumps(report["canonical_manifest"], indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temporary.write_text(payload, encoding="utf-8")
    os.replace(str(temporary), str(target))
    return report


__all__ = [
    "ALLOWED_SPLITS",
    "CORPUS_MANIFEST_SCHEMA_VERSION",
    "CorpusManifestError",
    "DEFAULT_REQUIRED_STRATA",
    "DEFAULT_STRATA_VOCABULARY",
    "DEFAULT_TAG_VOCABULARY",
    "require_valid_corpus_manifest",
    "validate_corpus_coverage",
    "validate_corpus_manifest",
    "write_corpus_manifest",
]

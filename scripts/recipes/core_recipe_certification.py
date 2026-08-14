#!/usr/bin/env python3
"""Run the release Core-recipe matrix and create a human-review worksheet.

Usage intentionally requires named corpus cases, for example::

    python scripts/recipes/core_recipe_certification.py \
      --case skin_dark=/path/dark.jpg \
      --case glasses=/path/glasses.jpg \
      --output /tmp/core-cert

The default is face-aware. ``--global-only`` is available only for a
diagnostic smoke run and is written into every manifest; it can never be
treated as face-retouch certification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retouch.core_recipes import CORE_RECIPE_NAMES, CORE_RECIPE_REVIEW_DIMENSIONS
from retouch.certification import (
    core_corpus_coverage,
    certification_fields,
    core_case_automatic_pass,
    core_matrix_final_certified,
)
from retouch.recipes import RECIPES


V2_SCHEMA_VERSION = 2
V2_CONTRACT_VERSION = "certification-evidence-v2"
_SHA256_LENGTH = 64
_V2_REQUIRED_DIMENSIONS = (
    "skin_tone",
    "lighting",
    "glasses",
    "wig",
    "hands_on_face",
    "group_portrait",
)


def _json_safe(value: Any) -> Any:
    """Return a JSON-safe copy without importing optional image dependencies."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> Optional[str]:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 16), b""):
                digest.update(chunk)
    except (OSError, IOError):
        return None
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value))


def _optional_module(module_name: str) -> Any:
    """Load a v2 extension module without making it a v1 dependency."""
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _call_hook(function: Any, candidates: Sequence[Tuple[Any, ...]]) -> Tuple[Any, Optional[str]]:
    """Call a developing extension hook using the first compatible signature."""
    if not callable(function):
        return None, None
    last_type_error: Optional[str] = None
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        signature = None
    for arguments in candidates:
        if signature is not None:
            try:
                signature.bind(*arguments)
            except TypeError:
                continue
        try:
            return function(*arguments), None
        except TypeError as exc:
            last_type_error = str(exc)
            if signature is not None:
                return None, last_type_error
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
    return None, last_type_error


def _module_hook(module: Any, names: Sequence[str]) -> Any:
    if module is None:
        return None
    for name in names:
        function = getattr(module, name, None)
        if callable(function):
            return function
    return None


def _first_value(objects: Iterable[Any], names: Sequence[str], default: Any = None) -> Any:
    for item in objects:
        if not isinstance(item, Mapping):
            continue
        for name in names:
            if name in item:
                return item[name]
    return default


def _first_mapping(objects: Iterable[Any], names: Sequence[str]) -> Dict[str, Any]:
    value = _first_value(objects, names)
    return dict(value) if isinstance(value, Mapping) else {}


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "ok", "pass", "passed", "ready"}
    return bool(value)


def _normalise_tag(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text.startswith("skin_tone") or text.startswith("tone"):
        return "skin_tone"
    if text.startswith("lighting") or text.startswith("light") or text in {"lowlight", "venue"}:
        return "lighting"
    if text.startswith("glasses") or text.startswith("eyeglass") or text.startswith("spectacle"):
        return "glasses"
    if text.startswith("wig") or text in {"cosplay", "hair"}:
        return "wig"
    if text.startswith("hands_on_face") or text.startswith("hand_face") or text.startswith("handsonface"):
        return "hands_on_face"
    if text.startswith("group") or text.startswith("family") or text.startswith("multi_face") or text.startswith("multiface"):
        return "group_portrait"
    return text


def _asset_tags(asset: Mapping[str, Any]) -> List[str]:
    raw = asset.get("validated_tags", asset.get("tags", asset.get("strata", [])))
    if isinstance(raw, Mapping):
        raw = list(raw.keys())
    if isinstance(raw, str):
        raw = [raw]
    return sorted({_normalise_tag(value) for value in (raw or []) if str(value).strip()})


def _manifest_assets(payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    candidates: List[Any] = [payload]
    nested = payload.get("corpus")
    if isinstance(nested, Mapping):
        candidates.append(nested)
    for candidate in candidates:
        for key in ("assets", "images", "cases", "entries"):
            values = candidate.get(key)
            if isinstance(values, list):
                return [item for item in values if isinstance(item, Mapping)]
    return []


def _asset_identity(asset: Mapping[str, Any]) -> List[str]:
    values: List[str] = []
    for key in ("case", "case_name", "name", "id", "asset_id", "assetId"):
        value = str(asset.get(key) or "").strip()
        if value:
            values.append(value)
    return values


def _asset_path(asset: Mapping[str, Any], manifest_path: Path) -> Optional[Path]:
    raw = asset.get("path", asset.get("input", asset.get("file", asset.get("filename"))))
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    return (manifest_path.parent / path).resolve() if not path.is_absolute() else path.resolve()


def _asset_expected_faces(asset: Mapping[str, Any]) -> Optional[int]:
    value = _first_value(
        (asset,),
        ("expected_face_count", "expected_faces", "face_count", "faces_expected"),
    )
    return _as_int(value)


def _load_and_validate_corpus_manifest(
    manifest_path: Path,
    cases: Sequence[Tuple[str, Path]],
) -> Dict[str, Any]:
    """Load a private v2 corpus manifest and reject unverifiable coverage."""
    raw: Dict[str, Any]
    try:
        decoded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "valid": False,
            "complete": False,
            "errors": [f"corpus manifest could not be read: {type(exc).__name__}: {exc}"],
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
            "assets": [],
            "by_case": {},
            "coverage": {},
            "validation_backend": "runner-fallback",
        }
    raw = dict(decoded) if isinstance(decoded, Mapping) else {}
    errors: List[str] = []
    warnings: List[str] = []
    module_backend = "runner-fallback"

    corpus_module = _optional_module("retouch.corpus_manifest")
    loaded = raw
    hook = _module_hook(
        corpus_module,
        ("load_and_validate", "validate_corpus_manifest", "validate_manifest", "load_manifest", "read_manifest"),
    )
    if hook is not None:
        try:
            # The in-repo v2 validator consumes decoded JSON plus the
            # manifest directory as its explicit path root. Older/developing
            # validators may instead consume a path, so retain the generic
            # fallback for those signatures.
            if getattr(hook, "__name__", "") == "validate_corpus_manifest":
                result = hook(raw, root=manifest_path.parent)
                hook_error = None
            else:
                result, hook_error = _call_hook(hook, ((manifest_path,), (str(manifest_path),), (raw,)))
        except Exception as exc:
            result = None
            hook_error = f"{type(exc).__name__}: {exc}"
        if hook_error:
            errors.append(f"corpus manifest validator failed: {hook_error}")
        elif result is not None:
            module_backend = "retouch.corpus_manifest"
            if hasattr(result, "to_dict") and callable(result.to_dict):
                result = result.to_dict()
            if isinstance(result, Mapping):
                if isinstance(result.get("manifest"), Mapping):
                    loaded = dict(result["manifest"])
                elif isinstance(result.get("data"), Mapping):
                    loaded = dict(result["data"])
                elif any(key in result for key in ("assets", "images", "cases", "entries", "corpus")):
                    loaded = dict(result)
                for key in ("errors", "issues"):
                    values = result.get(key)
                    if isinstance(values, list):
                        errors.extend(str(value) for value in values if str(value).strip())
                if result.get("valid") is False:
                    errors.append("corpus manifest validator reported valid=false")
            elif result is False:
                errors.append("corpus manifest validator reported false")

            # The canonical in-repo validator returns normalized assets,
            # explicit strata coverage, and machine-readable issues. Consume
            # that report as the source of truth rather than reinterpreting
            # its labels with the v1 filename aliases below.
            if (
                isinstance(result, Mapping)
                and getattr(hook, "__name__", "") == "validate_corpus_manifest"
                and "canonical_manifest" in result
            ):
                raw_by_id = {
                    str(item.get("asset_id")): item
                    for item in _manifest_assets(raw)
                    if item.get("asset_id")
                }
                module_assets = result.get("assets") if isinstance(result.get("assets"), list) else []
                module_errors = [
                    _json_safe(issue)
                    for issue in result.get("errors", [])
                ] if isinstance(result.get("errors"), list) else []
                matched: Dict[str, Dict[str, Any]] = {}
                normalized_assets: List[Dict[str, Any]] = []
                by_key: Dict[str, Dict[str, Any]] = {}
                for item in module_assets:
                    if not isinstance(item, Mapping):
                        continue
                    asset_id = str(item.get("asset_id", ""))
                    raw_item = raw_by_id.get(asset_id, {})
                    relative_path = str(item.get("path", ""))
                    resolved_path = (manifest_path.parent / relative_path).resolve()
                    summary = {
                        "asset_id": asset_id,
                        "sha256": str(item.get("sha256", "")),
                        "tags": list(item.get("tags", [])) if isinstance(item.get("tags"), list) else [],
                        "strata": dict(item.get("strata", {})) if isinstance(item.get("strata"), Mapping) else {},
                        "split": item.get("split"),
                        "consent_ref": result.get("consent_reference"),
                        "expected_face_count": _asset_expected_faces(raw_item),
                        "path": str(resolved_path),
                    }
                    normalized_assets.append(summary)
                    for key in (asset_id, str(resolved_path)):
                        if key:
                            by_key[key] = summary
                for case_name, image_path in cases:
                    match = by_key.get(case_name) or by_key.get(str(image_path.resolve()))
                    if match is None:
                        module_errors.append({
                            "code": "case_not_in_corpus_manifest",
                            "case": case_name,
                        })
                    else:
                        matched[case_name] = dict(match)
                valid = bool(result.get("valid")) and not module_errors
                return {
                    "valid": valid,
                    "complete": valid and bool(matched),
                    "errors": module_errors,
                    "warnings": [_json_safe(item) for item in result.get("warnings", [])] if isinstance(result.get("warnings"), list) else [],
                    "path": str(manifest_path),
                    "sha256": _sha256_file(manifest_path),
                    "validation_backend": module_backend,
                    "validator_report": _json_safe(dict(result)),
                    "assets": normalized_assets,
                    "by_case": matched,
                    "coverage": _json_safe(result.get("coverage", {})),
                    "declared_complete": bool(result.get("valid")),
                    "tag_vocabulary_version": raw.get("tag_vocabulary_version"),
                }

    assets = _manifest_assets(loaded)
    if not assets:
        errors.append("v2 corpus manifest must contain an assets/images/cases list")

    hashes: Dict[str, str] = {}
    by_case: Dict[str, Dict[str, Any]] = {}
    coverage: Dict[str, List[str]] = {dimension: [] for dimension in _V2_REQUIRED_DIMENSIONS}
    normalised_assets: List[Dict[str, Any]] = []
    for index, asset in enumerate(assets):
        identities = _asset_identity(asset)
        asset_id = identities[0] if identities else f"asset-{index + 1}"
        source_hash = str(
            asset.get("sha256", asset.get("source_sha256", asset.get("content_sha256", "")))
            or ""
        ).strip().lower()
        if len(source_hash) != _SHA256_LENGTH or any(char not in "0123456789abcdef" for char in source_hash):
            errors.append(f"asset {asset_id!r} must declare a 64-character SHA-256")
        elif source_hash in hashes:
            errors.append(
                f"duplicate corpus asset hash {source_hash} for {asset_id!r} and {hashes[source_hash]!r}"
            )
        else:
            hashes[source_hash] = asset_id
        consent = str(
            asset.get("consent_ref", asset.get("consent_reference", asset.get("consent", "")))
            or ""
        ).strip()
        if not consent:
            errors.append(f"asset {asset_id!r} is missing a consent reference")
        tags = _asset_tags(asset)
        if not tags:
            errors.append(f"asset {asset_id!r} has no validated tags")
        split = str(asset.get("split", asset.get("partition", "")) or "").strip().lower()
        if split not in {"pilot", "holdout", "calibration", "locked_holdout"}:
            errors.append(f"asset {asset_id!r} has no valid pilot/holdout split")
        expected_faces = _asset_expected_faces(asset)
        if expected_faces is None or expected_faces < 0:
            errors.append(f"asset {asset_id!r} must declare expected_face_count")
        source_path = _asset_path(asset, manifest_path)
        if source_path is None:
            errors.append(f"asset {asset_id!r} is missing a source path")
        case_keys = set(identities)
        if source_path is not None:
            case_keys.add(str(source_path))
        for key in case_keys:
            if key:
                by_case[key] = {
                    "asset_id": asset_id,
                    "sha256": source_hash,
                    "tags": tags,
                    "split": split,
                    "consent_ref": consent,
                    "expected_face_count": expected_faces,
                    "path": str(source_path) if source_path else None,
                }
        for dimension in _V2_REQUIRED_DIMENSIONS:
            if dimension in tags:
                coverage[dimension].append(asset_id)
        normalised_assets.append({
            "asset_id": asset_id,
            "sha256": source_hash,
            "tags": tags,
            "split": split,
            "consent_ref": consent,
            "expected_face_count": expected_faces,
            "path": str(source_path) if source_path else None,
        })

    matched: Dict[str, Dict[str, Any]] = {}
    for case_name, image_path in cases:
        match = by_case.get(case_name) or by_case.get(str(image_path.resolve()))
        if match is None:
            errors.append(f"case {case_name!r} is not explicitly present in the corpus manifest")
            continue
        actual_hash = _sha256_file(image_path)
        if actual_hash is None:
            errors.append(f"case {case_name!r} could not be hashed")
        elif actual_hash != match["sha256"]:
            errors.append(f"case {case_name!r} source SHA-256 does not match the corpus manifest")
        matched[case_name] = dict(match)

    declared_complete = loaded.get("complete") if isinstance(loaded, Mapping) else None
    if declared_complete is False:
        warnings.append("corpus manifest explicitly declares complete=false")
    coverage_complete = all(coverage[dimension] for dimension in _V2_REQUIRED_DIMENSIONS)
    complete = bool(
        not errors
        and coverage_complete
        and (bool(declared_complete) if declared_complete is not None else True)
        and matched
    )
    return {
        "valid": not errors,
        "complete": complete,
        "errors": errors,
        "warnings": warnings,
        "path": str(manifest_path),
        "sha256": _sha256_file(manifest_path),
        "validation_backend": module_backend,
        "assets": normalised_assets,
        "by_case": matched,
        "coverage": {key: sorted(set(value)) for key, value in coverage.items()},
        "declared_complete": declared_complete,
        "tag_vocabulary_version": loaded.get("tag_vocabulary_version"),
    }


def _sanitise_runtime(value: Any, key: str = "") -> Any:
    """Remove machine-specific paths from a Runtime Doctor snapshot."""
    lower_key = key.lower()
    if lower_key in {
        "path", "module_path", "executable", "project_root", "pyproject", "manifest",
        "manifest_path", "output_dir", "input", "filename",
    } or lower_key.endswith(("_path", "_dir")):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(name): _sanitise_runtime(item, str(name)) for name, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitise_runtime(item, key) for item in value]
    if isinstance(value, Path):
        return "<redacted>"
    if isinstance(value, str):
        replacements = [str(PROJECT_ROOT), str(Path.cwd()), str(Path.home())]
        result = value
        for replacement in replacements:
            if replacement:
                result = result.replace(replacement, "<redacted>")
        return result
    return _json_safe(value)


def _runtime_doctor_snapshot() -> Dict[str, Any]:
    """Capture a probed, sanitized Runtime Doctor report and its content hash."""
    report: Dict[str, Any]
    probe_call_used = False
    module = _optional_module("retouch.runtime_doctor")
    function = _module_hook(module, ("runtime_doctor_report", "collect_runtime_report", "run_doctor"))
    if function is None:
        report = {
            "schema_version": 1,
            "status": "error",
            "reason_code": "runtime_doctor_unavailable",
            "reason": "Runtime Doctor could not be imported",
        }
    else:
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            signature = None
        kwargs_supported: Optional[bool] = None
        if signature is not None:
            try:
                signature.bind(probe_detector=True, probe_parser=True)
                kwargs_supported = True
            except TypeError:
                kwargs_supported = False
        try:
            if kwargs_supported is False:
                # Only collectors whose signature rejects the probe kwargs
                # take this compatibility path. Their report must determine
                # whether a probe actually occurred; the wrapper never does.
                result = function()
            else:
                result = function(probe_detector=True, probe_parser=True)
                probe_call_used = True
            hook_error = None
        except Exception as exc:
            result = None
            hook_error = f"{type(exc).__name__}: {exc}"
        if hook_error or not isinstance(result, Mapping):
            report = {
                "schema_version": 1,
                "status": "error",
                "reason_code": "runtime_doctor_failed",
                "reason": hook_error or "Runtime Doctor returned a non-mapping report",
            }
        else:
            report = dict(result)
    detector_report = report.get("detector") if isinstance(report.get("detector"), Mapping) else {}
    parser_report = report.get("parser") if isinstance(report.get("parser"), Mapping) else {}
    detector_probed = bool(_first_value(
        (detector_report,),
        ("probed", "probe_attempted", "probe_performed", "initialization_probed"),
        False,
    ))
    parser_probed = bool(_first_value(
        (parser_report,),
        ("probed", "probe_attempted", "probe_performed", "initialization_probed"),
        False,
    ))
    evidence_module = _optional_module("retouch.certification_evidence_v2")
    builder = getattr(evidence_module, "build_runtime_doctor_evidence", None) if evidence_module else None
    if callable(builder):
        try:
            built = builder(report)
        except Exception:
            built = None
        if isinstance(built, Mapping) and isinstance(built.get("snapshot"), Mapping):
            snapshot = _json_safe(built["snapshot"])
            digest = str(built.get("snapshot_sha256") or _sha256_json(snapshot))
            return {
                "schema_version": built.get("schema_version", report.get("schema_version", 1)),
                "contract": built.get("contract", V2_CONTRACT_VERSION),
                "probe_requested_detector": True,
                "probe_requested_parser": True,
                "probe_detector": detector_probed,
                "probe_parser": parser_probed,
                "probe_call_used": probe_call_used,
                "snapshot": snapshot,
                # Keep both names during the v2 module transition. The
                # canonical evidence module calls this snapshot_sha256.
                "snapshot_sha256": digest,
                "sha256": digest,
                "status": str(report.get("status", "unknown")),
            }
    snapshot = _sanitise_runtime(report)
    digest = _sha256_json(snapshot)
    return {
        "schema_version": report.get("schema_version", 1),
        "probe_requested_detector": True,
        "probe_requested_parser": True,
        "probe_detector": detector_probed,
        "probe_parser": parser_probed,
        "probe_call_used": probe_call_used,
        "snapshot": snapshot,
        "snapshot_sha256": digest,
        "sha256": digest,
        "status": str(report.get("status", "unknown")),
    }


def _path_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([str(path.resolve()), str(root.resolve())]) == str(root.resolve())
    except (OSError, ValueError):
        return False


def _decode_image(path: Path) -> Dict[str, Any]:
    """Decode an output for v2 evidence without making a decoder import-time mandatory."""
    try:
        from PIL import Image  # type: ignore

        with Image.open(str(path)) as image:
            image.load()
            mode = str(image.mode)
            channels = len(image.getbands())
            dtype = "uint16" if mode in {"I;16", "I;16B", "I;16L"} else "uint8"
            return {
                "decoded": True,
                "width": int(image.width),
                "height": int(image.height),
                "channels": channels,
                "dtype": dtype,
                "mode": mode,
                "metadata": {
                    "icc_profile": bool(image.info.get("icc_profile")),
                    "exif": bool(image.info.get("exif")),
                    "format": str(image.format or "").lower() or None,
                },
            }
    except Exception as pil_error:
        try:
            import cv2  # type: ignore

            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError("OpenCV returned no image")
            shape = tuple(int(value) for value in image.shape)
            channels = int(shape[2]) if len(shape) == 3 else 1
            return {
                "decoded": True,
                "width": int(shape[1]),
                "height": int(shape[0]),
                "channels": channels,
                "dtype": str(image.dtype),
                "mode": None,
                "metadata": {"icc_profile": None, "exif": None, "format": path.suffix.lower().lstrip(".")},
            }
        except Exception as cv_error:
            return {
                "decoded": False,
                "error": f"PIL: {type(pil_error).__name__}: {pil_error}; OpenCV: {type(cv_error).__name__}: {cv_error}",
            }


def _module_render_artifact(
    output_path: Path,
    row: Mapping[str, Any],
    recipe: str,
) -> Optional[Dict[str, Any]]:
    """Use the canonical v2 artifact builder when it is available."""
    module = _optional_module("retouch.certification_evidence_v2")
    builder = getattr(module, "build_render_artifact_evidence", None) if module else None
    if not callable(builder):
        return None
    observed = _decode_image(output_path) if output_path.is_file() else {"decoded": False}
    dimensions = [observed.get("width"), observed.get("height")]
    supplied = row.get("render_evidence")
    if isinstance(supplied, Mapping):
        supplied_dimensions = supplied.get("dimensions")
        if isinstance(supplied_dimensions, list) and len(supplied_dimensions) >= 2:
            dimensions = supplied_dimensions[:2]
        elif supplied.get("width") is not None and supplied.get("height") is not None:
            dimensions = [supplied.get("width"), supplied.get("height")]
    dimensions = [int(value) for value in dimensions] if all(value is not None for value in dimensions) else None
    dtype = str(
        row.get("dtype")
        or (supplied.get("dtype") if isinstance(supplied, Mapping) else None)
        or observed.get("dtype")
        or ""
    ) or None
    metadata = observed.get("metadata") if isinstance(observed.get("metadata"), Mapping) else {}
    metadata_present = bool(
        observed.get("metadata_present")
        if observed.get("metadata_present") is not None
        else metadata.get("icc_profile") or metadata.get("exif")
    )
    supplied_hash = row.get("output_sha256", row.get("sha256"))
    def decoder(_path: Path) -> Dict[str, Any]:
        return {
            "decodable": bool(observed.get("decoded")),
            "dimensions": dimensions,
            "dtype": dtype,
            "metadata_present": metadata_present,
            "metadata": metadata,
        }
    try:
        built = builder(
            output_path,
            expected_recipe=recipe,
            actual_recipe=row.get("recipe", recipe),
            expected_dimensions=dimensions,
            expected_dtype=dtype,
            expected_metadata=metadata,
            expected_metadata_present=metadata_present,
            metadata=metadata,
            metadata_present=metadata_present,
            decoder=decoder,
            expected_sha256=str(supplied_hash) if supplied_hash else None,
        )
    except Exception:
        return None
    return dict(built) if isinstance(built, Mapping) else None


def _render_evidence(
    case_manifest: Mapping[str, Any],
    case_output: Path,
    expected_recipes: Sequence[str],
    source_path: Path,
) -> Dict[str, Any]:
    outputs = case_manifest.get("outputs")
    reasons: List[str] = []
    rows: List[Dict[str, Any]] = []
    if not isinstance(outputs, list) or len(outputs) != len(expected_recipes):
        reasons.append("output_count_mismatch")
        outputs = outputs if isinstance(outputs, list) else []
    seen_recipes: List[str] = []
    for raw_row in outputs:
        row = dict(raw_row) if isinstance(raw_row, Mapping) else {}
        recipe = str(row.get("recipe", ""))
        seen_recipes.append(recipe)
        output_name = str(row.get("output", ""))
        output_path = (case_output / output_name).resolve()
        artifact: Dict[str, Any] = {
            "recipe": recipe,
            "output": output_name,
            "status": row.get("status"),
            "output_sha256": _sha256_file(output_path) if _path_within(output_path, case_output) else None,
            "output_bytes": output_path.stat().st_size if _path_within(output_path, case_output) and output_path.is_file() else 0,
            "source_sha256": _sha256_file(source_path),
            "row": _json_safe(row),
        }
        if recipe not in expected_recipes:
            reasons.append(f"unexpected_recipe:{recipe}")
        if row.get("status") != "done":
            reasons.append(f"render_not_done:{recipe}")
        if not _path_within(output_path, case_output) or not output_path.is_file() or output_path.stat().st_size <= 0:
            reasons.append(f"missing_output:{recipe}")
            artifact["decode"] = {"decoded": False, "error": "output missing or empty"}
        else:
            artifact["decode"] = _decode_image(output_path)
            if not artifact["decode"].get("decoded"):
                reasons.append(f"output_not_decodable:{recipe}")
        supplied_hash = row.get("output_sha256", row.get("sha256"))
        if supplied_hash and artifact["output_sha256"] != str(supplied_hash).lower():
            reasons.append(f"output_hash_mismatch:{recipe}")
        module_artifact = None
        if output_path.is_file() and _path_within(output_path, case_output):
            module_artifact = _module_render_artifact(output_path, row, recipe)
            if module_artifact is not None:
                artifact["canonical_v2"] = module_artifact
                artifact["render_completed"] = bool(module_artifact.get("render_completed"))
                if not artifact["render_completed"]:
                    reasons.append(f"render_evidence_incomplete:{recipe}")
        supplied_decode = row.get("decode", row.get("render_evidence"))
        if isinstance(supplied_decode, Mapping):
            dimensions = supplied_decode.get("dimensions")
            expected_width = supplied_decode.get(
                "width",
                dimensions[0] if isinstance(dimensions, list) and len(dimensions) > 0 else None,
            )
            expected_height = supplied_decode.get(
                "height",
                dimensions[1] if isinstance(dimensions, list) and len(dimensions) > 1 else None,
            )
            if expected_width is not None and artifact["decode"].get("width") != int(expected_width):
                reasons.append(f"dimension_mismatch:{recipe}")
            if expected_height is not None and artifact["decode"].get("height") != int(expected_height):
                reasons.append(f"dimension_mismatch:{recipe}")
        rows.append(artifact)
    if sorted(seen_recipes) != sorted(str(recipe) for recipe in expected_recipes):
        reasons.append("recipe_identity_mismatch")
    contact_name = str(case_manifest.get("contact_sheet", ""))
    contact_path = (case_output / contact_name).resolve()
    contact_ok = _path_within(contact_path, case_output) and contact_path.is_file() and contact_path.stat().st_size > 0
    if not contact_ok:
        reasons.append("missing_contact_sheet")
    contact = {
        "path": contact_name,
        "sha256": _sha256_file(contact_path) if contact_ok else None,
        "bytes": contact_path.stat().st_size if contact_ok else 0,
    }
    return {
        "passed": not reasons,
        "reasons": reasons,
        "source_sha256": _sha256_file(source_path),
        "outputs": rows,
        "contact_sheet": contact,
    }


def _evidence_objects(case_manifest: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    objects: List[Mapping[str, Any]] = [case_manifest]
    for key in (
        "evidence", "evidence_v2", "image_evidence", "face_evidence", "detector_evidence",
        "parser_evidence", "provider_evidence", "automatic_quality", "quality_evidence",
        "render_evidence",
    ):
        value = case_manifest.get(key)
        if isinstance(value, Mapping):
            objects.append(value)
        elif isinstance(value, list):
            objects.extend(item for item in value if isinstance(item, Mapping))
    outputs = case_manifest.get("outputs")
    if isinstance(outputs, list):
        objects.extend(item for item in outputs if isinstance(item, Mapping))
    return objects


def _face_gate(
    case_manifest: Mapping[str, Any],
    expected_face_count: Optional[int],
) -> Dict[str, Any]:
    objects = _evidence_objects(case_manifest)
    face_evidence = _first_mapping(objects, ("face_evidence", "face", "face_aware_evidence", "detector_evidence"))
    detector = _first_mapping(
        (face_evidence,) + tuple(objects),
        ("detector", "detector_probe", "detector_status", "detector_evidence"),
    )
    parser = _first_mapping(
        (face_evidence,) + tuple(objects),
        ("parser", "parser_evidence", "parser_status"),
    )
    provider = _first_value(
        (face_evidence, detector) + tuple(objects),
        (
            "active_provider", "active_execution_provider", "actual_provider",
            "execution_provider", "provider", "provider_evidence",
        ),
    )
    if isinstance(provider, Mapping):
        provider_verified = _truthy(
            provider.get("verified", provider.get("active", provider.get("actual", False)))
        )
        provider_name = provider.get(
            "name",
            provider.get(
                "actual_provider",
                provider.get("active_provider", provider.get("provider")),
            ),
        )
        if provider_name is None and isinstance(provider.get("active"), list) and provider.get("active"):
            provider_name = provider["active"][0]
    else:
        provider_verified = _truthy(
            _first_value(
                (face_evidence, detector) + tuple(objects),
                ("active_provider_verified", "provider_verified", "execution_provider_verified"),
                False,
            )
        )
        provider_name = provider
    expected = expected_face_count
    if expected is None:
        expected = _as_int(_first_value(
            (face_evidence, detector) + tuple(objects),
            ("expected_face_count", "expected_faces", "faces_expected"),
        ))
    detected = _as_int(_first_value(
        (face_evidence, detector) + tuple(objects),
        ("detected_face_count", "actual_face_count", "face_count", "faces_detected"),
    ))
    faces = _first_value(
        (face_evidence,) + tuple(objects),
        ("faces", "detected_faces", "face_observations"),
        [],
    )
    faces = faces if isinstance(faces, list) else []
    probed = _truthy(_first_value(
        (face_evidence, detector) + tuple(objects),
        (
            "probed", "probe_performed", "detector_probed", "initialization_probed",
            "probe_attempted",
        ),
        False,
    ))
    if not probed:
        probed = _truthy(_first_value(
            (face_evidence, detector) + tuple(objects),
            ("probe_succeeded", "probe_completed"),
            False,
        ))
    initialized = _truthy(_first_value(
        (face_evidence, detector) + tuple(objects),
        ("initialized", "initialization_succeeded", "detector_initialized"),
        False,
    ))
    backend = str(_first_value(
        (face_evidence, detector) + tuple(objects),
        ("backend", "backend_name", "detector_backend"),
        "",
    ) or "")
    parser_backend = str(_first_value(
        (parser,) + tuple(objects),
        ("backend", "backend_name", "parser_backend"),
        "",
    ) or "")
    parser_ok = bool(parser) and parser_backend.lower() not in {"", "unavailable", "unknown", "none"}
    parser_status = str(_first_value((parser,) + tuple(objects), ("status", "parser_status"), "ok") or "").lower()
    parser_ok = parser_ok and parser_status not in {"error", "failed", "unavailable"}
    landmarks_ok = bool(faces) and all(
        isinstance(face, Mapping)
        and (
            isinstance(face.get("landmarks"), (list, tuple)) and len(face.get("landmarks")) > 0
            or _as_int(face.get("landmark_count")) is not None and _as_int(face.get("landmark_count")) > 0
        )
        for face in faces
    )
    if not landmarks_ok:
        landmark_count = _as_int(_first_value(
            (face_evidence,) + tuple(objects),
            ("landmark_count", "landmarks_detected"),
            0,
        ))
        landmarks_ok = bool(detected and landmark_count and landmark_count >= detected)
    checks = {
        "global_only_false": case_manifest.get("global_only") is False,
        "probed": probed,
        "initialized": initialized,
        "backend": backend.lower() not in {"", "unavailable", "unknown", "none"},
        "expected_face_count": expected is not None and expected > 0,
        "detected_face_count": detected is not None and detected == expected and detected > 0,
        "landmarks": landmarks_ok,
        "parser": parser_ok,
        "actual_provider": provider_verified and bool(str(provider_name or "").strip()),
    }
    reasons = [name for name, passed in checks.items() if not passed]
    canonical = _module_face_evidence(
        case_manifest,
        detector=detector,
        parser=parser,
        provider=provider,
        faces=faces,
        expected_face_count=expected,
        detected_face_count=detected,
    )
    if canonical is not None and canonical.get("face_aware") is not True:
        reasons.append("canonical_face_evidence_invalid")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "expected_face_count": expected,
        "detected_face_count": detected,
        "detector": _json_safe(detector),
        "parser": _json_safe(parser),
        "provider": _json_safe(provider),
        "provider_name": provider_name,
        "faces": _json_safe(faces),
        "checks": checks,
        "raw": _json_safe(face_evidence),
        "canonical_v2": _json_safe(canonical) if canonical is not None else None,
    }


def _module_face_evidence(
    case_manifest: Mapping[str, Any],
    *,
    detector: Mapping[str, Any],
    parser: Mapping[str, Any],
    provider: Any,
    faces: Sequence[Any],
    expected_face_count: Optional[int],
    detected_face_count: Optional[int],
) -> Optional[Dict[str, Any]]:
    """Normalize sweep detector/parser/provider observations through v2."""
    module = _optional_module("retouch.certification_evidence_v2")
    builder = getattr(module, "build_face_aware_evidence", None) if module else None
    if not callable(builder):
        return None
    probe = dict(detector)
    probe.setdefault("probe_attempted", detector.get("probed", False))
    probe.setdefault("probe_succeeded", detector.get("probe_succeeded", detector.get("initialized", False)))
    probe.setdefault("detector_initialized", detector.get("initialized", False))
    parser_backend = parser.get("backend", parser.get("parser_backend"))
    provider_value = provider
    provider_verified = None
    if isinstance(provider, Mapping):
        provider_verified = bool(provider.get("verified", provider.get("active", False)))
        provider_value = provider.get(
            "actual_provider",
            provider.get("name", provider.get("active_provider", provider.get("provider"))),
        )
        if provider_value is None and isinstance(provider.get("active"), list) and provider.get("active"):
            provider_value = provider["active"][0]
    landmarks = []
    for face in faces:
        if not isinstance(face, Mapping):
            continue
        value = face.get("landmarks")
        if value is None and isinstance(face.get("landmark_evidence"), Mapping):
            value = face["landmark_evidence"].get("landmarks")
        landmarks.append(value or [])
    try:
        built = builder(
            requested_mode="global-only" if case_manifest.get("global_only") is True else "full",
            detector_probe=probe,
            expected_face_count=expected_face_count,
            detected_face_count=detected_face_count,
            landmarks=landmarks,
            parser_backend=parser_backend,
            actual_provider=provider_value,
            provider_verified=provider_verified,
        )
    except Exception:
        return None
    return dict(built) if isinstance(built, Mapping) else None


def _quality_vector(value: Any) -> Tuple[List[Mapping[str, Any]], bool, List[str]]:
    if isinstance(value, Mapping):
        complete = value.get(
            "complete",
            value.get("coverage_complete", value.get("vector_complete", value.get("full_vector"))),
        )
        metrics = value.get(
            "metrics",
            value.get(
                "observations",
                value.get("vector", value.get("quality_metrics", value.get("qa"))),
            ),
        )
        if not isinstance(metrics, list):
            metrics = []
        raw_complete = bool(complete) if complete is not None else None
    elif isinstance(value, list):
        metrics = value
        raw_complete = None
    else:
        return [], False, ["quality_vector_missing"]
    rows = [item for item in metrics if isinstance(item, Mapping)]
    reasons: List[str] = []
    for index, metric in enumerate(rows):
        prefix = f"metric_{index}"
        if not str(metric.get("metric", metric.get("name", ""))).strip():
            reasons.append(f"{prefix}_name_missing")
        if not str(metric.get("version", "")).strip():
            reasons.append(f"{prefix}_version_missing")
        if not str(metric.get("threshold_version", "")).strip():
            reasons.append(f"{prefix}_threshold_version_missing")
        if "flagged" not in metric:
            reasons.append(f"{prefix}_flag_missing")
        if "available" not in metric and "status" not in metric:
            reasons.append(f"{prefix}_availability_missing")
        if "threshold" not in metric:
            reasons.append(f"{prefix}_threshold_missing")
        if "roi" not in metric and "roi_provenance" not in metric:
            reasons.append(f"{prefix}_roi_missing")
        if "details" not in metric:
            reasons.append(f"{prefix}_details_missing")
        status = str(metric.get("status", "pass") or "pass").lower()
        if _truthy(metric.get("flagged")) or status in {"fail", "failed", "error", "unavailable"}:
            reasons.append(f"{prefix}_flagged_or_failed")
        if metric.get("available") is False:
            reasons.append(f"{prefix}_unavailable")
    complete = bool(rows) and not reasons and (raw_complete is not False)
    if raw_complete is None and rows:
        # A developing v2 sweep may not yet expose a top-level marker. The
        # per-metric fields above are the conservative fallback proof.
        complete = bool(rows) and not reasons
    if not rows:
        reasons.append("quality_vector_empty")
    return rows, complete, reasons


def _module_quality_evidence(
    value: Any,
    metrics: Sequence[Mapping[str, Any]],
    complete: bool,
) -> Optional[Dict[str, Any]]:
    """Normalize sweep QA through the shared Certification Evidence v2 API."""
    module = _optional_module("retouch.certification_evidence_v2")
    builder = getattr(module, "build_automatic_quality_evidence", None) if module else None
    if not callable(builder):
        return None
    container = value if isinstance(value, Mapping) else {}
    metric_version = container.get("metric_version")
    threshold_version = container.get("threshold_version")
    if not metric_version and metrics:
        metric_version = metrics[0].get("version")
    if not threshold_version and metrics:
        threshold_version = metrics[0].get("threshold_version")
    thresholds = {
        str(metric.get("metric", metric.get("name", metric.get("detector", index)))): metric.get("threshold")
        for index, metric in enumerate(metrics)
        if metric.get("threshold") is not None
    }
    roi = container.get("roi_provenance") or container.get("roi")
    if not roi:
        roi = {
            str(metric.get("metric", metric.get("name", metric.get("detector", index)))): (
                metric.get("roi_provenance", metric.get("roi"))
            )
            for index, metric in enumerate(metrics)
            if metric.get("roi_provenance", metric.get("roi"))
        }
    failures = container.get("detector_failures")
    if failures is None:
        failures = container.get("unavailable", [])
    flagged = [
        str(metric.get("metric", metric.get("name", metric.get("detector", index))))
        for index, metric in enumerate(metrics)
        if metric.get("flagged")
    ]
    if not complete:
        outcome = "unresolved"
    elif flagged or failures:
        outcome = "rejected"
    else:
        outcome = "approved"
    try:
        built = builder(
            list(metrics),
            metric_version=str(metric_version) if metric_version else None,
            threshold_version=str(threshold_version) if threshold_version else None,
            thresholds=thresholds,
            roi_provenance=roi,
            detector_failures=list(failures) if isinstance(failures, Sequence) and not isinstance(failures, (str, bytes)) else [],
            outcome=outcome,
            flagged_metrics=flagged,
        )
    except Exception:
        return None
    return dict(built) if isinstance(built, Mapping) else None


def _quality_gate(case_manifest: Mapping[str, Any], expected_recipes: Sequence[str]) -> Dict[str, Any]:
    outputs = case_manifest.get("outputs") if isinstance(case_manifest.get("outputs"), list) else []
    per_recipe: Dict[str, Dict[str, Any]] = {}
    all_reasons: List[str] = []
    all_metrics: List[Mapping[str, Any]] = []
    for raw_row in outputs:
        row = raw_row if isinstance(raw_row, Mapping) else {}
        recipe = str(row.get("recipe", ""))
        value = _first_value(
            (row,),
            (
                "qa_evidence", "quality_evidence", "automatic_quality", "quality_metrics",
                "metrics", "qa_vector", "quality",
            ),
        )
        metrics, complete, reasons = _quality_vector(value)
        canonical = _module_quality_evidence(value, metrics, complete)
        if canonical is not None and canonical.get("automatic_pass") is not True:
            reasons = list(reasons) + ["canonical_quality_evidence_invalid"]
        if not complete:
            all_reasons.extend(f"{recipe}:{reason}" for reason in reasons)
        per_recipe[recipe] = {
            "complete": complete,
            "metrics": _json_safe(metrics),
            "reasons": reasons,
            "canonical_v2": _json_safe(canonical) if canonical is not None else None,
        }
        all_metrics.extend(metrics)
    missing = [recipe for recipe in expected_recipes if recipe not in per_recipe]
    all_reasons.extend(f"missing_quality:{recipe}" for recipe in missing)
    if not outputs:
        # Some developing sweep manifests place a source-level vector under
        # quality_evidence. It is accepted only when it explicitly covers all
        # recipes; otherwise every render remains unproven.
        source_value = _first_value((case_manifest,), ("quality_evidence", "automatic_quality", "quality_metrics"))
        metrics, complete, reasons = _quality_vector(source_value)
        if complete and isinstance(source_value, Mapping) and source_value.get("recipes") == list(expected_recipes):
            all_metrics.extend(metrics)
        else:
            all_reasons.extend(f"source_quality:{reason}" for reason in reasons)
    return {
        "passed": not all_reasons and bool(all_metrics),
        "reasons": all_reasons,
        "per_recipe": per_recipe,
        "metrics": _json_safe(all_metrics),
        "canonical_v2": [
            item["canonical_v2"]
            for item in per_recipe.values()
            if item.get("canonical_v2") is not None
        ],
        "raw": _json_safe({
            key: case_manifest[key]
            for key in ("qa_evidence", "quality_evidence", "automatic_quality", "quality_metrics")
            if key in case_manifest
        }),
    }


def _preserved_sweep_evidence(case_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep the sweep's raw per-image/backend evidence losslessly in the matrix."""
    keys = (
        "evidence", "evidence_v2", "image_evidence", "per_image_evidence",
        "face_evidence", "detector_evidence", "parser_evidence", "provider_evidence",
        "qa_evidence", "automatic_quality", "quality_evidence", "quality_metrics", "metrics",
    )
    selected = {key: case_manifest[key] for key in keys if key in case_manifest}
    outputs = case_manifest.get("outputs")
    if isinstance(outputs, list):
        selected["outputs"] = [
            {
                key: row[key]
                for key in (
                    "recipe", "render_evidence", "face_evidence", "detector_evidence",
                    "parser_evidence", "provider_evidence", "quality_evidence",
                    "qa_evidence", "automatic_quality", "quality_metrics", "metrics", "qa",
                )
                if isinstance(row, Mapping) and key in row
            }
            for row in outputs
            if isinstance(row, Mapping)
        ]
    return _json_safe(selected)


def _optional_evidence_validation(
    case_manifest: Mapping[str, Any],
    case_output: Path,
    expected_recipes: Sequence[str],
) -> Dict[str, Any]:
    module = _optional_module("retouch.certification_evidence_v2")
    hook = _module_hook(
        module,
        ("validate_case_evidence", "validate_case", "evaluate_case", "automatic_pass", "case_automatic_pass"),
    )
    if hook is None:
        return {"available": False, "used": False}
    result, hook_error = _call_hook(
        hook,
        (
            (case_manifest, case_output, len(expected_recipes)),
            (case_manifest, case_output, expected_recipes),
            (case_manifest,),
        ),
    )
    if hook_error:
        return {"available": True, "used": False, "error": hook_error}
    if isinstance(result, Mapping):
        return {"available": True, "used": True, "result": _json_safe(dict(result))}
    return {"available": True, "used": True, "result": bool(result)}


def _v2_case_evidence(
    case_manifest: Mapping[str, Any],
    case_output: Path,
    source_path: Path,
    expected_recipes: Sequence[str],
    expected_face_count: Optional[int],
) -> Dict[str, Any]:
    render = _render_evidence(case_manifest, case_output, expected_recipes, source_path)
    face = _face_gate(case_manifest, expected_face_count)
    quality = _quality_gate(case_manifest, expected_recipes)
    extension = _optional_evidence_validation(case_manifest, case_output, expected_recipes)
    reasons = list(render["reasons"]) + list(face["reasons"]) + list(quality["reasons"])
    if extension.get("result") is False:
        reasons.append("certification_evidence_v2_hook_rejected_case")
    if isinstance(extension.get("result"), Mapping):
        result = extension["result"]
        if result.get("passed") is False or result.get("automatic_pass") is False:
            reasons.append("certification_evidence_v2_hook_rejected_case")
    return {
        "automatic_pass": not reasons,
        "face_aware_run": bool(face["passed"]),
        "render": render,
        "face": face,
        "quality": quality,
        "extension": extension,
        "reasons": reasons,
    }


def _parse_cases(raw_cases: Sequence[str]) -> List[Tuple[str, Path]]:
    cases: List[Tuple[str, Path]] = []
    for raw in raw_cases:
        if "=" not in raw:
            raise ValueError(f"case must be NAME=PATH, got {raw!r}")
        name, raw_path = raw.split("=", 1)
        name = name.strip()
        path = Path(raw_path).expanduser().resolve()
        if not name or not path.is_file():
            raise ValueError(f"case {raw!r} must name an existing image file")
        cases.append((name, path))
    if not cases:
        raise ValueError("provide at least one --case NAME=PATH")
    return cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True, help="Corpus case as NAME=IMAGE_PATH; repeatable")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--corpus-manifest",
        default=None,
        help=(
            "Private JSON corpus manifest for Certification Evidence v2. "
            "When supplied, filename-derived coverage is disabled and strict "
            "render/face/quality evidence gates are applied."
        ),
    )
    parser.add_argument("--max-dim", type=int, default=None)
    parser.add_argument("--global-only", action="store_true", help="Diagnostic only; never face-aware certification")
    parser.add_argument(
        "--allow-incomplete-corpus",
        action="store_true",
        help="Generate a diagnostic matrix without the required six-case corpus gate",
    )
    parser.add_argument("--stop-on-fail", action="store_true")
    return parser


def _human_review_markdown(rows: Sequence[Dict[str, object]], corpus: Dict[str, object]) -> str:
    """Create an inspectable companion to the machine-readable review JSON."""
    lines = [
        "# Core Recipe Human Review Worksheet",
        "",
        "Automatic and human evidence are separate. Do not mark this review approved until the contact sheet is inspected at 100%.",
        "",
        "## Corpus coverage",
        "",
        f"- Complete: `{bool(corpus.get('complete'))}`",
        f"- Missing: {', '.join(json.dumps(item, sort_keys=True) if isinstance(item, Mapping) else str(item) for item in (corpus.get('missing') or corpus.get('errors') or [])) or 'none'}",
        "",
    ]
    for case in rows:
        lines.extend([
            f"## {case['case']}",
            "",
            f"- Input: `{case['input']}`",
            f"- Face-aware run: `{case['face_aware_run']}`",
            f"- Automatic pass: `{case['automatic_pass']}`",
            f"- Corpus dimensions: {', '.join(case.get('corpus_dimensions') or []) or 'none'}",
            "",
            "| Recipe | Review dimension | Natural output | Reviewer | Notes |",
            "| --- | --- | --- | --- | --- |",
        ])
        for recipe in case["recipes"]:
            lines.append(f"| {recipe} | {CORE_RECIPE_REVIEW_DIMENSIONS[recipe]} | pending |  |  |")
        lines.append("")
    lines.extend([
        "## Finalisation",
        "",
        "Copy approved/rejected decisions into `human_review.json`, then run `finalize_core_recipe_certification.py`. A completed worksheet alone does not make the matrix certified.",
        "",
    ])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    unknown = [name for name in CORE_RECIPE_NAMES if name not in RECIPES]
    if unknown:
        raise RuntimeError(f"Core recipe selection contains unknown recipes: {unknown}")
    cases = _parse_cases(args.case)
    allow_incomplete = bool(getattr(args, "allow_incomplete_corpus", False))
    corpus_manifest_arg = getattr(args, "corpus_manifest", None)
    v2 = bool(corpus_manifest_arg)
    if v2:
        corpus_manifest_path = Path(corpus_manifest_arg).expanduser().resolve()
        corpus = _load_and_validate_corpus_manifest(corpus_manifest_path, cases)
        if (not corpus.get("valid") or not corpus.get("complete")) and not allow_incomplete:
            details = corpus.get("errors") or ["required coverage is incomplete"]
            raise ValueError(
                "Certification Evidence v2 corpus manifest failed validation: "
                + "; ".join(str(item) for item in details)
                + ". Use --allow-incomplete-corpus only for diagnostics."
            )
        runtime_doctor = _runtime_doctor_snapshot()
    else:
        # v1 remains a diagnostic compatibility path. Its semantic case-name
        # coverage and artifact-only automatic gate are intentionally retained
        # when no private v2 manifest is supplied.
        corpus = core_corpus_coverage(case_name for case_name, _ in cases)
        if corpus["missing"] and not allow_incomplete:
            raise ValueError(
                "representative certification corpus is incomplete; missing: "
                + ", ".join(corpus["missing"])
                + ". Use --allow-incomplete-corpus only for diagnostics."
            )
        runtime_doctor = None
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    recipes = ",".join(CORE_RECIPE_NAMES)
    rows: List[Dict[str, object]] = []
    for case_name, image_path in cases:
        case_output = output / case_name
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/recipes/recipe_sweep.py"),
            str(image_path), "-o", str(case_output),
            "--recipes", recipes, "--compare",
        ]
        if args.max_dim is not None:
            command.extend(["--max-dim", str(args.max_dim)])
        if args.global_only:
            command.append("--global-only")
        completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
        manifest_path = case_output / "manifest.json"
        case_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        if not isinstance(case_manifest, Mapping):
            case_manifest = {}
        if v2:
            asset = corpus.get("by_case", {}).get(case_name, {})
            evidence = _v2_case_evidence(
                case_manifest,
                case_output,
                image_path,
                list(CORE_RECIPE_NAMES),
                _as_int(asset.get("expected_face_count")) if isinstance(asset, Mapping) else None,
            )
            if completed.returncode != 0:
                evidence["reasons"].append(f"recipe_sweep_returncode:{completed.returncode}")
                evidence["automatic_pass"] = False
                evidence["face_aware_run"] = False
            face_aware_run = bool(evidence["face_aware_run"] and completed.returncode == 0)
            automatic_pass = bool(evidence["automatic_pass"] and completed.returncode == 0)
            dimensions = list(asset.get("tags", [])) if isinstance(asset, Mapping) else []
            row: Dict[str, object] = {
                "evidence_version": V2_SCHEMA_VERSION,
                "case": case_name,
                "input": str(image_path),
                "source_sha256": evidence["render"].get("source_sha256"),
                "returncode": completed.returncode,
                "manifest": str(manifest_path),
                "global_only": bool(case_manifest.get("global_only", args.global_only)),
                "status": "done" if completed.returncode == 0 else "failed",
                "recipes": list(CORE_RECIPE_NAMES),
                "recipe_count": len(case_manifest.get("outputs", [])) if isinstance(case_manifest.get("outputs"), list) else 0,
                "corpus_asset_id": asset.get("asset_id") if isinstance(asset, Mapping) else None,
                "corpus_dimensions": dimensions,
                "expected_face_count": evidence["face"].get("expected_face_count"),
                "detected_face_count": evidence["face"].get("detected_face_count"),
                "render_evidence": evidence["render"],
                "face_evidence": evidence["face"],
                "quality_evidence": evidence["quality"],
                "automatic_gate": {
                    "render_artifacts": bool(evidence["render"]["passed"]),
                    "face_aware": bool(evidence["face"]["passed"]),
                    "automatic_quality": bool(evidence["quality"]["passed"]),
                    "reasons": list(evidence["reasons"]),
                },
                # Keep both the selected evidence view and the complete raw
                # sweep manifest so later calibration can recover successful
                # detector observations, parser/provider data, and details.
                "per_image_evidence": _preserved_sweep_evidence(case_manifest),
                "recipe_sweep_manifest": _json_safe(dict(case_manifest)),
                "v2_extension_validation": evidence["extension"],
                **certification_fields(
                    face_aware_run=face_aware_run,
                    automatic_pass=automatic_pass,
                    human_review="pending",
                ),
            }
        else:
            face_aware_run = bool(
                completed.returncode == 0
                and case_manifest.get("global_only") is False
            )
            automatic_pass = core_case_automatic_pass(
                case_manifest,
                case_output,
                len(CORE_RECIPE_NAMES),
                returncode=completed.returncode,
            )
            row = {
                "case": case_name,
                "input": str(image_path),
                "returncode": completed.returncode,
                "manifest": str(manifest_path),
                "global_only": bool(case_manifest.get("global_only", args.global_only)),
                "status": "done" if completed.returncode == 0 else "failed",
                "recipes": list(CORE_RECIPE_NAMES),
                "recipe_count": len(case_manifest.get("outputs", [])) if isinstance(case_manifest.get("outputs"), list) else 0,
                "corpus_dimensions": [
                    dimension for dimension, matched in corpus["covered"].items() if case_name.lower() in matched
                ],
                **certification_fields(
                    face_aware_run=face_aware_run,
                    automatic_pass=automatic_pass,
                    human_review="pending",
                ),
            }
        rows.append(row)
        if completed.returncode != 0 and args.stop_on_fail:
            break

    matrix = {
        "version": V2_SCHEMA_VERSION if v2 else 1,
        "recipes": list(CORE_RECIPE_NAMES),
        "recipe_count": len(CORE_RECIPE_NAMES),
        "full_catalog_count_at_run": len(RECIPES),
        "global_only": bool(args.global_only),
        "corpus_complete": bool(corpus["complete"]),
        "corpus_coverage": corpus["coverage"] if v2 else corpus,
        "cases": rows,
    }
    if v2:
        matrix.update({
            "schema_version": V2_SCHEMA_VERSION,
            "contract_version": V2_CONTRACT_VERSION,
            "runtime_doctor": runtime_doctor,
            "runtime_doctor_snapshot": runtime_doctor["snapshot"],
            "runtime_doctor_sha256": runtime_doctor["sha256"],
            "corpus": corpus,
            "corpus_manifest": {
                "path": corpus.get("path"),
                "sha256": corpus.get("sha256"),
                "validation_backend": corpus.get("validation_backend"),
            },
        })
    review_rows = []
    for case_name, _ in cases:
        for recipe in CORE_RECIPE_NAMES:
            review_rows.append({
                "case": case_name,
                "recipe": recipe,
                "dimension": CORE_RECIPE_REVIEW_DIMENSIONS[recipe],
                "automatic_status": "pending",
                "human_natural_output": "pending",
                "reviewer": "",
                "notes": "",
            })
    automatic_pass = (
        bool(corpus.get("valid", True))
        and bool(corpus["complete"])
        and bool(rows)
        and all(row["automatic_pass"] for row in rows)
    )
    matrix.update({
        **certification_fields(
            face_aware_run=bool(rows) and all(row["face_aware_run"] for row in rows),
            automatic_pass=automatic_pass,
            human_review="pending",
        ),
        "final_certified": core_matrix_final_certified(
            rows,
            review_rows,
            len(CORE_RECIPE_NAMES),
            corpus_complete=bool(corpus["complete"]),
        ),
    })
    if v2:
        matrix["gates"] = {
            "render_artifacts": bool(rows) and all(row["automatic_gate"]["render_artifacts"] for row in rows),
            "face_aware": bool(rows) and all(row["automatic_gate"]["face_aware"] for row in rows),
            "automatic_quality": bool(rows) and all(row["automatic_gate"]["automatic_quality"] for row in rows),
            "corpus": bool(corpus.get("valid")) and bool(corpus.get("complete")),
            "runtime_doctor_snapshot": bool(runtime_doctor.get("snapshot")) and bool(runtime_doctor.get("sha256")),
            "human_review": False,
            "certified": bool(matrix["final_certified"]),
        }
    (output / "matrix_manifest.json").write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    (output / "human_review.json").write_text(json.dumps({"version": V2_SCHEMA_VERSION if v2 else 1, "rows": review_rows}, indent=2), encoding="utf-8")
    (output / "human_review.md").write_text(_human_review_markdown(rows, corpus), encoding="utf-8")
    return 0 if matrix["automatic_pass"] else 1


if __name__ == "__main__":
    parser = build_parser()
    try:
        raise SystemExit(run(parser.parse_args()))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)

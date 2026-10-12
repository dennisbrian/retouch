"""Pure evidence contracts for Retouch certification v2.

The v1 certification helpers intentionally remain unchanged.  This module is
an isolated, dependency-light contract for collecting evidence before a
future integration layer writes a certification matrix.  It imports only the
Python standard library and returns ordinary JSON-compatible values.

The contract treats missing observations as unresolved rather than inferring
success.  A requested full-mode run is therefore never equivalent to a
face-aware run: the detector must have been probed and initialized, and the
per-image face evidence must be present.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import re
import struct
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 2
CONTRACT_NAME = "retouch.certification.evidence"
CONTRACT_VERSION = "v2"

OUTCOMES = {"approved", "rejected", "uncertain", "unresolved", "pending"}
_RESOLVED_OUTCOMES = {"approved", "rejected"}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

_SENSITIVE_KEY_PARTS = (
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "private_key",
)
_VOLATILE_KEY_NAMES = {
    "cache_dir",
    "cache_path",
    "cwd",
    "current_directory",
    "executable",
    "finished_at",
    "generated_at",
    "manifest_path",
    "path",
    "project_root",
    "pyproject",
    "started_at",
    "timestamp",
    "working_directory",
}


def _key_name(key: Any) -> str:
    return str(key).strip().lower().replace("-", "_")


def _is_sensitive_key(key: Any) -> bool:
    name = _key_name(key)
    return name in _VOLATILE_KEY_NAMES or any(part in name for part in _SENSITIVE_KEY_PARTS)


def _json_safe(value: Any) -> Any:
    """Convert supported values to JSON-compatible values without imports.

    ``bytes`` are represented by a length and digest instead of being dropped;
    that keeps metadata content auditable without putting binary payloads in a
    certification manifest.  Unsupported objects fail loudly so evidence is
    never silently truncated.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite numbers are not valid certification evidence")
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        return {
            "__bytes_sha256__": hashlib.sha256(payload).hexdigest(),
            "length": len(payload),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_safe(item) for item in value), key=lambda item: repr(item))
    raise TypeError("unsupported evidence value: %s" % type(value).__name__)


def canonical_json(value: Any) -> str:
    """Serialize a value deterministically for hashing and manifests."""

    safe = _json_safe(value)
    return json.dumps(
        safe,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 digest of :func:`canonical_json` output."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: os.PathLike[str] | str, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading the whole artifact into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _safe_error(exc: BaseException) -> str:
    detail = str(exc).strip().replace("\x00", " ")
    if len(detail) > 400:
        detail = detail[:397] + "..."
    return "%s: %s" % (type(exc).__name__, detail) if detail else type(exc).__name__


def _nonempty_text(value: Any) -> Optional[str]:
    text = str(value).strip() if value is not None else ""
    return text or None


def _normalise_dimensions(value: Any) -> Optional[List[int]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        dimensions = [int(value[0]), int(value[1])]
    except (TypeError, ValueError):
        return None
    return dimensions if all(item > 0 for item in dimensions) else None


def _normalise_sha256(value: Any) -> Optional[str]:
    text = _nonempty_text(value)
    return text.lower() if text and _HEX64.fullmatch(text.lower()) else None


def _normalise_outcome(value: Any) -> str:
    outcome = str(value or "unresolved").strip().lower()
    if outcome not in OUTCOMES:
        raise ValueError("unknown evidence outcome: %r" % outcome)
    return outcome


def sanitize_runtime_doctor_snapshot(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove paths, volatile fields, and credentials from a doctor report.

    The useful compatibility facts (versions, statuses, backends, provider
    lists, model hashes, and reason codes) remain intact.  Keys are filtered at
    every nesting level so a nested package or environment report cannot leak
    a credential or machine-specific path into a certification matrix.
    """

    if not isinstance(snapshot, Mapping):
        raise TypeError("runtime doctor snapshot must be a mapping")

    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): clean(item)
                for key, item in value.items()
                if not _is_sensitive_key(key)
            }
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        return _json_safe(value)

    return clean(snapshot)


def build_runtime_doctor_evidence(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a sanitized Runtime Doctor snapshot and its content hash."""

    sanitized = sanitize_runtime_doctor_snapshot(snapshot)
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_NAME,
        "snapshot": sanitized,
        "snapshot_sha256": canonical_sha256(sanitized),
    }


runtime_doctor_evidence = build_runtime_doctor_evidence
sanitize_runtime_doctor = sanitize_runtime_doctor_snapshot


def _png_decoder(data: bytes) -> Dict[str, Any]:
    if len(data) < 26 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    width, height, bit_depth, color_type = struct.unpack(">IIBB", data[16:26])
    if not width or not height:
        raise ValueError("PNG dimensions are empty")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        raise ValueError("unsupported PNG color type")
    return {
        "decodable": True,
        "decoder_backend": "png-header",
        "dimensions": [width, height],
        "dtype": "uint16" if bit_depth == 16 else "uint8",
        "metadata_present": b"eXIf" in data or b"iCCP" in data or b"tEXt" in data,
        "metadata": {
            "format": "PNG",
            "bit_depth": bit_depth,
            "channels": channels,
        },
    }


def _jpeg_dimensions(data: bytes) -> Tuple[int, int]:
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("not a JPEG file")
    index = 2
    while index + 3 < len(data):
        while index < len(data) and data[index] != 0xFF:
            index += 1
        while index < len(data) and data[index] == 0xFF:
            index += 1
        if index >= len(data):
            break
        marker = data[index]
        index += 1
        if marker in (0xD8, 0xD9):
            continue
        if index + 2 > len(data):
            break
        segment_length = struct.unpack(">H", data[index:index + 2])[0]
        if segment_length < 2 or index + segment_length > len(data):
            raise ValueError("truncated JPEG segment")
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if segment_length < 7:
                raise ValueError("truncated JPEG frame")
            height, width = struct.unpack(">HH", data[index + 3:index + 7])
            if width and height:
                return [width, height]
        index += segment_length
    raise ValueError("JPEG frame dimensions were not found")


def _default_image_decoder(path: Path) -> Dict[str, Any]:
    """Decode common images with Pillow when available, with PNG/JPEG fallback."""

    data = path.read_bytes()
    try:
        from PIL import Image  # type: ignore

        with Image.open(io.BytesIO(data)) as image:
            image.load()
            exif: Dict[str, Any] = {}
            try:
                exif = {str(key): _json_safe(value) for key, value in image.getexif().items()}
            except Exception:
                exif = {}
            metadata = {
                "format": str(image.format or "").upper() or None,
                "info_keys": sorted(str(key) for key in image.info.keys()),
                "exif": exif,
            }
            metadata_present = bool(metadata["info_keys"] or exif)
            dtype = "float32" if image.mode == "F" else "uint16" if image.mode.startswith("I;16") else "uint8"
            return {
                "decodable": True,
                "decoder_backend": "Pillow",
                "dimensions": [int(image.width), int(image.height)],
                "dtype": dtype,
                "metadata_present": metadata_present,
                "metadata": metadata,
            }
    except ImportError:
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return _png_decoder(data)
        if data.startswith(b"\xff\xd8"):
            dimensions = _jpeg_dimensions(data)
            return {
                "decodable": True,
                "decoder_backend": "jpeg-header",
                "dimensions": dimensions,
                "dtype": "uint8",
                "metadata_present": False,
                "metadata": {},
            }
        raise ValueError("no built-in decoder is available for this format")


def _decoder_observation(
    path: Path,
    decoder: Optional[Callable[[Path], Mapping[str, Any]]],
) -> Dict[str, Any]:
    try:
        observed = decoder(path) if decoder is not None else _default_image_decoder(path)
        if not isinstance(observed, Mapping):
            raise TypeError("decoder must return a mapping")
        return _json_safe(observed)
    except Exception as exc:
        return {
            "decodable": False,
            "decoder_backend": None,
            "error": _safe_error(exc),
        }


def build_render_artifact_evidence(
    output_path: os.PathLike[str] | str,
    *,
    expected_recipe: Any,
    actual_recipe: Any = None,
    expected_dimensions: Optional[Sequence[int]] = None,
    expected_dtype: Optional[str] = None,
    expected_metadata: Optional[Mapping[str, Any]] = None,
    expected_metadata_present: Optional[bool] = True,
    metadata: Optional[Mapping[str, Any]] = None,
    metadata_present: Optional[bool] = None,
    decoder: Optional[Callable[[Path], Mapping[str, Any]]] = None,
    expected_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect render output evidence without trusting a return code.

    The optional decoder is deliberately injectable for headless tests and
    alternate image stacks.  A decoder must report ``decodable``,
    ``dimensions``, and ``dtype``.  Metadata may come from the decoder or from
    the explicit arguments; recipe identity is supplied by the caller because
    it is a pipeline fact rather than something an image decoder can infer.
    Dimensions use ``[width, height]`` order.
    """

    path = Path(output_path)
    file_exists = path.is_file()
    artifact: Dict[str, Any] = {
        "path": path.name,
        "file_exists": file_exists,
        "size_bytes": int(path.stat().st_size) if file_exists else None,
        "sha256": None,
    }
    if file_exists:
        try:
            artifact["sha256"] = sha256_file(path)
        except OSError as exc:
            artifact["hash_error"] = _safe_error(exc)

    observed: Dict[str, Any] = {}
    if file_exists:
        observed = _decoder_observation(path, decoder)
    else:
        observed = {"decodable": False, "error": "artifact_missing"}

    if actual_recipe is None and "recipe" in observed:
        actual_recipe = observed.get("recipe")
    observed_dimensions = _normalise_dimensions(observed.get("dimensions"))
    observed_dtype = _nonempty_text(observed.get("dtype"))
    if metadata is None and "metadata" in observed:
        raw_metadata = observed.get("metadata")
        metadata = raw_metadata if isinstance(raw_metadata, Mapping) else None
    if metadata_present is None:
        if "metadata_present" in observed:
            metadata_present = bool(observed.get("metadata_present"))
        else:
            metadata_present = metadata is not None

    expected_dims = _normalise_dimensions(expected_dimensions)
    expected_hash = _normalise_sha256(expected_sha256)
    actual_hash = artifact.get("sha256")
    recipe_matches: Optional[bool] = (
        None if actual_recipe is None else _json_safe(actual_recipe) == _json_safe(expected_recipe)
    )
    dimensions_match: Optional[bool] = (
        None
        if expected_dims is None or observed_dimensions is None
        else expected_dims == observed_dimensions
    )
    dtype_matches: Optional[bool] = (
        None
        if expected_dtype is None or observed_dtype is None
        else str(expected_dtype) == observed_dtype
    )
    metadata_content = None if metadata is None else _json_safe(metadata)
    expected_content = None if expected_metadata is None else _json_safe(expected_metadata)
    metadata_content_matches: Optional[bool] = (
        None
        if metadata is None
        else metadata_content == expected_content if expected_metadata is not None else True
    )
    metadata_presence_matches: Optional[bool] = (
        None
        if metadata_present is None or expected_metadata_present is None
        else bool(metadata_present) == bool(expected_metadata_present)
    )
    hash_matches: Optional[bool] = None if actual_hash is None else (
        actual_hash == expected_hash if expected_hash is not None else True
    )

    checks = {
        "decodable": observed.get("decodable") is True,
        "recipe_identity": recipe_matches,
        "dimensions": dimensions_match,
        "dtype": dtype_matches,
        "metadata_presence": metadata_presence_matches,
        "metadata_content": metadata_content_matches,
        "sha256": hash_matches,
    }
    required_checks = [value for value in checks.values() if value is not None]
    missing_observations = [name for name, value in checks.items() if value is None]
    failed_checks = [name for name, value in checks.items() if value is False]
    render_completed = bool(
        file_exists
        and required_checks
        and not missing_observations
        and not failed_checks
        and all(required_checks)
    )
    status = "complete" if render_completed else "failed" if failed_checks or not file_exists else "unresolved"

    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_NAME,
        "artifact": artifact,
        "decode": observed,
        "recipe": {
            "expected": _json_safe(expected_recipe),
            "observed": None if actual_recipe is None else _json_safe(actual_recipe),
            "matches": recipe_matches,
        },
        "dimensions": {
            "expected": expected_dims,
            "observed": observed_dimensions,
            "matches": dimensions_match,
        },
        "dtype": {
            "expected": _nonempty_text(expected_dtype),
            "observed": observed_dtype,
            "matches": dtype_matches,
        },
        "metadata": {
            "expected_present": expected_metadata_present,
            "present": metadata_present,
            "presence_matches": metadata_presence_matches,
            "expected": expected_content,
            "observed": metadata_content,
            "content_sha256": None if metadata_content is None else canonical_sha256(metadata_content),
            "expected_sha256": None if expected_content is None else canonical_sha256(expected_content),
            "content_matches": metadata_content_matches,
        },
        "hash": {
            "sha256": actual_hash,
            "expected_sha256": expected_hash,
            "matches": hash_matches,
        },
        "checks": checks,
        "missing_observations": missing_observations,
        "failed_checks": failed_checks,
        "render_completed": render_completed,
        "status": status,
    }


render_artifact_evidence = build_render_artifact_evidence


def _probe_bool(probe: Mapping[str, Any], *names: str) -> Optional[bool]:
    for name in names:
        if name in probe:
            value = probe.get(name)
            if value is None:
                return None
            return bool(value)
    return None


def _provider_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (list, tuple, set, frozenset)):
        values = [_nonempty_text(item) for item in value]
        return sorted(item for item in values if item)
    return _json_safe(value)


def build_face_aware_evidence(
    *,
    requested_mode: str = "full",
    detector_probe: Optional[Mapping[str, Any]] = None,
    expected_face_count: Optional[int] = None,
    detected_face_count: Optional[int] = None,
    landmarks: Any = None,
    parser_backend: Any = None,
    actual_provider: Any = None,
    provider_verified: Optional[bool] = None,
) -> Dict[str, Any]:
    """Record actual per-image face-processing evidence.

    ``detector_probe`` must contain explicit probe/initialization facts.  A
    static ``available`` value alone is intentionally insufficient.  The
    provider is the observed execution provider, not the requested provider
    order; callers may set ``provider_verified=False`` when they only know the
    requested order.
    """

    probe = dict(detector_probe or {})
    requested_text = str(requested_mode or "unresolved").strip().lower()
    requested_full_mode = requested_text in {"full", "face-aware", "face_aware"}
    probe_attempted = _probe_bool(probe, "probe_attempted", "probed")
    probe_succeeded = _probe_bool(probe, "probe_succeeded", "initialized", "detector_initialized")
    initialized = _probe_bool(probe, "detector_initialized", "initialized")
    if probe_succeeded is True and initialized is None:
        initialized = True
    backend = probe.get("backend") or probe.get("detector_backend")
    backend = _nonempty_text(backend)
    if provider_verified is None:
        provider_verified = _provider_value(actual_provider) is not None
    provider = _provider_value(actual_provider)

    expected_count: Optional[int]
    detected_count: Optional[int]
    try:
        expected_count = None if expected_face_count is None else int(expected_face_count)
        detected_count = None if detected_face_count is None else int(detected_face_count)
    except (TypeError, ValueError):
        expected_count = detected_count = None
    if expected_count is not None and expected_count < 0:
        expected_count = None
    if detected_count is not None and detected_count < 0:
        detected_count = None
    face_count_matches: Optional[bool] = (
        None
        if expected_count is None or detected_count is None
        else expected_count == detected_count
    )

    landmark_value = None if landmarks is None else _json_safe(landmarks)
    landmark_count: Optional[int] = None
    if isinstance(landmarks, (list, tuple)):
        landmark_count = len(landmarks)
    landmarks_present = landmarks is not None
    landmarks_match: Optional[bool] = None
    if landmarks_present and detected_count is not None:
        landmarks_match = landmark_count == detected_count
        if landmarks_match and detected_count:
            landmarks_match = all(bool(item) for item in landmarks)

    parser = None if parser_backend is None else _json_safe(parser_backend)
    parser_known = parser_backend is not None and bool(_nonempty_text(parser_backend) or parser_backend)
    provider_known = provider is not None and bool(provider_verified)
    negative = (
        requested_full_mode is False
        or probe_attempted is False
        or probe_succeeded is False
        or initialized is False
        or face_count_matches is False
        or landmarks_match is False
        or provider_verified is False
    )
    missing = [
        name
        for name, value in {
            "probe_attempted": probe_attempted,
            "probe_succeeded": probe_succeeded,
            "detector_initialized": initialized,
            "expected_face_count": expected_count,
            "detected_face_count": detected_count,
            "landmarks": landmark_value,
            "parser_backend": parser,
            "actual_provider": provider,
        }.items()
        if value is None
    ]
    face_aware = bool(
        requested_full_mode
        and probe_attempted is True
        and probe_succeeded is True
        and initialized is True
        and face_count_matches is True
        and landmarks_match is True
        and parser_known
        and provider_known
    )
    status = "face-aware" if face_aware else "failed" if negative else "unresolved"
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_NAME,
        "requested_mode": requested_text,
        "requested_full_mode": requested_full_mode,
        "detector_probe": {
            "attempted": probe_attempted,
            "succeeded": probe_succeeded,
            "initialized": initialized,
            "backend": backend,
            "reason": _json_safe(probe.get("reason")) if "reason" in probe else None,
            "raw": _json_safe(probe),
        },
        "expected_face_count": expected_count,
        "detected_face_count": detected_count,
        "face_count_matches": face_count_matches,
        "landmarks": {
            "observed": landmark_value,
            "count": landmark_count,
            "matches_detected_faces": landmarks_match,
        },
        "parser_backend": parser,
        "actual_provider": provider,
        "provider_verified": bool(provider_verified) if provider_verified is not None else None,
        "face_aware": face_aware,
        "mode": "face-aware" if face_aware else "global-only" if requested_full_mode is False else "unresolved",
        "missing_observations": missing,
        "status": status,
    }


face_aware_evidence = build_face_aware_evidence


def _infer_flagged_metrics(metrics: Any) -> List[str]:
    flagged: List[str] = []

    def walk(value: Any, prefix: str = "") -> None:
        if isinstance(value, Mapping):
            if value.get("flagged") is True and prefix:
                flagged.append(prefix)
            for key, item in value.items():
                if str(key) == "flagged":
                    continue
                child = "%s.%s" % (prefix, key) if prefix else str(key)
                walk(item, child)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, "%s[%d]" % (prefix, index))

    walk(metrics)
    return sorted(set(flagged))


def build_automatic_quality_evidence(
    metrics: Any,
    *,
    metric_version: Optional[str],
    threshold_version: Optional[str],
    thresholds: Optional[Mapping[str, Any]],
    roi_provenance: Any,
    detector_failures: Optional[Sequence[Any]],
    outcome: str = "unresolved",
    flagged_metrics: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Preserve a complete metric vector and make unresolved states explicit."""

    quality_outcome = _normalise_outcome(outcome)
    metric_value = _json_safe(metrics)
    threshold_value = None if thresholds is None else _json_safe(thresholds)
    roi_value = None if roi_provenance is None else _json_safe(roi_provenance)
    failures_value = None if detector_failures is None else _json_safe(list(detector_failures))
    explicit_flags = [] if flagged_metrics is None else [str(item) for item in flagged_metrics]
    all_flags = sorted(set(explicit_flags + _infer_flagged_metrics(metrics)))
    versions_present = bool(_nonempty_text(metric_version) and _nonempty_text(threshold_version))
    complete = bool(
        isinstance(metrics, (Mapping, list, tuple))
        and bool(metrics)
        and versions_present
        and isinstance(thresholds, Mapping)
        and bool(thresholds)
        and isinstance(roi_provenance, (Mapping, list, tuple))
        and bool(roi_provenance)
        and roi_provenance is not None
        and detector_failures is not None
    )
    automatic_pass = bool(
        complete
        and quality_outcome == "approved"
        and not all_flags
        and not failures_value
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_NAME,
        "metric_version": _nonempty_text(metric_version),
        "threshold_version": _nonempty_text(threshold_version),
        "thresholds": threshold_value,
        "metric_vector": metric_value,
        "flagged_metrics": all_flags,
        "roi_provenance": roi_value,
        "detector_failures": failures_value,
        "outcome": quality_outcome,
        "outcome_resolved": quality_outcome in _RESOLVED_OUTCOMES,
        "evidence_complete": complete,
        "automatic_pass": automatic_pass,
        "status": "complete" if complete else "unresolved",
    }


automatic_quality_evidence = build_automatic_quality_evidence
quality_vector_evidence = build_automatic_quality_evidence


def build_corpus_evidence(
    assets: Sequence[Mapping[str, Any]],
    *,
    consent_reference: Optional[str],
    required_tags: Sequence[str] = (),
    allowed_splits: Sequence[str] = ("pilot", "holdout"),
) -> Dict[str, Any]:
    """Validate private corpus identity, consent, tags, and calibration splits."""

    if not isinstance(assets, Sequence) or isinstance(assets, (str, bytes, bytearray)):
        raise TypeError("corpus assets must be a sequence of mappings")
    required = {str(tag).strip() for tag in required_tags if str(tag).strip()}
    allowed = {str(split).strip() for split in allowed_splits if str(split).strip()}
    rows: List[Dict[str, Any]] = []
    seen_hashes: set[str] = set()
    duplicate_hashes: List[str] = []
    invalid_assets: List[Dict[str, Any]] = []
    covered_tags: set[str] = set()
    split_counts: Dict[str, int] = {split: 0 for split in sorted(allowed)}

    for index, asset in enumerate(assets):
        if not isinstance(asset, Mapping):
            invalid_assets.append({"index": index, "reason": "asset_not_mapping"})
            continue
        asset_hash = _normalise_sha256(asset.get("sha256") or asset.get("asset_sha256"))
        asset_id = _nonempty_text(asset.get("asset_id") or asset.get("id"))
        tags_raw = asset.get("tags")
        tags = sorted({str(tag).strip() for tag in tags_raw if str(tag).strip()}) if isinstance(tags_raw, Sequence) and not isinstance(tags_raw, (str, bytes, bytearray)) else []
        split = _nonempty_text(asset.get("split"))
        tags_validated = asset.get("tags_validated") is True or asset.get("validated_tags") is True
        problems: List[str] = []
        if not asset_hash:
            problems.append("invalid_sha256")
        elif asset_hash in seen_hashes:
            duplicate_hashes.append(asset_hash)
            problems.append("duplicate_sha256")
        else:
            seen_hashes.add(asset_hash)
        if not asset_id:
            problems.append("missing_asset_id")
        if not split or split not in allowed:
            problems.append("invalid_split")
        if not tags_validated:
            problems.append("tags_not_validated")
        if problems:
            invalid_assets.append({"index": index, "asset_id": asset_id, "reasons": problems})
        if split in split_counts:
            split_counts[split] += 1
        covered_tags.update(tags)
        rows.append({
            "asset_id": asset_id,
            "sha256": asset_hash,
            "tags": tags,
            "tags_validated": tags_validated,
            "split": split,
        })

    missing_tags = sorted(required - covered_tags)
    missing_splits = sorted(split for split in allowed if split_counts.get(split, 0) == 0)
    consent = _nonempty_text(consent_reference)
    complete = bool(
        bool(rows)
        and bool(consent)
        and not duplicate_hashes
        and not invalid_assets
        and not missing_tags
        and not missing_splits
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_NAME,
        "consent_reference": consent,
        "assets": rows,
        "asset_count": len(rows),
        "unique_asset_hashes": len(seen_hashes) == len(rows) and not duplicate_hashes,
        "duplicate_sha256": sorted(set(duplicate_hashes)),
        "required_tags": sorted(required),
        "covered_tags": sorted(covered_tags),
        "missing_tags": missing_tags,
        "allowed_splits": sorted(allowed),
        "split_counts": split_counts,
        "missing_splits": missing_splits,
        "invalid_assets": invalid_assets,
        "tags_validated": not invalid_assets and bool(rows),
        "complete": complete,
        "status": "complete" if complete else "failed",
    }


corpus_evidence = build_corpus_evidence


def build_human_review_evidence(
    reviews: Sequence[Mapping[str, Any]],
    *,
    critical_defect_labels: Sequence[str] = ("critical", "identity", "geometry", "likeness"),
    minimum_independent_reviewers: int = 2,
) -> Dict[str, Any]:
    """Summarize blinded independent reviews without collapsing uncertainty."""

    if not isinstance(reviews, Sequence) or isinstance(reviews, (str, bytes, bytearray)):
        raise TypeError("reviews must be a sequence of mappings")
    critical = {str(label).strip().lower() for label in critical_defect_labels if str(label).strip()}
    normalized: List[Dict[str, Any]] = []
    reviewer_ids: set[str] = set()
    decisions: List[str] = []
    critical_rows: List[int] = []
    invalid_rows: List[Dict[str, Any]] = []
    for index, review in enumerate(reviews):
        if not isinstance(review, Mapping):
            invalid_rows.append({"index": index, "reason": "review_not_mapping"})
            continue
        reviewer = _nonempty_text(review.get("reviewer_id") or review.get("reviewer"))
        decision = str(review.get("decision") or review.get("outcome") or "unresolved").strip().lower()
        if decision not in OUTCOMES:
            invalid_rows.append({"index": index, "reason": "unknown_decision"})
            continue
        blinded = review.get("blinded") is True
        confidence = review.get("confidence")
        confidence_valid = isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and 0 <= float(confidence) <= 1
        labels_raw = review.get("defect_labels", [])
        labels = sorted({str(label).strip() for label in labels_raw if str(label).strip()}) if isinstance(labels_raw, Sequence) and not isinstance(labels_raw, (str, bytes, bytearray)) else []
        critical_here = sorted(label for label in labels if label.lower() in critical)
        row = {
            "item_id": _nonempty_text(review.get("item_id") or review.get("asset_id") or review.get("case")),
            "recipe": _nonempty_text(review.get("recipe")),
            "reviewer_id": reviewer,
            "blinded": blinded,
            "decision": decision,
            "confidence": float(confidence) if confidence_valid else None,
            "defect_labels": labels,
            "critical_defect_labels": critical_here,
        }
        normalized.append(row)
        if reviewer:
            reviewer_ids.add(reviewer)
        decisions.append(decision)
        if critical_here:
            critical_rows.append(index)
        if not reviewer or not blinded or not confidence_valid:
            invalid_rows.append({
                "index": index,
                "reason": "missing_reviewer_blinding_or_confidence",
            })

    decision_counts = {outcome: decisions.count(outcome) for outcome in sorted(OUTCOMES)}
    disagreement = len(set(decisions)) > 1
    third_reviewer_required = disagreement or bool(critical_rows)
    third_reviewer_present = len(reviewer_ids) >= max(3, minimum_independent_reviewers + 1) if third_reviewer_required else True
    critical_review_count = sum(1 for row in normalized if row["critical_defect_labels"])
    critical_defect_confirmed = critical_review_count >= 2
    reviewers_sufficient = len(reviewer_ids) >= max(2, int(minimum_independent_reviewers))
    all_blinded = bool(normalized) and all(row["blinded"] for row in normalized)
    confidence_complete = bool(normalized) and all(row["confidence"] is not None for row in normalized)
    no_uncertain_outcomes = bool(normalized) and not any(
        row["decision"] in {"uncertain", "unresolved", "pending"} for row in normalized
    )
    approved_count = decision_counts["approved"]
    majority_approval = bool(approved_count > decision_counts["rejected"] and approved_count > decision_counts["uncertain"] + decision_counts["unresolved"] + decision_counts["pending"])
    human_accepted = bool(
        reviewers_sufficient
        and all_blinded
        and confidence_complete
        and no_uncertain_outcomes
        and majority_approval
        and (not third_reviewer_required or third_reviewer_present)
        and not critical_defect_confirmed
        and not invalid_rows
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_NAME,
        "reviews": normalized,
        "review_count": len(normalized),
        "independent_reviewer_count": len(reviewer_ids),
        "minimum_independent_reviewers": max(2, int(minimum_independent_reviewers)),
        "blinded": all_blinded,
        "confidence_complete": confidence_complete,
        "decision_counts": decision_counts,
        "disagreement": disagreement,
        "third_reviewer_required": third_reviewer_required,
        "third_reviewer_present": third_reviewer_present,
        "critical_defect_review_count": critical_review_count,
        "critical_defect_confirmed": critical_defect_confirmed,
        "invalid_reviews": invalid_rows,
        "outcome": "approved" if human_accepted else "rejected" if normalized and not any(row["decision"] in {"uncertain", "unresolved", "pending"} for row in normalized) else "uncertain",
        "human_accepted": human_accepted,
        "status": "complete" if human_accepted else "unresolved" if not normalized or invalid_rows or not confidence_complete else "failed",
    }


human_review_evidence = build_human_review_evidence


def seal_final_manifest(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep-copy and hash a manifest so later mutation is detectable."""

    if not isinstance(manifest, Mapping):
        raise TypeError("final manifest must be a mapping")
    body = _json_safe(copy.deepcopy(dict(manifest)))
    if not isinstance(body, Mapping):
        raise TypeError("final manifest must serialize as a mapping")
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_NAME,
        "sealed": True,
        "manifest": dict(body),
        "manifest_sha256": canonical_sha256(body),
    }


def verify_sealed_manifest(sealed: Mapping[str, Any]) -> Dict[str, Any]:
    """Verify that a sealed manifest still matches its immutable digest."""

    if not isinstance(sealed, Mapping) or sealed.get("sealed") is not True:
        return {"valid": False, "reason_code": "manifest_not_sealed"}
    body = sealed.get("manifest")
    expected = _normalise_sha256(sealed.get("manifest_sha256"))
    if not isinstance(body, Mapping) or expected is None:
        return {"valid": False, "reason_code": "manifest_hash_missing"}
    actual = canonical_sha256(body)
    return {
        "valid": actual == expected,
        "reason_code": "manifest_hash_matches" if actual == expected else "manifest_hash_mismatch",
        "manifest_sha256": actual,
        "expected_sha256": expected,
    }


def _runtime_snapshot_valid(runtime: Any) -> bool:
    if not isinstance(runtime, Mapping):
        return False
    snapshot = runtime.get("snapshot")
    expected = _normalise_sha256(runtime.get("snapshot_sha256"))
    return isinstance(snapshot, Mapping) and expected is not None and canonical_sha256(snapshot) == expected


def _render_evidence_valid(render: Any) -> bool:
    if not isinstance(render, Mapping):
        return False
    if render.get("render_completed") is not True:
        return False
    artifact = render.get("artifact")
    digest = artifact.get("sha256") if isinstance(artifact, Mapping) else None
    recipe = render.get("recipe")
    dimensions = render.get("dimensions")
    dtype = render.get("dtype")
    metadata = render.get("metadata")
    decode = render.get("decode")
    return bool(
        _normalise_sha256(digest)
        and isinstance(decode, Mapping)
        and decode.get("decodable") is True
        and isinstance(recipe, Mapping)
        and recipe.get("matches") is True
        and isinstance(dimensions, Mapping)
        and dimensions.get("matches") is True
        and isinstance(dtype, Mapping)
        and dtype.get("matches") is True
        and isinstance(metadata, Mapping)
        and metadata.get("presence_matches") is True
        and metadata.get("content_matches") is True
        and isinstance(render.get("hash"), Mapping)
        and render["hash"].get("matches") is True
    )


def _face_evidence_valid(face: Any) -> bool:
    if not isinstance(face, Mapping) or face.get("face_aware") is not True:
        return False
    probe = face.get("detector_probe")
    landmarks = face.get("landmarks")
    return bool(
        face.get("requested_full_mode") is True
        and isinstance(probe, Mapping)
        and probe.get("attempted") is True
        and probe.get("succeeded") is True
        and probe.get("initialized") is True
        and face.get("face_count_matches") is True
        and isinstance(landmarks, Mapping)
        and landmarks.get("matches_detected_faces") is True
        and _nonempty_text(face.get("parser_backend"))
        and _provider_value(face.get("actual_provider")) is not None
        and face.get("provider_verified") is True
    )


def _quality_evidence_valid(quality: Any, policy: Mapping[str, Any]) -> bool:
    if not isinstance(quality, Mapping):
        return False
    metric_vector = quality.get("metric_vector")
    roi = quality.get("roi_provenance")
    return bool(
        quality.get("evidence_complete") is True
        and isinstance(metric_vector, (Mapping, list))
        and bool(metric_vector)
        and isinstance(roi, (Mapping, list))
        and bool(roi)
        and quality.get("automatic_pass") is True
        and quality.get("outcome") == "approved"
        and not quality.get("detector_failures")
        and not quality.get("flagged_metrics")
        and _versions_consistent(quality, policy)
    )


def _corpus_evidence_valid(corpus: Any) -> bool:
    if not isinstance(corpus, Mapping):
        return False
    assets = corpus.get("assets")
    return bool(
        corpus.get("complete") is True
        and corpus.get("unique_asset_hashes") is True
        and bool(corpus.get("consent_reference"))
        and corpus.get("tags_validated") is True
        and isinstance(assets, list)
        and assets
        and corpus.get("asset_count") == len(assets)
        and not corpus.get("duplicate_sha256")
        and not corpus.get("invalid_assets")
        and not corpus.get("missing_tags")
        and not corpus.get("missing_splits")
    )


def _human_evidence_valid(human: Any) -> bool:
    if not isinstance(human, Mapping):
        return False
    return bool(
        human.get("status") == "complete"
        and human.get("human_accepted") is True
        and human.get("independent_reviewer_count", 0) >= 2
        and human.get("blinded") is True
        and human.get("confidence_complete") is True
        and not human.get("invalid_reviews")
        and not human.get("critical_defect_confirmed")
        and (not human.get("third_reviewer_required") or human.get("third_reviewer_present") is True)
    )


def _versions_consistent(quality: Mapping[str, Any], policy: Mapping[str, Any]) -> bool:
    metric_version = _nonempty_text(quality.get("metric_version"))
    threshold_version = _nonempty_text(quality.get("threshold_version"))
    required_metric = _nonempty_text(policy.get("metric_version"))
    required_threshold = _nonempty_text(policy.get("threshold_version"))
    return bool(
        metric_version
        and threshold_version
        and (required_metric is None or metric_version == required_metric)
        and (required_threshold is None or threshold_version == required_threshold)
    )


def certification_gate(
    sealed_manifest: Mapping[str, Any],
    *,
    policy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate every v2 prerequisite and compute certification from scratch.

    Any top-level ``final_certified`` value supplied by a caller is ignored.
    The returned value is a decision report, while the sealed manifest remains
    the immutable evidence record.
    """

    seal = verify_sealed_manifest(sealed_manifest)
    if not seal["valid"]:
        return {
            "schema_version": SCHEMA_VERSION,
            "contract": CONTRACT_NAME,
            "manifest_valid": False,
            "manifest_sha256": seal.get("manifest_sha256"),
            "prerequisites": {},
            "reason_codes": [seal["reason_code"]],
            "final_certified": False,
        }

    manifest = sealed_manifest.get("manifest", {})
    embedded_policy = manifest.get("policy", {}) if isinstance(manifest, Mapping) else {}
    policy_value = dict(_json_safe(policy if policy is not None else embedded_policy or {}))
    evidence = manifest.get("evidence", manifest) if isinstance(manifest, Mapping) else {}
    runtime = evidence.get("runtime_doctor") if isinstance(evidence, Mapping) else None
    renders = evidence.get("render_artifacts") if isinstance(evidence, Mapping) else None
    face = evidence.get("face_aware") if isinstance(evidence, Mapping) else None
    quality = evidence.get("automatic_quality") if isinstance(evidence, Mapping) else None
    corpus = evidence.get("corpus") if isinstance(evidence, Mapping) else None
    human = evidence.get("human_review") if isinstance(evidence, Mapping) else None

    prerequisites: Dict[str, bool] = {
        "runtime_doctor": _runtime_snapshot_valid(runtime),
        "render_artifacts": isinstance(renders, Sequence) and not isinstance(renders, (str, bytes, bytearray)) and bool(renders) and all(_render_evidence_valid(item) for item in renders),
        "face_aware": _face_evidence_valid(face),
        "automatic_quality": _quality_evidence_valid(quality, policy_value),
        "corpus": _corpus_evidence_valid(corpus),
        "human_review": _human_evidence_valid(human),
        "versions": isinstance(quality, Mapping) and _versions_consistent(quality, policy_value),
    }
    reasons = [name + "_missing_or_invalid" for name, passed in prerequisites.items() if not passed]
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_NAME,
        "manifest_valid": True,
        "manifest_sha256": seal["manifest_sha256"],
        "policy": policy_value,
        "prerequisites": prerequisites,
        "reason_codes": reasons,
        "final_certified": all(prerequisites.values()),
    }


def build_final_manifest(
    *,
    runtime_doctor: Mapping[str, Any],
    render_artifacts: Sequence[Mapping[str, Any]],
    face_aware: Mapping[str, Any],
    automatic_quality: Mapping[str, Any],
    corpus: Mapping[str, Any],
    human_review: Mapping[str, Any],
    policy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a sealed v2 manifest from its independently collected evidence."""

    return seal_final_manifest({
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_NAME,
        "evidence": {
            "runtime_doctor": _json_safe(runtime_doctor),
            "render_artifacts": _json_safe(list(render_artifacts)),
            "face_aware": _json_safe(face_aware),
            "automatic_quality": _json_safe(automatic_quality),
            "corpus": _json_safe(corpus),
            "human_review": _json_safe(human_review),
        },
        "policy": _json_safe(policy or {}),
    })


final_manifest = build_final_manifest
final_certification_gate = certification_gate


__all__ = [
    "SCHEMA_VERSION",
    "CONTRACT_NAME",
    "CONTRACT_VERSION",
    "OUTCOMES",
    "canonical_json",
    "canonical_sha256",
    "sha256_file",
    "sanitize_runtime_doctor_snapshot",
    "sanitize_runtime_doctor",
    "build_runtime_doctor_evidence",
    "runtime_doctor_evidence",
    "build_render_artifact_evidence",
    "render_artifact_evidence",
    "build_face_aware_evidence",
    "face_aware_evidence",
    "build_automatic_quality_evidence",
    "automatic_quality_evidence",
    "quality_vector_evidence",
    "build_corpus_evidence",
    "corpus_evidence",
    "build_human_review_evidence",
    "human_review_evidence",
    "seal_final_manifest",
    "verify_sealed_manifest",
    "certification_gate",
    "final_certification_gate",
    "build_final_manifest",
    "final_manifest",
]

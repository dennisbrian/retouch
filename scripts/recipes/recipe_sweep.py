#!/usr/bin/env python3
"""Render one input image through the curated retouch recipe catalog.

This is a visual-validation utility: it exports one output per recipe plus a
contact sheet and manifest so recipe drift can be reviewed quickly.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli import _apply_global_finish  # noqa: E402
from retouch import RetouchEngine  # noqa: E402
from retouch.batch_processor import generate_contact_sheet  # noqa: E402
from retouch.io import (  # noqa: E402
    IMAGE_EXTENSIONS,
    encode_write_params,
    imread_exif,
    make_comparison,
    resize_for_processing,
    write_image_with_icc,
)
from retouch.recipes import (  # noqa: E402
    CONDITIONAL_RECIPE_NAMES,
    CURATED_RECIPE_NAMES,
    RECOMMENDED_RECIPE_NAMES,
    RECIPES,
)


EVIDENCE_VERSION = 2
# These identifiers are part of the evidence contract.  Changing detector
# definitions or threshold policy requires a new value rather than silently
# reusing an old certification vector.
QA_METRIC_VERSION = "qa-vector-v2"
QA_THRESHOLD_VERSION = "qa-thresholds-v1"


def _json_safe(value: Any) -> Any:
    """Convert runtime evidence into deterministic JSON-safe values.

    Detector details are normally scalar dictionaries, but test doubles and
    optional backends may expose dataclasses, NumPy scalars, arrays, or native
    objects. Small arrays retain their values; larger arrays retain shape,
    dtype, range, and a content hash so serialization never silently drops
    provenance or attempts to encode a non-JSON native object.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "length": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        payload: Dict[str, Any] = {
            "type": "ndarray",
            "shape": [int(item) for item in array.shape],
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
        if array.size:
            try:
                payload.update({
                    "min": _json_safe(np.min(array)),
                    "max": _json_safe(np.max(array)),
                })
            except (TypeError, ValueError):
                pass
        if array.size <= 4096:
            payload["values"] = _json_safe(array.tolist())
        else:
            payload["values_omitted"] = True
        return payload
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]

    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in attributes.items()
            if not str(key).startswith("_")
        }
    return repr(value)


def _safe_number(value: Any) -> Optional[float]:
    """Return a finite numeric value, or ``None`` for unavailable metrics."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _array_provenance(value: Any, source: str) -> Dict[str, Any]:
    """Describe an ROI/mask without embedding a potentially huge pixel array."""
    array = np.asarray(value)
    payload: Dict[str, Any] = {
        "available": True,
        "source": source,
        "shape": [int(item) for item in array.shape],
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }
    if array.size:
        try:
            payload.update({
                "nonzero": int(np.count_nonzero(array)),
                "min": _json_safe(np.min(array)),
                "max": _json_safe(np.max(array)),
            })
        except (TypeError, ValueError):
            pass
    return payload


def _extract_roi_fields(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep ROI/mask/region fields from a detector's raw result intact."""
    roi: Dict[str, Any] = {}
    for key, item in value.items():
        key_text = str(key).lower()
        if any(token in key_text for token in ("roi", "mask", "region", "bbox", "box")):
            roi[str(key)] = _json_safe(item)
    return roi


def _normalise_qa_observation(
    detector_name: str,
    value: Any,
    source: str,
) -> Dict[str, Any]:
    """Normalize one QA result without throwing away unflagged/raw fields."""
    if dataclasses.is_dataclass(value):
        raw: Dict[str, Any] = dict(dataclasses.asdict(value))
    elif isinstance(value, Mapping):
        raw = dict(value)
    elif isinstance(getattr(value, "__dict__", None), Mapping):
        raw = dict(vars(value))
    else:
        raw = {
            "value": value,
            "available": False,
            "error": "QA observation was not mapping-like",
        }

    detector = str(raw.get("detector") or detector_name)
    details_value = raw.get("details")
    if isinstance(details_value, Mapping):
        details = dict(details_value)
    else:
        canonical = {
            "detector", "score", "threshold", "flagged", "message",
            "available", "error", "status", "source",
        }
        details = {key: item for key, item in raw.items() if key not in canonical}

    available_value = raw.get("available", details.get("available", True))
    available = bool(available_value) if available_value is not None else True
    error = raw.get("error") or details.get("error")
    flagged = bool(raw.get("flagged", False))
    if error is not None and raw.get("flagged") is None:
        flagged = True
    if not available:
        status = "unavailable"
    elif error is not None:
        status = "error"
    elif flagged:
        status = "flagged"
    else:
        status = "ok"

    roi_fields = _extract_roi_fields(raw)
    roi_fields.update(_extract_roi_fields(details))
    observation: Dict[str, Any] = {
        # These fields intentionally retain the old warning-row shape.
        "detector": detector,
        "score": _safe_number(raw.get("score")),
        "threshold": _safe_number(raw.get("threshold", details.get("threshold"))),
        "flagged": flagged,
        "message": str(raw.get("message") or ""),
        # New evidence fields.
        "available": available,
        "status": status,
        "source": source,
        "metric": detector,
        "version": QA_METRIC_VERSION,
        "threshold_version": QA_THRESHOLD_VERSION,
        "details": _json_safe(details),
        "raw_details": _json_safe(raw),
        "roi_provenance": _json_safe(roi_fields),
        # ``roi`` is the short contract spelling; keep the descriptive alias
        # for existing consumers and human-readable manifests.
        "roi": _json_safe(roi_fields),
    }
    if error is not None:
        observation["error"] = str(error)
    return observation


def _iter_qa_values(raw: Any) -> Iterable[Tuple[str, Any]]:
    """Yield QA observations from either a detector mapping or a list."""
    if isinstance(raw, Mapping):
        for name, value in raw.items():
            yield str(name), value
    elif isinstance(raw, (list, tuple)):
        for index, value in enumerate(raw):
            name = getattr(value, "detector", None)
            if isinstance(value, Mapping):
                name = value.get("detector", name)
            yield str(name or f"observation_{index}"), value
    elif raw is not None:
        name = getattr(raw, "detector", None)
        yield str(name or "qa_observation"), raw


def _extract_result_qa(result: Any) -> List[Dict[str, Any]]:
    """Extract ``result.qa`` without filtering out clean observations."""
    values = getattr(result, "qa", []) if result is not None else []
    return [
        _normalise_qa_observation(name, value, "processing_result.qa")
        for name, value in _iter_qa_values(values)
    ]


def _extract_raw_qa(result: Any) -> Optional[Any]:
    """Find an explicitly attached complete QA vector, if a backend provides one."""
    if result is None:
        return None
    for attribute in ("qa_observations", "qa_results", "raw_qa", "qa_raw"):
        value = getattr(result, attribute, None)
        if value is not None:
            return value
    return None


def _extract_context_qa(result: Any) -> Optional[Any]:
    """Return the engine context's legacy raw QA map for provenance only."""
    if result is None:
        return None
    params = getattr(result, "params", None)
    return getattr(params, "_qa_results", None) if params is not None else None


def _qa_runner_kwargs(
    result: Any,
    reference_bgr: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Build QA-runner inputs from result-owned masks and context."""
    params = getattr(result, "params", None) if result is not None else None
    face_skin_mask = getattr(result, "skin_mask", None) if result is not None else None
    person_mask = getattr(result, "person_mask", None) if result is not None else None
    body_skin_mask = getattr(result, "body_skin_mask", None) if result is not None else None
    return {
        "skin_mask": person_mask if person_mask is not None else face_skin_mask,
        "reference_img_bgr": reference_bgr,
        "img_before": reference_bgr,
        "person_mask": person_mask,
        "face_skin_mask": face_skin_mask,
        "body_skin_mask": body_skin_mask,
        "mark_policy": getattr(params, "mark_policy", None) if params is not None else None,
        "warp_field": getattr(params, "_aa6_warp_field", None) if params is not None else None,
    }


def _default_qa_runner(image_bgr: np.ndarray, **kwargs: Any) -> Any:
    """Run the complete QA vector; kept injectable for native-free tests."""
    from retouch import qa_detectors

    return qa_detectors.run_all(image_bgr, **kwargs)


def _merge_qa_observations(
    result_observations: List[Dict[str, Any]],
    complete_observations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge legacy warning metadata into the complete detector vector."""
    merged: List[Dict[str, Any]] = [dict(item) for item in complete_observations]
    by_detector = {item["detector"]: index for index, item in enumerate(merged)}
    for warning in result_observations:
        detector = warning["detector"]
        if detector not in by_detector:
            merged.append(dict(warning))
            by_detector[detector] = len(merged) - 1
            continue
        target = merged[by_detector[detector]]
        if warning.get("flagged"):
            target["flagged"] = True
            target["status"] = "flagged" if target.get("available", True) else "unavailable"
        if warning.get("message"):
            target["message"] = warning["message"]
        if warning.get("threshold") is not None:
            target["threshold"] = warning["threshold"]
        if target.get("score") is None and warning.get("score") is not None:
            target["score"] = warning["score"]
        # Keep the actual ProcessingResult warning distinct from context/raw
        # overlays; later provenance merges must not overwrite it.
        target.setdefault("result_qa", warning)
        if warning.get("source") == "processing_context._qa_results":
            target["context_qa"] = warning
        elif warning.get("source") == "processing_result.raw_qa":
            target["raw_observation"] = warning
    return merged


def _extract_qa_evidence(
    result: Any,
    output_bgr: np.ndarray,
    reference_bgr: Optional[np.ndarray] = None,
    *,
    global_only: bool,
    qa_runner: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Collect complete QA observations and explicit failure states.

    The helper is deterministic for a given result and injected runner. It
    never assumes that an absent ``result.qa`` means clean output: the runner
    is asked for the full vector and the legacy warning list is retained
    separately in ``result_qa``.
    """
    result_observations = _extract_result_qa(result)
    raw_qa = _extract_raw_qa(result)
    context_qa = _extract_context_qa(result)
    raw_observations = [
        _normalise_qa_observation(name, value, "processing_result.raw_qa")
        for name, value in _iter_qa_values(raw_qa)
    ]
    context_observations = [
        _normalise_qa_observation(name, value, "processing_context._qa_results")
        for name, value in _iter_qa_values(context_qa)
    ]
    runner_observations: List[Dict[str, Any]] = []
    runner_status = "not_run"
    runner_error: Optional[str] = None

    runner = qa_runner or _default_qa_runner
    try:
        try:
            raw_runner_result = runner(
                output_bgr,
                **_qa_runner_kwargs(result, reference_bgr),
            )
        except TypeError as exc:
            # Tiny test doubles often accept only the image. Retry only for
            # an argument-shape mismatch; genuine TypeErrors from the runner
            # remain explicit failures below.
            if "unexpected keyword" not in str(exc) and "positional argument" not in str(exc):
                raise
            raw_runner_result = runner(output_bgr)
        runner_observations = [
            _normalise_qa_observation(name, value, "qa_detectors.run_all")
            for name, value in _iter_qa_values(raw_runner_result)
        ]
        runner_status = "complete" if runner_observations else "empty"
    except Exception as exc:  # noqa: BLE001 - evidence must survive QA failures
        runner_status = "error"
        runner_error = f"{type(exc).__name__}: {exc}"
        runner_observations = [_normalise_qa_observation(
            "qa_runner",
            {
                "score": 1.0,
                "flagged": True,
                "available": False,
                "error": runner_error,
                "message": "Complete QA vector could not be collected",
            },
            "qa_runner",
        )]

    complete_source = runner_observations or raw_observations or result_observations
    observations = _merge_qa_observations(result_observations, complete_source)
    if context_observations:
        observations = _merge_qa_observations(context_observations, observations)
    if raw_observations and runner_observations:
        observations = _merge_qa_observations(raw_observations, observations)

    coverage_complete = runner_status == "complete" or bool(raw_observations)
    status = "diagnostic_only" if global_only else ("complete" if coverage_complete else "partial")
    return {
        "version": EVIDENCE_VERSION,
        "metric_version": QA_METRIC_VERSION,
        "threshold_version": QA_THRESHOLD_VERSION,
        "status": status,
        "certifying": bool(not global_only and coverage_complete and runner_status != "error"),
        "coverage_complete": coverage_complete,
        "complete": coverage_complete,
        "metrics": observations,
        "runner": {
            "status": runner_status,
            "name": getattr(qa_runner, "__name__", "default_qa_runner") if qa_runner else "retouch.qa_detectors.run_all",
            "error": runner_error,
        },
        "observations": observations,
        "result_qa": result_observations,
        "raw_qa": raw_observations,
        "context_qa": context_observations,
        "flagged": [item for item in observations if item.get("flagged")],
        "unavailable": [item for item in observations if not item.get("available", True)],
    }


def _landmark_evidence(landmarks: Any) -> Dict[str, Any]:
    """Serialize normalized landmark coordinates for one detected face."""
    values = getattr(landmarks, "landmark", None)
    if values is None and isinstance(landmarks, (list, tuple)):
        values = landmarks
    if values is None and isinstance(landmarks, np.ndarray):
        values = landmarks.tolist()
    if values is None:
        return {"available": False, "count": 0, "landmarks": []}

    points: List[Any] = []
    for item in values:
        if isinstance(item, Mapping):
            point = {
                "x": _safe_number(item.get("x")),
                "y": _safe_number(item.get("y")),
                "z": _safe_number(item.get("z", 0.0)),
            }
        elif isinstance(item, (list, tuple, np.ndarray)):
            point = {
                "x": _safe_number(item[0]) if len(item) > 0 else None,
                "y": _safe_number(item[1]) if len(item) > 1 else None,
                "z": _safe_number(item[2]) if len(item) > 2 else 0.0,
            }
        else:
            point = {
                "x": _safe_number(getattr(item, "x", None)),
                "y": _safe_number(getattr(item, "y", None)),
                "z": _safe_number(getattr(item, "z", 0.0)),
            }
        points.append(point)
    return {"available": True, "count": len(points), "landmarks": points}


def _extract_face_evidence(result: Any, *, global_only: bool) -> Dict[str, Any]:
    """Record actual per-image face count, landmarks, boxes, and ROI masks."""
    if global_only:
        return {
            "status": "not_run",
            "available": False,
            "face_count": None,
            "face_count_source": "global_only_requested",
            "faces": [],
        }
    if result is None:
        return {
            "status": "not_collected",
            "available": False,
            "face_count": None,
            "face_count_source": "processing_failed_before_result",
            "faces": [],
        }

    contexts = getattr(result, "face_contexts", None) if result is not None else None
    candidates = list(contexts or [])
    if not candidates and result is not None:
        candidates = list(getattr(result, "faces", None) or getattr(result, "detected_faces", None) or [])

    faces: List[Dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        face_data = getattr(candidate, "face_data", candidate)
        landmarks = getattr(face_data, "landmarks", getattr(candidate, "landmarks", None))
        regions = getattr(candidate, "regions", None)
        region_masks: Dict[str, Any] = {}
        if regions is not None:
            region_attributes = getattr(regions, "__dict__", {})
            for name, value in region_attributes.items():
                if value is not None and isinstance(value, np.ndarray):
                    region_masks[name] = _array_provenance(value, f"face_contexts[{index}].regions.{name}")
        landmark_record = _landmark_evidence(landmarks)
        faces.append({
            "index": index,
            "bbox": _json_safe(getattr(face_data, "bbox", getattr(candidate, "bbox", None))),
            "ied": _safe_number(getattr(face_data, "ied", getattr(candidate, "ied", None))),
            "confidence": _safe_number(getattr(face_data, "confidence", getattr(candidate, "confidence", None))),
            "confidence_source": str(getattr(face_data, "confidence_source", "unknown")),
            "landmark_evidence": landmark_record,
            "landmarks": landmark_record.get("landmarks", []),
            "region_masks": region_masks,
        })

    reported_count = getattr(result, "face_count", None) if result is not None else None
    face_count = int(reported_count) if reported_count is not None else len(faces)
    return {
        "status": "detected" if face_count else "initialized_no_faces",
        "available": True,
        "face_count": face_count,
        "face_count_source": "processing_result.face_count" if reported_count is not None else "face_contexts",
        "faces": faces,
        "context_count": len(candidates),
        "landmarks_complete": bool(faces) and all(face["landmark_evidence"]["available"] for face in faces),
    }


def _extract_detector_evidence(
    engine: Any,
    face_evidence: Mapping[str, Any],
    *,
    global_only: bool,
) -> Dict[str, Any]:
    """Capture initialized backend state, including unavailable failures."""
    if global_only:
        return {
            "status": "not_run",
            "available": False,
            "initialized": False,
            "probed": False,
            "probe_succeeded": False,
            "backend": "not_run",
            "reason_code": "global_only_requested",
            "reason": "global-only diagnostic intentionally skipped face detection",
            "face_count": None,
            "certifying": False,
        }

    detector = getattr(engine, "_detector", None) if engine is not None else None
    if detector is None:
        return {
            "status": "unavailable",
            "available": False,
            "initialized": False,
            "probed": False,
            "probe_succeeded": False,
            "backend": "unavailable",
            "reason_code": "detector_missing",
            "reason": "engine did not expose a face detector",
            "face_count": face_evidence.get("face_count"),
            "certifying": False,
        }

    runtime_status: Dict[str, Any] = {}
    status_method = getattr(detector, "runtime_status", None)
    if callable(status_method):
        try:
            runtime_status = dict(status_method())
        except Exception as exc:  # noqa: BLE001 - preserve detector evidence
            runtime_status = {"status_error": f"{type(exc).__name__}: {exc}"}
    available = bool(runtime_status.get("available", getattr(detector, "available", False)))
    backend = str(runtime_status.get("backend", getattr(detector, "backend_name", "unknown")))
    reason = runtime_status.get("reason", getattr(detector, "unavailable_reason", None))
    initialized = bool(
        runtime_status.get("probe_state") == "initialized"
        or available
        or reason is not None
        or backend not in {"", "unknown", "uninitialized"}
    )
    return {
        "status": "initialized" if initialized and available else "unavailable",
        "available": available,
        "initialized": initialized,
        "probed": bool(runtime_status.get("probe_state") in {"initialized", "blocked"} or initialized),
        "probe_succeeded": bool(available),
        "detector_initialized": initialized,
        "backend": backend,
        "provider": str(getattr(detector, "provider", backend)),
        "probe_state": runtime_status.get("probe_state", "initialized" if initialized else "unprobed"),
        "reason_code": "detector_initialized" if available else "detector_unavailable",
        "reason": reason,
        "runtime_status": _json_safe(runtime_status),
        "face_count": face_evidence.get("face_count"),
        "certifying": bool(available and initialized),
    }


def _extract_parser_evidence(engine: Any, *, global_only: bool) -> Dict[str, Any]:
    """Capture parser initialization and the actual active ONNX providers."""
    if global_only:
        return {
            "status": "not_run",
            "available": False,
            "initialized": False,
            "backend": "not_run",
            "provider_evidence": {
                "active": [],
                "active_providers": [],
                "actual_provider": None,
                "available": [],
                "verified": False,
            },
            "reason_code": "global_only_requested",
        }
    parser = getattr(engine, "_parser", None) if engine is not None else None
    if parser is None:
        return {
            "status": "unavailable",
            "available": False,
            "initialized": False,
            "backend": "unavailable",
            "provider_evidence": {
                "active": [],
                "active_providers": [],
                "actual_provider": None,
                "available": [],
                "verified": False,
            },
            "reason_code": "parser_missing",
        }

    session = getattr(parser, "_sess", None)
    active: List[str] = []
    provider_error = None
    if session is not None:
        try:
            active = [str(item) for item in session.get_providers()]
        except Exception as exc:  # noqa: BLE001 - keep provider failure explicit
            provider_error = f"{type(exc).__name__}: {exc}"
    available = bool(getattr(parser, "available", session is not None))
    backend = str(getattr(parser, "backend_name", "bisenet_onnx" if session is not None else "landmark_only"))
    if session is not None and not active and provider_error:
        status = "error"
    elif session is not None:
        status = "initialized"
    else:
        status = "fallback"
    providers = getattr(parser, "providers", getattr(parser, "available_providers", [])) or []
    provider_evidence = {
        "active": active,
        "active_providers": active,
        "actual_provider": active[0] if active else None,
        "available": [str(item) for item in providers],
        "requested": _json_safe(getattr(parser, "requested_providers", [])),
        "verified": bool(active),
    }
    if provider_error:
        provider_evidence["error"] = provider_error
    return {
        "status": status,
        "available": available,
        "initialized": True,
        "backend": backend,
        "model_path": _json_safe(getattr(parser, "_model_path", None)),
        "provider_evidence": provider_evidence,
        "reason_code": "parser_initialized" if session is not None else "parser_session_unavailable",
        "reason": None if session is not None else "parser is using landmark-only fallback",
    }


def _extract_roi_provenance(result: Any) -> Dict[str, Any]:
    """Record the result-owned masks and their coordinate-space provenance."""
    masks: Dict[str, Any] = {}
    if result is not None:
        for attribute in ("skin_mask", "skin_hair_mask", "sharpen_mask", "lips_mask", "person_mask", "body_skin_mask"):
            value = getattr(result, attribute, None)
            if value is not None:
                masks[attribute] = _array_provenance(value, f"processing_result.{attribute}")
    return {
        "coordinate_space": "processed_output_pixels",
        "image_shape": [int(item) for item in getattr(result, "shape", ())[:2]] if result is not None else [],
        "masks": masks,
        "available": bool(masks),
    }


def _extract_runtime_evidence(
    result: Any,
    engine: Any,
    *,
    global_only: bool,
) -> Dict[str, Any]:
    """Assemble per-output detector, face, parser, provider, and ROI evidence."""
    face_evidence = _extract_face_evidence(result, global_only=global_only)
    detector_evidence = _extract_detector_evidence(
        engine,
        face_evidence,
        global_only=global_only,
    )
    parser_evidence = _extract_parser_evidence(engine, global_only=global_only)
    diagnostics = getattr(result, "runtime_diagnostics", {}) if result is not None else {}
    face_aware_run = bool(
        not global_only
        and result is not None
        and detector_evidence.get("certifying")
    )
    actual_mode = "global-only" if global_only else (
        "face-aware" if face_aware_run
        else "face-aware-not-completed" if detector_evidence.get("certifying")
        else "global-only-fallback"
    )
    return {
        "version": EVIDENCE_VERSION,
        "mode": actual_mode,
        "certifying": face_aware_run,
        "face_aware_run": face_aware_run,
        "detector": detector_evidence,
        "face": face_evidence,
        "parser": parser_evidence,
        "provider_evidence": parser_evidence.get("provider_evidence", {}),
        "runtime_diagnostics": _json_safe(diagnostics),
        "roi_provenance": _extract_roi_provenance(result),
    }


def _parse_recipe_list(raw: Optional[str]) -> List[str]:
    if raw is None or raw.strip().lower() in {"recommended", "default"}:
        return list(RECOMMENDED_RECIPE_NAMES)
    if raw.strip().lower() == "curated":
        return list(CURATED_RECIPE_NAMES)
    if raw.strip().lower() == "conditional":
        return list(CONDITIONAL_RECIPE_NAMES)
    if raw.strip().lower() == "all":
        return sorted(RECIPES.keys())
    names = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [name for name in names if name not in RECIPES]
    if unknown:
        raise ValueError(f"unknown recipe(s): {', '.join(unknown)}")
    return names


def _safe_stem(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def _write_image(path: Path, img_bgr: np.ndarray, fmt: str, quality: int) -> None:
    if fmt == "png":
        write_image_with_icc(str(path), img_bgr, bit_depth=8, quality=quality)
    else:
        ok = cv2.imwrite(str(path), img_bgr, encode_write_params(fmt, quality))
        if not ok:
            raise IOError(f"failed to write output image: {path}")
    if not path.exists() or path.stat().st_size <= 0:
        raise IOError(f"output image was not written: {path}")


def _checkpoint_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    """Atomically save progress so a native crash leaves a resumable sweep."""
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _process_recipe_with_evidence(
    img_bgr: np.ndarray,
    recipe: str,
    engine: Optional[RetouchEngine],
    global_only: bool,
    fail_on_qa: bool,
    qa_runner: Optional[Callable[..., Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """Process one recipe and return the image, backend result, and evidence."""
    result_object: Any = None
    if global_only:
        result_object = _apply_global_finish(img_bgr, {"recipe": recipe})
        result = np.asarray(result_object)
    else:
        if engine is None:
            raise RuntimeError("engine is required unless global_only=True")
        result_object = engine.process(img_bgr, recipe=recipe)
        result = np.asarray(result_object)

    qa_evidence = _extract_qa_evidence(
        result_object,
        result,
        img_bgr,
        global_only=global_only,
        qa_runner=qa_runner,
    )
    runtime_evidence = _extract_runtime_evidence(
        result_object,
        engine,
        global_only=global_only,
    )

    return result, result_object, {
        "qa": qa_evidence,
        "runtime": runtime_evidence,
    }


def _process_recipe(
    img_bgr: np.ndarray,
    recipe: str,
    engine: Optional[RetouchEngine],
    global_only: bool,
    fail_on_qa: bool,
    qa_runner: Optional[Callable[..., Any]] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Backward-compatible wrapper returning the full QA vector as ``qa``."""
    result, _result_object, evidence = _process_recipe_with_evidence(
        img_bgr=img_bgr,
        recipe=recipe,
        engine=engine,
        global_only=global_only,
        fail_on_qa=fail_on_qa,
        qa_runner=qa_runner,
    )
    if fail_on_qa and not global_only and evidence["qa"]["flagged"]:
        flagged = ", ".join(row["detector"] for row in evidence["qa"]["flagged"])
        raise RuntimeError(f"QA_FAIL: {flagged}")
    return result, evidence["qa"]["observations"]


def run_sweep(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")
    if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"unsupported input image extension: {input_path.suffix}")

    recipes = _parse_recipe_list(args.recipes)
    skip = set(_parse_recipe_list(args.skip)) if args.skip else set()
    recipes = [name for name in recipes if name not in skip]
    if not recipes:
        raise ValueError("no recipes selected")

    img_bgr = imread_exif(input_path)
    original_shape = img_bgr.shape[:2]
    img_bgr, scale = resize_for_processing(img_bgr, args.max_dim)

    source_path = output_dir / f"00_source.{args.format}"
    _write_image(source_path, img_bgr, args.format, args.quality)

    manifest_path = output_dir / "manifest.json"
    manifest: Dict[str, Any] = {
        "evidence_version": EVIDENCE_VERSION,
        "input": str(input_path),
        "output_dir": str(output_dir),
        "recipes_requested": recipes,
        "global_only": bool(args.global_only),
        "diagnostic_only": bool(args.global_only),
        "face_aware_run": False,
        "certifying": False,
        "max_dim": args.max_dim,
        "original_shape": list(original_shape),
        "processed_shape": list(img_bgr.shape[:2]),
        "scale": scale,
        "outputs": [],
    }
    if args.resume and manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("input") != str(input_path):
            raise ValueError("cannot resume: manifest input does not match the requested image")
        if prior.get("recipes_requested") != recipes:
            raise ValueError("cannot resume: recipe list does not match the existing manifest")
        manifest = prior
        manifest.setdefault("evidence_version", EVIDENCE_VERSION)
        manifest.setdefault("diagnostic_only", bool(args.global_only))
        manifest.setdefault("face_aware_run", False)
        manifest.setdefault("certifying", False)

    completed_rows = {
        row.get("recipe"): row
        for row in manifest.get("outputs", [])
        if (
            row.get("status") == "done"
            and row.get("evidence_version") == EVIDENCE_VERSION
            and isinstance(row.get("qa_evidence"), Mapping)
            and isinstance(row.get("runtime_evidence"), Mapping)
            and (output_dir / str(row.get("output", ""))).exists()
        )
    }

    output_paths: List[Path] = [
        output_dir / str(row["output"])
        for row in completed_rows.values()
    ]
    _checkpoint_manifest(manifest_path, manifest)
    engine: Optional[RetouchEngine] = None
    if not args.global_only and not args.restart_engine_per_recipe:
        engine = RetouchEngine()

    try:
        for idx, recipe in enumerate(recipes, start=1):
            if recipe in completed_rows:
                print(f"[{idx:02d}/{len(recipes):02d}] {recipe}: resume skip")
                continue
            started = time.perf_counter()
            out_name = f"{idx:02d}_{_safe_stem(recipe)}.{args.format}"
            out_path = output_dir / out_name
            compare_path = output_dir / f"{idx:02d}_{_safe_stem(recipe)}_compare.{args.format}"

            row: Dict[str, Any] = {
                "evidence_version": EVIDENCE_VERSION,
                "recipe": recipe,
                "output": out_name,
                "status": "pending",
                "global_only": bool(args.global_only),
                "diagnostic_only": bool(args.global_only),
                "qa": [],
                "qa_warnings": [],
                "qa_evidence": {
                    "version": EVIDENCE_VERSION,
                    "status": "not_collected",
                    "certifying": False,
                    "coverage_complete": False,
                    "observations": [],
                    "flagged": [],
                    "unavailable": [],
                },
            }
            recipe_engine = engine
            try:
                if not args.global_only and args.restart_engine_per_recipe:
                    recipe_engine = RetouchEngine()
                result, _result_object, evidence = _process_recipe_with_evidence(
                    img_bgr=img_bgr,
                    recipe=recipe,
                    engine=recipe_engine,
                    global_only=bool(args.global_only),
                    # Store the complete evidence before applying the optional
                    # failure gate so a QA failure is itself auditable.
                    fail_on_qa=False,
                )
                qa_evidence = evidence["qa"]
                runtime_evidence = evidence["runtime"]
                row.update({
                    "qa": qa_evidence["observations"],
                    "qa_warnings": qa_evidence["flagged"],
                    "qa_evidence": qa_evidence,
                    "quality_evidence": qa_evidence,
                    "automatic_quality": qa_evidence,
                    "runtime_evidence": runtime_evidence,
                    "detector_evidence": runtime_evidence["detector"],
                    "face_evidence": runtime_evidence["face"],
                    "parser_evidence": runtime_evidence["parser"],
                    "provider_evidence": runtime_evidence["provider_evidence"],
                    "roi_provenance": runtime_evidence["roi_provenance"],
                    "runtime_diagnostics": runtime_evidence["runtime_diagnostics"],
                    "face_count": runtime_evidence["face"].get("face_count"),
                    "face_aware_run": bool(runtime_evidence["face_aware_run"]),
                    "certifying": bool(runtime_evidence["certifying"] and qa_evidence["certifying"]),
                })
                if args.fail_on_qa and not args.global_only and qa_evidence["flagged"]:
                    flagged = ", ".join(row["detector"] for row in qa_evidence["flagged"])
                    raise RuntimeError(f"QA_FAIL: {flagged}")
                _write_image(out_path, result, args.format, args.quality)
                output_paths.append(out_path)

                if args.compare:
                    make_comparison(img_bgr, result, compare_path, args.format, args.quality)
                    row["compare"] = compare_path.name

                row.update(
                    {
                        "status": "done",
                        "seconds": round(time.perf_counter() - started, 3),
                    }
                )
                print(f"[{idx:02d}/{len(recipes):02d}] {recipe}: wrote {out_name}")
            except cv2.error as exc:
                row.update({"status": "failed", "error": f"cv2.error: {exc}"})
                print(f"[{idx:02d}/{len(recipes):02d}] {recipe}: FAILED cv2.error: {exc}")
                if not args.keep_going:
                    raise
            except Exception as exc:
                row.update({"status": "failed", "error": str(exc)})
                print(f"[{idx:02d}/{len(recipes):02d}] {recipe}: FAILED {exc}")
                if not args.keep_going:
                    raise
            finally:
                row.setdefault(
                    "runtime_evidence",
                    _extract_runtime_evidence(
                        None,
                        recipe_engine,
                        global_only=bool(args.global_only),
                    ),
                )
                row.setdefault("detector_evidence", row["runtime_evidence"]["detector"])
                row.setdefault("face_evidence", row["runtime_evidence"]["face"])
                row.setdefault("parser_evidence", row["runtime_evidence"]["parser"])
                row.setdefault("provider_evidence", row["runtime_evidence"]["provider_evidence"])
                row.setdefault("roi_provenance", row["runtime_evidence"]["roi_provenance"])
                row.setdefault("runtime_diagnostics", row["runtime_evidence"]["runtime_diagnostics"])
                row.setdefault("face_count", row["face_evidence"].get("face_count"))
                row.setdefault("face_aware_run", False)
                row.setdefault("certifying", False)
                if args.restart_engine_per_recipe and recipe_engine is not None:
                    recipe_engine.close()
                    gc.collect()
                manifest["outputs"] = [
                    existing for existing in manifest["outputs"]
                    if existing.get("recipe") != recipe
                ]
                manifest["outputs"].append(row)
                _checkpoint_manifest(manifest_path, manifest)
    finally:
        if engine is not None:
            engine.close()

    if args.contact_sheet and output_paths:
        sheet_path = output_dir / "contact_sheet.jpg"
        generate_contact_sheet(output_paths, sheet_path, cols=args.sheet_cols, cell_size=args.cell_size)
        manifest["contact_sheet"] = sheet_path.name
        print(f"contact sheet: {sheet_path}")

    done_rows = [row for row in manifest["outputs"] if row.get("status") == "done"]
    manifest["face_aware_run"] = bool(done_rows) and all(
        row.get("face_aware_run") is True for row in done_rows
    )
    manifest["certifying"] = bool(done_rows) and not bool(args.global_only) and all(
        row.get("certifying") is True for row in done_rows
    )
    manifest["diagnostic_only"] = bool(args.global_only)

    _checkpoint_manifest(manifest_path, manifest)
    print(f"manifest: {manifest_path}")

    failed = [row for row in manifest["outputs"] if row["status"] != "done"]
    if failed:
        print(f"completed with {len(failed)} failed recipe(s)")
        return 1
    print(f"completed {len(output_paths)} recipe render(s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one image through curated recipes, or all recipes when explicitly requested."
    )
    parser.add_argument("input", help="Input image path")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument(
        "--recipes",
        default="recommended",
        help=(
            "Comma-separated recipe names; 'recommended' (default), 'conditional', "
            "'curated' (all 50 maintained), or 'all' for legacy/experimental recipes."
        ),
    )
    parser.add_argument(
        "--skip",
        default="",
        help="Comma-separated recipe names to skip",
    )
    parser.add_argument("--format", choices=["jpg", "png", "webp"], default="jpg")
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--max-dim", type=int, default=None)
    parser.add_argument(
        "--global-only",
        action="store_true",
        help="Skip face detection and render only global recipe finishing.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also export side-by-side original/result comparison images.",
    )
    parser.add_argument(
        "--no-contact-sheet",
        action="store_false",
        dest="contact_sheet",
        help="Do not generate contact_sheet.jpg.",
    )
    parser.add_argument("--sheet-cols", type=int, default=4)
    parser.add_argument("--cell-size", type=int, default=320)
    parser.add_argument(
        "--fail-on-qa",
        action="store_true",
        help="Mark recipes failed if QA detectors flag artifacts.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        default=True,
        help="Continue after a recipe fails (default: enabled).",
    )
    parser.add_argument(
        "--stop-on-fail",
        action="store_false",
        dest="keep_going",
        help="Stop at the first recipe failure.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a matching partial sweep, skipping outputs recorded as done.",
    )
    parser.add_argument(
        "--restart-engine-per-recipe",
        action="store_true",
        help="Close and recreate the native engine after every recipe to reduce long-run memory pressure.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_sweep(args)
    except cv2.error as exc:
        print(f"cv2.error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

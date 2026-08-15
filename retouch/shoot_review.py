"""Persistent, non-destructive shoot review manifests.

The manifest is deliberately an evidence and decision record, not a culling
engine. Automatic observations may be refreshed by a later scan, while a
human decision, rating, labels, and its history remain intact. No operation
in this module deletes, moves, or rejects a source file.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
from uuid import uuid4

from .shoot_intelligence import BurstGroup, CullingCandidate, ShootAsset


DECISIONS = {"select", "reject", "hold"}
DECISION_ORIGINS = {"automatic", "human"}
EYES_OPEN_VALUES = {"yes", "no", "uncertain"}


class StateLoadError(ValueError):
    """A durable review state file exists but cannot be trusted."""


def _quarantine_malformed_state(path: Path) -> Optional[Path]:
    """Preserve a malformed state file beside itself for human recovery."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine = path.with_name(f"{path.name}.corrupt-{stamp}")
    try:
        shutil.copy2(path, quarantine)
    except OSError:
        return None
    return quarantine


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: Path, text: str, *, prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temp_name, path)
    except OSError:
        try:
            os.remove(temp_name)
        except OSError:
            pass
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_id_for(relative_path: str, content_sha256: str) -> str:
    """Return a stable project-local ID for one path/content version."""
    normalized = Path(str(relative_path)).as_posix()
    digest = hashlib.sha256((normalized + "\0" + str(content_sha256)).encode("utf-8")).hexdigest()
    return "asset-" + digest[:24]


def _relative_path(project_root: Path, source_path: Union[str, Path]) -> str:
    source = Path(source_path).expanduser().resolve()
    try:
        relative = os.path.relpath(str(source), str(project_root))
    except ValueError:
        relative = source.name
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        raise ValueError("source path must be inside the shoot project root")
    return Path(relative).as_posix()


def _bounded_optional(value: Any, lower: float = 0.0, upper: float = 1.0) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    if not lower <= number <= upper:
        raise ValueError(f"value must be in [{lower}, {upper}]")
    return number


def _normalized_bbox(value: Any) -> Optional[tuple[float, float, float, float]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox must contain normalized x, y, width, height")
    x, y, width, height = (float(item) for item in value)
    if width <= 0.0 or height <= 0.0:
        raise ValueError("bbox width and height must be positive")
    if min(x, y, width, height) < 0.0 or max(x, y, width, height) > 1.0:
        raise ValueError("bbox values must be in [0, 1]")
    if x + width > 1.000001 or y + height > 1.000001:
        raise ValueError("bbox must stay inside normalized image bounds")
    return (x, y, width, height)


def _bbox_iou(
    left: Optional[tuple[float, float, float, float]],
    right: Optional[tuple[float, float, float, float]],
) -> float:
    if left is None or right is None:
        return 0.0
    lx1, ly1, lw, lh = left
    rx1, ry1, rw, rh = right
    lx2, ly2 = lx1 + lw, ly1 + lh
    rx2, ry2 = rx1 + rw, ry1 + rh
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0.0 else 0.0


@dataclass
class FaceQualityEvidence:
    """Inspectable quality evidence for one detected face.

    ``uncertain`` is represented explicitly by ``eyes_open='uncertain'`` and
    the optional uncertainty list. Missing detector output is not converted
    into a false pass or fail.
    """

    face_id: str
    bbox: Optional[tuple[float, float, float, float]] = None
    coverage: Optional[float] = None
    detector_confidence: Optional[float] = None
    detector_confidence_source: str = "unknown"
    face_sharpness: Optional[float] = None
    left_eye_sharpness: Optional[float] = None
    right_eye_sharpness: Optional[float] = None
    eyes_open: str = "uncertain"
    measurement_version: str = "face-quality-v1"
    sharpness_method: str = "unspecified"
    measurement_details: Dict[str, Any] = field(default_factory=dict)
    uncertainty: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.face_id:
            raise ValueError("face_id is required")
        self.bbox = _normalized_bbox(self.bbox)
        self.coverage = _bounded_optional(self.coverage)
        self.detector_confidence = _bounded_optional(self.detector_confidence)
        self.detector_confidence_source = str(self.detector_confidence_source or "unknown")
        for name in ("face_sharpness", "left_eye_sharpness", "right_eye_sharpness"):
            value = getattr(self, name)
            if value is not None and float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative")
            if value is not None:
                setattr(self, name, float(value))
        self.eyes_open = str(self.eyes_open).lower()
        if self.eyes_open not in EYES_OPEN_VALUES:
            raise ValueError("eyes_open must be yes, no, or uncertain")
        self.measurement_version = str(self.measurement_version or "unknown")
        self.sharpness_method = str(self.sharpness_method or "unspecified")
        self.measurement_details = dict(self.measurement_details or {})
        self.uncertainty = [str(item) for item in self.uncertainty]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FaceQualityEvidence":
        return cls(
            face_id=str(payload["face_id"]),
            bbox=payload.get("bbox"),
            coverage=payload.get("coverage"),
            detector_confidence=payload.get("detector_confidence"),
            detector_confidence_source=str(payload.get("detector_confidence_source", "unknown")),
            face_sharpness=payload.get("face_sharpness"),
            left_eye_sharpness=payload.get("left_eye_sharpness"),
            right_eye_sharpness=payload.get("right_eye_sharpness"),
            eyes_open=str(payload.get("eyes_open", "uncertain")),
            measurement_version=str(payload.get("measurement_version", "face-quality-v1")),
            sharpness_method=str(payload.get("sharpness_method", "unspecified")),
            measurement_details=dict(payload.get("measurement_details") or {}),
            uncertainty=[str(item) for item in payload.get("uncertainty", [])],
        )


def _reconcile_face_evidence(
    asset_id: str,
    current: Sequence[FaceQualityEvidence],
    previous: Sequence[FaceQualityEvidence],
    *,
    minimum_iou: float = 0.5,
) -> List[FaceQualityEvidence]:
    """Preserve IDs only for unambiguous, mutually unique geometry matches."""
    current_items = list(current)
    previous_items = list(previous)
    current_candidates: Dict[int, List[int]] = {}
    previous_candidates: Dict[int, List[int]] = {}
    for current_index, current_face in enumerate(current_items):
        for previous_index, previous_face in enumerate(previous_items):
            if _bbox_iou(current_face.bbox, previous_face.bbox) >= minimum_iou:
                current_candidates.setdefault(current_index, []).append(previous_index)
                previous_candidates.setdefault(previous_index, []).append(current_index)

    matches: Dict[int, int] = {}
    for current_index, candidates in current_candidates.items():
        if len(candidates) != 1:
            continue
        previous_index = candidates[0]
        if previous_candidates.get(previous_index) == [current_index]:
            matches[current_index] = previous_index

    output: List[FaceQualityEvidence] = []
    for index, face in enumerate(current_items):
        uncertainty = list(face.uncertainty)
        if index in matches:
            face_id = previous_items[matches[index]].face_id
        else:
            # Preserve explicit IDs supplied by an external analyzer; replace
            # only the provisional IDs emitted by Retouch's measurement pass.
            face_id = face.face_id
            if face_id.startswith("observation-"):
                face_id = f"{asset_id}:face-{uuid4().hex}"
            if previous_items:
                if current_candidates.get(index):
                    uncertainty.append("face_identity_ambiguous")
                else:
                    uncertainty.append("face_identity_unmatched")
        output.append(replace(
            face,
            face_id=face_id,
            uncertainty=sorted(set(uncertainty)),
        ))
    return output


@dataclass
class ReviewOverride:
    timestamp: str
    previous_decision: str
    decision: str
    rating: Optional[int] = None
    labels: List[str] = field(default_factory=list)
    reviewer: str = ""
    note: str = ""
    origin: str = "human"

    def __post_init__(self) -> None:
        if self.previous_decision not in DECISIONS or self.decision not in DECISIONS:
            raise ValueError("review decisions must be select, reject, or hold")
        if self.origin not in DECISION_ORIGINS:
            raise ValueError("review origin must be automatic or human")
        if self.rating is not None and int(self.rating) not in range(0, 6):
            raise ValueError("rating must be an integer from 0 to 5")
        self.rating = int(self.rating) if self.rating is not None else None
        self.labels = [str(label) for label in self.labels if str(label).strip()]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReviewOverride":
        return cls(
            timestamp=str(payload.get("timestamp", _now())),
            previous_decision=str(payload.get("previous_decision", "hold")),
            decision=str(payload.get("decision", "hold")),
            rating=payload.get("rating"),
            labels=[str(item) for item in payload.get("labels", [])],
            reviewer=str(payload.get("reviewer", "")),
            note=str(payload.get("note", "")),
            origin=str(payload.get("origin", "human")),
        )


@dataclass
class ReviewAsset:
    asset_instance_id: str
    relative_path: str
    source_sha256: str
    content_id: str
    width: int
    height: int
    burst_ids: List[str] = field(default_factory=list)
    culling_evidence: Dict[str, Any] = field(default_factory=dict)
    faces: List[FaceQualityEvidence] = field(default_factory=list)
    uncertainty: List[str] = field(default_factory=list)
    decision: str = "hold"
    decision_origin: str = "automatic"
    rating: Optional[int] = None
    labels: List[str] = field(default_factory=list)
    review_required: bool = True
    override_history: List[ReviewOverride] = field(default_factory=list)
    identity_history: List[Dict[str, Any]] = field(default_factory=list)
    job_provenance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    present: bool = True
    created: str = field(default_factory=_now)
    updated: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        if not self.asset_instance_id:
            raise ValueError("asset_instance_id is required")
        if not self.content_id:
            self.content_id = self.source_sha256
        if not self.source_sha256:
            self.source_sha256 = self.content_id
        if self.decision not in DECISIONS:
            raise ValueError("decision must be select, reject, or hold")
        if self.decision_origin not in DECISION_ORIGINS:
            raise ValueError("decision_origin must be automatic or human")
        if self.rating is not None and int(self.rating) not in range(0, 6):
            raise ValueError("rating must be an integer from 0 to 5")
        self.rating = int(self.rating) if self.rating is not None else None
        self.burst_ids = [str(item) for item in self.burst_ids]
        self.labels = [str(label) for label in self.labels if str(label).strip()]
        self.uncertainty = [str(item) for item in self.uncertainty]

    @property
    def asset_id(self) -> str:
        """Backward-compatible name for the persistent asset instance ID."""
        return self.asset_instance_id

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["asset_id"] = self.asset_instance_id
        payload["asset_instance_id"] = self.asset_instance_id
        payload["faces"] = [face.to_dict() for face in self.faces]
        payload["override_history"] = [item.to_dict() for item in self.override_history]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReviewAsset":
        values = dict(payload)
        values["faces"] = [FaceQualityEvidence.from_dict(item) for item in payload.get("faces", [])]
        values["override_history"] = [ReviewOverride.from_dict(item) for item in payload.get("override_history", [])]
        return cls(
            asset_instance_id=str(values.get("asset_instance_id", values.get("asset_id", ""))),
            relative_path=str(values["relative_path"]),
            source_sha256=str(values.get("source_sha256", "")),
            content_id=str(values.get("content_id", values.get("source_sha256", ""))),
            width=int(values.get("width", 0)),
            height=int(values.get("height", 0)),
            burst_ids=values.get("burst_ids", []),
            culling_evidence=dict(values.get("culling_evidence") or {}),
            faces=values["faces"],
            uncertainty=values.get("uncertainty", []),
            decision=str(values.get("decision", "hold")),
            decision_origin=str(values.get("decision_origin", "automatic")),
            rating=values.get("rating"),
            labels=values.get("labels", []),
            review_required=bool(values.get("review_required", True)),
            override_history=values["override_history"],
            identity_history=[dict(item) for item in values.get("identity_history", [])],
            job_provenance={str(key): dict(value) for key, value in (values.get("job_provenance") or {}).items()},
            present=bool(values.get("present", True)),
            created=str(values.get("created", _now())),
            updated=str(values.get("updated", _now())),
        )


class ShootReviewManifest:
    """Versioned review artifact for one project/shoot."""

    schema_version = 2

    def __init__(self, project_root: Union[str, Path], *, created: Optional[str] = None, updated: Optional[str] = None):
        self.project_root = str(Path(project_root).expanduser().resolve())
        self.created = created or _now()
        self.updated = updated or self.created
        self.assets: Dict[str, ReviewAsset] = {}

    @staticmethod
    def default_path(project_root: Union[str, Path]) -> Path:
        return Path(project_root).expanduser().resolve() / ".retouch-shoot-review.json"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "retouch.shoot_review",
            "schema_version": self.schema_version,
            "project_root": self.project_root,
            "created": self.created,
            "updated": self.updated,
            "assets": [self.assets[key].to_dict() for key in sorted(self.assets)],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShootReviewManifest":
        if not isinstance(payload, Mapping):
            raise ValueError("shoot review state must be a JSON object")
        if payload.get("schema") != "retouch.shoot_review":
            raise ValueError("unsupported or missing shoot review schema")
        schema_version = payload.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version < 1
            or schema_version > cls.schema_version
        ):
            raise ValueError(f"unsupported shoot review schema version: {schema_version!r}")
        asset_values = payload.get("assets", [])
        if not isinstance(asset_values, list):
            raise ValueError("shoot review assets must be a JSON array")
        manifest = cls(
            str(payload.get("project_root", ".")),
            created=str(payload.get("created", _now())),
            updated=str(payload.get("updated", _now())),
        )
        for item in asset_values:
            if not isinstance(item, Mapping):
                raise ValueError("shoot review asset entries must be JSON objects")
            asset = ReviewAsset.from_dict(item)
            manifest.assets[asset.asset_id] = asset
        return manifest

    @classmethod
    def from_json(cls, payload: str) -> "ShootReviewManifest":
        return cls.from_dict(json.loads(payload))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ShootReviewManifest":
        target = Path(path).expanduser()
        if not target.exists():
            return cls(target.parent)
        try:
            return cls.from_json(target.read_text(encoding="utf-8"))
        except FileNotFoundError:
            # The file may have been removed between exists() and read().
            return cls(target.parent)
        except OSError as exc:
            raise StateLoadError(
                f"Could not read shoot review state {target}: {exc}"
            ) from exc
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            quarantine = _quarantine_malformed_state(target)
            detail = f"; preserved copy: {quarantine}" if quarantine else "; original preserved"
            raise StateLoadError(
                f"Malformed shoot review state {target}{detail}: {exc}"
            ) from exc

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        target = Path(path).expanduser() if path else self.default_path(self.project_root)
        _atomic_write_text(target, self.to_json(), prefix=".retouch-shoot-review-")
        return target

    def observe(self, asset: ReviewAsset) -> ReviewAsset:
        """Refresh automatic evidence while preserving human review state."""
        existing = self.assets.get(asset.asset_id)
        if existing is not None:
            asset.created = existing.created
            asset.override_history = existing.override_history
            asset.identity_history = list(existing.identity_history) + [
                event for event in asset.identity_history
                if event not in existing.identity_history
            ]
            asset.job_provenance = existing.job_provenance
            content_changed = existing.content_id != asset.content_id
            if not content_changed:
                asset.rating = existing.rating
                asset.labels = list(existing.labels)
                asset.decision = existing.decision
                asset.decision_origin = "human"
                asset.review_required = existing.review_required
            elif content_changed:
                # The old decision remains auditable in override_history, but
                # it must not silently approve a different byte sequence.
                asset.decision = "hold"
                asset.decision_origin = "automatic"
                asset.rating = None
                asset.labels = []
                asset.review_required = True
                asset.identity_history.append({
                    "kind": "content_changed",
                    "timestamp": _now(),
                    "relative_path": asset.relative_path,
                    "previous_content_id": existing.content_id,
                    "content_id": asset.content_id,
                })
        asset.updated = _now()
        self.assets[asset.asset_id] = asset
        self.updated = asset.updated
        return asset

    def set_human_review(
        self,
        asset_id: str,
        decision: str,
        *,
        rating: Optional[int] = None,
        labels: Optional[Sequence[str]] = None,
        reviewer: str = "",
        note: str = "",
    ) -> ReviewAsset:
        if decision not in DECISIONS:
            raise ValueError("decision must be select, reject, or hold")
        if asset_id not in self.assets:
            raise KeyError(asset_id)
        if rating is not None and int(rating) not in range(0, 6):
            raise ValueError("rating must be an integer from 0 to 5")
        asset = self.assets[asset_id]
        override = ReviewOverride(
            timestamp=_now(),
            previous_decision=asset.decision,
            decision=decision,
            rating=rating,
            labels=[str(item).strip() for item in (labels or []) if str(item).strip()],
            reviewer=str(reviewer or "").strip(),
            note=str(note or "").strip(),
        )
        asset.decision = decision
        asset.decision_origin = "human"
        asset.rating = override.rating
        asset.labels = list(override.labels)
        asset.review_required = decision == "hold"
        asset.override_history.append(override)
        asset.updated = override.timestamp
        self.updated = override.timestamp
        return asset

    def attach_job_provenance(self, asset_id: str, kind: str, details: Mapping[str, Any]) -> ReviewAsset:
        if asset_id not in self.assets:
            raise KeyError(asset_id)
        if kind not in {"preview", "final"}:
            raise ValueError("job kind must be preview or final")
        asset = self.assets[asset_id]
        asset.job_provenance[kind] = dict(details)
        asset.updated = _now()
        self.updated = asset.updated
        return asset

    def export_csv_text(self, *, selected_only: bool = False) -> str:
        output = io.StringIO(newline="")
        fields = [
            "manifest_schema_version", "asset_id", "asset_instance_id", "relative_path",
            "source_sha256", "content_id", "width", "height",
            "burst_ids", "candidate_rank", "candidate_score", "decision",
            "decision_origin", "rating", "labels", "review_required",
            "uncertainty", "culling_evidence_json", "face_count",
            "face_evidence_json", "job_provenance_json", "identity_history_json", "present",
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for asset in sorted(self.assets.values(), key=lambda item: item.relative_path):
            if selected_only and asset.decision != "select":
                continue
            candidate = asset.culling_evidence.get("candidate") or {}
            writer.writerow({
                "manifest_schema_version": self.schema_version,
                "asset_id": asset.asset_id,
                "asset_instance_id": asset.asset_instance_id,
                "relative_path": asset.relative_path,
                "source_sha256": asset.source_sha256,
                "content_id": asset.content_id,
                "width": asset.width,
                "height": asset.height,
                "burst_ids": ";".join(asset.burst_ids),
                "candidate_rank": candidate.get("rank", ""),
                "candidate_score": candidate.get("score", ""),
                "decision": asset.decision,
                "decision_origin": asset.decision_origin,
                "rating": "" if asset.rating is None else asset.rating,
                "labels": ";".join(asset.labels),
                "review_required": str(asset.review_required).lower(),
                "uncertainty": json.dumps(asset.uncertainty, sort_keys=True),
                "culling_evidence_json": json.dumps(asset.culling_evidence, sort_keys=True),
                "face_count": len(asset.faces),
                "face_evidence_json": json.dumps(
                    [face.to_dict() for face in asset.faces], sort_keys=True,
                ),
                "job_provenance_json": json.dumps(asset.job_provenance, sort_keys=True),
                "identity_history_json": json.dumps(asset.identity_history, sort_keys=True),
                "present": str(asset.present).lower(),
            })
        return output.getvalue()

    def export_csv(self, path: Union[str, Path], *, selected_only: bool = False) -> Path:
        target = Path(path).expanduser()
        _atomic_write_text(target, self.export_csv_text(selected_only=selected_only), prefix=".retouch-shoot-review-csv-")
        return target


class ShootReviewManifestStore:
    """Small persistence facade used by GUI handlers and scripts."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path).expanduser()
        self.manifest = ShootReviewManifest.load(self.path)

    def save(self) -> Path:
        return self.manifest.save(self.path)

    def export_csv(self, path: Optional[Union[str, Path]] = None, *, selected_only: bool = False) -> Path:
        target = Path(path).expanduser() if path else self.path.with_suffix(".csv")
        return self.manifest.export_csv(target, selected_only=selected_only)


def build_review_manifest(
    project_root: Union[str, Path],
    assets: Sequence[ShootAsset],
    bursts: Sequence[BurstGroup],
    candidates_by_group: Mapping[str, Sequence[CullingCandidate]],
    *,
    face_evidence_by_path: Optional[Mapping[str, Sequence[FaceQualityEvidence]]] = None,
    face_uncertainty_by_path: Optional[Mapping[str, Sequence[str]]] = None,
    manifest_path: Optional[Union[str, Path]] = None,
) -> ShootReviewManifest:
    """Create/update a manifest from explainable scan observations."""
    root = Path(project_root).expanduser().resolve()
    target = Path(manifest_path).expanduser() if manifest_path else ShootReviewManifest.default_path(root)
    store = ShootReviewManifestStore(target)
    manifest = store.manifest
    # Storage location and shoot identity are independent. This is essential
    # when a caller puts the manifest in a shared project/cache directory.
    manifest.project_root = str(root)
    group_by_path: Dict[str, List[BurstGroup]] = {}
    candidate_by_path: Dict[str, CullingCandidate] = {}
    for group in bursts:
        for path in group.asset_paths:
            group_by_path.setdefault(str(Path(path).resolve()), []).append(group)
        for candidate in candidates_by_group.get(group.group_id, []):
            candidate_by_path[str(Path(candidate.path).resolve())] = candidate

    observations: List[Dict[str, Any]] = []
    for asset in assets:
        source = Path(asset.path).expanduser().resolve()
        content_hash = _sha256(source)
        relative = _relative_path(root, source)
        groups = group_by_path.get(str(source), [])
        candidate = candidate_by_path.get(str(source))
        culling_evidence: Dict[str, Any] = {
            "policy": "capture-quality-only; human review required",
            "burst_groups": [group.to_dict() for group in groups],
            "candidate": candidate.to_dict() if candidate else None,
        }
        uncertainty: List[str] = []
        if asset.capture_time is None:
            uncertainty.append("capture_timestamp_missing")
        if not groups:
            uncertainty.append("not_grouped_as_burst")
        analysis_performed = face_evidence_by_path is not None and str(source) in face_evidence_by_path
        if not analysis_performed:
            uncertainty.append("face_quality_unavailable")
        faces = list(face_evidence_by_path.get(str(source), ())) if analysis_performed else []
        if analysis_performed and not faces:
            uncertainty.append("no_face_detected")
        if face_uncertainty_by_path:
            uncertainty.extend(str(item) for item in face_uncertainty_by_path.get(str(source), ()))
        for face in faces:
            if face.eyes_open == "uncertain":
                uncertainty.append("eyes_open_uncertain")
        observations.append({
            "asset": asset,
            "source": source,
            "relative": relative,
            "content_id": content_hash,
            "groups": groups,
            "candidate": candidate,
            "culling_evidence": culling_evidence,
            "faces": faces,
            "face_analysis_performed": analysis_performed,
            "uncertainty": sorted(set(uncertainty)),
        })

    current_paths = {item["relative"] for item in observations}
    path_matches = {
        record.relative_path: record
        for record in manifest.assets.values()
        if record.relative_path in current_paths
    }
    unmatched_observations = [item for item in observations if item["relative"] not in path_matches]
    old_missing_by_content: Dict[str, List[ReviewAsset]] = {}
    for record in manifest.assets.values():
        if record.relative_path not in current_paths:
            old_missing_by_content.setdefault(record.content_id or record.source_sha256, []).append(record)
    new_by_content: Dict[str, List[Dict[str, Any]]] = {}
    for item in unmatched_observations:
        new_by_content.setdefault(item["content_id"], []).append(item)

    # Reuse an old instance only for one unambiguous missing->new path match.
    # Multiple identical copies remain separate instances rather than being
    # merged into one reviewed asset.
    rename_matches: Dict[str, ReviewAsset] = {}
    for content_id, items in new_by_content.items():
        candidates = old_missing_by_content.get(content_id, [])
        if len(items) == 1 and len(candidates) == 1:
            rename_matches[items[0]["relative"]] = candidates[0]

    seen: set[str] = set()
    for item in observations:
        existing = path_matches.get(item["relative"]) or rename_matches.get(item["relative"])
        asset_id = existing.asset_id if existing is not None else "asset-" + uuid4().hex
        identity_history = list(existing.identity_history) if existing is not None else []
        if existing is not None and existing.relative_path != item["relative"]:
            identity_history.append({
                "kind": "rename",
                "timestamp": _now(),
                "from": existing.relative_path,
                "to": item["relative"],
                "content_id": item["content_id"],
            })
        content_unchanged = existing is not None and existing.content_id == item["content_id"]
        if item["face_analysis_performed"]:
            faces = _reconcile_face_evidence(
                asset_id,
                item["faces"],
                existing.faces if content_unchanged else (),
            )
        elif content_unchanged and existing.faces:
            # A metadata-only rescan must not erase valid evidence for the same
            # source bytes merely because the optional analyzer was disabled.
            faces = list(existing.faces)
            item["uncertainty"] = sorted(set(
                value for value in item["uncertainty"]
                if value != "face_quality_unavailable"
            ) | {"face_quality_not_refreshed"})
        else:
            faces = []
        record = ReviewAsset(
            asset_instance_id=asset_id,
            relative_path=item["relative"],
            source_sha256=item["content_id"],
            content_id=item["content_id"],
            width=item["asset"].width,
            height=item["asset"].height,
            burst_ids=[group.group_id for group in item["groups"]],
            culling_evidence=item["culling_evidence"],
            faces=faces,
            uncertainty=item["uncertainty"],
            decision="hold",
            decision_origin="automatic",
            review_required=True,
            identity_history=identity_history,
        )
        manifest.observe(record)
        seen.add(asset_id)
    for asset_id, record in manifest.assets.items():
        record.present = asset_id in seen
    manifest.updated = _now()
    store.save()
    return manifest


__all__ = [
    "DECISIONS", "EYES_OPEN_VALUES", "FaceQualityEvidence", "ReviewOverride",
    "ReviewAsset", "ShootReviewManifest", "ShootReviewManifestStore",
    "asset_id_for", "build_review_manifest",
]

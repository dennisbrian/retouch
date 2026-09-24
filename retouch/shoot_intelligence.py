"""Explainable project/shoot intelligence primitives.

The first slice is deliberately non-destructive: it builds a dependency/status
graph and groups likely burst captures, but never deletes, rejects, or edits a
source photo. Every grouping decision carries reasons and evidence so a GUI or
future culling assistant can ask for review.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from .capture_fidelity import CaptureMetadata, read_capture_metadata
from .io import open_image


GRAPH_STATUSES = {"pending", "ready", "running", "succeeded", "failed", "blocked", "skipped"}


@dataclass
class ProjectNode:
    node_id: str
    kind: str
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    message: str = ""
    outputs: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in GRAPH_STATUSES:
            raise ValueError(f"unknown project node status: {self.status!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProjectGraph:
    """Small serializable dependency graph for a project/shoot workflow."""

    def __init__(self, nodes: Optional[Iterable[ProjectNode]] = None):
        self.nodes: Dict[str, ProjectNode] = {}
        for node in nodes or ():
            self.add_node(node)

    def add_node(self, node: ProjectNode) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate project node: {node.node_id}")
        self.nodes[node.node_id] = node
        self.validate()

    def validate(self) -> None:
        """Reject missing dependencies and cycles before the graph is used."""
        for node in self.nodes.values():
            missing = [dep for dep in node.dependencies if dep not in self.nodes]
            if missing:
                raise ValueError(f"{node.node_id} has missing dependencies: {missing}")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError(f"project graph cycle detected at {node_id}")
            if node_id in visited:
                return
            visiting.add(node_id)
            for dependency in self.nodes[node_id].dependencies:
                visit(dependency)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in self.nodes:
            visit(node_id)

    def refresh_ready(self) -> None:
        for node in self.nodes.values():
            if node.status not in {"pending", "ready", "blocked"}:
                continue
            blockers = [self.nodes[dep] for dep in node.dependencies if self.nodes[dep].status not in {"succeeded", "skipped"}]
            if blockers:
                node.status = "blocked"
                node.message = "Waiting for: " + ", ".join(dep.node_id for dep in blockers)
                node.evidence["blocked_by"] = [dep.node_id for dep in blockers]
            else:
                node.status = "ready"
                node.message = "Dependencies satisfied."
                node.evidence.pop("blocked_by", None)

    def ready_nodes(self) -> List[ProjectNode]:
        self.refresh_ready()
        return [node for node in self.nodes.values() if node.status == "ready"]

    def set_status(
        self,
        node_id: str,
        status: str,
        *,
        message: str = "",
        outputs: Optional[Sequence[str]] = None,
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if status not in GRAPH_STATUSES:
            raise ValueError(f"unknown project node status: {status!r}")
        if node_id not in self.nodes:
            raise KeyError(node_id)
        self.nodes[node_id].status = status
        self.nodes[node_id].message = message
        if outputs is not None:
            self.nodes[node_id].outputs = [str(output) for output in outputs]
        if evidence:
            self.nodes[node_id].evidence.update(dict(evidence))
        self.refresh_ready()

    def to_dict(self) -> Dict[str, Any]:
        return {"version": 1, "nodes": [node.to_dict() for node in self.nodes.values()]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProjectGraph":
        return cls(ProjectNode(**dict(row)) for row in payload.get("nodes", []))

    @classmethod
    def from_json(cls, payload: str) -> "ProjectGraph":
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True)
class ShootAsset:
    path: str
    capture_time: Optional[float]
    camera_model: Optional[str]
    lens_model: Optional[str]
    focal_length_mm: Optional[float]
    width: int
    height: int
    fingerprint: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BurstGroup:
    group_id: str
    asset_paths: Tuple[str, ...]
    confidence: float
    reasons: Tuple[str, ...]
    evidence: Mapping[str, Any]
    review_required: bool = True

    @property
    def representative(self) -> str:
        return self.asset_paths[0]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["asset_paths"] = list(self.asset_paths)
        payload["reasons"] = list(self.reasons)
        return payload


@dataclass(frozen=True)
class CullingCandidate:
    """Explainable, non-destructive hero-frame candidate score."""

    path: str
    rank: int
    score: float
    evidence: Mapping[str, Any]
    review_required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["evidence"] = dict(self.evidence)
        return payload


def _capture_timestamp(path: Path) -> Optional[float]:
    try:
        with open_image(path) as image:
            exif = image.getexif()
            raw = exif.get(36867) or exif.get(306)
        if not raw:
            return None
        parsed = datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S")
        return parsed.replace(tzinfo=timezone.utc).timestamp()
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None


def _fingerprint(path: Path) -> Tuple[int, int, str]:
    with open_image(path) as image:
        width, height = image.size
        rgb = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32)
    small = cv2.resize(rgb, (8, 8), interpolation=cv2.INTER_AREA)
    bits = (small >= float(np.mean(small))).astype(np.uint8).flatten()
    packed = np.packbits(bits).tobytes()
    return width, height, hashlib.sha256(packed).hexdigest()[:16]


def inspect_asset(path: Union[str, Path]) -> ShootAsset:
    """Read non-destructive capture facts for a JPEG/RAW-compatible path."""
    source = Path(path).expanduser().resolve()
    width, height, fingerprint = _fingerprint(source)
    metadata = read_capture_metadata(source)
    return ShootAsset(
        path=str(source),
        capture_time=_capture_timestamp(source),
        camera_model=metadata.camera_model,
        lens_model=metadata.lens_model,
        focal_length_mm=metadata.focal_length_mm,
        width=width,
        height=height,
        fingerprint=fingerprint,
    )


def _hamming_distance(left: str, right: str) -> int:
    a = int(left, 16)
    b = int(right, 16)
    return bin(a ^ b).count("1")


def group_bursts(
    assets: Sequence[ShootAsset],
    *,
    max_gap_seconds: float = 2.0,
    max_hamming_distance: int = 18,
) -> List[BurstGroup]:
    """Group likely bursts using time, capture geometry, and perceptual evidence.

    Missing capture times never create a burst solely from filename order. A
    group is always marked ``review_required`` because expression, blink,
    hands, and hair motion require visual QA before culling or fusion.
    """
    if max_gap_seconds < 0 or max_hamming_distance < 0:
        raise ValueError("burst thresholds must be non-negative")
    ordered = sorted(assets, key=lambda asset: (asset.capture_time is None, asset.capture_time or 0.0, asset.path))
    groups: List[List[ShootAsset]] = []
    for asset in ordered:
        if not groups:
            groups.append([asset])
            continue
        previous = groups[-1][-1]
        if asset.capture_time is None or previous.capture_time is None:
            groups.append([asset])
            continue
        gap = asset.capture_time - previous.capture_time
        same_camera = bool(asset.camera_model and previous.camera_model and asset.camera_model == previous.camera_model)
        same_lens = not asset.lens_model or not previous.lens_model or asset.lens_model == previous.lens_model
        similar = _hamming_distance(asset.fingerprint, previous.fingerprint) <= max_hamming_distance
        if 0 <= gap <= max_gap_seconds and same_camera and same_lens and similar:
            groups[-1].append(asset)
        else:
            groups.append([asset])

    output: List[BurstGroup] = []
    for index, group in enumerate(groups):
        if len(group) == 1:
            continue
        gaps = [group[i].capture_time - group[i - 1].capture_time for i in range(1, len(group))]
        distances = [_hamming_distance(group[i].fingerprint, group[i - 1].fingerprint) for i in range(1, len(group))]
        confidence = float(np.clip(
            0.45 * (1.0 - max(gaps) / max(max_gap_seconds, 1e-6))
            + 0.35 * (1.0 - max(distances) / max(max_hamming_distance, 1))
            + 0.20,
            0.0,
            1.0,
        ))
        output.append(BurstGroup(
            group_id=f"burst-{index:04d}",
            asset_paths=tuple(asset.path for asset in group),
            confidence=confidence,
            reasons=("capture timestamps are close", "camera/lens match", "thumbnail fingerprints are similar"),
            evidence={"gaps_seconds": gaps, "hamming_distances": distances, "count": len(group)},
        ))
    return output


def _frame_quality(path: str) -> Dict[str, float]:
    """Measure simple capture quality signals without face or identity inference."""
    with open_image(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    sharpness = float(np.var(laplacian))
    normalized = gray.astype(np.float32) / 255.0
    mean = float(np.mean(normalized))
    exposure = float(np.exp(-((mean - 0.5) / 0.28) ** 2))
    highlight_fraction = float(np.mean(normalized >= 0.99))
    shadow_fraction = float(np.mean(normalized <= 0.01))
    clipping = float(np.clip(1.0 - (highlight_fraction + shadow_fraction) / 0.20, 0.0, 1.0))
    resolution = float(np.log1p(rgb.shape[0] * rgb.shape[1]))
    return {
        "sharpness": sharpness,
        "exposure": exposure,
        "highlight_fraction": highlight_fraction,
        "shadow_fraction": shadow_fraction,
        "clipping": clipping,
        "resolution": resolution,
    }


# A frame whose smaller eyelid aperture is below this fraction of the same
# person's widest aperture in the burst is flagged as a blink or squint. Being
# relative, it holds for narrow-eyed subjects that an absolute cut would flag
# in every frame. Checked 2026-09-24 on simulated 5-frame bursts from 12 public
# talking-head clips: every clear blink was flagged (lowest ratio 0.07), and 11
# of 1,175 frames in blink-free bursts were flagged, mostly real half-closures
# or downward looks when checked by eye.
BLINK_RELATIVE_APERTURE = 0.55

# Faces at least this fraction of the main face's area count as subjects whose
# closed eyes should flag the frame (a group shot), not background people.
SUBJECT_FACE_AREA_FRACTION = 0.25

CAPTURE_ONLY_POLICY = "sharpness 0.50, exposure 0.25, clipping 0.15, resolution 0.10"
FACE_AWARE_POLICY = (
    "face focus 0.40, eyes open 0.25, frame sharpness 0.10, exposure 0.10, "
    "clipping 0.10, resolution 0.05"
)


def _face_value(face: Any, name: str) -> Any:
    if isinstance(face, Mapping):
        return face.get(name)
    return getattr(face, name, None)


def _face_focus(face: Any) -> Optional[float]:
    """Sharpness of the sharper eye, else of the whole face crop."""
    eyes = [
        float(value) for value in (
            _face_value(face, "left_eye_sharpness"), _face_value(face, "right_eye_sharpness"),
        ) if value is not None
    ]
    if eyes:
        return max(eyes)
    value = _face_value(face, "face_sharpness")
    return None if value is None else float(value)


def _face_aperture(face: Any) -> Optional[float]:
    details = _face_value(face, "measurement_details") or {}
    values = [details.get("left_eye_aperture"), details.get("right_eye_aperture")]
    if any(value is None for value in values):
        return None
    return float(min(values))


def _frame_face_evidence(faces: Sequence[Any]) -> Dict[str, Any]:
    """Summarize one frame's face evidence: main subject focus and eye state."""
    faces = [face for face in faces if _face_value(face, "coverage") is not None]
    if not faces:
        return {"faces_detected": 0}
    main = max(faces, key=lambda face: float(_face_value(face, "coverage")))
    main_area = float(_face_value(main, "coverage"))
    subjects = [
        face for face in faces
        if float(_face_value(face, "coverage")) >= SUBJECT_FACE_AREA_FRACTION * main_area
    ]
    states = [str(_face_value(face, "eyes_open") or "uncertain") for face in subjects]
    return {
        "faces_detected": len(faces),
        "subject_faces": len(subjects),
        "face_focus": _face_focus(main),
        "eye_aperture": _face_aperture(main),
        "subject_eyes_open": states,
    }


def rank_burst_candidates(
    group: BurstGroup,
    face_evidence_by_path: Optional[Mapping[str, Sequence[Any]]] = None,
) -> List[CullingCandidate]:
    """Rank burst frames for review without automatically culling any source.

    Without face evidence the score is capture-quality-only. With face evidence
    (``FaceQualityEvidence`` records keyed by asset path, as produced by
    :class:`retouch.face_quality.FaceQualityAnalyzer`) the score favours the
    frame whose main face has the sharpest eye and whose subjects have their
    eyes open. Frames with closed eyes are ranked lower and carry an
    ``eyes_closed`` flag; nothing is removed, and every candidate still
    requires human review. Identity, expression and likeness are not inspected.
    """
    if len(group.asset_paths) < 2:
        return []
    measurements: List[Tuple[str, Dict[str, float]]] = []
    for path in group.asset_paths:
        try:
            measurements.append((path, _frame_quality(path)))
        except (FileNotFoundError, OSError, ValueError):
            measurements.append((path, {
                "sharpness": 0.0, "exposure": 0.0,
                "highlight_fraction": 1.0, "shadow_fraction": 1.0,
                "clipping": 0.0, "resolution": 0.0,
            }))
    sharpness_values = np.asarray([item[1]["sharpness"] for item in measurements], dtype=np.float32)
    resolution_values = np.asarray([item[1]["resolution"] for item in measurements], dtype=np.float32)
    sharp_min, sharp_max = float(sharpness_values.min()), float(sharpness_values.max())
    res_min, res_max = float(resolution_values.min()), float(resolution_values.max())

    face_by_path: Dict[str, Dict[str, Any]] = {}
    if face_evidence_by_path is not None:
        lookup = {str(Path(key).expanduser().resolve()): value for key, value in face_evidence_by_path.items()}
        for path, _ in measurements:
            faces = lookup.get(str(Path(path).expanduser().resolve()))
            if faces is not None:
                face_by_path[path] = _frame_face_evidence(list(faces))
    focus_values = [item["face_focus"] for item in face_by_path.values() if item.get("face_focus") is not None]
    face_aware = bool(focus_values)
    best_focus = max(focus_values) if focus_values else 0.0
    apertures = [item["eye_aperture"] for item in face_by_path.values() if item.get("eye_aperture") is not None]
    baseline_aperture = max(apertures) if len(apertures) >= 2 else None

    scored: List[Tuple[str, float, Dict[str, Any]]] = []
    for path, evidence in measurements:
        sharpness_norm = 1.0 if sharp_max <= sharp_min else (evidence["sharpness"] - sharp_min) / (sharp_max - sharp_min)
        resolution_norm = 1.0 if res_max <= res_min else (evidence["resolution"] - res_min) / (res_max - res_min)
        details: Dict[str, Any] = {
            **evidence,
            "sharpness_normalized": float(sharpness_norm),
            "resolution_normalized": float(resolution_norm),
        }
        if not face_aware:
            score = float(np.clip(
                0.50 * sharpness_norm
                + 0.25 * evidence["exposure"]
                + 0.15 * evidence["clipping"]
                + 0.10 * resolution_norm,
                0.0,
                1.0,
            ))
            details["scoring_policy"] = CAPTURE_ONLY_POLICY
            scored.append((path, score, details))
            continue

        face = face_by_path.get(path, {"faces_detected": 0})
        flags: List[str] = []
        focus = face.get("face_focus")
        focus_norm = 0.0 if focus is None or best_focus <= 0.0 else float(focus) / best_focus
        if not face.get("faces_detected"):
            flags.append("no_face_detected")
        aperture = face.get("eye_aperture")
        aperture_ratio = (
            float(aperture) / baseline_aperture
            if aperture is not None and baseline_aperture and baseline_aperture > 0.0
            else None
        )
        states = face.get("subject_eyes_open", [])
        closed = "no" in states or (aperture_ratio is not None and aperture_ratio < BLINK_RELATIVE_APERTURE)
        if closed:
            flags.append("eyes_closed")
            eyes_term = 0.0
        elif states and all(state == "yes" for state in states):
            eyes_term = 1.0
        else:
            eyes_term = 0.5
        score = float(np.clip(
            0.40 * focus_norm
            + 0.25 * eyes_term
            + 0.10 * sharpness_norm
            + 0.10 * evidence["exposure"]
            + 0.10 * evidence["clipping"]
            + 0.05 * resolution_norm,
            0.0,
            1.0,
        ))
        details.update({
            "scoring_policy": FACE_AWARE_POLICY,
            "faces_detected": int(face.get("faces_detected", 0)),
            "face_focus": focus,
            "face_focus_relative": focus_norm,
            "eye_aperture": aperture,
            "eye_aperture_burst_max": baseline_aperture,
            "eye_aperture_relative": aperture_ratio,
            "subject_eyes_open": list(states),
            "eyes_term": eyes_term,
            "flags": flags,
        })
        scored.append((path, score, details))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [
        CullingCandidate(path=path, rank=index + 1, score=score, evidence=evidence)
        for index, (path, score, evidence) in enumerate(scored)
    ]

"""Resumable, non-destructive watch-folder ingestion.

The service uses polling rather than requiring a platform-specific watcher.
Files are emitted only after their size/mtime signature is unchanged across
two scans. Processing callbacks are supplied by the caller; failures remain
pending and are recorded with an explanation.
"""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from .io import EXT_MAP, IMAGE_EXTENSIONS


WATCH_JOB_STATES = {"queued", "processing", "done", "failed", "cancelled"}


class PermanentWatchJobError(RuntimeError):
    """Processor error that needs changed settings or explicit human retry."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class WatchJob:
    """Durable work item; ``done`` is valid only after output verification."""

    job_id: str
    source_path: str
    content_id: str
    asset_instance_id: str
    kind: str  # preview | final
    settings: Dict[str, Any]
    settings_fingerprint: str
    pipeline_fingerprint: str
    output_path: str
    status: str = "queued"
    attempts: int = 0
    error_code: Optional[str] = None
    error: Optional[str] = None
    queued_at: str = field(default_factory=_now)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    output_size: Optional[int] = None
    output_sha256: Optional[str] = None
    retryable: bool = True

    def __post_init__(self) -> None:
        if self.kind not in {"preview", "final"}:
            raise ValueError("watch job kind must be preview or final")
        if self.status not in WATCH_JOB_STATES:
            raise ValueError(f"unknown watch job status: {self.status!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WatchJob":
        return cls(
            job_id=str(payload["job_id"]),
            source_path=str(payload["source_path"]),
            content_id=str(payload.get("content_id", "")),
            asset_instance_id=str(payload.get("asset_instance_id", "")),
            kind=str(payload.get("kind", "preview")),
            settings=dict(payload.get("settings") or {}),
            settings_fingerprint=str(payload.get("settings_fingerprint", "")),
            pipeline_fingerprint=str(payload.get("pipeline_fingerprint", "")),
            output_path=str(payload["output_path"]),
            status=str(payload.get("status", "queued")),
            attempts=int(payload.get("attempts", 0)),
            error_code=payload.get("error_code"),
            error=payload.get("error"),
            queued_at=str(payload.get("queued_at", _now())),
            started_at=payload.get("started_at"),
            completed_at=payload.get("completed_at"),
            output_size=payload.get("output_size"),
            output_sha256=payload.get("output_sha256"),
            retryable=bool(payload.get("retryable", True)),
        )


@dataclass
class WatchRecord:
    path: str
    signature: str
    status: str = "pending"  # pending | queued | processing | done | failed
    attempts: int = 0
    error: Optional[str] = None
    content_id: Optional[str] = None
    asset_instance_id: Optional[str] = None
    job_ids: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WatchFolderState:
    version: int = 2
    records: Dict[str, WatchRecord] = field(default_factory=dict)
    jobs: Dict[str, WatchJob] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "records": {path: record.to_dict() for path, record in self.records.items()},
            "jobs": {job_id: job.to_dict() for job_id, job in self.jobs.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WatchFolderState":
        records = {
            str(path): WatchRecord(**dict(value))
            for path, value in (payload.get("records") or {}).items()
        }
        jobs = {
            str(job_id): WatchJob.from_dict(value)
            for job_id, value in (payload.get("jobs") or {}).items()
        }
        return cls(version=int(payload.get("version", 1)), records=records, jobs=jobs)


class WatchFolder:
    """Poll a folder and process only stable, changed image files."""

    def __init__(
        self,
        input_dir: Union[str, Path],
        *,
        state_path: Optional[Union[str, Path]] = None,
        recursive: bool = True,
    ) -> None:
        self.input_dir = Path(input_dir).expanduser().resolve()
        if not self.input_dir.is_dir():
            raise NotADirectoryError(str(self.input_dir))
        self.recursive = bool(recursive)
        self.state_path = Path(state_path).expanduser() if state_path else self.input_dir / ".retouch-watch.json"
        self.state = self._load_state()

    def _load_state(self) -> WatchFolderState:
        try:
            return WatchFolderState.from_dict(json.loads(self.state_path.read_text(encoding="utf-8")))
        except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
            return WatchFolderState()

    def save(self) -> None:
        """Persist state atomically so interruption cannot corrupt the queue."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=".retouch-watch-", suffix=".tmp", dir=str(self.state_path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.state.to_dict(), handle, indent=2, sort_keys=True)
            os.replace(temp_name, self.state_path)
        except OSError:
            try:
                os.remove(temp_name)
            except OSError:
                pass
            raise

    def _paths(self) -> Iterable[Path]:
        iterator = self.input_dir.rglob("*") if self.recursive else self.input_dir.iterdir()
        return sorted(
            path for path in iterator
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.name != self.state_path.name
        )

    @staticmethod
    def _signature(path: Path) -> str:
        stat = path.stat()
        return f"{stat.st_size}:{stat.st_mtime_ns}"

    def scan_once(self) -> List[WatchRecord]:
        """Update stability state and return files ready for processing.

        A newly observed or changed file is held for one scan. It becomes
        ready only when its signature is observed unchanged on the next scan.
        """
        ready: List[WatchRecord] = []
        current_paths = set()
        for path in self._paths():
            resolved = str(path.resolve())
            current_paths.add(resolved)
            try:
                signature = self._signature(path)
            except OSError:
                continue
            previous = self.state.records.get(resolved)
            if previous is None or previous.signature != signature:
                self.state.records[resolved] = WatchRecord(path=resolved, signature=signature)
                continue
            if previous.status in {"pending", "failed"}:
                ready.append(previous)
        # Stale records are retained for audit/retry, but no longer appear in
        # the active queue when the source has been removed.
        for path, record in self.state.records.items():
            if path not in current_paths and record.status == "processing":
                record.status = "failed"
                record.error = "source disappeared while processing"
        self.save()
        return ready

    def _default_output_path(self, source: Path, output_root: Path, kind: str, settings: Mapping[str, Any]) -> Path:
        try:
            relative = source.resolve().relative_to(self.input_dir.resolve())
        except ValueError:
            relative = Path(source.name)
        extension = EXT_MAP.get(str(settings.get("export_fmt", "JPEG")), ".jpg")
        return output_root / kind / relative.parent / f"{relative.stem}_retouched{extension}"

    def queue_jobs(
        self,
        *,
        kind: str,
        output_root: Union[str, Path],
        settings: Mapping[str, Any],
        pipeline_fingerprint: str,
        asset_instance_resolver: Optional[Callable[[Path, str], str]] = None,
        output_path_resolver: Optional[Callable[[Path, str, Mapping[str, Any]], Union[str, Path]]] = None,
        limit: Optional[int] = None,
        retry_failed: bool = False,
    ) -> List[WatchJob]:
        """Create idempotent durable jobs for stable files."""
        if kind not in {"preview", "final"}:
            raise ValueError("kind must be preview or final")
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        output_base = Path(output_root).expanduser().resolve()
        try:
            output_base.relative_to(self.input_dir.resolve())
        except ValueError:
            pass
        else:
            raise ValueError("job output root must be outside the watched folder")
        ready = self.scan_once()
        ready_paths = {record.path for record in ready}
        # A preview may already be done while a final job is still absent.
        # Include stable records that are not in the scan queue solely because
        # their aggregate WatchRecord status reflects an earlier job kind.
        for path in self._paths():
            resolved = str(path.resolve())
            record = self.state.records.get(resolved)
            if record is None or resolved in ready_paths:
                continue
            if record.status == "pending" and not record.job_ids:
                # This is the first stability observation, not an eligible
                # job yet.
                continue
            try:
                signature = self._signature(path)
            except OSError:
                continue
            if record.signature != signature:
                continue
            job_id = record.job_ids.get(kind)
            job = self.state.jobs.get(job_id) if job_id else None
            needs_job = (
                job is None
                or job.status == "failed"
                or (job.status == "done" and not self._output_matches(job))
            )
            if needs_job:
                ready.append(record)
        if limit is not None:
            ready = ready[:limit]
        normalized_settings = json.loads(json.dumps(dict(settings), sort_keys=True, default=str))
        settings_fingerprint = _hash_payload(normalized_settings)
        queued: List[WatchJob] = []
        for record in ready:
            source = Path(record.path).resolve()
            content_id = _file_sha256(source)
            asset_instance_id = (
                asset_instance_resolver(source, content_id)
                if asset_instance_resolver is not None
                else content_id
            )
            output_path = Path(output_path_resolver(source, kind, normalized_settings)) if output_path_resolver else self._default_output_path(source, output_base, kind, normalized_settings)
            logical_payload = {
                "source_content_id": content_id,
                "asset_instance_id": asset_instance_id,
                "kind": kind,
                "settings_fingerprint": settings_fingerprint,
                "pipeline_fingerprint": str(pipeline_fingerprint),
                "output_path": str(output_path),
            }
            job_id = "watch-job-" + _hash_payload(logical_payload)[:24]
            existing = self.state.jobs.get(job_id)
            if existing is not None:
                if existing.status == "done" and self._output_matches(existing):
                    record.status = "done"
                    record.job_ids[kind] = existing.job_id
                    record.content_id = content_id
                    record.asset_instance_id = asset_instance_id
                    continue
                if existing.status in {"queued", "processing"}:
                    record.status = "queued"
                    record.job_ids[kind] = existing.job_id
                    continue
                if not existing.retryable and not retry_failed:
                    record.status = "failed"
                    record.job_ids[kind] = existing.job_id
                    continue
                existing.status = "queued"
                existing.error_code = None
                existing.error = None
                existing.completed_at = None
                existing.output_size = None
                existing.output_sha256 = None
                existing.queued_at = _now()
                job = existing
            else:
                job = WatchJob(
                    job_id=job_id,
                    source_path=str(source),
                    content_id=content_id,
                    asset_instance_id=str(asset_instance_id),
                    kind=kind,
                    settings=normalized_settings,
                    settings_fingerprint=settings_fingerprint,
                    pipeline_fingerprint=str(pipeline_fingerprint),
                    output_path=str(output_path),
                )
                self.state.jobs[job.job_id] = job
            record.status = "queued"
            record.content_id = content_id
            record.asset_instance_id = str(asset_instance_id)
            record.job_ids[kind] = job.job_id
            queued.append(job)
            self.save()
        self.save()
        return queued

    @staticmethod
    def _output_matches(job: WatchJob) -> bool:
        if not job.output_sha256 or not job.output_size:
            return False
        output = Path(job.output_path)
        if not output.is_file() or output.stat().st_size != job.output_size:
            return False
        try:
            return _file_sha256(output) == job.output_sha256
        except OSError:
            return False

    def recover_stale_jobs(self, *, max_age_seconds: float = 3600.0) -> List[WatchJob]:
        """Return stale processing attempts to the durable queued state."""
        now = datetime.now(timezone.utc)
        recovered: List[WatchJob] = []
        for job in self.state.jobs.values():
            if job.status != "processing" or not job.started_at:
                continue
            try:
                age = (now - datetime.fromisoformat(job.started_at)).total_seconds()
            except ValueError:
                age = max_age_seconds + 1
            if age >= max_age_seconds:
                job.status = "queued"
                job.error_code = "worker_lost"
                job.error = "stale processing attempt recovered"
                job.started_at = None
                for record in self.state.records.values():
                    if record.job_ids.get(job.kind) == job.job_id:
                        record.status = "queued"
                        record.error = job.error
                recovered.append(job)
        if recovered:
            self.save()
        return recovered

    def run_queued(
        self,
        processor: Callable[[WatchJob], Optional[Union[str, Path]]],
        *,
        limit: Optional[int] = None,
        verifier: Optional[Callable[[WatchJob, Path], None]] = None,
    ) -> List[WatchJob]:
        """Run queued jobs and mark done only after output hash verification."""
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        self.recover_stale_jobs()
        jobs = [job for job in self.state.jobs.values() if job.status == "queued"]
        jobs.sort(key=lambda job: (job.queued_at, job.job_id))
        if limit is not None:
            jobs = jobs[:limit]
        completed: List[WatchJob] = []
        for job in jobs:
            job.status = "processing"
            job.attempts += 1
            job.started_at = _now()
            job.error_code = None
            job.error = None
            job.retryable = True
            for record in self.state.records.values():
                if record.job_ids.get(job.kind) == job.job_id:
                    record.status = "processing"
                    record.attempts = job.attempts
                    record.error = None
            self.save()
            before_output = None
            output_before = Path(job.output_path)
            if output_before.is_file():
                try:
                    before_output = (output_before.stat().st_size, output_before.stat().st_mtime_ns)
                except OSError:
                    before_output = None
            try:
                returned_path = processor(job)
                if returned_path is not None and Path(returned_path).resolve() != Path(job.output_path).resolve():
                    raise ValueError("processor returned an unexpected output path")
                output = Path(job.output_path)
                if not output.is_file() or output.stat().st_size <= 0:
                    raise FileNotFoundError("verified output is missing or empty")
                after_output = (output.stat().st_size, output.stat().st_mtime_ns)
                if before_output is not None and before_output == after_output:
                    raise IOError("processor did not update the intended output")
                if verifier is not None:
                    verifier(job, output)
                job.output_size = output.stat().st_size
                job.output_sha256 = _file_sha256(output)
                job.status = "done"
                job.completed_at = _now()
                for record in self.state.records.values():
                    if record.job_ids.get(job.kind) == job.job_id:
                        record.status = "done"
                        record.error = None
            except Exception as exc:
                job.status = "failed"
                job.error_code = type(exc).__name__
                job.error = str(exc)
                job.retryable = not isinstance(exc, PermanentWatchJobError)
                job.output_size = None
                job.output_sha256 = None
                job.completed_at = None
                for record in self.state.records.values():
                    if record.job_ids.get(job.kind) == job.job_id:
                        record.status = "failed"
                        record.attempts = job.attempts
                        record.error = f"{job.error_code}: {job.error}"
            completed.append(job)
            self.save()
        return completed

    def process_pending(
        self,
        processor: Callable[[Path], Any],
        *,
        limit: Optional[int] = None,
    ) -> List[WatchRecord]:
        """Process stable files; failed callbacks remain retryable."""
        ready = self.scan_once()
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative")
            ready = ready[:limit]
        completed: List[WatchRecord] = []
        for record in ready:
            record.status = "processing"
            record.attempts += 1
            record.error = None
            self.save()
            try:
                processor(Path(record.path))
            except Exception as exc:  # callback errors are queue state, not watcher crashes
                record.status = "failed"
                record.error = f"{type(exc).__name__}: {exc}"
            else:
                record.status = "done"
            completed.append(record)
            self.save()
        return completed

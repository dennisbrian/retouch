"""Job/FileRecord model and JobStore persistence for the batch Job Dashboard.

A Job records one ``BatchProcessor.process_folder`` run: which recipe/style,
which files, their per-file outcome, and any QA warnings the engine raised.
Jobs persist to disk so past batch runs survive an app restart and can be
reviewed (or partially re-run) from the "Job Dashboard" GUI tab.

Public API:
    FileRecord — per-file outcome within a job
    Job — one batch run
    JobStore — reads/writes jobs under ~/.retouch/jobs/

Design principles (mirrors retouch/session.py):
    - Forward-compatible: unknown keys in loaded JSON warn, don't crash
    - Atomic writes: temp file + os.replace, never a partially-written job
    - job_id is timestamp-sortable (no uuid precedent elsewhere in the repo)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

JOB_VERSION = 1

# QA states:
#   unknown — no .qa on the result (no-person short-circuit, or QA stripped
#             by a downstream cv2 op such as the export resize or a custom
#             style's skin-tone correction branch; these two cases are not
#             currently distinguishable, see docs/plans for the Job Dashboard)
#   clean   — QA ran and .qa == []
#   flagged — .qa has one or more real artifact warnings
#   error   — .qa has a warning whose details.available is False, i.e. the
#             QA pipeline itself failed (not a genuine artifact flag)
QA_STATE_UNKNOWN = "unknown"
QA_STATE_CLEAN = "clean"
QA_STATE_FLAGGED = "flagged"
QA_STATE_ERROR = "error"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def make_job_id(input_dir: str) -> str:
    """Timestamp-sortable job id: compact UTC timestamp + short hash of input_dir."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    return f"{ts}_{_short_hash(input_dir)}"


@dataclass
class FileRecord:
    """Outcome of processing one file within a Job."""

    source_path: str
    output_path: Optional[str] = None
    status: str = "pending"  # pending | done | failed
    qa_state: str = QA_STATE_UNKNOWN
    qa_warnings: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_path": self.source_path,
            "output_path": self.output_path,
            "status": self.status,
            "qa_state": self.qa_state,
            "qa_warnings": self.qa_warnings,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileRecord":
        known_keys = {
            "source_path", "output_path", "status", "qa_state",
            "qa_warnings", "error",
        }
        unknown = set(data.keys()) - known_keys
        if unknown:
            logger.warning(
                "FileRecord JSON contains unknown keys (ignored): %s",
                sorted(unknown),
            )
        return cls(
            source_path=data["source_path"],
            output_path=data.get("output_path"),
            status=data.get("status", "pending"),
            qa_state=data.get("qa_state", QA_STATE_UNKNOWN),
            qa_warnings=data.get("qa_warnings", []),
            error=data.get("error"),
        )


@dataclass
class Job:
    """One batch ``process_folder`` run."""

    job_id: str
    created: str = field(default_factory=_now_iso)
    updated: str = field(default_factory=_now_iso)
    version: int = JOB_VERSION
    status: str = "running"  # running | done | failed
    input_dir: str = ""
    output_dir: str = ""
    style_type: str = ""  # "Use Standard Recipe" | "Use Custom Style"
    recipe_or_style: str = ""
    export_fmt: str = "JPEG"
    export_quality: int = 95
    export_res: str = "Original"
    total_files: int = 0
    files: List[FileRecord] = field(default_factory=list)
    log: str = ""

    @property
    def flagged_count(self) -> int:
        return sum(1 for f in self.files if f.qa_state == QA_STATE_FLAGGED)

    @property
    def done_count(self) -> int:
        return sum(1 for f in self.files if f.status == "done")

    @property
    def failed_count(self) -> int:
        return sum(1 for f in self.files if f.status == "failed")

    def flagged_files(self) -> List[FileRecord]:
        return [f for f in self.files if f.qa_state == QA_STATE_FLAGGED]

    def touch(self) -> None:
        self.updated = _now_iso()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "created": self.created,
            "updated": self.updated,
            "version": self.version,
            "status": self.status,
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "style_type": self.style_type,
            "recipe_or_style": self.recipe_or_style,
            "export_fmt": self.export_fmt,
            "export_quality": self.export_quality,
            "export_res": self.export_res,
            "total_files": self.total_files,
            "files": [f.to_dict() for f in self.files],
            "log": self.log,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        known_keys = {
            "job_id", "created", "updated", "version", "status", "input_dir",
            "output_dir", "style_type", "recipe_or_style", "export_fmt",
            "export_quality", "export_res", "total_files", "files", "log",
        }
        unknown = set(data.keys()) - known_keys
        if unknown:
            logger.warning("Job JSON contains unknown keys (ignored): %s", sorted(unknown))

        version = data.get("version", JOB_VERSION)
        if version > JOB_VERSION:
            logger.warning(
                "Job version %s is newer than supported %s — loading with "
                "best-effort compatibility",
                version, JOB_VERSION,
            )

        return cls(
            job_id=data["job_id"],
            created=data.get("created", _now_iso()),
            updated=data.get("updated", _now_iso()),
            version=version,
            status=data.get("status", "running"),
            input_dir=data.get("input_dir", ""),
            output_dir=data.get("output_dir", ""),
            style_type=data.get("style_type", ""),
            recipe_or_style=data.get("recipe_or_style", ""),
            export_fmt=data.get("export_fmt", "JPEG"),
            export_quality=data.get("export_quality", 95),
            export_res=data.get("export_res", "Original"),
            total_files=data.get("total_files", 0),
            files=[FileRecord.from_dict(f) for f in data.get("files", [])],
            log=data.get("log", ""),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "Job":
        return cls.from_dict(json.loads(json_str))


class JobStore:
    """Reads/writes Job records as one JSON file per job under a jobs directory.

    One file per job (rather than a single jobs.json) sidesteps concurrent-
    write corruption when a job is being updated while the dashboard lists
    others. Writes are atomic (temp file + os.replace), mirroring the
    download pattern in retouch/model_fetch.py.
    """

    def __init__(self, jobs_dir: Optional[str] = None):
        self.jobs_dir = Path(jobs_dir) if jobs_dir else Path.home() / ".retouch" / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def save(self, job: Job) -> None:
        job.touch()
        dest = self._path_for(job.job_id)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{job.job_id}.", suffix=".json.tmp", dir=str(self.jobs_dir)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(job.to_json())
            os.replace(tmp_name, dest)
        except OSError:
            if os.path.exists(tmp_name):
                try:
                    os.remove(tmp_name)
                except OSError:
                    pass
            raise

    def load(self, job_id: str) -> Job:
        path = self._path_for(job_id)
        with open(path, "r", encoding="utf-8") as f:
            return Job.from_json(f.read())

    def list_jobs(self) -> List[Job]:
        jobs: List[Job] = []
        for path in self.jobs_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    jobs.append(Job.from_json(f.read()))
            except (OSError, json.JSONDecodeError, KeyError) as e:
                logger.error("Skipping unreadable job file %s: %s", path, e)
        jobs.sort(key=lambda j: j.created, reverse=True)
        return jobs

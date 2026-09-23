"""Deterministic, explainable input planning for the Retouch CLI.

The CLI used to discover paths and immediately hand the resulting list to the
renderer.  This module keeps selection, inspection, and execution as separate
steps so a dry-run can describe the same files and destinations that execution
will use.  It deliberately does not import the engine; header inspection and
path planning must remain usable without model initialization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatchcase
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .io import IMAGE_EXTENSIONS, RAW_EXTENSIONS


INPUT_PLAN_SCHEMA_VERSION = 1
_JPEG_EXTENSIONS = {".jpg", ".jpeg", ".jpe"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def path_key(path: Path) -> str:
    """Return a stable comparison key without changing the displayed path."""
    resolved = str(Path(path).expanduser().resolve(strict=False))
    if os.name == "nt" or sys_platform_is_macos():
        return resolved.casefold()
    return resolved


def sys_platform_is_macos() -> bool:
    # Kept as a function so tests can monkeypatch the platform decision without
    # changing the public planner contract.
    import sys

    return sys.platform == "darwin"


def _json_default(value: Any) -> str:
    return repr(value)


def stable_fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> Optional[str]:
    """Hash one file in bounded chunks for plan/result identity."""
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _normalise_relative(path: Path, root: Optional[Path]) -> str:
    if root is not None:
        try:
            return path.resolve(strict=False).relative_to(
                root.resolve(strict=False)
            ).as_posix()
        except ValueError:
            pass
    return path.name


def _pattern_matches(path: Path, root: Optional[Path], pattern: str) -> bool:
    relative = _normalise_relative(path, root)
    candidates = {relative, path.name, str(path).replace(os.sep, "/")}
    wanted = str(pattern).replace(os.sep, "/")
    wanted_lower = wanted.casefold()
    return any(
        fnmatchcase(candidate, wanted)
        or fnmatchcase(candidate.casefold(), wanted_lower)
        for candidate in candidates
    )


def _is_hidden(path: Path, root: Optional[Path]) -> bool:
    relative = _normalise_relative(path, root)
    return any(part.startswith(".") for part in Path(relative).parts)


def _safe_stat(path: Path) -> Tuple[Optional[int], Optional[int]]:
    try:
        stat = path.stat()
    except OSError:
        return None, None
    return stat.st_size, stat.st_mtime_ns


def _file_identity(path: Path) -> Optional[Tuple[int, int]]:
    try:
        stat = path.stat()
    except OSError:
        return None
    return int(stat.st_dev), int(stat.st_ino)


def _source_kind(path: Path) -> str:
    return "raw" if path.suffix.lower() in RAW_EXTENSIONS else "image"


@dataclass
class InputRecord:
    """One selected, excluded, or rejected input observation."""

    token: str
    path: Optional[str] = None
    root: Optional[str] = None
    relative_path: Optional[str] = None
    status: str = "candidate"
    reason: Optional[str] = None
    selected: bool = False
    extension: Optional[str] = None
    source_kind: Optional[str] = None
    size_bytes: Optional[int] = None
    modified_ns: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    mode: Optional[str] = None
    detected_format: Optional[str] = None
    icc_present: Optional[bool] = None
    exif_orientation: Optional[int] = None
    frame_count: Optional[int] = None
    decoder_available: Optional[bool] = None
    decoder_requested: Optional[str] = None
    decoder_actual: Optional[str] = None
    header_status: str = "not_checked"
    decode_status: str = "not_checked"
    alpha_policy: Optional[str] = None
    planned_output_format: Optional[str] = None
    estimated_working_bytes: Optional[int] = None
    output_stem: Optional[str] = None
    planned_output: Optional[str] = None
    planned_artifacts: List[str] = field(default_factory=list)
    source_sha256: Optional[str] = None
    output_sha256: Optional[str] = None
    artifact_sha256: Dict[str, str] = field(default_factory=dict)
    result_status: Optional[str] = None
    result_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InputPlan:
    """Serializable selection and preflight plan shared by preview/run."""

    input_tokens: List[str]
    recursive: bool
    include: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)
    include_hidden: bool = False
    raw_jpeg_policy: str = "error"
    created_at: str = field(default_factory=_utc_now)
    rows: List[InputRecord] = field(default_factory=list)
    issues: List[Dict[str, str]] = field(default_factory=list)
    output_dir: Optional[str] = None
    output_format: Optional[str] = None
    bit_depth: Optional[int] = None
    compare: Optional[bool] = None
    config_fingerprint: Optional[str] = None

    @property
    def selected_records(self) -> List[InputRecord]:
        return [row for row in self.rows if row.selected and row.path]

    @property
    def selected_paths(self) -> List[Path]:
        return [Path(row.path) for row in self.selected_records]

    @property
    def execution_paths(self) -> List[Path]:
        """Selected paths that still require work after verified resume."""
        return [
            Path(row.path)
            for row in self.selected_records
            if row.status != "resume_verified"
        ]

    @property
    def blocking_issues(self) -> List[Dict[str, str]]:
        return [issue for issue in self.issues if issue.get("severity") == "error"]

    @property
    def estimated_working_bytes(self) -> int:
        return sum(row.estimated_working_bytes or 0 for row in self.selected_records)

    def add_issue(self, code: str, message: str, severity: str = "error") -> None:
        self.issues.append({"code": code, "message": message, "severity": severity})

    def row_for(self, path: Path) -> Optional[InputRecord]:
        key = path_key(path)
        for row in self.rows:
            if row.path and path_key(Path(row.path)) == key:
                return row
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": INPUT_PLAN_SCHEMA_VERSION,
            "created_at": self.created_at,
            "input_tokens": list(self.input_tokens),
            "selection": {
                "recursive": self.recursive,
                "include": list(self.include),
                "exclude": list(self.exclude),
                "include_hidden": self.include_hidden,
                "raw_jpeg_policy": self.raw_jpeg_policy,
            },
            "run": {
                "output_dir": self.output_dir,
                "output_format": self.output_format,
                "bit_depth": self.bit_depth,
                "compare": self.compare,
                "config_fingerprint": self.config_fingerprint,
            },
            "summary": {
                "selected": len(self.selected_records),
                "rows": len(self.rows),
                "estimated_working_bytes": self.estimated_working_bytes,
                "blocking_issues": len(self.blocking_issues),
            },
            "issues": list(self.issues),
            "rows": [row.to_dict() for row in self.rows],
        }

    def write(self, path: Path) -> Path:
        target = Path(path).expanduser().resolve(strict=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return target


def load_input_list(path: Path) -> List[str]:
    """Load a versioned literal-path JSON list.

    Relative entries are resolved against the list's directory unless the JSON
    supplies a relative ``base_dir``. Nested includes and shell expressions are
    intentionally not interpreted.
    """
    source = Path(path).expanduser().resolve()
    data = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(data, list):
        entries = data
        base_dir = source.parent
    elif isinstance(data, dict):
        version = data.get("schema_version", 1)
        if version != 1:
            raise ValueError(f"Unsupported input-list schema_version: {version!r}")
        entries = data.get("paths", data.get("inputs"))
        if not isinstance(entries, list):
            raise ValueError("input-list JSON requires a 'paths' array")
        configured_base = data.get("base_dir")
        base_dir = Path(configured_base) if configured_base else source.parent
        if not base_dir.is_absolute():
            base_dir = source.parent / base_dir
        base_dir = base_dir.expanduser().resolve(strict=False)
    else:
        raise ValueError("input-list JSON must be an object or array")

    result: List[str] = []
    for entry in entries:
        if not isinstance(entry, str) or not entry:
            raise ValueError("input-list paths must be non-empty strings")
        if "\x00" in entry:
            raise ValueError("input-list paths cannot contain NUL characters")
        candidate = Path(entry).expanduser()
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        result.append(str(candidate))
    return result


def _record_path(
    plan: InputPlan,
    token: str,
    candidate: Path,
    root: Optional[Path],
    discovered: bool,
) -> Optional[InputRecord]:
    """Add one filesystem observation and return it when it is selectable."""
    record = InputRecord(
        token=token,
        path=str(candidate.expanduser().resolve(strict=False)),
        root=str(root.resolve(strict=False)) if root else None,
        relative_path=_normalise_relative(candidate, root),
        extension=candidate.suffix.lower() or None,
    )
    plan.rows.append(record)

    if not candidate.exists():
        record.status = "missing"
        record.reason = "path does not exist"
        return None
    if not candidate.is_file():
        record.status = "rejected_not_regular_file"
        record.reason = "path is not a regular file"
        return None

    record.source_kind = _source_kind(candidate)
    record.size_bytes, record.modified_ns = _safe_stat(candidate)
    record.estimated_working_bytes = (
        max(24 * 1024 * 1024, (record.size_bytes or 0) * 4)
    )

    if discovered and not plan.include_hidden and _is_hidden(candidate, root):
        record.status = "excluded_hidden"
        record.reason = "hidden path excluded by default discovery policy"
        return None
    if plan.include and not any(
        _pattern_matches(candidate, root, pattern) for pattern in plan.include
    ):
        record.status = "excluded_include_filter"
        record.reason = "did not match any --include pattern"
        return None
    if any(_pattern_matches(candidate, root, pattern) for pattern in plan.exclude):
        record.status = "excluded_exclude_filter"
        record.reason = "matched an --exclude pattern"
        return None
    if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
        record.status = "rejected_unsupported_extension"
        record.reason = "extension is not in the CLI image allowlist"
        return None

    key = path_key(candidate)
    identity = _file_identity(candidate)
    if any(
        row.selected
        and row.path
        and (
            path_key(Path(row.path)) == key
            or (
                identity is not None
                and _file_identity(Path(row.path)) == identity
            )
        )
        for row in plan.rows
    ):
        record.status = "duplicate_reference"
        record.reason = "same filesystem object was already selected"
        return None

    record.status = "selected"
    record.selected = True
    return record


def _iter_directory(path: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        # Path.rglob does not follow directory symlinks by default on the
        # supported Python versions, which avoids cycles and outside-root walks.
        yield from sorted(path.rglob("*"), key=lambda item: str(item).casefold())
    else:
        yield from sorted(path.iterdir(), key=lambda item: str(item).casefold())


def _apply_raw_jpeg_policy(plan: InputPlan) -> None:
    grouped: Dict[Tuple[str, str], List[InputRecord]] = {}
    for row in plan.selected_records:
        path = Path(row.path or "")
        if path.suffix.lower() not in RAW_EXTENSIONS and path.suffix.lower() not in _JPEG_EXTENSIONS:
            continue
        key = (str(path.parent.resolve(strict=False)).casefold(), path.stem.casefold())
        grouped.setdefault(key, []).append(row)

    for pair in grouped.values():
        raw = [row for row in pair if row.extension in RAW_EXTENSIONS]
        jpeg = [row for row in pair if row.extension in _JPEG_EXTENSIONS]
        if not raw or not jpeg:
            continue
        if plan.raw_jpeg_policy == "raw-only":
            for row in jpeg:
                row.selected = False
                row.status = "excluded_raw_jpeg_policy"
                row.reason = "RAW selected; matching JPEG excluded by policy"
        elif plan.raw_jpeg_policy == "jpeg-only":
            for row in raw:
                row.selected = False
                row.status = "excluded_raw_jpeg_policy"
                row.reason = "JPEG selected; matching RAW excluded by policy"
        elif plan.raw_jpeg_policy == "suffix":
            for row in raw:
                row.output_stem = f"{Path(row.path or '').stem}_raw"
            for row in jpeg:
                row.output_stem = f"{Path(row.path or '').stem}_jpeg"
        else:
            plan.add_issue(
                "raw_jpeg_pair",
                "Matching RAW and JPEG inputs require --raw-jpeg-policy raw-only, jpeg-only, or suffix",
            )


def build_input_plan(
    tokens: Sequence[str],
    *,
    recursive: bool = False,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    include_hidden: bool = False,
    raw_jpeg_policy: str = "error",
) -> InputPlan:
    """Discover and classify literal input tokens deterministically."""
    if raw_jpeg_policy not in {"error", "raw-only", "jpeg-only", "suffix"}:
        raise ValueError(f"unknown raw_jpeg_policy: {raw_jpeg_policy}")
    plan = InputPlan(
        input_tokens=[str(token) for token in tokens],
        recursive=bool(recursive),
        include=list(include or []),
        exclude=list(exclude or []),
        include_hidden=bool(include_hidden),
        raw_jpeg_policy=raw_jpeg_policy,
    )

    for token in plan.input_tokens:
        candidate = Path(token).expanduser()
        if not candidate.exists():
            _record_path(plan, token, candidate, None, False)
            continue
        if candidate.is_dir():
            root = candidate.resolve(strict=False)
            for child in _iter_directory(candidate, recursive):
                _record_path(plan, token, child, root, True)
        else:
            _record_path(plan, token, candidate, candidate.parent, False)

    _apply_raw_jpeg_policy(plan)
    if not plan.selected_records:
        plan.add_issue(
            "no_selected_inputs",
            "No supported image files were selected",
            severity="warning",
        )
    return plan


def inspect_plan_headers(
    plan: InputPlan,
    max_pixels: Optional[int] = None,
    multi_frame_policy: str = "error",
    raw_decoder: str = "rawpy",
    raf2jpeg_path: Optional[str] = None,
) -> None:
    """Inspect non-RAW headers and report optional RAW decoder availability."""
    if multi_frame_policy not in {"error", "first"}:
        raise ValueError(f"unknown multi_frame_policy: {multi_frame_policy}")
    for row in plan.selected_records:
        path = Path(row.path or "")
        if path.suffix.lower() in RAW_EXTENSIONS:
            row.decoder_requested = raw_decoder
            if raw_decoder == "raf2jpeg":
                try:
                    from .io import resolve_raf2jpeg

                    resolve_raf2jpeg(raf2jpeg_path)
                    row.decoder_available = True
                except (FileNotFoundError, OSError):
                    row.decoder_available = False
            elif raw_decoder == "rawpy-fuji-match":
                raf_available = False
                try:
                    from .io import resolve_raf2jpeg

                    resolve_raf2jpeg(raf2jpeg_path)
                    raf_available = True
                except (FileNotFoundError, OSError):
                    pass
                row.decoder_available = (
                    importlib.util.find_spec("rawpy") is not None
                    and raf_available
                )
            else:
                row.decoder_available = importlib.util.find_spec("rawpy") is not None
            row.header_status = "available" if row.decoder_available else "unavailable"
            if not row.decoder_available:
                plan.add_issue(
                    "raw_decoder_unavailable",
                    f"{path.name}: rawpy is unavailable for RAW input",
                )
            continue

        try:
            from PIL import Image

            with Image.open(path) as image:
                row.detected_format = image.format
                row.width, row.height = image.size
                row.mode = image.mode
                row.frame_count = int(getattr(image, "n_frames", 1) or 1)
                row.icc_present = bool(image.info.get("icc_profile"))
                row.decoder_requested = "pillow"
                if image.mode in ("RGBA", "LA", "PA"):
                    row.alpha_policy = "flattened-white"
                orientation = image.getexif().get(0x0112)
                row.exif_orientation = int(orientation) if orientation else None
                if row.frame_count > 1 and multi_frame_policy == "error":
                    raise ValueError(
                        f"multi-frame input has {row.frame_count} frames; "
                        "choose --multi-frame-policy first explicitly"
                    )
                if max_pixels is not None and row.width * row.height > max_pixels:
                    raise ValueError(
                        f"{row.width}x{row.height} exceeds max input pixels {max_pixels}"
                    )
            # Pillow requires verify() to be called directly after open(); it
            # also requires a fresh open if metadata was inspected first.
            with Image.open(path) as image:
                image.verify()
            row.header_status = "passed"
            if row.frame_count and row.frame_count > 1:
                plan.add_issue(
                    "multi_frame_first",
                    f"{path.name}: only the first of {row.frame_count} frames will be processed",
                    severity="warning",
                )
            row.estimated_working_bytes = max(
                24 * 1024 * 1024,
                int(row.width or 0) * int(row.height or 0) * 3 * 4,
            )
        except Exception as exc:
            row.header_status = "failed"
            row.reason = f"header check failed: {exc}"
            plan.add_issue("header_check_failed", f"{path.name}: {exc}")


def decode_plan_inputs(
    plan: InputPlan,
    *,
    raw_decoder: str = "rawpy",
    raf2jpeg_path: Optional[str] = None,
    raf2jpeg_quality: int = 100,
    fuji_match_strength: float = 0.85,
    optical_correction: bool = False,
) -> None:
    """Run the configured decoder route without starting the retouch engine."""
    from .io import imread_engine_with_context

    for row in plan.selected_records:
        path = Path(row.path or "")
        try:
            image, _context = imread_engine_with_context(
                path,
                raw_decoder=raw_decoder,
                raf2jpeg_path=raf2jpeg_path,
                raf2jpeg_quality=raf2jpeg_quality,
                fuji_match_strength=fuji_match_strength,
                optical_correction=optical_correction,
            )
            if image is None or getattr(image, "ndim", 0) != 3:
                raise ValueError("decoder returned no 3-channel image")
            row.decode_status = "passed"
            row.decoder_actual = (
                raw_decoder
                if path.suffix.lower() in RAW_EXTENSIONS
                else "pillow/opencv"
            )
            row.width, row.height = int(image.shape[1]), int(image.shape[0])
            row.estimated_working_bytes = max(
                24 * 1024 * 1024, row.width * row.height * 3 * 4
            )
        except Exception as exc:
            row.decode_status = "failed"
            row.reason = f"decode check failed: {exc}"
            plan.add_issue("decode_check_failed", f"{path.name}: {exc}")


def apply_resume_plan(
    plan: InputPlan,
    path: Path,
    config_fingerprint: str,
    *,
    force: bool = False,
) -> None:
    """Mark only cryptographically verified completed rows as resumable.

    A previous plan is advisory until its configuration, source hash, output
    hash, and output path all match the current filesystem. Any mismatch leaves
    the current row selected for a fresh render; an existing stale output
    blocks unless ``force`` explicitly authorizes replacement. A config
    mismatch blocks the entire run so an operator cannot accidentally mix
    settings.
    """
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        plan.add_issue("resume_plan_invalid", f"Could not read resume plan: {exc}")
        return
    previous_run = payload.get("run") if isinstance(payload, dict) else None
    if not isinstance(previous_run, dict):
        plan.add_issue("resume_plan_invalid", "Resume plan has no run metadata")
        return
    previous_fingerprint = previous_run.get("config_fingerprint")
    if previous_fingerprint != config_fingerprint:
        plan.add_issue(
            "resume_config_mismatch",
            "Resume plan settings do not match this run; no rows were skipped",
        )
        return

    previous_rows = payload.get("rows", []) if isinstance(payload, dict) else []
    by_path: Dict[str, Dict[str, Any]] = {}
    for previous in previous_rows:
        if not isinstance(previous, dict) or not previous.get("path"):
            continue
        by_path[path_key(Path(previous["path"]))] = previous

    for row in plan.selected_records:
        current_path = Path(row.path or "")
        previous = by_path.get(path_key(current_path))
        if not previous or previous.get("result_status") != "done":
            continue
        previous_source_hash = previous.get("source_sha256")
        previous_output_hash = previous.get("output_sha256")
        previous_output = previous.get("planned_output")
        current_source_hash = sha256_file(current_path)
        current_output_hash = (
            sha256_file(Path(previous_output)) if previous_output else None
        )
        previous_artifacts = previous.get("artifact_sha256") or {}
        artifacts_match = bool(previous_artifacts)
        if artifacts_match:
            for artifact, expected_hash in previous_artifacts.items():
                if sha256_file(Path(artifact)) != expected_hash:
                    artifacts_match = False
                    break
        if (
            previous_source_hash
            and previous_output_hash
            and previous_output
            and current_source_hash == previous_source_hash
            and current_output_hash == previous_output_hash
            and artifacts_match
            and row.planned_output == previous_output
        ):
            row.status = "resume_verified"
            row.result_status = "resume_verified"
            row.source_sha256 = current_source_hash
            row.output_sha256 = current_output_hash
            row.artifact_sha256 = dict(previous_artifacts)
        else:
            row.reason = "prior result did not match current source/output; rerendering"
            if previous_output and Path(previous_output).exists() and not force:
                plan.add_issue(
                    "resume_output_mismatch",
                    f"{current_path.name}: prior output is stale or incomplete; "
                    "use --force to replace it explicitly",
                )


def attach_destinations(
    plan: InputPlan,
    *,
    output_dir: Optional[Path],
    format_arg: str,
    bit_depth: int,
    compare: bool,
    save_session: Any,
    recursive_root: Optional[Path],
    destination_builder: Callable[..., Path],
    output_format_resolver: Callable[[Path, str], str],
) -> None:
    """Attach planned output/artifact paths using the CLI's existing policy."""
    plan.output_dir = str(output_dir) if output_dir else None
    plan.output_format = format_arg
    plan.bit_depth = bit_depth
    plan.compare = compare
    for row in plan.selected_records:
        source = Path(row.path or "")
        fmt = output_format_resolver(source, format_arg)
        if bit_depth == 16 and fmt not in ("png", "tif", "tiff"):
            fmt = "png"
        row.planned_output_format = fmt
        output = destination_builder(
            source,
            output_dir,
            fmt,
            input_root=recursive_root,
            output_stem=row.output_stem,
        )
        row.planned_output = str(output)
        artifacts = [output]
        if compare:
            artifacts.append(output.with_name(f"{output.stem}_compare{output.suffix}"))
        if save_session is True:
            artifacts.append(output.with_suffix(".session.json"))
        elif save_session not in (None, False):
            artifacts.append(Path(str(save_session)).expanduser().resolve(strict=False))
        row.planned_artifacts = [str(item) for item in artifacts]


def print_input_plan(plan: InputPlan) -> None:
    """Print a compact human-readable plan without leaking pixel data."""
    selected = plan.selected_records
    print(
        f"Input plan — {len(selected)} selected, {len(plan.rows)} observed, "
        f"{len(plan.blocking_issues)} blocking issue(s)"
    )
    for row in plan.rows:
        label = row.path or row.token
        detail = row.reason or row.planned_output or ""
        if row.selected:
            if "failed" in (row.header_status, row.decode_status):
                marker = "✖"
            elif row.status == "resume_verified":
                marker = "↺"
            else:
                marker = "✓"
            suffix = " [verified resume]" if row.status == "resume_verified" else ""
            observations = []
            if row.header_status != "not_checked":
                observations.append(f"header={row.header_status}")
            if row.decode_status != "not_checked":
                observations.append(f"decode={row.decode_status}")
            if row.decoder_requested:
                observations.append(f"decoder={row.decoder_requested}")
            if row.width and row.height:
                observations.append(f"{row.width}x{row.height}")
            observation_text = f" ({', '.join(observations)})" if observations else ""
            print(
                f"  {marker} {label} -> {row.planned_output or '(output unresolved)'}"
                f"{observation_text}{suffix}"
            )
        else:
            print(f"  · {label} [{row.status}] {detail}".rstrip())
    if plan.estimated_working_bytes:
        print(f"Estimated working memory (sum, not peak): {plan.estimated_working_bytes / (1024 ** 3):.2f} GiB")
    for issue in plan.issues:
        print(f"  {'✖' if issue['severity'] == 'error' else '⚠'} {issue['message']}")

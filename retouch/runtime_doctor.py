"""Dependency-light runtime diagnostics for Retouch.

This module deliberately imports only Python's standard library at module load
time.  Retouch's image stack has several optional/native dependencies, and a
runtime doctor must still be useful when one of those imports is absent or
broken.  Checks that need MediaPipe, OpenCV, ONNX Runtime, or Retouch's heavy
modules are lazy and best-effort.

The public entry point is :func:`collect_runtime_report`.  It returns a JSON
serializable dictionary with stable ``status`` and ``reason_code`` fields.
``python -m retouch.runtime_doctor`` prints the same report as JSON when the
package initializer permits the submodule to be imported.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse


SCHEMA_VERSION = 1

STATUS_OK = "ok"
STATUS_WARNING = "warning"
STATUS_ERROR = "error"
STATUS_UNAVAILABLE = "unavailable"
STATUS_UNKNOWN = "unknown"

_STATUSES = {
    STATUS_OK,
    STATUS_WARNING,
    STATUS_ERROR,
    STATUS_UNAVAILABLE,
    STATUS_UNKNOWN,
}

_DEFAULT_ROOT = Path(__file__).resolve().parents[1]
_OPENCV_DISTRIBUTIONS = (
    "opencv-python",
    "opencv-contrib-python",
    "opencv-python-headless",
    "opencv-contrib-python-headless",
)
_PLACEHOLDER_URL_PART = "github.com/owner/retouch-models"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_DISTRIBUTION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*")
_SPECIFIER_RE = re.compile(r"^(~=|==|!=|>=|<=|>|<)\s*(.+)$")


def _result(status: str, reason_code: str, reason: str, **fields: Any) -> Dict[str, Any]:
    """Build a consistently shaped check result."""
    if status not in _STATUSES:
        raise ValueError("unknown runtime-doctor status: %s" % status)
    value: Dict[str, Any] = {
        "status": status,
        "reason_code": reason_code,
        "reason": reason,
    }
    value.update(fields)
    return value


def _aggregate_status(results: Iterable[Mapping[str, Any]], *, empty: str = STATUS_UNKNOWN) -> str:
    """Return the most important status from child checks."""
    statuses = [str(item.get("status")) for item in results if item]
    if not statuses:
        return empty
    if STATUS_ERROR in statuses:
        return STATUS_ERROR
    if STATUS_WARNING in statuses:
        return STATUS_WARNING
    if STATUS_UNKNOWN in statuses:
        return STATUS_UNKNOWN
    if STATUS_UNAVAILABLE in statuses:
        return STATUS_UNAVAILABLE
    return STATUS_OK


def _safe_detail(exc: BaseException) -> str:
    """Return bounded exception detail without allowing diagnostics to raise."""
    text = str(exc).strip().replace("\x00", " ")
    if len(text) > 500:
        text = text[:497] + "..."
    return "%s: %s" % (type(exc).__name__, text) if text else type(exc).__name__


def _canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", str(name).strip().lower())


def _safe_import(module_name: str) -> Tuple[Optional[ModuleType], Optional[str]]:
    """Import an optional module without making it a doctor dependency."""
    try:
        return importlib.import_module(module_name), None
    except Exception as exc:  # optional/native imports can fail in many ways
        return None, _safe_detail(exc)


def _module_exists(module_name: str) -> Tuple[bool, Optional[str]]:
    """Check for an importable module without importing its native code."""
    try:
        return importlib.util.find_spec(module_name) is not None, None
    except Exception as exc:
        return False, _safe_detail(exc)


def _strip_toml_comment(line: str) -> str:
    """Strip a TOML comment while preserving ``#`` inside quoted strings."""
    quote: Optional[str] = None
    escaped = False
    for index, char in enumerate(line):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\" and quote == '"':
                escaped = True
            elif char == quote:
                quote = None
        elif char in {'"', "'"}:
            quote = char
        elif char == "#":
            return line[:index]
    return line


def _project_section(text: str) -> str:
    match = re.search(r"(?ms)^\[project\]\s*(.*?)(?=^\[|\Z)", text)
    return match.group(1) if match else ""


def _project_optional_section(text: str) -> str:
    match = re.search(
        r"(?ms)^\[project\.optional-dependencies\]\s*(.*?)(?=^\[|\Z)",
        text,
    )
    return match.group(1) if match else ""


def _fallback_array(section: str, key: str) -> Optional[List[Any]]:
    """Read the simple string arrays used by this project's pyproject.toml.

    Python 3.9 has no stdlib TOML parser.  This fallback is intentionally
    narrow: it handles a TOML array of quoted values and returns ``None`` for
    unsupported syntax instead of guessing.
    """
    lines = section.splitlines()
    start: Optional[int] = None
    first_value = ""
    for index, line in enumerate(lines):
        clean = _strip_toml_comment(line).strip()
        match = re.match(r"^%s\s*=\s*(.*)$" % re.escape(key), clean)
        if match:
            start = index
            first_value = match.group(1)
            break
    if start is None:
        return None

    chunks = [first_value]
    balance = first_value.count("[") - first_value.count("]")
    index = start + 1
    while balance > 0 and index < len(lines):
        chunk = _strip_toml_comment(lines[index]).strip()
        chunks.append(chunk)
        balance += chunk.count("[") - chunk.count("]")
        index += 1
    if balance != 0:
        return None
    try:
        value = ast.literal_eval("\n".join(chunks))
    except (SyntaxError, ValueError):
        return None
    return value if isinstance(value, list) else None


def _fallback_string(section: str, key: str) -> Optional[str]:
    match = re.search(
        r"(?m)^\s*%s\s*=\s*([\"'])(.*?)\1\s*(?:#.*)?$" % re.escape(key),
        section,
    )
    return match.group(2) if match else None


def _dependency_spec(raw: Any) -> Optional[Dict[str, str]]:
    if not isinstance(raw, str):
        return None
    value = raw.split(";", 1)[0].strip()
    match = _DISTRIBUTION_RE.match(value)
    if not match:
        return None
    name = match.group(0)
    remainder = value[match.end():].strip()
    if remainder.startswith("["):
        closing = remainder.find("]")
        if closing < 0:
            return None
        remainder = remainder[closing + 1:].strip()
    return {
        "name": name,
        "canonical_name": _canonical_name(name),
        "specifier": remainder,
        "raw": raw,
    }


def read_declared_project(pyproject_path: Optional[os.PathLike[str] | str] = None) -> Dict[str, Any]:
    """Read ``requires-python`` and core dependency constraints.

    The function uses ``tomllib``/``tomli`` when available and a small safe
    fallback on Python 3.9.  It never imports a project dependency.
    """
    path = Path(pyproject_path) if pyproject_path is not None else _DEFAULT_ROOT / "pyproject.toml"
    base: Dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "parser": None,
        "requires_python": None,
        "dependencies": [],
        "optional_dependencies": {},
    }
    if not path.is_file():
        base.update(_result(
            STATUS_UNKNOWN,
            "declaration_file_missing",
            "declared project constraints are unavailable",
        ))
        return base
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        base.update(_result(
            STATUS_ERROR,
            "declaration_file_unreadable",
            "declared project constraints could not be read",
            detail=_safe_detail(exc),
        ))
        return base

    parsed: Optional[Mapping[str, Any]] = None
    parser_name = "fallback"
    try:
        toml_parser: Any = None
        try:
            import tomllib as toml_parser  # type: ignore
        except ImportError:
            try:
                import tomli as toml_parser  # type: ignore
            except ImportError:
                toml_parser = None
        if toml_parser is not None:
            parsed = toml_parser.loads(text)
            parser_name = "tomllib"
    except Exception:
        # A malformed TOML file can still be diagnosed by the narrow fallback.
        parsed = None

    if parsed is not None:
        project = parsed.get("project", {})
        if isinstance(project, Mapping):
            requires_python = project.get("requires-python")
            dependencies = project.get("dependencies", [])
            optional_dependencies = project.get("optional-dependencies", {})
        else:
            requires_python = None
            dependencies = []
            optional_dependencies = {}
    else:
        section = _project_section(text)
        requires_python = _fallback_string(section, "requires-python")
        dependencies = _fallback_array(section, "dependencies") or []
        optional_section = _project_optional_section(text)
        optional_dependencies = {}
        for group_match in re.finditer(
            r"(?m)^\s*([A-Za-z0-9_.-]+)\s*=", optional_section
        ):
            group = group_match.group(1)
            values = _fallback_array(optional_section, group)
            if values is not None:
                optional_dependencies[group] = values
        parser_name = "fallback"

    parsed_dependencies = []
    for dependency in dependencies if isinstance(dependencies, list) else []:
        item = _dependency_spec(dependency)
        if item is not None:
            parsed_dependencies.append(item)
    parsed_optional_dependencies: Dict[str, List[Dict[str, str]]] = {}
    if isinstance(optional_dependencies, Mapping):
        for group, group_dependencies in optional_dependencies.items():
            parsed_group = []
            if isinstance(group_dependencies, list):
                for dependency in group_dependencies:
                    item = _dependency_spec(dependency)
                    if item is not None:
                        parsed_group.append(item)
            parsed_optional_dependencies[str(group)] = parsed_group
    base.update({
        "parser": parser_name,
        "requires_python": str(requires_python) if requires_python else None,
        "dependencies": parsed_dependencies,
        "optional_dependencies": parsed_optional_dependencies,
    })
    if not base["requires_python"] and not parsed_dependencies:
        base.update(_result(
            STATUS_WARNING,
            "declaration_constraints_empty",
            "project file contains no readable runtime constraints",
        ))
    else:
        base.update(_result(
            STATUS_OK,
            "declarations_read",
            "declared runtime constraints were read",
        ))
    return base


def _numeric_version(value: Any) -> Tuple[int, ...]:
    text = str(value).strip().lstrip("vV")
    numbers = re.findall(r"\d+", text)
    return tuple(int(number) for number in numbers) or (0,)


def version_satisfies(version: str, specifier: str) -> bool:
    """Evaluate common PEP 440 specifiers without requiring ``packaging``."""
    specifier = str(specifier or "").split(";", 1)[0].strip()
    if not specifier:
        return True
    try:
        from packaging.specifiers import SpecifierSet  # type: ignore

        return bool(SpecifierSet(specifier).contains(str(version), prereleases=True))
    except Exception:
        installed = _numeric_version(version)
        for part in specifier.split(","):
            match = _SPECIFIER_RE.match(part.strip())
            if not match:
                return False
            operator, requested_text = match.groups()
            requested = _numeric_version(requested_text.rstrip(".*"))
            if operator == "==" and requested_text.endswith(".*"):
                if installed[:len(requested)] != requested:
                    return False
                continue
            if operator == "==" and installed != requested:
                return False
            if operator == "!=" and installed == requested:
                return False
            if operator == ">=" and installed < requested:
                return False
            if operator == "<=" and installed > requested:
                return False
            if operator == ">" and installed <= requested:
                return False
            if operator == "<" and installed >= requested:
                return False
            if operator == "~=":
                if installed < requested:
                    return False
                if len(requested) <= 1:
                    upper = (requested[0] + 1,)
                else:
                    upper = requested[:-1] + (requested[-1] + 1,)
                if installed >= upper:
                    return False
        return True


def check_interpreter(declarations: Mapping[str, Any]) -> Dict[str, Any]:
    """Check the active interpreter against the declared Python constraint."""
    version = platform.python_version()
    specifier = declarations.get("requires_python")
    fields: Dict[str, Any] = {
        "python_version": version,
        "version_info": [
            int(sys.version_info.major),
            int(sys.version_info.minor),
            int(sys.version_info.micro),
        ],
        "executable": str(sys.executable),
        "implementation": platform.python_implementation(),
        "platform": sys.platform,
        "declared_requires_python": specifier,
    }
    if not specifier:
        return _result(
            STATUS_UNKNOWN,
            "python_constraint_unavailable",
            "declared Python constraint is unavailable",
            compatible=None,
            **fields,
        )
    compatible = version_satisfies(version, str(specifier))
    if compatible:
        return _result(
            STATUS_OK,
            "python_satisfies_constraint",
            "interpreter satisfies the declared Python constraint",
            compatible=True,
            **fields,
        )
    return _result(
        STATUS_ERROR,
        "python_constraint_mismatch",
        "interpreter does not satisfy the declared Python constraint",
        compatible=False,
        **fields,
    )


def _installed_distribution_version(name: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        return str(importlib_metadata.version(name)), None
    except importlib_metadata.PackageNotFoundError:
        return None, None
    except Exception as exc:
        return None, _safe_detail(exc)


def inspect_packages(
    dependencies: Sequence[Mapping[str, Any]],
    *,
    required: bool = True,
) -> Dict[str, Any]:
    """Report installed versions and compatibility for declared packages."""
    items: Dict[str, Dict[str, Any]] = {}
    for dependency in dependencies:
        name = str(dependency.get("name", ""))
        canonical = str(dependency.get("canonical_name") or _canonical_name(name))
        specifier = str(dependency.get("specifier") or "")
        installed, metadata_error = _installed_distribution_version(name)
        fields: Dict[str, Any] = {
            "distribution": name,
            "installed_version": installed,
            "declared_specifier": specifier,
            "declared_raw": dependency.get("raw"),
            "compatible": None,
        }
        if metadata_error:
            item = _result(
                STATUS_ERROR,
                "distribution_metadata_error",
                "installed distribution metadata could not be read",
                detail=metadata_error,
                **fields,
            )
        elif installed is None:
            item = _result(
                STATUS_ERROR if required else STATUS_UNAVAILABLE,
                "distribution_not_installed" if required else "optional_distribution_not_installed",
                "declared distribution is not installed"
                if required
                else "optional distribution is not installed",
                **fields,
            )
        elif not specifier:
            item = _result(
                STATUS_OK,
                "installed_without_constraint",
                "distribution is installed; no version constraint was declared",
                compatible=True,
                **fields,
            )
        else:
            compatible = version_satisfies(installed, specifier)
            fields["compatible"] = compatible
            item = _result(
                STATUS_OK if compatible else STATUS_ERROR,
                "distribution_satisfies_constraint" if compatible else "distribution_constraint_mismatch",
                "installed distribution satisfies the declared constraint"
                if compatible
                else "installed distribution does not satisfy the declared constraint",
                **fields,
            )
        items[canonical] = item

    status = _aggregate_status(items.values())
    if status == STATUS_OK:
        reason_code = "declared_distributions_compatible"
        reason = "declared distributions are installed and compatible"
    elif status == STATUS_ERROR:
        reason_code = "declared_distribution_problem"
        reason = "one or more declared distributions are missing or incompatible"
    else:
        reason_code = "declared_distribution_status_unresolved"
        reason = "declared distribution compatibility is not fully resolved"
    return {
        "status": status,
        "reason_code": reason_code,
        "reason": reason,
        "items": items,
    }


def inspect_optional_packages(
    optional_dependencies: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Inspect optional dependency groups without making them core blockers."""
    return {
        str(group): inspect_packages(dependencies, required=False)
        for group, dependencies in optional_dependencies.items()
        if isinstance(dependencies, Sequence)
    }


def check_pip_check() -> Dict[str, Any]:
    """Run ``pip check`` without changing the environment."""
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _result(
            STATUS_UNKNOWN,
            "pip_check_unavailable",
            "pip check could not be executed",
            detail=_safe_detail(exc),
        )
    output = "\n".join(
        value.strip() for value in (completed.stdout, completed.stderr) if value and value.strip()
    )
    if completed.returncode == 0:
        return _result(
            STATUS_OK,
            "pip_check_clean",
            "pip check reports no broken requirements",
            returncode=0,
            output=output,
        )
    return _result(
        STATUS_ERROR,
        "pip_check_failed",
        "pip check reports broken requirements",
        returncode=int(completed.returncode),
        output=output[:4000],
    )


def _opencv_metadata() -> Dict[str, str]:
    """Find installed OpenCV distributions without assuming one package name."""
    found: Dict[str, str] = {}
    for name in _OPENCV_DISTRIBUTIONS:
        version, error = _installed_distribution_version(name)
        if version is not None:
            found[name] = version
        elif error:
            # A broken metadata record is still useful evidence to expose.
            found[name] = "metadata-error:%s" % error

    # Python 3.9's importlib.metadata has no packages_distributions(), so scan
    # distribution metadata as a compatibility fallback for renamed packages.
    try:
        distributions = importlib_metadata.distributions()
        for distribution in distributions:
            name = getattr(distribution, "name", None)
            if not name:
                try:
                    name = distribution.metadata.get("Name")
                except Exception:
                    name = None
            if name and _canonical_name(str(name)).startswith("opencv-"):
                try:
                    found.setdefault(str(name), str(distribution.version))
                except Exception:
                    found.setdefault(str(name), "metadata-error")
    except Exception:
        pass
    return found


def check_opencv() -> Dict[str, Any]:
    """Detect duplicate OpenCV distributions and import/build failures."""
    distributions = _opencv_metadata()
    cv2_module, import_error = _safe_import("cv2")
    imported_version = getattr(cv2_module, "__version__", None) if cv2_module else None
    module_path = getattr(cv2_module, "__file__", None) if cv2_module else None
    fields = {
        "distributions": distributions,
        "distribution_count": len(distributions),
        "imported_version": str(imported_version) if imported_version else None,
        "module_path": str(module_path) if module_path else None,
    }
    if len(distributions) > 1:
        return _result(
            STATUS_ERROR,
            "duplicate_opencv_distributions",
            "multiple OpenCV distributions are installed",
            import_error=import_error,
            **fields,
        )
    if import_error:
        return _result(
            STATUS_ERROR if distributions else STATUS_UNAVAILABLE,
            "opencv_import_failed" if distributions else "opencv_not_installed",
            "OpenCV could not be imported" if distributions else "OpenCV is not installed",
            import_error=import_error,
            **fields,
        )
    if not distributions:
        return _result(
            STATUS_WARNING,
            "opencv_metadata_missing",
            "cv2 imports but its distribution metadata is unavailable",
            import_error=None,
            **fields,
        )
    return _result(
        STATUS_OK,
        "single_opencv_distribution",
        "one OpenCV distribution is installed and importable",
        import_error=None,
        **fields,
    )


def _default_cache_dir() -> Path:
    explicit = os.environ.get("RETOUCH_CACHE_DIR")
    if explicit:
        return Path(explicit).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "retouch"
    return Path.home() / ".cache" / "retouch"


def _sha256_file(path: Path) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 16), b""):
                digest.update(chunk)
                size += len(chunk)
    except Exception as exc:
        return None, None, _safe_detail(exc)
    return digest.hexdigest(), size, None


def _manifest_entry_integrity(entry: Mapping[str, Any]) -> Tuple[bool, Optional[str], Optional[int]]:
    expected_hash = str(entry.get("sha256") or "").strip().lower()
    try:
        expected_size = int(entry.get("size_bytes") or 0)
    except (TypeError, ValueError):
        expected_size = 0
    valid = bool(_SHA256_RE.fullmatch(expected_hash)) and expected_size > 0
    return valid, expected_hash or None, expected_size or None


def _controlled_url(url: Any) -> bool:
    value = str(url or "").strip()
    parsed = urlparse(value)
    return bool(
        parsed.scheme == "https"
        and parsed.netloc
        and _PLACEHOLDER_URL_PART not in value
    )


def _model_distribution(entry: Mapping[str, Any]) -> Dict[str, Any]:
    declared = str(entry.get("availability") or "unspecified")
    url = str(entry.get("url") or "").strip()
    integrity_valid, _, _ = _manifest_entry_integrity(entry)
    controlled = _controlled_url(url)
    downloadable = controlled and integrity_valid
    if declared == "downloadable" and not downloadable:
        status = STATUS_ERROR
        reason_code = "download_metadata_invalid"
        reason = "model is declared downloadable but lacks controlled URL or integrity metadata"
    elif downloadable:
        status = STATUS_OK
        reason_code = "controlled_distribution_available"
        reason = "model has a controlled HTTPS distribution URL and integrity metadata"
    else:
        status = STATUS_UNAVAILABLE
        reason_code = "controlled_distribution_unavailable"
        reason = "model has no controlled distribution URL"
    return {
        "status": status,
        "reason_code": reason_code,
        "reason": reason,
        "declared_availability": declared,
        "url": url or None,
        "url_present": bool(url),
        "controlled_url": controlled,
        "downloadable": downloadable,
        "integrity_metadata_valid": integrity_valid,
    }


def _model_candidates(manifest_path: Path, filename: str) -> List[Path]:
    if not filename or Path(filename).name != filename:
        return []
    candidates = [manifest_path.parent / filename, _default_cache_dir() / "models" / filename]
    result: List[Path] = []
    for candidate in candidates:
        if candidate not in result:
            result.append(candidate)
    return result


def _check_one_model(name: str, entry: Mapping[str, Any], manifest_path: Path) -> Dict[str, Any]:
    expected_valid, expected_hash, expected_size = _manifest_entry_integrity(entry)
    distribution = _model_distribution(entry)
    filename = str(entry.get("filename") or "")
    candidates: List[Dict[str, Any]] = []
    verified_path: Optional[str] = None
    mismatch_seen = False
    read_error_seen = False
    for path in _model_candidates(manifest_path, filename):
        candidate: Dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        if not path.is_file():
            candidate.update(_result(
                STATUS_UNAVAILABLE,
                "model_file_missing",
                "model file is not present at this candidate path",
            ))
            candidates.append(candidate)
            continue
        if not expected_valid:
            read_error_seen = True
            candidate.update(_result(
                STATUS_ERROR,
                "model_integrity_metadata_invalid",
                "present model file has invalid manifest integrity metadata",
            ))
            candidates.append(candidate)
            continue
        actual_hash, actual_size, error = _sha256_file(path)
        candidate.update({
            "expected_sha256": expected_hash,
            "expected_size_bytes": expected_size,
            "actual_sha256": actual_hash,
            "actual_size_bytes": actual_size,
        })
        if error:
            read_error_seen = True
            candidate.update(_result(
                STATUS_ERROR,
                "model_hash_read_failed",
                "model file could not be hashed",
                detail=error,
            ))
        elif actual_hash == expected_hash and actual_size == expected_size:
            candidate.update(_result(
                STATUS_OK,
                "model_hash_verified",
                "model file matches the manifest SHA-256 and size",
            ))
            verified_path = str(path)
        else:
            mismatch_seen = True
            candidate.update(_result(
                STATUS_ERROR,
                "model_hash_mismatch",
                "model file does not match the manifest SHA-256 or size",
            ))
        candidates.append(candidate)

    if distribution["status"] == STATUS_ERROR:
        status = STATUS_ERROR
        reason_code = "model_distribution_metadata_invalid"
        reason = "model distribution metadata is invalid for a clean installation"
    elif verified_path:
        status = STATUS_OK
        reason_code = "model_verified"
        reason = "a local model file passed manifest integrity verification"
    elif mismatch_seen or read_error_seen or distribution["status"] == STATUS_ERROR:
        status = STATUS_ERROR
        reason_code = "model_integrity_or_distribution_problem"
        reason = "model integrity or distribution metadata needs attention"
    elif distribution["downloadable"]:
        status = STATUS_WARNING
        reason_code = "model_downloadable_not_local"
        reason = "model is not local but has controlled distribution metadata"
    else:
        status = STATUS_UNAVAILABLE
        reason_code = "model_not_available"
        reason = "model is neither verified locally nor available from a controlled distribution"

    return {
        "status": status,
        "reason_code": reason_code,
        "reason": reason,
        "name": name,
        "filename": filename or None,
        "expected_sha256": expected_hash,
        "expected_size_bytes": expected_size,
        "available": bool(verified_path),
        "path": verified_path,
        "distribution": distribution,
        "candidates": candidates,
        "manifest_entry": dict(entry),
    }


def inspect_models(
    manifest_path: Optional[os.PathLike[str] | str] = None,
) -> Dict[str, Any]:
    """Verify model files and report controlled distribution metadata.

    This is deliberately side-effect-free: it hashes local files but never
    downloads, creates cache directories, or calls ``model_fetch``.
    """
    path = Path(manifest_path) if manifest_path is not None else _DEFAULT_ROOT / "models" / "manifest.json"
    manifest_info: Dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "schema_version": None,
    }
    if not path.is_file():
        manifest_info.update(_result(
            STATUS_ERROR,
            "model_manifest_missing",
            "model manifest is not present",
        ))
        return {
            "status": STATUS_ERROR,
            "reason_code": "model_manifest_missing",
            "reason": "model manifest is not present",
            "manifest": manifest_info,
            "items": {},
        }
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception as exc:
        manifest_info.update(_result(
            STATUS_ERROR,
            "model_manifest_invalid",
            "model manifest could not be parsed",
            detail=_safe_detail(exc),
        ))
        return {
            "status": STATUS_ERROR,
            "reason_code": "model_manifest_invalid",
            "reason": "model manifest could not be parsed",
            "manifest": manifest_info,
            "items": {},
        }
    models = manifest.get("models") if isinstance(manifest, Mapping) else None
    if not isinstance(models, Mapping):
        manifest_info.update(_result(
            STATUS_ERROR,
            "model_manifest_schema_invalid",
            "model manifest has no models mapping",
        ))
        return {
            "status": STATUS_ERROR,
            "reason_code": "model_manifest_schema_invalid",
            "reason": "model manifest has no models mapping",
            "manifest": manifest_info,
            "items": {},
        }
    manifest_info.update({
        "schema_version": manifest.get("schema_version"),
        "status": STATUS_OK,
        "reason_code": "model_manifest_read",
        "reason": "model manifest was read",
    })
    items = {
        str(name): _check_one_model(str(name), entry, path)
        for name, entry in models.items()
        if isinstance(entry, Mapping)
    }
    status = _aggregate_status(items.values())
    if status == STATUS_ERROR:
        reason_code = "model_checks_failed"
        reason = "one or more model integrity or distribution checks failed"
    elif status in {STATUS_WARNING, STATUS_UNAVAILABLE}:
        reason_code = "model_checks_degraded"
        reason = "model checks completed with unavailable or not-local assets"
    else:
        reason_code = "model_checks_passed"
        reason = "all declared model checks passed"
    return {
        "status": status,
        "reason_code": reason_code,
        "reason": reason,
        "manifest": manifest_info,
        "items": items,
    }


def _mediapipe_static_info() -> Dict[str, Any]:
    version, metadata_error = _installed_distribution_version("mediapipe")
    exists, spec_error = _module_exists("mediapipe")
    fields = {
        "installed_version": version,
        "module_present": exists,
        "metadata_error": metadata_error,
        "module_spec_error": spec_error,
    }
    if metadata_error or spec_error:
        fields["status"] = STATUS_ERROR
        fields["reason_code"] = "mediapipe_metadata_error"
        fields["reason"] = "MediaPipe metadata could not be read"
    elif version is None and not exists:
        fields["status"] = STATUS_UNAVAILABLE
        fields["reason_code"] = "mediapipe_not_installed"
        fields["reason"] = "MediaPipe is not installed"
    else:
        fields["status"] = STATUS_OK
        fields["reason_code"] = "mediapipe_distribution_present"
        fields["reason"] = "MediaPipe distribution is present"
    return fields


def check_detector(
    models: Mapping[str, Any],
    *,
    probe: bool = False,
    platform_name: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Report the detector backend without making MediaPipe mandatory.

    ``probe=False`` is the safe default: it uses distribution metadata and
    known backend rules.  ``probe=True`` additionally imports and initializes
    ``FaceDetector(allow_unavailable=True)`` when the static check says that is
    reasonable.  Any Python exception is reported rather than propagated.
    """
    info = _mediapipe_static_info()
    version = info.get("installed_version")
    env = dict(environ or os.environ)
    requested = str(env.get("RETOUCH_MEDIAPIPE_BACKEND", "auto")).strip().lower()
    platform_value = platform_name or sys.platform
    model = models.get("face_landmarker", {}) if isinstance(models, Mapping) else {}
    model_available = bool(model.get("available"))
    fields: Dict[str, Any] = {
        "backend_requested": requested,
        "backend": "unavailable",
        "available": False,
        "probed": bool(probe),
        "mediapipe_version": version,
        "mediapipe_importable": bool(info.get("module_present")),
        "has_legacy_solutions": None,
        "has_tasks_api": None,
        "face_landmarker_model_available": model_available,
        "runtime_status": None,
    }

    if requested not in {"auto", "legacy", "tasks"}:
        return _result(
            STATUS_ERROR,
            "invalid_mediapipe_backend_setting",
            "RETOUCH_MEDIAPIPE_BACKEND has an invalid value",
            **fields,
        )
    if info["status"] == STATUS_UNAVAILABLE:
        return _result(
            STATUS_UNAVAILABLE,
            "mediapipe_not_installed",
            "face-aware detection is unavailable because MediaPipe is not installed",
            **fields,
        )
    if info["status"] == STATUS_ERROR:
        return _result(
            STATUS_ERROR,
            "mediapipe_metadata_error",
            "face-aware detection could not be assessed from MediaPipe metadata",
            detail=info.get("metadata_error") or info.get("module_spec_error"),
            **fields,
        )

    # detection.py has an explicit macOS guard for this known Tasks runtime.
    # Do not import MediaPipe merely to rediscover the same fact: importing a
    # native vision stack can be slow or unsafe in a damaged environment.
    if (
        platform_value == "darwin"
        and version == "0.10.35"
        and requested in {"auto", "tasks"}
    ):
        fields["backend"] = "unavailable"
        fields["available"] = False
        fields["has_legacy_solutions"] = False
        fields["has_tasks_api"] = True
        fields["probe_skipped"] = True
        return _result(
            STATUS_UNAVAILABLE,
            "mediapipe_tasks_unsupported_on_macos",
            "MediaPipe 0.10.35 Tasks FaceLandmarker is unsupported by this macOS runtime",
            **fields,
        )

    legacy = None
    tasks = None
    imported_mp: Optional[ModuleType] = None
    import_error: Optional[str] = None
    if probe:
        imported_mp, import_error = _safe_import("mediapipe")
        if imported_mp is not None:
            legacy = hasattr(imported_mp, "solutions")
            tasks = hasattr(imported_mp, "tasks")
        fields["mediapipe_import_error"] = import_error
    else:
        # MediaPipe 0.10.31+ removed the legacy Solutions API.  This is only a
        # conservative metadata inference; probing records the actual attrs.
        legacy = version == "0.10.5"
        tasks = version is not None and version != "0.10.5"
    fields["has_legacy_solutions"] = legacy
    fields["has_tasks_api"] = tasks

    if requested == "legacy":
        backend = "mediapipe_legacy"
        backend_available = bool(legacy)
    elif requested == "tasks":
        backend = "mediapipe_tasks"
        backend_available = bool(tasks) and model_available
    elif legacy:
        backend = "mediapipe_legacy"
        backend_available = True
    else:
        backend = "mediapipe_tasks"
        backend_available = bool(tasks) and model_available

    fields["backend"] = backend if backend_available else "unavailable"
    fields["available"] = backend_available

    if import_error:
        fields["backend"] = "unavailable"
        fields["available"] = False
        return _result(
            STATUS_ERROR,
            "mediapipe_import_failed",
            "MediaPipe could not be imported for detector probing",
            detail=import_error,
            **fields,
        )
    if not backend_available:
        reason_code = "legacy_backend_missing" if requested == "legacy" else (
            "face_landmarker_model_missing" if tasks else "mediapipe_backend_missing"
        )
        reason = (
            "requested MediaPipe legacy backend is unavailable"
            if requested == "legacy"
            else "face-aware detection is unavailable because its backend or verified model is missing"
        )
        return _result(STATUS_UNAVAILABLE, reason_code, reason, **fields)

    if not probe:
        return _result(
            STATUS_OK,
            "detector_backend_capable_not_probed",
            "detector backend and required metadata are present; native initialization was not probed",
            **fields,
        )

    detection_module, detection_error = _safe_import("retouch.detection")
    detector_class = getattr(detection_module, "FaceDetector", None) if detection_module else None
    if detection_error or detector_class is None:
        return _result(
            STATUS_ERROR,
            "detector_module_import_failed",
            "detector module could not be imported for probing",
            detail=detection_error or "FaceDetector class is missing",
            **fields,
        )
    detector = None
    try:
        detector = detector_class(allow_unavailable=True)
        runtime_status_method = getattr(detector, "runtime_status", None)
        if callable(runtime_status_method):
            runtime_snapshot = runtime_status_method()
            if isinstance(runtime_snapshot, Mapping):
                fields["runtime_status"] = _json_safe(dict(runtime_snapshot))
                actual_available = bool(runtime_snapshot.get("available"))
                actual_backend = str(runtime_snapshot.get("backend", "unknown"))
                actual_reason = runtime_snapshot.get("reason")
            else:
                actual_available = bool(getattr(detector, "available", False))
                actual_backend = str(getattr(detector, "backend_name", "unknown"))
                actual_reason = getattr(detector, "unavailable_reason", None)
        else:
            actual_available = bool(getattr(detector, "available", False))
            actual_backend = str(getattr(detector, "backend_name", "unknown"))
            actual_reason = getattr(detector, "unavailable_reason", None)
    except Exception as exc:
        return _result(
            STATUS_ERROR,
            "detector_initialization_failed",
            "detector native initialization raised an exception",
            detail=_safe_detail(exc),
            **fields,
        )
    finally:
        if detector is not None:
            try:
                detector.close()
            except Exception as exc:
                fields["detector_close_error"] = _safe_detail(exc)
    fields["backend"] = actual_backend
    fields["available"] = actual_available
    if actual_available:
        return _result(
            STATUS_OK,
            "detector_initialized",
            "face-aware detector initialized successfully",
            **fields,
        )
    return _result(
        STATUS_UNAVAILABLE,
        "detector_reported_unavailable",
        "detector initialized in unavailable mode; global-only processing is required",
        detail=str(actual_reason) if actual_reason else None,
        **fields,
    )


def check_onnx_providers() -> Dict[str, Any]:
    """Report ONNX Runtime providers without importing Retouch's engine."""
    version, metadata_error = _installed_distribution_version("onnxruntime")
    exists, spec_error = _module_exists("onnxruntime")
    fields: Dict[str, Any] = {
        "installed_version": version,
        "module_present": exists,
        "available": [],
        "available_providers": [],
        "requested_provider_order": [],
        "active_session_providers": [],
        "active_provider_verified": False,
        "selected": None,
        "selected_is_active": False,
        "import_error": None,
    }
    if metadata_error or spec_error:
        return _result(
            STATUS_ERROR,
            "onnxruntime_metadata_error",
            "ONNX Runtime metadata could not be read",
            detail=metadata_error or spec_error,
            **fields,
        )
    if version is None and not exists:
        return _result(
            STATUS_UNAVAILABLE,
            "onnxruntime_not_installed",
            "ONNX Runtime is not installed",
            **fields,
        )
    ort, import_error = _safe_import("onnxruntime")
    fields["import_error"] = import_error
    if import_error or ort is None:
        return _result(
            STATUS_ERROR,
            "onnxruntime_import_failed",
            "ONNX Runtime could not be imported",
            detail=import_error,
            **fields,
        )
    try:
        available = [str(provider) for provider in ort.get_available_providers()]
    except Exception as exc:
        return _result(
            STATUS_ERROR,
            "onnxruntime_provider_probe_failed",
            "ONNX Runtime execution providers could not be queried",
            detail=_safe_detail(exc),
            **fields,
        )
    preference = (
        "CoreMLExecutionProvider",
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "ROCMExecutionProvider",
        "TensorrtExecutionProvider",
        "CPUExecutionProvider",
    )
    selected = next((name for name in preference if name in available), available[0] if available else None)
    fields["available"] = available
    fields["available_providers"] = available
    fields["requested_provider_order"] = list(preference)
    fields["selected"] = selected
    if not available:
        return _result(
            STATUS_WARNING,
            "onnxruntime_no_execution_providers",
            "ONNX Runtime is installed but reports no execution providers",
            **fields,
        )
    return _result(
        STATUS_OK,
        "onnxruntime_providers_available",
        "ONNX Runtime execution providers were queried successfully",
        **fields,
    )


def check_parser(
    models: Mapping[str, Any],
    onnx: Mapping[str, Any],
    *,
    probe: bool = False,
) -> Dict[str, Any]:
    """Report BiSeNet parser capability and its landmark fallback."""
    model = models.get("resnet18_bisenet", {}) if isinstance(models, Mapping) else {}
    model_available = bool(model.get("available"))
    onnx_available = onnx.get("status") == STATUS_OK if isinstance(onnx, Mapping) else False
    fields: Dict[str, Any] = {
        "backend": "landmark_only",
        "available": True,
        "bisenet_model_available": model_available,
        "onnxruntime_available": onnx_available,
        "probed": bool(probe),
        "session_initialized": None,
    }
    if not onnx_available:
        return _result(
            STATUS_WARNING,
            "parser_landmark_fallback_onnx_unavailable",
            "parser will use landmark-only masks because ONNX Runtime is unavailable",
            **fields,
        )
    if not model_available:
        return _result(
            STATUS_WARNING,
            "parser_landmark_fallback_model_unavailable",
            "parser will use landmark-only masks because the verified BiSeNet model is unavailable",
            **fields,
        )

    fields["backend"] = "bisenet_onnx"
    if not probe:
        return _result(
            STATUS_OK,
            "parser_backend_capable_not_probed",
            "verified BiSeNet model and ONNX Runtime are present; parser initialization was not probed",
            **fields,
        )
    parsing_module, parsing_error = _safe_import("retouch.parsing")
    parser_class = getattr(parsing_module, "FaceParser", None) if parsing_module else None
    if parsing_error or parser_class is None:
        fields["backend"] = "landmark_only"
        return _result(
            STATUS_WARNING,
            "parser_module_import_failed",
            "parser module could not be imported; landmark-only masks remain available",
            detail=parsing_error or "FaceParser class is missing",
            **fields,
        )
    try:
        parser = parser_class()
        session = getattr(parser, "_sess", None)
    except Exception as exc:
        fields["backend"] = "landmark_only"
        return _result(
            STATUS_WARNING,
            "parser_initialization_failed_fallback",
            "parser initialization failed; landmark-only masks remain available",
            detail=_safe_detail(exc),
            **fields,
        )
    fields["session_initialized"] = session is not None
    if session is None:
        fields["backend"] = "landmark_only"
        return _result(
            STATUS_WARNING,
            "parser_session_unavailable_fallback",
            "parser has no active ONNX session; landmark-only masks remain available",
            **fields,
        )
    return _result(
        STATUS_OK,
        "parser_initialized",
        "BiSeNet parser initialized successfully",
        **fields,
    )


def check_mode(detector: Mapping[str, Any], requested_mode: str = "auto") -> Dict[str, Any]:
    """Make the effective face-aware/global-only mode explicit."""
    requested = str(requested_mode or "auto").strip().lower()
    if requested not in {"auto", "face-aware", "global-only"}:
        return _result(
            STATUS_ERROR,
            "invalid_requested_mode",
            "requested runtime mode is invalid",
            requested=requested,
            effective="global-only",
            face_aware=False,
            global_only=True,
        )
    detector_available = bool(detector.get("available"))
    if requested == "global-only":
        return _result(
            STATUS_WARNING,
            "explicit_global_only_request",
            "global-only mode was explicitly requested",
            requested=requested,
            effective="global-only",
            face_aware=False,
            global_only=True,
            detector_available=detector_available,
        )
    if requested == "face-aware" and not detector_available:
        return _result(
            STATUS_ERROR,
            "face_aware_requested_but_unavailable",
            "face-aware mode was requested but the detector is unavailable",
            requested=requested,
            effective="global-only",
            face_aware=False,
            global_only=True,
            detector_available=False,
        )
    if detector_available:
        return _result(
            STATUS_OK,
            "face_aware_mode_available",
            "face-aware mode is available from the detector status",
            requested=requested,
            effective="face-aware",
            face_aware=True,
            global_only=False,
            detector_available=True,
        )
    return _result(
        STATUS_ERROR,
        "global_only_detector_unavailable",
        "global-only mode is required because the face-aware detector is unavailable",
        requested=requested,
        effective="global-only",
        face_aware=False,
        global_only=True,
        detector_available=False,
    )


def check_entrypoint_seam(root: Optional[os.PathLike[str] | str] = None) -> Dict[str, Any]:
    """Detect the package-initializer limitation around ``python -m``.

    Python imports a package's ``__init__.py`` before its submodule.  This
    check makes an eager engine import visible without editing that file in
    this bounded slice.
    """
    base = Path(root) if root is not None else _DEFAULT_ROOT
    init_path = base / "retouch" / "__init__.py"
    try:
        text = init_path.read_text(encoding="utf-8")
    except Exception as exc:
        return _result(
            STATUS_UNKNOWN,
            "package_initializer_unreadable",
            "package initializer could not be inspected",
            path=str(init_path),
            detail=_safe_detail(exc),
        )
    try:
        tree = ast.parse(text, filename=str(init_path))
    except SyntaxError as exc:
        return _result(
            STATUS_ERROR,
            "package_initializer_invalid",
            "package initializer could not be parsed",
            path=str(init_path),
            detail=_safe_detail(exc),
        )
    # Only imports in the module body are eager.  A lazy ``__getattr__`` may
    # legitimately import ``.engine`` inside its function body.
    eager = any(
        (
            isinstance(node, ast.ImportFrom)
            and node.level >= 1
            and node.module == "engine"
        )
        or (
            isinstance(node, ast.Import)
            and any(alias.name == "retouch.engine" for alias in node.names)
        )
        for node in tree.body
    )
    if eager:
        return _result(
            STATUS_WARNING,
            "package_initializer_eager_engine_import",
            "retouch.runtime_doctor may be blocked by retouch/__init__.py when native dependencies are missing",
            path=str(init_path),
            safe_module_import=True,
            required_follow_up="lazy-load retouch.engine exports in retouch/__init__.py",
        )
    return _result(
        STATUS_OK,
        "package_initializer_lazy_enough",
        "package initializer does not eagerly import retouch.engine",
        path=str(init_path),
        safe_module_import=True,
        required_follow_up=None,
    )


def collect_runtime_report(
    *,
    project_root: Optional[os.PathLike[str] | str] = None,
    pyproject_path: Optional[os.PathLike[str] | str] = None,
    manifest_path: Optional[os.PathLike[str] | str] = None,
    mode: str = "auto",
    probe_detector: bool = False,
    probe_parser: bool = False,
) -> Dict[str, Any]:
    """Collect a complete, JSON-serializable Runtime Doctor report."""
    root = Path(project_root) if project_root is not None else _DEFAULT_ROOT
    declarations_path = Path(pyproject_path) if pyproject_path is not None else root / "pyproject.toml"
    models_path = Path(manifest_path) if manifest_path is not None else root / "models" / "manifest.json"

    declarations = read_declared_project(declarations_path)
    interpreter = check_interpreter(declarations)
    packages = inspect_packages(declarations.get("dependencies", []))
    optional_packages = inspect_optional_packages(
        declarations.get("optional_dependencies", {})
    )
    pip_check = check_pip_check()
    opencv = check_opencv()
    model_report = inspect_models(models_path)
    model_items = model_report.get("items", {})
    onnx = check_onnx_providers()
    detector = check_detector(model_items, probe=probe_detector)
    parser = check_parser(model_items, onnx, probe=probe_parser)
    execution_mode = check_mode(detector, mode)
    entrypoint = check_entrypoint_seam(root)

    major = [
        interpreter,
        packages,
        optional_packages.get("gui", {}),
        pip_check,
        opencv,
        model_report,
        detector,
        parser,
        onnx,
        execution_mode,
    ]
    overall_status = _aggregate_status(major)
    if overall_status == STATUS_ERROR:
        overall_reason_code = "runtime_requires_recovery"
        overall_reason = "runtime doctor found one or more blocking compatibility or capability problems"
    elif overall_status in {STATUS_WARNING, STATUS_UNAVAILABLE}:
        overall_reason_code = "runtime_degraded"
        overall_reason = "runtime doctor completed with degraded or unavailable optional capability"
    elif overall_status == STATUS_UNKNOWN:
        overall_reason_code = "runtime_status_incomplete"
        overall_reason = "runtime doctor could not resolve every check"
    else:
        overall_reason_code = "runtime_healthy"
        overall_reason = "runtime doctor found no blocking problems"

    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": overall_status,
        "reason_code": overall_reason_code,
        "reason": overall_reason,
        "interpreter": interpreter,
        "declared": declarations,
        "packages": packages,
        "optional_packages": optional_packages,
        "pip_check": pip_check,
        "opencv": opencv,
        "models": model_report,
        "detector": detector,
        "parser": parser,
        "onnx": onnx,
        "mode": execution_mode,
        "face_aware": bool(execution_mode.get("face_aware")),
        "global_only": bool(execution_mode.get("global_only")),
        "entrypoint": entrypoint,
    }
    return _json_safe(report)


def _json_safe(value: Any) -> Any:
    """Normalize extension values so callers can always call json.dumps."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def run_doctor(**kwargs: Any) -> Dict[str, Any]:
    """Stable GUI/library alias for :func:`collect_runtime_report`."""
    return collect_runtime_report(**kwargs)


def runtime_doctor_report(**kwargs: Any) -> Dict[str, Any]:
    """Stable GUI/library alias for :func:`collect_runtime_report`."""
    return collect_runtime_report(**kwargs)


def face_mode(report: Mapping[str, Any]) -> str:
    """Return the effective public mode label from a doctor report."""
    mode = report.get("mode") if isinstance(report, Mapping) else None
    effective = mode.get("effective") if isinstance(mode, Mapping) else None
    if effective in {"face-aware", "global-only"}:
        return str(effective)
    return "global-only" if bool(report.get("global_only")) else "unknown"


def format_runtime_report(report: Mapping[str, Any]) -> str:
    """Format the high-value doctor facts for a GUI status panel or log."""
    if not isinstance(report, Mapping):
        return "Runtime Doctor: invalid report"
    interpreter = report.get("interpreter", {})
    detector = report.get("detector", {})
    parser = report.get("parser", {})
    onnx = report.get("onnx", {})
    mode = report.get("mode", {})
    pip_check = report.get("pip_check", {})
    optional_packages = report.get("optional_packages", {})
    gui = optional_packages.get("gui", {}) if isinstance(optional_packages, Mapping) else {}
    lines = [
        "Runtime Doctor: %s" % str(report.get("status", STATUS_UNKNOWN)),
        "Python: %s (%s)" % (
            str(interpreter.get("python_version", "unknown")),
            str(interpreter.get("status", STATUS_UNKNOWN)),
        ),
        "pip check: %s" % str(pip_check.get("status", STATUS_UNKNOWN)),
        "GUI: %s [%s]" % (
            "available" if gui.get("status") == STATUS_OK else "optional dependencies unavailable",
            str(gui.get("status", STATUS_UNKNOWN)),
        ),
        "Detector: %s [%s]" % (
            str(detector.get("backend", "unavailable")),
            str(detector.get("status", STATUS_UNKNOWN)),
        ),
        "Parser: %s [%s]" % (
            str(parser.get("backend", "unavailable")),
            str(parser.get("status", STATUS_UNKNOWN)),
        ),
        "ONNX providers: %s" % ", ".join(str(item) for item in onnx.get("available", [])),
        "Mode: %s" % str(mode.get("effective", face_mode(report))),
    ]
    return "\n".join(lines)


# Backwards-friendly aliases for callers that used the shorter names during
# development.  The explicit functions above keep a stable introspection and
# documentation surface for GUI code.
runtime_doctor = collect_runtime_report
doctor_report = collect_runtime_report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point; return non-zero only for an error-level report."""
    parser = argparse.ArgumentParser(description="Inspect Retouch runtime compatibility and capabilities.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--pyproject", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=("auto", "face-aware", "global-only"),
        default="auto",
        help="requested effective processing mode (default: auto)",
    )
    parser.add_argument(
        "--probe-detector",
        action="store_true",
        help="initialize FaceDetector when its static prerequisites are present",
    )
    parser.add_argument(
        "--probe-parser",
        action="store_true",
        help="initialize FaceParser when its static prerequisites are present",
    )
    parser.add_argument("--compact", action="store_true", help="emit one-line JSON")
    args = parser.parse_args(argv)
    report = collect_runtime_report(
        project_root=args.project_root,
        pyproject_path=args.pyproject,
        manifest_path=args.manifest,
        mode=args.mode,
        probe_detector=args.probe_detector,
        probe_parser=args.probe_parser,
    )
    print(json.dumps(report, sort_keys=True, indent=None if args.compact else 2))
    return 1 if report.get("status") == STATUS_ERROR else 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI tests
    raise SystemExit(main())

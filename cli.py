#!/usr/bin/env python3

import argparse
import atexit
import multiprocessing
import queue
import re
import signal
import shutil
import sys
import os
import time
import warnings
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retouch import RetouchEngine
from retouch.batch_progress import (
    BatchProgress,
    ProgressReporter,
    emit,
    image_megapixels,
    make_event_sink,
)
from retouch.engine import _adjust_contrast
from retouch.grading import PRESETS, ColorGrader
from retouch.io import (
    IMAGE_EXTENSIONS,
    RAW_EXTENSIONS,
    _resolve_safe_path,
    apply_raw_exposure_gain,
    color_context_for_path,
    imread_engine_with_context,
    imread_exif,
    make_comparison,
    output_format,
    read_exif_bytes,
    read_c2pa_manifest,
    resize_for_processing,
    raw_exposure_decode_info,
    write_image_with_color_context,
)
from retouch.recipes import CURATED_RECIPE_NAMES, RECIPES
from retouch.params import PROCESSING_PARAMS, recipe_to_params
from retouch.session import Session, create_session_from_params
from retouch.recipe_cookbook import list_recipes, search_recipes
from retouch.look_extractor import LookExtractor
from retouch.cli_input import (
    apply_resume_plan,
    attach_destinations,
    build_input_plan,
    decode_plan_inputs,
    inspect_plan_headers,
    load_input_list,
    print_input_plan,
    sha256_file,
    stable_fingerprint,
)
from retouch.review_page import (
    REVIEW_DIRNAME,
    ReviewRecord,
    build_review_page,
    record_key,
    result_review_meta,
    review_root_for,
    write_review_record,
)


class _DeprecatedAliasAction(argparse.Action):
    """Argparse action for a deprecated flag aliased to a newer flag.

    Stores the value into ``dest`` (a separate "legacy" attribute) and emits a
    ``DeprecationWarning`` pointing the user at the replacement option. The
    build_params() step is responsible for picking the new flag's value over
    the legacy one when both are set.
    """

    def __init__(self, option_strings, dest, deprecated_to, **kwargs):
        kwargs.setdefault("default", None)
        self.deprecated_to = deprecated_to
        super().__init__(option_strings=option_strings, dest=dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        warnings.warn(
            f"'{option_string}' is deprecated and will be removed in a future"
            f" release; please use '{self.deprecated_to}' instead.",
            DeprecationWarning,
            stacklevel=1,
        )
        setattr(namespace, self.dest, values)

RECIPE_CHOICES = list(CURATED_RECIPE_NAMES)
PRESET_CHOICES = RECIPE_CHOICES


def _finalize_params(params):
    """Load color reference once; safe to share read-only across batch jobs."""
    finalized = dict(params)
    ref_path = finalized.pop("color_ref_path", None)
    if ref_path is not None:
        finalized["color_ref"] = imread_exif(Path(ref_path))
    return finalized


def _smart_params_for_image(img_bgr, base_params):
    """F10 — run smart analysis on one image, return merged engine params.

    Uses :class:`retouch.smart_default.SmartProcessor` to pick a recipe +
    parameter overrides for ``img_bgr``, then layers any explicit CLI
    params from ``base_params`` on top (CLI flags win over smart
    suggestions). Returns a new params dict suitable for
    ``engine.process(**params)``.

    The smart suggestion emits GUI-scale values; we convert them to
    engine kwargs via ``gui_values_to_engine_kwargs``.
    """
    from retouch.smart_default import SmartProcessor
    from retouch.params import gui_values_to_engine_kwargs

    sp = SmartProcessor()
    suggestion = sp.analyze_and_suggest(img_bgr)
    smart_engine_kwargs = gui_values_to_engine_kwargs(suggestion.params)
    smart_engine_kwargs["recipe"] = suggestion.recipe

    # CLI explicit flags (base_params) override smart suggestions.
    merged = dict(smart_engine_kwargs)
    merged.update(base_params)
    # If the CLI explicitly set a recipe, it wins; otherwise keep smart's.
    if "recipe" in base_params:
        merged["recipe"] = base_params["recipe"]
    return merged, suggestion


def _resolve_session_path(path: str) -> Path:
    """Resolve a session file path with a path-traversal guard.

    Session files are user-supplied JSON; reject any path that resolves
    to a system directory (mirrors the guard used for image I/O).
    """
    return _resolve_safe_path(path, base_dir=None)


def _load_session_params(path: str) -> Dict[str, Any]:
    """Load a session JSON file and return its params dict + recipe.

    Returns a dict with two keys:
        ``recipe`` — the session's recipe name (or None)
        ``params`` — the session's params dict (engine-scale kwargs for
                     ``engine.process``), with ``color_ref_path`` stripped
                     (the file referenced at save time may not exist now)

    Raises ValueError on path-traversal rejection; raises FileNotFoundError
    if the session file does not exist; raises json.JSONDecodeError on
    malformed JSON (surfaced by Session.from_file).
    """
    resolved = _resolve_session_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Session file not found: {path}")
    session = Session.from_file(str(resolved))
    params = dict(session.params)
    # ``color_ref_path`` references an image at save time; drop it so
    # _finalize_params doesn't try to load a possibly-stale path. The
    # caller can re-supply --color-ref on the CLI.
    params.pop("color_ref_path", None)
    return {"recipe": session.recipe, "params": params}


def _save_session(
    params: Dict[str, Any],
    image_path: Path,
    out_path: Path,
    save_path: Optional[str],
) -> str:
    """Save the effective params as a session JSON next to the output image.

    If ``save_path`` is provided, write there (after path-traversal guard);
    otherwise auto-name it ``<output_stem>.session.json`` in the same
    directory as ``out_path``. Returns the absolute path written.
    """
    recipe = params.get("recipe")
    # Strip non-engine kwargs before saving so the session round-trips
    # cleanly through engine.process(**session.params).
    save_params = {
        k: v for k, v in params.items()
        if k not in ("color_ref", "color_ref_path")
    }
    if save_path:
        resolved = _resolve_safe_path(save_path, base_dir=None)
        target = str(resolved)
    else:
        target = str(out_path.with_suffix(".session.json"))
    session = create_session_from_params(
        save_params,
        recipe=recipe,
        image_path=str(image_path),
    )
    return session.to_file(target)


_worker_engine = None
_worker_progress_q = None


def _init_worker(progress_q=None, with_engine=True):
    global _worker_engine, _worker_progress_q
    _worker_progress_q = progress_q
    if progress_q is not None:
        progress_q.cancel_join_thread()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _terminate_worker)
    if with_engine:
        _worker_engine = RetouchEngine()
    # Each pool worker owns a RetouchEngine, which owns its own internal
    # FaceProcessorPool (a nested ProcessPoolExecutor, up to 4 more
    # processes). This atexit hook is a backstop for the single-face case
    # (inner pool never started) and for abrupt interpreter exits; it is
    # NOT sufficient by itself on multi-face images — see the inline
    # engine._face_pool.shutdown() in _process_single's finally block and
    # cli-batch-hangs-on-exit-after-done for why atexit alone cannot reach
    # RetouchEngine.close() once the inner pool's grandchildren are alive.
    atexit.register(_close_worker_engine)


def _terminate_worker(signum, _frame):
    for child in multiprocessing.active_children():
        child.terminate()
    os._exit(128 + signum)


_ATOMIC_TEMP_RE = re.compile(r"^\..+\.tmp-\d+-[0-9a-f]{8}\.[A-Za-z0-9]+$")


def _stop_pool(pool):
    procs = list((getattr(pool, "_processes", None) or {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for proc in procs:
        if proc.is_alive():
            proc.terminate()
    for proc in procs:
        proc.join(timeout=10)


def _remove_partial_outputs(output_dir):
    if output_dir is None or not Path(output_dir).is_dir():
        return
    for tmp in Path(output_dir).rglob(".*.tmp-*"):
        if _ATOMIC_TEMP_RE.match(tmp.name) and tmp.is_file():
            try:
                tmp.unlink()
            except OSError:
                pass


def _close_worker_engine():
    global _worker_engine
    if _worker_engine is not None:
        try:
            _worker_engine.close()
        except Exception:
            pass
        _worker_engine = None


def _review_relative(img_path, input_root):
    """Display path for a review record: relative to the recursive input
    root when batching recursively, else just the file name."""
    if input_root is not None:
        try:
            return str(Path(img_path).resolve().relative_to(Path(input_root).resolve()))
        except ValueError:
            pass
    return Path(img_path).name


def _safe_write_review_record(review_root, record):
    """Write a ReviewRecord, logging (never raising) on failure.

    Record writing must never take down a batch: a bad path, a full disk,
    or a permissions error here should surface as a warning, not a crash
    partway through an image the engine already finished processing.
    """
    if review_root is None:
        return
    try:
        write_review_record(review_root, record)
    except Exception as e:
        print(f"  ⚠ review record write failed for {record.source}: {e}")


def _maybe_write_skip_record(review_root, img_path, out_path, input_root, recipe):
    """On a skip (existing output, no --force), backfill a minimal record
    only if this image has no review record yet, so a re-run doesn't
    clobber a richer record from the run that actually processed it."""
    if review_root is None:
        return
    try:
        existing = Path(review_root) / REVIEW_DIRNAME / "records" / f"{record_key(img_path)}.json"
        if existing.exists():
            return
        write_review_record(review_root, ReviewRecord(
            source=str(Path(img_path).resolve()),
            output=str(out_path),
            status="done",
            recipe=recipe,
            relative=_review_relative(img_path, input_root),
            qa_recorded=False,
        ))
    except Exception as e:
        print(f"  ⚠ review record (skip) failed for {img_path}: {e}")


def _linear_raw_to_engine_bgr(
    path,
    exposure: float = 0.0,
    contrast: float = 1.0,
    *,
    apply_exposure_bias: bool = True,
    decode_info: Optional[Dict[str, Any]] = None,
):
    """T5 path: linear decode → develop → gamma-encode → float32 BGR [0,255]."""
    from retouch.raw_develop import RAWDeveloper

    dev = RAWDeveloper()
    linear_rgb, _meta = dev.load_raw(path)
    exposure_info = raw_exposure_decode_info(path, apply_exposure_bias)
    gain_ev = float(exposure_info["raw_exposure_gain_ev"] or 0.0)
    if decode_info is not None:
        decode_info.clear()
        decode_info.update(exposure_info)
    if gain_ev > 0.0:
        linear_rgb = apply_raw_exposure_gain(linear_rgb, gain_ev)
    linear_rgb = dev.develop(
        linear_rgb, exposure=exposure, contrast=contrast,
    )
    from retouch.white_balance import linear_to_srgb
    srgb = linear_to_srgb(np.clip(linear_rgb, 0.0, 1.0))
    bgr = (srgb[..., ::-1] * 255.0).astype(np.float32)
    return bgr


def _evidence_scalar(value):
    """Return one bounded JSON scalar, excluding non-finite/path-like text."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        value = value.strip()
        if (
            value
            and len(value) <= 128
            and "/" not in value
            and "\\" not in value
            and "\n" not in value
            and "\r" not in value
        ):
            return value
    return None


def _evidence_fields(value, allowed):
    if not isinstance(value, dict):
        return {}
    compact = {}
    for key in allowed:
        if key not in value:
            continue
        item = value[key]
        if isinstance(item, (list, tuple)):
            compact[key] = [
                normalized
                for normalized in (_evidence_scalar(child) for child in item[:32])
                if normalized is not None
            ]
        else:
            normalized = _evidence_scalar(item)
            if normalized is not None:
                compact[key] = normalized
    return compact


def _processing_evidence(result, *, global_only=False, color_context=None):
    """Capture a small, versioned per-image diagnostic envelope.

    Only selected scalar diagnostics cross the CLI/ProcessPool boundary. No
    pixels, masks, face crops, raw exception text, or arbitrary engine attrs
    are persisted in the input-plan ledger.
    """
    face_count = getattr(result, "face_count", None) if result is not None else None
    if isinstance(face_count, np.generic):
        face_count = face_count.item()
    if not isinstance(face_count, int) or isinstance(face_count, bool) or face_count < 0:
        face_count = None

    runtime = getattr(result, "runtime_diagnostics", {}) if result is not None else {}
    runtime = runtime if isinstance(runtime, dict) else {}
    detector_raw = runtime.get("face_detection", {})
    detector = _evidence_fields(
        detector_raw,
        ("mode", "available", "backend", "probe_state", "reason"),
    )
    detector_mode = detector.get("mode")
    if global_only:
        detector = {"mode": "not_run_global_only"}
        detector_mode = "global_only"
    runtime_summary = {}
    for name in ("denoise", "super_resolution", "body_reshape"):
        summary = _evidence_fields(
            runtime.get(name),
            (
                "backend", "providers", "provider_order", "fallback_chain",
                "available", "stage", "action", "status", "fallback_reason",
                "reason", "confidence", "scale",
            ),
        )
        if summary:
            runtime_summary[name] = summary
    healing = runtime.get("healing")
    if isinstance(healing, list):
        runtime_summary["healing"] = [
            _evidence_fields(
                item,
                (
                    "index", "status", "requested", "executed", "reason",
                    "error_type", "valid_source_centres", "permitted_donor_pixels",
                    "fallback_pixels", "source_map_available",
                ),
            )
            for item in healing[:32]
            if isinstance(item, dict)
        ]

    qa_evidence = getattr(result, "qa_evidence", {}) if result is not None else {}
    detectors = {}
    if isinstance(qa_evidence, dict):
        for name, item in sorted(qa_evidence.items(), key=lambda pair: str(pair[0]))[:32]:
            if not isinstance(name, str) or not isinstance(item, dict):
                continue
            summary = _evidence_fields(item, ("status", "score", "flagged", "available"))
            if summary:
                detectors[name[:64]] = summary
    if global_only:
        qa = {"status": "not_applicable_global_only", "detectors": {}}
    elif detectors:
        qa = {"status": "captured", "detectors": detectors}
    else:
        qa = {"status": "not_captured", "detectors": {}}

    safe_auto = []
    decisions = getattr(result, "safe_auto_decisions", []) if result is not None else []
    if isinstance(decisions, (list, tuple)):
        for decision in decisions[:32]:
            compact = _evidence_fields(
                decision,
                ("stage", "action", "confidence", "reason", "strength_scale"),
            )
            if compact:
                safe_auto.append(compact)

    fa02 = []
    fa02_values = getattr(result, "fa02_diagnostics", []) if result is not None else []
    if isinstance(fa02_values, (list, tuple)):
        for item in fa02_values[:32]:
            if not isinstance(item, dict):
                fa02.append({"status": "not_run"})
                continue
            fa02.append(
                _evidence_fields(
                    item,
                    ("face_width_px", "mode", "eligible", "reason", "ran"),
                )
            )

    timings = getattr(result, "timings", {}) if result is not None else {}
    timing_summary = {}
    if isinstance(timings, dict):
        timing_summary = {
            str(name)[:64]: number
            for name, value in list(timings.items())[:64]
            if (number := _evidence_scalar(value)) is not None
            and isinstance(number, (int, float))
        }
    precision = getattr(result, "precision_metadata", {}) if result is not None else {}
    precision_summary = _evidence_fields(
        precision,
        (
            "source_dtype", "source_bit_depth", "storage_dtype", "storage_bit_depth",
            "processed_precision", "processed_bit_depth", "precision_status",
            "downgraded", "downgrade_reason", "is_true_16bit", "supports_16bit_export",
        ),
    )
    color_summary = {}
    to_dict = getattr(color_context, "to_dict", None)
    if callable(to_dict):
        color_summary = _evidence_fields(
            to_dict(),
            (
                "schema", "working_space", "source_kind", "assumed_srgb",
                "is_tagged", "conversion_applied", "source_profile_name",
                "source_profile_sha256", "working_profile_sha256",
                "source_bit_depth", "working_bit_depth", "transform_intent",
                "black_point_compensation", "alpha_mode", "raw_exposure_bias_ev",
                "raw_exposure_gain_ev",
            ),
        )

    return {
        "schema": "retouch.cli_processing_evidence",
        "version": 1,
        "capture_status": "captured",
        "execution_mode": "global_only" if global_only else "engine",
        "face_detection": detector or {"mode": "unknown"},
        "face_count": face_count,
        "face_aware_execution": (
            False
            if global_only
            else (
                detector_mode == "face_aware" and face_count > 0
                if face_count is not None and detector_mode in ("face_aware", "global_only")
                else None
            )
        ),
        "qa": qa,
        "runtime_diagnostics": runtime_summary,
        "safe_auto_decisions": safe_auto,
        "fa02_diagnostics": fa02,
        "timings_ms": timing_summary,
        "precision": precision_summary,
        "color_context": color_summary,
    }


def _progress_key(img_path, input_root=None):
    """Stable per-image progress key, relative to a recursive input root."""
    path = Path(img_path)
    if input_root is not None:
        try:
            return path.resolve().relative_to(Path(input_root)).as_posix()
        except ValueError:
            pass
    return path.name


def _result_info(result, info):
    """Copy small picklable result details before ndarray conversions drop them."""
    faces = getattr(result, "face_count", None)
    if faces is not None:
        info["faces"] = int(faces)
    timings = getattr(result, "timings", None)
    if timings:
        info["timings"] = {
            str(key): round(float(value), 1)
            for key, value in dict(timings).items()
            if isinstance(value, (int, float))
        }
    qa = getattr(result, "qa", None) or []
    info["qa"] = [
        {
            "detector": warning.detector,
            "score": round(float(warning.score), 4),
            "threshold": None if warning.threshold is None else float(warning.threshold),
            "flagged": bool(warning.flagged),
            "message": warning.message,
        }
        for warning in qa
    ]


def _process_single(args):
    """Pool entry point: process one image and stream stage progress."""
    # Keep compatibility with the 25-item worker tuple used by callers/tests
    # predating branch-specific destination stems and RAF exposure metadata.
    if len(args) == 25:
        args = (*args[:23], None, True, *args[23:])
    supplied_key = args[27] if len(args) > 27 else None
    if len(args) > 27:
        args = args[:27]
    img_path, input_root = args[0], args[22]
    key = supplied_key or _progress_key(img_path, input_root)
    q = _worker_progress_q
    sink = make_event_sink(q, key) if q is not None else None
    if q is not None:
        emit(q, key, "start", worker=os.getpid())
    info: Dict[str, Any] = {}
    t_start = time.time()
    name, status, evidence = _process_single_with_evidence(args, info, sink)
    info["seconds"] = round(time.time() - t_start, 3)
    if evidence is not None:
        info["_processing_evidence"] = evidence
    return Path(name).name, status, info


def _process_single_with_evidence(args, info=None, sink=None):
    if info is None:
        info = {}
    stage = sink if sink is not None else (lambda *_a, **_k: None)
    (img_path, output_dir, params, format_arg, quality, force, copy_exif_flag,
     max_dim, compare_flag, global_only, bit_depth, fail_on_qa, save_session,
     smart, linear_raw, raw_exposure, raw_contrast, raf_decoder,
     raf2jpeg_path, raf2jpeg_quality, fuji_match_strength, optical_correction,
     input_root, destination_stem, raf_exposure_bias, review_root, review) = args
    processing_evidence = None
    t_start = time.time()
    try:
        fmt = output_format(img_path, format_arg)
        # For 16-bit, force PNG or TIFF
        if bit_depth == 16 and fmt not in ("png", "tif", "tiff"):
            fmt = "png"

        out_path = _destination_for_image(
            img_path,
            output_dir,
            fmt,
            input_root=input_root,
            output_stem=destination_stem,
        )
        _assert_safe_destination(img_path, out_path)

        if os.path.lexists(os.fspath(out_path)) and not force:
            if review:
                _maybe_write_skip_record(
                    review_root, img_path, out_path, input_root, params.get("recipe"),
                )
            return (str(img_path), "skipped", None)

        stage("stage", {"stage": "decode"})
        if linear_raw and Path(img_path).suffix.lower() in RAW_EXTENSIONS:
            decode_info = {}
            img_bgr = _linear_raw_to_engine_bgr(
                img_path,
                exposure=raw_exposure,
                contrast=raw_contrast,
                apply_exposure_bias=raf_exposure_bias,
                decode_info=decode_info,
            )
            color_context = color_context_for_path(
                img_path,
                apply_exposure_bias=raf_exposure_bias,
                raw_exposure_info=decode_info,
            )
        else:
            correction_status = {}
            img_bgr, color_context = imread_engine_with_context(
                img_path,
                raw_decoder=raf_decoder,
                raf2jpeg_path=raf2jpeg_path,
                raf2jpeg_quality=raf2jpeg_quality,
                fuji_match_strength=fuji_match_strength,
                optical_correction=optical_correction,
                correction_status=correction_status,
                apply_exposure_bias=raf_exposure_bias,
            )
            if optical_correction:
                print(f"  ℹ {img_path.name}: Lensfun {correction_status.get('reason') or correction_status.get('applied', ())}")
        orig_shape = img_bgr.shape[:2]
        # 16-bit RAF ingest returns float32 [0,255]; comparison stitching is
        # uint8-only, so snapshot a uint8 original for the compare image.
        original_full = (
            (np.clip(img_bgr, 0, 255).astype(np.uint8) if img_bgr.dtype != np.uint8 else img_bgr.copy())
            if compare_flag else None
        )
        img_bgr, _scale = resize_for_processing(img_bgr, max_dim)

        # F10: --smart — per-image analysis overrides recipe/params.
        # CLI explicit flags (in `params`) win over the smart suggestion.
        effective_params = params
        if smart:
            try:
                effective_params, suggestion = _smart_params_for_image(
                    img_bgr, params
                )
            except (ValueError, RuntimeError) as e:
                # Analysis failed — fall back to the base params and log.
                tqdm.write(
                    f"  ⚠ {img_path.name}: smart analysis failed ({e}), "
                    f"using base params"
                )

        review_meta = {"faces": [], "qa": []}
        if global_only:
            stage("stage", {"stage": "global_finish"})
            result = _apply_global_finish(img_bgr, dict(effective_params))
            _result_info(result, info)
            processing_evidence = _processing_evidence(
                result, global_only=True, color_context=color_context
            )
        else:
            global _worker_engine
            if _worker_engine is not None:
                engine = _worker_engine
                should_close = False
            else:
                engine = RetouchEngine()
                should_close = True

            try:
                result = engine.process(
                    img_bgr, progress_cb=sink, **dict(effective_params)
                )
                _result_info(result, info)
                processing_evidence = _processing_evidence(
                    result, color_context=color_context
                )
                if review:
                    review_meta = result_review_meta(result, _scale)
                if fail_on_qa:
                    qa = getattr(result, 'qa', [])
                    if qa:
                        flagged = [w for w in qa if w.flagged]
                        if flagged:
                            reasons = "; ".join(f"{w.detector}={w.score:.2f}" for w in flagged)
                            if review:
                                _safe_write_review_record(review_root, ReviewRecord(
                                    source=str(Path(img_path).resolve()),
                                    status="qa_fail",
                                    recipe=params.get("recipe"),
                                    faces=review_meta.get("faces", []),
                                    qa=review_meta.get("qa", []),
                                    relative=_review_relative(img_path, input_root),
                                    elapsed_s=time.time() - t_start,
                                ))
                            return (str(img_path), f"QA_FAIL: {reasons}", processing_evidence)
            finally:
                if should_close:
                    engine.close()
                else:
                    # Shut down the reused engine's inner FaceProcessorPool
                    # (its own nested ProcessPoolExecutor, started lazily on
                    # multi-face images) after every task rather than
                    # deferring to atexit. Under --workers>1, this worker
                    # process is non-daemon and so are the pool's own
                    # grandchild processes; multiprocessing's own exit
                    # handler joins non-daemon children before running our
                    # atexit-registered _close_worker_engine, so as long as
                    # any grandchildren are alive, this worker never reaches
                    # interpreter shutdown and _close_worker_engine's
                    # RetouchEngine.close() call is never reached — the
                    # ProcessPoolExecutor's own final shutdown() then blocks
                    # forever joining this worker. Shutting the inner pool
                    # down here (a no-op if it was never started) breaks
                    # that cycle. See cli-batch-hangs-on-exit-after-done.
                    engine._face_pool.shutdown()

        # Upscale back to original dimensions
        if _scale < 1.0:
            result = cv2.resize(result, (orig_shape[1], orig_shape[0]),
                                interpolation=cv2.INTER_LINEAR)

        # Embed the context-selected ICC at write time so the working profile,
        # bit depth, quality, and metadata survive the delivery boundary.
        stage("stage", {"stage": "write"})
        exif_bytes = read_exif_bytes(img_path) if copy_exif_flag else None
        c2pa_manifest = read_c2pa_manifest(img_path) if copy_exif_flag else None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _assert_safe_destination(img_path, out_path)
        write_image_with_color_context(
            str(out_path),
            result,
            color_context,
            bit_depth=bit_depth,
            quality=quality,
            exif=exif_bytes,
            c2pa_manifest=c2pa_manifest,
        )
        info["out_path"] = str(out_path)

        if compare_flag:
            stage("stage", {"stage": "compare"})
            compare_path = out_path.with_name(f"{out_path.stem}_compare{out_path.suffix}")
            _assert_safe_destination(img_path, compare_path)
            make_comparison(original_full, result, compare_path, fmt, quality)
            info["compare_path"] = str(compare_path)

        if save_session is not None:
            save_path = None if save_session is True else save_session
            try:
                _save_session(effective_params, img_path, out_path, save_path)
            except (OSError, ValueError) as e:
                return (
                    str(img_path),
                    f"saved_image_but_session_failed: {e}",
                    processing_evidence,
                )

        if review:
            _safe_write_review_record(review_root, ReviewRecord(
                source=str(Path(img_path).resolve()),
                output=str(out_path),
                compare=str(compare_path) if compare_flag else None,
                status="done",
                recipe=params.get("recipe"),
                faces=review_meta.get("faces", []),
                qa=review_meta.get("qa", []),
                relative=_review_relative(img_path, input_root),
                elapsed_s=time.time() - t_start,
            ))

        return (str(img_path), "done", processing_evidence)
    except Exception as e:
        from retouch.utils import log_crash
        log_crash(e, {
            "image_path": str(img_path),
            "output_dir": str(output_dir) if output_dir else None,
            "params": str(params),
            "format_arg": format_arg,
            "quality": quality,
            "max_dim": max_dim,
            "compare_flag": compare_flag,
            "global_only": global_only,
            "bit_depth": bit_depth,
        })
        if review:
            _safe_write_review_record(review_root, ReviewRecord(
                source=str(Path(img_path).resolve()),
                status="failed",
                recipe=params.get("recipe") if isinstance(params, dict) else None,
                error=str(e),
                relative=_review_relative(img_path, input_root),
                elapsed_s=time.time() - t_start,
            ))
        return (str(img_path), f"failed: {e}", processing_evidence)


def _recipe_defaults(recipe_name):
    """Return the recipe-derived defaults for the global-finish path.

    Delegates to ``retouch.params.recipe_to_params`` (the canonical spec
    shared with the engine and the GUI) and converts the GUI-scale
    values it returns into the engine-scale values that
    ``_apply_global_finish`` consumes.  ``grade_intensity`` is stored
    0-100 in the GUI scale but ``grader.grade`` expects 0.0-1.0, and
    ``color_grade`` uses ``None`` (not ``"none"``) to mean "no grade".
    """
    d = recipe_to_params(recipe_name or "natural")
    if d.get("grade_intensity") is not None:
        d["grade_intensity"] = float(d["grade_intensity"]) / 100.0
    if d.get("color_grade") == "none":
        d["color_grade"] = None
    return d


def _apply_global_finish(img_bgr, params):
    """Fast retouch path that avoids face detection and local facial edits."""
    result = img_bgr.copy()
    defaults = _recipe_defaults(params.get("recipe"))
    grader = ColorGrader()

    if params.get("contrast"):
        result = _adjust_contrast(result, params["contrast"])

    color_grade = params.get("color_grade", defaults["color_grade"])
    grade_intensity = params.get(
        "grade_intensity",
        1.0 if params.get("color_grade") else defaults["grade_intensity"],
    )

    settings = PRESETS.get(color_grade, PRESETS["natural"]).copy() if color_grade else {}
    if params.get("chromatic_aberration") is not None:
        settings["chromatic_aberration"] = params["chromatic_aberration"]
    if params.get("halation") is not None:
        settings["halation"] = params["halation"]
    if params.get("grain") is not None:
        settings["grain"] = params["grain"]
    if params.get("lut") is not None:
        settings["lut"] = params["lut"]
    if params.get("gamut_compress") is not None:
        settings["gamut_compress"] = params["gamut_compress"]
    if params.get("saturation_mode") is not None:
        settings["saturation_mode"] = params["saturation_mode"]

    if settings:
        result = grader.grade(result, settings, grade_intensity)

    impact = params.get("impact", defaults["impact"])
    if impact > 0:
        result = grader.add_impact_finish(result, impact)

    return result


def find_images(input_path: str, recursive: bool) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]
    pattern = "**/*" if recursive else "*"
    files = []
    for f in path.glob(pattern):
        if f.name.startswith("."):
            continue
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(f)
    return sorted(files)


def _destination_for_image(
    image_path: Path,
    output_dir: Optional[Path],
    fmt: str,
    *,
    input_root: Optional[Path] = None,
    output_stem: Optional[str] = None,
) -> Path:
    """Build one deterministic destination without flattening recursive input."""
    stem = output_stem or image_path.stem
    if output_dir is None:
        return image_path.with_name(f"{stem}.{fmt}")
    if input_root is not None:
        try:
            relative = image_path.resolve().relative_to(Path(input_root).resolve())
        except ValueError as exc:
            raise ValueError(
                f"Input {image_path} is outside recursive input root {input_root}"
            ) from exc
    else:
        relative = Path(image_path.name)
    return output_dir / relative.parent / f"{output_stem or relative.stem}.{fmt}"


def _path_key(path: Path) -> str:
    """Return a collision-safe key, including macOS case-insensitive paths."""
    resolved = str(Path(path).expanduser().resolve(strict=False))
    normalized = os.path.normcase(resolved)
    # ``posixpath.normcase`` is a no-op even on the default case-insensitive
    # macOS filesystem, so fold there explicitly. Linux remains case-sensitive.
    return normalized.casefold() if sys.platform == "darwin" else normalized


def _assert_safe_destination(source: Path, destination: Path) -> None:
    # Path.exists() follows a symlink and returns False for a dangling one.
    # Writers may then follow that link and create an external target, so
    # output leaf symlinks are never valid CLI destinations.
    if os.path.islink(os.fspath(destination)):
        raise ValueError(
            f"Refusing to write through symlink destination {destination}"
        )
    if _path_key(source) == _path_key(destination):
        raise ValueError(
            f"Refusing to overwrite source image {source} with its own output"
        )
    if os.path.lexists(os.fspath(source)) and os.path.lexists(os.fspath(destination)):
        try:
            aliases_source = os.path.samefile(str(source), str(destination))
        except OSError:
            aliases_source = False
        if aliases_source:
            raise ValueError(
                f"Refusing to overwrite source image {source} through filesystem alias {destination}"
            )


def _preflight_destinations(
    files: list[Path],
    output_dir: Optional[Path],
    format_arg: str,
    bit_depth: int,
    *,
    recursive_root: Optional[Path],
    compare: bool,
    save_session: Any,
    output_stems: Optional[Dict[str, str]] = None,
) -> None:
    """Reject source overwrites and any duplicate artifact before processing."""
    if output_dir is not None and recursive_root is not None:
        output_resolved = Path(output_dir).expanduser().resolve(strict=False)
        input_resolved = Path(recursive_root).expanduser().resolve(strict=False)
        try:
            output_resolved.relative_to(input_resolved)
        except ValueError:
            pass
        else:
            raise ValueError(
                f"Recursive output directory {output_resolved} must be outside input tree {input_resolved}"
            )
    source_keys = {_path_key(path): path for path in files}
    destinations: Dict[str, Path] = {}
    for image_path in files:
        fmt = output_format(image_path, format_arg)
        if bit_depth == 16 and fmt not in ("png", "tif", "tiff"):
            fmt = "png"
        output_path = _destination_for_image(
            image_path,
            output_dir,
            fmt,
            input_root=recursive_root,
            output_stem=(output_stems or {}).get(_path_key(image_path)),
        )
        artifacts = [output_path]
        if compare:
            artifacts.append(
                output_path.with_name(
                    f"{output_path.stem}_compare{output_path.suffix}"
                )
            )
        if save_session is True:
            artifacts.append(output_path.with_suffix(".session.json"))
        elif save_session not in (None, False):
            artifacts.append(_resolve_safe_path(str(save_session), base_dir=None))

        for destination in artifacts:
            _assert_safe_destination(image_path, destination)
            if os.path.lexists(os.fspath(destination)):
                for source_path in files:
                    try:
                        aliases_input = os.path.samefile(
                            str(source_path), str(destination)
                        )
                    except OSError:
                        aliases_input = False
                    if aliases_input:
                        raise ValueError(
                            f"Destination {destination} aliases input {source_path}"
                        )
            key = _path_key(destination)
            if key in source_keys:
                raise ValueError(
                    f"Destination {destination} would overwrite input "
                    f"{source_keys[key]}"
                )
            previous = destinations.get(key)
            if previous is not None:
                raise ValueError(
                    f"Duplicate output destination {destination} for "
                    f"multiple input images (already claimed by {previous})"
                )
            destinations[key] = image_path


_ESTIMATED_OUTPUT_BYTES_PER_INPUT_BYTE = 2.0
_MIN_FREE_BYTES_AFTER_RUN = 5 * 1024 * 1024 * 1024


def _export_social_crops(files, output_dir, args, recursive_root, formats) -> None:
    """Post-batch: face-aware social crops of every written output."""
    from retouch.social_crops import export_folder, format_summary

    outputs = []
    for f in files:
        fmt = output_format(f, args.format)
        if args.bit_depth == 16 and fmt not in ("png", "tif", "tiff"):
            fmt = "png"
        out_path = _destination_for_image(f, output_dir, fmt, input_root=recursive_root)
        if out_path.exists():
            outputs.append(out_path)
    if not outputs:
        print("Social crops: no retouched outputs to crop")
        return
    social_dir = (output_dir or outputs[0].parent) / "social"
    summary = export_folder(
        outputs, social_dir, formats, size=args.social_size, force=args.force,
    )
    for r in summary["results"]:
        if r.status == "failed":
            print(f"  ✖ {r.path.name}: {r.error}")
    print(format_summary(summary, social_dir))


def _nearest_existing_ancestor(path: Path) -> Path:
    """Find a real directory for disk-usage checks before output creation."""
    probe = Path(path)
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    return probe


def _check_disk_space(
    files: list[Path],
    output_dir: Optional[Path],
    compare: bool,
    *,
    enforce: bool = True,
) -> list[str]:
    """Estimate destination capacity without confusing compressed size for RAM.

    The estimate is intentionally conservative and warn-oriented. Explicit
    output paths use their nearest existing ancestor; in-place exports are
    grouped by device because each source directory is a destination volume.
    """
    try:
        if output_dir is not None:
            probe = _nearest_existing_ancestor(output_dir)
            volume_checks = [
                (output_dir, probe, sum(path.stat().st_size for path in files), len(files))
            ]
        else:
            grouped: dict[int, tuple[Path, Path, int, int]] = {}
            for image_path in files:
                target = image_path.parent
                probe = _nearest_existing_ancestor(target)
                device = probe.stat().st_dev
                prior = grouped.get(device)
                size = image_path.stat().st_size
                if prior is None:
                    grouped[device] = (target, probe, size, 1)
                else:
                    first_target, first_probe, total, count = prior
                    grouped[device] = (
                        first_target,
                        first_probe,
                        total + size,
                        count + 1,
                    )
            volume_checks = list(grouped.values())
    except OSError:
        return []

    warnings_found: list[str] = []
    for target, probe, total_input, count in volume_checks:
        try:
            free_bytes = shutil.disk_usage(probe).free
        except OSError:
            continue
        estimated = total_input * _ESTIMATED_OUTPUT_BYTES_PER_INPUT_BYTE
        if compare:
            estimated *= 2
        projected = free_bytes - estimated
        if projected >= _MIN_FREE_BYTES_AFTER_RUN:
            continue
        message = (
            f"{count} input file(s), estimated output ~{estimated / (1024 ** 3):.1f} GiB; "
            f"{free_bytes / (1024 ** 3):.1f} GiB free on {target}'s volume, "
            f"projected remainder ~{max(projected, 0) / (1024 ** 3):.1f} GiB "
            f"(want at least {_MIN_FREE_BYTES_AFTER_RUN / (1024 ** 3):.1f} GiB)"
        )
        warnings_found.append(message)
        if enforce:
            print(
                f"⚠ Disk space warning: {message}. Free space, choose another "
                "volume, or pass --skip-disk-check to proceed."
            )
            sys.exit(1)
    return warnings_found


def _add_processing_arg(parser, spec):
    """Add an argparse argument for a single ``PROCESSING_PARAMS`` entry.

    Booleans get both ``--{flag}`` and ``--no-{flag}`` variants so the
    user can explicitly enable or disable a toggle.  Non-boolean args
    use ``type=cli_type, default=None``; argparse converts hyphens in
    the flag to underscores in the dest, which matches ``spec.name``.
    """
    if spec.cli_flag is None:
        return
    if spec.cli_type is bool:
        parser.add_argument(
            f"--{spec.cli_flag}",
            action="store_true",
            default=None,
            help=f"Enable {spec.name}",
        )
        parser.add_argument(
            f"--no-{spec.cli_flag}",
            action="store_false",
            dest=spec.name,
            default=None,
            help=f"Disable {spec.name}",
        )
        return
    parser.add_argument(
        f"--{spec.cli_flag}",
        type=spec.cli_type,
        choices=spec.choices,
        default=None,
        help=f"{spec.name} parameter",
    )


def build_params(args: argparse.Namespace) -> dict:
    params = {}
    if getattr(args, "recipe", None):
        params["recipe"] = args.recipe
    elif getattr(args, "preset", None):
        params["recipe"] = args.preset

    if getattr(args, "face_params", None):
        if args.face_params.lower() == "auto":
            params["face_params"] = "auto"
        else:
            from retouch.face_params import load_face_params_json
            params["face_params"] = load_face_params_json(args.face_params)

    # Process the simple scalar/int/float parameters from the spec list.
    # Each spec maps a CLI flag (e.g. "--smooth") to the engine kwarg name
    # ("smooth").  When a flag is not provided on the command line we leave
    # the engine kwarg unset (None) so the recipe default takes over.
    for spec in PROCESSING_PARAMS:
        if spec.cli_flag is None:
            continue
        # Translate the CLI flag to the argparse attribute name.
        attr = spec.cli_flag.replace("-", "_")
        if not hasattr(args, attr):
            continue
        val = getattr(args, attr)
        if val is None:
            continue
        # auto_exposure uses ``store_true`` semantics: only ``True`` is
        # forwarded (the explicit ``--no-auto-exposure`` sets ``False``,
        # which is the default and should be dropped).
        if spec.name == "auto_exposure":
            if val:
                params[spec.name] = True
            continue
        # For other boolean flags (nose_blush, under_eye_blush,
        # white_costume_lift) the caller can explicitly set ``False`` to
        # override the recipe.  Forward both ``True`` and ``False``.
        if spec.cli_type is bool:
            params[spec.name] = bool(val)
            continue
        params[spec.name] = val

    if args.color_grade:
        params["color_grade"] = args.color_grade
    if args.lip_tint:
        params["lip_tint"] = args.lip_tint
    if getattr(args, "whiten_tone", None):
        params["whiten_tone"] = args.whiten_tone
    if getattr(args, "specular_bloom_tone", None):
        params["specular_bloom_tone"] = args.specular_bloom_tone
    if getattr(args, "lip_finish", None):
        params["lip_finish"] = args.lip_finish
    if getattr(args, "auto_exposure", False):
        params["auto_exposure"] = True

    if getattr(args, "chromatic_aberration", None) is not None:
        params["chromatic_aberration"] = args.chromatic_aberration
    if getattr(args, "grain", None) is not None:
        params["grain"] = args.grain
    if getattr(args, "lut", None) is not None:
        params["lut"] = args.lut

    halation_intensity = getattr(args, "halation_intensity", None)
    if halation_intensity is None:
        halation_intensity = getattr(args, "halation", None)
    if halation_intensity is not None:
        threshold = getattr(args, "halation_threshold", None)
        radius = getattr(args, "halation_radius", None)
        params["halation"] = {
            "intensity": float(halation_intensity),
            "threshold": int(threshold) if threshold is not None else 210,
            "radius": int(radius) if radius is not None else 21,
        }

    if args.color_ref:
        params["color_ref_path"] = args.color_ref

    color_transfer_intensity = getattr(args, "color_transfer_intensity", None)
    if color_transfer_intensity is None:
        color_transfer_intensity = getattr(args, "color_ref_strength", None)
    if color_transfer_intensity is not None:
        params["color_transfer_intensity"] = float(color_transfer_intensity)

    return params


def main() -> None:
    from retouch.diagnostics import enable_native_crash_log

    enable_native_crash_log()
    warnings.filterwarnings(
        "always",
        category=DeprecationWarning,
        module=r"^(cli|__main__)(\.|$)",
    )

    parser = argparse.ArgumentParser(
        description="Professional batch face retouching tool"
    )
    parser.add_argument(
        "input",
        nargs="*",
        help="One or more image files/directories (quote paths containing spaces)",
    )
    parser.add_argument(
        "--input-list",
        type=str,
        default=None,
        metavar="PATH.json",
        help="Load literal input paths from a versioned JSON list",
    )
    parser.add_argument(
        "--input-plan",
        type=str,
        default=None,
        metavar="PATH.json",
        help="Write the input selection/preflight plan and update per-file results",
    )
    parser.add_argument(
        "--resume-plan",
        type=str,
        default=None,
        metavar="PATH.json",
        help="Reuse only rows whose prior source/output hashes and settings match",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Include discovered paths matching PATTERN (repeatable)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Exclude discovered paths matching PATTERN (repeatable)",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden files/directories during folder discovery",
    )
    parser.add_argument(
        "--raw-jpeg-policy",
        choices=["error", "raw-only", "jpeg-only", "suffix"],
        default="error",
        help="Handle same-basename RAW+JPEG pairs (default: error)",
    )
    parser.add_argument(
        "--input-check",
        choices=["paths", "headers", "decode"],
        default="paths",
        help="Input validation level: paths, container headers, or full decode",
    )
    parser.add_argument(
        "--max-input-pixels",
        type=int,
        default=None,
        metavar="N",
        help="Reject header-checked images larger than N pixels (opt-in safety cap)",
    )
    parser.add_argument(
        "--multi-frame-policy",
        choices=["error", "first"],
        default="error",
        help="Header-check policy for multi-frame images (default: error)",
    )
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("-q", "--quality", type=int, default=95,
                        help="Output quality 1-100 (default: 95)")
    parser.add_argument("--format", choices=["jpg", "png", "webp", "same"],
                        default="same", help="Output format (default: same as input)")
    parser.add_argument("--bit-depth", type=int, choices=[8, 16], default=8,
                        help="Output bit depth (default: 8). 16-bit requires PNG or TIFF format.")
    parser.add_argument("--max-dim", type=int, default=None,
                        help="Downscale so longest side ≤ N px before processing (faster)")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="Search subdirectories")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Overwrite existing files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without processing")
    parser.add_argument("--global-only", action="store_true",
                        help="Skip face detection and apply only global color/impact retouch")
    parser.add_argument("--optical-correction", action="store_true",
                        help="Apply verified Lensfun distortion/TCA/vignetting from EXIF; reports unavailable, unmatched, or precision-preserving skips")
    parser.add_argument("--preflight-check", action="store_true",
                        help="Run system pre-flight integrity checks (color "
                             "science, QA detectors, acceleration layer) and "
                             "exit; does not process images")

    # Processing controls
    parser.add_argument("--recipe", choices=RECIPE_CHOICES,
                        help="Retouch recipe name")
    parser.add_argument("--preset", choices=PRESET_CHOICES,
                        help="Quick parameter preset (legacy alias)")
    parser.add_argument("--color-ref", type=str, default=None,
                        help="Reference image path for colour transfer")
    parser.add_argument("--color-ref-strength", type=float, default=None,
                        action=_DeprecatedAliasAction,
                        deprecated_to="--color-transfer-intensity",
                        dest="color_ref_strength",
                        help="[DEPRECATED, use --color-transfer-intensity] "
                             "Colour transfer blend intensity 0-1")
    parser.add_argument("--halation-intensity", type=float, default=None,
                        help="Film halation bleed intensity (0.0 - 1.0)")
    parser.add_argument("--halation-threshold", type=int, default=None,
                        help="Halation highlight threshold 0-255 (default: 210)")
    parser.add_argument("--halation-radius", type=int, default=None,
                        help="Halation blur radius in pixels (default: 21)")
    parser.add_argument("--halation", type=float, default=None,
                        action=_DeprecatedAliasAction,
                        deprecated_to="--halation-intensity",
                        dest="halation",
                        help="[DEPRECATED, use --halation-intensity] "
                             "Film halation bleed intensity (0.0 - 1.0)")

    # Auto-generate processing parameter arguments from PROCESSING_PARAMS
    for spec in PROCESSING_PARAMS:
        _add_processing_arg(parser, spec)

    # Batch
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() // 2),
                        help="Parallel workers (default: CPU count / 2)")
    parser.add_argument(
        "--ram-budget-gib",
        type=float,
        default=None,
        metavar="GIB",
        help="Cap workers from the input-plan working-memory estimate (opt-in)",
    )
    parser.add_argument(
        "--skip-disk-check",
        action="store_true",
        help="Skip the destination-volume free-space estimate",
    )
    parser.add_argument(
        "--progress-file", type=str, default=None,
        help="Write live batch progress JSON (default: <output>/.retouch-progress.json)",
    )
    parser.add_argument(
        "--no-progress-file", action="store_true",
        help="Disable the live progress JSON file",
    )
    parser.add_argument("--no-compare", action="store_false", dest="compare",
                        help="Skip side-by-side comparison output")
    parser.add_argument("--no-exif", action="store_true",
                        help="Skip EXIF metadata copying")
    parser.add_argument("--fail-on-qa", action="store_true",
                        help="Exit with code 1 if any QA detector flags an artifact")
    parser.add_argument("--social-crops", nargs="?", const="4:5,9:16,1:1", default=None,
                        metavar="FORMATS",
                        help="After the batch, export face-aware crops for posting into "
                             "<output>/social/ (4:5, 9:16, 1:1, 3:4 or 'all'; "
                             "default with no value: 4:5,9:16,1:1)")
    parser.add_argument("--social-size", choices=["platform", "full"], default="platform",
                        help="Social crop size: platform = 1080 px wide (default), "
                             "full = native crop resolution")
    parser.add_argument("--no-review", action="store_false", dest="review", default=True,
                        help="Skip writing review.html (per-batch review page)")

    # Session save/load (F2)
    parser.add_argument("--session", type=str, default=None,
                        metavar="SESSION.json",
                        help="Load params from a session JSON file. Session "
                             "params override --recipe but are overridden by "
                             "explicit CLI flags.")
    parser.add_argument("--save-session", nargs="?", const=True, default=None,
                        metavar="PATH",
                        help="After processing, save the effective params to a "
                             "session JSON file. With no value, saves to "
                             "<output_stem>.session.json next to each output "
                             "image. Path traversal is guarded.")

    # F10: Smart Default — per-image auto-analysis + param suggestion.
    parser.add_argument("--smart", action="store_true",
                        help="Analyze each image with F9 (ImageAnalyzer) and "
                             "auto-select the recipe + params. Overrides "
                             "--recipe per-image; explicit CLI flags still win.")

    # T4: Recipe cookbook browsing (no processing).
    parser.add_argument("--list-recipes", nargs="?", const="", default=None,
                        metavar="CATEGORY",
                        help="List all recipes (optionally filtered by CATEGORY) and exit")
    parser.add_argument("--search-recipes", type=str, default=None,
                        metavar="QUERY",
                        help="Search recipes by name/description and exit")
    parser.add_argument("--export-lut", nargs="?", const="", default=None,
                        metavar="PATH",
                        help="Save --recipe's colour look as a .cube 3D LUT "
                             "(for Photoshop, Premiere, Resolve) and exit. "
                             "PATH is a .cube file or folder; default is "
                             "<recipe>.cube in -o or the current folder. "
                             "Only per-pixel colour/tone steps are baked in.")
    parser.add_argument("--lut-size", type=int, default=33,
                        help="Points per axis for --export-lut (default 33)")
    parser.add_argument("--reload-luts", action="store_true",
                        help="Force-reload the 3D LUT registry (clears cache, "
                             "re-scans luts/) and exit. Symmetric with the GUI "
                             "Reload LUTs button; useful after adding .cube files "
                             "mid-batch.")

    # F6: Look extraction from a reference image.
    parser.add_argument("--extract-look", type=str, default=None,
                        metavar="REF_IMG",
                        help="Extract a look from REF_IMG and apply it to the processed images")
    parser.add_argument("--look-base", type=str, default=None,
                        metavar="BASE_IMG",
                        help="Optional original image for paired look extraction (delta vs REF_IMG)")

    # Per-face recipe / param overrides (JSON keyed by detection-order index).
    parser.add_argument("--face-params", type=str, default=None,
                        metavar="PATH",
                        help="JSON file of per-face overrides, e.g. "
                             '{"0": {"recipe": "cosplay", "smooth": 70}, '
                             '"1": {"recipe": "natural"}}. Engine units (0-100).')

    # T5: optional true-linear RAW develop (gamma=1) before engine sRGB path.
    parser.add_argument("--linear-raw", action="store_true",
                        help="For RAW inputs: decode linear (gamma=1,1), optional "
                             "exposure/contrast develop, then gamma-encode into engine.")
    parser.add_argument("--raw-exposure", type=float, default=0.0,
                        help="With --linear-raw: additional exposure stops after RAF bias (default 0)")
    parser.add_argument("--raw-contrast", type=float, default=1.0,
                        help="With --linear-raw: linear contrast factor (default 1)")
    parser.add_argument(
        "--raf-decoder",
        choices=["rawpy", "raf2jpeg", "rawpy-fuji-match"],
        default="rawpy",
        help="RAF decoder: native 16-bit rawpy (default), external raf2jpeg, "
             "or full-resolution rawpy calibrated to the Fuji preview",
    )
    parser.add_argument(
        "--raf2jpeg-path",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to the raf2jpeg executable (auto-discovers ../raf2jpeg/bin/raf2jpeg)",
    )
    parser.add_argument(
        "--raf2jpeg-quality",
        type=int,
        choices=range(1, 101),
        default=100,
        metavar="1-100",
        help="Fallback JPEG quality with --raf-decoder raf2jpeg (default: 100)",
    )
    parser.add_argument(
        "--fuji-match-strength",
        type=float,
        default=0.85,
        metavar="0-1",
        help="Camera-preview calibration strength with --raf-decoder rawpy-fuji-match (default: 0.85)",
    )
    parser.add_argument(
        "--no-raf-exposure-bias",
        action="store_true",
        help="With --raf-decoder rawpy (default): do not undo the RAF's recorded "
             "RawExposureBias at decode, including --linear-raw",
    )

    args = parser.parse_args()

    if not 0.0 <= args.fuji_match_strength <= 1.0:
        parser.error("--fuji-match-strength must be between 0 and 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.ram_budget_gib is not None and args.ram_budget_gib <= 0:
        parser.error("--ram-budget-gib must be greater than 0")
    if args.max_input_pixels is not None and args.max_input_pixels <= 0:
        parser.error("--max-input-pixels must be greater than 0")
    social_formats = None
    if args.social_crops is not None:
        from retouch.social_crops import parse_formats
        try:
            social_formats = parse_formats(args.social_crops)
        except ValueError as exc:
            parser.error(f"--social-crops: {exc}")
    if args.linear_raw and args.raf_decoder != "rawpy":
        parser.error("--linear-raw can only be combined with --raf-decoder rawpy")

    # T4: recipe cookbook browse mode — exit before requiring an input image.
    if args.list_recipes is not None:
        infos = list_recipes(args.list_recipes or None)
        if not infos:
            print("No recipes found"
                  + (f" in category '{args.list_recipes}'" if args.list_recipes else ""))
            return
        print(f"Recipes ({len(infos)}):")
        for info in infos:
            print(f"  [{info.category}] {info.name}: {info.description}")
        return
    if args.search_recipes:
        infos = search_recipes(args.search_recipes)
        print(f"Search '{args.search_recipes}' -> {len(infos)} match(es):")
        for info in infos:
            print(f"  [{info.category}] {info.name}: {info.description}")
        return

    if args.export_lut is not None:
        if not args.recipe:
            parser.error("--export-lut needs --recipe NAME")
        from retouch.lut import load_cube
        from retouch.lut_export import changes_colour, export_recipe_lut
        target = args.export_lut or args.output or "."
        try:
            path, skipped = export_recipe_lut(args.recipe, target, args.lut_size)
        except (KeyError, ValueError) as exc:
            parser.error(f"--export-lut: {exc}")
        print(f"✓ Wrote {path}")
        if not changes_colour(load_cube(path)):
            print(f"  Note: {args.recipe} has no colour look (it only "
                  "retouches), so this LUT leaves colours unchanged.")
        if skipped:
            print("  Not in the LUT (needs the full app): " + ", ".join(skipped))
        return

    if args.reload_luts:
        from retouch.lut import get_registry
        try:
            get_registry().reload()
            from retouch.lut import list_available_luts
            available = list_available_luts()
            print(f"✓ LUT registry reloaded — {len(available)} LUT(s):")
            for stem in available:
                print(f"  {stem}")
        except Exception as e:
            print(f"✖ LUT reload failed: {e}")
            sys.exit(1)
        return

    # BB6: opt-in pre-flight sanity check. Runs the integrity checks on
    # demand and exits; never implicit per-init overhead.
    if args.preflight_check:
        from retouch.benchmark import run_preflight_checks, benchmark_engine_throughput
        checks = run_preflight_checks()
        all_ok = True
        for name, ok in checks.items():
            print(f"  {'✓' if ok else '✖'} {name}")
            if not ok:
                all_ok = False
        if all_ok:
            try:
                engine = RetouchEngine()
                try:
                    results = benchmark_engine_throughput(
                        lambda img: engine.process(img, recipe="natural"),
                        resolutions={"720p": (1280, 720)},
                        iterations=1,
                    )
                    for label, r in results.items():
                        print(f"  ⏱ {label}: {r.elapsed_seconds:.2f}s "
                              f"({r.mpx_per_sec:.1f} Mpx/s)")
                finally:
                    engine.close()
            except Exception as e:
                print(f"  ⚠ throughput benchmark skipped: {e}")
            print("✓ Pre-flight checks passed")
        else:
            print("✖ Pre-flight checks failed")
            sys.exit(1)
        return

    input_tokens = list(args.input or [])
    if args.input_list:
        try:
            input_tokens.extend(load_input_list(Path(args.input_list)))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"✖ Could not load input list: {exc}")
            sys.exit(1)
    if args.input_plan and args.resume_plan:
        if _path_key(Path(args.input_plan)) == _path_key(Path(args.resume_plan)):
            print("✖ --input-plan and --resume-plan must be different files")
            sys.exit(1)
    if not input_tokens and args.resume_plan:
        try:
            resume_payload = json.loads(
                Path(args.resume_plan).expanduser().read_text(encoding="utf-8")
            )
            input_tokens.extend(resume_payload.get("input_tokens", []))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"✖ Could not load resume plan inputs: {exc}")
            sys.exit(1)
    if not input_tokens:
        parser.error("an input file/directory or --input-list is required")

    try:
        input_plan = build_input_plan(
            input_tokens,
            recursive=args.recursive,
            include=args.include,
            exclude=args.exclude,
            include_hidden=args.include_hidden,
            raw_jpeg_policy=args.raw_jpeg_policy,
        )
    except (OSError, ValueError) as exc:
        print(f"✖ Input planning failed: {exc}")
        sys.exit(1)

    for row in input_plan.selected_records:
        row.decoder_requested = (
            args.raf_decoder
            if row.source_kind == "raw"
            else "pillow/opencv"
        )

    input_path = Path(input_tokens[0]).expanduser()
    output_dir = Path(args.output).expanduser().resolve() if args.output else None
    recursive_root = (
        input_path.resolve()
        if len(input_tokens) == 1 and input_path.is_dir() and args.recursive
        else None
    )

    if args.input_check in ("headers", "decode"):
        inspect_plan_headers(
            input_plan,
            max_pixels=args.max_input_pixels,
            multi_frame_policy=args.multi_frame_policy,
            raw_decoder=args.raf_decoder,
            raf2jpeg_path=args.raf2jpeg_path,
        )
    if args.input_check == "decode":
        decode_plan_inputs(
            input_plan,
            raw_decoder=args.raf_decoder,
            raf2jpeg_path=args.raf2jpeg_path,
            raf2jpeg_quality=args.raf2jpeg_quality,
            fuji_match_strength=args.fuji_match_strength,
            optical_correction=args.optical_correction,
            apply_exposure_bias=not args.no_raf_exposure_bias,
        )

    files = input_plan.selected_paths
    if not files:
        print_input_plan(input_plan)
        if args.input_plan:
            input_plan.write(Path(args.input_plan))
        hard_input_failure = any(
            row.status in {"missing", "rejected_not_regular_file"}
            for row in input_plan.rows
        )
        if args.dry_run and not hard_input_failure:
            return
        print("✖ No image files found")
        sys.exit(1)

    params = build_params(args)

    # Session loading: session params override recipe defaults but are
    # overridden by explicit CLI flags. Precedence: CLI > session > recipe.
    if args.session:
        try:
            loaded = _load_session_params(args.session)
        except FileNotFoundError as e:
            print(f"✖ Could not load session: {e}")
            sys.exit(1)
        except ValueError as e:
            print(f"✖ Could not load session: {e}")
            sys.exit(1)
        except OSError as e:
            print(f"✖ Could not load session: {e}")
            sys.exit(1)
        # Start from session params, then layer CLI-provided keys on top.
        # CLI params keys take precedence (explicit flag wins).
        merged: Dict[str, Any] = dict(loaded["params"])
        for k, v in params.items():
            merged[k] = v
        # If the CLI didn't set a recipe but the session has one, use it.
        if "recipe" not in params and loaded["recipe"]:
            merged["recipe"] = loaded["recipe"]
        params = merged

    # F6: extract a look from a reference image and apply it over the current
    # params (look wins). Drops internal keys (e.g. _split_tone_three_way) that
    # are not valid engine kwargs. Failure exits loudly so the user knows.
    if getattr(args, "extract_look", None):
        try:
            ref = imread_exif(Path(args.extract_look))
            base = imread_exif(Path(args.look_base)) if args.look_base else None
            look = LookExtractor().extract(ref, base)
            look_params = {
                k: v for k, v in look["engine_params"].items()
                if not k.startswith("_")
            }
            params.update(look_params)
            print(
                f"Applied extracted look ({look['mode']}): "
                f"{ {k: (round(v, 3) if isinstance(v, float) else v) for k, v in look_params.items()} }"
            )
        except Exception as e:
            print(f"✖ Look extraction failed: {e}")
            sys.exit(1)

    output_stems = {
        _path_key(Path(row.path)): row.output_stem
        for row in input_plan.selected_records
        if row.path and row.output_stem
    }
    attach_destinations(
        input_plan,
        output_dir=output_dir,
        format_arg=args.format,
        bit_depth=args.bit_depth,
        compare=args.compare,
        save_session=args.save_session,
        recursive_root=recursive_root,
        destination_builder=_destination_for_image,
        output_format_resolver=output_format,
    )
    input_plan.config_fingerprint = stable_fingerprint({
        "params": params,
        "format": args.format,
        "quality": args.quality,
        "bit_depth": args.bit_depth,
        "compare": args.compare,
        "global_only": args.global_only,
        "max_dim": args.max_dim,
        "max_input_pixels": args.max_input_pixels,
        "multi_frame_policy": args.multi_frame_policy,
        "raw_decoder": args.raf_decoder,
        "raf_exposure_bias": not args.no_raf_exposure_bias,
        "raw_jpeg_policy": args.raw_jpeg_policy,
    })
    if args.resume_plan:
        apply_resume_plan(
            input_plan,
            Path(args.resume_plan),
            input_plan.config_fingerprint,
            force=args.force,
        )
        verified = [
            row for row in input_plan.selected_records
            if row.status == "resume_verified"
        ]
        rerender = [
            row for row in input_plan.selected_records
            if row.status != "resume_verified"
        ]
        reasons: Dict[str, int] = {}
        for row in rerender:
            key = row.resume_reason or "not in resume plan"
            reasons[key] = reasons.get(key, 0) + 1
        reason_text = ", ".join(f"{count} {why}" for why, count in reasons.items())
        print(
            f"↺ Resume plan: {len(verified)} verified (skipped), "
            f"{len(rerender)} to re-render"
            + (f" ({reason_text})" if reason_text else "")
        )
    files = input_plan.execution_paths
    # Rows the resume plan authorized to replace their own hash-verified prior
    # output (see apply_resume_plan). Everything else keeps the global
    # --force policy: an existing output is skipped unless -f is given.
    replace_keys = {
        _path_key(Path(row.path))
        for row in input_plan.selected_records
        if row.path and row.replace_prior_output
    }

    def _force_for(path: Path) -> bool:
        return bool(args.force) or _path_key(path) in replace_keys

    if args.ram_budget_gib is not None and files:
        selected_sizes = [
            row.estimated_working_bytes or 24 * 1024 * 1024
            for row in input_plan.selected_records
            if row.status != "resume_verified"
        ]
        largest_input = max(selected_sizes, default=24 * 1024 * 1024)
        worker_cap = max(
            1,
            int((args.ram_budget_gib * (1024 ** 3)) // largest_input),
        )
        if worker_cap < args.workers:
            print(
                f"⚠ RAM budget caps workers from {args.workers} to {worker_cap} "
                f"(largest estimated input {largest_input / (1024 ** 3):.2f} GiB)"
            )
            args.workers = worker_cap

    if not files:
        print_input_plan(input_plan)
        if args.input_plan:
            input_plan.write(Path(args.input_plan))
        if input_plan.blocking_issues:
            sys.exit(1)
        print("✓ All selected inputs are already verified in the resume plan")
        return

    validation_errors = [issue["message"] for issue in input_plan.blocking_issues]
    preflight_error = "; ".join(validation_errors) if validation_errors else None
    try:
        _preflight_destinations(
            files,
            output_dir,
            args.format,
            args.bit_depth,
            recursive_root=recursive_root,
            compare=args.compare,
            save_session=args.save_session,
            output_stems=output_stems,
        )
    except (OSError, ValueError) as exc:
        destination_error = str(exc)
        preflight_error = "; ".join(
            item for item in (preflight_error, destination_error) if item
        )
        # Record only the new destination problem; the plan's own blocking
        # issues are already listed individually.
        input_plan.add_issue("destination_preflight", destination_error)

    if not args.skip_disk_check:
        for warning in _check_disk_space(
            files,
            output_dir,
            args.compare,
            enforce=not args.dry_run,
        ):
            input_plan.add_issue("disk_space_estimate", warning, severity="warning")

    if args.input_plan:
        input_plan.write(Path(args.input_plan))

    if args.dry_run:
        print_input_plan(input_plan)
        print(f"\nSettings: {params}")
        print(f"Workers: {args.workers}")
        if preflight_error:
            print(f"⚠ Output preflight would block execution: {preflight_error}")
        return

    if preflight_error:
        print(f"✖ Output preflight failed: {preflight_error}")
        sys.exit(1)

    params = _finalize_params(params)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    review_root = review_root_for(output_dir, input_path) if args.review else None

    t0 = time.time()
    done = skipped = failed = 0
    use_pool = args.workers > 1 and len(files) > 1
    progress_keys = {}
    used_progress_keys = set()
    for index, path in enumerate(files):
        key = _progress_key(path, recursive_root)
        if key in used_progress_keys:
            key = f"{index + 1}:{key}"
            suffix = 2
            while key in used_progress_keys:
                key = f"{index + 1}.{suffix}:{_progress_key(path, recursive_root)}"
                suffix += 1
        used_progress_keys.add(key)
        progress_keys[path] = key
    progress = BatchProgress(
        [(progress_keys[path], image_megapixels(path)) for path in files]
    )
    if args.no_progress_file:
        progress_path = None
    elif args.progress_file:
        progress_path = Path(args.progress_file).expanduser()
    elif output_dir:
        progress_path = output_dir / ".retouch-progress.json"
    else:
        progress_path = None
    progress_q = multiprocessing.Queue() if use_pool else queue.Queue()
    reporter = ProgressReporter(progress, progress_q, progress_path)
    reporter.__enter__()
    interrupted = False
    active_progress = {
        "path": None, "key": None, "info": {}, "started": None, "finished": False,
    }

    def _record_result(
        path_value: str,
        status: str,
        processing_evidence: Optional[Dict[str, Any]] = None,
    ) -> None:
        # In serial mode the input-plan write is the common completion path
        # for successful, skipped, and rejected images. Use it to guarantee a
        # finish event on every ordinary exit path without duplicating logic.
        path_key = _path_key(Path(path_value))
        if (
            not use_pool
            and active_progress["key"] is not None
            and not active_progress["finished"]
            and path_key == active_progress["path"]
        ):
            info = active_progress["info"]
            info["seconds"] = round(time.time() - active_progress["started"], 3)
            emit(
                progress_q, active_progress["key"], "finish",
                status=status, **info,
            )
            active_progress["finished"] = True
        row = input_plan.row_for(Path(path_value))
        if row is None:
            return
        if processing_evidence is not None:
            row.processing_evidence = processing_evidence
        row.result_status = "done" if status == "done" else (
            "skipped_existing" if status == "skipped" else "failed"
        )
        if status not in ("done", "skipped"):
            row.result_message = status
        row.source_sha256 = sha256_file(Path(path_value))
        if status == "done" and row.planned_output:
            row.output_sha256 = sha256_file(Path(row.planned_output))
            row.artifact_sha256 = {
                artifact: digest
                for artifact in row.planned_artifacts
                if (digest := sha256_file(Path(artifact))) is not None
            }

    if args.workers > 1 and len(files) > 1:
        pool_args = [
            (f, output_dir, params, args.format, args.quality, _force_for(f),
             not args.no_exif, args.max_dim, args.compare, args.global_only,
             args.bit_depth, args.fail_on_qa, args.save_session, args.smart,
             args.linear_raw, args.raw_exposure, args.raw_contrast,
             args.raf_decoder, args.raf2jpeg_path, args.raf2jpeg_quality,
             args.fuji_match_strength, args.optical_correction,
             recursive_root, output_stems.get(_path_key(f)),
             not args.no_raf_exposure_bias, review_root, args.review,
             progress_keys[f])
            for f in files
        ]
        pool = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(progress_q, not args.global_only),
        )
        try:
            futures = {pool.submit(_process_single, a): a[0] for a in pool_args}
            for future in as_completed(futures):
                path = futures[future]
                key = progress_keys[path]
                try:
                    name, status, info = future.result()
                except Exception as exc:
                    name, status, info = path.name, f"failed: {exc}", {}
                processing_evidence = info.pop("_processing_evidence", None)
                _record_result(str(path), status, processing_evidence)
                emit(progress_q, key, "finish", status=status, **info)
                if status == "done":
                    done += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    tqdm.write(f"  ✖ {name}: {status}")
        except KeyboardInterrupt:
            interrupted = True
            tqdm.write("\n  ■ Stopping: cancelling queued images and ending active workers…")
            _stop_pool(pool)
            _remove_partial_outputs(output_dir)
        finally:
            if not interrupted:
                pool.shutdown(wait=True)
    else:
        engine = None if args.global_only else RetouchEngine()
        try:
            for f in files:
                active_progress.update({
                    "path": _path_key(f),
                    "key": progress_keys[f],
                    "info": {},
                    "started": time.time(),
                    "finished": False,
                })
                sink = make_event_sink(progress_q, progress_keys[f])
                emit(progress_q, progress_keys[f], "start", worker=None)
                img_t_start = active_progress["started"]
                fmt = output_format(f, args.format)
                # For 16-bit, force PNG or TIFF
                if args.bit_depth == 16 and fmt not in ("png", "tif", "tiff"):
                    fmt = "png"
                out_path = _destination_for_image(
                    f,
                    output_dir,
                    fmt,
                    input_root=recursive_root,
                    output_stem=output_stems.get(_path_key(f)),
                )
                try:
                    _assert_safe_destination(f, out_path)
                except Exception as e:
                    failed += 1
                    tqdm.write(f"  ✖ {f.name}: {e}")
                    _record_result(str(f), f"failed: {e}")
                    continue
                if os.path.lexists(os.fspath(out_path)) and not _force_for(f):
                    skipped += 1
                    active_progress["info"]["out_path"] = str(out_path)
                    _record_result(str(f), "skipped")
                    if args.review:
                        _maybe_write_skip_record(
                            review_root, f, out_path, recursive_root, params.get("recipe"),
                        )
                    continue

                processing_evidence = None
                sink("stage", {"stage": "decode"})

                if args.linear_raw and f.suffix.lower() in RAW_EXTENSIONS:
                    try:
                        decode_info = {}
                        img_bgr = _linear_raw_to_engine_bgr(
                            f,
                            exposure=args.raw_exposure,
                            contrast=args.raw_contrast,
                            apply_exposure_bias=not args.no_raf_exposure_bias,
                            decode_info=decode_info,
                        )
                        color_context = color_context_for_path(
                            f,
                            apply_exposure_bias=not args.no_raf_exposure_bias,
                            raw_exposure_info=decode_info,
                        )
                    except Exception as e:
                        failed += 1
                        tqdm.write(f"  ✖ {f.name}: linear-raw {e}")
                        _record_result(str(f), f"linear-raw {e}")
                        if args.review:
                            _safe_write_review_record(review_root, ReviewRecord(
                                source=str(f.resolve()), status="failed", error=str(e),
                                recipe=params.get("recipe"),
                                relative=_review_relative(f, recursive_root),
                                elapsed_s=time.time() - img_t_start,
                            ))
                        continue
                else:
                    correction_status = {}
                    try:
                        img_bgr, color_context = imread_engine_with_context(
                            f,
                            raw_decoder=args.raf_decoder,
                            raf2jpeg_path=args.raf2jpeg_path,
                            raf2jpeg_quality=args.raf2jpeg_quality,
                            fuji_match_strength=args.fuji_match_strength,
                            optical_correction=args.optical_correction,
                            correction_status=correction_status,
                            apply_exposure_bias=not args.no_raf_exposure_bias,
                        )
                    except (OSError, ValueError, RuntimeError) as e:
                        failed += 1
                        tqdm.write(f"  ✖ {f.name}: {e}")
                        _record_result(str(f), str(e))
                        if args.review:
                            _safe_write_review_record(review_root, ReviewRecord(
                                source=str(f.resolve()), status="failed", error=str(e),
                                recipe=params.get("recipe"),
                                relative=_review_relative(f, recursive_root),
                                elapsed_s=time.time() - img_t_start,
                            ))
                        continue
                    if args.optical_correction:
                        tqdm.write(
                            f"  ℹ {f.name}: Lensfun "
                            f"{correction_status.get('reason') or correction_status.get('applied', ())}"
                        )
                if img_bgr is None:
                    failed += 1
                    tqdm.write(f"  ✖ {f.name}: failed to read")
                    _record_result(str(f), "failed to read")
                    if args.review:
                        _safe_write_review_record(review_root, ReviewRecord(
                            source=str(f.resolve()), status="failed", error="failed to read",
                            recipe=params.get("recipe"),
                            relative=_review_relative(f, recursive_root),
                            elapsed_s=time.time() - img_t_start,
                        ))
                    continue

                orig_shape = img_bgr.shape[:2]
                original_full = img_bgr.copy() if args.compare else None
                img_bgr, _scale = resize_for_processing(img_bgr, args.max_dim)

                # F10: --smart — per-image analysis overrides recipe/params.
                effective_params = params
                if args.smart:
                    try:
                        effective_params, suggestion = _smart_params_for_image(
                            img_bgr, params
                        )
                        tqdm.write(
                            f"  🧠 {f.name}: {suggestion.recipe} "
                            f"({len(suggestion.params)} overrides)"
                        )
                    except (ValueError, RuntimeError) as e:
                        tqdm.write(
                            f"  ⚠ {f.name}: smart analysis failed ({e}), "
                            f"using base params"
                        )

                review_meta = {"faces": [], "qa": []}
                if args.global_only:
                    sink("stage", {"stage": "global_finish"})
                    result = _apply_global_finish(img_bgr, dict(effective_params))
                    _result_info(result, active_progress["info"])
                    processing_evidence = _processing_evidence(
                        result, global_only=True, color_context=color_context
                    )
                else:
                    result = engine.process(
                        img_bgr, progress_cb=sink, **dict(effective_params)
                    )
                    _result_info(result, active_progress["info"])
                    processing_evidence = _processing_evidence(
                        result, color_context=color_context
                    )
                    if args.review:
                        review_meta = result_review_meta(result, _scale)
                    if args.fail_on_qa:
                        qa = getattr(result, 'qa', [])
                        if qa:
                            flagged = [w for w in qa if w.flagged]
                            if flagged:
                                reasons = "; ".join(f"{w.detector}={w.score:.2f}" for w in flagged)
                                print(f"  ✖ {f.name}: QA_FAIL: {reasons}")
                                failed += 1
                                _record_result(
                                    str(f), f"QA_FAIL: {reasons}", processing_evidence
                                )
                                if args.review:
                                    _safe_write_review_record(review_root, ReviewRecord(
                                        source=str(f.resolve()), status="qa_fail",
                                        recipe=params.get("recipe"),
                                        faces=review_meta.get("faces", []),
                                        qa=review_meta.get("qa", []),
                                        relative=_review_relative(f, recursive_root),
                                        elapsed_s=time.time() - img_t_start,
                                    ))
                                continue

                if _scale < 1.0:
                    result = cv2.resize(result, (orig_shape[1], orig_shape[0]),
                                        interpolation=cv2.INTER_LINEAR)

                try:
                    # Embed the working-space ICC selected by ColorContext.
                    # The source ICC is never reattached to converted pixels.
                    from retouch.io import read_exif_bytes
                    exif_bytes = read_exif_bytes(f) if not args.no_exif else None
                    c2pa_manifest = read_c2pa_manifest(f) if not args.no_exif else None
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    _assert_safe_destination(f, out_path)
                    sink("stage", {"stage": "write"})
                    write_image_with_color_context(
                        str(out_path),
                        result,
                        color_context,
                        bit_depth=args.bit_depth,
                        quality=args.quality,
                        exif=exif_bytes,
                        c2pa_manifest=c2pa_manifest,
                    )
                    active_progress["info"]["out_path"] = str(out_path)

                    if args.compare:
                        sink("stage", {"stage": "compare"})
                        compare_path = out_path.with_name(
                            f"{out_path.stem}_compare{out_path.suffix}"
                        )
                        _assert_safe_destination(f, compare_path)
                        make_comparison(
                            original_full, result, compare_path, fmt, args.quality
                        )
                        active_progress["info"]["compare_path"] = str(compare_path)
                except Exception as e:
                    failed += 1
                    tqdm.write(f"  ✖ {f.name}: export failed: {e}")
                    _record_result(str(f), f"failed: {e}", processing_evidence)
                    continue

                if args.save_session is not None:
                    save_path = None if args.save_session is True else args.save_session
                    try:
                        written = _save_session(effective_params, f, out_path, save_path)
                    except (OSError, ValueError) as e:
                        tqdm.write(f"  ⚠ {f.name}: could not save session: {e}")
                        _record_result(
                            str(f),
                            f"saved_image_but_session_failed: {e}",
                            processing_evidence,
                        )
                        failed += 1
                        continue
                    else:
                        tqdm.write(f"  💾 session → {written}")
                if args.review:
                    _safe_write_review_record(review_root, ReviewRecord(
                        source=str(f.resolve()),
                        output=str(out_path),
                        compare=str(compare_path) if args.compare else None,
                        status="done",
                        recipe=params.get("recipe"),
                        faces=review_meta.get("faces", []),
                        qa=review_meta.get("qa", []),
                        relative=_review_relative(f, recursive_root),
                        elapsed_s=time.time() - img_t_start,
                    ))
                done += 1
                _record_result(str(f), "done", processing_evidence)
        except KeyboardInterrupt:
            interrupted = True
            if active_progress["key"] is not None and not active_progress["finished"]:
                info = active_progress["info"]
                info["seconds"] = round(time.time() - active_progress["started"], 3)
                emit(
                    progress_q, active_progress["key"], "finish",
                    status="failed: interrupted", **info,
                )
                active_progress["finished"] = True
            tqdm.write("\n  ■ Stopped.")
        finally:
            if engine is not None:
                engine.close()

    summary = reporter.close()
    if use_pool:
        progress_q.cancel_join_thread()
        progress_q.close()
    elapsed = time.time() - t0
    heading = "Stopped" if interrupted else "Done"
    print(f"\n{heading} — {done} processed, {skipped} skipped, {failed} failed"
          f"  ({elapsed:.1f}s)")
    for line in (summary if interrupted else summary[1:]):
        print(line)
    if progress_path is not None:
        print(f"Progress file: {progress_path}")
    if args.input_plan:
        input_plan.write(Path(args.input_plan))

    if social_formats:
        _export_social_crops(
            files, output_dir, args, recursive_root, social_formats,
        )
    if args.review and not args.dry_run and len(files) > 0:
        try:
            page = build_review_page(review_root, workers=args.workers)
            print(f"Review page → {page}")
        except Exception as e:
            print(f"⚠ Review page failed: {e}")

    if interrupted:
        print("Run the same command again to continue: finished images are "
              "skipped and the rest are rendered.")
        sys.exit(130)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

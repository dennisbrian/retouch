#!/usr/bin/env python3
"""Shoot Intelligence / Watch Folder / Review / Profiles / Look Board cluster.

Pure move from gui.py. gui.py re-exports every public name and injects a
``gui`` module reference after import; handlers resolve patchable shared
names (``FaceDetector``, ``BatchProcessor``, ``ProjectProfileStore``, …)
through that reference at call time so test monkeypatches of ``gui.X`` keep
working. This module must never import gui directly (no import cycles).
"""

import json
import logging
from pathlib import Path

from retouch.project_profiles import (
    LookBoard,
    LookReference,
    ProjectProfile,
    ProjectProfileStore,
)
from retouch.shoot_intelligence import (
    ProjectGraph,
    ProjectNode,
    group_bursts,
    inspect_asset,
    rank_burst_candidates,
)
from retouch.shoot_review import (
    ShootReviewManifest,
    ShootReviewManifestStore,
    build_review_manifest,
)
from retouch.face_quality import FaceQualityAnalyzer
from retouch.watch_folder import WatchFolder
from retouch.batch_processor import BatchProcessor
from retouch.look_extractor import LookExtractor
from retouch import __version__
from retouch.io import imread_exif

_logger = logging.getLogger(__name__)

# Assigned by gui.py right after import (same pattern as
# gui_advanced.get_engine). Handlers resolve patchable shared names
# (FaceDetector, BatchProcessor, ProjectProfileStore, imread_exif,
# model_status) through this module reference at call time so test
# monkeypatches of ``gui.X`` keep working. Never import gui here.
gui = None


def _resolve_image_path(value):
    """Best-effort extraction of a filesystem path from a Gradio input value."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("name") or value.get("path")
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _resolve_image_path(value[0])
    if hasattr(value, "name"):
        return value.name
    return str(value)


def _resolve_image_paths(value):
    """Extract all filesystem paths from single/multiple Gradio file values."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        paths = []
        for item in value:
            path = _resolve_image_path(item)
            if path:
                paths.append(path)
        return paths
    path = _resolve_image_path(value)
    return [path] if path else []


def on_shoot_intelligence_scan(
    input_dir,
    recursive=True,
    manifest_path=None,
    analyze_face_quality=False,
):
    """Scan a shoot and persist explainable review evidence without culling."""
    if not input_dir or not str(input_dir).strip():
        return [], "Enter a shoot folder first.", "{}"
    try:
        root = Path(str(input_dir)).expanduser().resolve()
        resolved_manifest_path = (
            Path(str(manifest_path)).expanduser().resolve()
            if manifest_path and str(manifest_path).strip()
            else ShootReviewManifest.default_path(root)
        )
        candidates = root.rglob("*") if recursive else root.iterdir()
        assets = [
            inspect_asset(path)
            for path in sorted(candidates)
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
        ]
        bursts = group_bursts(assets)
        candidates_by_group = {
            burst.group_id: rank_burst_candidates(burst)
            for burst in bursts
        }
        face_evidence_by_path = None
        face_uncertainty_by_path = None
        face_status = "Face quality was not requested."
        if bool(analyze_face_quality):
            face_evidence_by_path = {}
            face_uncertainty_by_path = {}
            detector = None
            try:
                detector = gui.FaceDetector(allow_unavailable=True)
                if not detector.available:
                    for asset in assets:
                        face_uncertainty_by_path[asset.path] = ["face_detector_unavailable"]
                    face_status = (
                        "Face quality unavailable: "
                        f"{detector.unavailable_reason or 'detector initialization failed'}."
                    )
                else:
                    import mediapipe as mp

                    face_model_status = gui.model_status("face_landmarker")
                    analyzer = FaceQualityAnalyzer(detector_details={
                        "runtime": "mediapipe",
                        "runtime_version": str(getattr(mp, "__version__", "unknown")),
                        "backend": str(getattr(detector, "backend_name", "unknown")),
                        "bbox_pipeline": "retinaface-when-available; mediapipe fallback",
                        "model_id": "face_landmarker",
                        "model_sha256": str(face_model_status.get("sha256", "")),
                        "model_integrity_verified": bool(face_model_status.get("available")),
                        "retinaface_model_integrity_verified": False,
                    })
                    analyzed_assets = 0
                    detected_faces = 0
                    failed_assets = 0
                    for asset in assets:
                        try:
                            image_bgr = gui.imread_exif(asset.path)
                            detected = detector.detect(image_bgr)
                            evidence = analyzer.analyze(image_bgr, detected)
                            face_evidence_by_path[asset.path] = evidence
                            analyzed_assets += 1
                            detected_faces += len(evidence)
                        except Exception as exc:  # one unreadable asset must not erase the shoot scan
                            failed_assets += 1
                            face_uncertainty_by_path[asset.path] = [
                                f"face_quality_analysis_failed:{type(exc).__name__}"
                            ]
                            _logger.warning("Face-quality analysis failed for %s: %s", asset.path, exc)
                    face_status = (
                        f"Face quality measured for {analyzed_assets} asset(s): "
                        f"{detected_faces} face(s), {failed_assets} failed asset(s)."
                    )
            except Exception as exc:
                for asset in assets:
                    face_uncertainty_by_path.setdefault(
                        asset.path, [f"face_detector_initialization_failed:{type(exc).__name__}"]
                    )
                face_status = f"Face quality unavailable: {type(exc).__name__}: {exc}."
                _logger.warning("Face-quality initialization failed: %s", exc)
            finally:
                if detector is not None:
                    try:
                        detector.close()
                    except Exception as exc:
                        _logger.warning("Face detector close failed after shoot scan: %s", exc)
        manifest = build_review_manifest(
            root,
            assets,
            bursts,
            candidates_by_group,
            face_evidence_by_path=face_evidence_by_path,
            face_uncertainty_by_path=face_uncertainty_by_path,
            manifest_path=resolved_manifest_path,
        )
        asset_ids_by_path = {
            record.relative_path: record.asset_id
            for record in manifest.assets.values()
            if record.present
        }
        asset_ids_by_absolute_path = {
            str((root / relative_path).resolve()): asset_id
            for relative_path, asset_id in asset_ids_by_path.items()
        }
        review_by_absolute_path = {
            str((root / record.relative_path).resolve()): record
            for record in manifest.assets.values()
            if record.present
        }

        def format_evidence_metric(value):
            return "unavailable" if value is None else f"{float(value):.6g}"

        rows = []
        for asset in assets:
            review_record = review_by_absolute_path.get(asset.path)
            asset_status = "inspected"
            if review_record and review_record.faces:
                asset_status = f"{len(review_record.faces)} face evidence record(s); review required"
            rows.append([
                "asset", "", asset_ids_by_absolute_path.get(asset.path, ""), asset.path, "",
                asset.camera_model or "unknown", asset.capture_time or "", asset_status,
            ])
            if review_record:
                for face in review_record.faces:
                    evidence_text = (
                        f"coverage={format_evidence_metric(face.coverage)}; "
                        f"confidence={format_evidence_metric(face.detector_confidence)} "
                        f"({face.detector_confidence_source}); "
                        f"eye_sharpness=L:{format_evidence_metric(face.left_eye_sharpness)} "
                        f"R:{format_evidence_metric(face.right_eye_sharpness)}; "
                        f"method={face.sharpness_method}; "
                        f"uncertainty={','.join(face.uncertainty) or 'none'}"
                    )
                    rows.append([
                        "face", "", review_record.asset_id, asset.path,
                        format_evidence_metric(face.face_sharpness), evidence_text, face.face_id,
                        "evidence only; human review required",
                    ])
        for burst in bursts:
            candidates = candidates_by_group[burst.group_id]
            for candidate in candidates:
                rows.append([
                    "burst", burst.group_id, asset_ids_by_absolute_path.get(candidate.path, ""),
                    candidate.path, round(candidate.score, 4), " / ".join(burst.reasons),
                    candidate.rank, "review required",
                ])
        graph = ProjectGraph([
            ProjectNode("ingest", "ingest", status="succeeded", outputs=[str(input_dir)]),
            ProjectNode("burst_grouping", "burst_grouping", dependencies=["ingest"], status="succeeded", outputs=[f"{len(bursts)} burst group(s)"]),
            ProjectNode("human_cull_review", "human_cull_review", dependencies=["burst_grouping"]),
        ])
        graph.refresh_ready()
        status = (
            f"Inspected {len(assets)} asset(s), found {len(bursts)} burst group(s). "
            f"Review manifest saved to {resolved_manifest_path}. "
            f"{face_status} "
            "Candidate ranks and face evidence are recommendations; human review is required."
        )
        return rows, status, graph.to_json()
    except Exception as exc:
        _logger.exception("Shoot Intelligence scan failed: %s", exc)
        return [], f"Shoot scan failed: {exc}", "{}"


def on_watch_folder_process(input_dir, state_path, limit=None, output_dir=None,
                            job_kind="preview", recipe_name="natural", manifest_path=None):
    """Scan, queue, and run real preview/final jobs when an output is supplied."""
    if not input_dir:
        return "Enter a watch folder first."
    try:
        watcher = WatchFolder(input_dir, state_path=state_path or None)
        if limit is not None and int(limit) < 0:
            raise ValueError("limit must be non-negative")
        if output_dir and str(output_dir).strip():
            input_root = Path(input_dir).expanduser().resolve()
            output_root = Path(str(output_dir)).expanduser().resolve()
            try:
                output_root.relative_to(input_root)
            except ValueError:
                pass
            else:
                raise ValueError("output folder must be outside the watched folder")
            selected_kind = str(job_kind or "preview").strip().lower()
            if selected_kind not in {"preview", "final"}:
                raise ValueError("job kind must be preview or final")
            settings = {
                "recipe": str(recipe_name or "natural").strip(),
                "settings_source": "selected_recipe",
                "export_fmt": "JPEG",
                "export_quality": 95 if selected_kind == "final" else 85,
                "export_res": "Original" if selected_kind == "final" else "720px",
            }
            review_path = (
                Path(str(manifest_path)).expanduser().resolve()
                if manifest_path and str(manifest_path).strip()
                else ShootReviewManifest.default_path(input_root)
            )
            review = ShootReviewManifest.load(review_path)
            review.project_root = str(input_root)

            def asset_instance_resolver(source_path, content_id):
                relative = source_path.relative_to(input_root).as_posix()
                matches = [
                    asset for asset in review.assets.values()
                    if asset.relative_path == relative and asset.content_id == content_id
                ]
                if len(matches) == 1:
                    return matches[0].asset_id
                # An unscanned or ambiguous path remains a distinct instance;
                # it must not inherit another asset's human decision.
                return f"unmanifested:{relative}:{content_id}"

            queued = watcher.queue_jobs(
                kind=selected_kind,
                output_root=output_root,
                settings=settings,
                pipeline_fingerprint=f"retouch-{__version__}-batch-v3",
                asset_instance_resolver=asset_instance_resolver,
                limit=int(limit) if limit is not None else None,
            )

            def process_watch_job(job):
                job_settings = dict(job.settings)
                source_path = Path(job.source_path).resolve()
                try:
                    relative_source = source_path.relative_to(input_root)
                except ValueError as exc:
                    raise ValueError(
                        f"watch job source is outside the watched folder: {source_path}"
                    ) from exc

                # BatchProcessor preserves the input-relative directory below
                # its output root. Derive that root from the durable output
                # contract instead of the current UI selection.
                job_output = Path(job.output_path).resolve()
                batch_output_root = job_output.parent
                for _ in relative_source.parent.parts:
                    batch_output_root = batch_output_root.parent
                processor = gui.BatchProcessor()
                try:
                    processed, _, _, log = processor.process_folder(
                        input_dir=input_root,
                        output_dir=batch_output_root,
                        style_name_or_recipe=str(job_settings.get("recipe", "natural")),
                        export_fmt=str(job_settings.get("export_fmt", "JPEG")),
                        export_quality=int(job_settings.get("export_quality", 95)),
                        export_res=str(job_settings.get("export_res", "Original")),
                        auto_group=False,
                        generate_sheet=False,
                        export_zip=False,
                        num_workers=1,
                        only_files=[source_path],
                        output_path_overrides={source_path: job_output},
                    )
                    if len(processed) != 1:
                        raise RuntimeError(
                            f"batch processor returned {len(processed)} output(s): {log}"
                        )
                    return Path(processed[0])
                finally:
                    processor.close()

            watcher.run_queued(
                process_watch_job,
                kind=selected_kind,
                limit=int(limit) if limit is not None else None,
            )
            job_ids = {
                record.job_ids.get(selected_kind)
                for record in watcher.state.records.values()
                if record.job_ids.get(selected_kind)
            }
            relevant_jobs = [
                watcher.state.jobs[job_id]
                for job_id in job_ids
                if job_id in watcher.state.jobs
            ]
            succeeded = sum(job.status == "done" for job in relevant_jobs)
            failed = sum(job.status == "failed" for job in relevant_jobs)
            retryable_failed = sum(job.status == "failed" and job.retryable for job in relevant_jobs)
            permanent_failed = failed - retryable_failed
            queued_count = sum(job.status in {"queued", "processing"} for job in relevant_jobs)

            for job in relevant_jobs:
                if not job.asset_instance_id.startswith("unmanifested:") and job.asset_instance_id in review.assets:
                    review.attach_job_provenance(job.asset_instance_id, selected_kind, {
                        "job_id": job.job_id,
                        "status": job.status,
                        "output_path": job.output_path,
                        "output_size": job.output_size,
                        "output_sha256": job.output_sha256,
                        "settings_fingerprint": job.settings_fingerprint,
                        "pipeline_fingerprint": job.pipeline_fingerprint,
                    })
            review.save(review_path)
            return (
                f"Watch jobs: {succeeded} {selected_kind} succeeded, {failed} failed "
                f"({retryable_failed} retryable, {permanent_failed} permanent), "
                f"{queued_count} still queued. {len(queued)} new durable job(s); "
                "completion requires a verified output hash. Preview and final jobs are independent."
            )
        ready = watcher.scan_once()
        stable_count = len(ready)
        shown_count = min(stable_count, int(limit)) if limit is not None else stable_count
        return (
            f"Watch scan: {stable_count} stable file(s) detected ({shown_count} shown by the limit); "
            "no processing job was submitted, "
            f"so no files were marked done. State saved to {watcher.state_path}."
        )
    except Exception as exc:
        _logger.exception("Watch-folder pass failed: %s", exc)
        return f"Watch-folder pass failed: {exc}"


def _review_store(manifest_path):
    if not manifest_path or not str(manifest_path).strip():
        raise ValueError("Enter the saved review manifest path first.")
    return ShootReviewManifestStore(str(manifest_path).strip())


def on_save_shoot_review(manifest_path, asset_id, decision, rating=None, labels="", reviewer="", note=""):
    """Persist a human review override; this never mutates the source image."""
    try:
        store = _review_store(manifest_path)
        parsed_rating = None if rating in (None, "") else int(rating)
        parsed_labels = [item.strip() for item in str(labels or "").split(",") if item.strip()]
        asset = store.manifest.set_human_review(
            str(asset_id or "").strip(),
            str(decision or "hold").strip(),
            rating=parsed_rating,
            labels=parsed_labels,
            reviewer=reviewer,
            note=note,
        )
        store.save()
        return (
            f"Saved human review for {asset.relative_path}: {asset.decision}, "
            f"rating={asset.rating if asset.rating is not None else 'unset'}. "
            "Source file was not changed."
        )
    except Exception as exc:
        return f"Review save failed: {exc}"


def on_export_shoot_review_json(manifest_path):
    try:
        store = _review_store(manifest_path)
        target = store.save()
        return str(target), f"JSON manifest ready: {target}"
    except Exception as exc:
        return None, f"JSON export failed: {exc}"


def on_export_shoot_review_csv(manifest_path, selected_only=False):
    try:
        store = _review_store(manifest_path)
        target = store.export_csv(selected_only=bool(selected_only))
        return str(target), f"CSV export ready: {target}"
    except Exception as exc:
        return None, f"CSV export failed: {exc}"


def on_save_project_profile(profile_id, subject_key, display_name, recipe_name, style_name, preferred_json, marks_json):
    """Persist a subject-linked profile without storing image pixels."""
    if not profile_id or not subject_key or not display_name:
        return "Profile ID, subject key, and display name are required."
    try:
        preferred = json.loads(preferred_json or "{}")
        marks = json.loads(marks_json or "[]")
        if not isinstance(preferred, dict) or not isinstance(marks, list):
            raise ValueError("preferred parameters must be an object and marks must be a list")
        profile = ProjectProfile(
            profile_id=str(profile_id).strip(), subject_key=str(subject_key).strip(),
            display_name=str(display_name).strip(), recipe=recipe_name or "natural",
            style_name=style_name or None, preferred_params=preferred,
            protected_marks=marks,
        )
        ProjectProfileStore().upsert_profile(profile)
        return f"Saved project profile '{profile.display_name}' ({profile.profile_id}). No face pixels were stored."
    except Exception as exc:
        return f"Profile save failed: {exc}"


def on_save_look_board(board_id, project_id, reference_files):
    """Persist a multi-reference Look Board linked to an optional profile."""
    if not board_id:
        return "Enter a Look Board ID.", "{}"
    paths = _resolve_image_paths(reference_files)
    if not paths:
        return "Add at least one reference image.", "{}"
    try:
        board = LookBoard(str(board_id).strip(), project_id=project_id or None)
        for index, path in enumerate(paths):
            board.add_reference(LookReference(path=str(Path(path).expanduser().resolve()), label=f"Reference {index + 1}"))
        ProjectProfileStore().upsert_board(board)
        summary = f"Saved Look Board '{board.board_id}' with {len(board.references)} reference(s)."
        return summary, board.to_json()
    except Exception as exc:
        return f"Look Board save failed: {exc}", "{}"


def on_apply_look_board(board_id, img_input):
    """Load a saved Look Board and place its extracted params in Process state."""
    if not board_id:
        return {}, "Enter a Look Board ID first."
    try:
        board = gui.ProjectProfileStore().boards.get(str(board_id).strip())
        if board is None:
            return {}, f"Look Board '{board_id}' was not found."
        references = []
        missing = []
        for reference in board.references:
            path = Path(reference.path).expanduser()
            if not path.is_file():
                missing.append(str(path))
                continue
            references.append((gui.imread_exif(path), reference.weight))
        if not references:
            detail = f" Missing: {missing[0]}" if missing else ""
            return {}, f"Look Board '{board.board_id}' has no readable references.{detail}"

        base = None
        base_path = _resolve_image_path(img_input)
        if base_path:
            base = gui.imread_exif(base_path)
        result = LookExtractor().extract_board(references, base_img=base)
        clean = {
            key: value for key, value in (result.get("engine_params") or {}).items()
            if not str(key).startswith("_")
        }
        suffix = f" {len(missing)} reference(s) unavailable." if missing else ""
        return clean, (
            f"Applied Look Board '{board.board_id}' from {result['reference_count']} reference(s)."
            f" Click Process to render.{suffix}"
        )
    except Exception as exc:
        _logger.exception("Look Board apply failed: %s", exc)
        return {}, f"Look Board apply failed: {exc}"

#!/usr/bin/env python3
"""Batch processing / style learn / job dashboard cluster.

Pure move from gui.py. gui.py re-exports every public name and injects a
``gui`` module reference after import; handlers resolve patchable shared
names (``gr``, ``_run_batch``, ``get_custom_style_names``, ``list_styles``)
through that reference at call time so test monkeypatches of ``gui.X`` keep
working. This module must never import gui directly (no import cycles).
"""

import logging
from pathlib import Path

import gradio as gr

from retouch.style_library import list_styles, save_style_profile, learn_dataset_style
from retouch.batch_processor import BatchProcessor, validate_batch_roots
from retouch.style import StyleProfile

_logger = logging.getLogger(__name__)

# Assigned by gui.py right after import (same pattern as
# gui_advanced.get_engine). Handlers resolve patchable shared names through
# this module reference at call time. Never import gui here.
gui = None


def on_save_style(style_name, author, tags_str,
                  smooth, mid_reduction, texture_opacity,
                  whiten, contrast, brightness):
    style_name = style_name.strip()
    if not style_name:
        return gui.gr.update(), gui.gr.update(), "Error: Style name cannot be empty."
    if len(style_name) > 100:
        return gui.gr.update(), gui.gr.update(), "Error: Style name must be 100 characters or less."

    tags = [t.strip() for t in tags_str.split(",") if t.strip()]

    profile = StyleProfile(
        brightness_delta=float(brightness / 2.0),
        contrast_delta=float(contrast),
        saturation_delta=0.0,
        skin_l_mean_delta=float(whiten / 4.0),
        skin_a_mean_delta=0.0,
        skin_b_mean_delta=0.0,
        skin_smooth_strength=float(smooth / 100.0),
        skin_mid_reduction=float(mid_reduction),
        skin_texture_opacity=float(texture_opacity),
    )

    try:
        save_style_profile(
            name=style_name,
            profile=profile,
            author=author or "Dennis",
            tags=tags,
        )
        choices = gui.get_custom_style_names()
        gui.gr.Info(f"Style '{style_name}' saved!")
        return gui.gr.update(choices=choices, value=style_name), gui.gr.update(choices=choices, value=style_name), f"Style '{style_name}' saved successfully!"
    except Exception as e:
        _logger.exception("Failed to save style: %s", e)
        return gui.gr.update(), gui.gr.update(), f"Failed to save style: {e}"


def on_learn_style(orig_dir, edit_dir, style_name, author, tags_str, prg=gr.Progress()):
    if not orig_dir or not edit_dir:
        return gui.gr.update(), gui.gr.update(), "Error: Original and Edited folders must be specified."
    style_name = style_name.strip()
    if not style_name:
        return gui.gr.update(), gui.gr.update(), "Error: Style name cannot be empty."
    if len(style_name) > 100:
        return gui.gr.update(), gui.gr.update(), "Error: Style name must be 100 characters or less."

    def prg_cb(progress, message):
        prg(progress, desc=message)

    try:
        profile, count = learn_dataset_style(orig_dir, edit_dir, progress_callback=prg_cb)
        if count == 0:
            return gui.gr.update(), gui.gr.update(), "No matching image pairs were found or analyzed successfully."

        tags = [t.strip() for t in tags_str.split(",") if t.strip()]
        save_style_profile(
            name=style_name,
            profile=profile,
            author=author or "Dennis",
            tags=tags,
        )

        choices = gui.get_custom_style_names()
        gui.gr.Info(f"Learned style '{style_name}' from {count} pairs!")
        return gui.gr.update(choices=choices, value=style_name), gui.gr.update(choices=choices, value=style_name), f"Extracted & saved style '{style_name}' from {count} image pairs!"
    except Exception as e:
        _logger.exception("Error during dataset learning: %s", e)
        return gui.gr.update(), gui.gr.update(), f"Error during dataset learning: {e}"


def on_process_folder(input_dir, output_dir, style_type, custom_style_name, recipe_name,
                      export_fmt, export_quality, export_res, auto_group, generate_sheet, export_zip,
                      prg=gr.Progress()):
    if not input_dir or not output_dir:
        return None, None, "Error: Both Input and Output directories must be specified."
    try:
        validate_batch_roots(input_dir, output_dir)
    except (OSError, ValueError) as exc:
        return None, None, f"Error: {exc}"

    gui.gr.Info("Batch processing started...")
    processor = BatchProcessor()

    try:
        return _run_batch(processor, input_dir, output_dir, style_type,
                          custom_style_name, recipe_name, export_fmt,
                          export_quality, export_res, auto_group,
                          generate_sheet, export_zip, prg)
    finally:
        # Release MediaPipe before the GC can finalize it — FaceLandmarker's
        # __del__ blocks forever on a serial-dispatcher future, which shows up
        # as the GUI hanging after the batch reports complete.
        processor.close()


def _run_batch(processor, input_dir, output_dir, style_type, custom_style_name,
               recipe_name, export_fmt, export_quality, export_res, auto_group,
               generate_sheet, export_zip, prg, only_files=None):
    from retouch.jobs import Job, FileRecord, JobStore, make_job_id, QA_STATE_UNKNOWN, QA_STATE_CLEAN, QA_STATE_FLAGGED, QA_STATE_ERROR

    profile = None
    style_val = recipe_name

    if style_type == "Use Custom Style":
        style_val = ""
        if not custom_style_name:
            return None, None, "Error: Please select a custom style profile."
        styles = list_styles()
        for s in styles:
            if s["name"] == custom_style_name:
                profile = StyleProfile(**s["profile"])
                break
        if profile is None:
            return None, None, f"Error: Custom style '{custom_style_name}' not found."

    def prg_cb(progress, message):
        print(f"[Batch Progress {progress * 100:.1f}%]: {message}")
        prg(progress, desc=message)

    job_store = JobStore()
    job = Job(
        job_id=make_job_id(input_dir),
        input_dir=input_dir,
        output_dir=output_dir,
        style_type=style_type,
        recipe_or_style=custom_style_name if style_type == "Use Custom Style" else recipe_name,
        export_fmt=export_fmt,
        export_quality=export_quality,
        export_res=export_res,
    )
    job_store.save(job)

    def _qa_state_for(qa_list):
        if qa_list is None:
            return QA_STATE_UNKNOWN, []
        warnings = [dict(w) if isinstance(w, dict) else vars(w) for w in qa_list]
        if any(w.get("details", {}).get("available") is False for w in warnings):
            return QA_STATE_ERROR, warnings
        if warnings:
            return QA_STATE_FLAGGED, warnings
        return QA_STATE_CLEAN, warnings

    def on_file_result(source_path, output_path, qa_list, error):
        if error is not None:
            record = FileRecord(source_path=str(source_path), status="failed", error=error)
        else:
            qa_state, warnings = _qa_state_for(qa_list)
            record = FileRecord(
                source_path=str(source_path),
                output_path=str(output_path) if output_path else None,
                status="done",
                qa_state=qa_state,
                qa_warnings=warnings,
            )
        job.files.append(record)
        job_store.save(job)

    try:
        processed, sheet_path, zip_path, log = processor.process_folder(
            input_dir=input_dir,
            output_dir=output_dir,
            style_name_or_recipe=style_val,
            custom_style_profile=profile,
            export_fmt=export_fmt,
            export_quality=export_quality,
            export_res=export_res,
            auto_group=auto_group,
            generate_sheet=generate_sheet,
            export_zip=export_zip,
            progress_callback=prg_cb,
            on_file_result=on_file_result,
            only_files=only_files,
        )

        job.total_files = len(job.files)
        if job.total_files > 0 and job.failed_count == 0 and job.done_count == job.total_files:
            job.status = "done"
        elif job.done_count:
            job.status = "partial"
        else:
            job.status = "failed"
        job.log = log
        job_store.save(job)

        if job.status == "done":
            gui.gr.Info("Batch processing complete!")
        elif job.status == "partial":
            gui.gr.Warning(f"Batch processing partial: {job.done_count}/{job.total_files} files succeeded.")
        else:
            gui.gr.Warning("Batch processing failed: no files completed.")
        return sheet_path, zip_path, log
    except Exception as e:
        _logger.exception("Batch processing failed: %s", e)
        job.status = "partial" if job.done_count else "failed"
        job.log = str(e)
        job_store.save(job)
        gui.gr.Warning(f"Batch processing failed: {e}")
        return None, None, f"Exception during batch processing: {e}"


# ---------------------------------------------------------------------------
# Job Dashboard handlers
# ---------------------------------------------------------------------------

def _job_row(job) -> list:
    return [job.job_id, job.created, job.status, job.recipe_or_style, job.total_files, job.flagged_count]


def on_refresh_jobs():
    from retouch.jobs import JobStore
    jobs = JobStore().list_jobs()
    return [_job_row(j) for j in jobs]


def _file_row(rec) -> list:
    detectors = ", ".join(w.get("detector", "?") for w in rec.qa_warnings)
    return [rec.source_path, rec.status, rec.qa_state, detectors]


def on_select_job(evt: gr.SelectData, job_rows):
    from retouch.jobs import JobStore

    if evt.index is None:
        return gui.gr.update(visible=False), [], "", None

    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if job_rows is None or row_idx >= len(job_rows):
        return gui.gr.update(visible=False), [], "", None

    job_id = job_rows[row_idx][0]
    try:
        job = JobStore().load(job_id)
    except (OSError, KeyError, ValueError) as e:
        gui.gr.Warning(f"Could not load job {job_id}: {e}")
        return gui.gr.update(visible=False), [], "", None

    rows = [_file_row(f) for f in job.files]
    return gui.gr.update(visible=True), rows, job.log, job_id


def on_rerun_flagged(job_id, prg=gr.Progress()):
    from retouch.jobs import JobStore
    from retouch.batch_processor import BatchProcessor

    if not job_id:
        gui.gr.Warning("Select a job first.")
        return "No job selected."

    job_store = JobStore()
    try:
        job = job_store.load(job_id)
    except (OSError, KeyError, ValueError) as e:
        gui.gr.Warning(f"Could not load job {job_id}: {e}")
        return f"Error loading job: {e}"

    flagged = job.flagged_files()
    if not flagged:
        gui.gr.Info("No flagged files in this job.")
        return "No flagged files to re-run."

    gui.gr.Info(f"Re-running {len(flagged)} flagged file(s)...")
    processor = BatchProcessor()
    try:
        _, _, log = gui._run_batch(
            processor, job.input_dir, job.output_dir, job.style_type,
            job.recipe_or_style if job.style_type == "Use Custom Style" else "",
            job.recipe_or_style if job.style_type != "Use Custom Style" else "natural",
            job.export_fmt, job.export_quality, job.export_res,
            False, False, False, prg,
            only_files=[Path(f.source_path) for f in flagged],
        )
        return log
    finally:
        processor.close()

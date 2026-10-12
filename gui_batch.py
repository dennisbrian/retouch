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

# retouch/review_page.py is owned by another agent and may not exist yet in
# this working tree. Import lazily/best-effort so gui_batch.py keeps working
# (review wiring simply no-ops) whether or not it has landed.
try:
    from retouch.review_page import (
        ReviewRecord,
        build_review_page,
        qa_entries as _review_qa_entries_impl,
        review_root_for,
        write_review_record,
    )
    _REVIEW_AVAILABLE = True
except ImportError:
    _REVIEW_AVAILABLE = False

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
                      social_crop_formats=None, watermark_text=None,
                      watermark_position="auto", write_xmp=False, prg=gr.Progress()):
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
                          generate_sheet, export_zip, prg,
                          social_crop_formats=social_crop_formats,
                          watermark_text=watermark_text,
                          watermark_position=watermark_position,
                          write_xmp=write_xmp)
    finally:
        # Release MediaPipe before the GC can finalize it — FaceLandmarker's
        # __del__ blocks forever on a serial-dispatcher future, which shows up
        # as the GUI hanging after the batch reports complete.
        processor.close()


def _batch_watermark_spec(watermark_text, watermark_position):
    """WatermarkSpec for the batch tab's credit box, or None when it is empty."""
    if not (watermark_text or "").strip():
        return None
    from retouch.watermark import WatermarkSpec

    return WatermarkSpec(text=watermark_text, position=watermark_position or "auto")


def _export_watermarked(job, output_dir, spec):
    """Write watermarked copies of the batch's outputs; returns a log line or ""."""
    if spec is None:
        return ""
    try:
        from retouch.social_crops import find_crop_sources
        from retouch.watermark import WATERMARK_DIRNAME, export_folder, format_summary
    except Exception as e:
        _logger.exception("Watermark skipped: retouch.watermark unavailable: %s", e)
        return f"\nWatermark: skipped (retouch.watermark unavailable: {e})"
    output_dir = Path(output_dir)
    sources = [Path(f.output_path) for f in job.files if f.status == "done" and f.output_path]
    if not sources:
        sources = find_crop_sources(output_dir)
    out_dir = output_dir / WATERMARK_DIRNAME
    try:
        result = export_folder(sources, out_dir, spec)
        return "\n" + format_summary(result, out_dir)
    except Exception as e:
        _logger.exception("Watermark export failed: %s", e)
        return f"\nWatermark: failed ({e})"


def _export_social_crops(job, output_dir, social_crop_formats, watermark=None):
    """Export platform crops for the batch's outputs; returns a log line or "".

    Imports retouch.social_crops lazily so gui.py's/gui_batch.py's own import
    never depends on it. Prefers the output paths this batch actually wrote
    (from on_file_result), falling back to scanning output_dir when none are
    available (e.g. a rerun of only-flagged files with nothing newly written).
    """
    if not social_crop_formats:
        return ""

    try:
        from retouch.social_crops import export_folder, find_crop_sources, format_summary, parse_formats
    except Exception as e:
        _logger.exception("Social crop export skipped: retouch.social_crops unavailable: %s", e)
        return f"\nSocial crops: skipped (retouch.social_crops unavailable: {e})"

    output_dir = Path(output_dir)
    sources = [Path(f.output_path) for f in job.files if f.status == "done" and f.output_path]
    if not sources:
        sources = find_crop_sources(output_dir)

    social_dir = output_dir / "social"
    try:
        formats = parse_formats(list(social_crop_formats))
        result = export_folder(sources, social_dir, formats, watermark=watermark)
        return "\n" + format_summary(result, social_dir)
    except Exception as e:
        _logger.exception("Social crop export failed: %s", e)
        return f"\nSocial crops: failed ({e})"


def _run_batch(processor, input_dir, output_dir, style_type, custom_style_name,
               recipe_name, export_fmt, export_quality, export_res, auto_group,
               generate_sheet, export_zip, prg, only_files=None, social_crop_formats=None,
               watermark_text=None, watermark_position="auto", write_xmp=False):
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

    recipe_label = custom_style_name if style_type == "Use Custom Style" else recipe_name
    review_root = None
    if _REVIEW_AVAILABLE:
        try:
            review_root = review_root_for(
                Path(output_dir) if output_dir else None, Path(input_dir)
            )
        except Exception:
            _logger.warning("Could not resolve review root for %s", input_dir, exc_info=True)
            review_root = None

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

        _write_batch_review_record(
            review_root, source_path, output_path, qa_list, error, input_dir, recipe_label,
        )

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

        try:
            watermark = _batch_watermark_spec(watermark_text, watermark_position)
        except ValueError as e:
            watermark = None
            log = log + f"\nWatermark: skipped ({e})"
        log = log + _export_watermarked(job, output_dir, watermark)
        log = log + _export_social_crops(job, output_dir, social_crop_formats, watermark=watermark)
        review_page_path = _build_batch_review_page(review_root)
        if review_page_path is not None:
            log = f"{log}\nReview page: {review_page_path}"
        if write_xmp:
            log = f"{log}\n{_write_batch_xmp(review_root)}"

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
# Review page wiring (best-effort — never lets a review-page failure fail the
# batch or drop a file result). See docs/plans or REVIEW_SPEC for the shared
# retouch/review_page.py API this targets.
# ---------------------------------------------------------------------------


def _review_qa_entries(qa_list):
    """Normalize a batch_processor qa_list (QAWarning objects or dicts,
    or None) into the plain-dict shape ReviewRecord.qa expects, via
    retouch.review_page's own normalizer so both call sites agree."""
    if not _REVIEW_AVAILABLE:
        return []
    return _review_qa_entries_impl(qa_list)


def _review_relative_name(source_path, input_dir):
    src = Path(source_path)
    try:
        return src.resolve().relative_to(Path(input_dir).resolve()).as_posix()
    except ValueError:
        return src.name


def _write_batch_review_record(review_root, source_path, output_path, qa_list, error,
                                input_dir, recipe_label):
    """Best-effort ReviewRecord write for one batch file. Never raises —
    a review-page problem must not affect the batch's own result or the
    Job Dashboard record it runs alongside."""
    if not _REVIEW_AVAILABLE or review_root is None:
        return
    try:
        src = Path(source_path)
        record = ReviewRecord(
            source=str(src.resolve()),
            output=str(Path(output_path).resolve()) if output_path else None,
            compare=None,  # gui_batch/batch_processor never writes *_compare files
            status="failed" if error is not None else "done",
            recipe=recipe_label or None,
            faces=[],  # not available at this call site (no FaceContext here)
            qa=[] if error is not None else _review_qa_entries(qa_list),
            error=error,
            elapsed_s=None,
            relative=_review_relative_name(source_path, input_dir),
        )
        write_review_record(review_root, record)
    except Exception:
        _logger.warning("Failed to write review record for %s", source_path, exc_info=True)


def _build_batch_review_page(review_root):
    """Best-effort review.html (re)build; returns its path or None."""
    if not _REVIEW_AVAILABLE or review_root is None:
        return None
    try:
        return build_review_page(review_root)
    except Exception:
        _logger.warning("Failed to build review page at %s", review_root, exc_info=True)
        return None


def _write_batch_xmp(review_root):
    """Best-effort XMP export for a finished batch; returns one log line."""
    if not _REVIEW_AVAILABLE or review_root is None:
        return "XMP: skipped (no review records for this batch)"
    try:
        from retouch.xmp_sidecar import format_counts, write_review_xmp
        return format_counts(write_review_xmp(review_root))
    except Exception as e:
        _logger.warning("XMP export failed at %s", review_root, exc_info=True)
        return f"XMP: failed ({e})"


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

"""Live progress tracking for multi-image batch runs (``cli.py``).

A batch of 26 MP portraits takes 27-325 s *per image*, so a bar that only
ticks when a whole image finishes looks frozen for minutes at a time.  This
module turns fine-grained events emitted by the render workers into:

* a tqdm bar measured in **megapixels** (so a 6 MP and a 26 MP image advance
  it proportionally) whose description shows ``images done/total`` + ETA and
  whose postfix lists what each worker is doing right now;
* a periodic "still working on X" line when an image goes quiet for too long;
* an atomically rewritten ``progress.json`` that another tool (or a human with
  ``cat``) can poll;
* a short end-of-run summary.

Events are plain, picklable dicts so they can cross process boundaries via a
``multiprocessing.Manager().Queue()``::

    {"image": str, "kind": "start" | "stage" | "face" | "finish",
     "t": float (time.time()), ...}

    start:  {"worker": int | None}
    stage:  {"stage": str}
    face:   {"index": int, "total": int}
    finish: {"status": str, "seconds": float | None, "faces": int | None,
             "timings": dict | None, "qa": list[dict] | None,
             "megapixels": float | None, **passthrough (out_path, ...)}

Once an image has finished, later non-finish events for it (queue
reordering between worker and parent) are ignored.

Worker side::

    sink = make_event_sink(q, img_path.name)
    emit(q, img_path.name, "start", worker=os.getpid())
    engine.process(img, progress_cb=sink, ...)
    emit(q, img_path.name, "finish", status="done", seconds=dt, ...)

Parent side::

    progress = BatchProgress([(f.name, image_megapixels(f)) for f in files])
    with ProgressReporter(progress, q, output_dir / "progress.json") as rep:
        ... submit work, wait for futures ...
    for line in rep.summary: tqdm.write(line)
"""

from __future__ import annotations

import json
import logging
import os
import queue as _queue
import statistics
import tempfile
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from tqdm import tqdm

# Read once at import (os.umask can only be read by setting it, which is not
# thread-safe once the reporter thread is running).
_UMASK = os.umask(0)
os.umask(_UMASK)

_logger = logging.getLogger(__name__)

__all__ = [
    "SNAPSHOT_VERSION",
    "image_megapixels",
    "make_event_sink",
    "emit",
    "BatchProgress",
    "ProgressReporter",
    "format_duration",
]

SNAPSHOT_VERSION = 1

# Mirrors ``retouch.io.RAW_EXTENSIONS``; used only if that import fails so this
# module stays importable without cv2 (and cheap to import in tests).
_FALLBACK_RAW_EXTENSIONS = frozenset({
    ".raf", ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".dng", ".orf", ".rw2",
    ".pef", ".srw", ".x3f",
})

_TERMINAL = ("done", "skipped", "failed", "qa_fail")

# Finish-event keys consumed explicitly; anything else (``out_path``,
# ``compare_path`` ...) is passed through into the image's snapshot entry.
_FINISH_KNOWN_KEYS = frozenset({
    "image", "kind", "t", "status", "seconds", "faces", "timings", "qa", "megapixels",
})


def _qa_flags(qa: Any) -> List[str]:
    """Human labels for the *flagged* QA entries of a finish event.

    Entries are dicts ``{"detector", "score", "threshold", "flagged",
    "message"}``; plain strings are accepted too (always treated as flagged).
    """
    if not qa:
        return []
    if isinstance(qa, (str, dict)):
        qa = [qa]
    out = []
    for entry in qa:
        if isinstance(entry, dict):
            if not entry.get("flagged"):
                continue
            label = str(entry.get("detector") or "qa")
            score, thr = entry.get("score"), entry.get("threshold")
            if isinstance(score, (int, float)) and isinstance(thr, (int, float)):
                label += f" {score:.3g}/{thr:.3g}"
            out.append(label)
        elif entry:
            out.append(str(entry))
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_duration(seconds: Optional[float]) -> str:
    """Compact human duration: ``45s``, ``3m10s``, ``1h02m``; ``?`` for None."""
    if seconds is None:
        return "?"
    s = int(round(max(0.0, float(seconds))))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _raw_extensions() -> frozenset:
    try:
        from retouch.io import RAW_EXTENSIONS  # lazy: pulls in cv2
        return frozenset(e.lower() for e in RAW_EXTENSIONS)
    except Exception:  # pragma: no cover - io unavailable/being edited
        return _FALLBACK_RAW_EXTENSIONS


def _jsonable(value: Any) -> Any:
    """Coerce *value* into something ``json.dumps`` accepts (numpy scalars,
    Paths, tuples ...).  Unknown objects become ``str``."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    # numpy scalars and similar expose .item()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:
            pass
    return str(value)


def _stem(name: str) -> str:
    base = os.path.basename(str(name))
    stem, _ext = os.path.splitext(base)
    return stem or base


def _classify_status(status: Any) -> Tuple[str, Optional[str]]:
    """Map a worker status string to (state, reason).

    ``"done"`` → done, ``"skipped"`` → skipped, ``"QA_FAIL: ..."`` → qa_fail,
    anything else (``"failed: ..."``, unknown) → failed, matching how
    ``cli.py`` counts non-done/non-skipped results as failures.
    """
    text = "" if status is None else str(status)
    low = text.strip().lower()
    if low == "done":
        return "done", None
    if low.startswith("skip"):
        return "skipped", None
    if low.startswith("qa_fail"):
        return "qa_fail", text
    return "failed", text or "failed"


# ---------------------------------------------------------------------------
# Header-only size probe
# ---------------------------------------------------------------------------

def image_megapixels(path: Union[str, os.PathLike]) -> Optional[float]:
    """Return the image's pixel count in megapixels without decoding it.

    Non-RAW files: ``PIL.Image.open(...).size`` (header only).  RAW files
    (``retouch.io.RAW_EXTENSIONS``): ``rawpy`` ``raw.sizes`` when the optional
    dependency is installed, else ``None``.  Never raises; returns ``None``
    for unreadable/unknown files.
    """
    try:
        p = Path(path)
        if p.suffix.lower() in _raw_extensions():
            try:
                import rawpy  # optional dependency
            except Exception:
                return None
            with rawpy.imread(str(p)) as raw:
                sizes = raw.sizes
                w, h = int(sizes.width), int(sizes.height)
        else:
            from PIL import Image

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with Image.open(p) as im:
                    w, h = im.size
        if w <= 0 or h <= 0:
            return None
        return (w * h) / 1e6
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Worker-side emitters
# ---------------------------------------------------------------------------

def _put(q: Any, event: Dict[str, Any]) -> None:
    try:
        q.put(event)
    except Exception:
        # A dead Manager / closed queue must never break a render.
        pass


def emit(q: Any, image: str, kind: str, **info: Any) -> None:
    """Put one event ``{"image", "kind", "t", **info}`` on *q*; never raises."""
    if q is None:
        return
    try:
        event = dict(info)
        event.update(image=image, kind=kind, t=time.time())
    except Exception:
        return
    _put(q, event)


class _EventSink:
    """Picklable ``progress_cb(kind, info)`` bound to one image and queue."""

    __slots__ = ("q", "image")

    def __init__(self, q: Any, image: str) -> None:
        self.q = q
        self.image = image

    def __call__(self, kind: str, info: Optional[Dict[str, Any]] = None) -> None:
        if self.q is None:
            return
        try:
            event = dict(info) if info else {}
            event.update(image=self.image, kind=kind, t=time.time())
        except Exception:
            return
        _put(self.q, event)

    def __getstate__(self):
        return (self.q, self.image)

    def __setstate__(self, state):
        self.q, self.image = state

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"_EventSink(image={self.image!r})"


def make_event_sink(q: Any, image: str) -> Callable[..., None]:
    """Return a ``progress_cb(kind, info=None)`` that forwards events for
    *image* to *q*.  Any exception (e.g. a shut-down Manager) is swallowed."""
    return _EventSink(q, image)


# ---------------------------------------------------------------------------
# Pure state
# ---------------------------------------------------------------------------

class BatchProgress:
    """Aggregated, I/O-free state of a batch (except :meth:`write_json`).

    Megapixel accounting: skipped images are removed from *both* ``total_mp``
    and ``done_mp`` (and from the throughput estimate), so a resumed batch
    that skips 30 already-rendered files does not report a bogus ETA.
    """

    def __init__(
        self,
        items: Sequence[Tuple[str, Optional[float]]],
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.clock = clock
        self.created_at = float(clock())
        self._first_start: Optional[float] = None
        self._max_running = 0
        known = [float(mp) for _, mp in items if mp is not None and mp > 0]
        self.default_mp = float(statistics.median(known)) if known else 1.0
        self.images: Dict[str, Dict[str, Any]] = {}
        for name, mp in items:
            self._add(str(name), mp)

    # -- internals ---------------------------------------------------------

    def _add(self, name: str, mp: Optional[float]) -> Dict[str, Any]:
        rec = {
            "status": "pending",
            "mp": float(mp) if (mp is not None and mp > 0) else self.default_mp,
            "mp_estimated": not (mp is not None and mp > 0),
            "worker": None,
            "stage": None,
            "stage_started_at": None,
            "face_index": None,
            "faces_done": 0,
            "faces_total": None,
            "started_at": None,
            "last_event_at": None,
            "finished_at": None,
            "seconds": None,
            "faces": None,
            "timings": None,
            "qa": None,
            "reason": None,
        }
        self.images[name] = rec
        return rec

    def _running(self) -> List[Tuple[str, Dict[str, Any]]]:
        return [(n, r) for n, r in self.images.items() if r["status"] == "running"]

    def _processed(self) -> List[Dict[str, Any]]:
        """Finished, non-skipped images with a known positive duration."""
        return [
            r for r in self.images.values()
            if r["status"] in ("done", "failed", "qa_fail")
            and r["seconds"] is not None and r["seconds"] > 0
        ]

    # -- event ingestion ---------------------------------------------------

    def on_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply one event; returns the image's record (None if malformed)."""
        if not isinstance(event, dict) or "image" not in event:
            return None
        name = str(event["image"])
        kind = event.get("kind")
        t = event.get("t")
        try:
            t = float(t) if t is not None else float(self.clock())
        except (TypeError, ValueError):
            t = float(self.clock())
        rec = self.images.get(name)
        if rec is None:
            rec = self._add(name, None)
        if rec["status"] in _TERMINAL and kind != "finish":
            # Late stage/face event after finish (queue reordering): ignore.
            return rec
        rec["last_event_at"] = t

        if kind == "start":
            rec["status"] = "running"
            rec["started_at"] = t
            rec["worker"] = event.get("worker")
            rec["stage"] = None
            rec["stage_started_at"] = t
            if self._first_start is None or t < self._first_start:
                self._first_start = t
        elif kind == "stage":
            self._ensure_running(rec, t)
            rec["stage"] = None if event.get("stage") is None else str(event["stage"])
            rec["stage_started_at"] = t
        elif kind == "face":
            self._ensure_running(rec, t)
            rec["faces_done"] += 1
            rec["face_index"] = event.get("index")
            total = event.get("total")
            if total is not None:
                rec["faces_total"] = total
        elif kind == "finish":
            state, reason = _classify_status(event.get("status"))
            rec["status"] = state
            rec["reason"] = reason
            rec["finished_at"] = t
            secs = event.get("seconds")
            if secs is None and rec["started_at"] is not None:
                secs = t - rec["started_at"]
            try:
                rec["seconds"] = None if secs is None else float(secs)
            except (TypeError, ValueError):
                rec["seconds"] = None
            if event.get("faces") is not None:
                rec["faces"] = event.get("faces")
            if event.get("timings") is not None:
                rec["timings"] = _jsonable(event.get("timings"))
            qa = event.get("qa")
            rec["qa"] = _jsonable(qa) if qa is not None else None
            mp = event.get("megapixels")
            try:
                if mp is not None and float(mp) > 0:
                    rec["mp"] = float(mp)
                    rec["mp_estimated"] = False
            except (TypeError, ValueError):
                pass
            for key, value in event.items():
                if key in _FINISH_KNOWN_KEYS:
                    continue
                target = key if key not in rec else f"finish_{key}"
                rec[target] = _jsonable(value)
        else:
            return rec

        n_running = sum(1 for r in self.images.values() if r["status"] == "running")
        self._max_running = max(self._max_running, n_running)
        return rec

    def _ensure_running(self, rec: Dict[str, Any], t: float) -> None:
        # Tolerate a lost/late "start" event.
        if rec["status"] == "pending":
            rec["status"] = "running"
            rec["started_at"] = t
            if self._first_start is None or t < self._first_start:
                self._first_start = t

    # -- aggregates --------------------------------------------------------

    @property
    def started_at(self) -> float:
        return self._first_start if self._first_start is not None else self.created_at

    @property
    def total_mp(self) -> float:
        return sum(r["mp"] for r in self.images.values() if r["status"] != "skipped")

    @property
    def done_mp(self) -> float:
        return sum(
            r["mp"] for r in self.images.values()
            if r["status"] in ("done", "failed", "qa_fail")
        )

    @property
    def counts(self) -> Dict[str, int]:
        c = {k: 0 for k in ("pending", "running", "done", "skipped", "failed", "qa_fail")}
        for r in self.images.values():
            c[r["status"]] = c.get(r["status"], 0) + 1
        c["finished"] = c["done"] + c["skipped"] + c["failed"] + c["qa_fail"]
        c["total"] = len(self.images)
        return c

    def eta_seconds(self) -> Optional[float]:
        """Seconds until the batch finishes, or None before the first
        non-skipped image has finished.

        Per-worker throughput is ``sum(MP) / sum(seconds)`` over processed
        images (skipped ones excluded on both sides).  It is multiplied by the
        observed parallelism (max concurrently running images, capped by the
        number of images still to do) so ``--workers 4`` is not reported as
        4x slower than it is.
        """
        processed = self._processed()
        if not processed:
            return None
        mp = sum(r["mp"] for r in processed)
        secs = sum(r["seconds"] for r in processed)
        if secs <= 0 or mp <= 0:
            return None
        rate = mp / secs
        remaining = [r for r in self.images.values() if r["status"] in ("pending", "running")]
        if not remaining:
            return 0.0
        remaining_mp = sum(r["mp"] for r in remaining)
        parallel = max(1, min(self._max_running, len(remaining)))
        return remaining_mp / (rate * parallel)

    def active(self, now: Optional[float] = None) -> List[Dict[str, Any]]:
        """Running images: name, stage, faces done/total, elapsed seconds."""
        now = float(self.clock()) if now is None else float(now)
        out = []
        for name, r in self._running():
            out.append({
                "image": name,
                "stage": r["stage"],
                "faces_done": r["faces_done"],
                "faces_total": r["faces_total"],
                "elapsed": None if r["started_at"] is None else max(0.0, now - r["started_at"]),
                "stage_elapsed": (
                    None if r["stage_started_at"] is None
                    else max(0.0, now - r["stage_started_at"])
                ),
                "worker": r["worker"],
            })
        return out

    def stalled(self, now: Optional[float] = None, after: float = 90.0) -> List[Dict[str, Any]]:
        """Running images whose last event is more than *after* seconds old."""
        now = float(self.clock()) if now is None else float(now)
        out = []
        for a in self.active(now):
            r = self.images[a["image"]]
            last = r["last_event_at"]
            if last is not None and now - last > after:
                a["silent_for"] = now - last
                out.append(a)
        return out

    # -- serialisation -----------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """JSON-serialisable view of the whole batch."""
        now = float(self.clock())
        images = {}
        for name, r in self.images.items():
            d = {k: _jsonable(v) for k, v in r.items()}
            d["elapsed"] = (
                None if r["started_at"] is None
                else (r["seconds"] if r["status"] in _TERMINAL and r["seconds"] is not None
                      else max(0.0, now - r["started_at"]))
            )
            images[name] = d
        return {
            "version": SNAPSHOT_VERSION,
            "started_at": self.started_at,
            "updated_at": now,
            "counts": self.counts,
            "done_mp": round(self.done_mp, 3),
            "total_mp": round(self.total_mp, 3),
            "eta_seconds": self.eta_seconds(),
            "images": images,
        }

    def write_json(self, path: Union[str, os.PathLike]) -> None:
        """Atomically write :meth:`snapshot` to *path* (tmp + ``os.replace``)."""
        path = Path(path)
        data = json.dumps(self.snapshot(), indent=1, default=str)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(data)
            # mkstemp creates 0600; give the progress file normal umask
            # permissions so another user or tool can poll it.
            os.chmod(tmp, 0o666 & ~_UMASK)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # -- human summary -----------------------------------------------------

    def summary_lines(self, slowest: int = 3, max_qa_lines: int = 10) -> List[str]:
        c = self.counts
        parts = [f"{c['done']} done"]
        for key, label in (("skipped", "skipped"), ("failed", "failed"),
                           ("qa_fail", "QA fail"), ("running", "unfinished"),
                           ("pending", "not started")):
            if c[key]:
                parts.append(f"{c[key]} {label}")
        lines = [f"Batch: {c['total']} images — " + ", ".join(parts)]

        finish_times = [r["finished_at"] for r in self.images.values()
                        if r["finished_at"] is not None]
        unfinished = c["running"] or c["pending"]
        end = (max(finish_times) if finish_times and not unfinished
               else float(self.clock()))
        wall = max(0.0, end - self.started_at)
        processed = self._processed()
        timing = f"Total time {format_duration(wall)}"
        if processed:
            secs = sum(r["seconds"] for r in processed)
            mp = sum(r["mp"] for r in processed)
            timing += (f"; avg {secs / len(processed):.1f} s/image, "
                       f"{secs / mp:.1f} s/MP over {len(processed)} processed")
        lines.append(timing)

        ranked = sorted(
            ((n, r) for n, r in self.images.items()
             if r["status"] in ("done", "failed", "qa_fail") and r["seconds"] is not None),
            key=lambda nr: nr[1]["seconds"], reverse=True,
        )[:max(0, slowest)]
        if ranked:
            bits = []
            for n, r in ranked:
                extra = f"{r['mp']:.1f} MP"
                if r["faces"] is not None:
                    extra += f", {r['faces']} face{'s' if r['faces'] != 1 else ''}"
                bits.append(f"{n} {r['seconds']:.1f}s ({extra})")
            lines.append("Slowest: " + "; ".join(bits))

        qa_only: List[str] = []
        for n, r in self.images.items():
            flags = _qa_flags(r["qa"])
            flag_txt = f" [QA: {', '.join(flags)}]" if flags else ""
            if r["status"] == "failed":
                lines.append(f"  FAILED {n}: {r['reason']}{flag_txt}")
            elif r["status"] == "qa_fail":
                lines.append(f"  QA_FAIL {n}: {r['reason']}{flag_txt}")
            elif flags:
                qa_only.append(f"  QA flagged {n}: {', '.join(flags)}")
        # Failures always print; QA notes on delivered images are capped so a
        # 300-image shoot does not bury the summary (all are in the JSON).
        lines.extend(qa_only[:max(0, max_qa_lines)])
        if len(qa_only) > max_qa_lines:
            lines.append(f"  … {len(qa_only) - max_qa_lines} more QA-flagged "
                         "images (see the progress file)")
        return lines


# ---------------------------------------------------------------------------
# Reporter: tqdm bar + drain thread + progress.json + stall lines
# ---------------------------------------------------------------------------

_QUEUE_DEAD_ERRORS = (EOFError, BrokenPipeError, ConnectionError, OSError, ValueError)


class ProgressReporter:
    """Consume events from *q* on a daemon thread and render them.

    Use as a context manager (or call :meth:`close`).  ``close()`` drains any
    remaining events, writes a final ``progress.json``, closes the bar and
    returns :meth:`BatchProgress.summary_lines` (also kept on ``.summary``).
    """

    def __init__(
        self,
        progress: BatchProgress,
        q: Any,
        progress_path: Optional[Union[str, os.PathLike]],
        stall_after: float = 90.0,
        write: Callable[[str], Any] = tqdm.write,
        disable: bool = False,
        poll_interval: float = 0.5,
        json_interval: float = 1.0,
    ) -> None:
        self.progress = progress
        self.q = q
        self.progress_path = Path(progress_path) if progress_path else None
        self.stall_after = float(stall_after)
        self.write = write
        self.poll_interval = float(poll_interval)
        self.json_interval = float(json_interval)
        self.summary: Optional[List[str]] = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._queue_dead = False
        self._dirty = True
        self._last_json: Optional[float] = None
        self._last_stall_warn: Dict[str, float] = {}
        self._closed = False
        self.shown_mp = 0.0
        self.bar = tqdm(
            total=progress.total_mp,
            unit="MP",
            disable=disable,
            dynamic_ncols=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.0f}/{total:.0f} MP [{elapsed}{postfix}]",
        )
        self._render()
        self._thread = threading.Thread(target=self._run, name="batch-progress", daemon=True)
        self._thread.start()

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> "ProgressReporter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- public ------------------------------------------------------------

    def handle(self, event: Dict[str, Any]) -> None:
        """Apply one event in the caller's thread (e.g. the serial path)."""
        with self._lock:
            self.progress.on_event(event)
            self._dirty = True
            self._render()

    def close(self) -> List[str]:
        if self._closed:
            return self.summary or []
        self._closed = True
        self._stop.set()
        self._thread.join(timeout=5.0)
        with self._lock:
            self._drain_nowait()
            self._render()
            self._write_json(force=True)
            try:
                self.bar.close()
            except Exception:
                pass
            self.summary = self.progress.summary_lines()
        return self.summary

    # -- thread ------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._step()
            except Exception:  # never let the UI thread die silently
                _logger.exception("batch progress reporter step failed")
                self._stop.wait(self.poll_interval)

    def _step(self) -> None:
        event = None
        if self._queue_dead or self.q is None:
            self._stop.wait(self.poll_interval)
        else:
            try:
                event = self.q.get(timeout=self.poll_interval)
            except _queue.Empty:
                event = None
            except _QUEUE_DEAD_ERRORS:
                self._queue_dead = True
        with self._lock:
            if event is not None:
                self.progress.on_event(event)
                self._dirty = True
            self._render()
            self._check_stalls()
            self._write_json(force=False)

    def _drain_nowait(self) -> None:
        if self._queue_dead or self.q is None:
            return
        while True:
            try:
                event = self.q.get_nowait()
            except _queue.Empty:
                return
            except Exception:
                self._queue_dead = True
                return
            self.progress.on_event(event)
            self._dirty = True

    # -- rendering ---------------------------------------------------------

    def _render(self) -> None:
        p = self.progress
        try:
            total = p.total_mp
            done = p.done_mp
            if self.bar.total != total:
                self.bar.total = total
            # Track our own position: a disabled tqdm ignores update().
            delta = done - self.shown_mp
            if delta:
                self.shown_mp = done
                self.bar.update(delta)
            c = p.counts
            eta = p.eta_seconds()
            desc = f"Retouching {c['finished']}/{c['total']} img"
            desc += f" ETA {format_duration(eta)}" if eta is not None else " ETA ?"
            self.bar.set_description_str(desc, refresh=False)
            self.bar.set_postfix_str(self._postfix(), refresh=True)
        except Exception:
            _logger.debug("progress bar render failed", exc_info=True)

    def _postfix(self) -> str:
        bits = []
        for a in self.progress.active():
            s = _stem(a["image"])
            if a["stage"]:
                s += f" {a['stage']}"
            if a["faces_total"]:
                s += f" {a['faces_done']}/{a['faces_total']}"
            s += f" {format_duration(a['elapsed'])}"
            bits.append(s)
        return " | ".join(bits)

    def _check_stalls(self) -> None:
        now = float(self.progress.clock())
        for a in self.progress.stalled(now=now, after=self.stall_after):
            name = a["image"]
            last = self._last_stall_warn.get(name)
            if last is not None and now - last < self.stall_after:
                continue
            self._last_stall_warn[name] = now
            stage = a["stage"] or "starting"
            if a["faces_total"]:
                stage += f" (face {a['faces_done']}/{a['faces_total']})"
            dur = a["stage_elapsed"] if a["stage_elapsed"] is not None else a["elapsed"]
            try:
                self.write(f"  … still working on {name}: {stage} for {format_duration(dur)}")
            except Exception:
                pass

    def _write_json(self, force: bool) -> None:
        if self.progress_path is None:
            return
        now = float(self.progress.clock())
        if not force:
            if not self._dirty and self.progress.counts["running"] == 0:
                return
            if self._last_json is not None and now - self._last_json < self.json_interval:
                return
        try:
            self.progress.write_json(self.progress_path)
            self._last_json = now
            self._dirty = False
        except Exception:
            _logger.debug("progress.json write failed", exc_info=True)

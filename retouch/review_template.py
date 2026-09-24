"""Self-contained HTML template for the per-batch review page (``review.html``).

The page is one HTML string with inline CSS and JS and no external requests,
so it works from ``file://``, survives the output folder being moved or
zipped, and is bundled by PyInstaller as ordinary Python source (no data
files). :mod:`retouch.review_page` builds the page data and calls
:func:`render_review_html`; this module only owns presentation.

Page data contract (see ``retouch.review_page.build_page_data``)::

    {"version": 1, "title": str, "batch_id": str, "generated_at": iso str,
     "root": str, "apply_command": str,
     "summary": {"total", "done", "qa_fail", "failed", "skipped", "flagged"},
     "items": [{"key", "name", "status", "recipe", "elapsed_s",
                "before", "after",               # thumb paths relative to the page, or None
                "output_href", "compare_href",   # URL-escaped relative links, or None
                "faces": [{"before", "after"}], "qa_recorded", "flagged",
                "qa": [{"detector", "score", "threshold", "flagged",
                        "message", "available"}],
                "error"}]}

The data is embedded as JSON in ``<script id="review-data"
type="application/json">``. Every ``<`` that could end or comment out that
block is escaped (``</`` -> ``<\\/``, ``<!--`` -> ``\\u003c!--``) and all
item strings reach the DOM through ``textContent`` / ``createElement`` only.

Browser behaviour: grid of cards with filter chips and progress, a detail
view (hold Space / press-and-hold for the before image, ``C`` side-by-side,
face crops, QA table, links), keyboard review (P pick, X reject, U clear,
auto-advance), decisions persisted to ``localStorage`` under
``retouch-review:<batch_id>``, and Export to ``<batch_id>-decisions.json``
(``{"version":1,"batch_id","root","decisions":{key: "pick"|"reject"}}``)
for ``./run review apply``.
"""

from __future__ import annotations

import html
import json
import math
import re
from pathlib import PurePath
from typing import Any, Dict, Mapping

__all__ = ["render_review_html"]

_PLACEHOLDER_RE = re.compile(r"__REVIEW_(TITLE|DATA)__")


def _jsonable(value: Any) -> Any:
    """Return *value* with non-finite floats as ``None`` and paths as strings.

    ``JSON.parse`` rejects ``NaN``/``Infinity``, so a stray ``elapsed_s=nan``
    would otherwise blank the whole page.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, PurePath):
        return str(value)
    return value


def _json_for_script(data: Mapping[str, Any]) -> str:
    """Serialise *data* for embedding inside a ``<script>`` data block.

    ``ensure_ascii`` keeps undecodable-filename surrogates from breaking the
    UTF-8 write. ``</`` becomes ``<\\/`` (valid JSON escape of ``/``) so no
    ``</script>`` can close the block; ``<!--`` becomes ``\\u003c!--`` so the
    HTML parser never enters the script-data-escaped state.
    """
    text = json.dumps(_jsonable(data), ensure_ascii=True, default=str,
                      separators=(",", ":"))
    return text.replace("</", "<\\/").replace("<!--", "\\u003c!--")


def render_review_html(data: Dict[str, Any]) -> str:
    """Render the self-contained review page for *data*.

    Args:
        data: Page data dict per the module docstring's contract. Missing
            keys are tolerated by the page (it renders an empty grid).

    Returns:
        The complete HTML document as a string (UTF-8 safe; the embedded
        JSON is pure ASCII).

    Raises:
        TypeError: if *data* is not a mapping.
    """
    if not isinstance(data, Mapping):
        raise TypeError(f"render_review_html expects a dict, got {type(data).__name__}")
    title = str(data.get("title") or "Batch review")
    title = title.encode("utf-8", "replace").decode("utf-8")
    subs = {
        "TITLE": html.escape(title, quote=True),
        "DATA": _json_for_script(data),
    }
    # Single pass so a title or item string containing a placeholder token
    # is never substituted a second time.
    return _PLACEHOLDER_RE.sub(lambda m: subs[m.group(1)], _TEMPLATE)


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="light dark">
<meta name="robots" content="noindex">
<title>__REVIEW_TITLE__</title>
<style>
:root {
  color-scheme: light dark;
  --bg: #f1f1f1;
  --surface: #ffffff;
  --surface-2: #e8e8e8;
  --well: #e3e3e3;
  --stage: #d9d9d9;
  --text: #1a1a1a;
  --muted: #5f5f5f;
  --faint: #8c8c8c;
  --line: #d4d4d4;
  --line-strong: #bdbdbd;
  --accent: #2563eb;
  --accent-ink: #ffffff;
  --accent-soft: rgba(37, 99, 235, 0.12);
  --pick: #15803d;
  --pick-ink: #ffffff;
  --pick-soft: rgba(21, 128, 61, 0.12);
  --reject: #d42f2f;
  --reject-ink: #ffffff;
  --reject-soft: rgba(212, 47, 47, 0.10);
  --flag: #e5a50a;
  --flag-ink: #2e2100;
  --flag-text: #8a5d00;
  --flag-soft: rgba(229, 165, 10, 0.15);
  --caption-bg: rgba(20, 20, 20, 0.62);
  --caption-ink: #f5f5f5;
  --shadow: 0 1px 2px rgba(0, 0, 0, 0.06), 0 2px 8px rgba(0, 0, 0, 0.05);
  --overlay: rgba(10, 10, 10, 0.45);
  --radius: 10px;
  --gutter: 24px;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #141414;
    --surface: #1d1d1d;
    --surface-2: #272727;
    --well: #232323;
    --stage: #0d0d0d;
    --text: #ececec;
    --muted: #a3a3a3;
    --faint: #767676;
    --line: #2d2d2d;
    --line-strong: #3d3d3d;
    --accent: #5b8def;
    --accent-ink: #0b0b0b;
    --accent-soft: rgba(91, 141, 239, 0.16);
    --pick: #3fb872;
    --pick-ink: #06150c;
    --pick-soft: rgba(63, 184, 114, 0.14);
    --reject: #ef5b5b;
    --reject-ink: #1a0505;
    --reject-soft: rgba(239, 91, 91, 0.13);
    --flag: #f0b429;
    --flag-ink: #231900;
    --flag-text: #f3c55a;
    --flag-soft: rgba(240, 180, 41, 0.13);
    --caption-bg: rgba(0, 0, 0, 0.62);
    --shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
    --overlay: rgba(0, 0, 0, 0.6);
  }
}
@media (max-width: 640px) { :root { --gutter: 16px; } }

*, *::before, *::after { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  overflow-x: hidden;
}
body.no-scroll { overflow: hidden; }
button { font: inherit; color: inherit; }
a { color: var(--accent); }
[hidden] { display: none !important; }
kbd {
  display: inline-block;
  min-width: 1.6em;
  padding: 0 5px;
  border: 1px solid var(--line-strong);
  border-bottom-width: 2px;
  border-radius: 4px;
  font: 600 11px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  text-align: center;
  color: var(--muted);
  background: var(--surface);
}
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

/* ---------- buttons ---------- */
.btn {
  display: inline-flex; align-items: center; justify-content: center; gap: 6px;
  height: 36px; padding: 0 14px;
  border: 1px solid var(--line-strong); border-radius: 8px;
  background: var(--surface); cursor: pointer; white-space: nowrap;
  font-weight: 550; -webkit-tap-highlight-color: transparent;
}
.btn:hover { background: var(--surface-2); }
.btn.primary { background: var(--accent); border-color: var(--accent); color: var(--accent-ink); }
.btn.primary:hover { filter: brightness(1.07); }
.btn.icon { width: 36px; padding: 0; }
.btn.ghost { border-color: transparent; background: transparent; }
.btn.ghost:hover { background: var(--surface-2); }
.btn:disabled { opacity: 0.45; cursor: default; }

/* ---------- header ---------- */
.top {
  position: sticky; top: 0; z-index: 5;
  background: var(--bg);
  border-bottom: 1px solid var(--line);
  padding: 14px var(--gutter) 12px;
}
.top-row { display: flex; align-items: flex-start; gap: 12px; }
.titles { flex: 1; min-width: 0; }
h1 {
  margin: 0; font-size: 18px; font-weight: 650; letter-spacing: -0.01em;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.summary { margin: 2px 0 0; color: var(--muted); font-size: 13px; font-variant-numeric: tabular-nums; }
.summary .warn { color: var(--flag-text); font-weight: 600; }
.summary .bad { color: var(--reject); font-weight: 600; }
.top-actions { display: flex; gap: 8px; flex: none; }
.progress-row { display: flex; align-items: center; gap: 12px; margin: 12px 0 10px; }
.bar {
  flex: 1; height: 6px; border-radius: 3px; background: var(--surface-2);
  overflow: hidden; display: flex; min-width: 60px;
}
.bar span { display: block; height: 100%; transition: width 0.2s ease; }
.bar-pick { background: var(--pick); }
.bar-reject { background: var(--reject); }
.progress-text { color: var(--muted); font-size: 13px; font-variant-numeric: tabular-nums; white-space: nowrap; }
.progress-text b { color: var(--text); font-weight: 650; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip {
  display: inline-flex; align-items: center; gap: 6px;
  height: 30px; padding: 0 11px; border-radius: 15px;
  border: 1px solid var(--line-strong); background: transparent; cursor: pointer;
  font-size: 13px; font-weight: 550; color: var(--muted);
  -webkit-tap-highlight-color: transparent;
}
.chip:hover { color: var(--text); border-color: var(--faint); }
.chip .n { font-variant-numeric: tabular-nums; color: var(--faint); font-weight: 600; }
.chip[aria-pressed="true"] { background: var(--text); border-color: var(--text); color: var(--bg); }
.chip[aria-pressed="true"] .n { color: var(--bg); opacity: 0.7; }
.chip .dot { width: 7px; height: 7px; border-radius: 50%; }
.hint-shortcuts { margin-left: auto; align-self: center; color: var(--faint); font-size: 12px; white-space: nowrap; }
@media (hover: none) { .hint-shortcuts { display: none; } }
.notice {
  margin: 10px 0 0; padding: 8px 12px; border-radius: 8px;
  background: var(--flag-soft); color: var(--flag-text); font-size: 13px;
}

/* ---------- grid ---------- */
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
  gap: 14px;
  padding: 18px var(--gutter) 48px;
}
@media (max-width: 640px) {
  .top { position: static; padding-top: 12px; }
  h1 { font-size: 17px; }
  .summary .when { display: none; }
  .progress-row { flex-wrap: wrap; row-gap: 6px; margin: 10px 0 10px; }
  .progress-text { order: 1; }
  .bar { order: 2; flex-basis: 100%; }
  .chips {
    flex-wrap: nowrap; overflow-x: auto; scrollbar-width: none;
    margin: 0 calc(-1 * var(--gutter)); padding: 0 var(--gutter);
  }
  .chips::-webkit-scrollbar { display: none; }
  .chip { flex: none; }
  .hint-shortcuts { display: none; }
}
@media (max-width: 520px) {
  .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; padding-top: 14px; }
}
.card {
  position: relative; display: block; width: 100%; min-width: 0;
  padding: 0; margin: 0; text-align: left;
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
  box-shadow: var(--shadow); cursor: pointer; overflow: hidden;
  -webkit-tap-highlight-color: transparent;
}
.card:focus-visible { outline: none; }
.card.is-cursor { outline: 2px solid var(--accent); outline-offset: 2px; }
.card[data-decision="pick"] { border-color: var(--pick); box-shadow: inset 0 0 0 1px var(--pick), var(--shadow); }
.card[data-decision="reject"] { border-color: var(--reject); box-shadow: inset 0 0 0 1px var(--reject), var(--shadow); }
.thumb {
  display: block; position: relative; aspect-ratio: 1 / 1; background: var(--well);
}
.thumb img {
  position: absolute; inset: 0; width: 100%; height: 100%;
  object-fit: contain; display: block; transition: opacity 0.15s ease;
}
.card[data-decision="reject"] .thumb img { opacity: 0.35; }
.placeholder {
  position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  padding: 12px; text-align: center; color: var(--faint); font-size: 13px;
}
.mark {
  position: absolute; top: 8px; left: 8px; width: 26px; height: 26px; border-radius: 50%;
  display: none; align-items: center; justify-content: center;
  font-size: 15px; font-weight: 700; line-height: 1;
}
.card[data-decision="pick"] .mark { display: flex; background: var(--pick); color: var(--pick-ink); }
.card[data-decision="reject"] .mark { display: flex; background: var(--reject); color: var(--reject-ink); }
.card[data-decision="pick"] .mark::before { content: "\2713"; }
.card[data-decision="reject"] .mark::before { content: "\2715"; }
.card-badges { position: absolute; top: 8px; right: 8px; display: flex; flex-direction: column; align-items: flex-end; gap: 4px; }
.card-foot { display: block; padding: 8px 10px 9px; border-top: 1px solid var(--line); }
.card-name {
  display: block; font-size: 13px; font-weight: 600;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.card-sub {
  display: block; margin-top: 1px; font-size: 12px; color: var(--muted);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-variant-numeric: tabular-nums;
}
.empty { text-align: center; color: var(--muted); padding: 64px var(--gutter); margin: 0; }

/* ---------- badges ---------- */
.badge {
  display: inline-flex; align-items: center; height: 20px; padding: 0 7px; border-radius: 4px;
  font-size: 11px; font-weight: 700; letter-spacing: 0.02em; text-transform: uppercase; white-space: nowrap;
}
.badge.flag { background: var(--flag); color: var(--flag-ink); }
.badge.fail { background: var(--reject); color: var(--reject-ink); }
.badge.muted { background: var(--caption-bg); color: var(--caption-ink); }
.d-badges .badge.muted { background: var(--surface-2); color: var(--muted); }
.badge.pick { background: var(--pick); color: var(--pick-ink); }
.badge.reject { background: var(--reject); color: var(--reject-ink); }

/* ---------- detail ---------- */
.detail {
  position: fixed; inset: 0; z-index: 20; background: var(--bg);
  display: grid; grid-template-rows: auto minmax(0, 1fr) auto;
  height: 100vh; height: 100dvh;
}
.d-bar {
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  padding: 8px var(--gutter); border-bottom: 1px solid var(--line); background: var(--bg);
  min-height: 52px;
}
.d-title { flex: 1; min-width: 0; display: flex; align-items: baseline; gap: 10px; }
.d-name { font-weight: 650; font-size: 15px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; }
.d-pos { color: var(--muted); font-size: 13px; white-space: nowrap; font-variant-numeric: tabular-nums; }
.d-badges { display: flex; gap: 6px; flex-wrap: wrap; }
.d-body { display: grid; grid-template-columns: minmax(0, 1fr) 360px; min-height: 0; }
.d-stage {
  position: relative; min-height: 0; min-width: 0; background: var(--stage);
  display: flex; gap: 12px; padding: 16px;
  touch-action: pan-y; user-select: none; -webkit-user-select: none; -webkit-touch-callout: none;
}
.pane { position: relative; flex: 1 1 0; min-width: 0; min-height: 0; margin: 0; }
.pane img {
  position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; display: block;
  -webkit-user-drag: none; user-select: none;
}
.pane .img-before { visibility: hidden; }
.pane.show-before .img-before { visibility: visible; }
.pane.show-before .img-after { visibility: hidden; }
.pane figcaption {
  position: absolute; left: 8px; top: 8px; z-index: 1;
  padding: 3px 8px; border-radius: 5px; background: var(--caption-bg); color: var(--caption-ink);
  font-size: 12px; font-weight: 600; pointer-events: none; white-space: nowrap;
  max-width: calc(100% - 16px); overflow: hidden; text-overflow: ellipsis;
}
.pane figcaption .hint { font-weight: 400; opacity: 0.8; }
.pane.show-before figcaption { background: var(--flag); color: var(--flag-ink); }
.hint-touch { display: none; }
@media (hover: none) and (pointer: coarse) { .hint-key { display: none; } .hint-touch { display: inline; } }
.d-empty {
  position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  color: var(--faint); text-align: center; padding: 24px;
}
.d-info { overflow-y: auto; border-left: 1px solid var(--line); padding: 16px 18px 24px; background: var(--bg); }
.d-info section + section { margin-top: 22px; }
.d-info h2 {
  margin: 0 0 8px; font-size: 11px; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--muted);
}
.callout { border-radius: 8px; padding: 10px 12px; margin-bottom: 18px; font-size: 13px; }
.callout.fail { background: var(--reject-soft); border-left: 3px solid var(--reject); }
.callout.flag { background: var(--flag-soft); border-left: 3px solid var(--flag); }
.callout strong { display: block; margin-bottom: 2px; }
.callout .err {
  margin: 6px 0 0; font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  white-space: pre-wrap; overflow-wrap: anywhere; max-height: 180px; overflow: auto;
}
.callout ul { margin: 4px 0 0; padding-left: 18px; }
.faces { display: grid; grid-template-columns: 1fr; gap: 12px; }
.face { margin: 0; }
.face-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }
.face-img { position: relative; aspect-ratio: 1 / 1; background: var(--well); border-radius: 6px; overflow: hidden; }
.face-img img { width: 100%; height: 100%; object-fit: contain; display: block; }
.face-img span {
  position: absolute; left: 5px; bottom: 5px; padding: 1px 6px; border-radius: 4px;
  background: var(--caption-bg); color: var(--caption-ink); font-size: 11px; font-weight: 600;
}
.face-img .none { position: static; background: none; color: var(--faint); display: flex; height: 100%; align-items: center; justify-content: center; }
.muted-note { color: var(--muted); font-size: 13px; margin: 0; }
.qa { width: 100%; border-collapse: collapse; font-size: 13px; }
.qa th {
  text-align: left; font-size: 11px; font-weight: 600; color: var(--faint);
  padding: 0 6px 6px; border-bottom: 1px solid var(--line);
}
.qa td { padding: 7px 6px 0; vertical-align: top; }
.qa tr.msg td { padding-top: 2px; padding-bottom: 8px; color: var(--muted); font-size: 12px; border-bottom: 1px solid var(--line); overflow-wrap: anywhere; }
.qa tbody.flagged td { background: var(--flag-soft); }
.qa tbody.flagged td:first-child { box-shadow: inset 3px 0 0 var(--flag); }
.qa .det { font-weight: 600; }
.qa .score { font-variant-numeric: tabular-nums; white-space: nowrap; }
.qa .score small { color: var(--faint); }
.qa .res { text-align: right; white-space: nowrap; font-weight: 600; font-size: 12px; }
.qa .res.ok { color: var(--muted); font-weight: 500; }
.qa .res.flag { color: var(--flag-text); }
.qa .res.na { color: var(--faint); font-weight: 500; font-style: italic; }
.meter { position: relative; height: 4px; margin-top: 4px; border-radius: 2px; background: var(--surface-2); width: 84px; }
.meter i { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 2px; background: var(--faint); }
.flagged .meter i { background: var(--flag); }
.meter b { position: absolute; top: -3px; bottom: -3px; width: 2px; margin-left: -1px; background: var(--text); border-radius: 1px; }
.links { display: flex; flex-direction: column; gap: 6px; }
.links a { font-weight: 550; text-decoration: none; overflow-wrap: anywhere; }
.links a:hover { text-decoration: underline; }
.meta { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 4px 14px; margin: 12px 0 0; font-size: 13px; }
.meta dt { color: var(--muted); }
.meta dd { margin: 0; overflow-wrap: anywhere; font-variant-numeric: tabular-nums; }

.d-actions {
  position: relative;
  display: flex; align-items: center; justify-content: center; gap: 10px;
  padding: 10px var(--gutter); padding-bottom: max(10px, env(safe-area-inset-bottom));
  border-top: 1px solid var(--line); background: var(--bg);
}
.act { height: 44px; min-width: 120px; padding: 0 18px; font-size: 15px; border-width: 1.5px; }
.act.pick { color: var(--pick); border-color: var(--pick); }
.act.reject { color: var(--reject); border-color: var(--reject); }
.act.clear { min-width: 88px; color: var(--muted); }
.act.pick[aria-pressed="true"] { background: var(--pick); color: var(--pick-ink); }
.act.reject[aria-pressed="true"] { background: var(--reject); color: var(--reject-ink); }
.act kbd { background: transparent; color: inherit; border-color: currentColor; opacity: 0.7; }
.act.flash { animation: flash 0.25s ease; }
@keyframes flash { 0% { transform: scale(0.96); } 100% { transform: scale(1); } }
.nav { width: 44px; height: 44px; padding: 0; font-size: 22px; line-height: 1; }
.nav:first-child { margin-right: 6px; }
#d-next { margin-left: 6px; }
.compare-btn { position: absolute; right: var(--gutter); top: 50%; transform: translateY(-50%); }
.compare-btn[aria-pressed="true"] { background: var(--accent-soft); border-color: var(--accent); color: var(--accent); }

@media (max-width: 899px) {
  .detail { display: block; overflow-y: auto; overscroll-behavior: contain; }
  .d-bar { position: sticky; top: 0; z-index: 2; }
  .d-body { display: block; }
  .d-stage { height: 62vh; height: 62dvh; min-height: 260px; padding: 10px; }
  .d-stage.compare { height: 80vh; height: 80dvh; }
  .d-info { border-left: 0; padding: 18px var(--gutter) 24px; }
  .faces { grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); }
  .d-actions { position: sticky; bottom: 0; z-index: 2; gap: 8px; }
  .act { min-width: 0; flex: 1; padding: 0 8px; }
  .act kbd, .compare-btn { display: none; }
  .nav:first-child, #d-next { margin: 0; }
}
@media (max-width: 699px) and (orientation: portrait) {
  .d-stage.compare { flex-direction: column; }
}

/* ---------- overlays ---------- */
.overlay {
  position: fixed; inset: 0; z-index: 40; background: var(--overlay);
  display: flex; align-items: center; justify-content: center; padding: 16px;
}
.sheet {
  width: 100%; max-width: 520px; max-height: calc(100dvh - 32px); overflow: auto;
  background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
  padding: 20px; box-shadow: 0 12px 40px rgba(0, 0, 0, 0.25);
}
.sheet h2 { margin: 0 0 12px; font-size: 17px; }
.sheet p { margin: 0 0 12px; }
.sheet .foot { display: flex; justify-content: flex-end; gap: 8px; margin-top: 16px; }
.keys { width: 100%; border-collapse: collapse; }
.keys td { padding: 5px 0; vertical-align: top; }
.keys td:first-child { width: 42%; white-space: nowrap; padding-right: 12px; }
.keys tr + tr td { border-top: 1px solid var(--line); }
.cmd {
  display: flex; gap: 8px; align-items: stretch; margin: 6px 0 12px;
}
.cmd textarea {
  flex: 1; min-width: 0; resize: none; height: 64px;
  font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  background: var(--well); color: var(--text); border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px;
}
.cmd .btn { height: auto; }
.small { font-size: 12.5px; color: var(--muted); }
.toast {
  position: fixed; left: 50%; bottom: 84px; z-index: 50; transform: translateX(-50%);
  background: var(--text); color: var(--bg); padding: 8px 14px; border-radius: 8px;
  font-size: 13px; font-weight: 550; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
  pointer-events: none; opacity: 0; transition: opacity 0.15s ease; max-width: calc(100vw - 32px);
}
.toast.on { opacity: 1; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
</head>
<body>
<header class="top" id="top">
  <div class="top-row">
    <div class="titles">
      <h1 id="title"></h1>
      <p class="summary" id="summary"></p>
    </div>
    <div class="top-actions">
      <button type="button" class="btn icon" id="help-btn" aria-label="Keyboard shortcuts" title="Keyboard shortcuts (?)">?</button>
      <button type="button" class="btn primary" id="export-btn" title="Download decisions as JSON">Export</button>
    </div>
  </div>
  <div class="progress-row">
    <div class="bar" aria-hidden="true"><span class="bar-pick" id="bar-pick"></span><span class="bar-reject" id="bar-reject"></span></div>
    <span class="progress-text" id="progress-text" aria-live="polite"></span>
  </div>
  <nav class="chips" id="chips" aria-label="Filter"></nav>
  <p class="notice" id="storage-notice" hidden>This browser is blocking local storage, so decisions will be lost when the page closes. Export before you leave.</p>
</header>
<main class="grid" id="grid" aria-label="Images"></main>
<p class="empty" id="empty" hidden>No images match this filter.</p>

<section class="detail" id="detail" hidden aria-label="Image detail">
  <div class="d-bar">
    <button type="button" class="btn ghost" id="d-back" title="Back to grid (Esc)">&#8249; Grid</button>
    <div class="d-title"><span class="d-name" id="d-name"></span><span class="d-pos" id="d-pos"></span></div>
    <div class="d-badges" id="d-badges"></div>
  </div>
  <div class="d-body">
    <div class="d-stage" id="d-stage">
      <figure class="pane" id="pane-before" hidden>
        <img class="img-solo" id="cmp-before" alt="" draggable="false">
        <figcaption>Before</figcaption>
      </figure>
      <figure class="pane" id="pane-main">
        <img class="img-after" id="d-after" alt="" draggable="false">
        <img class="img-before" id="d-before" alt="" draggable="false">
        <figcaption id="d-caption"></figcaption>
      </figure>
      <div class="d-empty" id="d-empty" hidden>No preview available</div>
    </div>
    <aside class="d-info" id="d-info">
      <div id="d-callout"></div>
      <section><h2>Faces</h2><div class="faces" id="d-faces"></div></section>
      <section><h2>Quality checks</h2><div id="d-qa"></div></section>
      <section><h2>Files</h2><div class="links" id="d-links"></div><dl class="meta" id="d-meta"></dl></section>
    </aside>
  </div>
  <div class="d-actions">
    <button type="button" class="btn nav" id="d-prev" aria-label="Previous image" title="Previous (&#8592; / K)">&#8249;</button>
    <button type="button" class="btn act reject" id="d-reject" aria-pressed="false">Reject <kbd>X</kbd></button>
    <button type="button" class="btn act clear" id="d-clear">Clear <kbd>U</kbd></button>
    <button type="button" class="btn act pick" id="d-pick" aria-pressed="false">Pick <kbd>P</kbd></button>
    <button type="button" class="btn nav" id="d-next" aria-label="Next image" title="Next (&#8594; / J)">&#8250;</button>
    <button type="button" class="btn compare-btn" id="d-compare" aria-pressed="false" title="Side by side (C)">Compare <kbd>C</kbd></button>
  </div>
</section>

<div class="overlay" id="help" hidden role="dialog" aria-modal="true" aria-labelledby="help-title">
  <div class="sheet">
    <h2 id="help-title">Keyboard shortcuts</h2>
    <table class="keys"><tbody>
      <tr><td><kbd>&#8592;</kbd> <kbd>&#8594;</kbd> &nbsp;or&nbsp; <kbd>K</kbd> <kbd>J</kbd></td><td>Previous / next image (within the filter)</td></tr>
      <tr><td><kbd>&#8593;</kbd> <kbd>&#8595;</kbd></td><td>Move by row in the grid</td></tr>
      <tr><td><kbd>Enter</kbd></td><td>Open the selected image</td></tr>
      <tr><td><kbd>P</kbd></td><td>Pick (advances to the next image)</td></tr>
      <tr><td><kbd>X</kbd></td><td>Reject (advances to the next image)</td></tr>
      <tr><td><kbd>U</kbd></td><td>Clear the decision</td></tr>
      <tr><td>Hold <kbd>Space</kbd></td><td>Show the before image (or press and hold the photo)</td></tr>
      <tr><td><kbd>C</kbd></td><td>Side-by-side before | after</td></tr>
      <tr><td><kbd>F</kbd> / <kbd>Shift</kbd>+<kbd>F</kbd></td><td>Next / previous filter</td></tr>
      <tr><td><kbd>Esc</kbd></td><td>Back to the grid</td></tr>
      <tr><td><kbd>?</kbd></td><td>Show or hide this help</td></tr>
    </tbody></table>
    <p class="small" style="margin-top:12px">Decisions are saved in this browser as you go. Export downloads them as a file for <code>./run review apply</code>; nothing is ever deleted.</p>
    <div class="foot"><button type="button" class="btn" id="help-close">Close</button></div>
  </div>
</div>

<div class="overlay" id="export" hidden role="dialog" aria-modal="true" aria-labelledby="export-title">
  <div class="sheet">
    <h2 id="export-title">Decisions exported</h2>
    <p id="export-summary"></p>
    <p class="small" style="margin-bottom:4px">Apply them from a terminal in the retouch folder:</p>
    <div class="cmd">
      <textarea id="export-cmd" readonly spellcheck="false" aria-label="Apply command"></textarea>
      <button type="button" class="btn" id="export-copy">Copy</button>
    </div>
    <p class="small">Picks are copied into <code>picks/</code> next to this page. Rejects stay where they are unless you add <code>--move-rejects</code>, which moves them into <code>rejected/</code>. Nothing is deleted and existing files are never overwritten.</p>
    <div class="foot">
      <button type="button" class="btn" id="export-again">Download again</button>
      <button type="button" class="btn primary" id="export-close">Done</button>
    </div>
  </div>
</div>

<div class="toast" id="toast" role="status" aria-live="polite"></div>

<script id="review-data" type="application/json">__REVIEW_DATA__</script>
<script>
(function () {
  'use strict';

  // ---------------------------------------------------------------- data
  var DATA = {};
  try {
    DATA = JSON.parse(document.getElementById('review-data').textContent || '{}') || {};
  } catch (err) {
    DATA = {};
  }
  var items = [];
  var byKey = Object.create(null);
  (Array.isArray(DATA.items) ? DATA.items : []).forEach(function (it) {
    if (!it || typeof it !== 'object' || typeof it.key !== 'string' || !it.key || byKey[it.key]) return;
    it._i = items.length;
    items.push(it);
    byKey[it.key] = it;
  });
  var BATCH = String(DATA.batch_id || 'default');
  var STORE_KEY = 'retouch-review:' + BATCH;
  var UI_KEY = STORE_KEY + ':ui';

  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = String(text);
    return e;
  }
  function str(v) { return (v === undefined || v === null) ? '' : String(v); }
  function safeUrl(p) {
    // Paths are page-relative; never let one be read as a scheme (javascript:, data:, ...).
    if (typeof p !== 'string' || !p) return null;
    return /^[a-zA-Z][a-zA-Z0-9+.\-]*:/.test(p) ? './' + p : p;
  }
  function isNum(v) { return typeof v === 'number' && isFinite(v); }
  function fmtNum(v) {
    if (!isNum(v)) return '—';
    var a = Math.abs(v);
    return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : v.toFixed(2);
  }
  function fmtSecs(s) {
    if (!isNum(s)) return '';
    if (s < 60) return s.toFixed(s < 10 ? 1 : 0) + ' s';
    var m = Math.floor(s / 60), r = Math.round(s - m * 60);
    return m + ':' + (r < 10 ? '0' : '') + r + ' min';
  }
  function human(det) {
    var s = str(det).replace(/[_\-]+/g, ' ').trim();
    return s ? s.charAt(0).toUpperCase() + s.slice(1) : 'Check';
  }
  function plural(n, word) { return n + ' ' + word + (n === 1 ? '' : 's'); }
  function qaList(it) { return Array.isArray(it.qa) ? it.qa.filter(function (q) { return q && typeof q === 'object'; }) : []; }
  function isFlagged(it) { return !!it.flagged || qaList(it).some(function (q) { return !!q.flagged; }); }
  function isFailed(it) { return it.status === 'failed' || it.status === 'qa_fail'; }
  function mainSrc(it) { return safeUrl(it.after) || safeUrl(it.before); }

  // ---------------------------------------------------------------- storage
  var decisions = Object.create(null);
  var foreign = {};           // stored decisions for keys not on this page (kept, not exported)
  var storageOK = true;
  var dirtySinceExport = false;
  (function probeStorage() {
    try {
      var t = '__retouch_review_probe__';
      window.localStorage.setItem(t, '1');
      window.localStorage.removeItem(t);
    } catch (err) { storageOK = false; }
  })();
  function loadDecisions() {
    try {
      var raw = window.localStorage.getItem(STORE_KEY);
      if (!raw) return;
      var obj = JSON.parse(raw);
      var d = (obj && typeof obj.decisions === 'object' && obj.decisions) || {};
      Object.keys(d).forEach(function (k) {
        if (d[k] !== 'pick' && d[k] !== 'reject') return;
        if (byKey[k]) decisions[k] = d[k]; else foreign[k] = d[k];
      });
    } catch (err) { /* unreadable or blocked storage: start empty */ }
  }
  function saveDecisions() {
    var all = {};
    Object.keys(foreign).forEach(function (k) { all[k] = foreign[k]; });
    Object.keys(decisions).forEach(function (k) { all[k] = decisions[k]; });
    try {
      window.localStorage.setItem(STORE_KEY, JSON.stringify({
        version: 1, batch_id: BATCH, updated_at: new Date().toISOString(), decisions: all
      }));
      storageOK = true;
    } catch (err) { storageOK = false; }
    $('storage-notice').hidden = storageOK;
  }
  function loadUi() {
    try { return JSON.parse(window.localStorage.getItem(UI_KEY) || '{}') || {}; } catch (err) { return {}; }
  }
  function saveUi() {
    try { window.localStorage.setItem(UI_KEY, JSON.stringify({ filter: filterId, compare: compare })); } catch (err) { /* optional */ }
  }

  // ---------------------------------------------------------------- filters
  var FILTERS = [
    { id: 'all', label: 'All', test: function () { return true; } },
    { id: 'flagged', label: 'Flagged', dot: 'var(--flag)', test: isFlagged },
    { id: 'failed', label: 'Failed', dot: 'var(--reject)', test: isFailed },
    { id: 'unreviewed', label: 'Unreviewed', test: function (it) { return !decisions[it.key]; } },
    { id: 'picked', label: 'Picked', dot: 'var(--pick)', test: function (it) { return decisions[it.key] === 'pick'; } },
    { id: 'rejected', label: 'Rejected', dot: 'var(--reject)', test: function (it) { return decisions[it.key] === 'reject'; } }
  ];
  var filterId = 'all';
  function filterById(id) {
    for (var i = 0; i < FILTERS.length; i++) if (FILTERS[i].id === id) return FILTERS[i];
    return FILTERS[0];
  }
  function filteredKeys() {
    var f = filterById(filterId);
    return items.filter(function (it) { return f.test(it); }).map(function (it) { return it.key; });
  }

  // ---------------------------------------------------------------- header
  function renderHeader() {
    var title = str(DATA.title) || 'Batch review';
    $('title').textContent = title;
    $('title').title = title;
    var s = DATA.summary && typeof DATA.summary === 'object' ? DATA.summary : {};
    var total = isNum(s.total) ? s.total : items.length;
    var sum = $('summary');
    sum.textContent = '';
    var parts = [[plural(total, 'image'), '']];
    if (s.qa_fail) parts.push([s.qa_fail + ' blocked by QA', 'bad']);
    if (s.failed) parts.push([s.failed + ' failed', 'bad']);
    if (s.flagged) parts.push([s.flagged + ' flagged', 'warn']);
    if (s.skipped) parts.push([s.skipped + ' skipped', '']);
    var when = DATA.generated_at ? new Date(DATA.generated_at) : null;
    if (when && !isNaN(when.getTime())) {
      parts.push(['generated ' + when.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }), 'when']);
    }
    parts.forEach(function (p, i) {
      var part = el('span', p[1] === 'when' ? 'when' : '');
      if (i) part.appendChild(document.createTextNode(' \u00b7 '));
      part.appendChild(el('span', p[1] === 'when' ? '' : p[1], p[0]));
      sum.appendChild(part);
    });
  }

  var chipEls = {};
  function buildChips() {
    var nav = $('chips');
    FILTERS.forEach(function (f) {
      var b = el('button', 'chip');
      b.type = 'button';
      b.setAttribute('aria-pressed', 'false');
      if (f.dot) { var d = el('span', 'dot'); d.style.background = f.dot; b.appendChild(d); }
      b.appendChild(el('span', '', f.label));
      var n = el('span', 'n', '0');
      b.appendChild(n);
      b.addEventListener('click', function () { setFilter(f.id); });
      chipEls[f.id] = { btn: b, n: n };
      nav.appendChild(b);
    });
    var hint = el('span', 'hint-shortcuts');
    [['Enter', ' open   '], ['P', ' pick   '], ['X', ' reject   '], ['?', ' all shortcuts']].forEach(function (h) {
      hint.appendChild(el('kbd', '', h[0]));
      hint.appendChild(document.createTextNode(h[1].replace(/   $/, '\u2002\u2002')));
    });
    nav.appendChild(hint);
  }
  function updateCounts() {
    FILTERS.forEach(function (f) {
      var c = 0;
      for (var i = 0; i < items.length; i++) if (f.test(items[i])) c++;
      chipEls[f.id].n.textContent = String(c);
      chipEls[f.id].btn.setAttribute('aria-pressed', f.id === filterId ? 'true' : 'false');
    });
    var picked = 0, rejected = 0;
    items.forEach(function (it) {
      if (decisions[it.key] === 'pick') picked++;
      else if (decisions[it.key] === 'reject') rejected++;
    });
    var n = items.length || 1;
    $('bar-pick').style.width = (100 * picked / n) + '%';
    $('bar-reject').style.width = (100 * rejected / n) + '%';
    var pt = $('progress-text');
    pt.textContent = '';
    pt.appendChild(document.createTextNode('Reviewed '));
    pt.appendChild(el('b', '', picked + rejected));
    pt.appendChild(document.createTextNode(' of ' + items.length + ' · ' + picked + ' picked · ' + rejected + ' rejected'));
  }

  // ---------------------------------------------------------------- grid
  var cards = Object.create(null);
  var cursor = null;          // key of the grid's selected card

  function badgesFor(it) {
    var out = [];
    if (it.status === 'failed') out.push(['Failed', 'fail']);
    else if (it.status === 'qa_fail') out.push(['QA fail', 'fail']);
    if (isFlagged(it) && it.status !== 'qa_fail') out.push(['Flagged', 'flag']);
    if (it.status === 'skipped') out.push(['Skipped', 'muted']);
    if (it.qa_recorded === false) out.push(['QA not recorded', 'muted']);
    if (!it.after && it.status !== 'failed' && it.status !== 'qa_fail') out.push(['No output', 'muted']);
    return out.map(function (b) { return el('span', 'badge ' + b[1], b[0]); });
  }

  function cardSub(it) {
    if (it.status === 'failed') return it.error ? str(it.error).split('\n')[0] : 'Processing failed';
    if (it.status === 'qa_fail') {
      var f = qaList(it).filter(function (q) { return q.flagged; }).map(function (q) { return human(q.detector); });
      return f.length ? 'Blocked: ' + f.join(', ') : 'Blocked by QA gate';
    }
    var bits = [];
    if (it.recipe) bits.push(str(it.recipe));
    if (isNum(it.elapsed_s)) bits.push(fmtSecs(it.elapsed_s));
    return bits.join(' · ');
  }

  function buildGrid() {
    var grid = $('grid');
    var frag = document.createDocumentFragment();
    items.forEach(function (it) {
      var card = el('button', 'card');
      card.type = 'button';
      card.dataset.key = it.key;
      card.setAttribute('aria-label', str(it.name) || it.key);
      var thumb = el('span', 'thumb');
      var src = safeUrl(it.grid) || mainSrc(it);
      if (src) {
        var img = el('img');
        img.loading = 'lazy';
        img.decoding = 'async';
        img.alt = '';
        img.draggable = false;
        img.src = src;
        img.addEventListener('error', function () {
          img.remove();
          thumb.appendChild(el('span', 'placeholder', 'Preview missing'));
        });
        thumb.appendChild(img);
      } else {
        thumb.appendChild(el('span', 'placeholder', it.status === 'failed' ? 'No preview' : 'Preview missing'));
      }
      thumb.appendChild(el('span', 'mark'));
      var badges = el('span', 'card-badges');
      badgesFor(it).forEach(function (b) { badges.appendChild(b); });
      thumb.appendChild(badges);
      card.appendChild(thumb);
      var foot = el('span', 'card-foot');
      var name = el('span', 'card-name', str(it.name) || it.key);
      name.title = str(it.name);
      foot.appendChild(name);
      foot.appendChild(el('span', 'card-sub', cardSub(it) || ' '));
      card.appendChild(foot);
      card.addEventListener('click', function () { setCursor(it.key, false); openDetail(it.key); });
      cards[it.key] = card;
      frag.appendChild(card);
    });
    grid.appendChild(frag);
  }

  function paintCard(key) {
    var c = cards[key];
    if (!c) return;
    var d = decisions[key];
    if (d) c.dataset.decision = d; else delete c.dataset.decision;
  }

  function applyFilter() {
    var f = filterById(filterId);
    var visible = 0;
    items.forEach(function (it) {
      var show = f.test(it);
      cards[it.key].hidden = !show;
      if (show) visible++;
    });
    $('empty').hidden = visible > 0;
    $('empty').textContent = items.length ? 'No images match this filter.' : 'This batch has no images to review.';
  }

  function visibleKeys() {
    return items.filter(function (it) { return !cards[it.key].hidden; }).map(function (it) { return it.key; });
  }

  function setCursor(key, scroll) {
    if (cursor && cards[cursor]) cards[cursor].classList.remove('is-cursor');
    cursor = key;
    var c = key && cards[key];
    if (!c) return;
    c.classList.add('is-cursor');
    if (scroll) {
      try { c.focus({ preventScroll: true }); } catch (err) { c.focus(); }
      c.scrollIntoView({ block: 'nearest' });
      // keep the card clear of the sticky header
      var top = $('top').getBoundingClientRect().bottom;
      var r = c.getBoundingClientRect();
      if (r.top < top + 8) window.scrollBy(0, r.top - top - 8);
    }
  }

  function ensureCursor(preferIndex) {
    var vis = visibleKeys();
    if (!vis.length) { setCursor(null); return; }
    if (cursor && vis.indexOf(cursor) >= 0) { setCursor(cursor, false); return; }
    var i = Math.max(0, Math.min(vis.length - 1, preferIndex || 0));
    setCursor(vis[i], false);
  }

  function gridColumns() {
    var cols = getComputedStyle($('grid')).gridTemplateColumns;
    return Math.max(1, cols ? cols.split(' ').filter(Boolean).length : 1);
  }

  function moveCursor(delta) {
    var vis = visibleKeys();
    if (!vis.length) return;
    var i = cursor ? vis.indexOf(cursor) : -1;
    if (i < 0) i = delta > 0 ? -1 : vis.length;
    var j = Math.max(0, Math.min(vis.length - 1, i + delta));
    setCursor(vis[j], true);
  }

  function setFilter(id) {
    var vis = visibleKeys();
    var idx = cursor ? vis.indexOf(cursor) : 0;
    filterId = filterById(id).id;
    applyFilter();
    updateCounts();
    ensureCursor(idx);
    saveUi();
    if (detailOpen) {
      navList = filteredKeys();
      renderDetail();
    }
    toast('Filter: ' + filterById(filterId).label);
  }
  function cycleFilter(dir) {
    var i = 0;
    for (var k = 0; k < FILTERS.length; k++) if (FILTERS[k].id === filterId) i = k;
    setFilter(FILTERS[(i + dir + FILTERS.length) % FILTERS.length].id);
  }

  // ---------------------------------------------------------------- decisions
  function decide(key, value) {
    if (!key || !byKey[key]) return;
    if (value) decisions[key] = value; else delete decisions[key];
    dirtySinceExport = true;
    saveDecisions();
    paintCard(key);
    updateCounts();
    if (!detailOpen) {
      var vis = visibleKeys();
      var idx = vis.indexOf(key);
      applyFilter();
      ensureCursor(idx);
    }
  }

  // ---------------------------------------------------------------- detail
  var detailOpen = false;
  var navList = [];
  var cur = null;
  var compare = false;
  var holdBefore = false;
  var pushedHistory = false;
  var preloadCache = [];

  function navIndex() { return navList.indexOf(cur); }
  function neighbour(dir) {
    var i = navIndex();
    if (i >= 0) {
      var j = i + dir;
      return (j >= 0 && j < navList.length) ? navList[j] : null;
    }
    // current item is outside the list (e.g. filter changed): step by global order
    var gi = byKey[cur] ? byKey[cur]._i : -1;
    var best = null;
    navList.forEach(function (k) {
      var ki = byKey[k]._i;
      if (dir > 0 && ki > gi && (best === null || ki < byKey[best]._i)) best = k;
      if (dir < 0 && ki < gi && (best === null || ki > byKey[best]._i)) best = k;
    });
    return best;
  }

  function openDetail(key, noPush) {
    if (!byKey[key]) return;
    navList = filteredKeys();
    detailOpen = true;
    $('detail').hidden = false;
    document.body.classList.add('no-scroll');
    if (!pushedHistory && !noPush) {
      try { history.pushState({ retouchReview: 1 }, '', '#' + encodeURIComponent(key)); pushedHistory = true; } catch (err) { /* file:// quirks */ }
    }
    show(key);
  }

  function closeDetail(fromPopState) {
    if (!detailOpen) return;
    detailOpen = false;
    setHold(false);
    $('detail').hidden = true;
    document.body.classList.remove('no-scroll');
    var wasPushed = pushedHistory;
    pushedHistory = false;
    try {
      if (wasPushed && !fromPopState) history.back();
      else if (!fromPopState) history.replaceState(null, '', location.pathname + location.search);
    } catch (err) { /* ignore */ }
    var last = cur;
    applyFilter();
    updateCounts();
    var vis = visibleKeys();
    if (last && vis.indexOf(last) >= 0) setCursor(last, true);
    else ensureCursor(0);
    if (cursor && cards[cursor]) setCursor(cursor, true);
  }

  function show(key) {
    cur = key;
    setHold(false);
    renderDetail();
    try { history.replaceState(history.state, '', '#' + encodeURIComponent(key)); } catch (err) { /* ignore */ }
    $('d-info').scrollTop = 0;
    preloadAround();
  }

  function preloadAround() {
    var keep = [];
    [1, -1, 2].forEach(function (d) {
      var k = cur;
      for (var s = 0; s < Math.abs(d) && k; s++) {
        var i = navList.indexOf(k);
        k = i >= 0 ? navList[i + (d > 0 ? 1 : -1)] : null;
      }
      if (!k || !byKey[k]) return;
      var it = byKey[k];
      [it.after, it.before].forEach(function (p) {
        var u = safeUrl(p);
        if (!u) return;
        var im = new Image();
        im.decoding = 'async';
        im.src = u;
        keep.push(im);
      });
    });
    preloadCache = keep;  // hold references so the fetches are not dropped
  }

  function setImg(img, src, alt) {
    if (src) {
      if (img.getAttribute('src') !== src) img.src = src;
      img.alt = alt || '';
      img.hidden = false;
    } else {
      img.removeAttribute('src');
      img.hidden = true;
    }
  }

  function renderDetail() {
    var it = byKey[cur];
    if (!it) return;
    var name = str(it.name) || it.key;
    $('d-name').textContent = name;
    $('d-name').title = name;
    var i = navIndex();
    $('d-pos').textContent = (i >= 0 ? (i + 1) : '–') + ' / ' + navList.length +
      (filterId !== 'all' ? ' · ' + filterById(filterId).label : '');

    var badges = $('d-badges');
    badges.textContent = '';
    var d = decisions[cur];
    if (d) badges.appendChild(el('span', 'badge ' + d, d === 'pick' ? 'Picked' : 'Rejected'));
    badgesFor(it).forEach(function (b) { badges.appendChild(b); });

    // stage
    var after = safeUrl(it.after), before = safeUrl(it.before);
    var stage = $('d-stage');
    var canCompare = !!(after && before);
    var cmp = compare && canCompare;
    stage.classList.toggle('compare', cmp);
    $('pane-before').hidden = !cmp;
    setImg($('cmp-before'), cmp ? before : null, 'Before: ' + name);
    var main = after || before;
    setImg($('d-after'), main, (after ? 'After: ' : 'Before: ') + name);
    setImg($('d-before'), (after && before) ? before : null, 'Before: ' + name);
    $('pane-main').hidden = !main;
    $('d-empty').hidden = !!main;
    renderCaption();
    $('d-compare').disabled = !canCompare;
    $('d-compare').setAttribute('aria-pressed', cmp ? 'true' : 'false');

    // actions
    $('d-pick').setAttribute('aria-pressed', d === 'pick' ? 'true' : 'false');
    $('d-reject').setAttribute('aria-pressed', d === 'reject' ? 'true' : 'false');
    $('d-clear').disabled = !d;
    $('d-prev').disabled = !neighbour(-1);
    $('d-next').disabled = !neighbour(1);

    renderCallout(it);
    renderFaces(it);
    renderQa(it);
    renderFiles(it);
  }

  function renderCallout(it) {
    var box = $('d-callout');
    box.textContent = '';
    if (it.status === 'failed') {
      var c = el('div', 'callout fail');
      c.appendChild(el('strong', '', 'Processing failed'));
      c.appendChild(document.createTextNode('No retouched output was written for this image.'));
      if (it.error) c.appendChild(el('pre', 'err', it.error));
      box.appendChild(c);
    } else if (it.status === 'qa_fail') {
      var q = el('div', 'callout flag');
      q.appendChild(el('strong', '', 'Blocked by the QA gate'));
      q.appendChild(document.createTextNode('The output was not saved because these checks failed:'));
      var ul = el('ul');
      var flagged = qaList(it).filter(function (x) { return x.flagged; });
      flagged.forEach(function (x) {
        var li = el('li');
        li.appendChild(el('b', '', human(x.detector)));
        if (x.message) li.appendChild(document.createTextNode(' — ' + str(x.message)));
        ul.appendChild(li);
      });
      if (!flagged.length) ul.appendChild(el('li', '', 'No reasons were recorded.'));
      q.appendChild(ul);
      if (it.error) q.appendChild(el('pre', 'err', it.error));
      box.appendChild(q);
    } else if (it.error) {
      var e = el('div', 'callout fail');
      e.appendChild(el('strong', '', 'Note'));
      e.appendChild(el('pre', 'err', it.error));
      box.appendChild(e);
    }
  }

  function faceImg(src, label, missing) {
    var w = el('div', 'face-img');
    var u = safeUrl(src);
    if (u) {
      var img = el('img');
      img.loading = 'lazy';
      img.decoding = 'async';
      img.alt = label;
      img.src = u;
      img.draggable = false;
      w.appendChild(img);
      w.appendChild(el('span', '', label));
    } else {
      w.appendChild(el('span', 'none', missing || (label + ' unavailable')));
    }
    return w;
  }

  function renderFaces(it) {
    var box = $('d-faces');
    box.textContent = '';
    var faces = Array.isArray(it.faces) ? it.faces.filter(function (f) { return f && (f.before || f.after); }) : [];
    if (!faces.length) {
      box.appendChild(el('p', 'muted-note',
        it.qa_recorded === false ? 'Not recorded for this image.' :
        it.status === 'failed' ? 'Not available (processing failed).' : 'No faces detected.'));
      return;
    }
    faces.forEach(function (f, i) {
      var fig = el('figure', 'face');
      var pair = el('div', 'face-pair');
      pair.appendChild(faceImg(f.before, 'Before'));
      pair.appendChild(faceImg(f.after, 'After', it.after ? '' : 'No retouched output'));
      fig.appendChild(pair);
      if (faces.length > 1) {
        var cap = el('figcaption', 'small', 'Face ' + (i + 1));
        cap.style.marginTop = '4px';
        fig.appendChild(cap);
      }
      box.appendChild(fig);
    });
  }

  function renderQa(it) {
    var box = $('d-qa');
    box.textContent = '';
    if (it.qa_recorded === false) {
      box.appendChild(el('p', 'muted-note', 'QA not recorded — this image was added from an earlier batch, so no checks are available.'));
      return;
    }
    var qa = qaList(it);
    if (!qa.length) {
      box.appendChild(el('p', 'muted-note', 'No quality checks ran for this image.'));
      return;
    }
    var nFlag = qa.filter(function (q) { return q.flagged; }).length;
    var nNa = qa.filter(function (q) { return q.available === false; }).length;
    var lead = nFlag ? plural(nFlag, 'check') + ' flagged' : 'All checks passed';
    if (nNa) lead += ' · ' + nNa + ' could not run';
    var p = el('p', 'muted-note', lead);
    p.style.marginBottom = '8px';
    box.appendChild(p);

    var table = el('table', 'qa');
    var thead = el('thead');
    var hr = el('tr');
    ['Check', 'Score / limit', ''].forEach(function (h) { hr.appendChild(el('th', '', h)); });
    thead.appendChild(hr);
    table.appendChild(thead);
    // flagged first, then the rest in their original order
    var ordered = qa.map(function (q, i) { return [q, i]; }).sort(function (a, b) {
      return (b[0].flagged ? 1 : 0) - (a[0].flagged ? 1 : 0) || a[1] - b[1];
    });
    ordered.forEach(function (pair) {
      var q = pair[0];
      var na = q.available === false;
      var tb = el('tbody', q.flagged ? 'flagged' : '');
      var tr = el('tr');
      tr.appendChild(el('td', 'det', human(q.detector)));
      var sc = el('td', 'score');
      if (na) {
        sc.textContent = '—';
      } else {
        sc.appendChild(document.createTextNode(fmtNum(q.score)));
        if (isNum(q.threshold)) sc.appendChild(el('small', '', ' / ' + fmtNum(q.threshold)));
        if (isNum(q.score)) {
          var scale = Math.max(1, Math.abs(q.score), isNum(q.threshold) ? Math.abs(q.threshold) : 0);
          var m = el('div', 'meter');
          var fill = el('i');
          fill.style.width = Math.max(0, Math.min(100, 100 * q.score / scale)) + '%';
          m.appendChild(fill);
          if (isNum(q.threshold)) {
            var tick = el('b');
            tick.style.left = Math.max(0, Math.min(100, 100 * q.threshold / scale)) + '%';
            m.appendChild(tick);
          }
          sc.appendChild(m);
        }
      }
      tr.appendChild(sc);
      tr.appendChild(el('td', 'res ' + (na ? 'na' : q.flagged ? 'flag' : 'ok'),
        na ? 'Could not check' : q.flagged ? 'Flagged' : 'OK'));
      tb.appendChild(tr);
      var mr = el('tr', 'msg');
      var md = el('td', '', str(q.message) || (na ? 'Detector unavailable for this image.' : ''));
      md.colSpan = 3;
      mr.appendChild(md);
      tb.appendChild(mr);
      table.appendChild(tb);
    });
    box.appendChild(table);
  }

  function renderFiles(it) {
    var links = $('d-links');
    links.textContent = '';
    function link(href, label) {
      var u = safeUrl(href);
      if (!u) return;
      var a = el('a', '', label + ' ↗');
      a.href = u;
      a.target = '_blank';
      a.rel = 'noopener';
      links.appendChild(a);
    }
    link(it.output_href, 'Open full-resolution output');
    link(it.compare_href, 'Open compare image');
    if (!links.childNodes.length) links.appendChild(el('p', 'muted-note', 'No output files for this image.'));

    var meta = $('d-meta');
    meta.textContent = '';
    function row(k, v) {
      if (v === '' || v === null || v === undefined) return;
      meta.appendChild(el('dt', '', k));
      meta.appendChild(el('dd', '', v));
    }
    var statusLabel = { done: 'Done', qa_fail: 'Blocked by QA', failed: 'Failed', skipped: 'Skipped' };
    row('Status', statusLabel[it.status] || str(it.status));
    row('Recipe', str(it.recipe));
    row('Time', fmtSecs(it.elapsed_s));
    row('File', str(it.name));
  }

  function go(dir) {
    var k = neighbour(dir);
    if (k) show(k);
    else toast(dir > 0 ? 'Last image in this view' : 'First image in this view');
  }

  function decideAndAdvance(value) {
    if (!cur) return;
    var next = neighbour(1);
    decide(cur, value);
    var btn = $(value === 'pick' ? 'd-pick' : 'd-reject');
    btn.classList.remove('flash');
    void btn.offsetWidth;
    btn.classList.add('flash');
    if (next) show(next);
    else { renderDetail(); toast((value === 'pick' ? 'Picked' : 'Rejected') + ' — last image in this view'); }
  }

  function renderCaption() {
    var it = byKey[cur];
    var cap = $('d-caption');
    cap.textContent = '';
    if (!it) return;
    var after = safeUrl(it.after), before = safeUrl(it.before);
    if (holdBefore) {
      cap.textContent = 'Before';
    } else if (after) {
      cap.appendChild(document.createTextNode('After'));
      if (before && !$('d-stage').classList.contains('compare')) {
        var h = el('span', 'hint');
        h.appendChild(el('span', 'hint-key', ' \u00b7 hold Space for before'));
        h.appendChild(el('span', 'hint-touch', ' \u00b7 hold for before'));
        cap.appendChild(h);
      }
    } else if (before) {
      cap.textContent = 'Before \u00b7 no retouched output';
    }
  }

  function setHold(on) {
    var it = byKey[cur];
    var ok = !!(on && detailOpen && it && safeUrl(it.after) && safeUrl(it.before) &&
                !$('d-stage').classList.contains('compare'));
    if (ok === holdBefore) return;
    holdBefore = ok;
    $('pane-main').classList.toggle('show-before', holdBefore);
    renderCaption();
  }

  function toggleCompare() {
    var it = byKey[cur];
    if (!it || !(safeUrl(it.after) && safeUrl(it.before))) { toast('Compare needs both before and after'); return; }
    compare = !compare;
    saveUi();
    setHold(false);
    renderDetail();
  }

  // press-and-hold (touch/mouse) for before, horizontal swipe for prev/next
  (function wireStage() {
    var stage = $('d-stage');
    var timer = null, sx = 0, sy = 0, active = false, pid = null;
    function clear() { if (timer) { clearTimeout(timer); timer = null; } }
    stage.addEventListener('pointerdown', function (e) {
      if (e.pointerType === 'mouse' && e.button !== 0) return;
      active = true; pid = e.pointerId; sx = e.clientX; sy = e.clientY;
      clear();
      timer = setTimeout(function () { timer = null; setHold(true); }, e.pointerType === 'mouse' ? 0 : 140);
    });
    stage.addEventListener('pointermove', function (e) {
      if (!active || e.pointerId !== pid || !timer) return;
      if (Math.abs(e.clientX - sx) > 10 || Math.abs(e.clientY - sy) > 10) clear();
    });
    function end(e, canSwipe) {
      if (!active || e.pointerId !== pid) return;
      active = false;
      var wasHolding = holdBefore;
      clear();
      setHold(false);
      if (!canSwipe || wasHolding) return;
      var dx = e.clientX - sx, dy = e.clientY - sy;
      if (Math.abs(dx) > 50 && Math.abs(dx) > 1.5 * Math.abs(dy)) go(dx < 0 ? 1 : -1);
    }
    stage.addEventListener('pointerup', function (e) { end(e, true); });
    stage.addEventListener('pointercancel', function (e) { end(e, false); });
    stage.addEventListener('pointerleave', function (e) { if (e.pointerType === 'mouse') end(e, false); });
    stage.addEventListener('contextmenu', function (e) { e.preventDefault(); });
  })();

  // ---------------------------------------------------------------- overlays
  function openOverlay(id) { $(id).hidden = false; }
  function closeOverlay(id) { $(id).hidden = true; }
  function anyOverlay() { return !$('help').hidden || !$('export').hidden; }

  var toastTimer = null;
  function toast(msg) {
    var t = $('toast');
    t.textContent = msg;
    t.classList.add('on');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { t.classList.remove('on'); }, 1400);
  }

  function exportPayload() {
    var out = {};
    items.forEach(function (it) { if (decisions[it.key]) out[it.key] = decisions[it.key]; });
    return { version: 1, batch_id: BATCH, root: DATA.root || null, decisions: out };
  }
  function exportName() { return BATCH + '-decisions.json'; }
  function download() {
    var payload = exportPayload();
    var blob = new Blob([JSON.stringify(payload, null, 2) + '\n'], { type: 'application/json' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = exportName();
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 5000);
    dirtySinceExport = false;
    return payload;
  }
  function doExport() {
    var payload = download();
    var picks = 0, rejects = 0;
    Object.keys(payload.decisions).forEach(function (k) {
      if (payload.decisions[k] === 'pick') picks++; else rejects++;
    });
    var s = $('export-summary');
    s.textContent = '';
    s.appendChild(document.createTextNode('Saved '));
    s.appendChild(el('code', '', exportName()));
    s.appendChild(document.createTextNode(' to your downloads — ' + plural(picks, 'pick') + ', ' +
      plural(rejects, 'reject') + (picks + rejects < items.length ? ', ' + (items.length - picks - rejects) + ' undecided.' : '.')));
    var cmd = str(DATA.apply_command) || ('./run review apply . ~/Downloads/' + exportName());
    $('export-cmd').value = cmd;
    $('export-copy').textContent = 'Copy';
    openOverlay('export');
    $('export-copy').focus();
  }
  function copyCommand() {
    var ta = $('export-cmd');
    var done = function () { $('export-copy').textContent = 'Copied'; };
    var fallback = function () {
      ta.focus(); ta.select();
      try { if (document.execCommand('copy')) done(); } catch (err) { /* user can copy manually */ }
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(ta.value).then(done, fallback);
    } else {
      fallback();
    }
  }

  // ---------------------------------------------------------------- wiring
  $('help-btn').addEventListener('click', function () { openOverlay('help'); });
  $('help-close').addEventListener('click', function () { closeOverlay('help'); });
  $('export-btn').addEventListener('click', doExport);
  $('export-close').addEventListener('click', function () { closeOverlay('export'); });
  $('export-again').addEventListener('click', function () { download(); toast('Downloaded ' + exportName()); });
  $('export-copy').addEventListener('click', copyCommand);
  ['help', 'export'].forEach(function (id) {
    $(id).addEventListener('click', function (e) { if (e.target === $(id)) closeOverlay(id); });
  });
  $('d-back').addEventListener('click', function () { closeDetail(false); });
  $('d-prev').addEventListener('click', function () { go(-1); });
  $('d-next').addEventListener('click', function () { go(1); });
  $('d-pick').addEventListener('click', function () { decideAndAdvance('pick'); });
  $('d-reject').addEventListener('click', function () { decideAndAdvance('reject'); });
  $('d-clear').addEventListener('click', function () { decide(cur, null); renderDetail(); });
  $('d-compare').addEventListener('click', toggleCompare);
  window.addEventListener('popstate', function () { if (detailOpen) closeDetail(true); });
  window.addEventListener('beforeunload', function (e) {
    if (!storageOK && dirtySinceExport && Object.keys(decisions).length) {
      e.preventDefault();
      e.returnValue = '';
    }
  });

  document.addEventListener('keydown', function (e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    var t = e.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) {
      if (e.key === 'Escape') { closeOverlay('export'); closeOverlay('help'); }
      return;
    }
    var k = e.key;
    if (k === '?') { e.preventDefault(); if ($('help').hidden) openOverlay('help'); else closeOverlay('help'); return; }
    if (k === 'Escape') {
      if (!$('help').hidden) closeOverlay('help');
      else if (!$('export').hidden) closeOverlay('export');
      else if (detailOpen) closeDetail(false);
      e.preventDefault();
      return;
    }
    if (anyOverlay()) return;
    var lower = k.length === 1 ? k.toLowerCase() : k;

    if (detailOpen) {
      switch (lower) {
        case 'ArrowRight': case 'j': go(1); break;
        case 'ArrowLeft': case 'k': go(-1); break;
        case 'p': decideAndAdvance('pick'); break;
        case 'x': decideAndAdvance('reject'); break;
        case 'u': decide(cur, null); renderDetail(); break;
        case 'c': toggleCompare(); break;
        case 'f': cycleFilter(e.shiftKey ? -1 : 1); break;
        case ' ': if (!e.repeat) setHold(true); break;
        case 'Enter': break;
        default: return;
      }
      e.preventDefault();
      return;
    }

    switch (lower) {
      case 'ArrowRight': case 'j': moveCursor(1); break;
      case 'ArrowLeft': case 'k': moveCursor(-1); break;
      case 'ArrowDown': moveCursor(gridColumns()); break;
      case 'ArrowUp': moveCursor(-gridColumns()); break;
      case 'Home': moveCursor(-1e9); break;
      case 'End': moveCursor(1e9); break;
      case 'Enter': if (cursor) openDetail(cursor); else { ensureCursor(0); if (cursor) openDetail(cursor); } break;
      case 'p': decide(cursor, 'pick'); break;
      case 'x': decide(cursor, 'reject'); break;
      case 'u': decide(cursor, null); break;
      case 'f': cycleFilter(e.shiftKey ? -1 : 1); break;
      case ' ': break;  // no native button activation / page jump on the grid
      default: return;
    }
    e.preventDefault();
  });
  document.addEventListener('keyup', function (e) {
    if (e.key === ' ' && detailOpen) { setHold(false); e.preventDefault(); }
  });
  window.addEventListener('blur', function () { setHold(false); });

  // ---------------------------------------------------------------- boot
  loadDecisions();
  $('storage-notice').hidden = storageOK;
  var ui = loadUi();
  if (ui && typeof ui.filter === 'string') filterId = filterById(ui.filter).id;
  compare = !!(ui && ui.compare);
  renderHeader();
  buildChips();
  buildGrid();
  items.forEach(function (it) { paintCard(it.key); });
  applyFilter();
  updateCounts();
  ensureCursor(0);
  var hashKey = '';
  try { hashKey = decodeURIComponent(location.hash.replace(/^#/, '')); } catch (err) { hashKey = ''; }
  if (hashKey && byKey[hashKey]) {
    setCursor(hashKey, false);
    openDetail(hashKey, true);
  }
})();
</script>
</body>
</html>
"""

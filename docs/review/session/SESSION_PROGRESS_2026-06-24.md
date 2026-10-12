# Session Progress Report — 2026-06-24

**Date:** 2026-06-24
**Session:** GUI layout — restore develop-panel internal scroll
**Branch:** main
**Commits:** 1
**Files changed:** `gui.py` (+9), `GUI.md` (+18)
**Tests:** 116/116 GUI tests pass

---

## Issue

User reported that the **Develop Adjustments** panel (right column of the Single Photo Editor) was clipping content at the bottom with no way to scroll to the remaining accordion sections (Color Transfer, Debug & Mask Preview) and the second "Apply Overrides & Process" button.

## Root Cause

A CSS specificity conflict between two rules in the embedded stylesheet:

| Rule | Location | Specificity | Effect |
|---|---|---|---|
| `.gradio-container .column { overflow: visible !important; }` | Line 906 (Overflow fix) | (0, 2, 0) | Forces all Gradio columns to `overflow: visible` |
| `.develop-panel { overflow-y: auto !important; }` | Line 743 | (0, 1, 0) | Intended to make the right column scroll |

Both use `!important`, so specificity decides the winner. The blanket "Overflow fix" rule has higher specificity, so the develop-panel's `overflow-y: auto` was silently being overridden to `overflow: visible`. The `max-height: 84vh` still applied, so content past 84vh overflowed visibly but was clipped by the parent row — no scrollbar was rendered.

## Fix

Added a higher-specificity rule immediately after the Overflow fix block:

```css
.gradio-container .column.develop-panel {
    overflow-y: auto !important;
    overflow-x: hidden !important;
}
```

Specificity: (0, 3, 0) — three class selectors. This beats the (0, 2, 0) blanket rule and restores the internal scrollbar.

## Documentation

Updated `GUI.md §2.3` with a callout explaining the specificity conflict and the guard rule, so future CSS edits to columns know not to disturb the (0, 3, 0) minimum.

## Verification

- `python3 -m py_compile gui.py` — OK
- `python3 -m pytest tests/test_gui.py -v` — 116 passed, 0 failed

## Files Touched

| Status | File | Change |
|---|---|---|
| M | `gui.py` | +9 lines (CSS rule + comment) |
| M | `GUI.md` | +18 lines (specificity callout) |

**End of session. Single commit, no behavior changes beyond restoring the intended scroll.**

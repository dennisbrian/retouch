"""Per-face "what was skipped" captions from a render's diagnostics.

The engine collects, per face, the reasons an op or the whole face was
skipped or dampened (``runtime_diagnostics["face_notes"]``, built in
``_stage_per_face`` and ``_finalize_face_notes``) and detections the person
gate dropped (``runtime_diagnostics["frame_notes"]``). This module turns
them into short lines for the GUI ("Face 2: eye edits skipped, eye closed")
and a comparable summary for the preview-vs-export parity check.

Faces are named by detection order in *this* render, plus a position
anchor (``anchor_bbox_norm``), so the preview and the export can be matched
even if their detections are ordered differently.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

# Actions shown in the caption. "review" (Safe Auto's verdict when it is
# advisory and the settings were applied as set) goes to the edit report
# only: it would sit on most faces and changes nothing.
CAPTION_ACTIONS = ("skipped", "dampened", "boosted", "kept")

_OP_LABELS = {
    "eye_edits": "eye edits",
    "face_strength": "face retouch",
    "protect": "protected area",
    "highlight_repair": "Blown Highlight Repair",
    "spot_heal": "Spot Healing",
    "hair_remove_flyaways": "Stray Hair Cleanup",
    "micro_restore": "texture restore",
    "skin_warmth": "Skin Warmth",
    "neck_tone_match": "Neck Tone Match",
    "cross_region_skin": "body skin match (P7)",
    "safe_auto": "Safe Auto",
    "detection": "face",
}


def op_label(op: str) -> str:
    return _OP_LABELS.get(op, op.replace("_", " "))


def _note_text(note: Mapping[str, Any]) -> str:
    op, action, reason = note.get("op", ""), note.get("action", ""), note.get("reason", "")
    if op == "face_strength":
        return str(reason)
    if op == "protect":
        return "protected area kept"
    return f"{op_label(op)} {action}, {reason}" if reason else f"{op_label(op)} {action}"


def caption_lines(diagnostics: Optional[Mapping[str, Any]]) -> List[str]:
    """One short line per face that met a skip/dampen, plus dropped faces."""
    diagnostics = diagnostics or {}
    lines: List[str] = []
    for entry in diagnostics.get("face_notes") or []:
        texts: List[str] = []
        for note in entry.get("notes") or []:
            if note.get("action") not in CAPTION_ACTIONS:
                continue
            text = _note_text(note)
            if text not in texts:
                texts.append(text)
        if texts:
            lines.append(f"Face {entry.get('face')}: " + "; ".join(texts))
    for note in diagnostics.get("frame_notes") or []:
        if note.get("action") == "skipped":
            x, y, w, h = (note.get("bbox_norm") or [0, 0, 0, 0])[:4]
            lines.append(
                f"Face at {round((x + w / 2) * 100)}% across, "
                f"{round((y + h / 2) * 100)}% down: {note.get('reason')}"
            )
    return lines


def caption_markdown(diagnostics: Optional[Mapping[str, Any]]) -> str:
    lines = caption_lines(diagnostics)
    if not lines:
        return ""
    return "**Skipped or dampened:**\n" + "\n".join(f"- {line}" for line in lines)


def _iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def borderline_sides(entry: Mapping[str, Any]) -> set:
    """Eye sides whose gate decision sat next to its threshold."""
    return {n.get("side") for n in entry.get("notes") or [] if n.get("borderline")}


def note_keys(entry: Mapping[str, Any], skip_sides: Iterable[str] = ()) -> Tuple[Tuple[str, str, str], ...]:
    """(op, action, side) of one face's shown notes, order-free, for parity.

    Only caption actions count (advisory Safe Auto reviews are evidence, not
    a change to the render); ``skip_sides`` drops eye notes for sides whose
    decision was borderline in either render.
    """
    skip = set(skip_sides)
    return tuple(sorted({
        (n.get("op", ""), n.get("action", ""), n.get("side") or "")
        for n in entry.get("notes") or []
        if n.get("action") in CAPTION_ACTIONS
        and not (n.get("op") == "eye_edits" and n.get("side") in skip)
    }))


def match_faces(
    a: Iterable[Mapping[str, Any]], b: Iterable[Mapping[str, Any]], min_iou: float = 0.3,
) -> List[Tuple[Mapping[str, Any], Optional[Mapping[str, Any]]]]:
    """Pair face entries of two renders by anchor position (best IoU)."""
    b_left = list(b)
    pairs = []
    for ea in a:
        best, best_iou = None, min_iou
        for eb in b_left:
            iou = _iou(ea.get("anchor_bbox_norm") or (0, 0, 0, 0),
                       eb.get("anchor_bbox_norm") or (0, 0, 0, 0))
            if iou >= best_iou:
                best, best_iou = eb, iou
        if best is not None:
            b_left.remove(best)
        pairs.append((ea, best))
    return pairs


def report_section(diagnostics: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """JSON-safe per-face notes for the edit report."""
    diagnostics = diagnostics or {}
    return {
        "faces": [
            {
                "face": e.get("face"),
                "position": e.get("anchor_bbox_norm"),
                "recipe": e.get("recipe"),
                "notes": [dict(n) for n in e.get("notes") or []],
            }
            for e in diagnostics.get("face_notes") or []
        ],
        "frame": [dict(n) for n in diagnostics.get("frame_notes") or []],
        "caption": caption_lines(diagnostics),
    }

#!/usr/bin/env python3
"""Core library for the Autonomous Image Quality & Regression Lab.

Deterministic. Heavy deps (cv2, numpy, retouch engine) are imported lazily
so the CLI stays stdlib-only for corpus/policy operations.

Subcommands (wired in quality_lab.py):
    corpus   --build-synthetic   build the seed synthetic corpus + manifest
    baseline                   render + fingerprint the stable reference
    run                        render candidate + full metric report
    compare                    diff a candidate report against a baseline
    report                     human summary of a report/comparison JSON
    benchmark                  performance timings + RAM vs baseline

Layout:
    quality_lab/corpus_manifest.json   tracked — case registry (tags, inputs)
    quality_lab/thresholds.json        tracked — triage policy (review-gated)
    test_output/quality_lab/<case>/{input.png,baseline/<recipe>.png,candidate/<recipe>.png}
    test_output/quality_lab/reports/*.json

Exit codes: 0 ok (or no REGRESSION for compare), 1 REGRESSION found / check
failure, 2 setup error.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LAB_DIR = ROOT / "test_output" / "quality_lab"          # gitignored: renders + reports
POLICY_DIR = ROOT / "quality_lab"                       # tracked: corpus policy
CORPUS_DIR = LAB_DIR / "corpus"                         # corpus images (untracked)
MANIFEST_PATH = POLICY_DIR / "corpus_manifest.json"     # tracked
REPORTS_DIR = LAB_DIR / "reports"
THRESHOLDS_PATH = POLICY_DIR / "thresholds.json"        # tracked, review-gated

DEFAULT_RECIPES = ["natural", "cosplay_clear_v1"]
# Recipes that exist in retouch.recipes.RECIPES are validated at runtime.

# ---------------------------------------------------------------------------
# Threshold policy (reviewable engineering data — never auto-loosened)
# ---------------------------------------------------------------------------
# One entry per check: (warn, regress). None = not applicable at that level.
# Units are documented per check in metrics.py docstrings. Direction matters:
# `low`  → value BELOW warn/regress is bad (e.g. texture retention)
# `high` → value ABOVE warn/regress is bad (e.g. chroma drift)
THRESHOLDS: Dict[str, Dict[str, Any]] = {
    # identity: how close candidate output is to baseline output (0..1, low=bad)
    "identity_ssim": {"warn": 0.995, "regress": 0.985, "direction": "low"},
    "identity_deltaE_mean": {"warn": 0.8, "regress": 2.0, "direction": "high"},
    # whole-image drift between INPUT and output (absolute quality, tone-relative)
    "luminance_drift": {"warn": 6.0, "regress": 12.0, "direction": "high"},   # ΔL mean
    "chroma_drift": {"warn": 5.0, "regress": 10.0, "direction": "high"},      # ΔE76 mean on skin
    "highlight_clip_delta": {"warn": 1.5, "regress": 3.0, "direction": "high"},  # × input clip frac
    "shadow_clip_delta": {"warn": 1.5, "regress": 3.0, "direction": "high"},
    # per-region (skin/eyes/hair/lips): texture + detail retention (0..1, low=bad)
    "region_texture_retention": {"warn": 0.75, "regress": 0.55, "direction": "low"},
    "region_luminance_drift": {"warn": 8.0, "regress": 15.0, "direction": "high"},
    "region_chroma_drift": {"warn": 6.0, "regress": 12.0, "direction": "high"},
    "edge_preservation": {"warn": 0.80, "regress": 0.65, "direction": "low"},
    # performance
    "perf_time_ratio": {"warn": 1.25, "regress": 1.75, "direction": "high"},
    "perf_ram_ratio": {"warn": 1.25, "regress": 2.0, "direction": "high"},
}

# ---------------------------------------------------------------------------
# PR-aware impact map: changed path prefix -> corpus tag filter.
# Evaluated most-specific-first; first match wins. "full" = every tag.
# ---------------------------------------------------------------------------
IMPACT_TAG_MAP: List[Tuple[str, List[str]]] = [
    ("retouch/eye", ["eyes", "face"]),
    ("retouch/red_eye", ["eyes", "face"]),
    ("retouch/lens_glare", ["eyes", "glasses", "face"]),
    ("retouch/lips", ["lips", "face"]),
    ("retouch/teeth", ["lips", "face"]),
    ("retouch/powder_finish", ["skin", "makeup", "face_paint", "face"]),
    ("retouch/nose_", ["skin", "face"]),
    ("retouch/undereye", ["skin", "face"]),
    ("retouch/spot_heal", ["skin", "blemish", "face"]),
    ("retouch/blemish", ["skin", "blemish", "face"]),
    ("retouch/skin", ["skin", "face"]),
    ("retouch/specular", ["skin", "makeup", "face"]),
    ("retouch/parsing", ["wig", "hair", "face"]),
    ("retouch/detection", ["full"]),
    ("retouch/hair", ["hair", "wig"]),
    ("retouch/grading", ["full"]),
    ("retouch/style", ["full"]),
    ("retouch/lut", ["full"]),
    ("retouch/engine", ["full"]),
    ("retouch/params", ["full"]),
    ("retouch/presets", ["full"]),
    ("presets/", ["full"]),
    ("retouch/", ["face", "skin"]),   # any other engine module: moderate sweep
    ("tests/", ["none"]),
    ("docs/", ["none"]),
    ("scripts/dev/quality_lab/", ["full"]),  # the lab itself: exercise everything
    ("scripts/", ["none"]),
    (".github/", ["none"]),
]

# Tags/prefixes that are infrastructure — if the PR touches ONLY these,
# the visual suite has nothing to do.
NONE_ONLY_PREFIXES = ("tests/", "docs/", "scripts/", ".github/")

# Recipes exercised per risk level for `run --pr-aware`.
RISK_RECIPES = {
    "LOW": ["natural"],
    "MEDIUM": ["natural", "cosplay_clear_v1"],
    "HIGH": ["natural", "cosplay_clear_v1", "portrait"],
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _np_cv2():
    import cv2  # noqa
    import numpy as np  # noqa
    return np, cv2


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
        return bool(out)
    except Exception:
        return True


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _engine_versions() -> Dict[str, str]:
    np_, cv2_ = _np_cv2()
    versions = {"cv2": cv2_.__version__, "numpy": np_.__version__}
    try:
        import mediapipe as mp  # noqa
        versions["mediapipe"] = mp.__version__
    except Exception:
        versions["mediapipe"] = "unavailable"
    return versions


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _load_manifest() -> Dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise SystemExit(
            f"corpus manifest missing: {MANIFEST_PATH}\n"
            "run: scripts/dev/quality-lab corpus --build-synthetic"
        )
    return json.loads(MANIFEST_PATH.read_text())


def load_thresholds() -> Dict[str, Dict[str, Any]]:
    """Active thresholds = built-in policy, optionally overridden by the
    committed thresholds.json. The file is reviewable; loosening it is a
    reviewable change. `run`/`compare` never write it."""
    policy = dict(THRESHOLDS)
    if THRESHOLDS_PATH.exists():
        override = json.loads(THRESHOLDS_PATH.read_text())
        for key, spec in override.items():
            if key in policy:
                policy[key] = {**policy[key], **spec}
            else:
                policy[key] = spec
    return policy


def check_threshold_change(base_ref: str = "origin/main") -> Dict[str, Any]:
    """Detect loosening of thresholds vs a git ref. Returns a report; CI can
    block on it. A threshold change that raises `warn`/`regress` (for
    direction=high) or lowers them (direction=low) is 'loosened'."""
    if not THRESHOLDS_PATH.exists():
        return {"changed": False, "loosened": [], "reason": "no thresholds.json"}
    try:
        old_raw = subprocess.check_output(
            ["git", "show", f"{base_ref}:{THRESHOLDS_PATH.relative_to(ROOT)}"],
            cwd=ROOT, text=True,
        )
        old = json.loads(old_raw)
    except Exception:
        return {"changed": True, "loosened": [], "reason": "new thresholds file (review it)"}
    new = json.loads(THRESHOLDS_PATH.read_text())
    loosened = []
    for key, spec in new.items():
        if key not in THRESHOLDS or key not in old:
            continue
        direction = THRESHOLDS[key]["direction"]
        for level in ("warn", "regress"):
            if level not in spec or old[key].get(level) is None:
                continue
            old_v, new_v = old[key][level], spec[level]
            bad = (new_v > old_v) if direction == "high" else (new_v < old_v)
            if bad:
                loosened.append({"check": key, "level": level,
                                 "old": old_v, "new": new_v})
    return {"changed": new != old, "loosened": loosened}


# ---------------------------------------------------------------------------
# Case selection (PR-aware)
# ---------------------------------------------------------------------------

def select_cases(manifest: Mapping[str, Any],
                 tags: Optional[Sequence[str]] = None,
                 changed_files: Optional[Sequence[str]] = None,
                 risk: str = "MEDIUM") -> Tuple[List[Dict[str, Any]], List[str]]:
    """Pick cases by explicit tags, or by mapping changed files -> tags.
    Returns (cases, active_tags). 'full' = all cases; 'none' = skip."""
    cases = list(manifest["cases"])
    if tags is None:
        changed = list(changed_files or [])
        active: List[str] = []
        for path in changed:
            for prefix, mapped in IMPACT_TAG_MAP:
                if path.startswith(prefix):
                    for t in mapped:
                        if t not in active:
                            active.append(t)
                    break
        if not active:
            if changed and all(p.startswith(NONE_ONLY_PREFIXES) for p in changed):
                return [], ["none"]
            active = ["face", "skin"]  # safe default for unknown changes
        if "none" in active and len(active) > 1:
            active = [t for t in active if t != "none"]
    else:
        active = list(tags)
    if "none" in active:
        return [], active
    if "full" in active:
        return cases, active
    selected = [c for c in cases if set(c.get("tags", [])) & set(active)]
    return selected, active


# ---------------------------------------------------------------------------
# Synthetic seed corpus
# ---------------------------------------------------------------------------

def build_synthetic_corpus(overwrite: bool = False) -> Dict[str, Any]:
    """Generate a small tagged synthetic corpus (no real photos).

    Images are procedural portrait stand-ins (oval face, eyes, lips, hair,
    textured skin) over a gradient background — enough to exercise region
    metrics and catch catastrophic regressions. Real-photo corpus cases are
    added by the owner via `corpus --add` (not implemented: drop files in +
    edit manifest.json), since private photos must not be committed.
    """
    np, cv2 = _np_cv2()
    rng = np.random.default_rng(20260928)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    POLICY_DIR.mkdir(parents=True, exist_ok=True)

    def portrait(skin_l: float, skin_chroma: float, *, face_paint: bool = False,
                 wig: bool = False, glasses: bool = False, eyes_closed: bool = False,
                 highlights: bool = False, low_light: bool = False,
                 texture: float = 1.0, blemishes: int = 0, w: int = 512, h: int = 640):
        img = np.zeros((h, w, 3), np.float32)
        # background gradient
        bg = np.linspace(0.25, 0.55, h, dtype=np.float32)[:, None, None]
        img[:] = bg * np.array([0.9, 1.0, 1.05], np.float32)
        cx, cy, fw, fh = w // 2, int(h * 0.42), int(w * 0.30), int(h * 0.26)
        Y, X = np.ogrid[:h, :w]
        face = ((X - cx) / fw) ** 2 + ((Y - cy) / fh) ** 2 <= 1.0
        # skin in Lab-ish BGR
        lab_skin = np.array([skin_l, 128 + skin_chroma * 0.55, 128 + skin_chroma], np.float32)
        bgr_skin = cv2.cvtColor(lab_skin.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_Lab2BGR).reshape(3).astype(np.float32)
        if face_paint:
            bgr_skin = np.array([230, 230, 235], np.float32)
        img[face] = bgr_skin
        # skin texture (speckle)
        if texture > 0:
            noise = rng.normal(0, 4.0 * texture, (h, w, 1)).astype(np.float32)
            img[face] += noise[face]
        # hair / wig: cap over top of head
        hair = ((X - cx) / (fw * 1.15)) ** 2 + ((Y - (cy - fh * 0.55)) / (fh * 0.75)) ** 2 <= 1.0
        hair &= Y < cy
        hair_color = np.array([235, 235, 240], np.float32) if wig else np.array([30, 40, 60], np.float32)
        img[hair] = hair_color
        # eyes
        eye_y, eye_dx = cy - int(fh * 0.15), int(fw * 0.38)
        for ex in (cx - eye_dx, cx + eye_dx):
            if eyes_closed:
                cv2.line(img, (ex - 12, eye_y), (ex + 12, eye_y), (60, 70, 90), 3)
            else:
                cv2.ellipse(img, (ex, eye_y), (14, 9), 0, 0, 360, (240, 240, 245), -1)
                cv2.circle(img, (ex, eye_y), 5, (70, 90, 120), -1)
        if glasses:
            for ex in (cx - eye_dx, cx + eye_dx):
                cv2.circle(img, (ex, eye_y), 22, (20, 20, 20), 2)
                cv2.circle(img, (ex - 6, eye_y - 6), 6, (255, 255, 255), -1)  # glare
            cv2.line(img, (cx - eye_dx + 22, eye_y), (cx + eye_dx - 22, eye_y), (20, 20, 20), 2)
        # lips
        cv2.ellipse(img, (cx, cy + int(fh * 0.42)), (int(fw * 0.32), int(fh * 0.10)),
                    0, 0, 360, (90, 90, 170), -1)
        # blemishes
        for _ in range(blemishes):
            bx = int(rng.integers(cx - fw // 2, cx + fw // 2))
            by = int(rng.integers(cy, cy + fh // 2))
            cv2.circle(img, (bx, by), 4, (60, 70, 110), -1)
        if highlights:
            hl = ((X - (cx - fw * 0.45)) / 30.0) ** 2 + ((Y - (cy - fh * 0.3)) / 22.0) ** 2 <= 1.0
            img[hl] = np.minimum(img[hl] + 90, 255)
        if low_light:
            img *= 0.35
        return np.clip(img, 0, 255).astype(np.uint8)

    specs = [
        # id, tags, kwargs
        ("synthetic-light-skin", ["face", "skin"], dict(skin_l=190, skin_chroma=18)),
        ("synthetic-dark-skin", ["face", "skin"], dict(skin_l=95, skin_chroma=26)),
        ("synthetic-face-paint", ["face", "skin", "makeup", "face_paint"],
         dict(skin_l=200, skin_chroma=5, face_paint=True)),
        ("synthetic-wig-glasses", ["face", "skin", "wig", "glasses", "eyes"],
         dict(skin_l=180, skin_chroma=16, wig=True, glasses=True)),
        ("synthetic-closed-eyes", ["face", "skin", "eyes"],
         dict(skin_l=170, skin_chroma=20, eyes_closed=True)),
        ("synthetic-blemish-texture", ["face", "skin", "blemish"],
         dict(skin_l=170, skin_chroma=20, texture=2.0, blemishes=8)),
        ("synthetic-harsh-highlight", ["face", "skin", "makeup"],
         dict(skin_l=180, skin_chroma=18, highlights=True)),
        ("synthetic-low-light", ["face", "skin"],
         dict(skin_l=150, skin_chroma=22, low_light=True)),
        # Real detectable face (MediaPipe landmarks), exercises the full
        # face-path region QA. Module import is lazy so corpus ops stay light.
        ("synthetic-real-face", ["face", "skin", "eyes"], None),
    ]

    cases = []
    for case_id, tags, kw in specs:
        case_dir = LAB_DIR / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        input_path = case_dir / "input.png"
        if input_path.exists() and not overwrite:
            cases.append({"id": case_id, "tags": tags, "input": str(input_path.relative_to(ROOT))})
            continue
        if kw is None:
            from tests.golden_face_fixture import make_synthetic_face_image
            img = make_synthetic_face_image(512, 640)
        else:
            img = portrait(**kw)
        cv2.imwrite(str(input_path), img)
        cases.append({"id": case_id, "tags": tags, "input": str(input_path.relative_to(ROOT))})

    manifest = {
        "version": 1,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "Synthetic seed corpus. Real photos: drop into test_output/quality_lab/<id>/input.png and append a case entry; do NOT commit private images.",
        "cases": cases,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return manifest

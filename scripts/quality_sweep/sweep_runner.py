"""Parameter-sweep runner for the face retouch pipeline.

Scores each configuration on:
  - golden_gate : does the code still reproduce the golden snapshots at
                  BASELINE params? (a per-RUN precondition, not per-config;
                  see NOTE below)
  - perceptual  : mean similarity vs the baseline-config reference render
  - runtime     : mean seconds per image (only trustworthy when measured
                  serially -- see --serial-timing)

NOTE on the golden gate: golden snapshots are byte-exact SHA-256 hashes
locked to baseline parameters. ANY swept parameter change necessarily
produces different bytes, so "does config X match the golden hash" would
be FAIL for every non-baseline config and carries no signal. Instead the
gate is evaluated once per run against the unmodified engine, and each
config is additionally checked for *hard regressions* (crash, NaN, degenerate
or identity output) which is the per-config failure signal that matters.

Sweeps NEVER mutate source: every parameter is passed as a keyword override
to RetouchEngine.process(), which accepts all recipe keys directly.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_CORPUS = Path(
    "/Users/dennis/Library/CloudStorage/GoogleDrive-alexanderlooi88@gmail.com"
    "/My Drive/Photography/Remielle/Photos/Total"
)


# ---------------------------------------------------------------- corpus

def _face_crop_and_mask(img):
    """Crop to the primary face and build its skin mask.

    Face-local edits are diluted to noise when measured over a full frame
    (whole-frame SSIM stayed at 0.9998 for genuinely different configs), so
    both the corpus and the metrics are restricted to the face region.
    """
    from retouch.detection import FaceDetector
    from retouch.parsing import FaceParser
    faces = FaceDetector().detect(img)
    if not faces:
        return None, None
    x, y, w, h = faces[0].bbox
    pad = int(max(w, h) * 1.1)
    cx, cy = x + w // 2, y + h // 2
    crop = img[max(0, cy - pad):cy + pad, max(0, cx - pad):cx + pad]
    cfaces = FaceDetector().detect(crop)
    if not cfaces:
        return crop, None
    parser = FaceParser()
    parser._sess = None  # landmark-only: deterministic, ONNX-free
    regions = parser._landmark_fallback_only(
        cfaces[0].landmarks, crop, person_mask=None, ied=cfaces[0].ied)
    m = getattr(regions, "skin", None)
    return crop, (np.asarray(m) > 0.5) if m is not None else None


def load_corpus(corpus_dir: Path, downscale: float, limit: int | None):
    """Return [(name, face_crop, skin_mask)]."""
    files = sorted(corpus_dir.glob("*.jpg"))
    if limit:
        files = files[:limit]
    out = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        if downscale != 1.0:
            img = cv2.resize(img, (0, 0), fx=downscale, fy=downscale,
                             interpolation=cv2.INTER_AREA)
        crop, mask = _face_crop_and_mask(img)
        if crop is None:
            continue
        out.append((f.name, crop, mask))
    return out


# ------------------------------------------------------------ perceptual

def _gray_f32(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def ssim(a: np.ndarray, b: np.ndarray, mask=None) -> float:
    """SSIM on luma (Wang et al. 2004), 11x11 Gaussian, no deps.

    When ``mask`` is given the SSIM map is averaged over the mask only.
    """
    A, B = _gray_f32(a), _gray_f32(b)
    if A.shape != B.shape:
        return float("nan")
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    win, sig = (11, 11), 1.5
    mu_a = cv2.GaussianBlur(A, win, sig)
    mu_b = cv2.GaussianBlur(B, win, sig)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = cv2.GaussianBlur(A * A, win, sig) - mu_a2
    sb = cv2.GaussianBlur(B * B, win, sig) - mu_b2
    sab = cv2.GaussianBlur(A * B, win, sig) - mu_ab
    num = (2 * mu_ab + C1) * (2 * sab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sa + sb + C2)
    smap = num / den
    if mask is not None and mask.shape == smap.shape and mask.any():
        return float(smap[mask].mean())
    return float(np.mean(smap))


def delta_e_mean(a: np.ndarray, b: np.ndarray, mask=None) -> float:
    """Mean CIE76 dE in Lab -- perceptual color drift vs reference."""
    if a.shape != b.shape:
        return float("nan")
    la = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).astype(np.float32)
    lb = cv2.cvtColor(b, cv2.COLOR_BGR2LAB).astype(np.float32)
    de = np.sqrt(np.sum((la - lb) ** 2, axis=-1))
    if mask is not None and mask.shape == de.shape and mask.any():
        return float(de[mask].mean())
    return float(np.mean(de))


# ----------------------------------------------------------- golden gate

def golden_gate(engine) -> Dict[str, Any]:
    """Verify the engine at BASELINE params still reproduces both golden
    snapshot sets. Computed in-process -- deliberately does NOT invoke
    pytest, because the golden tests WRITE their snapshot JSON on missing
    keys, which is unsafe under parallel slices."""
    sys.path.insert(0, str(REPO / "tests"))
    from test_golden_pipeline import _make_synthetic_image
    from golden_face_fixture import make_face_context, make_synthetic_face_image

    def h(x):
        return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()[:16]

    res: Dict[str, Any] = {"base": {}, "face": {}, "ok": True}

    base_snaps = json.loads((REPO / "tests" / "golden_pipeline_snapshots.json").read_text())
    img = _make_synthetic_image()
    for name, exp in base_snaps.items():
        got = h(engine.process(img, recipe=name))
        res["base"][name] = {"expected": exp, "got": got, "match": got == exp}
        res["ok"] &= got == exp

    face_snaps = json.loads((REPO / "tests" / "golden_pipeline_face_snapshots.json").read_text())
    fimg = make_synthetic_face_image()
    fh, fw = fimg.shape[:2]
    fctx = make_face_context(fw, fh, fimg)
    for name, exp in face_snaps.items():
        got = h(engine.process(fimg, recipe=name, face_contexts=[fctx]))
        res["face"][name] = {"expected": exp, "got": got, "match": got == exp}
        res["ok"] &= got == exp
    return res


# ------------------------------------------------------------ param space

def build_space(spec: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = sorted(spec)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(spec[k] for k in keys))]


def config_id(cfg: Dict[str, Any]) -> str:
    return "|".join(f"{k}={cfg[k]}" for k in sorted(cfg))


# ---------------------------------------------------------------- scoring

def hard_regression(out: np.ndarray, inp: np.ndarray) -> str | None:
    """Per-config failure signal: the things that are always wrong."""
    a = np.asarray(out)
    if a is None or a.size == 0:
        return "empty output"
    if a.shape != inp.shape:
        return f"shape changed {inp.shape}->{a.shape}"
    if not np.isfinite(a.astype(np.float32)).all():
        return "non-finite values"
    if np.array_equal(a, inp):
        return "identity (no-op)"
    if float(a.std()) < 1.0:
        return f"degenerate flat output (std={a.std():.3f})"
    return None


def score_config(engine, cfg, corpus, refs, recipe) -> Dict[str, Any]:
    ssims, des, times, fails = [], [], [], []
    for name, img, mask in corpus:
        kw = dict(cfg)
        t0 = time.perf_counter()
        try:
            out = np.asarray(engine.process(img, recipe=recipe, **kw))
        except Exception as e:
            fails.append(f"{name}: {type(e).__name__}: {e}")
            continue
        times.append(time.perf_counter() - t0)
        bad = hard_regression(out, img)
        if bad:
            fails.append(f"{name}: {bad}")
            continue
        ref = refs[name]
        ssims.append(ssim(out, ref, mask))
        des.append(delta_e_mean(out, ref, mask))
    return {
        "config_id": config_id(cfg),
        "params": cfg,
        "recipe": recipe,
        "n_images": len(corpus),
        "n_failures": len(fails),
        "failures": fails[:5],
        "ssim_face_vs_baseline": round(float(np.mean(ssims)), 6) if ssims else None,
        "delta_e_face_vs_baseline": round(float(np.mean(des)), 4) if des else None,
        "runtime_s_mean": round(float(np.mean(times)), 4) if times else None,
        "runtime_s_total": round(float(np.sum(times)), 4) if times else None,
    }


# ------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", type=str, help="JSON file describing the parameter space")
    ap.add_argument("--slice", type=int, default=0, help="this slice index")
    ap.add_argument("--num-slices", type=int, default=1)
    ap.add_argument("--out", type=str, required=True, help="result JSON path")
    ap.add_argument("--recipe", type=str, default="natural")
    ap.add_argument("--corpus", type=str, default=str(DEFAULT_CORPUS))
    ap.add_argument("--downscale", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-gate", action="store_true")
    args = ap.parse_args()

    from retouch.engine import RetouchEngine
    engine = RetouchEngine()

    corpus = load_corpus(Path(args.corpus), args.downscale, args.limit)
    if not corpus:
        raise SystemExit(f"no images found in {args.corpus}")

    gate = None if args.skip_gate else golden_gate(engine)

    # Baseline reference renders = recipe at stock params, no overrides.
    refs, base_times = {}, []
    for name, img, _mask in corpus:
        t0 = time.perf_counter()
        refs[name] = np.asarray(engine.process(img, recipe=args.recipe))
        base_times.append(time.perf_counter() - t0)

    spec = json.loads(Path(args.space).read_text()) if args.space else {}
    configs = build_space(spec) if spec else [{}]
    mine = configs[args.slice::args.num_slices]

    rows = [score_config(engine, cfg, corpus, refs, args.recipe) for cfg in mine]

    payload = {
        "slice": args.slice,
        "num_slices": args.num_slices,
        "recipe": args.recipe,
        "corpus": args.corpus,
        "downscale": args.downscale,
        "n_images": len(corpus),
        "golden_gate": gate,
        "baseline_runtime_s_mean": round(float(np.mean(base_times)), 4),
        "n_configs_total": len(configs),
        "n_configs_this_slice": len(mine),
        "results": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"slice {args.slice}/{args.num_slices}: {len(mine)} configs -> {args.out}")
    if gate is not None:
        print(f"golden gate: {'PASS' if gate['ok'] else 'FAIL'}")


if __name__ == "__main__":
    main()

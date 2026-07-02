# AGENTS.md — Retouch Engine (Python 3.9+ / OpenCV / NumPy)

> Companion docs (load on demand): `docs/PRECISION.md` · `docs/PIPELINE.md` · `docs/PERCEPTUAL.md` · `docs/VISUAL_QA.md`

## Philosophy
Reference standard: 像素蛋糕 (PixCake) — skin texture and tone solved simultaneously, not traded.
Hierarchy: **Visual Fidelity > Correctness > Performance > Elegance**
A correct fix that introduces a halo artifact is not a fix. Revert. A fast fix that blurs pores is not an optimization. Revert.

## Critical Rules
- **Python 3.9+**: Type hints required. Explicit imports. Never `except: pass` — catch specific exceptions; log or re-raise. `cv2.error` caught separately from `Exception`.
- **float32 internal**: Never use uint8 intermediates in pixel arithmetic.
- **Colorspace**: Explicit `cv2.cvtColor` at every boundary. Never assume BGR/RGB/LAB/HSV — verify the conversion path. LAB for skin tone ops.
- **Masks**: BiSeNet masks must be float32 [0.0, 1.0] before compositing.
- **Skin**: Never `cv2.GaussianBlur` directly on skin for "smoothing" (destroys texture). Preserve high-frequency detail band from `frequency.py`.
- **Kernel sizing**: Scale with resolution via `KERNEL_SCALE`, never hardcoded.

## Auto-CRITICAL Findings (non-negotiable)
- uint8 intermediate in pixel arithmetic
- Missing `cv2.cvtColor` at colorspace boundary
- `GaussianBlur` on skin pixels for smoothing
- LAB not used for skin tone operations
- BiSeNet mask not float32 [0.0, 1.0] before compositing
- Bare `except: pass` or `except Exception` without logging
- Skin op that doesn't preserve high-frequency detail band
- Any pipeline invariant violation from `docs/PERCEPTUAL.md` §2

## Visual QA (Mandatory for pipeline stages)
- Any change touching `frequency.py`, `skin.py`, `grading.py`, `parsing.py`, `geometry.py` = Visual-Critical. Visual QA gates cannot be bypassed regardless of task size.
- Run pipeline on reference test images. Diff against baseline per `docs/VISUAL_QA.md` gates. Document PASS/FAIL/IMPROVED per gate.
- "Tests pass" without visual confirmation = protocol violation.
- After 2 consecutive visual QA failures for same gate: STUCK, escalate.

## Performance (10M+ pixels)
- No Python loops over pixels. Vectorized NumPy only.
- In-place ops (`out=`, `[:]`) where safe. Delete large temporaries immediately.
- Runtime must not increase >10% on 1080p/4K. Memory peak must not increase >15%.

## Patterns
- Thin UI/entry points (`gui.py`, `cli.py`) — all logic in `retouch/`.
- All parameters in `retouch/params.py` registry. No hardcoded values elsewhere.
- Each pipeline stage = isolated module class. No cross-module calls bypassing engine orchestrator.
- Recipe validation against `recipe_schema.py` mandatory before pipeline entry.

## Security
- Path traversal guard on all GUI/CLI file inputs.
- Gradio threading: engine instances and `FaceContext` caches must be thread-safe.
- No API keys or hardcoded system paths in committed code.

## Complexity
- **SMALL** (typo/<10 lines non-pipeline): lint only, visual QA bypassed.
- **MEDIUM** (bug fix/feature/single-module): full loops + visual QA if blast radius includes pipeline stage.
- **LARGE** (arch/new pipeline stage/cross-cutting): strict loops, precision audit, visual QA mandatory.
- **Visual-Critical** (touches frequency/skin/grading/parsing/geometry): treat as MEDIUM for visual QA even if SMALL by line count.

## Workflow
- One-shot: solve in one response. Ask only if blocked.
- Tag claims `[VERIFIED]` / `[ASSUMED]` / `[STALE]`. Colorspace/dtype claims MUST be verified by reading source — never from memory.
- No "while I'm here" edits. Exception: precision violations (uint8 intermediate, missing colorspace conversion) MAY be fixed but documented as separate finding in REVIEW_OUTPUT.
- After 2 failed attempts: STOP, escalate. Do not invent fixes.

## Verification (run before shipping Python)
1. `python3 -m py_compile path/to/file.py`
2. `python3 -m pytest tests/ -v`
3. `python3 scripts/benchmark.py` (if performance-affecting)
4. Visual QA gates (if pipeline stage touched) — per `docs/VISUAL_QA.md`

## Quota Mode (429/TPM pressure)
Serial execution, reduce context depth, compress summaries. Defer benchmarks/integration suites. **Never defer visual QA for Visual-Critical modules.**

## Directory Map
`retouch/` (pipeline stages) · `presets/` (LUTs/recipes) · `tests/` · `scripts/` · `styles/` · `models/` · `docs/`

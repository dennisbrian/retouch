# Tier 1 Implementation Plan — Foundation Stages (F1–F3)

**Date:** 2026-07-02
**Source:** `PHOTOSHOP_PARITY_ROADMAP.md` Tier 1, turned into executable stages (same format as Stages 0–B).
**Theme:** every stage below is mostly *leverage* — wiring together infrastructure that already exists (`precision.py`, `_F_*` grader variants, `PROCESS_INPUT_KEYS` params dict, `reset_*` GUI pattern, `regions.py` mask/blend toolkit, Gradio ≥4 `ImageEditor`) rather than building new machinery.

---

## Stage F1 — Float32 global pipeline + 16-bit export

**Goal:** eliminate inter-stage uint8 truncation where banding actually originates, and export 16-bit. Raises the quality ceiling for every dark/gradient look (`moonlight_porcelain` class).

**Key insight from exploration:** the per-face Phase 2 work (crops, frequency separation) is not the banding source — the global Phase 3 chain is: `_stage_subject_separation → _stage_global → _stage_grade → _stage_finish`, each stage quantizing to uint8 on exit (`engine.py:1458, 1508, 1745, 1782, 1790, 1819, 1844` + `grade()` returning uint8). `grading.grade()` already computes internally in float via `ensure_float`/`_F_*` — it just re-quantizes on every hand-off.

**Scope decision:** migrate **Phase 3 (global stages) only** to a float32 [0,1] contract. Phase 2 per-face stays uint8 for now (blends through masks, low banding risk, huge migration surface). This gets ~90% of the quality win for ~30% of the risk.

### Steps
1. **Float stage contract:** after `_composite_faces`, convert once via `precision.to_float()`. `_stage_subject_separation`, `_stage_global`, `_stage_grade`, `_stage_finish` accept/return float32 [0,1]. Single `to_uint8()` at `ProcessingResult` build (skipped when 16-bit export requested — see step 4).
2. **Port module-level helpers** used by those stages (`_adjust_contrast`, `_adjust_tonal`, brightness LUT at `engine.py:1521`, sharpen at `:1844`, HSV op at `:1782`) to float variants — follow the existing `_F_*` naming convention in `grading.py`. The brightness LUT becomes a direct `np.power` on float (no 256-entry LUT).
3. **`grade()` float entry:** add `return_float=True` path (or an `_F_grade` wrapper) so `_stage_grade` doesn't round-trip through uint8. Internals already float — only the entry/exit points change.
4. **16-bit export:** in `io.py`, extend `write_image_with_icc()` to accept float32 input + `bit_depth=16` for PNG/TIFF (`cv2.imwrite` with uint16, or PIL for ICC embedding). GUI export-format dropdown gains "TIFF 16-bit" / "PNG 16-bit".
5. **RAW in 16-bit:** `io.py:imread_exif` rawpy postprocess with `output_bps=16`; downscale to uint8 only if the pipeline entry needs it (Phase 2 is uint8) — keep the 16-bit original for the final grade re-render later (defer full linear develop to Tier 3).

### Files
`retouch/engine.py`, `retouch/precision.py` (may gain helpers), `retouch/grading.py`, `retouch/tonal.py` (check `apply_hd_curve` dtype), `retouch/io.py`, `gui.py` (export dropdown), tests.

### Tests / artifact QA
- Per-stage SSIM regression: float output vs current uint8 baseline ≥ 0.99 on the test corpus (catches range bugs; expect *small* diffs — that's the point).
- **cv2 range trap (top risk):** `cvtColor` on float32 expects [0,1] for BGR but LAB L is [0,100] and HSV H is [0,360] in float mode (vs 0–255/0–180 in uint8). Every ported conversion needs an explicit range test.
- Synthetic dark-gradient image through `moonlight_porcelain`: assert unique-value count in shadow region increases vs uint8 baseline (banding metric), and 16-bit export contains >256 distinct levels.
- Full pytest suite; benchmark before/after (float should be ≈neutral; no LUT means brightness slightly slower — acceptable).

**Effort:** ~1–2 weeks. **Do this first** — F2/F3 re-render through this path.

---

## Stage F2 — Session model, undo/redo, snapshots

**Goal:** revisability — the actual reason people call Photoshop "powerful."

**Key leverage:** `gui.py:273` already materializes the complete edit state: `params = dict(zip(PROCESS_INPUT_KEYS, args))`. A **session IS this dict** (+ recipe name + image path + future local-adjustment list). Serialization is nearly free. The `reset_*` functions + `gr.State` (`_original_state` at `gui.py:1300`) establish the pattern for pushing values back into components.

### Steps
1. **`retouch/session.py` (new):** `Session` dataclass = `{version, recipe, params: dict, image_path, created}` with `to_json`/`from_json` + schema validation (reuse `recipe_schema.py` validation approach). Forward-compat: unknown params warn, don't crash.
2. **Save/Load session in GUI:** "💾 Save Session" / "📂 Load Session" buttons → `gr.File` download/upload. Load pushes values into all components — a scaled-up `reset_lch`-style function returning the full `PROCESS_INPUT_KEYS` tuple. **Alignment guard:** extend the existing positional-alignment regression tests (`test_gui.py`) to sessions, since this is the same 1-position-shift failure mode fixed in Stage 0.
3. **Undo/redo:** `gr.State` holding a bounded list of param dicts (~50) + cursor. Push on every Process click; Undo/Redo buttons restore params into components and re-render. Sessions are tiny dicts — memory is trivial; re-render cost is the normal process time.
4. **Snapshots (A/B):** named session saves in a `gr.State` dict + dropdown; "Compare" renders snapshot vs current side-by-side (reuse `make_comparison` in `io.py:165`).
5. **CLI parity:** `cli.py --session file.json` (load params, run) and `--save-session` (write params next to output). Makes any GUI edit batch-reproducible → *this is the killer leverage: tune one photo in GUI, apply the session to a whole shoot via `batch_processor.py`*.

### Files
`retouch/session.py` (new), `gui.py`, `cli.py`, `retouch/batch_processor.py` (accept session dict), `tests/test_session.py` (new), `tests/test_gui.py`.

### Tests
Round-trip (params → JSON → params identical); load-session component-order alignment; undo/redo cursor logic (pure functions, easy); session-vs-GUI render equivalence (same params dict → same output hash); batch-with-session smoke test.

**Effort:** ~1.5–2 weeks. Independent of F1 (merge either order), but renders benefit from F1.

---

## Stage F3 — Brush-painted local adjustments

**Goal:** Lightroom-class local edits (brush + radial + linear), with a differentiator PS doesn't have: intersect painted masks with semantic masks ("only where skin").

### Steps
1. **GUI paint surface:** new "🖌️ Local Adjustments" accordion with `gr.ImageEditor` (brush mode) seeded with the current image. Extract the painted layer's alpha as the raw mask. Up to N (e.g. 4) local adjustments, each: mask + op dropdown + strength slider + optional "restrict to: skin / hair / background / none" (semantic intersect via `combine_masks`, `regions.py:125`).
2. **Mask post-processing:** `feather_mask()` (`regions.py:158`) with radius scaled to image size; downscale-safe storage (masks saved as PNG alongside session).
3. **Engine hook:** new `process()` kwarg `local_adjustments: list[dict]` (`{mask, op, strength}`). New `_stage_local_adjustments` — runs **after `_stage_grade`, before `_stage_finish`** so local exposure/warmth edits sit on top of the look, like LR. Ops v0 (all exist as functions already — this stage only masks them): `exposure`, `warmth` (`_F_adjust_warmth`), `saturation` (`_F_adjust_saturation`), `clarity` (`_F_add_clarity`), `smooth` (guided filter), `dodge`, `burn` (L-channel gain, same math as `_stage_subject_separation`).
4. **Radial/linear generators:** two pure functions in `regions.py` (center/radius/feather; start/end points) → same mask path, no painting needed.
5. **Params/session integration:** local adjustments are NOT ParamSpecs (they're structured, not scalar) — carry them as a separate `process()` kwarg and a `Session.local_adjustments` field (F2 dependency for persistence; the GUI feature itself can ship before F2).

### Files
`gui.py`, `retouch/engine.py` (new stage), `retouch/regions.py` (radial/linear + op registry), `retouch/session.py` (field), `tests/test_regions.py`, `tests/test_engine.py`, `tests/test_gui.py`.

### Tests / artifact QA
- Mask edge halos: op applied at strength 100 must show no visible seam at feathered boundary (gradient-continuity assertion at mask edge).
- ImageEditor alpha extraction across Gradio versions (pin + adapter function; this is the flakiest dependency — timebox and isolate).
- Semantic intersect correctness (painted ∩ skin ⊆ skin mask).
- Preview latency: brush round-trip < 2s at 800px proxy (existing proxy path `_process_with_proxy`).

**Effort:** ~2 weeks. Depends on: F1 (ops in float), soft-depends on F2 (persistence).

---

## Order & milestones

```
F1 (float32 + 16-bit)  →  F2 (session/undo)  →  F3 (brush masks)
        quality               workflow              capability
```

Milestone after F3: **"tune one photo, batch a whole shoot, undo anything, paint anywhere"** — at that point the system covers the daily loop of a working retoucher without Photoshop.

## How this leverages what you already built (summary)

| Existing asset | Gets leveraged by |
|---|---|
| `precision.py` + `_F_*` grader methods (Phase 1.b/1.d work) | F1 — the float pipeline is mostly plumbing, not math |
| `PROCESS_INPUT_KEYS` params dict (`gui.py:274`) | F2 — the session format already exists in memory |
| `reset_*` + alignment regression tests (Stage 0 work) | F2 — load-session uses the same pattern and guards |
| `batch_processor.py` + recipes | F2 — session-driven batch = tune once, apply to 500 shots |
| `regions.py` masks/blends + BiSeNet semantic masks | F3 — local adjustments are 90% existing mask code |
| `make_comparison()` (`io.py:165`) | F2 snapshots A/B view |
| `_process_with_proxy` fast path | F3 brush preview latency |

## Out of scope for Tier 1 (tracked in roadmap)
Spot heal/LaMa (2.1), liquify (2.2), curve editor (2.3), denoise/super-res (2.4), background replace, linear RAW develop, plugin API. Open bug: `anime_crystal_void` dead keys (still report-only).

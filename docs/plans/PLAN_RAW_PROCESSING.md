# PLAN — RAW Processing (full-data RAF ingestion)

**Date:** 2026-07-10
**Status:** Step 1+2 largely done 2026-07-14 — `imread_engine` live for RAW (GUI/CLI); `load_raw` gamma=(1,1)+ReconstructDefault; CLI `--linear-raw` wires LinearGrader.develop. Step 3 develop UX still backlog.
**Scope:** What "import raw files with full data" actually requires, what already exists in the codebase, what's broken, and the recommended wiring plan.

---

## 1. Current state — three RAW decode sites, only one live

| Site | Output | Used by live pipeline? | Notes |
|---|---|---|---|
| `io.py::imread_exif` (io.py:250) | **uint8** BGR, sRGB, camera WB, `bright=1.0`, `HighlightMode.ReconstructDefault` | **YES** — gui.py, cli.py, batch_processor.py, style_library.py all ingest RAW through this | Fixed 2026-07-10 (was `bright=1.5`). Faithful, but throws away ~6 bits of tonal precision at the first step. |
| `io.py::read_image_16bit` (io.py:214) | **float32 [0,255]** BGR from 16-bit decode (`output_bps=16`), same dev params | **NO** — exists since F1/E2 Wave 1, zero live callers for RAW input | Numerically agrees with imread_exif (mean diff 0.17 on `_DSF1853.RAF`). This is the natural full-data entry point. |
| `retouch/raw_develop.py::RAWDeveloper.load_raw` (T5) | float32 [0,1] RGB, claims "linear" | **NO** — module is an island: no non-test callers anywhere | **BUG (reported, not fixed — ground rule #4):** claims linear RGB but omits `gamma=` from `postprocess()`; rawpy's default gamma is `(2.222, 4.5)` (BT.709 encode), so `load_raw` returns **gamma-encoded** sRGB. Every `LinearGrader` op ("assume linear tristimulus RGB without gamma encoding") therefore operates on the wrong domain. True linear requires `gamma=(1, 1)`. Also missing `output_color` (defaults sRGB — fine) and `highlight_mode` (defaults **Clip** — worse than io.py's ReconstructDefault). |

**T5 status discrepancy:** MASTER_PLAN.md line 12 marks T5 ✅ done, but the Phase 6 table row (line 112) has no completion note, the module has the gamma bug above, and nothing calls it. T5 shipped a library, not a feature.

**Engine ingestion contract:** `RetouchEngine.process(img_bgr)` assumes uint8 BGR at entry (docstring at engine.py:441; MediaPipe/BiSeNet detection requires uint8). Float32 exists *internally* (E1 per-face float canvas, F1 float global stages via `precision.to_float`), but the front door is 8-bit. This is the actual blocker for full-data RAF — not the decoder.

---

## 2. rawpy/LibRaw parameter research (verified against rawpy API docs, 2026-07-10)

Defaults that matter (all confirmed from `rawpy.Params` docs):

- `gamma` — default `(2.222, 4.5)` = BT.709 encode. `(1, 1)` = true linear. **Any module claiming "linear" must pass this explicitly.**
- `output_bps` — default 8; `16` gives uint16 [0, 65535]. Cost is decode-side only (~same speed, 2× memory for the decoded array).
- `output_color` — `ColorSpace` enum, default sRGB. Wide/ProPhoto available if we ever want a wide-gamut working space; engine is sRGB-native today so sRGB is correct.
- `highlight_mode` — default **Clip**. `Blend` (2) and `ReconstructDefault` (5) recover blown channels; io.py already uses ReconstructDefault (2026-07-10 fix). `raw_develop.py` still silently clips.
- `use_camera_wb` — default **False** (daylight WB!). Must pass `True` for as-shot WB; io.py does, raw_develop.py does (good). `user_wb` (4 multipliers) is the hook for a future WB-before-demosaic slider — WB applied in raw domain is cleaner than post-hoc Kelvin shifts on rendered pixels.
- `no_auto_bright` — default False (auto-brightens!). Both io.py paths pass True (correct for faithful dev).
- `exp_shift` (0.25–8.0, linear-domain) + `exp_preserve_highlights` (0–1) — the *right* way to do exposure correction at develop time: applied pre-gamma with soft highlight preservation, vs. the crude post-hoc `bright` multiplier. Candidate backing for a future "develop exposure" knob.
- `demosaic_algorithm` — default AHD. DHT/AAHD are generally higher quality on Bayer sensors. **Fuji caveat (this project is RAF-centric):** X-Trans sensors use LibRaw's dedicated X-Trans interpolation; the Bayer `demosaic_algorithm` choice has little/no effect on RAF from X-series bodies. Don't burn time benchmarking demosaic options on RAF.
- `fbdd_noise_reduction` (Off/Light/Full) — pre-demosaic denoise; `noise_thr` — wavelet denoise. Both operate in the raw domain, upstream of (and complementary to) F7 NAFNet. Worth a high-ISO A/B later; not required for wiring.
- `half_size=True` — half-resolution decode, measured **54× faster on X-Trans** (0.17 s vs 9.2 s, §5.1): ideal for the GUI `fast=True` interactive path where RAF decode latency hurts slider responsiveness.
- Orientation: LibRaw applies sensor flip/rotation from metadata during postprocess by default (`user_flip` unset), so RAW paths don't need the PIL `exif_transpose` step.

## 3. What full-data ingestion actually buys

- **Tonal headroom:** RAF is 14-bit sensor data. 8-bit sRGB ingestion quantizes to 256 levels *before* the engine's tonal moves; 16-bit → float32 keeps ~16k levels. Payoff concentrates where the pipeline pushes hard: `face_exposure` lifts, shadow recovery, strong film curves, `equalize` on pale skin — exactly the ops this project leans on. Well-exposed, lightly-graded shots will look identical; banding-prone gradients (skies, studio backdrops, smooth skin) under heavy grading are where 8-bit shows.
- **Highlight recovery:** already shipped at the decoder (2026-07-10 fix), but its recovered detail lives in the top stops — the region 8-bit quantization crushes hardest. 16-bit ingestion is what lets the recovery survive into the output.
- **Not** more dynamic range per se: `postprocess` still renders to a display-referred image. True scene-referred latitude (multi-stop exposure moves after decode) is the linear-develop story (T5's original intent), a separate, larger step.

## 4. Recommended wiring plan (priority order)

**Step 1 — 16-bit RAF into the live path (~2–4 d incl. parity tests).** The first slice of "T5 as a feature":
1. `process()` entry: accept float32 [0,255] BGR in addition to uint8. Normalize once at the top; derive a uint8 view *only* for stage-0 detection/segmentation (MediaPipe/BiSeNet). Per-face (E1) and global (F1) stages already run float.
2. Ingestion switch: in `imread_exif` callers (gui.py, cli.py, batch_processor.py), route `RAW_EXTENSIONS` through `read_image_16bit` (float32 [0,255]) instead. Non-RAW formats unchanged.
3. Export: PNG-16/TIFF via existing `write_image_16bit` (already GUI-wired from F1 Wave 3) so precision survives end-to-end; 8-bit JPEG export quantizes last, after all grading — still a win.
4. Guards: uint8-input path must stay byte-identical (golden test); float-input parity test vs uint8 within quantization tolerance; GUI `fast=True` RAF preview should use `half_size=True` decode to keep sliders responsive.

**Step 2 — fix `raw_develop.py` linear bug (~0.5 d).** Add `gamma=(1,1)`, `output_color=ColorSpace.sRGB`, `highlight_mode=ReconstructDefault` to `load_raw`; regression test asserting linearity (gray-ramp DNG → linear ramp out). Do this before anything ever calls the module.

**Step 3 (later, optional) — true linear develop UX.** Expose `exp_shift`/`exp_preserve_highlights` (develop-time exposure), `user_wb` (develop-time WB), `fbdd`/`noise_thr` (raw-domain NR feeding F7). This is the remainder of T5's original 2-wk intent; only worth it after Step 1 proves demand.

## 5. Extended research (2026-07-10, second pass — measured on `_DSF1853.RAF`, X-T5-class 26MP X-Trans, rawpy 0.27.0)

### 5.1 Decode economics — 16-bit is FREE, half_size is ~10× faster
Measured locally on the reference RAF (6246×4170):

| Decode | Time | Output |
|---|---|---|
| 8-bit full postprocess | **9.22 s** | 6246×4170 uint8 |
| 16-bit full postprocess | **9.09 s** | 6246×4170 uint16 |
| `half_size=True` | **1.00 s** | 3123×2085 |
| `extract_thumb()` | **0.003 s** | **large embedded JPEG (4416×2944), 3.7 MB** |

Implications: (a) **16-bit ingestion has zero decode-time cost** — the 9 s is X-Trans 3-pass Markesteijn demosaic, identical either way; there is no performance argument for staying 8-bit. (b) The 9 s full decode is far too slow for the GUI slider loop → RAF decode result **must be cached** (keyed on path + dev params), analogous to FaceContext caching. (c) `half_size` (1.00 s, ~10× faster, still 3123px wide > the 2048 proxy) is a fast path for *processing*; the truly instant first-paint (0.003 s) is `extract_thumb()`.

### 5.2 X-Trans demosaic — already optimal, stop worrying
LibRaw's default X-Trans path is **3-pass Markesteijn**, judged at least as good as Fuji's own in-camera demosaic and better than AHD on near-diagonal detail (LibRaw forum + dpreview + RawPedia consensus). `demosaic_algorithm` and `fbdd_noise_reduction` are confirmed **no-ops on X-Trans** (also measured independently, see PROGRESS.md §5). Nothing to tune here; the 9 s cost *is* the quality.

### 5.3 Instant preview via embedded JPEG — `extract_thumb()`
Fuji embeds a **large camera-rendered JPEG (4416×2944, ~70% of full 6246×4170)** (film simulation, DR mode, all in-camera processing applied) inside every RAF; `raw.extract_thumb()` returns it as JPEG bytes in milliseconds. Two uses: (1) **GUI first paint** — show the embedded JPEG immediately while the 9 s develop runs in background; (2) **camera-rendition reference** — it is the ground-truth "what the camera intended", usable as an A/B baseline or even as `color_ref` input for style matching against the in-camera film sim.

### 5.4 Fuji film-sim + shooting metadata — auto-recipe selection
`exiftool` (installed at `/opt/homebrew/bin/exiftool`) reads the RAF makernotes; the reference file yields `FilmMode: F2/Fujichrome (Velvia)`, plus `WhiteBalance`, `HighlightTone`/`ShadowTone`, `GrainEffect`, `ColorChromeEffect`, `DynamicRange`, and full lens info (`XF35mmF1.4 R`, 35mm, f/1.4). Caveat: Acros/mono sims are identified via `Saturation`, not `FilmMode`. Opportunity: **F10 Smart Default could auto-select the matching Fuji film recipe from the RAF's own FilmMode tag** — the shooter already told us the look they wanted. WB needs no exiftool at all: rawpy exposes `raw.camera_whitebalance` ([641, 302, 529] as-shot) and `raw.daylight_whitebalance` directly — enough to compute an as-shot↔daylight Kelvin estimate for a develop-time WB slider (`user_wb`).

### 5.5 Lens corrections — lensfunpy (deferred, Step 3+)
`lensfunpy` (same author as rawpy) corrects distortion, lateral CA, and vignetting from a lens database, driven by the EXIF lens/focal/aperture above. Two catches: not currently installed, and corrections **assume linear (not gamma-encoded) input** — so it belongs inside a true linear develop (after the raw_develop.py gamma fix), not bolted onto the current sRGB path. Portrait payoff is mostly vignetting + CA on fast primes (the f/1.4 reference shot is exactly the use case). Optional dependency; defer with Step 3.

### 5.6 Updated priority order (supersedes §4 ordering nuances)
1. **RAF decode cache + `extract_thumb` first-paint + `half_size` fast path** — GUI usability blocker; without the cache, RAF in the slider loop is unusable (9 s per re-decode). `extract_thumb()` paints in 0.003 s (embedded film-sim JPEG), `half_size` (~1.0 s) is the fast *processing* path. Cheap (~1–2 d), highest leverage; ship first.
2. **16-bit live ingestion, end-to-end (§4 Step 1 + §5.7)** — zero decode cost vs 8-bit; kills the 108-vs-4182-levels banding gap (PROGRESS.md §5). MUST cover both ends (ingest *and* `process()` `return_float` → `write_image_16bit`): `process()` currently quantizes to uint8 at exit (engine.py:1973), so ingestion alone is half a fix. ~3–5 d with caller audit (gui/imwrite/batch/style_library/tests).
3. **`raw_develop.py` gamma fix** (§4 Step 2) — unchanged, before anything calls the module.
4. **FilmMode → recipe auto-select in F10 smart path** (~1 d, exiftool subprocess + mapping table) — unique differentiator for the Fuji-shooting owner; no pipeline risk.
5. **Linear develop UX + lensfunpy corrections** (§4 Step 3, extended) — develop-time exposure/WB, then vignetting/TCA in linear space. Later, demand-gated.

### 5.7 Whole-flow finding — PNG-16 export currently writes 8-bit data (2026-07-10 third pass)
**Headline finding of this research.** Traced the full precision chain end-to-end. The middle of the pipeline is float (E1 per-face float32 [0,255] canvas; F1 global stages float32 [0,1] via `to_float` at engine.py:1862), but **both ends quantize**:
- **Input:** RAF → uint8 via `imread_exif` (known, §1).
- **Output:** `_run_global_phases` ends with `result = to_uint8(result)` (engine.py:1973; same in `_no_face_fallback` at :2262) — `process()` always returns uint8. The GUI's PNG-16 export (gui.py:551-552) then feeds that uint8 result into `write_image_16bit`, so **"PNG-16" is 8-bit data in a 16-bit container** — 256 levels stretched to 65k, zero real precision gained. The F1 float pipeline's precision never reaches disk.

Implication for §4 Step 1: the float boundary work must cover **both ends**, not just ingestion — `process()` needs a float32 return option (e.g. `return_float=True`, a param several internal stage methods already have) so the chain is RAF 16-bit → float in → float through → `write_image_16bit` from float. Fixing ingestion alone would still quantize at engine exit; fixing either end alone buys almost nothing. Effort unchanged (~2–4 d) — it's the same boundary refactor, applied symmetrically.

## 6. Bugs/discrepancies found during this research (reported, not fixed)

1. `raw_develop.py::load_raw` — "linear" output is actually gamma-encoded (missing `gamma=(1,1)`); all `LinearGrader` math runs in the wrong domain. Highlights also hard-clip (default `highlight_mode`).
2. T5 marked ✅ in MASTER_PLAN.md resume line but the module has zero live callers and the Phase 6 table row carries no completion evidence.
3. Live pipeline still ingests RAF at 8-bit (`imread_exif`) — the "full data" gap this plan closes.
4. **PNG-16 export is 8-bit in disguise** (§5.7): `process()` quantizes to uint8 at exit (engine.py:1973), so the GUI's PNG-16 path writes uint8-derived data into a 16-bit file. The F1 "float32 global pipeline + 16-bit export" claim holds inside the engine but not at the export boundary.

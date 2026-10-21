# Retouch Codebase Survey — Master Report

Oct 2026 · 111 research agents · repo `/Applications/htdocs/retouch` @ c3274b6
Scope: full survey — engine, editor port, GUI, video, QA, batch, CLI, infra.
Calibration: findings cross-checked against CLAUDE.md Outstanding Fixes and repo docs (COLOR_DELIVERY_TRUTH.md, ARCHITECTURE.md); already-tracked items (poster-FP veto, EXR E-decisions, tiling seams, eye-gate calibration) excluded from NEW lists.

---

## A. Confirmed bugs — fix first

### A1. Editor curves_plot slot shift (empirically verified)
`_tool_view` emits `curve_graph(*tables)` at ui.py:452 (after cb_preserve) but output slot is index 78 (ui.py:988, after noise_monochromatic) — one-slot positional shift. curves_plot HTML receives `50` (tonal_amount); cb_preserve checkbox receives `<svg>` HTML. Tests pass by accident: view[105] svg assertion encodes the buggy layout.
**Fix:** move `curve_graph(*tables)` before `tonal.get('amount', 50)` (ui.py:440); update test comment view[105]→view[78]. Probe: `/private/var/folders/.../opencode/probe.py`.
**Status: awaiting user go-ahead.**

### A2. detect_halo dead — uint8 overflow (empirically verified)
qa_detectors.py:936 `np.clip(L,0,100).astype(np.uint8)*255//100` wraps mod 256 → result always {0,1,2}; Canny threshold 50 never fires → **detector always returns score 0**.
**Fix:** `np.clip(L,0,100)*2.55` in float then cast.
(Note: earlier L977-978 shape-bug claim refuted — indexing consistent.)

### A3. Batch processor consumer hang
batch_processor.py:416/426 — unguarded `worker_fn` + `progress_callback` in consumer thread. Failure → poison-pill desync → silent partial batch OR producer blocks forever at join (L437). Gradio Progress callback is also not thread-safe (N workers race, gui_batch.py:229-231).
**Fix:** wrap both in try/except (patch sketch in survey); guards keep `task_done()` in finally.

### A4. engine.close() unguarded chain
engine.py:6424-6427 — parser raise skips detector close → the exact post-batch GUI hang the docstring warns about.
**Fix:** wrap each close in try/except+warn, mirroring L6421-6423.

### A5. body_reshape `_max_radius_for_warps` outside class
body_reshape.py:705 defined module-level but called as method at :544 → AttributeError risk on `_apply_warps`.

### A6. Plugin stages never wired
engine.py:1348-1362 calls `discover()`+`init_all()` but never `register_all()`; plugin stages land in manager's private registry, never run. Plus module-leak in sys.modules (plugin_api.py:244-250) and per-plugin partial-registration on stage-name collision.
**Fix order:** wire register_all after global registry build (L1362), then leak fixes, then manifest/priority (hardening sketch in survey).

### A7. blemish.py red detector dead code
blemish.py:178-179 `AND` then `OR` of same set = identity. Delete.

### A8. healing.py opacity message wrong unit
healing.py:238/281 — check is 0..1, message says "0..100". Wrong both directions.

### A9. Liquify/Smudge single-click misleading error
local_retouch.py:378-379 — fires "try different region or healing method" for what is actually "drag to move pixels".

### A10. Ombre lips default keys not in palette
makeup_v2.py:683-684 default `color1="red"`, `color2="pink"` — absent from `EYESHADOW_COLORS` → silent fallback to rose/nude (L708-709). Docstring misdocuments.

### A11. `hair_ring_position` dead in production
hairwork.py:340-341/387 — only used when landmarks is None; production always passes landmarks. Slider no-op. Also `int(... or 30)` makes position 0 unreachable (perf_optimizations.py:1490).

### A12. bloom_threshold/bloom_softness parsed but ignored
recipe_loader.py:137-143 parses them; `_grade_bloom` (editor/adjustments.py:875) ignores both. Wire-up gap.

### A13. `draw.textsize` fallback crashes on Pillow≥10
batch_processor.py:719/742 — fallback path always raises AttributeError (textsize removed in Pillow 10, floor is `>=10`). Use `font.getbbox`.

### A14. selections.py `selection()` no width/height validation
selections.py:128-137 — negative dims pass budget check → bare TypeError. Also: fractional morph amounts silently floored (L200/210/229); `project_selection` ignores `mask_placement` for unlinked masks (L244-263, wrong grid); semantic worker catches only 3 exception types (semantic_selection.py:147 — ONNX/cv2 errors lose traceback).

### A15. session.py:149 imports root-level `gui`
`to_params_tuple` imports `gui` (repo root) — sys.path fragile, gradio pulled in. Function is dead in prod (tests only); docstring lies ("use their defaults" — actually None).
**Fix:** move `PROCESS_INPUT_KEYS` to `retouch/params.py`; fix or delete function.

### A16. XMP sidecar path uncontained
xmp_sidecar.py:354/380 — write path derived from record's source, no containment/symlink check vs review root. `.orig` backup follows symlinks (L390-392). Both fixed by one guard function reusing project_store `_read_file` pattern.

### A17. RetinaFace partial-loss fallback over-eager
detection.py:505 — any single dropped box (usually a correctly-dropped RetinaFace FP) triggers ~60-100ms full-image fallback at 24MP.
**Fix:** `lost * 2 >= retinaface_box_count` fraction gate.

### A18. Video cancel stale-file hazard
video/export.py:250 — cancel file never auto-deleted; stale file kills next job immediately.

### A19. Golden tests green-on-failure
test_golden_pipeline.py:128-136 — `--update-snapshot` skips read as green (bit them once, EXECUTION_LOG.md:32); missing key auto-creates+skips; coverage 3/121 deterministic recipes; read-modify-write races under xdist.
**Fix (<20 lines):** fail on missing key; assert set-equality count test; atomic snapshot writes.

### A20. `segment_person` all-ones fallback mask
detection.py:771/780/792 — segmentation failure → all-ones mask → "protect subject" ops grade the subject. Return zeros.
Also: background.py `replace()` L502-503 uses naive composite, skipping the alpha-matte path others use (hair fringe).

### A21. nose_restore baseline reintroduces defects
engine per-face flow — uses pristine `canvas_original` (L1629-1630) not post-heal `pre_smooth_canvas` → can reintroduce blown highlights/spots/stray hairs the heals removed.

### A22. Worker/inline processor drift
`_get_worker_processors` (engine.py:1713-1739) missing `frequency` processor; `_ctx_for_face` (engine.py:3980-3984) drops ndarray ctx fields on multi-face path. Assert processor-key parity in a test.

### A23. misc
- gui_advanced.py:755 `float(overlay_opacity or 42)` — 0 → 42.
- chromophore.py:62-71 `−log` on gamma-encoded sRGB, no linearization; absolute `clip(0)` floor → hb=0 for 8/10 Monk tones (self-audited PLAN_S1). Route magnitude consumers to chromophore_v2.
- JobStore: no stale-"running" recovery (crashed batch = permanently misleading dashboard row), no flock/merge (below watch_folder bar).
- E1 debug block perf_optimizations.py:213-253 self-marked "remove before finishing"; ~50 `_tr()` call sites; chain-order list already stale.

---

## B. Fake/unfaithful features

| Feature | Truth | Cheap fix |
|---|---|---|
| CLI `--bit-depth 16` | uint8 ×257 stretch; engine quantizes at engine.py:3261; io.py:266 warning never fires on CLI path | Warn at cli.py:875; real fix = `return_float=True` on process() |
| GUI PNG-16 (non-editor) | Same upcast; GUI labels honestly | - |
| face_polish | GUI promises nose/cheekbone highlights + iris crispness; implementation = 3 floor-raises only (perf_optimizations.py:274-278) | Implement or reword |
| Watch folder "auto" | Manual button only; no poller; `recover_stale_jobs` never invoked outside click | Tier-1 read-only gr.Timer(5) dashboard poll is safe now |

---

## C. Dead code — delete list

| Item | Location |
|---|---|
| hair.py `cleanup_flyaway_strands` (+tests) | hair.py:168; superseded by stray_hair.py |
| cosplay_moat.py `WigLaceBlender` (+tests) | :42; superseded by wig_hairline.py (keep module — HosierySmoother live) |
| highlight.py `tone_map_reinhard`/`tone_map_filmic`/`recover_highlights` | :104/:136/:74 |
| style.py `StyleAnalyzer.extract_look` + orphaned helpers | :262; superseded by LookExtractor |
| io.py `imwrite_16bit_png` | :192; zero callers |
| runtime_doctor bare aliases `runtime_doctor`/`doctor_report` | :1559-1560 |
| certification_evidence_v2 aliases `runtime_doctor_evidence`/`sanitize_runtime_doctor` | :211-212 |
| matting.py `refine_alpha_matte` | - |
| undereye.py v1 path | - |
| neural_boosters.py | zero callers |
| Numba block (imported, never used) | perf_optimizations.py:80-98 |
| blemish red detector | blemish.py:171-179 |
| skin_protect `protect_skin_chromatic`/`skin_aware_apply` | skin_protect.py:77-141; tests only |
| teeth `_WID_WPT` | teeth.py:35; defined, never used |
| E1 debug block + `_tr` call sites | perf_optimizations.py:213-253 |
| Relight `spec_factor=1.0` dead var | background.py:442 |
| `generate_film_grain` low-value path | grain.py:144-186 |
| plugin `_E1`/teeth misc | see A-items |

Frequency `separate`/`combine` deprecated wrappers still load-bearing — keep, migrate later. eye_enhancement.py vs eyes.py — keep both (split feature sets).

---

## D. Perf opportunities — ranked

1. **skin.py LAB round-trips** — ~20 full-frame LAB conversions per face ≈ 1.5-3s at 24MP. Biggest single win.
2. **Grading LUT-bake** — point-ops in `_color_ops` (grading.py:425-484) bakeable to 33³ LUT at load; ~10-12× fewer full-frame conversions. `_LatticeGrader` pattern already proven in lut_export.py. Blockers documented (warmth skin_mask branch, split-tone masks, clarity/haze stay post-LUT).
3. **Grading chain duplication** — engine.py:3431 vs 5618+; ~15-25 simultaneous full-frame float32 @24MP. Merge chains; use in-place ops.
4. **Preview cache** — stat-first identity (drop `_sha256_file` from hot path, ~6 lines, kills 40-150ms/tick); byte-budget eviction; arrays-out-of-Gradio-State (removes deepcopy spikes, ~400MB ceiling). Ladder in survey.
5. **`reduce_hemoglobin_variance`** — 23.4s cumulative hotspot; 3× redundant chromophore re-decompose per face (skin.py:1892/1929/1964). Cache one decomposition; downsample-edit-upsample.
6. **Pore synthesis** — Python loop full-frame. Vectorize.
7. **stray_hair** — 36 full-res filter2D, no work-scale (inconsistent with wig_hairline). Add work-res stage.
8. **Costume shine+clarity** — both recompute `painted_skin_mask`/segmenter/costume_mask at same work res. Share one pass. Biggest costume win.
9. **CA** — meshgrid recomputed per call; cache keyed (h,w,disp). Heaviest post-effect @24MP.
10. **Batch analysis gating** — full detection runs even when `auto_group=False` (default) for a status string; sha256 computed but `content_id` never compared on hit. Gate + drop.
11. **Certification renders** — double decode + per-output source re-hash in render gate; parallelize case renders; tiered cert via `extends` graph (biggest lever for 146 recipes).
12. **Blemish component loop** — O(n·HW) per-component mask writes; vectorize via area LUT.
13. **ZIP compression** — `ZIP_STORED` → `ZIP_DEFLATED`, one word (batch_processor.py:1213).
14. **Detection fallbacks** — see A17 (~60-100ms at 24MP).

---

## E. Fairness / tone-invariance violations

House rule: no absolute intensity thresholds on skin-tone-scaled signals.

| Violation | Location |
|---|---|
| `skin_mask_lch` chroma_min=8 absolute gate — breaks deep-skin for skin_protect AND P7 cross-region; **protection fails exactly the tones that need it** | color_space.py:547; skin_protect.py:72/101; cross_region_skin.py:163 |
| blemish absolute 8-23 threshold + darker-only polarity — dark skin under-fires, whiteheads missed | blemish.py:160/163; marks.py has the robust-stats pattern to copy |
| chromophore v1 — no linearization + absolute clip(0) | chromophore.py:62-71/104-105 (A23) |
| Legacy hair specular floor 120 | hair.py:127-161 |
| eyes.py specular boost absolute L>220 contradicts module's own rule | eyes.py:195-218 |
| facepaint crack fixed 12/255 threshold | skin_chromophore.py:464 |
| `paint_coverage_even` global mean pull flattens legit shading | skin_chromophore.py:505 (also never engine-wired) |

Compliant exemplars (use as templates): body_skin_even.py, teeth.py, powder_finish.py, nose_shape/highlight, skin_warmth.py, costume_clarity.py.

---

## F. Structural / architecture findings

- **God function** `_process_face_core` ~1400 lines, 40 stages (perf_optimizations.py:301-1699). Refactor ladder: delete E1+Numba → dedupe 3× `frequency.combine` blocks → extract FA-02/safe_auto/makeup_v2 helpers → cache transient processors (FreckleRemover/EyeEnhancerV0/ShadowLifter per face).
- **Three undo systems** non-communicating (UndoRedoStack / AdvancedHistory / DocumentHistory). Full unification infeasible (different payloads/budgets). Cheap: shared `can_undo/can_redo/truncated` protocol. Biggest immediate fix: positional-tuple plumbing (gui.py:1233-1241 manual index arithmetic) → one HistoryState dataclass. gui_advanced.py is the epicenter (11/14/15-tuple returns, arity mismatches between success/error paths).
- **UndoRedoStack not persisted** across session save/load (advanced canvas history IS — inconsistent).
- **No cooperative cancel for stills** — `_emit_progress` swallows cb exceptions by design; editor cancel = subprocess kill. Minimal sketch: `_check_cancel()` at `_emit_stage` + `_FaceProgress.done`, cancel-file pattern mirrors video. ~3 lines + keep raise outside the swallow-try.
- **Stage F2 opacity unwired; no-face fallback bypasses stage registry; fabric runs hardcoded "Stage 7.6" outside registry** — registry consistency gaps.
- **Text layers opaque baked PNGs** — no rasterizer port (upstream TypeTool.swift ~NSLayoutManager; Python equivalent needs Pango ~150-300 lines). Metadata/validation side already done.
- **Video temporal gaps** — blend sawtooth at parse refresh (adapter.py:223-224, mask amplitude modulated at parse cadence); occlusion-exit stale reference → parse quality cliff (adapter.py:205-210); QA bbox-crop jitter conflates motion with flicker (qa.py:56); no temporal grain.
- **RAM budget not threaded** to `process_folder` (batch_processor) — GUI 4 workers × 45MP ≈ 2.2GiB regardless; mirror cli.py:2055-2069 cap-by-max.
- **CLI 104 params unreachable** (reshape 18, film 16, MV2 12, background 10, ...) — only via undiscoverable session JSON. Serial vs pool worker paths drifting (uint8/float compare divergence, cli.py:2371 vs 775-777).
- **JobStore below watch_folder durability bar** — no flock/merge/stale-recovery; per-file JSON save from N worker threads races (gui_batch.py:279-280).
- **Recipe scale conventions split** — slimming/body_reshape `gui_direct` 0-100 vs cosplay `recipe_pct` 0-1; dodge_burn dict-or-float with `<=1.0` ambiguity boundary. Migration sketch in survey (~15 recipes change behavior; no version field needed — in-code single writer).
- **Smart intents expansion cheap** — Skin card sketch ready (smart_intents.py:456 registry append; validation enforced at import).
- **Params spec consistency actually healthy** — engine.py:290-292 comment stale (91/14/23 actual, not 26/33/5); 1 benign recipe_key collision (`eyes.whites`); bounds clean.
- **Engine progress/cancellation** — ContextVar design clean; multi-face pool path parent-side only by design.

---

## G. Hygiene

- **No lint config at all** — add `[tool.ruff]` + `ruff check retouch scripts tests`; catches trailing-ws class (~150 lines) forever.
- Root clutter `ds.png orig.png plain.png` — delete. `third_party/compositor/Compositor-11d8d7a.../` untracked — commit or ignore.
- `test_output/` on-disk worktree copies of stale code confuse greps — purge.
- Dependencies: mediapipe `==0.10.5` + protobuf `<4` load-bearing (High risk pins); starlette cap-only, add floor `>=0.27`; numba/scipy/torch undeclared — declare extras or delete paths.
- C2PA: solid, but signed-PNG/WebP/TIFF outputs undetected by JPEG-only sniff → XMP embed could break signature on non-JPEG outputs.
- Error-message normalization: three phrasings for opacity bounds across editor modules (fix list in survey).

---

## H. Suggested execution order

1. **Quick wins (1-2 days):** A1 curves_plot · A2 detect_halo · A4 close() guards · A7 red detector · A8-A11 message/param fixes · A13 textsize · D13 ZIP deflate · A18 cancel file · B watch-folder tier-1 timer · A19 golden guards.
2. **Hang/correctness (2-4 days):** A3 consumer guards · A20 segment fallback + replace() matte · A21 nose_restore baseline · A16 sidecar containment · D10 batch gating · A17 fallback fraction gate · JobStore stale recovery.
3. **Dead code sweep (1 day):** Section C list + E1 block + ruff.
4. **Perf wave (1-2 weeks):** D1-D6 ranked; grading LUT-bake is the highest-leverage structural piece.
5. **Fairness wave:** E-table fixes (skin_mask_lch gate first — worst offender).
6. **Structural (later):** F-items — HistoryState dataclass, cooperative cancel, plugin wiring, registry consistency, text rasterizer, video temporal.

---

## Verification commands

- Editor suite: `.venv/bin/python -m pytest tests/test_editor_*.py -q -p no:warnings`
- Gate: `scripts/dev/local-gate` (long timeout), push only after PASS
- View arity current: loaded=123, plain edits=122, grade flows=125
- Never commit untracked `third_party/compositor/Compositor-11d8d7a.../`

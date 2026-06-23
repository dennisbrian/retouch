# Future Roadmap

**Date:** 2026-06-23 (revised — Session 2, v1 scope locked)
**Horizon:** 10-12 weeks (v1 only); 18-24 months (full vision)
**Status:** Planning document — v1 scope is locked, v2+ is aspirational
**Revision notes:** 2026-06-23 — locked v1 = Fuji-quality color recipe system (3 sims, 10-12 weeks). All face features, polish, plugin ecosystem, mobile, cloud, AI agents, video deferred to v2+.

---

## v1 SCOPE (LOCKED 2026-06-23) — Fuji-Quality Color Recipe System

**Total duration:** 10-12 weeks (2.5-3 months)
**Goal:** Ship the best color recipe system in any retouching app. 90-95% match to Fujifilm JPEG.

### Pre-Phase 0: Foundation (2 weeks)

- [ ] **Fix 2 pre-existing bloom-scaling bugs** in `build_context` (`0.16` → `16.0`, `0.1` → `10.0`)
- [ ] **Resolve 33 default mismatches** (4.6) — at minimum the ones affecting color grading
- [ ] **Add missing `color_transfer_intensity` ParamSpec** (registry gap)
- [ ] **Measure baseline** with `pytest --cov` + benchmarks
- [ ] **Set up CI** with `pytest` + benchmark tracking
- [ ] **LCH color space utility spike** (3-5 days) — unblocks perceptual HSL
- [ ] **3D LUT pipeline spike** (2-3 days) — `.cube` loader, trilinear interp
- [ ] **Mask system v0** (1 week) — unified `apply_to_region(img, mask, op)` API
- [ ] **Fuji color research** (1 week) — what makes Fuji look like Fuji (papers, teardowns)

### Phase 1: Fuji-Quality Color (8-10 weeks)

#### 1.a Fuji Foundation Layer (3 weeks)
- [ ] **Tonal response curve** — film H&D model, not digital sCurve. Single biggest "Fuji-look" unlock.
- [ ] **Skin-tone protection** — preserve skin hues during all color operations. Required for Astia-quality portraits.
- [ ] **Film grain synthesis** — organic, clumped, luminance-correlated. NOT Gaussian noise.
- [ ] **Highlight rolloff** — soft film-like clip, not digital hard clip.

#### 1.b 3D LUT Pipeline (2 weeks)
- [ ] **Real `.cube` / `.3dl` loader** with trilinear interpolation
- [ ] **LUT directory registration** (`luts/` like `presets/`)
- [ ] **10+ film stocks** as data (Kodak Portra 400, Fuji Pro 400H, Cinestill 800T, etc.)
- [ ] **Hot-load** support
- [ ] **ICC profile input** — read embedded ICC, convert to working space
- [ ] **ICC profile embedding** on export — preserve color fidelity
- [ ] **Wide-gamut working space** — ProPhoto RGB internally, sRGB at output
- [ ] **16-bit float** internal pipeline

#### 1.c 3 Official Fuji Film Simulations (3 weeks)
- [ ] **Classic Chrome** — flagship, the "Fuji look" most people mean
- [ ] **Astia** — portrait specialist, beautiful skin tones
- [ ] **Provia** — neutral/standard, accurate reproduction
- Each sim is a carefully crafted recipe, not a hardcoded LUT. Combines tonal curve + skin protection + LUT + grain.

#### 1.d Power User Color Tools (2 weeks)
- [ ] **LCH-based HSL panel** (replace HSV math — "luminance" actually controls L*)
- [ ] **Channel mixer for B&W** (per-channel R/G/B weights, -200 to +200)
- [ ] **White balance GUI** (Kelvin + tint sliders, optional eyedropper)
- [ ] **Soft-light blend mode** for grade layer (preserves original detail)
- [ ] **Master HSL controls** (Lightroom "All →" sliders)
- [ ] **Negative split toning** (de-saturation in shadows/highlights)

#### 1.e Recipe System v1 (1-2 weeks, parallel to 1.d)
- [ ] **Recipe builder UI** — power users craft custom recipes
- [ ] **Recipe version diff** — compare two recipes side by side
- [ ] **Recipe export/import** workflow + format docs
- [ ] **Recipe validation** + schema enforcement

### v1 Success Criteria

- ✅ 3 official Fuji sims (Classic Chrome, Astia, Provia) that look 90-95% like real Fuji JPEGs
- ✅ Recipe save/load as JSON
- ✅ Recipe share (export/import)
- ✅ 10+ film stocks in 3D LUT library (data)
- ✅ Skin tones survive grading (Astia-quality portraits)
- ✅ Film grain (organic, not digital)
- ✅ Wide-gamut color fidelity (ICC input/output)
- ✅ Recipe builder UI
- ✅ All 213 current tests pass + new tests for the 3 sims
- ✅ 90-95% subjective match to a Fuji JPEG in casual viewing

---

## v2+ (DEFERRED — out of v1 scope)

### v2: Face Editing (after v1 ships)
- Makeup engine: lipstick, eyeshadow, eyeliner, foundation, contour, highlighter, mascara
- Body reshaping: waist, legs, arms, shoulders, posture, height
- Expression editing: subtle smile, eye-open, brow lift (identity-preserving)
- Per-region face editing: forehead, T-zone, cheeks, chin

### v3: Polish + Workflow (after v2)
- Smart Default button (auto-enhance)
- Adjustment brush / radial filter / graduated filter (Lightroom-class local adjustments)
- Undo/redo stack
- Save/load `.json` workspaces
- Match-to-reference curve generation
- Plugin API v0
- Recipe cookbook (community-contributed)

### v4: Strategic Expansion (2027+)
- Mobile companion app (deferred)
- Plugin ecosystem maturity
- (KILLED) Cloud processing
- (DEFERRED) AI Style Agents
- (DROPPED) Video support

---

## Strategic Direction (revised 2026-06-23)

Pro Max Retouch sits in the **professional desktop portrait retouching** space. The current code is solid (641+ tests, clean architecture, central param registry). The v1 strategy is **Fuji-quality color first**:

1. **v1 (10-12 weeks)** — Fuji-quality color recipe system. The most important thing per the user.
2. **v2 (3-4 months)** — Face editing MVP (lipstick, basic body reshape).
3. **v3 (3-4 months)** — Polish + workflow.
4. **v4 (2027+)** — Mobile companion, plugin ecosystem.

**Strategic rationale:** The "recipe" is the unit of value. A user can save a recipe, share it, and apply it to thousands of photos. Getting color right (Fuji quality) is the highest-leverage thing — once the color foundation is solid, every later feature (face editing, body reshaping, expression) benefits from the same color science.

---

## Pre-Phase 0: Tech Debt & Foundation (2 weeks) — see v1 above for details

The same tasks as v1 Pre-Phase 0 above.

---

---

## Pre-Phase 0: Tech Debt & Foundation (3-5 days)

**Why this is here:** Performance and Smart Default features both depend on having correct defaults, working bloom-scaling, and benchmark tracking in CI. Doing Pre-Phase 0 first means every later decision has a solid baseline.

- [ ] **Fix 2 pre-existing bloom-scaling bugs** in `build_context` — `bloom.opacity=0.16` should become `16.0`, not `0.16`. Confirmed pre-existing via `git stash`. Likely 1-2 lines per field.
- [ ] **Resolve 33 default mismatches** (4.6) between `ProcessingContext` dataclass and `ParamSpec.default`. Either sync or document why they should differ. 16 are type-only (0.0 vs 0); 17 are substantive value differences; 4 are None-vs-sentinel.
- [ ] **Add missing `color_transfer_intensity` ParamSpec** to `retouch/params.py` (registry gap found during 4.6 audit).
- [ ] **Measure actual test coverage** with `pytest --cov` — replace the "60% estimate" in success metrics with a real number.
- [ ] **Set up CI** with `pytest` + benchmark tracking on every PR. Without this, perf regressions are invisible.
- [ ] **Address 2 OPEN audit items** — `4.8` module pattern (cosmetic, low value, optional) and `4.14` `sys.path.insert` (needs `pyproject.toml`).

**Effort:** 3-5 days
**Impact:** Every later phase benefits. Smart Default is meaningless if the recipe→context conversion is wrong.

---

## Phase 1: Polish & Performance (1-2 months) — REPLACED BY v1 (see top of file)

The original Phase 1 (perf + UX) has been replaced by the v1 Fuji-quality color recipe system above. The old content is preserved below for reference but is **not the active plan**.

<details>
<summary>Original Phase 1 content (deprecated, kept for reference)</summary>

### 1.1 Performance — The #1 User Pain Point

**Current state:** ~3.5s per photo (full quality, single face). The bilateral filter in `FrequencySeparator.combine` is the known bottleneck (~700ms per face).

**Target:** < 2s per photo for typical 4K input, < 1s for 1080p.

**Actions:**
- [ ] **Bilateral → guided filter** — replace `cv2.bilateralFilter` with guided filter (4-8× faster, similar quality)
- [ ] **Metal Performance Shaders** — use `cv2.dnn` with CoreML/Metal backend for ONNX inference (2-3× on Apple Silicon)
- [ ] **MediaPipe GPU delegate** — switch from explicit CPU to GPU delegate (architectural change since we currently force CPU for stability)
- [ ] **Batch ONNX inference** — when multiple faces detected, batch their BiSeNet runs in one inference call
- [ ] **WebGPU client-side preview** — run the 800px fast pipeline in the browser via WebGPU, eliminate server round-trip
- [ ] **Lazy model loading** — don't load ONNX models until first process() call (faster startup)

**Effort:** 3-4 weeks
**Impact:** 2-3× throughput = the difference between 9min/161 photos and 3-4min/161 photos

### 1.2 UX Polish

**Current state:** Gradio is functional but intimidating. 59 sliders across 11 accordions. New users don't know where to start.

**Actions:**
- [ ] **"Smart Default" button** — one-tap auto-enhance: detect face, analyze image, apply optimal recipe + parameter tuning
- [ ] **Preset gallery with thumbnails** — visual grid of recipes with before/after preview thumbnails (not a dropdown)
- [ ] **Inline tooltips with examples** — every slider shows a hover tooltip with a real photo example at different values
- [ ] **Onboarding wizard** — first-launch guided tour: upload → pick recipe → tweak → export
- [ ] **Undo/redo stack** — for slider adjustments (Gradio doesn't ship this)
- [ ] **Save/load slider state** — export current slider values as a `.json` workspace, share with others

**Effort:** 3-4 weeks
**Impact:** Reduces onboarding friction. "Smart Default" alone would close the UX gap with Meitu for casual users.

### 1.3 Test Coverage Gaps

**Current state:** 641+ tests, but the following are still untested:
- `gui.py` event bindings (only helpers are tested)
- `batch_processor.py` for non-JPEG formats
- `style.py` end-to-end (StyleAnalyzer.extract on real images, StyleApplier.apply)
- Engine multi-face parallelization code path
- Engine `_process_with_proxy` (auto 2048px downscale)
- Engine `_stage_subject_separation`
- Engine `_apply_white_costume_lift`

**Actions:**
- [ ] Add engine integration tests with real (or better synthetic) face images
- [ ] Add end-to-end batch processing tests (folder → ZIP)
- [ ] Add property-based tests for `FrequencySeparator` (invariants like round-trip)

**Effort:** 1-2 weeks

---

## Phase 2: Face Edit — Close the Pro Tool Gap (3-6 months)

**Priority:** 🟠 P1 (promoted from P2 in original draft — user prioritization 2026-06-23)

The three sub-phases below can be tackled in any order, but the recommended sequence is **Makeup MVP first** (highest user demand) → **Body Reshaping** (next Meitu gap) → **Neural Upgrades** (quality boost, last because each model is a dependency).



### 2.1 Body Reshaping (extend `geometry.py`)

**Current state:** Face slimming only via `ctx.slimming` (jaw + cheek + chin).

**Target:** Body reshaping to compete with Meitu body tools and Luminar Neo Body AI.

**Actions:**
- [ ] Pose estimation (MediaPipe Pose) to detect body landmarks
- [ ] Liquid warp for: waist, legs, arms, shoulders
- [ ] Posture correction (straighten back)
- [ ] Height adjustment (legs length)
- [ ] Head-to-body ratio adjustment (common in Asian beauty retouch)

**Effort:** 6-8 weeks
**Risk:** High — pose estimation is hard, warp quality matters
**Impact:** Opens up a whole new use case beyond portrait

### 2.2 Makeup Engine (extend `makeup.py`)

**Current state:** Blush + nose blush + under-eye blush + lip tint only.

**Target:** Full makeup application like Meitu's cosmetic features.

**Actions:**
- [ ] **Lipstick** — color picker, matte/glossy/satin finishes, blend with natural lip texture
- [ ] **Eyeshadow** — color picker, multiple shades, gradient blending
- [ ] **Eyeliner** — thickness + wing styles (cat eye, natural, dramatic)
- [ ] **Mascara/lashes** — enhance existing lashes, add synthetic lashes
- [ ] **Foundation** — even out skin tone without changing underlying texture
- [ ] **Contour** — subtle shadow placement for face shape definition
- [ ] **Highlighter** — add glow to cheekbones, brow bone, nose bridge

**Effort:** 8-10 weeks
**Risk:** Medium — makeup looks very different at different quality levels
**Impact:** Could match Meitu's cosmetic feature set

### 2.3 Neural Upgrades

**Current state:** Classical CV approaches (bilateral filter, CLAHE, color transfer).

**Target:** Lightweight ML upgrades for quality boost without massive compute.

**Actions:**
- [ ] **Neural blemish removal** — replace OpenCV fast-marching with small U-Net ONNX (~5MB)
- [ ] **GAN-based skin smoothing** — preserve texture while removing blemishes (alternatives: GFPGAN, RestoreFormer)
- [ ] **Hair segmentation** — replace selfie-segmenter approximation with BiSeNet-trained hair model
- [ ] **Clothing segmentation** — for white costume lift, color transfer, etc.
- [ ] **Background removal/replacement** — useful for headshots, ID photos

**Effort:** 4-6 weeks each
**Risk:** Medium — ONNX model quality varies, need to vet carefully
**Impact:** Quality jump, but each model is a dependency

### 2.4 Video Support

**Current state:** Photo only.

**Target:** Process video clips frame-by-frame with face tracking continuity.

**Actions:**
- [ ] Video input/output (mp4, mov)
- [ ] Per-frame face detection with temporal smoothing (avoid jitter)
- [ ] Face tracking across frames (avoid re-detecting every frame)
- [ ] Style profile application to entire video consistently
- [ ] Preview scrubbing
- [ ] Audio preservation

**Effort:** 8-12 weeks
**Risk:** High — memory management for video is tricky
**Impact:** Massive — opens up YouTube creator, vlogger, filmmaker markets

---

## Phase 3: Plugin Ecosystem (parallel to Phase 2, months 3-6)

**Priority:** 🟠 P1 (promoted from 🟢 P3 in original draft — user prioritization 2026-06-23)

**Why this is a foundation, not a feature:** A well-designed plugin API unlocks community contributions to face edit (Phase 2) without adding to the core team. By month 6, the community can ship lipstick variants, niche makeup styles, and post-effect filters while the core team focuses on the hard stuff.

### 3.1 Plugin API v0 (2 weeks, do first)

**Actions:**
- [ ] Define plugin protocol: a function with a name, docstring, and `ParamSpec` declaration that takes an image + context and returns an image
- [ ] Plugin manifest (JSON or TOML): name, version, author, dependencies, entry point
- [ ] Type-safe registration: `register_plugin(stage_fn, spec)` validates at registration time
- [ ] Sandbox boundaries: plugins can't access engine internals, only the public API + provided image/ctx

**Risk:** API design is hard to change later. v0 should be minimal but stable.

### 3.2 Plugin Discovery (2 weeks)

- [ ] Entry-point discovery via `pyproject.toml` (PEP 621)
- [ ] CLI: `retouch plugin list`, `retouch plugin install <name>`, `retouch plugin validate`
- [ ] Hot-load for development (no restart needed)

### 3.3 Reference Plugin Implementations (1 week each)

Ship 2-3 official plugins to validate the API and seed the ecosystem:
- [ ] **"Fujifilm Classic Chrome" recipe plugin** — port an existing recipe to plugin form
- [ ] **"Cinematic Teal & Orange" post-effect plugin** — a popular color grade
- [ ] **"Vignette Pro" post-effect plugin** — demonstrates the post-effect slot

### 3.4 Community / Marketplace (later)

- [ ] Community recipes sharing platform (GitHub Discussions → tagged recipes)
- [ ] Plugin submission guidelines + CI validation
- [ ] Optional: lightweight registry service for discovery (deferred until demand exists)

**Effort:** 4-6 weeks (3.1-3.3), ongoing for 3.4
**Risk:** Medium — API stability, security (plugins can do anything with images)
**Impact:** Long-term ecosystem play. Compounds with every Phase 2 face edit feature.

---

## Phase 4: Deferred / Killed

### 4.1 AI Style Agents (LLM-driven) — ⏸️ DEFERRED

**Status:** Deferred to 2027+ (no current budget / unclear value)

**Original idea:** Natural language interface: "make this look like a Fujifilm portrait" → applies optimal settings.

**Why deferred:** LLM cost is real ($0.01-0.10 per query, adds up fast at scale). Latency. Reliability. The "natural language → recipe" mapping needs a curated dataset of good outputs that doesn't exist yet. **Plugin ecosystem is the more reliable path to "community-extensible retouching."**

**Re-evaluate:** After 6 months of plugin ecosystem data, see if there's a real demand signal. If yes, revisit.

### 4.2 Cloud Processing — ❌ KILLED

**Status:** Killed 2026-06-23 (user decision: $0 cloud budget)

**Original idea:** Optional cloud backend for users without GPUs.

**Why killed:** Ops cost (servers, bandwidth, support) without revenue model. Privacy concerns (no user trust for retouching photos in the cloud). Plugin ecosystem + local-first covers the same need for power users.

**Don't revisit** unless the business model changes.

### 4.3 Mobile Companion App — ⏸️ DEFERRED to 2027

**Status:** Deferred (16-24 weeks, native mobile is expensive)

**Why deferred:** Native iOS/Android dev is a separate skill set and infrastructure. Better to wait for plugin ecosystem maturity (some plugins can be reused) and prove the desktop product first.

**Re-evaluate:** Q4 2026, after plugin ecosystem + Phase 2 face edit are shipping.

---

## Cross-Cutting Initiatives (ongoing)

### Documentation

- [ ] User-facing tutorial videos (YouTube)
- [ ] Recipe cookbook with before/after galleries
- [ ] API documentation website (mkdocs or docusaurus)
- [ ] Community Discord / forum

### Community

- [ ] Contributing guide for new recipes
- [ ] Style profile sharing platform
- [ ] Bug bounty program
- [ ] Quarterly community calls

### Quality

- [ ] CI/CD with full test suite on every PR
- [ ] Performance regression tracking (track benchmarks per release)
- [ ] Memory profiling in CI
- [ ] Cross-platform testing (Linux ARM, Windows CUDA, macOS Metal)

---

## Prioritization Summary (revised 2026-06-23)

| Priority | Item | Effort | Impact | When |
|---|---|---|---|---|
| 🔴 P0 | **Pre-Phase 0: tech debt** (bloom bugs, 4.6, CI) | 3-5d | Foundation | **Pre-Phase 0** |
| 🔴 P0 | Performance: bilateral → guided filter | 2w | 4-8× speedup | Month 1 |
| 🔴 P0 | Smart Default button | 1w | Massive UX | Month 1 |
| 🟠 P1 | **Plugin API v0** (PROMOTED) | 2w | Foundation for ecosystem | Month 1-2 |
| 🟠 P1 | Metal/CoreML ONNX backend | 2w | 2-3× speedup | Month 2 |
| 🟠 P1 | Makeup MVP (lipstick, eyeshadow) (PROMOTED) | 6w | Meitu parity | Month 3-4 |
| 🟠 P1 | **Plugin discovery + reference plugins** (PROMOTED) | 4w | Community | Month 3-4 |
| 🟡 P2 | Body reshaping | 8w | New use case | Month 4-5 |
| 🟡 P2 | Neural blemish removal | 4w | Quality boost | Month 5-6 |
| 🟢 P3 | WebGPU client-side preview | 3w | Eliminates latency | Month 6+ |
| ⚫ KILLED | ~~Cloud processing~~ (decision 2026-06-23) | — | — | — |
| ⚫ DEFERRED | ~~AI style agents~~ → 2027+ | — | — | — |
| ⚫ DEFERRED | ~~Video support~~ → dropped from scope | — | — | — |
| 🔵 P4 | Mobile companion → 2027 | 20w | Mobile gap | 2027 |

**Key change from original draft:** Video support (Phase 2.4 in original) was dropped. Rationale: 8-12w effort for a feature that competes with mature tools (Premiere, DaVinci). Plugin ecosystem covers more ground per hour invested. Can re-add if community demand appears.

---

## Success Metrics (revised 2026-06-23)

| Metric | Current | Target (6mo) | Target (12mo) |
|---|---|---|---|
| Photo processing time (4K, full) | 3.5s | 1.5s | 1.0s |
| Test coverage | **TBD (measure in Pre-Phase 0)** | 75% | 85% |
| GUI onboarding time (new user) | 10+ min | 2 min | 30s |
| Feature parity vs Meitu | TBD (verify) | 65% | 80% |
| Community recipes | 0 | 50 | 500 |
| Community plugins | 0 | 10 | 50 |
| GitHub stars | (current) | 2× | 5× |

**Note:** Test coverage and Meitu parity metrics need to be measured, not estimated. Pre-Phase 0 task.

---

## What NOT to Build (revised 2026-06-23)

- **AI image generation** — different market (Stable Diffusion, Midjourney territory)
- **Social network** — too expensive, not our core competency
- **Mobile-first from scratch** — use a companion app (deferred to 2027) instead
- **Generic photo editor** — Lightroom/Capture One already own that space
- **Print ordering / e-commerce** — different business entirely
- **~~Cloud processing~~** — killed 2026-06-23 ($0 budget)
- **~~Video support~~** — dropped 2026-06-23 (high effort, mature competitors)

---

## Updated Decision Point

**The plan is now:**

1. **Pre-Phase 0** (3-5 days): Tech debt + CI + measurement — non-negotiable foundation
2. **Phase 1** (1-2 months): Polish + perf, in parallel with **Plugin API v0** (so plugin ecosystem is ready before face edit ships)
3. **Phase 2** (3-6 months): Makeup MVP → Body reshaping → Neural upgrades (the "face edit" priority)
4. **Phase 3** (parallel to Phase 2): Plugin discovery + reference plugins
5. **Phase 4** (deferred): Mobile companion in 2027. AI agents and cloud permanently parked.

**Strategic shape:** Foundation → Polish + Plugin API → Face Edit + Community → Strategic (2027)

**My recommendation:** Start Pre-Phase 0 today (the bloom-scaling bugs and 4.6 mismatches are the highest-leverage free wins). Then do Phase 1 perf + UX in parallel with the Plugin API v0 (the 2-week investment pays off as soon as Phase 2 face edit starts).

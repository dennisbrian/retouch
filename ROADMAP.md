# Future Roadmap

**Date:** 2026-06-22
**Horizon:** 12 months
**Status:** Planning document — not committed code

---

## Strategic Direction

Pro Max Retouch sits in the **professional desktop portrait retouching** space. The current code is solid (641+ tests, clean architecture, central param registry). The roadmap is organized in three concentric rings:

1. **Inner ring** (1-2 months) — Polish what we have, fix the rough edges
2. **Middle ring** (3-6 months) — Close the feature gap with pro tools (Capture One, Luminar Neo)
3. **Outer ring** (6-12 months) — Strategic expansion (video, mobile companion, AI agents)

---

## Phase 1: Polish & Performance (1-2 months)

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

## Phase 2: Feature Expansion — Close the Pro Tool Gap (3-6 months)

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

## Phase 3: Strategic Expansion (6-12 months)

### 3.1 AI Style Agents (LLM-driven retouching)

**Current state:** User manually picks recipes and tweaks sliders.

**Target:** Natural language interface: "make this look like a Fujifilm portrait with soft skin and warm tones" → applies optimal settings.

**Actions:**
- [ ] Integrate LLM API (Claude/GPT) that maps natural language to recipe + parameter suggestions
- [ ] "Style coach" mode — describes what's wrong with the current settings in plain English
- [ ] "Reference matching" — upload a target image, describe what you want, LLM picks the recipe
- [ ] Batch auto-style — "make all 200 photos look like this reference"

**Effort:** 6-8 weeks
**Risk:** Medium — LLM cost, latency, reliability
**Impact:** Revolutionary UX — no one else has this

### 3.2 Cloud Processing (Optional)

**Current state:** 100% local.

**Target:** Optional cloud backend for users without powerful GPUs.

**Actions:**
- [ ] Server API (FastAPI) wrapping the engine
- [ ] Job queue with progress tracking
- [ ] Web UI for remote processing
- [ ] Pricing tiers (free tier with limits, paid for higher)
- [ ] Privacy-first: ephemeral processing, no photo storage

**Effort:** 12-16 weeks
**Risk:** High — ongoing ops, cost, privacy concerns
**Impact:** Opens up to mobile users (thin client → server processing)

### 3.3 Plugin Ecosystem

**Current state:** Monolithic pipeline.

**Target:** Allow third-party extensions (custom stages, custom recipes, custom post-effects).

**Actions:**
- [ ] Define plugin API (register a stage function with a name, docstring, and param spec)
- [ ] Plugin discovery (entry points via `pyproject.toml`)
- [ ] Plugin marketplace / registry
- [ ] Community recipes sharing platform

**Effort:** 8-10 weeks
**Risk:** Medium — API design is hard to get right
**Impact:** Long-term ecosystem play

### 3.4 Mobile Companion App

**Current state:** Desktop only.

**Target:** iOS/Android app that captures photos and sends to local desktop or cloud for processing.

**Actions:**
- [ ] Mobile camera integration
- [ ] Live preview of retouch (server streaming)
- [ ] Quick retouch presets optimized for mobile viewing
- [ ] Sync with desktop workspace

**Effort:** 16-24 weeks
**Risk:** Very high — native mobile dev is expensive
**Impact:** Closes the mobile gap with Meitu

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

## Prioritization Summary

| Priority | Item | Effort | Impact | When |
|---|---|---|---|---|
| 🔴 P0 | Performance: bilateral → guided filter | 2w | 4-8× speedup | Month 1 |
| 🔴 P0 | Smart Default button | 1w | Massive UX | Month 1 |
| 🟠 P1 | Metal/CoreML ONNX backend | 2w | 2-3× speedup | Month 2 |
| 🟠 P1 | WebGPU client-side preview | 3w | Eliminates latency | Month 2 |
| 🟠 P1 | Makeup engine (lipstick, eyeshadow) | 6w | Meitu parity | Month 3-4 |
| 🟡 P2 | Body reshaping | 8w | New use case | Month 4-5 |
| 🟡 P2 | Video support | 10w | Massive new market | Month 5-6 |
| 🟢 P3 | Neural blemish removal | 4w | Quality boost | Month 6-7 |
| 🟢 P3 | AI style agents (LLM) | 8w | Revolutionary UX | Month 8-9 |
| 🔵 P4 | Mobile companion | 20w | Mobile gap | Month 10+ |
| 🔵 P4 | Cloud processing | 16w | Reach expansion | Month 12+ |

---

## Success Metrics

| Metric | Current | Target (6mo) | Target (12mo) |
|---|---|---|---|
| Photo processing time (4K, full) | 3.5s | 1.5s | 1.0s |
| Test coverage | ~60% (estimate) | 75% | 85% |
| GUI onboarding time (new user) | 10+ min | 2 min | 30s |
| Feature parity vs Meitu | 40% | 65% | 80% |
| Active community recipes | 0 | 50 | 500 |
| GitHub stars | (current) | 2× | 5× |

---

## What NOT to Build

- **AI image generation** — that's a different market (Stable Diffusion, Midjourney territory)
- **Social network** — too expensive, not our core competency
- **Mobile-first from scratch** — use a companion app + cloud instead
- **Generic photo editor** — Lightroom/Capture One already own that space
- **Print ordering / e-commerce** — different business entirely

---

## Decision Point: What to Tackle First?

The top 3 candidates for the next 2 months:

1. **Performance sprint** (bilateral → guided filter, Metal backend) — most users feel the speed, 4× improvement
2. **UX sprint** (Smart Default, preset gallery, onboarding) — most new users benefit, broader appeal
3. **Makeup MVP** (lipstick + eyeshadow only) — closes the biggest feature gap with Meitu

My recommendation: **Option 1 + 2 in parallel** — performance and UX together. That keeps current users happy (speed) while broadening appeal (UX). Then start makeup in month 3.

What's your priority?

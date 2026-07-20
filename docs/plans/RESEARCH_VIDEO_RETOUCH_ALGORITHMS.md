# Research — Face Retouch Algorithms for TikTok-Class Video Quality

**Status:** 📚 RESEARCH — companion to `PLAN_VIDEO_FACE_RETOUCH.md`; informs the
V0 technology bake-off. No pipeline code is authorized by this document.

**Question answered:** what algorithms do TikTok/ByteDance-class engines
actually use for live face retouch, and which of them close the gap between
our image engine and that quality bar in video.

---

## 1. What "TikTok parity" actually means — three tiers

Consumer engines are not one algorithm. They are three distinct quality tiers,
and being honest about which tier we target keeps the parity claim scoped:

**Tier A — classical shader stack** (TikTok "Retouch"/Enhance, most beauty
SDKs for years): landmark-driven skin mask → edge-preserving smoothing on a
downsampled frame → high-frequency reinjection → LUT tone ops → landmark mesh
warp for reshaping. Entirely deterministic, ~1–5 ms/frame on mobile GPU.

**Tier B — attribute-aware adaptive stack** (FabSoften-class; what separates
"filtered" from "natural"): same primitives, but per-face automatic strength
driven by measured skin attributes (blemish count, texture coarseness), guided
feathering of mask edges, and explicit texture *restoration* after smoothing
rather than just weaker smoothing. Still real-time on phone hardware.

**Tier C — generative re-render** (TikTok *Bold Glamour*, 2023+): a neural
network synthesizes the retouched face region per frame (GAN-style
image-to-image), which is why Bold Glamour survives hand-over-face occlusion
tests that break mesh-warp filters. Press claims of ByteDance model names
("FaceFormer", "TextureGAN") come from third-party vendor blogs, not ByteDance
disclosures — treat as unverified. **Tier C is identity-synthesizing and is
excluded by this track's own product boundary** ("no identity-changing
generative edits"). Our parity target is Tier B, with Tier A as the V1 floor.

A useful platform signal: TikTok's Effect House exposes retouch as a single
platform-level "Enhance Mode" toggle rather than per-effect shaders — retouch
is a centralized engine service, versioned and QA'd once. Our video adapter
should mirror that shape (one retouch service consuming stabilized context),
which the plan's architecture already does.

---

## 2. Algorithm inventory by primitive

### 2.1 Skin region estimation

Real-time engines do **not** run full semantic parsing per frame. The standard
stack is:

- landmark-polygon face prior (cheap, every frame) fused with a
  skin-probability color model built from the face crop's own pixels;
- **guided feathering** of the mask boundary (FabSoften's term): feather the
  alpha using the image itself as guidance so the mask edge follows hairline
  and beard texture instead of a geometric contour;
- heavier semantic parsing (our BiSeNet) run every N frames — or on scene
  change / low confidence — and flow-warped or track-warped between runs.

Our BiSeNet masks are higher quality than typical SDK color priors; the video
question is *cadence*, not model choice. **V0 experiment:** parsing every
frame vs every 4–8 frames with track-warp, measured on the corpus's
mask-edge-error and flicker metrics.

### 2.2 Smoothing operator

The field converged on the **guided filter** (He et al.) over the bilateral:
O(1) per pixel regardless of radius, no gradient-reversal artifacts, trivially
GPU-friendly. Open-source commercial-grade implementations (GPUPixel, the
GPUImage BeautifyFaceDemo lineage) all use guided-filter or surface-blur
variants. Two universal engineering tricks:

- **Downsample → filter → upsample** (patented as early as US 9390478): smooth
  at 2–4× reduced resolution, bilinear-upsample the low-frequency result. This
  is both the speed trick and a natural low-pass split.
- **High-pass reinjection**: add back the original high-frequency band
  (texture, pores) at reduced amplitude — the classic "surface blur + high
  pass + soft-light blend" recipe (see YUCIHighPassSkinSmoothing).

This is mathematically our existing frequency-separation module. **The gap to
TikTok is not the operator — we already have a better one at image quality —
it is (a) a real-time formulation and (b) temporal stability of its inputs.**

### 2.3 Texture retention (the Tier-B differentiator)

FabSoften (CVPR Workshops 2020, Samsung Research) is the best published match
for "natural" consumer smoothing and is the reference pipeline to study:

1. attribute-aware **dynamic** smoothing — filter strength set per face from
   measured blemish count and skin-texture coarseness, not a global slider;
2. **wavelet band manipulation** to restore underlying skin micro-texture
   after smoothing (restore, don't just under-smooth);
3. explicit **facial-hair preservation** so beards/brows/baby hair keep
   delicate texture;
4. minimal cost — runs on low-power mobile devices.

For video, per-face auto-strength must be smoothed over time (it becomes one
of the plan's "automatically chosen strengths" under temporal stabilization),
or the adaptivity itself flickers.

### 2.4 Blemish / spot removal

- Classical real-time: DoG/blob detection on the skin mask + small-kernel
  inpaint or median patch — what mobile engines ship.
- SOTA offline: **ABPN** (CVPR 2022) — a context-aware local retouching layer
  at low resolution plus an adaptive blend pyramid that lifts the edit to full
  resolution; real-time on 4K with a datacenter GPU. Its *edit-lifting pyramid*
  idea maps directly onto our proxy-resolution architecture (edit at proxy,
  blend-lift to native), even if we never adopt the network.
- **Video-specific requirement:** per-frame spot detection flickers by
  construction. Detections must be anchored in the plan's canonical face-local
  space and persisted across frames with confidence decay — a spot is an
  object with a lifetime, not a per-frame measurement.

### 2.5 Tone ops (brighten / even / whiten, teeth, sclera)

Industry engines apply luma/HSV LUT curves ("whitening", "yellow reduction").
Two repo-specific constraints:

- Chinese-market SDK "whitening" defaults are a documented fairness hazard.
  Per CLAUDE.md's tone-invariance rule, every tone op must be
  margin-above-baseline relative to the face's own diffuse baseline
  (`retouch/specular.py::extract_specular` is the reference pattern) — never
  an absolute luminance target.
- Teeth/sclera whitening already went tone-adaptive in the image engine
  (`fd095ad`); the video adapter must consume those, not reimplement.

Shine/specular control is the same story: our specular extraction is already
the right primitive; it needs temporal smoothing of its baseline estimate so
the baseline doesn't breathe with exposure changes.

### 2.6 Face reshaping (slim, jaw, eyes)

Consumer engines use landmark-driven mesh warps: Delaunay-triangulate the
landmark set, offset target vertices, render through the warped mesh in a
shader (equivalently MLS/TPS image deformation — our Stage 1 liquid warp is
this family). The two video-specific findings:

- **Expression compensation**: production engines dampen reshape strength
  during strong expressions (open jaw, wide smile) so the warp doesn't fight
  the deformation — a per-frame scalar derived from blendshape/expression
  coefficients.
- Warp *parameters* get temporally smoothed, never the warped pixels; and the
  warp must be computed in the stabilized face-local frame, or slimming
  visibly wobbles during head turns. This is the plan's "reuse geometry tools
  only after temporal QA proves they do not wobble" gate, now with a concrete
  mechanism.

---

## 3. Temporal-stability research (beyond the plan's current citations)

- **Deep Video Prior** (arXiv 2007.01466, already in the plan) — general
  blind temporal consistency.
- **BlazeBVD** (arXiv 2403.06243) — scale-time equalization for blind video
  deflickering; relevant as a QA-stage rescue, not a primary mechanism.
- **VToonify** (arXiv 2209.11224) — two directly reusable ideas despite being
  a stylization paper: a flicker-suppression loss, and **temporal smoothing of
  the face-parsing map itself** (smooth the masks, not the output — exactly
  the plan's "stabilize inputs, not pixels" policy, independently validated).
- **Blind face restoration → video benchmark** (arXiv 2410.11828) — documents
  that per-frame face processing produces characteristic jitter/"noise-shape"
  flicker and that alignment smoothing + a temporal consistency pass fixes
  most of it; useful framing for our V0 metric definitions.

Consistent conclusion across all sources and vendor practice: **smooth
geometry, masks, and control signals; never cross-frame-blend output pixels**
except as a last-resort QA back-off. The plan's temporal policy is aligned
with the literature as written.

---

## 4. Real-time engineering references (V2/V4 relevance)

| Reference | What to take from it |
|---|---|
| GPUPixel (github.com/pixpark/gpupixel) | C++11 + OpenGL/ES cross-platform beauty engine (smooth/whiten/slim); closest open-source architecture to a commercial SDK — study its filter graph and mask handling |
| GPUImage `BeautifyFaceDemo` | The canonical minimal shader recipe (bilateral + edge + curve mix) — Tier A floor in ~3 shaders |
| YUCIHighPassSkinSmoothing | High-pass reinjection recipe on Core Image; the texture-retention half of frequency separation in shader form |
| Intel scalable beautification patents (US 10152778, 10453270, 11328496) | Per-effect scheduling across GPU fixed-function vs GPGPU kernels; useful map of which ops are cheap where |
| TRTC beauty-filter engineering blog | Industry budget confirmation: full stack within ~16 ms/frame via OpenGL/Metal; frequency-domain split + guided filtering + Delaunay mesh reshaping described as the standard stack |

Budget arithmetic for V4: 30 fps leaves ~33 ms total; tracking (~5–10 ms
MediaPipe video mode) + retouch (~5 ms Tier A stack) + encode fits, but only
with the downsample-filter-upsample pattern and masks at proxy resolution —
i.e., the image engine's proxy architecture is already the right shape.

---

## 5. What this changes for the plan (recommendations)

1. **Target Tier B, floor Tier A.** State it in the product definition:
   FabSoften-class adaptive-natural is the quality bar; Bold Glamour-class
   generative synthesis is explicitly out of scope (already implied by the
   no-generative rule — make the competitive scoping explicit).
2. **Add two V0 bake-offs** to the technology evaluation: (a) mask cadence —
   per-frame BiSeNet vs every-N + track-warp vs landmark+color prior; (b)
   smoothing formulation — current frequency separation at proxy vs
   guided-filter downsample/upsample with high-pass reinjection, scored on the
   corpus flicker + texture-energy metrics.
3. **Add "attribute-aware auto-strength" as a V2 item** with its output
   routed through the temporal strength-smoothing layer — it is the single
   biggest published lever for "natural," and it is cheap.
4. **Spot removal in video needs track-attached persistence** (canonical-space
   anchoring + lifetime), not per-frame detection; record this as a V2 design
   constraint.
5. **Expression-compensated reshaping** joins the V2 geometry gate: dampen
   warp strength from expression coefficients before temporal QA will pass.
6. **Tone ops inherit the repo fairness rule** — margin-above-baseline only;
   audit any borrowed SDK-style "whitening" curve against Fitzpatrick V–VI
   corpus clips in V0.

## Sources

- FabSoften (CVPRW 2020): https://research.samsung.com/research-papers/FabSoften-Face-Beautification-via-Dynamic-Skin-Smoothing-Guided
- ABPN (CVPR 2022): https://openaccess.thecvf.com/content/CVPR2022/papers/Lei_ABPN_Adaptive_Blend_Pyramid_Network_for_Real-Time_Local_Retouching_of_CVPR_2022_paper.pdf
- Bold Glamour analysis (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC12558173/
- TRTC beauty filters engineering overview: https://trtc.io/blog/details/beauty-filters-explained
- Real-time skin smoothing filter patent (US 9390478): https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9390478
- Intel scalable real-time beautification patents: https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10152778 , https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10453270 , https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11328496
- GPUPixel: https://github.com/pixpark/gpupixel
- GPUImage BeautifyFaceDemo: https://github.com/Guikunzhi/BeautifyFaceDemo
- YUCIHighPassSkinSmoothing: https://github.com/YuAo/YUCIHighPassSkinSmoothing
- VToonify (parsing-map smoothing, flicker loss): https://arxiv.org/pdf/2209.11224
- BlazeBVD deflickering: https://arxiv.org/pdf/2403.06243
- Blind face restoration → video benchmark: https://arxiv.org/pdf/2410.11828
- Deep Video Prior (already cited by the plan): https://arxiv.org/abs/2007.01466
- Effect House Enhance Mode (platform-level retouch): https://effecthouse.tiktok.com/learn/guides/support/faqs
- Vendor-claim caveat source (unverified "FaceFormer/TextureGAN" naming): https://www.alibaba.com/product-insights/ai-beauty-filters-on-tiktok-vs-instagram-which-algorithm-alters-skin-texture-more-realistically.html

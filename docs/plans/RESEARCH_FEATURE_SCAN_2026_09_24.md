# Retouch feature scan: what to build next

Date: 2026-09-24. Source inspected: `bf76726eea4e33f4d824955a68c0b273c1f28c53` (main).
Scope: outward research by eight parallel agents (competitor tools, photographer
complaints, cosplay retouching, Instagram/Reels/TikTok output, culling and workflow,
open models that run locally, generative AI and labeling rules, video). Every
"retouch has / lacks" claim below was checked against the code on `main`.
Research only: no code, models, or photos were changed or processed.

Left out because they were being built in parallel on 2026-09-24: live batch
progress, the post-batch review page, splitting shoots by capture time, and
face-aware social crops.

## Top three

1. **Flag closed eyes and pick the frame with the sharpest face** (culling, S).
   Blink checking is the biggest culling time sink, and Aftershoot and FilterPixel
   both lead with it. `retouch/eye_visibility.py` already has a calibrated EAR gate
   (`_MIN_EAR=0.285`), but `retouch/face_quality.py` still hard-codes
   `eyes_open="uncertain"` with `blink_analysis_deferred`. Burst ranking in
   `retouch/shoot_intelligence.py` scores whole-frame sharpness only, so a sharp
   background can beat a sharp face, while per-eye sharpness already exists in
   `face_quality.py`.
2. **Switch on object removal, AI upscaling and AI denoise** (models, S–M).
   LaMa (`spot_heal.py::LamaHealer`), Real-ESRGAN and NAFNet (`enhance.py`) are
   integrated, but `models/manifest.json` marks `lama_inpaint`, `sr_real_esrgan`
   and `denoise_nafnet` as `"availability": "unavailable"` with no URL, so they fall
   back to classical methods. Licenses (Apache-2.0, BSD-3-Clause, MIT) allow shipping.
   The work is sourcing, hosting and checksumming the weights.
3. **Export a recipe as a `.cube` LUT** (output, S). Lets Reels graded in
   CapCut, Resolve or Premiere match the photo set. `retouch/lut.py` has
   `load_cube` only; there is no writer. Pure color math on the existing grading
   code, with no video processing.

## Full ranked list

Ranking weighs how often the problem shows up in a cosplay batch, how much
existing code it reuses, and license or runtime risk. Effort: S is a few days,
M is one to two weeks, L is larger or needs a new model.

| # | Feature | Area | Pain it fixes | What retouch has today | Effort |
|---|---|---|---|---|---|
| 1 | Closed-eye flag + face sharpness in burst picks | Culling | Manual blink checks; sharp background outranks sharp face | EAR gate exists; `eyes_open` hard-coded "uncertain"; whole-frame burst ranking | S |
| 2 | Turn on LaMa, Real-ESRGAN, NAFNet | Models | Con clutter and props; soft crops; high-ISO hall shots | Integrations written; weights unavailable, classical fallback | S–M |
| 3 | Export a recipe as a `.cube` LUT | Output | Reels don't match the photo grade | `lut.py` reads `.cube` only | S |
| 4 | Glasses, goggle and visor glare removal | Cosplay | Flash reflections on character eyewear | Nothing; Aftershoot and PortraitPro ship it. Wig deglare and `specular.py` are starting points | M |
| 5 | Body paint support (blue, green, grey skin) | Cosplay | Patchy paint; grey paint pushed toward human skin | `skin_chromophore.py::paint_coverage_even()` has no callers; the paint-protection gate in `engine.py` is chroma-only (OKLCh C > 0.18), so desaturated paint slips through | S–M |
| 6 | Match a whole set to one hero frame | Output | Color jumps between carousel slides (IG 20 slides, TikTok photo posts 35) | Per-image color transfer and look extraction; nothing set-level | S |
| 7 | Watermark / credit overlay | Output | Reposts lose attribution | Nothing | S |
| 8 | XMP sidecars with picks and ratings | Workflow | Culling decisions don't reach Lightroom / Capture One | No XMP code | M |
| 9 | Red-eye fix | Retouch | On-flash convention shots | Nothing | S |
| 10 | "What was changed" report, then signed Content Credentials | Trust | Disclosure rules; IG and TikTok read C2PA | `io.py` copies the camera's C2PA block unchanged onto edited pixels, so it will fail verification | S then M |
| 11 | Learned matting for wigs (MODNet or InSPyReNet) | Models | Hair/wig edges in background replace | MediaPipe mask + classical closed-form matting | M |
| 12 | Real depth for lens blur (Depth Anything V2 Small) | Models | Fake-looking blur | `background.py::lens_blur` uses a radial falloff, not a depth map | M |
| 13 | Prosthetic edge blending (elf ears, horns) | Cosplay | Visible silicone seams | Nothing; `cosplay_moat.py` wig-lace blend is the pattern; ears need new localization | M |
| 14 | Auto-find and heal straps, zips, tags | Cosplay | Repeated manual fix per batch | Heal/patch tools; no detection | M |
| 15 | Shoot-wide duplicate detection | Culling | Near-identical poses minutes apart | Duplicates caught only inside a burst | M |
| 16 | Group a shoot by cosplayer | Culling | Subjects rotate during con days | No face-identity code; needs a shippable face-recognition model | L |
| 17 | Click-to-select masks (SAM 2.1) | Models | Fixing what automatic masks miss | All masking is automatic; Apache-2.0 with an Apple CoreML build | L |
| 18 | Convention crowd removal | Cosplay | Busy halls behind the subject | Classical fill handles small gaps; needs generative fill (see decisions) | L |

## Decisions for the owner

- **Video stays dropped?** `docs/reference/ROADMAP.md` dropped video on
  2026-06-23. Evoto shipped tracked AI video portrait retouching on 2026-09-22
  with a DaVinci Resolve integration. Recommendation: keep full video retouching
  out; build the `.cube` export, and consider a color-only "grade this clip like
  recipe X" later since it carries no flicker risk.
- **Generative editing stays out?** `retouch/neural_boosters.py` records the
  2026-07-03 classical-only decision. FLUX.1 Kontext dev is non-commercial for
  local use, and Qwen-Image-Edit is reported as research-licensed (not verified
  from its license file). Recommendation: keep it; LaMa (pick 2) covers most of
  the benefit. Crowd removal waits.
- **Let culling look at eyes?** `rank_burst_candidates` deliberately avoids eyes,
  expression and identity. Recommendation: allow a closed-eye flag the user
  reviews, never an automatic reject, matching the module's evidence-only design.

## Model licenses for the standalone build

Several models have an Apache or MIT code repo but non-commercial weights. Check
the checkpoint itself, not the repo badge.

Safe to ship: LaMa (Apache-2.0), Real-ESRGAN (BSD-3-Clause), NAFNet (MIT),
MODNet (Apache-2.0), InSPyReNet (MIT), BiRefNet (MIT; its ONNX build is reported
slow), SAM 2.1 (Apache-2.0), Depth Anything V2 **Small** (Apache-2.0), IC-Light
(Apache-2.0, but diffusion is impractical on CPU).

Traps: RMBG-2.0 (CC BY-NC), Depth Anything V2 Base/Large (CC BY-NC), CodeFormer
(S-Lab, non-commercial), GFPGAN weights (depend on non-commercial StyleGAN2 /
DFDNet), jonathandinu segformer face-parsing (model card says research only),
AnimeGANv2 (non-commercial), Depth Pro (Apple custom license), FLUX.1 Kontext dev
(non-commercial locally). BEN2's shipped checkpoint terms could not be confirmed.

## Surprises

- Three AI models are fully coded but never download: the cheapest upgrade here.
- A body-paint evening function exists with no callers.
- Lens blur uses a radial falloff, so it is not true depth, despite looking
  comparable to Lightroom's AI Lens Blur on paper.
- Colored-contact handling is more mature than expected (`eye_artifact_safety.py`).
- Retouch4me's 531 Trustpilot reviews rarely mention price; the top complaints
  are forced updates and AI mistakes. `retouch/update_check.py` is already
  non-blocking.
- Adobe has an open community bug where its AI people filter misses a person of
  color in a group photo, so skin-tone parity is a real edge.
- EVA-foam armor seams are fixed during the build, not in post; that idea was dropped.

## Sources the agents opened

Competitors and complaints:
- https://aftershoot.com/roadmap-2026/ and https://aftershoot.com/retouching/
- https://www.evoto.ai/features/portrait-retouching and https://www.evoto.ai/features/video-portrait-retouching
- https://retouch4.me/retouchplugins, https://www.anthropics.com/portraitpro/, https://www.captureone.com/en/explore-features/whats-new
- https://fstoppers.com/lightroom/ai-lens-blur-lightroom-controls-make-it-look-real-719228
- https://www.digitalcameraworld.com/photography/photo-editing/big-luminar-neo-update-supercharges-portrait-retouching-and-adds-next-gen-bokeh-tool
- Reviews: janakukebal.com (Evoto, Retouch4me), tryretouchlab.com, Trustpilot retouch4.me, G2 Luminar Neo, Shotkit PortraitPro, Adobe Community bug report

Culling, output and video:
- https://filterpixel.com/best-ai-photo-culling-software
- https://support.aftershoot.com/en/articles/16556786-how-to-use-the-people-filter
- https://buffer.com/resources/instagram-image-size/, https://metricool.com/instagram-increases-carousel-content-limit/, https://wavegen.ai/tiktok-slideshow-size
- https://vanityfilter.com/blog/skin-smoothing-davinci-resolve, https://www.digitalanarchy.com/beauty-box-video, https://www.capcut.com/tools/face-retouching

Cosplay, models and labeling:
- https://www.therpf.com/forums/threads/how-to-blend-the-edges-of-reusable-silicone-elf-ears.332185/
- https://fstoppers.com/lifestyle/quick-tips-improve-your-cosplay-convention-photos-192081
- https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev and the LICENSE files / model cards of each model above
- https://artificialintelligenceact.eu/transparency-rules-article-50/
- https://www.legifrance.gouv.fr/jorf/id/JORFTEXT000034580217
- https://petapixel.com/2021/07/02/not-disclosing-that-a-photo-was-retouched-is-now-illegal-in-norway/
- https://editorsweblog.org/2026/04/12/c2pa-adoption-tracker-platforms-content-credentials-2026

Limits: model speed claims were not benchmarked on the owner's hardware.
Qwen-Image-Edit's license and Imagen's "learns your style" feature came from
search results, not pages the agents opened.

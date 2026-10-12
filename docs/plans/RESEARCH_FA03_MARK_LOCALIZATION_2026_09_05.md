# FA-03: facial mark localization and eyeliner confusion

Date: 2026-09-05. Research only; no production changes or candidate classifier implementation.
Baseline: `e072fdb9f26ab32fd21ecbe5d405c65f3ed2921a`.
Continues FA-02 §6's detector blocker. Closed FA-01 decisions remain closed.

## Decision

The DSCF2310 failure is reproduced on a fresh engine capture. The two previously described eyeliner wings are, more precisely at native crop inspection, components of the drawn lower lash/eyeliner structure. Both receive `beauty_mark=0.70` and `blemish=0.70`; dictionary insertion order selects beauty mark, which `detect_marks()` renames `mole`. No makeup hypothesis competes, and no shape or eye-context feature influences classification.

Start with an offline semantic/geometry reranker of frozen candidates. Compare against a small scored classifier using the same candidates; change candidate generation only in a separate recall experiment. Do not turn either into an automatic FA-02 exclusion until independently reviewed supports and preservation-harm checks pass. This research permits an FA-02 experiment with manually accepted supports; it does **not** establish that automatic default-path exclusions are safe.

## 1. Current code, precisely

Primary locations: [`freckle.py`](../../retouch/freckle.py), [`marks.py`](../../retouch/marks.py), [`parsing.py`](../../retouch/parsing.py), [`perf_optimizations.py`](../../retouch/perf_optimizations.py), [`harmony.py`](../../retouch/harmony.py).

### Candidate generation

`detect_marks()` (`marks.py:252`) requires uint8 BGR, normalizes/resizes the supplied mask, and calls `FreckleRemover.classify_anomalies()`. Without a mask it searches the entire image. Current pipeline callers supply `skin_n`.

`FreckleRemover._detect_components()` (`freckle.py:75`):

1. Convert BGR to grayscale; Gaussian local mean with fixed 11×11 kernel.
2. Candidate if `(local_mean - gray) / max(local_mean, 1) > 0.05`.
3. Intersect with `skin_mask > 0.3`.
4. HSV redness uses H=0–15 or 165–180, S>=60, V>=50. However `red_blemish = redness AND candidates`, followed by `candidates OR red_blemish`, is algebraically identical to `candidates`. There is **no independent redness proposal**, despite the comment.
5. Open using a fixed 2×2 elliptical kernel, then 8-connected components. OpenCV 4.11 produces kernel `[[0,1],[1,1]]`; at this size it is asymmetric. The resulting support can differ from original threshold pixels, including displacement. No post-opening skin intersection occurs.
6. Discard area <2 px; process at most 2,000 eligible components in connected-component scan order. There is no hard maximum mark area. Area ranges below add scores; they do not reject larger components.

Kernel sizes and class area ranges are absolute pixels, not face-normalized. This matters when the same portrait is resized or face crops have different resolutions. Small pale/red blemishes can fail before classification; geometry filtering cannot recover them.

### Classification and confidence

`_classify_anomaly()` (`freckle.py:126`) samples actual component pixels in OpenCV LAB units (L=0–255, a/b centered at 128). Reference statistics come from **all mask-selected skin**, not a local ring:

`a_norm = (component_mean_a - skin_median_a) / (skin_std_a + 1e-6)`

`L_norm = (component_mean_L - skin_median_L) / max(skin_std_L, 0.1*skin_median_L, 1)`

`chroma = hypot(component_mean_a - 128, component_mean_b - 128)`

| Class | Additive rules |
|---|---|
| freckle | +0.50 if area 4–25; +0.45 if a_norm>0.3; +0.05 if L_norm>=−2 |
| beauty_mark | +0.20 if area 10–70; +0.40 if a_norm<−0.3; +0.50 if L_norm<−2 |
| blemish | +0.20 if area 6–80; +0.30 if absolute chroma>25; +0.50 if component L std>8 |
| noise | score 0.90 if area<4 |

Winner is `max(scores, key=scores.get)` in order freckle, beauty_mark, blemish, noise. Ties silently favor the earlier class. Confidence is `min(1, winning_score)`: no normalization across classes, calibration, competing-score margin, or uncertainty estimate. The beauty score can reach 1.10 before clipping. Comments about flat marks do not impose a low-variance requirement on beauty marks or freckles; only blemish gets an internal-variance term.

Results below default 0.60 are dropped, not emitted as uncertain. `detect_marks()` maps beauty_mark→mole, blemish→acne_blemish, noise→unknown, and freckle→freckle. Thus `unknown` primarily means tiny/noise under these rules, not semantic ambiguity. The taxonomy includes drawn makeup, scars, and other classes that this classifier cannot produce.

### Features recorded but not used to decide

`marks.py:175–249` computes bbox-derived eccentricity, whole-skin-relative median LAB L/a margins, optional melanin/hemoglobin margins, and neighbor counts within 0.12×face_width. These are computed **after** class/confidence selection. They do not rerank candidates. Edge sharpness and symmetry are constant zero placeholders. Eccentricity uses axis-aligned bbox dimensions, not the actual component covariance; orientation, contour compactness, circularity, solidity, boundary following, and eye-relative scale are absent. Bbox appearance mixes mark and non-mark pixels. There is no b-margin feature, although b contributes to the earlier absolute-chroma rule.

`face_width` only normalizes record area and cluster radius; if omitted it is estimated as sqrt(mask area above 0.3). Production callers omit the explicit width even though the face core has `shifted_face.ied * 2.5`. Optional mole/blemish/stray-hair masks are appended as separate records, default confidence 1.0, without cross-source deduplication or conflict resolution. They are not supplied by the current `detect_marks()` production call sites. Optional pigment maps are also not supplied there; pigment decomposition is available elsewhere/later, not already computed for the initial detector call.

### Semantic information available upstream

`FaceRegions` (`parsing.py:378`) provides eyes, brows, hair, skin, lips, under-eye, and crow's-foot masks. The core also has shifted MediaPipe landmarks and inter-eye distance. These can support local eye contours, corners, tangents, signed boundary distances, component overlap, and normalized size. Currently only the collapsed skin mask reaches mark classification. Skin excludes eyes/brows, but drawn eyeliner on adjacent skin legitimately remains inside it. Skin membership is not evidence of natural pigmentation.

BiSeNet region and boundary confidence summaries already exist. They are explicitly observational, not calibrated per-pixel error probabilities; no production gating change is proposed here. Dense class logits/label maps exist during parsing but are not retained as a full probability field on `MarkRecord` or `FaceRegions`. There is no dedicated lash/eyeliner/beard class. Landmark eye visibility is available, but does not identify the material causing a dark stroke. These limits must be encoded as missing/uncertain context in an experiment.

### Downstream trust and support inflation

| Consumer | What it trusts and consequence |
|---|---|
| Initial policy protection (`perf_optimizations.py:413`) | Detects on pristine face canvas, compiles preserve mask. Guides the guided-engine base-smoothing protection and masks for flatten, micro dodge/burn, redness, HB operations, vein attenuation, equalize, hue/tone unification. Some operations also derive their statistical reference from the reduced mask. |
| Freckle stage (`:620`) | Redetects on the already modified canvas; uses policy preserve mask. Separately, legacy `FreckleRemover.remove()` classifies again, protects beauty marks with circles, and heals freckles. |
| Blemish stage (`:873`) | Redetects later; subtracts policy preserve mask before a **separate** `BlemishRemover._detect()` and repair. `acne_blemish` records are not directly the auto-heal targets. |
| Harmony H4 (`harmony.py:195`) | Counts policy-preserve records before/after, same-class matching within 6 px; thresholds 0.6/0.4. Its existing detector-derived denominator is not independent mark-retention ground truth; matching is not one-to-one. |
| Micro-texture restoration (`perf_optimizations.py:670`, `skin.py:985`) | Currently consumes no mark classification. Unconditional mark-derived exclusion was the rejected FA-02 proposal, not shipped behavior. |

`compile_mark_policy()` reconstructs a filled axis-aligned ellipse from bbox dimensions around centroid, discarding the true component support and orientation. `protect_identity` preserves moles, drawn makeup, scars, and unknowns; freckles have an attenuation policy. Only preserve masks are consumed at the above pipeline policy sites; compiled heal/attenuate/enhance outputs are not automatic new repair paths. Low confidence becomes unknown only when a retained record reaches policy resolution; already dropped candidates cannot be recovered there.

The separate blemish detector (`blemish.py:133`) has a face-scaled local-mean kernel, absolute grayscale deviation threshold, opening/closing, hard area filtering and dilation. Its redness union has the same redundancy. It does not share the four-class winner computation. A mark-classification improvement alone does not certify this detector's edit precision.

## 2. Reproduction and exact DSCF2310 mechanism

Fresh `RetouchEngine.process(source, recipe='xiaohongshu')`, observing `SkinProcessor.restore_micro_texture` in memory while calling the original unchanged method. Captured pre-smoothing image is byte-identical after uint8 conversion to the prior FA-02 saved image. One captured face; OpenCV 4.11.0, NumPy 1.26.4. No repaired portrait written. The observation script is `/tmp/fa03_mark_detector_audit.py` (temporary, not production).

Source: `/private/tmp/retouch-meitu-final-0071447/DSCF2310/00_source.jpg`.
SHA256: `78ca785a1898227ebd31fca26b962824a26878cfe8b1c12dc4cdf84b18a588ec`.
Captured canvas: 675×1262. Coordinates below are in that canvas, not the original full-resolution photograph.

Evidence: [full audit JSON](../../test_output/fa03_mark_localization_20260905/DSCF2310_audit.json), [visually reviewed diagnostic crops](../../test_output/fa03_mark_localization_20260905/DSCF2310_candidates.png). The two supports are visibly thin drawn lower-eye lines, not isolated round skin spots. Remaining candidates have not been relabeled as true freckles/moles.

| Measurement | Viewer-left eye component | Viewer-right eye component |
|---|---:|---:|
| bbox x,y,w,h | 236,381,18,12 | 367,428,22,13 |
| Component area | 46 px | 62 px |
| L mean / L norm | 98.96 / −2.505 | 111.71 / −2.134 |
| a norm | +1.856 | +0.117 |
| Absolute chroma | 20.93 | 14.87 |
| Internal L std | 13.03 | 15.95 |
| Scores: freckle / beauty / blemish / noise | .45 / .70 / .70 / 0 | 0 / .70 / .70 / 0 |
| Axis-aligned bbox aspect | 1.50 | 1.69 |
| Component PCA major/minor ratio | 6.89 | 8.14 |
| Component fill in bbox | 21.3% | 21.7% |
| Contour circularity, 4πA/P² | .129 | .163 |
| Minimum distance to eye mask >.3 | 4.12 px | 4.24 px |
| Eye overlap / pixels outside skin>.3 | 0 / 0 | 0 / 0 |
| Compiled preserve ellipse area | 189 px | 233 px |

Exact chain: locally dark line → survives morphology and skin gate → area adds .20 to both beauty and blemish → L_norm<−2 adds .50 to beauty → internal L std>8 adds .50 to blemish → tied .70 → beauty wins by insertion order → adapter emits mole. The cool-color beauty rule is **not** involved in either case. Full-skin baseline is L median 185, L std 34.34, a median 142, a std 3.736.

There are 55 accepted records, matching FA-02. Morphology leaves only 35/46 and 50/62 target pixels on the original threshold support, illustrating why features must be measured on both the proposal response and final support. The actual component shape, not bbox aspect, reveals elongation. Circularity here uses OpenCV contour area/perimeter; at tiny sizes it is a raster-dependent diagnostic, not a universal threshold.

A diagnostic local reference ring (15×15 dilation, other candidates excluded, skin>.8) gives median ΔLAB of (−40,−2,−2) and (−60,−1.5,−3). This does not establish a classifier; it shows that the first component's globally positive a-normalization is not a locally red anomaly. A local ring alone still cannot tell black makeup from a dark mole.

Both true supports remain entirely inside skin after clipping. Ellipse reconstruction expands their nominal areas by 4.11× and 3.76× before any feathering. This is a separate support-quality amplifier, not a reason to reopen FA-01.

Overlap-accounting correction: the earlier FA-02 note did not fully specify thresholds/rounding for its 8 candidates and 654→389 pixels. Eight centroids reproduce using the current feathered dimensional mask, threshold .5, and integer-truncated centroids (rounding yields seven). That definition yields 669 preserve-overlap pixels and 447 after thresholding skin-weighted preserve at .5, not 654/389. Raw-union>.3 with rounded centroids instead gives 12 candidates and 893→651 pixels using skin>.3. Do not carry forward the earlier exact area numbers or its ~40% reduction as fresh measurements. The essential result survives: both eyeliner supports have nonzero restoration-zone overlap, and skin clipping does not eliminate them. Prefer soft overlap mass plus explicitly defined binary summaries in future experiments.

## 3. Literature that changes the experiment

These are localization/classification sources, not a retouch survey. Proposed applications below are inferences, not claims of published cosplay accuracy. No external model or dataset was downloaded or run.

| Primary source | What to use | What it does not establish |
|---|---|---|
| Jain & Park, ICIP 2009, [Facial Marks: Soft Biometric for Face Recognition](https://biometrics.cse.msu.edu/Publications/SoftBiometrics/JainParkFacemarks_ICIP09.pdf) | Landmark/feature masking followed by LoG and morphology directly addresses facial mark proposals. Use scale-space blob evidence and spatial context as separate signals. Author-hosted indexed abstract inspected; full PDF fetch failed. | Recognition improvements are not mark precision/recall or proof that an eye exclusion preserves periocular moles. Do not copy blanket feature masks. |
| Lowe, IJCV 2004, [Distinctive Image Features, §4.1](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) | DoG produces strong edge responses; Hessian principal-curvature ratios distinguish edge-like from well-localized extrema. Supplies a complementary blob-versus-edge cue to component PCA. | A corner/line rejection method does not determine biological mark type. Test as a feature, not an unconditional veto. |
| Ko & Cheoi, 2021, [Image-processing based facial imperfection region detection and segmentation](https://link.springer.com/article/10.1007/s11042-020-10208-w) | Publisher abstract describes LAB illumination correction, skin extraction, Gabor responses and DBSCAN. Motivates oriented texture plus explicit illumination-stress controls. | Full article was not available through publisher access; no parameter transfer or quantitative robustness claim is adopted. No demonstrated makeup/identity separation. |
| Wang et al., 2022, [ILoveEye](https://arxiv.org/pdf/2203.13592), §4 | Eye-contour-relative construction of upper, lower and winged eyeliner motivates tangent alignment, corner continuation and eye-normalized size. | It generates guidance, not an existing-makeup segmenter. Its chosen 15° angle and 12%-eye-width wing are design presets, not valid universal detection thresholds. |
| Gazeau et al., MICCAI 2024, [AcneAI](https://papers.miccai.org/miccai-2024/paper/2216_paper.pdf), §§2–3 | Separates localization, individual lesion scoring and aggregate severity; documents annotation problems in ACNE04 and adjacent-lesion merging. Strong reason to evaluate supports independently from class and repair. | Locator intentionally includes acne-like lesions such as moles. Public conference [resource page](https://papers.miccai.org/miccai-2024/042-Paper2216.html) lists code N/A. Not a verified downloadable replacement. |
| Authors' [ACNE04-v2 annotations](https://github.com/AIpourlapeau/acne04v2), based on [Wu et al. ACNE04](https://github.com/xpwu95/LDL) | Practical public localization data: 1,204 images, 32,443 center/radius annotations, corrected small-lesion coverage. Enough to justify a future learned acne-proposal challenger. | Circles are not exact masks or eyeliner/mole labels. Original project specifies academic use; other use requires contacting authors. Do not assume unrestricted Retouch training rights. |
| Gangrade, Kag & Saligrama, AISTATS 2021, [Selective Classification via One-Sided Prediction](https://proceedings.mlr.press/v130/gangrade21a.html) | Explicit reject option with class-specific false-positive control motivates class-wise acceptance and risk/coverage evaluation. | This does not calibrate current .70 scores or guarantee safety under cosplay domain shift. |

Excluded from the recommendation: image-level acne severity models, generative makeup transfer/retouching, and dermoscopy-only diagnosis. They do not supply the needed portrait mark-type distinction. DSDH's [author preprint](https://arxiv.org/abs/2301.12219) leaves its code/data link temporarily anonymous; it is not counted as a verified runnable challenger.

## 4. Ranked strategies

### 1 — Semantic/geometry rules with explicit ambiguity: first experiment

Keep the current candidates fixed to isolate classification. Retain their true masks and measure PCA/minimum-area-rectangle elongation, contour compactness, fill/solidity, skeleton length-to-width, endpoint continuation and local structure-tensor/Hessian evidence. Use width/area relative to each visible eye and inter-eye distance, not absolute pixels alone.

Measure eye/lash-line/brow/hair proximity and overlap separately. Test whether a candidate follows a nearby eye/brow tangent or continues into a longer dark stroke, including adjacent fragments disconnected by the current threshold. High elongation **plus** boundary alignment/continuation supports makeup/hair; proximity alone never rejects a candidate. Compare structure to both eye and brow/hair context. A compact isolated spot with a clean surrounding skin ring remains eligible even immediately beside an eye. A compact drawn dot or a mole merged with eyeliner remains ambiguous.

Prefer coarse operational labels where fine attribution is unsupported: identity-like spot, blemish-like spot, eye makeup, hair/brow contamination, other cosmetic mark, illumination-like, ambiguous. Hair versus drawn eyebrow may itself require abstention; both differ from an approved skin defect. Stable identity is not observable from one image with certainty. Subject confirmation or multiple captures without the same makeup can strengthen the reviewed label.

Advantage: interpretable, cheap, directly tests the observed failure. Risks: fragmented makeup becomes compact; crossing hair becomes blob-like; profile foreshortens true marks; a parser boundary can be wrong. Do not promote measured PCA ratios from one portrait into tuned production constants.

### 2 — Scored classical classifier: next if rules do not generalize

Fit a regularized multinomial logistic model first; a shallow boosted-tree model can be a later comparator if enough reviewed examples exist. Inputs: true-support shape, normalized distances, parser overlap/missingness, boundary alignment, local ΔL/Δa/Δb, ring MAD, ring completeness, oriented texture/coherence and multiscale response persistence. Compute local reference statistics from several skin-only annulus sectors, excluding other proposals/features, and compare with a robust local illumination plane. Record when a safe local baseline cannot be estimated.

Keep scores for all classes. Use held-out calibration, class-specific acceptance criteria and winner-versus-runner-up separation; otherwise emit ambiguous and a reason. The exact current .70/.70 ties should not become confident identity marks. Do not relabel all ties as blemishes. Compare coverage at a fixed upper tolerance for harmful false assignments, not accuracy only among easy accepted samples.

This approach can combine moderate cues that no single rule resolves, but needs more labels than the rule experiment. Avoid claiming calibrated probabilities from a handful of examples. Split by subject/session; multiple crops and synthetics from one source stay together. Missing/uncertain parser context is evidence of uncertainty, not evidence of clean skin.

Candidate-generation ablation is separate: compare the frozen dark-only proposals with face-scaled LoG/DoG dark proposals **unioned with an independent local chroma-anomaly stream**. Never intersect that stream back into dark candidates. Audit normal blush, colored paint and illumination as negatives. Compare proposal recall before comparing class precision.

### 3 — Learned acne localization challenger: conditional, third

ACNE04-v2 makes a small detector trained on lesion centers/circles a practical *future* challenger, subject to permissible use. It can test whether faint or crowded blemishes missed by classical proposals are recoverable. It cannot supply the requested six-way taxonomy without additional portrait annotations and hard negatives for makeup, hair, shadows and genuine identity marks. Use a proposal model plus a separate candidate classifier; expose disagreement for review.

No compatible public weights resolving this full distinction were verified in this bounded search. AcneAI is methodological evidence, not a ready local model. Expected costs are annotation/training, input-resolution sensitivity, runtime/ONNX evaluation and domain-shift auditing. Do not train or integrate it before the simpler benchmark establishes what is missing.

## 5. Minimum reviewed benchmark and first experiment

Proposed pilot floor: **30 real face portraits, at least 20 subjects**, plus **192 controlled synthetic variants**. These are experiment-planning counts, not an assertion that the corpus is already available or a statistical production certification.

| Real-image allocation | Required content |
|---|---|
| 6 heavy eye-makeup portraits | Include DSCF2310; varied wing directions, lower-lash drawing, false lashes, closed/partly occluded eyes. Existing DSCF2306/2308/2362/2365 are candidates for review, not assumed labels. |
| 6 ordinary portraits | Little/no makeup, varied observable skin color and facial hair; include clean negatives. |
| 6 identity-mark portraits | Confirmed natural moles/freckles/stable scars; prioritize near-eye and near-brow marks. Obtain confirmation/repeated-image evidence where feasible. |
| 6 blemish portraits | Small dark and non-dark inflamed-looking spots, clustered spots and low contrast. Do not substitute image-level acne grades for spot annotation. |
| 6 challenge portraits | Profile, directional/colored light, hair across skin, dark face paint/drawn dots. Tags overlap earlier groups; primary allocation sums to 30. |

Reuse existing local portraits and prior lighting/occlusion lists where suitable; this does not require processing the full 83-image corpus. Existing documents explicitly lack confirmed natural-mole evidence. Small records called freckles in previous detector output are not confirmed positive labels. Periocular genuine-mark coverage remains a real collection/review requirement, not something synthetic marks can satisfy.

Instance targets across those portraits: at least 100 confirmed identity marks across >=5 subjects (>=20 near eyes/brows across >=4 subjects), 60 blemishes across >=5 subjects, 40 eye-makeup structures, 30 hair/brow structures, 20 other cosmetic marks and 30 shadow/illumination distractors. Add portraits if these minima or split coverage cannot be met. One densely freckled face must not dominate the macro results. Report natural moles, freckles and scars separately when counts permit; absent categories remain unevaluated.

Synthetic design: 6 reviewed clean host crops ×4 edge-to-boundary distances (0, .05, .15, .40 eye widths) ×4 signals (compact pigment-like dot, red spot without necessary luminance decrease, oriented dark stroke, soft shadow patch) ×2 sizes =192. Hosts cover both eye and brow boundaries, different skin colors, lighting and poses. Balance contrast and stroke angle within the grid and log them; retain unmodified hosts as paired controls. Composite only in a future authorized experiment. These are controlled appearances, never labeled as genuine biological moles/acne. Keep all derivatives with their host's split.

Partition before tuning: approximately 12 development, 8 calibration, 10 locked test images, with subject/session isolation overriding exact image counts. DSCF2310 belongs to development because this failure is already known. Require near-eye genuine positives and eye-makeup negatives in the locked test set; otherwise no near-eye safety claim. Two reviewers annotate source images at native and contextual zoom; adjudicate conflicts, retaining an explicit uncertain truth label when identity/material cannot be established. Annotate all relevant objects and clean distractor regions independently of detector output, including missed spots, and retain exact support polygons where visible.

Recommended first authorized experiment:

1. Freeze current proposals at both default acceptance and a permissive logging setting; save every class score and rejected candidate. Measure candidate recall independently.
2. On the reviewed development subset, compare current class selection, semantic context only, oriented shape only, combined context+shape, and the scored classifier. Never modify repair during this comparison.
3. Include near-eye compact positives and synthetic boundary-distance pairs immediately. An approach that removes the DSCF2310 errors by losing nearby real marks fails the experiment.
4. Freeze rule/model choices on development, calibrate acceptance separately, then run the locked set once. Report support and consumer-risk metrics below. If the minimum genuine-mark evidence is unavailable, finish the negative-case diagnostic but label positive safety untested.

No candidate strategy or benchmark beyond DSCF2310 reproduction was executed in this research tranche.

## 6. Metrics, harm, and FA-02 gate

Use one-to-one matching, fixed before evaluation. For usable support masks, match at IoU>=0.30 and also report IoU>=0.50 sensitivity; small lesions need a separately reported center criterion (distance <=max(2 native pixels, .5 annotated equivalent diameter)) because one-pixel shifts can dominate IoU. Do not count the permissive center result as pixel-accurate localization. A predicted class mismatch contributes FP to predicted class and FN to true class. Unmatched detections are FP; unmatched annotations are FN; duplicate matches are FP. For continuous liner fragments, also aggregate object-level connected structures and overlap area so fragmentation cannot improve the score.

Report per mark type: TP/FP/FN, precision, recall, F1, false detections/face and class confusion. Separately report proposal recall, accepted-class recall over **all** eligible truths, support IoU/coverage and excess support area. For abstention, report ambiguous count/rate over candidates and matched truths, acceptance coverage, conditional error among accepted predictions, and risk-versus-coverage curves. Abstained true instances are not silently omitted from automatic recall. Label-uncertain truth is reported separately, not forced into a class to boost accuracy.

Stratify by true mark type, makeup, visible skin color, lighting, pose, face scale, and eye/brow distance. Show raw denominators and subject-level summaries/bootstrap intervals; small strata cannot establish rare-event safety. Detector-driven H4 is not the scorekeeper.

Downstream preservation harm is a separate accounting layer:

- Identity/makeup/hair mistakenly eligible for removal or FA-02 exclusion: object count, affected support area, and overlap mass with the actual soft dimensional mask.
- Blemishes incorrectly protected: missed intended reduction, counted separately from identity damage.
- Unknowns lost before policy compilation: silent protection loss.
- Reconstructed ellipse/feather footprint spilling into neighboring feature pixels: excess support relative to reviewed annotation.

Only after detection is frozen should a separate FA-02 rendering experiment compare current restoration, restoration disabled and exclusions from **reviewer-approved reduction targets**, with identical source/smoothing/regions. Score mark contrast recovery, makeup/identity contrast retention, nearby texture loss and boundary artifacts independently of detection precision. Some genuine marks should be preserved, not treated as defects merely because they are moles; image class is not edit permission. Hold reviewed supports fixed when comparing repair algorithms later (FA-04).

Provisional pilot advancement criteria, to be agreed/frozen before running: both known liner cases cease being accepted as moles/blemishes; no accepted destructive/exclusion assignments on locked genuine near-eye marks or reviewed cosmetics; no new near-eye identity misses relative to baseline; useful coverage (report an example target >=80% of reviewed mark candidates) and blemish recall no more than 5 percentage points below baseline at matched harmful-FP burden. Abstentions remain visible failures of automation coverage, not hidden successes. Zero observed harm in 30 images is a pilot result, not a production guarantee.

**FA-02 remains blocked for automatic detector-derived exclusions today.** Manual-support FA-02 research can proceed without pretending the detector is solved. A future unblock needs independent mark-type evidence, retained true supports, an explicit ambiguous/action contract, and a separate successful rendering check. Merely changing .70 to another threshold, renaming eyeliner to `unknown` (currently preserved), or using all `protect_identity` pixels as a defect-exclusion mask does not satisfy that gate.

## 7. Main failure modes to watch

Compact cosmetic dots can be visually indistinguishable from natural marks in a single frame. Freckle clusters can merge into elongated strokes; fragmented eyeliner and crossing hairs can become compact blobs. Red makeup can resemble blemishes; genuine lesions may be achromatic or non-dark. Colored light and blush can corrupt both global and local skin references; shallow folds can have the same local darkness as marks. Profile and small faces destabilize shape and parser boundaries. Fixed-size morphology changes topology and sampled appearance. Sparse calibration can produce unjustified confidence, and detector-based retention metrics can reward preserving the wrong objects. These are reasons for controlled positive examples and explicit abstention, not for a blanket eye exclusion.

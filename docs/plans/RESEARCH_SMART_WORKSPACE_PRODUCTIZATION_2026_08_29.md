# Research & Implementation Plan — Smart Workspace Productization

**Date:** 2026-08-29

**Status:** Research complete; implementation is not authorized by this document

**Branch baseline:** `feat/color-science-k9-fix-and-frontier`

**Owner direction:** Research first; document the complete product and implementation plan before changing behavior

**Primary evidence:** Current repository, supplied Meitu Android workflow screenshots 1–12, supplied Meitu export, and official Meitu product listings

**Scope of this tranche:** Documentation only

---

## 1. Executive decision

Retouch should add a **Smart workspace** beside the existing **Classic workspace**. Smart is a compact, contextual presentation and orchestration layer over the engine that already exists. It is not a second processing engine, a replacement for Classic, or a generative beauty system.

The recommended product direction is:

1. Keep every current Classic control available and backward compatible.
2. Give Smart no more than five primary cards at one time and no more than three macro controls for the selected card.
3. Let a canvas tap, selected face, or explicit protection choose the editing context.
4. Recompute edits from an immutable base state so macro changes are reversible and never accumulate destructively.
5. Reuse the existing recipe, parameter, per-face, Safe Auto, Advanced Retouch, preview-cache, session, color, and delivery contracts.
6. Distinguish measured correction from artistic intent. The software may measure exposure, white balance, noise, and image structure; the user chooses appearance, geometry, makeup, and identity-sensitive changes.
7. Show the actual local model, deterministic method, fallback, and review status instead of applying a generic “AI” label.
8. Preserve export truth: preview remains a proxy; Full Quality remains the native-quality delivery path with its color, metadata, manifest, and inspection evidence.

The first implementation tranche should build and test the pure intent contract only. It should not redesign the GUI, alter engine output, or introduce new model dependencies until the contract is reviewed against real photographs.

### 1.1 Priority model

Priority and implementation sequence are related but not identical. A P0 item is a release blocker even when its final integration occurs after a visible P1 screen has been built.

| Priority | Meaning | Scope rule |
|---|---|---|
| P0 | Contract, correctness, safety, compatibility, or delivery blocker | Must be complete before Smart v1 can be called ready |
| P1 | Core Smart v1 user value | Build after its P0 foundation; required for the intended v1 experience |
| P2 | High-value expansion using the stable core | Cut or defer before weakening a P0 or P1 acceptance gate |
| P3 | Later product research | Not part of core Smart v1 |
| Parked | Explicitly excluded from the approved direction | Requires a separate proposal before implementation |

### 1.2 Priority-ranked backlog

This is the controlling priority order for scope and resourcing:

| Rank | Priority | Work item | Why it ranks here | Dependency | Tranche |
|---:|---|---|---|---|---|
| 1 | P0 | Product, intent, protection, and corpus contract | Prevents UI and engine work from encoding ambiguous or identity-sensitive behavior | None | T0 |
| 2 | P0 | Pure intent, macro, ownership, and capability contracts | Gives every later callback a deterministic and testable source of truth | T0 approval | T1 |
| 3 | P0 | Render lifecycle, stale-result rejection, cache identity, and session isolation | A fast-looking UI is unsafe if it can display or commit pixels from the wrong revision/session | T1 plus a minimal callback seam | T4 foundation |
| 4 | P0 | Session compatibility, Full Quality parity, manifest evidence, and model/fallback truth | Smart must remain reproducible and must not weaken existing delivery guarantees | T1; completed across shell and delivery work | T2 and T6 |
| 5 | P0 | Eye, occlusion, iris, mark, texture, hair, multi-face, and background safety calibration | Identity and artifact protection is a release gate, not post-launch polish | Corpus begins in T0; final integration needs T3 | T5 |
| 6 | P1 | Shared Smart/Classic shell and Active Adjustments | Establishes the compact mental model without removing Classic capability | T1 | T2 |
| 7 | P1 | Color vertical slice with responsive deterministic preview | Highest-value, lowest-identity-risk proof of the full Smart architecture | Ranks 2, 3, and 6 | T2 and T4 |
| 8 | P1 | Canvas context and explicit per-face selection | Removes irrelevant controls and differentiates Retouch without inventing a new algorithm | T2 and cached analysis context | T3 |
| 9 | P1 | Face card and face-aware Smart Analysis | High user value, but only after per-face scope and protection evidence are reliable | T3 and rank 5 release gate | T5 |
| 10 | P1 | Auto Polish with separate, inspectable components | Useful one-click outcome while preserving user control and preventing invisible stacking | T2, T4, and T5 | T5 |
| 11 | P1 | Background protection contract | Protects geometry and makes reshape behavior trustworthy before adding more effects | T3 and safety review | T3 and T5 |
| 12 | P2 | Clean card routed to Advanced Heal/Remove | Valuable, but the underlying manual workflow already exists and core Smart does not depend on it | Stable shell, lifecycle, and Advanced replay | T6 |
| 13 | P2 | Conditional Hair card | Surface only operations the active runtime can perform truthfully | Context routing and a reviewed supported operation | T3 and T6 |
| 14 | P3 | Deterministic Profile Photo workflow | Outcome-oriented expansion after core Smart is certified | T0–T6 | T7 |
| 15 | Parked | Free-form prompts, generative identity changes, new remote services, and automatic body shaping | Expands safety, privacy, reproducibility, and product scope without being necessary for Smart v1 | Separate future research | Not scheduled |

### 1.3 Dependency-respecting execution order

The practical build order remains:

1. T0 contract/corpus approval, including early safety-corpus preparation.
2. T1 pure intent foundation.
3. T2 minimal Smart/Classic shell and compatibility seam.
4. T4 lifecycle foundation demonstrated through the Color vertical slice.
5. T3 context and per-face integration.
6. T5 face-aware analysis, protection calibration, Face, Auto Polish, and Background safety.
7. T6 Clean integration and final delivery evidence.
8. T7 Profile Photo research only after core Smart certification.

If capacity is constrained, remove P2 and P3 scope first. Do not reduce stale-render, cache/session isolation, safety, backward-compatibility, model-truth, or Full Quality gates.

---

## 2. Research question

How can Retouch gain the speed and clarity demonstrated by the supplied mobile workflow without losing Retouch's stronger properties: explicit parameters, repeatability, local processing, model truth, face-aware safety, manual control, and professional delivery evidence?

The answer is not to copy the mobile interface literally. The useful pattern is **progressive disclosure around a large existing capability set**:

- Start with intent and context.
- Show only the controls relevant to the current intent.
- Keep the detailed control surface one deliberate switch away.
- Make every Smart action inspectable, reversible, and reproducible.

---

## 3. Evidence and limitations

### 3.1 Evidence used

- Supplied screenshots 1–12 show the Meitu renewal surface, Classic facial controls, Auto styles, main editor, profile-photo discovery, Smart cards, contextual sliders, and final save path.
- The supplied final file, `MEITU_20260829_202452457.jpg`, is a 1031 × 2304 baseline JPEG with an embedded sRGB profile and Meitu software metadata.
- Current Retouch source and documentation were inspected for GUI structure, parameter declarations, recipes, Smart Default, Advanced Retouch, per-face controls, Safe Auto, caches, sessions, render modes, workspaces, and delivery.
- Official Meitu mobile and desktop listings were used only to understand public positioning and the breadth of current competitor capability.

### 3.2 What the evidence does not prove

- The screenshots do not prove Meitu's latency, numerical quality, privacy behavior, model architecture, or offline behavior.
- The supplied Meitu result has no clean source pair in this research set. It cannot support a reliable before/after quality judgment or a claim about resolution loss.
- Static UI counts below are declaration counts from the 2026-08-29 worktree. They are not a runtime component census.
- The current worktree contains active eye-visibility and eye-artifact-safety changes. Their presence is not release certification.
- Performance targets in this plan are acceptance targets. They are not measurements of the current application.

### 3.3 Research boundary

This plan does not change code, models, defaults, recipes, thresholds, or images. Any implementation tranche begins only after product and safety review of its acceptance criteria.

---

## 4. Competitor workflow teardown

| Screenshots | Observed workflow | Useful pattern | Retouch response | Pattern not to copy |
|---|---|---|---|---|
| 1 | VIP feature and renewal discovery | Commercial features are grouped and discoverable | Keep optional capabilities discoverable in one capability panel | Countdown pressure, clipped copy, and commercial interruption inside editing |
| 2–3 | Classic face editor with Width, Lift, Smooth, Symmetry, Temple, and Cheekbones | One category and one short horizontal tool row at a time | Add contextual Classic filtering while retaining all controls | Ambiguous protection behavior and beauty-ideal geometry as a default |
| 4 | Auto styles with one Face strength | Preset first, intensity second | Map reviewed Retouch profiles to reversible macro amounts | Hiding the individual effects that a preset changed |
| 5 | Main editor with Smart/Classic switch, Save, history, and compact categories | Canvas-first hierarchy and a visible mode switch | Use one stable canvas shell shared by Smart and Classic | Treating Save as proof of native-quality or color-managed delivery |
| 6 | Profile Photo template gallery | Outcome-oriented discovery | Consider a deterministic Profile Photo workflow after core Smart is stable | Generative pose, wardrobe, or identity reconstruction inside the core editor |
| 7 | Smart mode with area guidance, four cards, and a prompt field | Contextual routing greatly reduces navigation | Route taps to reviewed cards and existing semantic masks | Free-form generative prompts or silent network/model dependency in v1 |
| 8–9 | AI Color card plus a single intensity | Clear selection and progressive disclosure | Use a macro amount over validated existing parameters | Calling deterministic color correction “AI” |
| 10 | Face card expands to Soften, V shape, and Jawline | A card can expose a small second level | Keep texture separate from explicit geometry controls | Automatically inferring a preferred face shape |
| 11 | Auto expands to Face, Filters, and Makeup | One high-level intent can expose component amounts | Show an Active Adjustments breakdown and allow component override | Invisible stacking or silent makeup and skin-tone changes |
| 12 | Final edited export | Short journey from edit to result | Target four primary actions from upload to export | Treating a single visual result as sufficient QA evidence |

### 4.1 Strongest lessons

- The canvas remains visually dominant.
- Smart and Classic are distinct mental models but share the same image and history.
- The current selection is unmistakable.
- Context removes irrelevant tools.
- A useful default is followed by a small number of understandable controls.
- Per-face and background protection are product concepts, not hidden implementation details.

### 4.2 Competitor gaps that Retouch can turn into advantages

- Fully named adjustments and an Active Adjustments ledger.
- Explicit model/fallback/local-processing truth.
- Source-preserving, color-managed, manifest-backed delivery.
- Protection for closed eyes, occlusions, beauty marks, hair/wigs, texture, and straight background lines.
- A reproducible session rather than only an opaque final image.
- A clear distinction between correction, style, geometry, and manual reconstruction.

---

## 5. Current Retouch baseline

### 5.1 Scale snapshot

The current source contains:

- 219 indented `ParamSpec(...)` declarations in `retouch/params.py`.
- 132 slider declarations and 25 accordion declarations in the Single Photo Editor block of `gui.py`.
- 133 `gr.Slider(...)` declarations across `gui.py` as a whole.
- 50 curated recipes, of which 18 are marked recommended in the current recipe metadata.

These figures explain the usability problem: Retouch already has breadth, but the first decision presented to a user is frequently a parameter decision rather than an outcome decision.

### 5.2 Capability inventory and gap

| Area | Implemented baseline | Product gap to close |
|---|---|---|
| Main editor | Three-column Single Photo Editor, recipes, cookbook, Smart Process, preview, full export, sessions | Large up-front control surface; no stable Smart/Classic workspace shell |
| Smart Default | `SmartProcessor` analyzes a proxy and returns recipe, parameter suggestions, and explanations; its API accepts face boxes and masks | GUI currently calls it without face boxes, skin mask, or person mask; it proposes settings but is not a contextual workspace |
| Rendering | Explicit Render Preview and Export Full Quality paths | Slider changes mark state stale but do not produce a fresh visual result automatically |
| Render contracts | Immutable JSON-compatible snapshots, settings hashes, preview/full intent, and stale-commit safety | No interaction-level “latest request wins” coordinator for Smart macro changes |
| Preview cache | Session-scoped bounded cache with source, orientation, geometry, color, bit depth, RAW, detector, and engine identities | Smart interactions do not yet expose cache-aware latency or lifecycle behavior |
| Temporary workspace | Session-owned collision-safe artifacts/cache with ownership and cleanup guards | Smart callbacks must reuse this boundary rather than create temporary paths ad hoc |
| Advanced Retouch | Adjust, Heal, Remove, Reshape, semantic intersections, mask controls, selected-face support, history, snapshots, and full export | These capable manual tools are not surfaced as contextual Smart destinations |
| Per-face editing | Face detection, selected face, face recipe, and an explicit face-local parameter allowlist | Face selection is a separate control cluster, not a first-class canvas context |
| Safe Auto | Apply/dampen/skip/review decisions with evidence and strength scaling; manual Advanced edits remain separate | Explanations and protection status are not the central Smart interaction model |
| Identity safety | Beauty-mark protection plus active eye visibility/artifact-safety work | Protections are scattered and the active eye work still requires calibrated visual certification |
| Semantic context | Face/skin/hair/person/background masks and subject/background separation | No common point-to-region routing contract for canvas interaction |
| Sessions/history | Versioned params, image identity, local adjustments, Advanced edit log, style events, undo/redo, forward-tolerant loading | Smart intent, macro values, selected face, evidence, and protections need a versioned envelope |
| Delivery | Proxy preview, native/full path, ICC/EXIF options, render manifests, and native inspection | Smart must keep these distinctions visible and must never make the preview look like the final deliverable |

### 5.3 Architectural conclusion

The processing foundation is already strong enough for the proposed product. The primary work is composition, interaction lifecycle, contextual state, truth labels, and validation. New image algorithms are not the prerequisite for Smart v1.

---

## 6. Product north star

> Upload a photograph, choose an intent, see a protected and explainable preview, refine at most three relevant controls, and export through the existing full-quality path—without losing access to Classic detail.

### 6.1 Target journey

1. **Upload** — Retouch creates the source identity, proxy, color contract, cached analysis context, and non-pixel session state.
2. **Choose** — the user chooses an explicit intent or taps a region; Retouch never invents a creative intent from demographics or perceived attractiveness.
3. **Preview** — a reviewed default is rendered on the proxy. Safety decisions and the actual processing capability are visible.
4. **Refine** — the selected card exposes zero to three macro controls. Classic and Advanced remain available.
5. **Export** — Full Quality replays the immutable state against the native source and records delivery evidence.

### 6.2 Proposed workspace shell

```text
┌──────────────────────────────────────────────────────────────┐
│ Back         [ Smart | Classic ]         Compare   Export   │
├───────────────────────────────────┬──────────────────────────┤
│                                   │ Context                  │
│              CANVAS               │ Face 1 · Background safe │
│       tap / zoom / select face     │ Local · MediaPipe        │
│                                   ├──────────────────────────┤
│                                   │ [Color] [Clean] [Face]   │
│                                   │ [Auto Polish] [Background]│
│                                   ├──────────────────────────┤
│                                   │ Selected: Face           │
│                                   │ Texture      ─────●──     │
│                                   │ Shape        ──●─────     │
│                                   │ Definition   ───●────     │
│                                   ├──────────────────────────┤
│                                   │ Active Adjustments  (4)  │
└───────────────────────────────────┴──────────────────────────┘
```

On narrow screens, the right column becomes a bottom sheet. The canvas, workspace switch, compare, undo/redo, and export remain stable.

### 6.3 Measurable product targets

| Target | Acceptance threshold |
|---|---|
| Primary actions from upload to full export | No more than four for the default Smart path |
| Primary Smart cards visible simultaneously | No more than five |
| Macro controls visible for one selected card | No more than three |
| Smart-to-Classic transition | One action; image, selected face, history, and settings remain intact |
| Context feedback after tap | p95 under 100 ms after analysis context is ready |
| Cached global-color proxy render | p95 under 500 ms on the reference machine and corpus |
| Cached single-face proxy render | p95 under 1.5 s on the reference machine and corpus |
| Stale render commit | Zero in automated lifecycle tests |
| Silent model/fallback switch | Zero |
| Silent geometry, makeup, skin-tone, or identity-mark change | Zero |

Latency thresholds must be measured with the same reference hardware, proxy dimension, warm/cold-cache labels, image set, and runtime-doctor output.

---

## 7. Product principles and hard rules

1. **Classic remains complete.** Smart may hide complexity, but it may not remove capability or change Classic defaults.
2. **One engine, two presentations.** Smart composes existing contracts; it does not fork the image pipeline.
3. **Zero is exact.** A macro at zero reproduces the base state exactly, not approximately.
4. **Recompute, do not compound.** Every change replays from the immutable base snapshot plus ordered edits.
5. **Correction can be measured; style must be chosen.** Exposure and neutral white-balance evidence may be suggested. Makeup, body/face geometry, skin tone, age appearance, and creative style require explicit intent.
6. **Protect first.** Closed eyes, occlusion, marks, hair/wigs, texture, and background geometry have visible protection state.
7. **Manual work is sovereign.** Safe Auto must not silently dampen or rewrite explicit Advanced Retouch actions.
8. **Name the real capability.** A deterministic transform is not called AI. If a model is used, show its local/remote state and exact fallback.
9. **Preview is not delivery.** Full Quality remains a separate replay and verification path.
10. **No pixels in session JSON, telemetry, or explanation records.** Persist only reproducible parameters, evidence summaries, model identities, and hashes.
11. **No demographic inference.** Do not infer gender, age group, ethnicity, attractiveness, or a preferred facial shape.
12. **No free-form prompt in v1.** A later command box may use a constrained, inspectable intent grammar, but may not become a hidden generative service.

---

## 8. Smart information architecture

### 8.1 Default cards

| Card | User promise | Existing foundation | Default controls | Explicit safeguards |
|---|---|---|---|---|
| Color | Correct or style global color | Smart analysis, color controls, recipes, color context | Amount, Warmth, Contrast | No skin-only tone inference; actual transform/model label |
| Clean | Remove a selected distraction | Advanced Remove/Heal and semantic masks | Amount, Edge protection, Method when needed | User mask or selected object required; preview boundary; fallback label |
| Face | Refine the selected face | Face-local params, selected face, texture and reshape controls | Texture, Definition, optional Shape | Shape is off until explicitly enabled; closed-eye/mark protection visible |
| Auto Polish | Apply a reviewed intent profile | Recipes, Safe Auto, style events | Face, Color, Makeup/Style component only when explicitly enabled | Active Adjustments breakdown; no silent geometry or skin-tone change |
| Background | Protect or refine background | Person/background masks, subject separation, background controls | Protect, Separation, Tone/Blur where supported | “BG Lock” means a protection contract, not an effect |

`Hair` becomes a contextual card when the tap resolves to hair or when a reviewed hair-capable operation is available. It is not shown as a promise when the current runtime cannot perform the requested operation truthfully.

User-facing cards should ship in this order:

| Order | Card | Priority | Shipping condition |
|---:|---|---|---|
| 1 | Color | P1 | Intent contract, latest-request-wins preview, and delivery parity pass |
| 2 | Face | P1 | Per-face routing and identity-safety gate pass |
| 3 | Auto Polish | P1 | Every component is separately visible, removable, and safety-reviewed |
| 4 | Background | P1 | Protection contract passes line/edge distortion review; effects may follow later |
| 5 | Clean | P2 | Advanced mask/replay and fallback truth are preserved |
| 6 | Hair | P2 conditional | A reviewed operation is actually available in the active runtime |

The shell may display a disabled or unavailable capability for research, but production must not show an enabled card before its shipping condition passes.

### 8.2 Card rules

- A card is a reviewed intent contract, not a folder of arbitrary parameters.
- A card may expose at most three macro controls in its first level.
- “More controls” opens a filtered Classic view containing only the underlying parameters and explanations.
- The Active Adjustments panel lists every non-base change, its scope, amount, origin, and protection decision.
- Selecting a card does not mutate the image. Applying its default or moving a macro does.
- Switching cards never discards prior edits; it changes only the visible control context.
- Removing an adjustment returns its owned parameters to the prior state without resetting unrelated work.

### 8.3 Multi-face behavior

- With one detected face, Face context selects it automatically.
- With multiple faces, a face tap selects one face. “All faces” is an explicit action, never the implicit default for geometry.
- Per-face texture/color adjustments use only the existing face-local allowlist.
- Global grade and body parameters cannot leak into a per-face payload.
- Group Auto may propose a shared correction, but it displays per-face safety decisions before apply.

---

## 9. Smart intent contract

### 9.1 New pure contract

Create `retouch/smart_intents.py` as a dependency-light, deterministic module. It must not import Gradio, initialize models, read images, or mutate engine state.

An illustrative contract is:

```python
@dataclass(frozen=True)
class SmartIntentSpec:
    intent_id: str
    label: str
    description: str
    supported_regions: Tuple[str, ...]
    macro_controls: Tuple[MacroSpec, ...]
    capability_requirements: Tuple[str, ...]
    fallback_policy: str
    explicit_only_effects: Tuple[str, ...]
    preview_stage_hint: str

@dataclass(frozen=True)
class MacroSpec:
    macro_id: str
    label: str
    minimum: float
    maximum: float
    default: float
    writes: Tuple[ParameterWrite, ...]

@dataclass(frozen=True)
class ParameterWrite:
    param_key: str
    scope: str
    target: Any
    curve: str
    activation: str
```

The actual names may change during implementation review, but the responsibilities must remain separate: product vocabulary, macro behavior, engine parameter mapping, capability truth, and safety activation.

### 9.2 Registry validation

At import/test time, validate that:

- every intent and macro ID is unique and stable;
- every referenced parameter exists in the canonical parameter registry;
- bounds and types agree with `ParamSpec`;
- per-face writes are contained in the face-local allowlist;
- global/body parameters cannot appear in a face-local mapping;
- explicit-only effects cannot be activated by a default macro;
- every optional model has a declared unavailable behavior and user-facing label;
- card and macro ordering is deterministic.

Do not add Smart metadata to all 219 parameter declarations in v1. A small reviewed intent registry is easier to audit and avoids turning the engine schema into a UI schema.

### 9.3 Macro evaluation

For a numeric parameter with base value `b`, reviewed target `t`, normalized amount `a`, and curve `c`:

```text
value = clamp(b + c(a) × (t - b), parameter bounds)
```

Required behavior:

- `a = 0` returns `b` exactly.
- `a = 1` returns `t` exactly.
- Integer parameters use documented deterministic rounding.
- Signed values preserve direction and never cross zero unless the mapping explicitly permits it.
- Booleans and enumerations are never interpolated. They require a discrete, visible user choice.
- Geometry writes are absent until the user explicitly enables Shape or an equivalent control.
- Re-evaluation starts from the base snapshot and all ordered active adjustments, never from the last rendered pixels.
- Removing one macro removes only its owned writes and reveals the prior value from lower layers.

### 9.4 Adjustment ownership and precedence

Use a layered state model:

```text
engine defaults
  → selected recipe/profile
  → Smart measured correction
  → Smart explicit intent macros
  → filtered Classic overrides
  → per-face overrides
  → ordered Advanced manual edit log
```

Later layers win only for parameters or pixels they explicitly own. Advanced manual operations remain separate from Safe Auto decisions. The Active Adjustments ledger must show this order rather than collapsing it into an unexplained final number.

---

## 10. Canvas context routing

### 10.1 Context object

Create a pure, serializable `SmartContextRef` containing only compact references:

- source identity and orientation contract;
- normalized image coordinate;
- resolved semantic region and confidence;
- selected face index and face-box digest;
- active protection IDs;
- detector/parser capability identity;
- analysis revision.

Images, masks, landmarks, and face crops stay in the existing session cache/runtime context. They are never serialized into UI state or session JSON.

### 10.2 Routing sequence

1. Convert displayed-canvas coordinates through crop, fit, zoom, orientation, and proxy transforms into normalized source coordinates.
2. Reject taps outside the rendered image, including letterbox areas.
3. Resolve semantic masks using the cache entry for the current source and analysis revision.
4. Resolve the selected face by containment and nearest valid face when appropriate.
5. Apply deterministic region precedence where masks overlap:
   `Eyes/Lips → Face → Hair → Clothing/Person → Background`.
6. Return a context and relevant cards without changing processing parameters.
7. Draw a non-destructive context highlight that is not part of export pixels.

### 10.3 Failure and fallback behavior

- If parsing is unavailable, face-box context still works where detection is available.
- If neither parsing nor detection is available, fall back to Global/Background selection and say why.
- Never fabricate a semantic selection from low-confidence evidence.
- A fallback must change the label before the user applies the operation.
- Changing source, orientation, crop, or geometry invalidates affected context and requires re-analysis.

---

## 11. Smart analysis v2

### 11.1 Preserve the current analyzer

`retouch/smart_default.py` already accepts optional face boxes, skin mask, and person mask. The first improvement is to pass cached analysis context from the GUI and enrich the proposal contract; it is not to replace the analyzer.

### 11.2 Separate evidence from intent

The proposal should contain:

```text
proposal_id
source/settings/analysis revisions
measured observations
recommended correction writes
selected user intent
affected regions and faces
per-effect confidence
Safe Auto action: apply | dampen | review | skip
protection decisions and reasons
model/method/fallback identity
plain-language explanations
```

Measured observations may support exposure, dynamic-range, white-balance, noise, blur, clipping, and detectable occlusion. They must not be presented as facts about identity, attractiveness, age, gender, ethnicity, or the “correct” facial shape.

### 11.3 Mutation rule

Analysis creates a proposal only. The user applies it explicitly. If the user selected a reviewed intent before analysis, the proposal may combine measured correction with that intent, but it must still list the components separately.

### 11.4 Protection chips

Display concise chips such as:

- `Closed eye protected`
- `Beauty mark protected`
- `Background geometry locked`
- `Face 2 needs review`
- `LaMa unavailable · Telea preview`
- `Local MediaPipe landmarks`

Every chip opens the underlying evidence and affected operation. A chip is not a vague confidence score.

---

## 12. Preview and interaction lifecycle

### 12.1 V1 rendering strategy

Use the existing full proxy pipeline at the current preview dimension on:

- card-default apply;
- macro release/change, not every pointer movement;
- adjustment removal;
- undo/redo;
- protection toggle;
- selected-face change when the visible result changes.

During pointer movement, update the numeric label and optional lightweight overlay only. This avoids creating a queue of expensive renders that can never be seen.

### 12.2 Latest-request-wins coordinator

Introduce a small render coordinator only if current GUI callbacks cannot guarantee these semantics:

1. Increment a settings revision for every committed interaction.
2. Capture an immutable render snapshot and unique request ID.
3. Reuse decoded source, face contexts, and semantic masks only when the existing cache key matches.
4. Allow at most one expensive render to execute per session.
5. Coalesce queued requests to the newest pending revision.
6. Commit a result only when source ID, settings revision, and request ID still match current state.
7. Mark discarded results as stale evidence; never flash them on canvas.
8. Clean request artifacts through `SessionWorkspace` on completion, cancellation, unload, and shutdown.

Do not kill shared model or engine state to simulate cancellation. Cooperative invalidation and stale-result rejection are safer.

### 12.3 Optimization gate

Start with whole-pipeline proxy recomputation because it is easiest to reason about and matches full export semantics. Add stage checkpoints only if measured targets are missed.

If checkpoints become necessary:

- each Smart macro declares its earliest invalidated stage;
- a checkpoint key includes every upstream setting and runtime capability that can affect pixels;
- cached intermediates remain session-owned and bounded;
- checkpoint and full-proxy output must pass equivalence tests within a documented tolerance.

### 12.4 Compare behavior

- Press-and-hold or keyboard compare shows the immutable base or previous committed adjustment state.
- Compare never changes history.
- The comparison state must be accessible without relying on color alone.
- Native-resolution comparison remains part of inspection, not a claim about the 800 px preview.

---

## 13. Safety and identity preservation

| Risk | Default behavior | Implementation gate |
|---|---|---|
| Closed, winking, or occluded eye | Do not apply eye-specific enhancement without sufficient evidence; expose review | Existing eye-visibility/artifact work must pass focused, mutation, corpus, and human review before Smart depends on it |
| Small or saturated visible iris | Attenuate artifact-prone enhancement independently from eye visibility | Separate safety decision and evidence in manifest/session event |
| Beauty marks, freckles, scars | Protect by default; user may explicitly edit manually | Mask/evidence regression cases and source/output crops |
| Hair, bangs, wigs, costume edges | Do not treat as skin or removable clutter | Semantic intersection plus boundary review cases |
| Skin texture | Preserve texture at Smart default; strength is visible | Frequency/texture metrics plus human review; no porcelain default |
| Face geometry | Off until explicit Shape control | Face-local allowlist, bounded warp, landmark and background-line tests |
| Body geometry | Excluded from Smart v1 | Remains in Classic; explicit future research only |
| Skin tone | No automatic lightening/darkening intent | Correction may neutralize a documented cast globally; skin-only style requires explicit choice |
| Straight background lines | Lock/protect by default during reshape | Before/after line and mask-edge corpus cases |
| Multiple faces | Select one face; “all” is explicit | No face-index drift after orientation/crop/re-analysis |
| Model unavailable | Label deterministic fallback before apply | Runtime capability contract and offline tests |
| Manual Advanced edit | Preserve exactly unless user edits/undoes it | Safe Auto never dampens Advanced edit-log operations |

Thresholds in the current eye research must not become Smart release defaults merely because the code path exists. The release gate is evidence from the agreed corpus and manual visual review.

---

## 14. Session, history, and compatibility

### 14.1 Session schema decision

When Smart state becomes persistable, introduce Session schema version 2 rather than silently changing version 1 semantics. The loader must support both:

- v1 loads into Classic with existing params, recipe, local adjustments, Advanced edit log, and style events unchanged.
- v2 adds an optional `smart_workspace` object.
- A v2 file without Smart activity remains semantically equivalent to v1.
- Newer unknown keys continue to warn and load best-effort where safe.

Suggested `smart_workspace` contents:

```text
workspace_mode
selected_intent_id
selected_face_ref
context_ref
active_adjustments[]
macro_values
protection_overrides
proposal summary and evidence
runtime capability identities
```

Do not store masks, face crops, image arrays, or temporary artifact paths.

### 14.2 History semantics

- Applying one card default is one history step.
- Releasing one macro is one history step; pointer movement is not.
- Selecting a face or opening a card is navigation, not history.
- Removing an adjustment is reversible and is one history step.
- Switching Smart/Classic is navigation unless it also commits a setting.
- Advanced Retouch keeps its ordered edit-log history and snapshot semantics.
- A loaded session must reproduce the same Active Adjustments ledger and settings hash before render.

### 14.3 Source mismatch

If a session source hash does not match the uploaded image, do not silently apply face indices, masks, or geometry. Offer global parameter import only, clearly separated from source-bound local edits.

---

## 15. Delivery, provenance, and capability truth

Smart must end in the current delivery architecture, not create a new “save preview” shortcut.

### 15.1 Preview

- Clearly label proxy dimension and stale/current state.
- Include source/settings revision and method/fallback status in the internal render contract.
- Do not present proxy output as the final downloadable photograph.

### 15.2 Full Quality

- Replay the immutable session against the native source with `fast=False` or the current equivalent full-quality contract.
- Preserve the selected ICC/EXIF and precision policies.
- Include Smart intent IDs, macro values, per-face scope, protection decisions, model/fallback identities, and settings hash in the render manifest where compatible.
- Keep native inspection and face-crop review available.
- Preserve the original source file unchanged.

### 15.3 Capability vocabulary

Use labels such as:

- `Deterministic color correction`
- `Local MediaPipe face landmarks`
- `Local BiSeNet semantic mask`
- `LaMa removal`
- `Telea fallback`
- `Model unavailable · manual mask only`

The word `AI` may appear only when a real model-backed capability is active and named in details. Marketing vocabulary cannot override runtime truth.

### 15.4 Privacy target

Core Smart v1 remains local and usable without network access after required local assets are installed. If a future optional remote capability is introduced, it needs a separate opt-in, data-boundary explanation, failure mode, and non-remote fallback. It cannot be silently invoked by a card or prompt.

---

## 16. Accessibility and interaction quality

- Cards use actual button/radio semantics with keyboard activation.
- Selected state uses border, icon/text, and accessible state—not color alone.
- Every control has an untruncated label, value, unit, and reset action.
- Touch targets are at least 44 × 44 CSS pixels.
- Focus remains stable after render and does not jump to the top of the page.
- Processing, stale, complete, review, and fallback states are announced through a polite live region.
- Canvas context has a non-pointer alternative: region selector plus face gallery.
- Compare has mouse, touch, and keyboard operation.
- Reduced-motion settings disable nonessential transitions.
- Error text explains recovery and never exposes internal paths or stack traces.

---

## 17. Implementation tranches

Estimates are engineering effort ranges, not calendar commitments. They exclude new model procurement, large-scale corpus labeling, external user recruitment, and release packaging.

The tranches below are retained in dependency order. Section 1.2 controls priority: P0 gates may span several tranches and remain mandatory even when their visible integration occurs later.

### T0 — Product contract and corpus baseline

**Priority:** P0 foundation

**Estimate:** 2–4 engineering days plus human review

**Code behavior:** None

Deliverables:

- Approve the five-card vocabulary, macro names, explicit-only effects, and protection language.
- Build a legally usable, consented validation corpus and source/output pairing protocol.
- Record reference hardware, runtime capabilities, proxy dimension, and cold/warm measurement rules.
- Create low-fidelity desktop/narrow-screen interaction prototypes.
- Review the supplied screenshots only as workflow evidence, not quality ground truth.

Acceptance:

- Product, engineering, and visual-review owners sign off on the hard rules.
- Every v1 card maps to existing capability or is removed from v1.
- Every appearance/identity-sensitive change has an explicit activation rule.

### T1 — Pure Smart intent foundation

**Priority:** P0 foundation

**Estimate:** 3–5 engineering days

**Likely files:** `retouch/smart_intents.py`, `tests/test_smart_intents.py`, documentation

Deliverables:

- Implement immutable intent, macro, write, capability, and adjustment contracts.
- Add registry validation against canonical parameter metadata and the face-local allowlist.
- Implement deterministic macro evaluation and layered ownership.
- Define serialized adjustment/proposal summaries without changing Session v1 yet.

Acceptance:

- Endpoint, rounding, clamping, determinism, ownership, and removal tests pass.
- No registry default can enable geometry, makeup, skin-tone style, body changes, or identity-mark removal.
- The module imports without Gradio, model loading, or image I/O.
- Engine output and current GUI remain byte/behavior unchanged.

### T2 — Shared Smart/Classic shell

**Priority:** P1 core experience with P0 compatibility responsibilities

**Estimate:** 5–8 engineering days

**Likely files:** new `gui_smart.py`, minimal `gui.py` wiring, `retouch/session.py`, focused GUI/session tests

Deliverables:

- Add the workspace switch, five-card shell, macro panel, Active Adjustments, capability details, and narrow layout.
- Keep existing Classic component identities and callbacks stable.
- Add Session v2 migration and optional Smart state only after contract approval.
- Make every Smart action inspectable in the current settings/history model.

Acceptance:

- Existing Session v1 fixtures load with unchanged parameters and Classic behavior.
- Smart/Classic switching preserves the image, history, selected face, and committed settings.
- No current Classic control is removed or default-changed.
- Keyboard and screen-reader smoke paths pass.

### T3 — Context routing and per-face integration

**Priority:** P1 core experience

**Estimate:** 7–10 engineering days

**Likely files:** `retouch/smart_context.py`, `gui_smart.py`, cache integration, context/per-face tests

Deliverables:

- Normalize canvas coordinates and route them through cached face/semantic context.
- Add canvas highlight, non-pointer region selection, selected-face behavior, and protection chips.
- Pass cached face boxes, skin mask, and person mask into Smart analysis.
- Enforce face-local parameter writes in the UI-to-engine bridge.

Acceptance:

- Orientation, crop, fit, zoom, letterbox, overlap precedence, and invalidation tests pass.
- Multi-face selection does not drift after re-analysis.
- No global or body parameter appears in a per-face payload.
- Parser/detector failure produces a truthful reduced-capability state.

### T4 — Responsive proxy preview and lifecycle

**Priority:** P0 correctness and lifecycle gate

**Estimate:** 10–15 engineering days

**Likely files:** `gui_smart.py`, optional `retouch/gui_render_coordinator.py`, `retouch/gui_preview_cache.py`, `retouch/gui_workspace.py`, lifecycle/performance tests

Deliverables:

- Render on committed Smart interactions using immutable snapshots.
- Add latest-request-wins coalescing and stale-result rejection.
- Reuse session-scoped decoded source and analysis context safely.
- Add structured latency evidence and user-visible current/stale status.
- Add stage checkpoints only if whole-proxy measurements fail the target.

Acceptance:

- Rapid macro, card, source, and face changes never commit stale pixels.
- Queue depth is bounded and does not grow with pointer movement.
- Workspace artifacts are cleaned without cross-session deletion.
- The agreed reference corpus meets targets or produces an evidence-backed optimization follow-up.
- Proxy and Full Quality settings hashes agree for the committed state.

### T5 — Face-aware Smart Analysis v2 and Safe Auto explanations

**Priority:** P0 safety gate and P1 user value

**Estimate:** 7–10 engineering days plus calibration

**Likely files:** `retouch/smart_default.py`, `retouch/safe_auto.py`, `gui_smart.py`, eye/mark safety integration, focused and corpus tests

Deliverables:

- Enrich proposals with revisions, regions, per-effect confidence, evidence, and Safe Auto actions.
- Surface protections and review decisions before apply.
- Calibrate closed-eye, iris-artifact, occlusion, marks, texture, hair, background, and multi-face behavior.
- Record accepted/rejected/reviewed style events without pixels.

Acceptance:

- Malformed or uncertain detector/parser evidence follows the approved fail-safe policy.
- Manual Advanced operations remain outside automatic dampening.
- Focused, mutation, broader automated, contact-sheet, and human source/output reviews pass.
- Automated tests are not described as visual certification.

### T6 — Manual Clean integration and delivery completion

**Priority:** P0 delivery completion plus P2 Clean expansion

**Estimate:** 5–8 engineering days

**Likely files:** `gui_smart.py`, `gui_advanced.py`, Advanced contracts, render manifest, delivery tests

Deliverables:

- Route Clean to the existing Advanced Heal/Remove mask workflow.
- Show actual removal engine, fallback, semantic intersection, and edge protection.
- Replay Smart state and Advanced edit log in Full Quality.
- Extend manifests/inspection with Smart adjustment and capability evidence.

Acceptance:

- No removal runs without an explicit selection/mask or reviewed deterministic selection.
- LaMa absence is visible before Telea or another fallback is applied.
- Preview/full replay, session reload, history, original preservation, ICC/EXIF, and manifest checks pass.

### T7 — Deterministic Profile Photo workflow, later and separate

**Priority:** P3 later research

**Estimate:** 10–20 engineering days after core Smart certification

Candidate scope:

- Reviewed crop/aspect templates.
- Background selection/replacement with explicit masks.
- Deterministic tonal and portrait profiles.
- Output-size and head-position validation for named template families.

Excluded without separate research and consent:

- Generated clothing, pose, hair, body, or identity.
- Claims of legal passport/visa compliance without jurisdiction-specific validation.
- Remote generation hidden behind a template.

---

## 18. Planned file-change map

| File or area | Planned responsibility | Boundary |
|---|---|---|
| `retouch/smart_intents.py` | Pure intent registry, macros, parameter writes, adjustment ownership | No GUI, models, or image I/O |
| `retouch/smart_context.py` | Pure point/region/face context references and routing | Runtime masks remain cache-owned |
| `gui_smart.py` | Gradio Smart components and callbacks | Thin orchestration; no new image algorithms |
| `gui.py` | Shared shell wiring and stable component integration | Keep changes narrow; do not expand another monolith inside this file |
| `retouch/smart_default.py` | Existing analysis plus richer proposal contract | Preserve current CLI/one-click compatibility |
| `retouch/safe_auto.py` | Existing per-effect safety decisions and evidence | Manual Advanced actions remain separate |
| `retouch/gui_preview_cache.py` | Reuse decoded source and analysis contexts | Maintain session scope and complete cache identity |
| `retouch/gui_workspace.py` | Session/request artifact lifecycle | No ad hoc global temporary files |
| optional `retouch/gui_render_coordinator.py` | Latest-request-wins and stale-result rejection | Add only when callback wiring needs a dedicated tested primitive |
| `retouch/session.py` | v1 reader plus v2 Smart envelope | No pixel/mask/temp-path serialization |
| `retouch/render_manifest.py` and inspection | Smart/capability/protection delivery evidence | Preserve existing schema compatibility or version explicitly |
| `gui_advanced.py` / Advanced contracts | Contextual entry to Heal/Remove and replay | Do not duplicate manual operations in Smart |
| focused tests | Unit, integration, lifecycle, migration, safety, performance | Do not replace human visual review |

---

## 19. Verification strategy

### 19.1 Unit tests

- Intent registry uniqueness, order, parameter existence, type, bounds, and capability declarations.
- Macro endpoints, curves, rounding, clamping, determinism, explicit activation, and exact removal.
- Adjustment ownership and precedence.
- Per-face allowlist containment.
- Context coordinate conversion, precedence, fallback, and invalidation.
- Session v1→v2 migration and v2 round-trip.
- Proposal schema, evidence sanitization, and no-pixel serialization.
- Latest-request-wins state machine and cleanup idempotence.

### 19.2 Integration tests

- Upload → choose intent → apply → refine → compare → undo/redo → Full Quality.
- Smart ↔ Classic without image/settings/history loss.
- Multi-face select-one and explicit all-faces behavior.
- Parser/model present, unavailable, malformed, and fallback paths.
- Advanced Clean mask → preview → session reload → full replay.
- Source replacement while an old render is running.
- Preview and full settings-hash agreement.
- Original input remains byte-for-byte unchanged.

### 19.3 Visual corpus

The corpus must include consented source files and, where relevant, masks or region labels for:

- one open eye and one intentional wink;
- closed eyes, squinting, glasses, reflections, small irises, and saturated contacts;
- bangs, flyaway hair, wigs, costume pieces, and hands across the face;
- beauty marks, freckles, scars, acne, and textured skin;
- frontal, profile, partial, small, and multiple faces;
- light, medium, and dark skin across warm, cool, mixed, low, and harsh lighting;
- straight walls, grids, door frames, and architecture behind face/body geometry;
- heavy makeup, no makeup, cosplay makeup, colored lighting, and unusual hair colors;
- RAW/high-bit-depth/color-managed sources supported by the current pipeline.

The supplied wink/cosplay image is valuable as a known-risk case, but the screenshots and final Meitu JPEG are not a clean A/B source pair. A clean original is required before it can certify Retouch output.

### 19.4 Visual review outputs

- Source/output pairs at matched scale.
- Native face and known-risk crops.
- Contact sheet with case ID, source, settings hash, method/fallback, and review status.
- Difference/edge visualizations where useful, never as a substitute for the photograph.
- Reviewer notes for identity, texture, eyes, hair, background, color, and artifacts.
- Explicit completed-suite versus full-suite boundaries.

### 19.5 Performance evidence

For every reported metric record:

- reference machine and OS;
- runtime-doctor/model capabilities;
- source dimensions and proxy dimensions;
- face count and enabled stages;
- cold or warm cache;
- p50, p95, maximum, sample count, errors, cancellations, and discarded stale results;
- queue depth and peak memory where available.

### 19.6 Accessibility review

- Keyboard-only upload-to-export smoke path.
- Screen-reader labels, selected states, value announcements, progress, review, and fallback messages.
- 200% zoom and narrow viewport without clipped labels.
- Touch compare and non-canvas context alternative.
- Contrast and non-color state checks.

---

## 20. Release definition of done

Core Smart v1 is ready only when all of the following are true:

- Product language, intent mappings, and explicit-only effects are approved.
- Classic behavior and existing sessions remain compatible.
- Smart is a composition layer over the current engine, not a fork.
- Active Adjustments explains every non-base edit and its scope.
- No silent geometry, body, makeup, skin-tone, identity-mark, network, model, or fallback action exists.
- Smart analysis consumes the available cached face/semantic context.
- Rapid interactions cannot commit stale pixels or create unbounded queues.
- Session artifacts and caches remain isolated and are cleaned safely.
- Preview/full replay, color, metadata, original preservation, manifest, and inspection checks pass.
- Closed-eye/occlusion, iris artifact, marks, hair, texture, multi-face, and background-line gates pass the approved corpus.
- Focused and broader automated suites pass with exact scope reported.
- Human source/output visual review is complete and recorded.
- Accessibility and narrow-layout smoke tests pass.
- Documentation explains Smart, Classic, fallbacks, delivery, privacy, and known limits.

---

## 21. Risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Smart macros become opaque presets | Users cannot understand or reproduce results | Active Adjustments, owned writes, explanations, exact session state |
| GUI monolith grows further | Fragile callback wiring and difficult testing | Put Smart UI in `gui_smart.py`; pure contracts under `retouch/` |
| Live preview queues stale work | Flicker, latency, wrong final preview | Render on release, coalesce requests, immutable revision checks |
| Cache key is incomplete | Pixels from wrong state or source | Reuse current complete identity; add context/version inputs explicitly |
| Per-face setting leaks global edits | Group inconsistency and unsafe geometry | Existing face-local allowlist plus registry validation |
| Automatic appearance ideals | Bias and identity harm | User-selected intent; exclude demographic and attractiveness inference |
| Protection gate is overconfident | Eye, mark, hair, or background damage | Review state, calibrated corpus, visible protection chips, manual override |
| Fallback is hidden | Trust and output inconsistency | Capability label before apply and evidence in session/manifest |
| Smart preview differs from full replay | False confidence at export | Immutable snapshot, settings hash parity, full replay integration tests |
| Session schema changes silently | Old files or automation break | Explicit v2, v1 loader/migration fixtures, forward-tolerant reading |
| Free-form prompt expands scope | Unbounded intent, network, safety, and reproducibility | Exclude from v1; later constrained grammar research only |
| “Profile Photo” implies compliance | Incorrect ID/legal claims | Separate deterministic templates and jurisdiction-specific validation |

---

## 22. Alternatives considered

### Replace Classic with a mobile-style editor

Rejected. It would discard expert capability, break learned workflows, and create migration pressure without proving better outcomes.

### Build a separate Smart processing pipeline

Rejected. Duplicate algorithms and state would drift from Classic, full export, sessions, and safety contracts.

### Make all 219 parameters searchable and stop there

Insufficient. Search improves discovery but still asks users to translate an outcome into parameter vocabulary. It remains useful inside filtered Classic.

### Render on every slider input event

Rejected for v1. It creates avoidable queue pressure. Render on committed changes, then optimize only from measured evidence.

### Use a free-form AI prompt as the main interface

Rejected for v1. It weakens reproducibility, makes capabilities ambiguous, and can imply unsupported or remote generative behavior.

### Infer a beauty profile automatically

Rejected. Measured correction is acceptable; preferred appearance, geometry, makeup, and skin tone require user intent.

### Launch Profile Photo first

Deferred. The core Smart shell, lifecycle, truth labels, and safety are reusable prerequisites. Generative identity changes remain out of scope.

---

## 23. Explicit non-goals for Smart v1

- Removing or simplifying away Classic controls.
- New generative face, pose, clothing, hair, or background synthesis.
- Automatic body reshaping.
- Automatic face geometry, makeup, skin-tone, age, or gender presentation.
- Demographic, attractiveness, or beauty-ideal inference.
- Free-form remote prompt execution.
- Silent model downloads or network calls.
- Claiming eye/occlusion certification before corpus and human review.
- Legal ID/passport/visa compliance claims.
- Replacing Full Quality export with a proxy save.
- Treating automated metrics as human visual approval.
- Rewriting the current engine before the Smart composition layer is measured.

---

## 24. Open research decisions

These decisions must be closed during T0:

1. ~~Which exact existing parameters and targets belong to each initial macro?~~ **Closed
   2026-08-30** for the Color card (Amount/Warmth/Contrast) — reviewed and shipped in
   `retouch/smart_intents.py`'s `COLOR_INTENT`. Review caught a sign inversion in the original
   Contrast macro (highlights/shadows were flipped, producing a flattening move instead of a
   contrast increase at `amount=+1`); fixed and verified against `retouch/engine.py::_adjust_tonal`'s
   sign convention. Still open for any future Face/Auto Polish/Background macros.
2. Should Auto Polish begin with one reviewed profile or a small intent set such as Natural, Polished, Cosplay, and Editorial?
3. Which capability combinations make Hair eligible to appear as a contextual card?
4. What is the approved uncertainty policy for the active eye-visibility and parser evidence?
5. What proxy dimension and reference machine define the interaction latency budget?
6. Does the current Gradio event model support safe coalescing directly, or is a dedicated render coordinator necessary?
7. Which Smart state belongs in the core Session v2 versus the render manifest or ephemeral GUI state?
8. Which operations, if any, may apply to all faces by default without identity risk?
9. What user-facing phrase best distinguishes global color correction from skin-only styling?
10. Who performs human visual certification, and what constitutes an unresolved review case?

---

## 25. Recommended next action

Approve T0 and review this document before implementation. The first code change after approval should be T1's pure `smart_intents.py` contract and focused tests. That tranche is intentionally reversible, has no GUI or pixel-output change, and forces the team to agree on product semantics before callback and rendering complexity begins.

Do not start by rearranging `gui.py`. The interaction design should be backed by validated intent mappings, explicit safety rules, and a testable state contract first.

---

## 26. Source index

### Repository evidence

- `gui.py` — Single Photo Editor, Smart Process callbacks, render/export wiring, face controls, eye gate, and current interaction lifecycle.
- `gui_advanced.py` — Advanced Retouch UI and callback composition.
- `retouch/params.py` — canonical parameter metadata.
- `retouch/recipes.py` — curated and recommended recipe metadata.
- `retouch/smart_default.py` — analyzable, explainable Smart Default contract.
- `retouch/advanced_retouch.py` — manual adjust/heal/remove/reshape capability and semantic masks.
- `retouch/face_params.py` — face-local allowlist and neutral auto-face policy.
- `retouch/safe_auto.py` — apply/dampen/review/skip safety decisions.
- `retouch/gui_preview_cache.py` — session-scoped cache and complete render identity.
- `retouch/gui_render_modes.py` — immutable preview/full render contracts and stale safety.
- `retouch/gui_workspace.py` — session-owned artifact/cache lifecycle.
- `retouch/session.py` — session versioning, compatibility, local adjustments, Advanced log, and style events.
- `retouch/eye_visibility.py` and `retouch/eye_artifact_safety.py` — active worktree safety implementation requiring release validation.
- `docs/guides/GUI.md` — current preview/full/native inspection behavior.
- `docs/plans/RESEARCH_EYE_OCCLUSION_2026_08_26.md` — research protocol and acceptance discipline for eye occlusion.

### External product references

- Meitu mobile product listing: <https://apps.apple.com/us/app/meitu-ai-photo-video-editor/id416048305>
- Meitu Android product listing: <https://play.google.com/store/apps/details?id=com.mt.mtxx.mtxx>
- Meitu desktop product page: <https://pc.meitu.com/en/pc>

External references support product/workflow comparison only. They do not establish Retouch implementation requirements, competitor internal behavior, or output-quality conclusions.

---

## 27. Separate proposal — Body/shoulder reshape (Parked item, §1.2 rank 15)

**Status of this section: a proposal, not an approval.** §1.2 ranks "automatic body shaping" as **Parked**, and §23 lists "Automatic body reshaping" as an explicit non-goal for Smart v1, both requiring "a separate proposal before implementation" before any code change. This section is that proposal. It does **not** promote body reshape into the §1.2 ranked backlog, does not change any T0–T7 tranche, and does not alter §13's "Body geometry | Excluded from Smart v1" row. Approving this section only approves it as a documented candidate for a future, separately-scheduled tranche — most plausibly after T7, and only if a product owner explicitly re-opens the Parked decision.

### 27.1 Why this exists

`docs/plans/RESEARCH_MEITU_COMPETITOR_QA_2026_08_29.md` ran a 5-pair landmark-registered diff study of Meitu output against this project's own DSCF sources (izunako cosplay shoots). One of two pairs with a heavy edit pass showed a clear, visually-confirmed shoulder-silhouette narrowing — a liquify-style body reshape — surviving registration (§2a of that doc). The other four pairs, including a same-subject different-shoot batch, showed no reshape at all, meaning Meitu's own body-reshape behavior is inconsistent and manually applied, not a fixed pipeline stage (§5 of that doc). This engine's current reshape stage (Stage 1, `engine.py`) is face-only — chin/jaw/slimming — with no torso/shoulder/body warp of any kind. If competitive parity on this axis is ever pursued, it is new pipeline scope, not a bug fix or extension of existing face-reshape code.

### 27.2 Why it is Parked and should very likely stay Parked for Smart v1

The existing Parked rationale (§1.2 row 15) is sound and this research does not weaken it:

- **Identity risk is structurally worse than face reshape.** Face geometry changes are already gated behind an explicit, off-by-default Shape control with a bounded warp and landmark/background-line tests (§13). Body reshape has no equivalent landmark set in this engine — there is no torso/shoulder skeleton or pose model in the current dependency set, only BiSeNet segmentation (skin/hair/person masks) and MediaPipe *face* landmarks. A body warp would have to be driven by segmentation-mask boundaries alone, which are far less stable under pose, clothing, and occlusion than face landmarks, raising the risk of visible distortion, particularly on body types and clothing this engine's corpus has not been validated against.
- **It is exactly the kind of appearance-ideal risk §7 rule 11 and §21 already flag.** "No demographic inference" and "Automatic appearance ideals" as a named risk apply with even more force to body shape than face shape — there is no neutral default for "narrower shoulders" the way there is for, say, white-balance correction.
- **The competitive evidence is weak and inconsistent.** §5 of the Meitu research found the *same competitor* did not apply body reshape in 3 of 5 pairs. This is not evidence of a stable feature users expect; it's evidence of an optional, manually-invoked tool (Meitu's own screenshots, referenced in that research, show it under a manual "Contour" tab, not an automatic pass). Matching an inconsistently-applied competitor behavior is a weak justification for taking on new identity-safety surface area.
- **It has no foundation to build on yet.** Everything in T0–T6 (intent contracts, lifecycle, per-face safety calibration, background protection) is a prerequisite this Parked item would still need even if un-parked — §1.2's dependency column already implies this by putting it after all ranked items.

### 27.3 What a future un-parking would require, if ever proposed

Recorded here only so a future proposal does not start from zero, **not** as authorized scope:

- An explicit product decision to re-open the Parked item, made by whoever owns §1.2's ranked backlog — this document does not make that call.
- A body-landmark or pose-estimation capability this engine does not currently have (segmentation-mask-only warping is not proposed as sufficient given the identity-risk bar face reshape is held to).
- A dedicated safety corpus analogous to §19.3's eye/mark/hair corpus, covering body types, poses, clothing, and occlusion, with the same "no automatic appearance ideal" rule extended to body shape.
- Landmark-registered or non-rigid before/after evidence (per the Meitu research's own §3/§6 caveats about needing non-rigid registration to measure reshape magnitude directly) as part of any calibration, not just visual review.
- Explicit-only activation, off by default, matching the existing Face Shape control's pattern (§13 row "Face geometry").
- A new §1.2 backlog row with its own priority, dependency, and tranche — this section does not assign one.

### 27.4 Recommendation

Leave Parked. Do not schedule engineering time against this section. Revisit only if a product owner decides competitive parity on body reshape is worth the identity-safety investment described in §27.3, and only after Smart v1's T0–T6 gates (particularly T5's safety calibration) are complete — running new identity-risk surface area in parallel with unfinished eye/occlusion safety work (§13, in progress per `RESEARCH_EYE_OCCLUSION_2026_08_26.md`) is not recommended.

# Per-Face Recipe Assignment — Implementation Plan

**Status:** 📋 BACKLOG (design locked 2026-07-14) · **Parent:** MASTER_PLAN post-plan row  
**Owner-approved:** 2026-07-10 · **Effort:** ~1–2 wk (Slice 1 engine+CLI+tests; Slice 2 GUI)  
**Goal:** each detected face in a group shot can take its own recipe / param overrides, matching Evoto’s 2026 per-face preset story without auto-classification in v1.

---

## 1. Problem (verified against code)

Today:

| Layer | Behavior | File |
|---|---|---|
| `RetouchEngine.process(...)` | One global kwargs bag → one `ProcessingContext` | `engine.py:846+` |
| `ProcessingContext` | Single typed param bag for the whole call | `engine.py:171` |
| `_stage_per_face` | Loops faces; **same `ctx`** for every face | `engine.py:2571–2721` |
| ProcessPool path | `_slim_ctx(ctx)` once-shape, same dict cloned per face | `engine.py:2665–2684` |
| `_process_one_face` → `_process_face_core` | Reads all face-local strengths from that one `ctx` | `engine.py:2723`, `perf_optimizations.py` |
| `_stage_reshape` | One `ctx` for **all** faces in one liquify pass | `engine.py:2528`, `geometry.py` |
| Global stages | Grade / finish / body / separation — intentionally image-wide | Stages 3–6 |
| `FaceContext` | `face_data`, `regions`, `index`, optional `face_image` — **no params** | `detection.py:48` |

Faces already process independently (ROI crop + composite). What is missing is **param plumbing**, not a new renderer.

**Audience why:** cosplay group shots — face A white-makeup porcelain, face B natural, face C heavy contour — need different `smooth` / `whiten` / `makeup_v2` / reshape. Evoto sells this; we cannot match it today.

---

## 2. Non-goals (v1)

- Auto Male/Female/Child/Senior classification (Evoto headline; **v2+**)
- Per-face **global** grade / LUT / WB / grain (those stay image-level)
- Per-face body_reshape / body_skin (body is person-level, not face-index)
- Changing face detection order / identity tracking across frames (P5 / R14)
- GUI face-picker (Slice 2)

---

## 3. Design principles

1. **Default path byte-identical** when `face_params is None` / empty (P3 discipline).
2. **Name-keyed overrides only** — never positional tuples of values (same lesson as `_process_inputs` / `_recipe_outputs` footguns).
3. **Split param domains** so merge rules are obvious:

| Domain | Examples | Per-face? |
|---|---|---|
| **Face-local** | `smooth`, `whiten`, eyes/lips/teeth, blush, `makeup_v2.*`, freckle, specular_*, albedo_*, chromophore ops, `cosplay_*` face ops, all `reshape_*`, `slimming`, `face_exposure`, relight/sculpt, wrinkle_*, undereye_* | **Yes** |
| **Image-global** | contrast/brightness/WB, color_grade, film/grain/bloom, subject_separation, backdrop/fabric (full-frame), body_*, session | **No** — ignore if present in override, log once |
| **Structural** | `face_contexts` cache, quality/proxy flags | Not overridable |

4. **Stable face index** = order of `faces` list from detection (already what `FaceContext.index` uses). Document: “face 0 = first detected; GUI Slice 2 may re-sort L→R for display but must map back to engine index.”
5. **Recipe string per face** resolves through existing recipe loader, then overlays on the **global base context** (not on empty defaults).

---

## 4. Public API

### 4.1 Engine

```python
def process(
    self,
    img_bgr,
    recipe: Optional[str] = None,
    ...,
    face_params: Optional[Mapping[int, Mapping[str, Any]]] = None,
) -> ProcessingResult:
```

Semantics:

- Keys = face index `0 .. N-1` after detection (indices for missing faces ignored; out-of-range logged).
- Values = dict of:
  - `"recipe": "natural"` (optional) — resolved via existing recipe path to a flat override dict
  - any face-local param name → value (same names as `process()` kwargs / `ProcessingContext` fields)
- Merge order per face **i**:

```
base_ctx  (global recipe + process kwargs, today’s path)
  ← face recipe flat params (if any)
  ← face explicit overrides (win)
  → face_ctx_i
```

Helper (new, pure, unit-tested):

```python
# retouch/face_params.py (new leaf module — no engine import cycle)
FACE_LOCAL_PARAM_NAMES: frozenset[str]  # derived from params.py tags or explicit allowlist
GLOBAL_ONLY_PARAM_NAMES: frozenset[str]

def resolve_face_context(
    base: ProcessingContext,
    overrides: Mapping[str, Any] | None,
    *,
    recipe_resolver: Callable[[str], dict] | None = None,
) -> ProcessingContext:
    """Return dataclasses.replace(base, **filtered) with recipe expansion."""
```

Prefer **allowlist of face-local names** (explicit) over “everything minus global” so new global params don’t silently become per-face.

### 4.2 CLI (Slice 1b, small)

```bash
python3 cli.py group.jpg -o out/ --recipe natural \
  --face-params faces.json
```

`faces.json` example:

```json
{
  "0": { "recipe": "cosplay", "smooth": 0.7 },
  "1": { "recipe": "natural", "whiten": 0.0, "slimming": 0 },
  "2": { "smooth": 0.4, "makeup_v2_contour": 40 }
}
```

Keys may be strings (JSON) → coerce to int.

### 4.3 GUI (Slice 2 — separate)

- After detection, show face thumbnails with index badges.
- Per-face recipe dropdown + “edit this face” that writes into a `gr.State` dict `face_params`.
- Smart Process / recipe change: update **global** only unless user has pinned a face override.
- Do **not** invent auto-classifiers here.

---

## 5. Engine integration points (minimal diff)

### 5.1 Build base context (unchanged)

`build_context(...)` / `process()` still produce one global `ctx`. Store:

```python
ctx.face_params = face_params  # Optional[Dict[int, dict]]; default None
```

Add field on `ProcessingContext` (or keep as process-local only — field preferred so ProcessPool can pickle it with slim dict).

### 5.2 `_stage_per_face` — the only required fork

Today every face gets `ctx`. Change:

```python
def _ctx_for_face(self, ctx, face_index: int) -> ProcessingContext:
    if not ctx.face_params:
        return ctx
    ov = ctx.face_params.get(face_index)
    if not ov:
        return ctx
    return resolve_face_context(ctx, ov, recipe_resolver=...)
```

Call sites:

| Site | Change |
|---|---|
| Single-face `_process_one_face(..., ctx)` | `ctx=_ctx_for_face(ctx, 0)` |
| ThreadPool submit | per-index `ctx_i` |
| ProcessPool `payloads` | `_slim_ctx(_ctx_for_face(ctx, i))` per face **i** |
| Built `FaceContext` | optional: stash resolved param snapshot for GUI debug (not required v1) |

### 5.3 `_stage_reshape` — explicit decision

`geometry.FaceReshaper.reshape(img, faces, ctx)` applies one param set across faces.

**v1 options (pick one, document in commit):**

| Option | Pros | Cons |
|---|---|---|
| **A. Defer** reshape/slimming to global only in v1 | Smallest diff; skin/makeup still per-face | Cosplay group can’t per-face eye enlarge |
| **B. Per-face reshape loop** | Full product story | Must verify multi-face warp compositing if sequential warps interact |

**Recommendation: B if cheap, else A with clear docs.**  
Geometry already loops faces internally (`geometry.py` face_oval_pts_per_face). If `reshape()` can take `List[ProcessingContext]` or per-index overrides without rewriting the warp solver, do B in Slice 1. If not, ship A and open a 0.5 d follow-up.

Spike check (1–2 h before coding): read `geometry.py` reshape loop — if params are read **inside** the per-face loop from `ctx`, switch to `ctxs[i]`; if one global warp field is built from one ctx, defer.

### 5.4 QA back-off (A5)

`qa_backoff` re-runs core with reduced strengths. With per-face params:

- Prefer **per-face back-off** only on faces that trip plastic/asymmetry detectors (ideal).
- v1 acceptable: global back-off multiplies face-local smooth/whiten on **all** faces (same as today). Document as known limitation.

### 5.5 Proxy / F8.2 / face_contexts cache

- Cached `face_contexts` remain detection/parsing only — **do not** cache per-face recipes inside them (params change every slider drag).
- When GUI passes `face_contexts` + `face_params`, merge still applies every process() call.

---

## 6. Recipe resolution detail

Reuse existing machinery; do not invent a second recipe format.

```python
from retouch.recipe_loader import resolve_recipe  # actual symbol may be get_recipe / _flat_recipe_to_engine

def expand_face_override(raw: dict) -> dict:
    out = {}
    if "recipe" in raw:
        name = raw["recipe"]
        flat = recipe_to_flat_engine_dict(name)  # same as process(recipe=name) base
        out.update(flat)
    for k, v in raw.items():
        if k == "recipe":
            continue
        out[k] = v
    return filter_face_local(out)  # drop global-only keys + warn
```

Nested recipe keys (`skin.smooth`) should already flatten through the same path as CLI/GUI.

---

## 7. Tests (must ship with Slice 1)

File: `tests/test_face_params.py` (+ thin engine integration).

| # | Test | Assert |
|---|---|---|
| 1 | `resolve_face_context` identity | `overrides=None` → same field values as base (or `is` same object) |
| 2 | Overlay wins | base smooth=0.3, face `{smooth:0.8}` → 0.8 |
| 3 | Recipe expand | face `{recipe: "natural"}` sets known natural keys |
| 4 | Explicit beats recipe | `{recipe, smooth: X}` → smooth X |
| 5 | Global key stripped | `{brightness: 10}` not applied; no crash |
| 6 | Unknown key ignored or TypeError | match project convention (prefer ignore+log) |
| 7 | Engine multi-face (synthetic 2 faces or real group) | face0 smooth high vs face1 zero → measurable skin variance difference **or** param echo via debug hook |
| 8 | Golden path | `face_params=None` byte-identical to pre-change on fixed seed/image (or hash of core face ROI) |
| 9 | Out-of-range index | `{99: {...}}` no crash |
| 10 | ProcessPool path | if pool enabled, two faces still get different slim_ctx (mock `_face_pool` or force thread path) |

Avoid hanging bare `RetouchEngine()` — use module `engine` fixture (`conftest.py`).

---

## 8. Implementation slices

### Slice 1 — Engine + unit tests (~3–5 d) ⭐ do first

1. `retouch/face_params.py` — allowlists + `resolve_face_context` + JSON loader helper  
2. `ProcessingContext.face_params` field  
3. `process(..., face_params=)` wire into build path  
4. `_ctx_for_face` in `_stage_per_face` (thread + process pool)  
5. Reshape decision A or B (spike first)  
6. Tests 1–10  
7. Optional CLI `--face-params`  

**Exit:** multi-face demo script + tests green; default path identical.

### Slice 2 — GUI (~3–5 d)

1. Detect → thumbnail strip with indices  
2. State: `face_params: dict[int, dict]`  
3. “Apply recipe to selected face” vs global recipe radio  
4. Pass `face_params` into `process_image`  
5. Drift-safe: face_params is a `gr.State`, **not** a new positional process-input slot unless registered in `params.py` (prefer State + explicit arg in `process_image` signature)

### Slice 3 — Auto class (later, optional)

- Lightweight rules on face bbox size / skin ITA / landmarks — **not** neural  
- Only after A1 evidence or owner demand  
- Emits suggested `face_params` user can edit

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| ProcessPool pickling huge ctx | Already slim; ensure face_params values are JSON-scalar only |
| Face index shuffle between runs | Document detection order; GUI pins by embedding later (R14) |
| Users put grade in face_params | Filter + one log line |
| Reshape multi-face interaction | Spike; sequential warp order = face index order |
| QA back-off unfair to one face | Document v1; per-face back-off = follow-up |
| GUI argument-order footgun | face_params via State/kwarg, never mid-list positional |

---

## 10. Acceptance

1. Group photo, `face_params={0: {recipe:"cosplay"}, 1: {recipe:"natural"}}` — visual difference face-to-face, global grade shared.  
2. `face_params=None` — full pytest + existing multi-face tests green; no new MediaPipe teardown hangs.  
3. Dead-key / recipe integrity guards still pass.  
4. Docs: `docs/guides/BATCH_GUIDE.md` or API.md one section; MASTER_PLAN row → ✅.

---

## 11. Why this before P4 makeup unmix

| | Per-face recipe | P4 unmix |
|---|---|---|
| Type | Product plumbing | Research algorithm |
| Risk | Low (merge + wire) | High (inverse model) |
| Competitor gap | Only missing Evoto-class feature we can match soon | Nobody has it — moat, but longer |
| Deps | FaceContext exists | R7 done; paint mask incomplete |
| Ship value | Immediate group-shot UX | Quality leap on painted faces |

**Order:** Slice 1 per-face → (visual QA / T5 / ship path can interleave) → P4 spike.

---

## 12. Open questions for owner

1. Reshape in v1 (option A global vs B per-face)?  
2. CLI JSON only, or also `--face0-recipe cosplay` flags? (recommend JSON only)  
3. Should `process()` return which face got which resolved recipe in `result.meta` for debugging?

Default if no answer: **B-lite** (agent-confirmed); JSON only; yes `result.face_recipes`.  
Full agent synthesis: `RESEARCH_PER_FACE_AND_P4.md`.

---

## 13. Research findings (code audit + agent waves 2026-07-14)

_No agent swarm — single-pass audit after rate-limit. Evidence below supersedes earlier “spike first” uncertainty on reshape._

### 13.1 Process pool already supports different ctx per face

`FaceProcessorPool.process_faces(payloads)` (`perf_optimizations.py:1034`) takes a **list of independent tuples**. Worker `_process_single_face_worker` (`:933`) unpacks `ctx` per payload and calls `_process_face_core(..., ctx, ...)`.

Today engine builds every payload with the **same** `_slim_ctx(ctx)` (`engine.py:2665–2684`).  
**Minimal change:** `_slim_ctx(_ctx_for_face(ctx, i))` inside the list comprehension. **Zero pool API changes.**

`_slim_ctx` strips ndarray fields only — recipe scalar overrides survive pickle. Keep `face_params` values JSON-scalar (no masks in overrides).

ThreadPool path (`engine.py:2709–2720`) same pattern: pass `ctx_i` into `submit`.

### 13.2 Reshape: recommend **B-lite** (cheap)

`geometry.FaceReshaper.reshape` (`geometry.py:68–204`):

| Step | Code | Implication |
|---|---|---|
| Read params | **Once** before loop (`:97–119`) | Same `reshape_vals` / `slimming_val` for all faces |
| Build warps | **Per face** in `for face in faces` (`:143–196`) | Warp list is already per-face geometry |
| Apply | **One** `_apply_warps` for entire image (`:204`) | All faces’ warps summed into one remap |

**B-lite (~30–50 LOC):** inside the face loop, if `face_ctxs` list provided, re-read `reshape_vals` from `face_ctxs[i]`; else keep today’s single `ctx`. Still one global remap — safe, no sequential warp interaction.

**Not** full sequential remap-per-face (expensive, seam risk).  
**A** only if we refuse to touch geometry this sprint — product weaker for cosplay eye-size.

### 13.3 Recipe expand — use existing chain

```
resolve_recipe(name)          # params.py:2324 → nested recipe dict
build_context(name, rec, ov)  # engine.py:524 → ProcessingContext
```

`recipe_loader._flat_to_engine_recipe` is for **JSON user recipes**, not the main path.

**Recommended expand:**

```python
def resolve_face_context(base: ProcessingContext, raw: Mapping[str, Any]) -> ProcessingContext:
    ov = {k: v for k, v in raw.items() if k != "recipe" and k in FACE_LOCAL}
    if "recipe" in raw:
        rec = resolve_recipe(raw["recipe"])
        # build face-only ctx from recipe, then overlay ov, then merge face-local fields onto base
        face_from_recipe = build_context(raw["recipe"], rec, ov)
        return merge_face_local(base, face_from_recipe)  # dataclasses.replace on FACE_LOCAL fields only
    return dataclasses.replace(base, **filter_face_local(ov))
```

**Critical:** merge face-local fields only — never let face recipe’s global grade/WB overwrite image-level `base`.

### 13.4 Slice 1 LOC estimate (revised)

| Piece | LOC |
|---|---|
| `face_params.py` allowlist + resolve + JSON load | ~80–120 |
| `ProcessingContext.face_params` + `process()` kwarg | ~15 |
| `_ctx_for_face` + `_stage_per_face` 3 call sites | ~40 |
| geometry B-lite optional | ~40 |
| tests | ~150–200 |
| CLI `--face-params` | ~30 |
| **Total Slice 1** | **~350–450** |

### 13.5 What does **not** exist

- No `face_params` anywhere in tree (grep clean)  
- No multi-face GUI picker  
- No per-face QA back-off  

### 13.6 Risks confirmed

1. Face recipe with nested grade keys → must strip via allowlist  
2. `build_context` quirks (blemish←smooth, whiten rosy/porcelain) apply per face — **desired**  
3. Index stability = detection order only  
4. Golden path: `face_params is None` → never call replace (same object identity OK)

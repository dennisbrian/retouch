# Research Synthesis — Per-Face Recipe + P4 Makeup Unmix

**Date:** 2026-07-14 · **Method:** 2 waves × 3 explore agents (rate-limit safe) + prior single-pass audit  
**Parents:** `PLAN_PER_FACE_RECIPE.md`, `PLAN_P4_MAKEUP_UNMIX.md`  
**Status:** research complete — ready to implement Slice 1 / P4 spike

---

## Executive ranking

| Rank | Work | Ready? | LOC / days | Blockers |
|---|---|---|---|---|
| **1** | Per-face recipe Slice 1 | ✅ | ~350–450 LOC / 3–5 d | none |
| **2** | Reshape B-lite (with Slice 1) | ✅ | ~100 LOC | none — do not sequential remap |
| **3** | P4 unmix **spike** | ✅ soft GO | 2–3 d | multi-cue α (not residual-only) |
| 4 | P4 product wire | after spike GO | 3–4 wk | synthetic + dark-skin + visual |
| — | GUI face picker | Slice 2 | 3–5 d | after Slice 1 |

---

## A. Per-face recipe — verified facts

### Call chain
```
process() → resolve_recipe → overrides dict (hand-built) → build_context
  → _process_with_proxy / _process_native_faces
    → _stage_reshape (one ctx today)
    → _stage_per_face (same ctx all faces today)
    → _composite_faces → _run_global_phases
```

### Merge points (only these)
1. `ProcessingContext.face_params` field  
2. `process(..., face_params=)` attach **after** `build_context` (not inside overrides)  
3. `_ctx_for_face` helper  
4. Four sites in `_stage_per_face`: single-face, ProcessPool listcomp, slot fallback, ThreadPool  
5. Optional geometry B-lite  
6. Optional `ProcessingResult.face_recipes`  
7. New `retouch/face_params.py`

### Process pool — zero API change
- Worker rehydrates: `dict → SimpleNamespace(**ctx)` (`perf_optimizations.py:302–304`)  
- Change only: `_slim_ctx(self._ctx_for_face(ctx, i))`  
- Different smooth: no race (separate processes; ThreadPool pure ops + private canvas)

### Recipe expand
```
resolve_recipe(name) → build_context(name, rec, explicit_overrides)
  → dataclasses.replace(base, **FACE_LOCAL fields only)
```
Never merge full face context (grade/WB leak).  
`face_params is None` → **same object identity**.

### Units warning
Engine `process(smooth=70)` is 0–100. Face JSON must use **engine units**, not 0–1 GUI recipe scale.

### Reshape — B-lite only
- Params read once before loop (`geometry.py:97–119`) — the only bug for multi-strength  
- Warps additive in one remap — different strengths safe when faces disjoint  
- Sequential remap-per-face: **rejected** (order-dependent, N× cost, landmark desync)  
- Ship B-lite (~40 LOC geometry)

### Tests gap
No test asserts different per-face params. Integration multi-face only checks `face_count`.  
Add `tests/test_face_params.py` + `test_multi_face_independent` reshape.

### GUI
`process_image` never passes `face_contexts`. No face picker. Slice 2 = `gr.State` + kwarg only.

---

## B. P4 makeup unmix — verified facts

### Exists vs missing

| Piece | Status |
|---|---|
| R7 `decompose_chromophores` | ✅ |
| R7 reconstruct / I_hat | ❌ ~15 LOC |
| R9 A×Sh | ✅ |
| R11 facepaint / coverage_even | ✅ lib, ❌ engine, need external mask |
| Auto paint_mask | ❌ invent multi-cue |
| BiSeNet makeup class | ❌ none (19-class, no paint) |
| Fitzpatrick classify | ✅ lib, ❌ residual thresholds |
| `makeup_unmix.py` | ❌ |

### Critical math finding (changes plan)
Unconstrained LS residual in blue-diff space is **always ~0** (2 DOF, 2 channel diffs).  
**R7 residual alone cannot detect white paint / sheer foundation** (on-manifold).

**Soft GO only with multi-cue α_init:**
```
hint = max(
  chroma_high,           # vivid makeup (OKLCh C > ~0.18 pattern inverted)
  L_high_and_C_low,      # white stage paint
  nonneg_clamp_residual, # off-manifold (negative mel/hb unconstrained)
) * base_skin * (1 - mole_protect)
```

### Call-site
```
perf_optimizations._process_face_core:
  ~L301 canvas_original
  → [P4 unmix] BEFORE albedo_even (~L306)   # paint must not enter even_albedo
  → frequency / skin ops on S
  → recompose
  → makeup_v2 (~L665+)
```

### Soft GO / NO-GO
| Gate | Call |
|---|---|
| Spike start | **GO** (multi-cue locked) |
| Residual-only prior | **NO-GO** |
| Product wire now | **NO** until S1–S5 + white/dark-skin |
| P1 SSS blocking | **No** |

---

## C. Frontier P1–P8 readiness (code, not docs)

| Item | Ready? | Evidence |
|---|---|---|
| P4 unmix spike | ✅ | R7/R9/R11 substrate |
| P2 CSF | partial | R13 metrics; no CSF curve |
| P7 cross-region | partial | S1 body skin exists |
| P6 preference | blocked | need owner edit pairs |
| P1 SSS | research | R7 ≠ layered transport |
| P5 temporal | blocked | no burst pipeline |
| P8 aging | blocked | needs M/I/P1 |

---

## D. Competitor (from MASTER_PLAN / existing docs)

- Evoto 2026: per-face preset (Male/Female/Child/Senior auto) — **only competitor gap we still miss**  
- Our v1: manual `face_params` by index — no auto-class (Slice 3 later)  
- Retouch4me / PixCake: no documented per-face recipe assignment in our notes  
- Acceptance already written: group photo face0 cosplay / face1 natural; global grade shared; `face_params=None` golden

---

## E. Implementation order (authoritative)

```
1. retouch/face_params.py + _ctx_for_face + 4 call sites + tests
2. geometry B-lite + test_multi_face_independent
3. CLI --face-params JSON
4. (parallel-ok) scripts/spike_p4_makeup_unmix.py multi-cue + reconstruct
5. P4 GO/NO-GO → product only if GO
6. GUI face picker Slice 2
```

**Do not spawn more research agents** unless a coding spike hits a concrete unknown.  
Detail lives in the two PLAN_* docs; this file is the index.

---

## F. Agent wave log

| Wave | Agents | Topics |
|---|---|---|
| 1 | 3 | per-face plumbing, P4 substrate, reshape geometry |
| 2 | 3 | recipe expand, face pool IPC, paint mask + R7 fidelity |

Cancelled earlier: 8-agent burst → 429 rate limit (max ~30 req / ~30s window).  
Rule: **≤3 agents** parallel on this API key.

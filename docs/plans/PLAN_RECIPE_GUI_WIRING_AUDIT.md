# Recipe / GUI Wiring Audit — Positional Param-Shift Footguns

**Status:** ✅ DONE 2026-07-13 — implemented and verified. `RECIPE_OUTPUT_KEYS` + `_recipe_output_components` dict derived from the original 111-slot `_recipe_outputs` (byte-identical components + order); import-time drift guard active. All three producers (`on_recipe_change`, `apply_custom_style`, `on_smart_process`) rewritten to `tuple(d[k] for k in RECIPE_OUTPUT_KEYS)`, structurally closing the live 107-vs-111 drift (missing `regional_modulation`, `smooth_engine`, `undereye_shadow_strength`, `freckle_removal`). `EXPECTED_UI_OUTPUT_COUNT` made self-updating; regression tests added. Remaining: 10 `reset_*` handlers still hand-ordered (latent low-risk); `test_ext_map_keys_match_radio_choices` fails on a separate pre-existing `EXT_MAP`/PNG-16 mismatch (out of scope).

---

## Executive Summary

The just-fixed `_process_inputs` refactor (commit `da1f8cb`) cleaned up **one** consumer family but left its **mirror image on the output side completely unguarded**. The audit found:

- **1 CONFIRMED, actively-drifted footgun** (high severity): the `_recipe_outputs` family. It is a bare 111-element hand-ordered component list fed by **three independent hand-ordered producer tuples** (`on_recipe_change`, `apply_custom_style`, `on_smart_process`), with **no name-keyed build and no drift guard**. Two of the three producers have **already drifted**: `apply_custom_style` and `on_smart_process` each emit only **107** values against **111** components, missing the same 4 params (`regional_modulation`, `smooth_engine`, `undereye_shadow_strength`, `freckle_removal`). Because the divergence begins at **index 5**, every slider from index 5 onward receives the *wrong neighbor's value* when "Smart Process" or a custom-style preset is applied — the exact silent-corruption signature this bug class produces.
- **1 CONFIRMED but currently-aligned** latent footgun (medium): `on_recipe_change` (111) vs `_recipe_outputs` (111) match today name-for-name, but are bound only by position with no assertion — one edit away from the same corruption.
- **1 stale test constant** (low, but it is the only thing that flagged the drift): `EXPECTED_UI_OUTPUT_COUNT = 91` in `tests/test_gui.py` vs actual **111**. This is a *symptom detector that fired late* — 20 params were added to the UI-output path over time without anyone updating the constant, and it took a hardcoded number to notice.
- **10 reset_* handler pairs** (low): small positional pairs, all currently in sync, no guard — latent but low-risk.
- **Everything else checked was SAFE by design**: `recipe_loader.py`, `params.py` assembly, `cli.py`, and `engine.py`/`ProcessingContext` are all name-keyed. The `_process_inputs` family itself is now correctly guarded.

**Highest priority: the `_recipe_outputs` family (Finding #1).** It is confirmed, live-drifted, and user-facing.

---

## Findings Table

| # | Location (file:lines) | What it binds | Verdict | Present-day drift | Priority |
|---|---|---|---|---|---|
| 1 | `gui.py:2035-2053` (`_recipe_outputs`) ← `apply_custom_style` `gui.py:101-119`, `on_smart_process` `gui.py:841-859` | 111 component slots ↔ producer value tuples | **CONFIRMED footgun** | **YES — live.** `apply_custom_style`=107, `on_smart_process`=107 vs 111 components. Both missing `regional_modulation`, `smooth_engine`, `undereye_shadow_strength`, `freckle_removal`; misalignment starts at index 5 → every subsequent slider corrupted | **P1 (high)** |
| 2 | `gui.py:624-642` (`on_recipe_change`) ↔ `gui.py:2035-2053` (`_recipe_outputs`) | 111 return values ↔ 111 components | **CONFIRMED (latent)** | No — aligned today (111=111, name-for-name), but positional & unguarded | **P2 (med)** |
| 3 | `tests/test_gui.py:98` (`EXPECTED_UI_OUTPUT_COUNT=91`) | Test constant ↔ actual output-tuple length | **CONFIRMED stale** | **YES** — 91 vs 111; `TestApplyCustomStyle` fails | **P3 (fix with #1)** |
| 4 | `gui.py:645-693` + `.click(...)` bindings ~`gui.py:2073-2110` (10 `reset_*` handlers) | Each handler's return tuple ↔ its `outputs=[...]` list | **CONFIRMED (latent, low)** | No — all 10 verified in sync | **P4 (low)** |
| 5 | `retouch/recipe_loader.py:80-260` (`_flat_recipe_to_engine` / `_engine_to_flat_recipe`) | Flat ↔ nested recipe round-trip | **SAFE by design** | n/a — fully name-keyed (`flat[key]`, `skin[key]`) | — |
| 6 | `retouch/params.py:2281-2316` (`PROCESSING_PARAMS`, `param_names()`) | Canonical ordered param list | **SAFE by design** | n/a — consumed by name (`_BY_NAME`, `spec.name`); order is a display convenience, not a binding contract | — |
| 7 | `cli.py:400-423` (`_build_params_from_args`) | argparse attrs ↔ engine params | **SAFE by design** | n/a — `getattr(args, spec.cli_flag→attr)` → `params[spec.name]`, both from same ParamSpec | — |
| 8 | `retouch/engine.py:706-735` (`ProcessingContext(...)`) | recipe/override values ↔ context fields | **SAFE by design** | n/a — pure kwargs (`**spec_kwargs`, `key=value`); no tuple unpacking | — |
| 9 | `gui.py:248-259` (`PROCESS_INPUT_KEYS`) + consumers (`gui.py:286,332,345,359,370,382,407`) + guard `gui.py:2428-2439` | 213-slot input family | **SAFE by design** (just fixed) | No — name-keyed dict build + import-time drift `AssertionError` | — |

---

## Detailed Findings

### Finding #1 — `_recipe_outputs` family: CONFIRMED, live-drifted (P1)

**The two (three) sides.** `_recipe_outputs` (`gui.py:2035-2053`) is a bare Python list of **111 Gradio components**, hand-ordered exactly like the old `_process_inputs` was. It is wired as `outputs=` on three events (`recipe.change`, `custom_style_preset.change`, `reset_btn.click` — `gui.py:2055-2071`) plus the Smart Process button (`gui.py:2462`, `outputs=[recipe] + list(_recipe_outputs) + [status, smart_analysis_html]`). Three separate functions produce the value tuples that land in those slots, and **each is its own independently hand-ordered tuple of `d["..."]` lookups**:

- `on_recipe_change` (`gui.py:624-642`) → **111** values ✅ aligned
- `apply_custom_style` (`gui.py:101-119`) → **107** values ❌ **drifted**
- `on_smart_process` slider tuple (`gui.py:841-859`) → **107** values ❌ **drifted**

**Why they drift / current drift status (measured).** The two 107-tuples open with `d["smooth"], d["mid_reduction"], d["texture_opacity"], d["pore_synthesis"], d["nose_smooth"], d["micro_restore"], …` — 6 items — whereas `_recipe_outputs`/`on_recipe_change` open with **10** items: `smooth, mid_reduction, texture_opacity, pore_synthesis, nose_smooth, regional_modulation, smooth_engine, undereye_shadow_strength, freckle_removal, micro_restore`. The four params `regional_modulation`, `smooth_engine`, `undereye_shadow_strength`, `freckle_removal` were added to the canonical list and to `on_recipe_change`/`_recipe_outputs`, but **never added to `apply_custom_style` or `on_smart_process`**. Set difference confirms these are exactly the 4 missing on both. Because the omission is at index 5, positions 5..106 are all shifted by one-to-four slots: e.g. component slot 47 (`lip_finish`) receives the smart-tuple's index-47 value (`contrast`), slot 48 (`blush`) receives `brightness`, etc. — a brightness value landing in a blush slider, the canonical symptom.

The early-return paths of these functions (`gui.py:77,87` for `apply_custom_style`; `gui.py:802,814,819,829` for `on_smart_process`) correctly emit `len(_recipe_outputs)` = 111 updates, so `apply_custom_style` is *internally inconsistent*: its no-op path returns 111, its populated path returns 107. This is why `tests/test_gui.py::TestApplyCustomStyle::test_none_style_returns_updates` asserts against the *early-return* path and observes 111 (failing only because the constant is stale at 91), while the populated path silently misfires at runtime.

**Fix sketch (fuller, per instructions).** Mirror the `_process_inputs` precedent exactly, on the output side:

1. Introduce a `RECIPE_OUTPUT_KEYS` tuple — the ordered list of the 111 param names (it already exists implicitly as the key sequence inside `on_recipe_change`; lift it to a named constant, ideally derived from `param_names()` filtered to the UI-visible subset so it *cannot* drift from the registry).
2. Build `_recipe_outputs` as `[_recipe_output_components[k] for k in RECIPE_OUTPUT_KEYS]` from a name→component dict, and add the same import-time drift guard (`set(RECIPE_OUTPUT_KEYS) - set(components)` / vice-versa → `AssertionError`) as `gui.py:2428-2439`.
3. Rewrite all three producers to build their return value **by key**: `return tuple(d[k] for k in RECIPE_OUTPUT_KEYS)` for `on_recipe_change` and `apply_custom_style`, and for `on_smart_process` `tuple(d[k] for k in RECIPE_OUTPUT_KEYS)` after overlaying `suggestion.params` onto `d`. This *structurally eliminates* the possibility of a producer emitting a differently-ordered or wrong-length tuple, and auto-picks up the 4 missing params.
4. Add a runtime/test assertion that all three producers return `len(RECIPE_OUTPUT_KEYS)` values for a real recipe (the existing `tests/test_gui.py:1203` `test_recipe_outputs_count_matches_on_recipe_change` covers `on_recipe_change` only — extend it to `apply_custom_style` and `on_smart_process`).

*Rough effort:* ~1–2 hours. Mechanical, but touches 4 code sites + tests; the payoff is the whole family becomes self-guarding like `_process_inputs`.

### Finding #2 — `on_recipe_change` ↔ `_recipe_outputs`: CONFIRMED latent (P2)

Currently aligned (111 = 111, verified name-for-name with zero mismatches), so **no live corruption today**. But it is the same construction: two hand-ordered sequences bound only by position, no assertion. It is folded into the Finding #1 fix (steps 1–3 above cover it). No separate work if #1 is done key-based.

### Finding #3 — stale `EXPECTED_UI_OUTPUT_COUNT` (P3, fix alongside #1)

`tests/test_gui.py:98` hardcodes `91`; actual is `111`. This constant is the *only* thing in the suite that noticed the output path grew, and it noticed by failing, not by preventing drift. When Finding #1 is fixed key-based, replace the magic `91` with `len(gui.RECIPE_OUTPUT_KEYS)` (or `len(gui._recipe_outputs)`) so the count assertion becomes self-updating and can never go stale again. *Effort: trivial, 5 min, bundled with #1.*

### Finding #4 — the ten `reset_*` handlers (P4, latent low)

Each `reset_*` function (`gui.py:645-693`) returns a small positional tuple matched against an `outputs=[...]` list on its button's `.click(...)`. All ten were verified in sync today (9, 7, 5, 4, 3, 16, 14, 2, 8, 6 items respectively). They are the same bug class in miniature — no guard — but each is small, single-consumer, and locally visible, so drift is easy to spot in review and low-blast-radius. Optional cleanup: once `RECIPE_OUTPUT_KEYS` exists, these could each become `tuple(d[k] for k in SUBSET)` against a named sub-key-list, but this is polish, not urgent.

---

## Recommended Priority Order

1. **P1 — Fix the `_recipe_outputs` family key-based (Finding #1 + #2 + #3 together).** ~1–2 h. This is the only *live user-facing corruption* found and it fixes the latent sibling and the stale test in one pass. Do this first.
2. **P4 — (Optional) Convert the 10 `reset_*` handlers to key-based** once `RECIPE_OUTPUT_KEYS` exists. ~30–45 min. Low urgency; do only if touching that region anyway.

No other fixes recommended — the rest is safe.

---

## Explicitly Checked and Found SAFE (do not re-audit)

- **`retouch/recipe_loader.py`** (`_flat_recipe_to_engine` `:80-161`, `_engine_to_flat_recipe` `:164-260`): every flat↔nested conversion uses explicit `flat[key]` / `skin[key]` / `eyes[key]` lookups and `for key in (…)` name loops. No `zip`, no positional dict iteration. The `skin.locus` nested pass-through (`:95-96`, `:181-182`) is name-keyed dict assignment. **Safe.**
- **`retouch/params.py`** (`PROCESSING_PARAMS` `:2281`, `param_names()` `:2314`): the list-of-lists concatenation is only ever consumed **by name** — `_BY_NAME = {p.name: p for p in …}` (`:2301`), and downstream lookups use `spec.name`/`spec.recipe_key`/`spec.cli_flag`. Its *order* feeds `PROCESS_INPUT_KEYS`/`_recipe_outputs` display order but nothing indexes into it positionally. **Safe as an ordering contract** (the order-driven consumers are guarded/covered separately above).
- **`cli.py`** (`_build_params_from_args` `:400-423`, plus `getattr` block `:425-462`): iterates `PROCESSING_PARAMS`, maps `spec.cli_flag → attr` via `getattr(args, attr)`, writes `params[spec.name]`. Both ends derive from the same ParamSpec; argparse binds by `dest`, not position. **Safe.**
- **`retouch/engine.py`** (`build_context` `:524`, `ProcessingContext(...)` `:706-735`): construction is pure keyword — `**spec_kwargs` (built by name via `_ov(key, …)`) plus explicit `field=value` kwargs. No tuple/list positional unpacking into the context. **Safe** — no engine-side echo of the footgun.
- **`_process_inputs` family** (`PROCESS_INPUT_KEYS` `:248`, consumers `save_session_handler`/`load_session_handler`/`push_undo_handler`/`undo_handler`/`redo_handler`/`save_snapshot_handler`/`compare_snapshot_handler`/`process_image`, guard `:2428-2439`): all consumers use `dict(zip(PROCESS_INPUT_KEYS, args))` or `params.get(k) for k in PROCESS_INPUT_KEYS`; the component list is `[_process_input_components[k] for k in PROCESS_INPUT_KEYS]` with an import-time `AssertionError` drift guard. **Safe (this is the fixed reference implementation).**

---

## Budget / Coverage Note

- **Fully verified (AST-parsed and diffed element-by-element, not eyeballed):** `_recipe_outputs` vs `on_recipe_change` (111 vs 111, 0 mismatches), vs `apply_custom_style` (111 vs 107), vs `on_smart_process` (111 vs 107) — the exact 4 missing params and index-5 divergence point are machine-confirmed. All 10 `reset_*` handlers diffed against their `.click` outputs. The `TestApplyCustomStyle` failure was reproduced live (`assert 111 == 91`).
- **Read and reasoned through in full (high confidence):** `recipe_loader.py` conversion pair, `cli.py:400-462`, `engine.py:524-735`, the `_process_inputs` guard. These are name-keyed by construction; verdict is solid.
- **Spot-checked, not exhaustively enumerated:** other `.change(`/`.click(`/`.select(`/`.load(` bindings in `gui.py` beyond those above. The audit searched for multi-element `inputs=`/`outputs=` list literals and followed the large ones (`_recipe_outputs`, `_process_inputs`, reset handlers). Small bindings (≤4 components, mostly dropdown/status/HTML triples like `on_save_style`, `on_search_recipes`) were not individually diffed — they are short, single-purpose, and low-risk, but not exhaustively proven aligned.
- **Not examined:** `retouch/session.py` internals (Session/Snapshot serialization) beyond confirming its callers pass name-keyed dicts; `retouch/smart_default.py` `SmartProcessor` internals (only its `.params`/`.recipe` contract matters here, which the fix normalizes by key). Neither showed a positional-binding surface from the caller side, but their internals were out of budget.

Trust level: the **"safe" verdicts on the six named subsystems are high-confidence**; the **"no other footguns" claim is medium-confidence** — bounded by the spot-check scope on small GUI bindings noted above.

---

## Critical Files for Implementation

- `/Applications/htdocs/retouch/gui.py` (lines 101-119, 624-642, 841-859, 2035-2071, 2455-2464 — the `_recipe_outputs` family + its 3 producers + wiring)
- `/Applications/htdocs/retouch/tests/test_gui.py` (line 98 stale constant; lines 1193-1205 count-match tests to extend)
- `/Applications/htdocs/retouch/retouch/params.py` (source `param_names()` for a registry-derived `RECIPE_OUTPUT_KEYS`)

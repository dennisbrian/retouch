---
name: retouch-add-face-op
description: Runbook for adding a new opt-in face-retouch operation (a new 0-100 slider/effect like red_eye, lens_glare, spot_heal, nose_shape, powder_finish, nose_highlight, prosthetic_blend, face_polish, watermark) to the Pro Max Face Retouch Engine at /Applications/htdocs/retouch. Use this whenever asked to add a new face op, a new opt-in retouch feature, wire up a new skin/eye/lens slider, add a new ParamSpec for a face effect, or anything that touches retouch/params.py plus gui.py together. This repo has shipped ~9 of these ops in the last few days and each one follows the same multi-file checklist with the same failure modes (forgotten GUI wiring throws an AssertionError at import time; a shared settings-count constant collides across parallel branches) — follow every step here even if the task sounds like "just write one function."
---

# Adding a face-retouch op

This engine has one repeating shape for a new opt-in effect: a new module, one
`ParamSpec`, a GUI wire, optionally a recipe key, and tests. Skipping any wire
doesn't fail quietly here — the repo has hard import-time and merge-time guards
specifically because this pattern was getting missed. Follow the steps in
order; don't jump straight to writing the algorithm.

Before starting, skim one or two recent ops for the current idiom — `retouch/nose_highlight.py`
and `retouch/powder_finish.py` are good, well-commented references (both explain
in their docstring *why* the naive approach failed on this engine's faces,
which is useful context for writing your own).

## 1. Write `retouch/<name>.py`

This is the actual effect. The one non-negotiable constraint in this codebase
(see CLAUDE.md "Tone-Invariance & Fairness"): **never gate on an absolute
intensity/luminance value** (`L > 170`, `I > 0.85`, etc). Every gate must be
relative to something measured from the face itself — a local skin baseline,
a cheek-ring median, the face's own texture spread. Absolute thresholds have
repeatedly shipped broken on darker skin tones in this repo (see the dark-circle
and shine-removal rewrites) because a fixed level that reads as "highlight" on
light skin is just normal skin tone on darker skin.

Shape to follow (see the reference files above):
- Takes a 0-100 `strength` (or whatever the ParamSpec calls it) plus the
  relevant landmarks/mask (skin mask, iris landmarks, etc. — see what's already
  threaded through `engine.py`'s per-face stage for what's available).
- Normalize any mask you're given with `utils.normalize_mask` — don't hand-roll
  a `max() > 1.0` check to detect 0-255 vs 0-1 masks. That exact pattern was a
  real, shipped bug here (silently discarded ops on ~24% of portraits) before
  it was replaced with a shared, epsilon-safe helper.
- Confine the effect to its region with a mask; don't touch the whole canvas
  when the edit is logically local (an earlier bug here dithered the entire
  canvas — hair, background, hands — for an eye-only edit because of a stray
  full-frame float/uint8 cast).
- Strength 0 must be a true no-op so default recipes render unchanged.

## 2. Register one `ParamSpec` in `retouch/params.py`

This is the single source of truth — adding one entry here auto-wires the CLI
flag and the engine default. Model it on the `nose_highlight` entry as a
concrete example:

```python
ParamSpec(
    name="<name>",
    cli_flag="<name-with-dashes>",
    cli_type=int,
    default=0,
    recipe_key="skin.<name>",   # omit if this op has no recipe exposure
    conversion="recipe_pct",
    min_val=0,
    max_val=100,
),
```

Nothing else needs to know about the CLI flag or engine default after this —
if you find yourself hardcoding the flag name somewhere else, stop and wire it
through the registry instead.

## 3. Wire the GUI input — `gui.py::_process_input_components`

Add one entry keyed by the *exact* param name from step 2. This is not
optional and not cosmetic: `tests/test_gui.py::TestProcessInputKeys` asserts
the key set matches `params.param_names()` and **raises `AssertionError` at
import time** if you skip it — meaning `import gui` itself fails, not just a
test. That's deliberate (an earlier version of this let a slider silently
drift out of sync with its argument position instead of failing loudly).

In practice this means touching several spots in `gui.py` that all key off the
same param name — search the file for an existing entry like `nose_highlight`
(grep it) to see the full set: the `gr.Slider(...)` construction, its entry in
the returned input dict, and its entry in `PROCESS_INPUT_KEYS`/the outputs
list if it's also a reset target. Match every place the reference param
appears.

## 4. Wire the recipe key (only if this op is recipe-exposed)

If step 2 set a `recipe_key` (e.g. `skin.<name>`), wire dispatch the same way
other `skin.*` keys are wired, and check whether it needs an entry in
`_recipe_output_components` / `RECIPE_OUTPUT_KEYS` in `gui.py` (needed if a
recipe or a reset handler should be able to set this slider). See the
`TestRecipeOutputKeys`-style tests in `tests/test_gui.py` for what's checked —
if this op gets a `reset_*` handler, note that reset handlers are hand-ordered
pairs, and `TestResetFunctions::test_reset_function_key_order_matches_click_output_order`
statically checks the handler's return-tuple order against its `.click(outputs=[...])`
order.

## 5. Write `tests/test_<name>.py`

At minimum cover:
- **Strength 0 is a no-op.**
- **Tone-invariance**: run the op on the same image with a synthetic tone/gain
  shift applied first (simulated darker skin), and assert the *relative*
  effect is comparable, not just "it runs." This repo doesn't have a real
  darker-skin corpus yet — simulated tone-shift is the accepted stand-in here,
  but say so in the test's docstring/comment as a known limitation. Don't
  present it as equivalent to real-corpus validation in commit messages or
  CLAUDE.md.
- **No-mask / empty-mask handling** if the op takes an optional mask.
- Whatever the op's actual claim is (e.g. "removes X", "restores Y") — measure
  it, don't just assert the array changed.

## 6. Grep before you run or commit

Two concrete ways this has bitten this repo:

- **Adjacent test files that reference the same literal.** A change once
  shipped with a sibling test file left red because only "the obvious" test
  file was run. Before committing, `grep -rn "<name>\|<cli-flag>"` across
  `tests/` to find every file that mentions the new key, not just the one you
  wrote.
- **The settings-count constant.** `tests/test_gui.py` defines
  `EXPECTED_RECIPE_KEY_COUNT`, an integer that must bump by exactly one for
  every new recipe-exposed param. Two parallel branches each bumping it to the
  same next number is a real merge collision that has happened here — git
  silently takes one side's line, and the other branch's count is now wrong by
  one even though the diff looked clean. `grep -n "EXPECTED_RECIPE_KEY_COUNT"
  tests/test_gui.py`, diff it against `main`/`origin` right before you commit
  (not just when you branched), and bump it if your op sets a `recipe_key`.
  (`EXPECTED_UI_OUTPUT_COUNT` is defined as `len(gui.RECIPE_OUTPUT_KEYS)` and
  self-updates — no manual bump needed there.)

## 7. Run tests with the pinned interpreter

```bash
RETOUCH_MEDIAPIPE_BACKEND=legacy .venv/bin/python -m pytest tests/test_<name>.py tests/test_gui.py -q
```

Not bare `python3` / bare `pytest` — the system Python resolves a different
`cv2` build than the pinned `.venv`, which changes detection behavior and
golden-hash comparisons. Only run the full suite if asked; per repo convention
this isn't run unprompted after every change.

## 8. Add one line to CLAUDE.md's "Outstanding Fixes"

That section is an index, not a changelog essay — one line, dated, pointing at
the commit hash for detail (`git show <hash>` has the real story). Don't write
a multi-paragraph writeup there; put design rationale in the module docstring
or commit message instead.

## A note on multi-site edits

If this op's ParamSpec/recipe entry means editing several near-identical spots
(e.g. the same literal pattern repeated across preset dictionaries), anchor
each `old_string` on unique surrounding context — the preceding key name, the
enclosing function, etc. — rather than the shared snippet alone. An ambiguous
`old_string` match fails loudly instead of silently editing the wrong one,
which is the safe failure mode, but better to give it unique context up front.

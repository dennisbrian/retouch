# Aesthetic reference targets (owner-supplied, 2026-07-03)

Real edited images the owner shared as "this is the quality bar" — not our output,
not synthetic test patterns. Kept here so future tuning (A2 competitive tuning, S4
shine-removal QA, F6 look-from-reference) has concrete targets instead of only
verbal descriptions.

## 1. Dreamy porcelain/milk-skin portrait (screenshot only, not saved to disk)
Owner shared this as an inline screenshot from a temp/sandboxed path Fable could not
read from disk, so the file itself isn't archived here — described from direct visual
inspection instead:
- White-haired character, soft diffused backlight through hair and veil, cool-blue grade.
- Extremely low local (mid-band) contrast; skin reads porcelain-smooth with no visible
  pore texture and **no visible specular shine anywhere** despite clearly being lit —
  this is the target for aggressive S4 shine-removal combined with C1's σ_C chroma
  uniformity and heavy `airy_haze`.
- Relevant existing primitives: `whiten(hue_stable=True)`, `airy_haze`, `fade_toe`,
  C1's `unify_hue_line`, S2's `micro_dodge_burn` at high strength.

## 2. `chang_e_cosplay_tamed_shine.jpg` (saved, 9-grid cosplay set)
Outdoor natural light, 嫦娥/moon-goddess cosplay, 9 poses. What makes this the S4
reference specifically:
- Forehead/nose-bridge/cheekbone highlights are **present but tamed** — not the flat
  "shine erased entirely" look, and not the blown/harsh con-hall highlights S4 is
  built to fix. This is the "≥30% residual prominence" QA bar from `PLAN_SKIN_PRO.md`
  §S4 made visible: real dimension survives, oiliness doesn't.
- Skin stays warm and evenly toned across all 9 shots despite varying outdoor light —
  consistent with S2 (micro dodge & burn) + C1 (hue-line unify) + S4 working together,
  not any single op in isolation.
- Under-eye blush/rouge reads as intentional makeup, not a retouch artifact — a
  reminder that S4/S2 must not misidentify makeup as shine (see makeup-aware
  smoothing, A3's `makeup-aware smoothing` item, for the general version of this
  concern).

## How to use these
Not automated QA fixtures (no ground truth, no ΔE targets) — visual references for
human judgment during A2-style tuning passes, or as F6 look-from-reference candidates
once that stage exists. If PixCake/Meitu edits of the SAME source photos become
available, that would upgrade these from "aspirational look" references to actual
A/B comparison fixtures.

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

## 3. Con-hall raw→edited tutorial pair (screenshot only, not saved to disk)
Owner shared a Chinese-language retouching tutorial screenshot (场照修图教程 = "convention
photo retouching tutorial") showing a genuine BEFORE/AFTER on the SAME source photo — this
is the most valuable reference type per the note below, upgrading from "aspirational look"
to real A/B evidence. Also a temp/sandboxed screenshot path Fable could not read from disk;
described from direct visual inspection:
- **Before** (small inset, bottom-left): raw convention-hall shot — harsh mixed venue
  lighting (visible overhead rigging/lights in frame), flat/muddy color, a second blurred
  figure in the background not fully separated from the subject, overall "phone-snapshot
  at a con" quality.
- **After** (main image): white/silver-haired cosplay character (dynamic action pose,
  fantasy costume with metallic dagger-like hair ornaments and a blue/dark ornate outfit).
  Clean saturated blues, sharp costume/prop detail (metal trim, fabric texture all crisp —
  NOT oversmoothed), controlled skin tone despite the original harsh lighting, background
  cleaned up/simplified vs. the busy con-hall original.
- **Why this matters for the roadmap:** this is a textbook T1 (background replace/scene
  relight) + C2 (structural sculpt) + S4 (shine removal, since con-hall lighting is
  EXACTLY the harsh-highlight scenario S4 targets) + A2 competitive-tuning case in one
  image. The raw→edited transformation is bigger than any single stage — it's the kind of
  full-pipeline result A2's blind A/B should be judged against once C2/S4 are both mature.
- If the owner can supply the RAW source photo (not just this tutorial screenshot), it
  becomes a real A1/A2 fixture: run it through our current recipes and directly compare
  against this tutorial's edited result on the identical starting point.

## How to use these
Not automated QA fixtures (no ground truth, no ΔE targets) — visual references for
human judgment during A2-style tuning passes, or as F6 look-from-reference candidates
once that stage exists. If PixCake/Meitu edits of the SAME source photos become
available, that would upgrade these from "aspirational look" references to actual
A/B comparison fixtures.

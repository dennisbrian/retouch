# P0 Tone and Demographic Safety Audit — 2026-08-12

## Result

The automatic face-treatment path no longer infers child, senior, male, or
female from landmarks, texture, beard-shadow, or skin color. `face_params="auto"`
is now a neutral no-op. Explicit user-selected per-face recipes remain
available.

`RetouchEngine.suggest_face_params()` reports `natural` plus a continuous
`tone_observation` diagnostic. The diagnostic contains CIELAB means, ITA mean
and spread, highlight/shadow coverage, texture variation, and measurement
uncertainty. It contains no Fitzpatrick/Monk label and is not used to choose a
beauty treatment.

## Verification

- `tests/test_face_params.py` proves child-like and ambiguous geometry both
  resolve to the neutral suggestion.
- `tests/test_tone_safety.py` covers uint8/float inputs, masks, clipped-light
  uncertainty, invalid masks, and the engine suggestion contract.
- Historical `classify_fitzpatrick()` and `tone_adaptation_params()` remain
  available for QA parity and compatibility only; no production caller uses
  them for automatic treatment selection.

## Remaining review boundary

This audit removes the unsafe selection behavior and adds the measurement
contract. It does not claim that skin-tone parity is visually certified. That
still requires the consented multi-tone, multi-light corpus and separate human
Natural Output review described in the certification plan.

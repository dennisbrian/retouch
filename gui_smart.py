#!/usr/bin/env python3
"""Smart Color card handlers (T2a — Smart/Classic shell, Color card only).

Pure handlers only — no gr.Blocks, no layout, no .click()/.change() wiring.
gui.py owns the layout and event registration, per the same pattern
gui_advanced.py uses for the Advanced Retouch cluster.

Smart is a thin control surface over the SAME Classic parameter components
Classic itself uses (doc "one engine, two presentations" — Smart never
calls process() differently, it only sets Classic component values via
gr.update()). See docs/plans/RESEARCH_SMART_WORKSPACE_PRODUCTIZATION_2026_08_29.md
§17 T2 and retouch/smart_intents.py for the underlying contract.

Scope: only the Color card (amount/warmth/contrast macros) is wired in T2a.
Macro target numbers in retouch.smart_intents were reviewed 2026-08-30
(doc §24 Q1 closed) — see the registry's own module comment in
retouch/smart_intents.py for the review record, including the
Contrast-macro sign fix caught during that review.
"""

import gradio as gr

from retouch.smart_intents import COLOR_INTENT, evaluate_write


def _macro_by_id(macro_id: str):
    for macro in COLOR_INTENT.macro_controls:
        if macro.macro_id == macro_id:
            return macro
    raise KeyError(macro_id)


_AMOUNT_MACRO = _macro_by_id("amount")
_WARMTH_MACRO = _macro_by_id("warmth")
_CONTRAST_MACRO = _macro_by_id("contrast")


def apply_color_macros(
    amount_value: float,
    warmth_value: float,
    contrast_value: float,
    base_vibrance: float,
    base_saturation: float,
    base_kelvin: float,
    base_tint: float,
    base_contrast: float,
    base_highlights: float,
    base_shadows: float,
):
    """Evaluate the Color card's three macros against caller-supplied base
    values and return gr.update() for each of the five underlying Classic
    parameters, in a fixed order matching the .change() outputs list:
    (vibrance, saturation, white_balance_kelvin, white_balance_tint,
    contrast, highlights, shadows).

    Callers must resolve base values themselves (e.g. read the current
    Classic slider values) -- this function never guesses a base, matching
    retouch.smart_intents.evaluate_write's own precondition.
    """
    vibrance_write, saturation_write = _AMOUNT_MACRO.writes
    kelvin_write, tint_write = _WARMTH_MACRO.writes
    contrast_write, highlights_write, shadows_write = _CONTRAST_MACRO.writes

    new_vibrance = evaluate_write(vibrance_write, base_vibrance, amount_value)
    new_saturation = evaluate_write(saturation_write, base_saturation, amount_value)
    new_kelvin = evaluate_write(kelvin_write, base_kelvin, warmth_value)
    new_tint = evaluate_write(tint_write, base_tint, warmth_value)
    new_contrast = evaluate_write(contrast_write, base_contrast, contrast_value)
    new_highlights = evaluate_write(highlights_write, base_highlights, contrast_value)
    new_shadows = evaluate_write(shadows_write, base_shadows, contrast_value)

    return (
        gr.update(value=new_vibrance),
        gr.update(value=new_saturation),
        gr.update(value=new_kelvin),
        gr.update(value=new_tint),
        gr.update(value=new_contrast),
        gr.update(value=new_highlights),
        gr.update(value=new_shadows),
    )


def on_smart_mode_toggle(mode: str):
    """mode is 'Classic' or 'Smart Color (preview)'. Returns gr.update(visible=...)
    for the Smart Color card group. T2a is purely additive: existing Classic
    controls are never hidden or reparented, so there is no classic_group to
    toggle -- only the new Smart Color card shows/hides."""
    is_smart = mode == "Smart Color (preview)"
    return gr.update(visible=is_smart)

"""Retouch Recipe System configuration.

Maps high-level recipe names to float/int ratios.
"""

from typing import List


RECIPES = {
    "natural": {
        "frequency": {"smooth": 0.30},
        "skin": {"equalize": 0.20, "rosy": 0.10},
        "eyes": {"whites": 0.05, "teeth_whiten": 0.05, "iris": 0.05, "catchlight": 0.05},
        "lips": {"tint": None, "gloss": 0.05},
        "hair": {"shine": 0.05},
        "dodge_burn": {"amount": 0.00},
        "color_harmony": {"preset": "natural", "amount": 0.00},
        "bloom": {"opacity": 0.00},
        "texture": {"opacity": 1.00},
        
        # Modular mappings
        "slimming": 0.0,
        "blush": 0.0,
        "lip_finish": "gloss",
        "nose_blush": False,
        "under_eye_blush": False,
        "white_costume_lift": False,
    },
    "portrait": {
        "extends": "natural",
        "frequency": {"smooth": 0.45},
        "skin": {"equalize": 0.35, "rosy": 0.20},
        "eyes": {"whites": 0.15, "teeth_whiten": 0.15, "iris": 0.15, "catchlight": 0.15},
        "lips": {"tint": None, "gloss": 0.10},
        "hair": {"shine": 0.10},
        "dodge_burn": {"amount": 0.10},
        "color_harmony": {"preset": "natural", "amount": 0.60},
        "bloom": {"opacity": 0.03},
        "texture": {"opacity": 1.00},
    },
    "cosplay": {
        "extends": "natural",
        "frequency": {"smooth": 0.55},
        "skin": {"equalize": 0.40, "rosy": 0.35},
        "eyes": {"whites": 0.30, "teeth_whiten": 0.30, "iris": 0.35, "catchlight": 0.25},
        "lips": {"tint": "cosplay", "gloss": 0.20},
        "hair": {"shine": 0.30},
        "dodge_burn": {"amount": 0.15},
        "color_harmony": {"preset": "cosplay", "amount": 0.20},
        "bloom": {"opacity": 0.06},
        "texture": {"opacity": 0.90},
        
        # Modular overrides
        "slimming": 30.0,
        "blush": 25.0,
        "nose_blush": True,
        "under_eye_blush": True,
    },
    "cosplay_3d": {
        "extends": "cosplay",
        "skin": {"equalize": 0.00, "rosy": 0.35},
    },
    "cosplay_no_eq": {
        "extends": "cosplay",
        "skin": {"equalize": 0.00, "rosy": 0.35},
    },
    "xiaohongshu": {
        "extends": "natural",
        "frequency": {"smooth": 0.50},
        "skin": {"equalize": 0.40, "rosy": 0.50, "relight": 0.40},
        "eyes": {"whites": 0.25, "teeth_whiten": 0.25, "iris": 0.25, "catchlight": 0.20},
        "lips": {"tint": "rose", "gloss": 0.20},
        "hair": {"shine": 0.25},
        "dodge_burn": {"amount": 0.20},
        "color_harmony": {"preset": "fantasy", "amount": 0.35},
        "bloom": {"opacity": 0.15},
        "texture": {"opacity": 0.90},
        # Xiaohongshu look: light sculpting + dreamy glow + warm/cool separation
        "specular_bloom": 30,
        "tonal_curve_strength": 0.25,
        "highlight_rolloff": 0.30,
        "skin_protect": 0.50,
        "shadow_hue": 220, "shadow_sat": 25,
        "midtone_hue": 30, "midtone_sat": 15,
        "micro_restore": 25,
        "grain_strength": 0.20,
        "finish": {"impact": 0.20},

        # Modular overrides
        "slimming": 30.0,
        "blush": 25.0,
    },
    "beauty": {
        "extends": "natural",
        "frequency": {"smooth": 0.50},
        "skin": {"equalize": 0.30, "rosy": 0.25},
        "eyes": {"whites": 0.20, "teeth_whiten": 0.20, "iris": 0.25, "catchlight": 0.20},
        "lips": {"tint": None, "gloss": 0.30},
        "hair": {"shine": 0.20},
        "dodge_burn": {"amount": 0.15},
        "color_harmony": {"preset": "beauty", "amount": 0.65},
        "bloom": {"opacity": 0.10},
        "texture": {"opacity": 0.85},
    },
    "korean_beauty": {
        "extends": "natural",
        "frequency": {"smooth": 0.45},
        "skin": {"equalize": 0.10, "rosy": 0.20},
        "eyes": {"whites": 0.15, "teeth_whiten": 0.15, "iris": 0.20, "catchlight": 0.15},
        "lips": {"tint": "pink", "gloss": 0.15},
        "hair": {"shine": 0.10},
        "dodge_burn": {"amount": 0.10},
        "color_harmony": {"preset": "natural", "amount": 0.30},
        "bloom": {"opacity": 0.04},
        "texture": {"opacity": 0.95},

        # Modular overrides
        "lip_finish": "velvet",
    },
    "porcelain_unified_v1": {
        "extends": "korean_beauty",
        "frequency": {"smooth": 0.40},
        "skin": {
            "equalize": 0.0,
            "rosy": 0.15,
            "hue_unify": 0.60,
            "chroma_even": 0.50,
            "whiten_hue_stable": 1,
        },
        "texture": {"opacity": 0.97},
    },
    "cosplay_sculpt_v1": {
        # Flagship #3: isolates C2 structural sculpt as the headline feature —
        # the landmark-derived light/shadow modeling that answers PixCake's
        # 自动中性灰/立体感 claim. Built on the porcelain_unified_v1 base (C1
        # hue-line targeting) so sculpt reads on genuinely unified skin tone,
        # not raw color variance.
        "extends": "porcelain_unified_v1",
        "skin": {
            "equalize": 0.0,
            "rosy": 0.15,
            "hue_unify": 0.60,
            "chroma_even": 0.50,
            "whiten_hue_stable": 1,
            "sculpt": 0.45,
            "shine_removal": 0.30,
        },
    },
    "convention_repair_v1": {
        # Flagship #4: S4 shine removal at con-hall-lighting strength — the
        # harsh mixed-venue-lighting scenario (overhead rigging, sweaty
        # highlights) documented in docs/reference_targets/README.md #3.
        # Strong shine_removal + moderate sculpt for dimension the harsh
        # lighting otherwise flattens, C1 hue-unify to fix venue color casts.
        "extends": "porcelain_unified_v1",
        # nose_smooth ~10% below the parent's 0.40 face smooth (texture realism).
        "frequency": {"nose_smooth": 0.36},
        "skin": {
            "equalize": 0.0,
            "rosy": 0.10,
            "hue_unify": 0.35,
            "chroma_even": 0.30,
            "whiten_hue_stable": 1,
            "shine_removal": 0.70,
            "sculpt": 0.30,
            "exposure_lock": 0.7,
        },
        "eyes": {
            "catchlight": 0.18,
        },
        # Subtle lash/brow definition via the eye-weighted selective sharpen mask.
        "sharpen": 16.0,
        # Con-hall wigs are often near clipping — halve the inherited shine push.
        "hair": {"shine": 0.05},
    },
    "milk_skin_v1": {
        # Flagship #5: 韩系牛奶皮/water-glow — high mean L, LOW chroma variance
        # (uniformity, not just brightness) per PLAN_EASTWEST_COLOR_SUPREMACY
        # §1a. Heavy chroma_even is the signature move here (not just a strong
        # whiten), plus airy_haze for the "water glow" atmospheric softness.
        "extends": "korean_beauty",
        "frequency": {"smooth": 0.55, "mid_reduction": 0.45},
        "skin": {
            "equalize": 0.0,
            "rosy": 0.20,
            "hue_unify": 0.55,
            "chroma_even": 0.75,
            "whiten_hue_stable": 1,
            "shine_removal": 0.35,
        },
        "finish": {
            "airy_haze": 0.20,
        },
        "texture": {"opacity": 0.95},
    },
    "float32_beauty_v1": {
        # Showcase recipe for the float32-native pipeline (F1/E2). The point
        # is fidelity, not style: every stage here leans on smooth tonal
        # transitions that the float path now renders without the uint8
        # banding/quantization the old fake-float path introduced —
        # tonal_curve + highlight_rolloff shape the highlights on a continuous
        # curve, a gentle skin.relight adds dimension, clarity_split adds
        # micro-definition, and a fine grain (float-native, no static) gives a
        # clean high-fidelity film-beauty finish. Pair with 16-bit PNG export
        # to carry the smooth gradients all the way to disk.
        #
        # Built on natural_polish_v1 (the restrained "just better" base) so the
        # result reads as flawless-but-real, not over-processed.
        "extends": "natural_polish_v1",
        "frequency": {"smooth": 0.28, "nose_smooth": 0.24},
        "skin": {
            "equalize": 0.06,
            "hue_unify": 0.22,
            "chroma_even": 0.18,
            "whiten_hue_stable": 1,
            "shine_removal": 0.30,
            "exposure_lock": 0.7,
            # Gentle relight for soft dimensional modeling — float-native so the
            # shading ramp stays smooth (no banding in the falloff).
            "relight": 0.22,
        },
        "eyes": {
            "dark_circles": 0.15,
            "catchlight": 0.10,
            "iris": 0.10,
            "whites": 0.12,
        },
        "lips": {"gloss": 0.12},
        # Smooth continuous tonal shaping — the core of the fidelity demo.
        # tonal_curve + highlight_rolloff render highlight/shadow rolloff on a
        # float curve; on the old path these were the most banding-prone ops.
        "tonal_curve_strength": 0.30,
        "highlight_rolloff": 0.40,
        # Light finish pack (all float-fixed today): clarity_split for micro
        # form/texture separation, a whisper of airy_haze for glow.
        "finish": {
            "clarity_split_neg": 0.12,
            "clarity_split_pos": 0.10,
            "airy_haze": 0.10,
        },
        # Fine film grain — float-native, clean (no full-frame static). Kept
        # low so it reads as fidelity texture, not stylization.
        "grain_strength": 0.10,
    },

    "float32_cinema_v1": {
        # Showcase #2 for the float32 pipeline: the cinematic film-finish.
        # Where float32_beauty_v1 is understated, this leans HARD on the finish
        # stack that was broken on the float path until today's fixes —
        # fade_toe (lifted matte black), highlight_drift (cyan highlights),
        # airy_haze, and clarity_split — plus a teal-shadow / warm-highlight
        # split tone and film grain. On the old fake-float path these ops
        # crushed to black or banded; here they compose into a clean, graded
        # filmic look. This recipe is the most direct proof the finish-stack
        # range fixes work end-to-end.
        "extends": "natural_polish_v1",
        "frequency": {"smooth": 0.30},
        "skin": {
            "equalize": 0.08,
            "hue_unify": 0.30,
            "chroma_even": 0.20,
            "whiten_hue_stable": 1,
            "shine_removal": 0.30,
            "relight": 0.25,
        },
        "eyes": {"dark_circles": 0.15, "catchlight": 0.12, "iris": 0.12},
        # Smooth highlight rolloff feeds the finish stack a clean range.
        "tonal_curve_strength": 0.35,
        "highlight_rolloff": 0.50,
        # The finish stack — the headline of this recipe (all float-fixed today).
        "finish": {
            "fade_toe": 0.30,          # lifted, hue-locked matte black
            "highlight_drift": 0.22,   # subtle cyan drift in highlights
            "airy_haze": 0.15,
            "clarity_split_neg": 0.18, # soft form band
            "clarity_split_pos": 0.12, # crisp texture band
        },
        # Teal shadows / warm highlights — the classic cinematic split.
        "shadow_hue": 200, "shadow_sat": 22,
        "midtone_hue": 40, "midtone_sat": 12,
        "highlight_hue": 45, "highlight_sat": 15,
        "vignette": 18.0,
        # Film grain — float-native, visible but clean.
        "grain_strength": 0.22,
    },

    "float32_glow_v1": {
        # Showcase #3 for the float32 pipeline: the gradient/glow stress case.
        # Big soft bloom + skin glow + airy_haze + strong highlight_rolloff +
        # a wide split tone produce large smooth gradients (backlight halos,
        # bloomed highlights, sky-like falloffs) — exactly where the old uint8
        # path showed visible banding/posterization. Rendered float-native
        # these gradients stay continuous. A dreamy high-key glow look that
        # doubles as a banding torture test.
        "extends": "natural",
        "frequency": {"smooth": 0.40},
        "skin": {
            "equalize": 0.15,
            "rosy": 0.30,
            "hue_unify": 0.25,
            "glow": 0.35,              # skin light-wrap diffusion (smooth halo)
        },
        "eyes": {"whites": 0.20, "catchlight": 0.20, "iris": 0.15},
        "lips": {"tint": "rose", "gloss": 0.25},
        # Big smooth bloom — the primary gradient generator.
        "bloom": {"opacity": 0.40},
        "specular_bloom": 30,
        # Strong highlight rolloff = long smooth highlight ramp (banding-prone).
        "tonal_curve_strength": 0.30,
        "highlight_rolloff": 0.60,
        "finish": {"airy_haze": 0.28},
        # Wide, gentle split tone across the whole tonal range.
        "shadow_hue": 230, "shadow_sat": 18,
        "midtone_hue": 35, "midtone_sat": 14,
        "highlight_hue": 50, "highlight_sat": 12,
        "grain_strength": 0.08,
    },

    "natural_polish_v1": {
        # Flagship #6: the "barely retouched, just better" case — proves the
        # algorithm stack doesn't require heavy strengths to be worth having.
        # Everything restrained (15-25 range): light C1 unify, light S4, no
        # sculpt (sculpt at low strength reads as noise, not modeling — the
        # plan's own over-modeling warning from PLAN_EASTWEST_COLOR_SUPREMACY
        # §C2 QA), no C4 finish pack (those are stylistic, not "natural").
        "extends": "natural",
        # nose_smooth ~10% below face smooth: keeps a touch more natural
        # texture on the nose/philtrum where pores read as realism.
        "frequency": {"smooth": 0.25, "nose_smooth": 0.22},
        "skin": {
            "equalize": 0.05,
            "hue_unify": 0.20,
            "chroma_even": 0.15,
            "whiten_hue_stable": 1,
            "shine_removal": 0.30,
            "exposure_lock": 0.7,
        },
        "eyes": {
            "dark_circles": 0.15,
            "catchlight": 0.08,
        },
        # Selective sharpen rides the lash/brow-weighted acc_sharpen mask
        # (eyes 1.0, brows 0.53) — subtle lash/brow definition only.
        "sharpen": 12.0,
        # Wig highlights on bright cosplay hair sit near clipping; no shine push.
        "hair": {"shine": 0.0},
    },
    "wrinkle_free_glow_v1": {
        # Flagship #7: isolates S5 wrinkle/line softening for close-up beauty
        # shots where nasolabial/crow's-feet lines actually matter. Built on
        # porcelain_unified_v1's C1 base + light S2 dodge&burn, since wrinkle
        # softening reads cleanest on already-unified skin tone. Strength kept
        # moderate (45) — S5's own 60%-cap design means even at max the fold
        # never fully disappears (reads as fake at full removal per the plan's
        # own QA warning), so there's little value pushing past ~50.
        "extends": "porcelain_unified_v1",
        "skin": {
            "equalize": 0.0,
            "rosy": 0.15,
            "hue_unify": 0.60,
            "chroma_even": 0.50,
            "whiten_hue_stable": 1,
            "wrinkle_soften": 0.45,
            "micro_db": 0.20,
        },
    },
    "pore_realism_v1": {
        # Flagship #9: S6 texture transplant, proving heavy smoothing doesn't
        # have to look synthetic. Deliberately uses a HIGH frequency.smooth
        # (0.65, well above porcelain_unified_v1's 0.40) to genuinely
        # over-smooth the skin first, then relies on texture_transplant to
        # clone real pore texture back in rather than pore_synthesis's
        # synthetic noise — the actual point of this recipe existing.
        "extends": "korean_beauty",
        "frequency": {"smooth": 0.65, "mid_reduction": 0.55},
        "skin": {
            "equalize": 0.0,
            "rosy": 0.15,
            "hue_unify": 0.55,
            "chroma_even": 0.45,
            "whiten_hue_stable": 1,
            "texture_transplant": 0.70,
        },
        "texture": {"opacity": 0.40},
    },
    "body_match_v1": {
        # Flagship #8: S1's headline feature — body_match_face, a first-in-
        # market op that tone-matches exposed body skin (arms, shoulders,
        # legs) to the already-retouched face so cosplay/swimwear/summer
        # shots don't show a visible face/body color seam. Moderate
        # body_smooth + body_equalize keep body skin consistent with the
        # porcelain face treatment; light body_whiten for the same hue-
        # stable lift the face gets (whiten_hue_stable=1 on the face side);
        # body_match_face does the actual seam fix. Strengths kept moderate
        # — the point is visible-but-natural body skin improvement, not an
        # over-processed plastic look (the "more = better" anti-pattern the
        # flagship list explicitly rejects).
        "extends": "porcelain_unified_v1",
        "skin": {
            "equalize": 0.0,
            "rosy": 0.15,
            "hue_unify": 0.60,
            "chroma_even": 0.50,
            "whiten_hue_stable": 1,
        },
        "body_skin": {
            "smooth": 0.35,
            "equalize": 0.25,
            "whiten": 0.15,
            "match_face": 0.70,
        },
    },
    "full_showcase_v1": {
        # Flagship #10: the "if you could only run one recipe" kitchen-sink
        # case. The design lesson stamped into the flagship list — "stacking
        # everything at max looks WORSE than tuned single-purpose recipes" —
        # means this is deliberately NOT a max-strength stack. Instead it's
        # the best LIGHT-TOUCH across every stage: every primitive the other
        # flagships isolate gets a restrained 15-30 dose here, so the
        # combined effect reads as "thoroughly retouched but not obviously
        # so" rather than the over-processed look a max stack produces.
        # Built on natural_polish_v1's "barely retouched, just better" base
        # (already the restrained template), then adds light doses of the
        # stages natural_polish_v1 deliberately omits (S2 sculpt at 0.15
        # reads as modeling not noise; C4 finish pack; S6 body match) plus
        # the face/body tone-match that ties the whole image together.
        "extends": "natural_polish_v1",
        "skin": {
            "equalize": 0.05,
            "hue_unify": 0.25,
            "chroma_even": 0.20,
            "whiten_hue_stable": 1,
            "shine_removal": 0.30,
            "exposure_lock": 0.7,
            # Light sculpt: natural_polish_v1 omits this because at low
            # strength it reads as noise — but at 0.15 on already-unified
            # skin it adds just enough dimension to justify "showcase".
            "sculpt": 0.15,
            # Light S5 wrinkle soften so the showcase covers close-up and
            # distance shots alike, without the 0.45 wrinkle_free_glow_v1
            # strength that would over-soften on a general-purpose recipe.
            "wrinkle_soften": 0.20,
        },
        "eyes": {
            "dark_circles": 0.15,
            "catchlight": 0.10,
            "iris": 0.10,
        },
        # Light S1 body treatment + face match — the body_match_v1 move at
        # reduced strength so it reads on full-body shots without dominating
        # the face-focused parts of the showcase.
        "body_skin": {
            "smooth": 0.25,
            "equalize": 0.15,
            "whiten": 0.10,
            "match_face": 0.45,
        },
        # C4 finish pack: airy_haze for the "showcase" polish natural_polish
        # deliberately omits (it's stylistic, not "natural"), kept light.
        "finish": {
            "airy_haze": 0.12,
        },
        # Subtle lash/brow sharpen inherited from natural_polish_v1 is
        # appropriate here too — no need to push it for a showcase.
        "hair": {"shine": 0.05},
    },

    # ------------------------------------------------------------------
    # Outdoor natural-light set (4) — each targets a genuinely distinct
    # light quality, not a re-skin. See MASTER_PLAN.md "50-recipe expansion"
    # entry for the full design rationale.
    # ------------------------------------------------------------------
    "outdoor_harsh_sun_v1": {
        # Midday direct sun: blown highlights, hard-edged shadows, squinting,
        # forehead/nose sweat-shine. Strong highlight_rolloff to recover the
        # blown sky/skin highlights, shadow lift to open up the hard shadow
        # side of the face, strong shine_removal for sun-sweat, cool-leaning
        # white balance correction since harsh sun reads slightly warm/yellow
        # on sensors and needs pulling back toward neutral.
        "extends": "natural",
        "frequency": {"smooth": 0.35},
        "skin": {
            "equalize": 0.10,
            "hue_unify": 0.45,
            "chroma_even": 0.30,
            "whiten_hue_stable": 1,
            "shine_removal": 0.55,
            "sculpt": 0.15,
        },
        "highlight_rolloff": 0.65,
        "shadows": 25.0,
        "highlights": -30.0,
        "white_balance_kelvin": 6200,
    },
    "outdoor_golden_hour_v1": {
        # Warm low-angle backlight / rim light. Amplify the warm rim via
        # relight pointed from behind-and-above (azimuth ~160° = behind
        # subject, matching a low sun position), add halation/bloom for the
        # characteristic golden-hour glow, warm white balance push (opposite
        # direction from harsh_sun_v1) since golden hour SHOULD read warm,
        # not be corrected away.
        "extends": "natural",
        "frequency": {"smooth": 0.35},
        "skin": {
            "equalize": 0.05,
            "rosy": 0.20,
            "hue_unify": 0.40,
            "chroma_even": 0.25,
            "whiten_hue_stable": 1,
            "relight": 0.35,
        },
        "relight_azimuth": 160.0,
        "relight_elevation": 20.0,
        "bloom": {"opacity": 0.18, "threshold": 0.75},
        "white_balance_kelvin": 4800,
        "grain_strength": 0.12,
    },
    "outdoor_overcast_v1": {
        # Flat, soft, shadowless light — the opposite problem from harsh sun.
        # Needs contrast/clarity LIFTED (not reduced) and C2 sculpt to
        # reintroduce the facial dimension flat light removes entirely —
        # this is sculpt's actual designed use case per PLAN_EASTWEST
        # §C2 ("removes lighting accidents... flat frontal flash faces
        # gain structure back"). Slight cool cast correction since overcast
        # skies skew blue/grey.
        "extends": "natural",
        "frequency": {"smooth": 0.30},
        "skin": {
            "equalize": 0.10,
            "hue_unify": 0.40,
            "chroma_even": 0.25,
            "whiten_hue_stable": 1,
            "sculpt": 0.40,
        },
        "contrast": 12.0,
        "clarity": 15.0,
        "vibrance": 10.0,
        "white_balance_kelvin": 5800,
    },
    "outdoor_backlit_v1": {
        # Subject silhouetted/underexposed against a bright background —
        # the classic backlighting mistake. subject_separation lifts the
        # subject's exposure relative to background, shadows lift opens up
        # the now-dark subject side, highlight_drift (C4) restrained ONLY
        # affects the bright background (it's L-thresholded so this is
        # naturally background-scoped), rim-light preserved via a light
        # relight pass from behind.
        "extends": "natural",
        "frequency": {"smooth": 0.35},
        "skin": {
            "equalize": 0.15,
            "hue_unify": 0.45,
            "chroma_even": 0.30,
            "whiten_hue_stable": 1,
            "relight": 0.20,
        },
        "relight_azimuth": 180.0,
        "relight_elevation": 10.0,
        "subject_separation": 55.0,
        "shadows": 30.0,
        "finish": {"highlight_drift": 0.15},
    },

    # ------------------------------------------------------------------
    # Studio/flash set (4)
    # ------------------------------------------------------------------
    "studio_hard_flash_v1": {
        # Direct on-camera/hard flash: harsh specular highlights, deep hard
        # shadow edges, flattened dimension (flash light is frontal and
        # shadowless in the highlight zone but creates a hard shadow line
        # at the silhouette edge). Strong shine_removal for the specular,
        # strong sculpt to rebuild the dimension flash flattens, vignette
        # to soften the harsh hard-flash background falloff look.
        "extends": "natural",
        "frequency": {"smooth": 0.35},
        "skin": {
            "equalize": 0.10,
            "hue_unify": 0.45,
            "chroma_even": 0.30,
            "whiten_hue_stable": 1,
            "shine_removal": 0.60,
            "sculpt": 0.50,
        },
        "vignette": 20.0,
        "highlight_rolloff": 0.50,
    },
    "studio_softbox_v1": {
        # Even, wraparound soft light — the studio-quality baseline, but
        # even softbox light still flattens dimension somewhat vs. a
        # deliberately modeled 3-point setup. Moderate sculpt is the
        # differentiator here (softbox_v1 vs overcast_v1 differ mainly in
        # base skin/color treatment: studio = cleaner/more saturated color,
        # overcast = slightly desaturated/cooler by nature of the light).
        "extends": "porcelain_unified_v1",
        "skin": {
            "equalize": 0.0,
            "rosy": 0.15,
            "hue_unify": 0.55,
            "chroma_even": 0.45,
            "whiten_hue_stable": 1,
            "sculpt": 0.30,
            "shine_removal": 0.25,
        },
    },
    "studio_ringlight_v1": {
        # Ring-light signature: very even frontal fill, distinctive circular
        # catchlight, minimal shadow. Skin is ALREADY flat-lit by the ring
        # (similar to softbox) but the defining feature is the catchlight,
        # so eyes.catchlight is boosted and preserved rather than the usual
        # skin-structure ops being the main lever — C1 hue-unify does the
        # heavy lifting on skin since there's little shadow/highlight to fix.
        "extends": "natural",
        "frequency": {"smooth": 0.40},
        "skin": {
            "equalize": 0.15,
            "hue_unify": 0.55,
            "chroma_even": 0.40,
            "whiten_hue_stable": 1,
            "micro_db": 0.25,
        },
        "eyes": {"catchlight": 0.35, "whites": 0.20, "iris": 0.25},
    },
    "studio_gel_color_v1": {
        # Colored gel wash — a cosplay-convention studio staple (blue/purple/
        # magenta gel backgrounds). The gel background cast is already present
        # in the source lighting, so C1's skin.hue_unify + white_balance_tint
        # pull the skin back toward its preferred locus while the environment
        # keeps its gel colour naturally.
        #
        # NOTE: color_harmony (blue_dream) was removed — it applied the gel
        # cast to the FACE too, darkening it ~18 L points and stripping skin
        # warmth (b +15 -> +1.4). The desaturated face then read as muddy /
        # grainy. Dropping it leaves the face clean and natural while the
        # source's own gel lighting keeps the background stylized.
        "extends": "natural",
        "frequency": {"smooth": 0.35},
        "skin": {
            "equalize": 0.10,
            "hue_unify": 0.70,
            "chroma_even": 0.50,
            "whiten_hue_stable": 1,
        },
        "white_balance_tint": 8.0,
    },

    # ------------------------------------------------------------------
    # Convention/con-hall set (3, beyond convention_repair_v1)
    # ------------------------------------------------------------------
    "con_fluorescent_v1": {
        # Harsh overhead fluorescent tubing — greenish-white cast, flat
        # top-down shadow under eyes/nose, no color warmth. Aggressive
        # white_balance correction (kelvin pulled cool-to-neutral + tint
        # pushed away from green), C1 hue_unify specifically counters the
        # green skin cast fluorescent light causes (the exact "venue-light
        # color cast" case H3 was built for on hair, applied here to skin).
        "extends": "natural",
        "frequency": {"smooth": 0.35},
        "skin": {
            "equalize": 0.15,
            "hue_unify": 0.70,
            "chroma_even": 0.45,
            "whiten_hue_stable": 1,
            "shine_removal": 0.30,
        },
        "white_balance_kelvin": 5200,
        "white_balance_tint": -12.0,
        "highlights": -15.0,
        "shadows": 20.0,
    },
    "con_mixed_temp_v1": {
        # Multiple light sources at different color temperatures in one
        # frame (LED panel + venue tungsten + daylight from a window) — the
        # exact scenario C1's hue-line unification was purpose-built for
        # per PLAN_EASTWEST_COLOR_SUPREMACY §1a: "记忆色... corrected toward
        # the preferred reproduction locus" regardless of the ambient color
        # chaos. subject_separation isolates the subject from a background
        # that may have entirely different color temperature.
        "extends": "porcelain_unified_v1",
        "skin": {
            "equalize": 0.0,
            "rosy": 0.15,
            "hue_unify": 0.75,
            "chroma_even": 0.60,
            "whiten_hue_stable": 1,
            "shine_removal": 0.35,
        },
        "subject_separation": 35.0,
    },
    "con_crowd_bg_v1": {
        # Crowded convention-hall background — need strong subject/background
        # separation so the subject reads cleanly against visual clutter.
        # subject_separation + vignette do the heavy lifting; airy_haze (C4)
        # is L-thresholded so it naturally softens bright background clutter
        # (booth lighting, other attendees) more than the subject.
        "extends": "natural",
        "frequency": {"smooth": 0.35},
        "skin": {
            "equalize": 0.10,
            "hue_unify": 0.50,
            "chroma_even": 0.35,
            "whiten_hue_stable": 1,
            "shine_removal": 0.30,
        },
        "subject_separation": 65.0,
        "vignette": 30.0,
        "finish": {"airy_haze": 0.20},
    },

    "idol": {
        "extends": "natural",
        "frequency": {"smooth": 0.50},
        "skin": {"equalize": 0.15, "rosy": 0.30},
        "eyes": {"whites": 0.25, "teeth_whiten": 0.25, "iris": 0.35, "catchlight": 0.30},
        "lips": {"tint": "rose", "gloss": 0.25},
        "hair": {"shine": 0.25},
        "dodge_burn": {"amount": 0.20},
        "color_harmony": {"preset": "cosplay", "amount": 0.40},
        "bloom": {"opacity": 0.07},
        "texture": {"opacity": 0.90},
        
        # Modular overrides
        "slimming": 30.0,
        "blush": 25.0,
    },
    "wedding": {
        "extends": "natural",
        "frequency": {"smooth": 0.50},
        "skin": {"equalize": 0.15, "rosy": 0.25},
        "eyes": {"whites": 0.15, "teeth_whiten": 0.15, "iris": 0.20, "catchlight": 0.15},
        "lips": {"tint": "rose", "gloss": 0.10},
        "hair": {"shine": 0.15},
        "dodge_burn": {"amount": 0.10},
        "color_harmony": {"preset": "beauty", "amount": 0.50},
        "bloom": {"opacity": 0.05},
        "texture": {"opacity": 0.90},
        
        # Modular overrides
        "blush": 25.0,
        "lip_finish": "matte",
    },
    "anime_cosplay": {
        "extends": "cosplay",
        "skin": {"equalize": 0.00, "rosy": 0.35},
        "eyes": {"whites": 0.30, "teeth_whiten": 0.30, "iris": 0.35, "catchlight": 0.25},
        "lips": {"tint": "cosplay", "gloss": 0.20},
        "hair": {"shine": 0.30},
        "dodge_burn": {"amount": 0.15},
        "color_harmony": {"preset": "cosplay", "amount": 0.20},
        "bloom": {"opacity": 0.06},
        "texture": {"opacity": 0.90},
    },
    "scifi_cosplay": {
        "extends": "natural",
        "frequency": {"smooth": 0.85, "mid_reduction": 0.75},
        "skin": {"equalize": 0.20, "rosy": 0.45},
        "eyes": {"whites": 0.35, "teeth_whiten": 0.35, "iris": 0.55, "catchlight": 0.50},
        "lips": {"tint": "cosplay", "gloss": 0.45},
        "hair": {"shine": 0.40},
        "dodge_burn": {"amount": 0.35},
        "color_harmony": {"preset": "cyberpunk", "amount": 0.70},
        "bloom": {"opacity": 0.15},
        "texture": {"opacity": 0.25},
        
        # Modular overrides
        "slimming": 0.0,
        "blush": 35.0,
        "nose_blush": True,
        "under_eye_blush": True,
    },
    "fantasy_goddess": {
        "extends": "natural",
        "frequency": {"smooth": 0.75, "mid_reduction": 0.60},
        "skin": {"equalize": 0.20, "rosy": 0.35},
        "eyes": {"whites": 0.30, "teeth_whiten": 0.30, "iris": 0.40, "catchlight": 0.35},
        "lips": {"tint": "pink", "gloss": 0.30},
        "hair": {"shine": 0.30},
        "dodge_burn": {"amount": 0.15},
        "color_harmony": {"preset": "fantasy", "amount": 0.80},
        "bloom": {"opacity": 0.45},
        "texture": {"opacity": 0.25},
        "contrast": -25,
        
        # Modular overrides
        "slimming": 35.0,
        "blush": 30.0,
        "nose_blush": True,
        "under_eye_blush": True,
    },
    "pink_dream": {
        "extends": "natural",
        "frequency": {"smooth": 0.25, "mid_reduction": 0.35},
        "skin": {"equalize": 0.30, "rosy": 0.45},
        "eyes": {"whites": 0.35, "teeth_whiten": 0.35, "iris": 0.50, "catchlight": 0.45},
        "lips": {"tint": None, "gloss": 0.00},
        "hair": {"shine": 0.50},
        "dodge_burn": {"amount": 0.30},
        "color_harmony": {"preset": "pink_dream", "amount": 0.85},
        "bloom": {"opacity": 0.05},
        "texture": {"opacity": 0.90},
        "finish": {"impact": 0.60},
        
        # Modular overrides
        "slimming": 30.0,
        "blush": 30.0,
        "nose_blush": True,
        "under_eye_blush": True,
        "white_costume_lift": True,
    },
    "blue_dream": {
        "extends": "natural",
        "frequency": {"smooth": 0.75, "mid_reduction": 0.60},
        "skin": {"equalize": 0.20, "rosy": 0.30},
        "eyes": {"whites": 0.30, "teeth_whiten": 0.30, "iris": 0.40, "catchlight": 0.35},
        "lips": {"tint": "cosplay", "gloss": 0.30},
        "hair": {"shine": 0.35},
        "dodge_burn": {"amount": 0.20},
        "color_harmony": {"preset": "blue_dream", "amount": 0.80},
        "bloom": {"opacity": 0.20},
        "texture": {"opacity": 0.35},
        "finish": {"impact": 0.50},
        
        # Modular overrides
        "slimming": 30.0,
        "blush": 25.0,
    },
    "xhs_ultrasoft": {
        "extends": "natural",
        "frequency": {"smooth": 0.85, "mid_reduction": 0.75},
        "skin": {"equalize": 0.30, "rosy": 0.50, "relight": 0.55},
        "eyes": {"whites": 0.25, "teeth_whiten": 0.25, "iris": 0.30, "catchlight": 0.25},
        "lips": {"tint": "rose", "gloss": 0.25},
        "hair": {"shine": 0.25},
        "dodge_burn": {"amount": 0.25},
        "color_harmony": {"preset": "xhs_ultrasoft", "amount": 0.90},
        "bloom": {"opacity": 0.12},
        "texture": {"opacity": 0.20},
        "finish": {"impact": 0.35},
        # Soft light sculpting + dreamy glow + soft tonal compression
        "specular_bloom": 20,
        "tonal_curve_strength": 0.35,
        "highlight_rolloff": 0.45,
        "skin_protect": 0.65,
        "shadow_hue": 220, "shadow_sat": 35,
        "midtone_hue": 30, "midtone_sat": 20,
        "highlight_hue": 50, "highlight_sat": 10,
        "micro_restore": 30,
        "grain_strength": 0.30,

        # Modular overrides
        "slimming": 30.0,
        "blush": 25.0,
        "lip_finish": "velvet",
    },
    "xhs_soft_glow": {
        "extends": "natural",
        "frequency": {"smooth": 0.70, "mid_reduction": 0.55},
        "skin": {"equalize": 0.40, "rosy": 0.55, "relight": 0.65},
        "eyes": {"whites": 0.30, "teeth_whiten": 0.30, "iris": 0.32, "catchlight": 0.28},
        "lips": {"tint": "rose", "gloss": 0.30},
        "hair": {"shine": 0.30},
        "dodge_burn": {"amount": 0.30},
        "color_harmony": {"preset": "xhs_ultrasoft", "amount": 0.95},
        "bloom": {"opacity": 0.15},
        "texture": {"opacity": 0.35},
        "finish": {"impact": 0.25},
        # Xiaohongshu soft glow — light sculpting with reduced halo artifacts:
        # face-relight + specular bloom + tonal curve + highlight rolloff
        # + cool/warm split-toning + restored micro-texture + organic grain.
        "specular_bloom": 25,
        "tonal_curve_strength": 0.45,
        "highlight_rolloff": 0.55,
        "skin_protect": 0.75,
        "shadow_hue": 220, "shadow_sat": 40,
        "midtone_hue": 30, "midtone_sat": 25,
        "highlight_hue": 50, "highlight_sat": 15,
        "micro_restore": 35,
        "grain_strength": 0.35,

        # Modular overrides
        "slimming": 30.0,
        "blush": 30.0,
        "lip_finish": "velvet",
        "nose_blush": True,
        "under_eye_blush": True,
    },
    "fuji_porcelain": {
        "extends": "natural",
        "frequency": {"smooth": 0.72, "mid_reduction": 0.75},
        "skin": {"equalize": 0.50, "rosy": 0.70},
        "eyes": {"whites": 0.30, "teeth_whiten": 0.30, "iris": 0.35, "catchlight": 0.30},
        "lips": {"tint": "pink", "gloss": 0.20},
        "hair": {"shine": 0.15},
        "dodge_burn": {"amount": 0.55},
        "color_harmony": {"preset": "natural", "amount": 0.50},
        "bloom": {"opacity": 0.05},
        "texture": {"opacity": 0.35},
        "specular_bloom": 50,
    },
    "anime_cinematic_v1": {
        "extends": "natural",
        "frequency": {"smooth": 0.32, "mid_reduction": 0.22},
        "skin": {"equalize": 0.38, "porcelain": 0.42, "relight": 0.35},
        "relight_azimuth": 45.0,
        "relight_elevation": 35.0,
        "eyes": {"whites": 0.18, "teeth_whiten": 0.18, "iris": 0.22, "catchlight": 0.25, "dark_circles": 0.22},
        "lips": {"tint": None, "gloss": 0.22},
        "hair": {"shine": 0.75},
        "dodge_burn": 18.0,
        "texture": {"opacity": 0.50},
        "brightness": 6.0,
        "contrast": 7.0,
        "highlights": -10.0,
        "shadows": -4.0,
        "whites": 8.0,
        "blacks": 1.0,
        "clarity": 14.0,
        "saturation": 6.0,
        "vibrance": 12.0,
        "shadow_hue": 225.0,
        "shadow_sat": 12.0,
        "midtone_hue": 0.0,
        "midtone_sat": 0.0,
        "highlight_hue": 320.0,
        "highlight_sat": 10.0,
        "bloom": {"opacity": 0.16, "threshold": 190.0},
        "glow": 8.0,
        "vignette": 3.0,
        "sharpen": 11.0,
        "sharpen_radius": 1.2,
        "chromatic_aberration": 1.5,
        "grain": 0.0,
        "specular_bloom_tone": "rosy",
        
        # Modular overrides
        "blush": 30.0,
        "nose_blush": True,
        "under_eye_blush": True,
        "white_costume_lift": True,
    },
    "anime_cinematic_soft": {
        "extends": "anime_cinematic_v1",
        "contrast": 8.0,
        "skin": {"porcelain": 0.50},
        "bloom": {"opacity": 0.22},
        "highlight_sat": 6.0,
        "clarity": 8.0,
        "chromatic_aberration": 2.0,
        "hair": {"shine": 0.85},
    },
    "anime_cinematic_action": {
        "extends": "anime_cinematic_v1",
        "contrast": 18.0,
        "clarity": 24.0,
        "sharpen": 30.0,
        "relight": 45.0,
        "chromatic_aberration": 6.0,
        "bloom": {"opacity": 0.10},
        "vignette": 10.0,
    },
    # NOTE: anime_crystal_void has 7 dead recipe keys that are NOT implemented in engine.py.
    # These keys are silently ignored by the engine (not an error, just no-op):
    # - background_blur, background_desaturation, light_wrap, blue_shadow_grade,
    #   cyan_midtone_grade, subject_sharpen, matte_black
    # See PLAN_MOONLIGHT_PORCELAIN.md §"Bugs found" for details.
    # TODO(Phase 2): Wire these as real ParamSpecs + engine stages if needed.
    "anime_crystal_void": {
        "extends": "anime_cinematic_v1",
        "subject_separation": 0.85,
        "bloom": {"opacity": 0.35},
        "chromatic_aberration": 1.5,
        "eyes": {"iris": 0.20},
        "background_blur": 0.55,  # DEAD KEY — not implemented
        "background_desaturation": 0.40,  # DEAD KEY — not implemented
        "light_wrap": 0.25,  # DEAD KEY — not implemented
        "blue_shadow_grade": 0.75,  # DEAD KEY — not implemented
        "cyan_midtone_grade": 0.45,  # DEAD KEY — not implemented
        "subject_sharpen": 0.30,  # DEAD KEY — not implemented
        "matte_black": 0.10,  # DEAD KEY — not implemented
    },
    "anime_cinematic_fantasy": {
        "extends": "anime_cinematic_v1",
        "brightness": 10.0,
        "bloom": {"opacity": 0.25},
        "glow": 18.0,
        "skin": {"porcelain": 0.55},
        "hair": {"shine": 0.90},
        "highlight_sat": 14.0,
        "shadow_sat": 14.0,
        "chromatic_aberration": 5.0,
    },
    "anime_v2": {
        "extends": "anime_cinematic_v1",
        "skin": {"equalize": 0.45, "porcelain": 0.42, "flatten": 0.55,
                 "quantize": 0.40, "relight": 0.42,
                 "unify": 0.50, "unify_hue": -1.0, "glow": 0.20},
        "frequency": {"smooth": 0.38, "mid_reduction": 0.30},
        "micro_restore": 8,
        "clarity": 8.0,
        "texture": {"opacity": 0.70},
        "bloom": {"opacity": 0.14, "threshold": 195.0},
    },
    "jp_transparent_v1": {
        "extends": "natural",
        "frequency": {"smooth": 0.25},
        "skin": {
            "equalize": 0.08,
            "whiten_hue_stable": 1,
            "micro_dodge_burn": 0.05,
        },
        "finish": {
            "fade_toe": 0.30,
            "highlight_drift": 0.25,
            "airy_haze": 0.20,
            "clarity_split_neg": 0.15,
            "clarity_split_pos": 0.10,
            "impact": 0.05,
        },
    },
    "game_character_v1": {
        # AAA-game-character look: rendered/sculpted lighting and punchy
        # color like a cinematic-trailer character model, while keeping
        # identity intact (moderate smoothing, texture retained, no
        # geometric warp). Contrast with anime_cinematic_v1 (flatter,
        # more illustrated) — this keeps more photographic skin micro-detail
        # and pushes directional relight + micro-contrast instead of porcelain.
        "extends": "natural",
        "frequency": {"smooth": 0.40, "mid_reduction": 0.30},
        # shadow_lift: opt-in local fill-light (shadow_lift.py) for real
        # photographic shade — e.g. a jaw/chin shadow next to a bright prop
        # — that would otherwise look like an untouched patch next to
        # heavily relit skin.
        "skin": {
            "equalize": 0.25,
            "rosy": 0.20,
            "porcelain": 0.15,
            "relight": 0.55,
            "relight_azimuth": 55.0,
            "relight_elevation": 40.0,
            "shadow_lift": 0.35,
            # Restores the real nose-bridge shadow toward the untouched
            # original (see perf_optimizations.py) — smoother surrounding
            # skin can make the source photo's own nose shading read as
            # over-defined even though no single op deepens it.
            "nose_restore": 0.40,
        },
        "eyes": {"whites": 0.25, "teeth_whiten": 0.20, "iris": 0.40, "catchlight": 0.45},
        "lips": {"tint": None, "gloss": 0.15},
        "hair": {"shine": 0.55},
        # Body smoothing/equalize raised (0.30/0.45 -> 0.50/0.60) so exposed
        # legs/arms/chest read closer to the face's retouch level instead of
        # looking comparatively untouched. relight/dodge_burn added (new
        # body_relight.py module, landmark-free) to close the remaining gap:
        # body_skin's smooth+equalize alone can't approach face-level
        # perceived retouch since it has no sculpting/shading step of its
        # own — these two give body a lightweight analog of the face's
        # relight + dodge_burn passes. shadow_lift added for the same local-
        # shade reason as the face-side key above.
        "body_skin": {
            "smooth": 0.50, "equalize": 0.60, "match_face": 0.65,
            "relight": 0.45, "dodge_burn": 0.40, "shadow_lift": 0.35,
        },
        # dodge_burn reduced 30.0 -> 15.0 — 30 produced visibly harsh
        # under-eye/cheek shadow sculpting on strong side-lit photos
        # (verified on DSCF7142); this recipe's own docstring calls for
        # "directional relight + micro-contrast instead of porcelain," not
        # hard sculpting.
        "dodge_burn": 15.0,
        "texture": {"opacity": 0.75},
        "brightness": 3.0,
        "contrast": 16.0,
        "highlights": -12.0,
        "shadows": -8.0,
        "whites": 6.0,
        "blacks": 3.0,
        "clarity": 22.0,
        "saturation": 10.0,
        "vibrance": 18.0,
        "shadow_hue": 210.0,
        "shadow_sat": 14.0,
        "midtone_hue": 25.0,
        "midtone_sat": 6.0,
        "highlight_hue": 40.0,
        "highlight_sat": 12.0,
        "bloom": {"opacity": 0.10, "threshold": 205.0},
        "glow": 4.0,
        "vignette": 8.0,
        "sharpen": 24.0,
        "sharpen_radius": 1.0,
        "chromatic_aberration": 0.5,
        "grain": 0.0,
        "specular_bloom": 35,
        "specular_bloom_tone": "neutral",

        # Modular overrides — kept subtle to preserve identity
        "slimming": 0.0,
        "blush": 10.0,
    },
    "aaa_photoreal_v1": {
        # Targets the "AAA Gold Quality" photoreal brief: over-smooth then
        # reclone real pore texture (texture_transplant, borrowed from
        # pore_realism_v1) instead of leaving skin flat/plastic, soft
        # wraparound relight instead of hard sculpt, gentle highlight
        # rolloff + skin_protect to avoid clipped highlights/crushed
        # shadows. No slimming/geometry change — identity untouched.
        # Note: this engine has no eyelash/eyebrow-specific or background-
        # separation ops, so those brief items aren't directly addressable.
        "extends": "natural",
        "frequency": {"smooth": 0.60, "mid_reduction": 0.35, "nose_smooth": 0.30},
        "skin": {
            "equalize": 0.10,
            "rosy": 0.15,
            "hue_unify": 0.35,
            "chroma_even": 0.30,
            "whiten_hue_stable": 1,
            "texture_transplant": 0.65,
            "relight": 0.30,
            "relight_azimuth": 40.0,
            "relight_elevation": 45.0,
            "shadow_lift": 0.35,
            "nose_restore": 0.40,
        },
        "eyes": {"whites": 0.15, "teeth_whiten": 0.12, "iris": 0.25, "catchlight": 0.30},
        "lips": {"tint": None, "gloss": 0.12},
        "hair": {"shine": 0.30},
        # Tone-match exposed chest/shoulder/arm skin to the retouched face
        # (body_match_v1's fix) — without this, cosplay/swimwear shots show
        # a visible seam where face relight/smoothing stops at the jawline.
        # smooth/equalize raised (0.30/0.45 -> 0.50/0.60, verified on
        # DSCF7117) so exposed legs/arms/chest read closer to the face's
        # retouch level. relight/dodge_burn added (new body_relight.py
        # module) to give body a landmark-free analog of the face's own
        # relight + dodge_burn passes, closing the perceived-intensity gap.
        # shadow_lift addresses real local shade (e.g. a shadowed jaw/chest
        # patch next to a bright prop) without flattening genuine contrast.
        "body_skin": {
            "smooth": 0.50, "equalize": 0.60, "match_face": 0.65,
            "relight": 0.30, "dodge_burn": 0.25, "shadow_lift": 0.35,
        },
        "dodge_burn": 12.0,
        "texture": {"opacity": 0.85},
        "skin_protect": 0.70,
        "highlight_rolloff": 0.50,
        "tonal_curve_strength": 0.25,
        "brightness": 2.0,
        "contrast": 10.0,
        "highlights": -14.0,
        "shadows": -6.0,
        "clarity": 10.0,
        "saturation": 4.0,
        "vibrance": 10.0,
        "bloom": {"opacity": 0.04, "threshold": 215.0},
        "vignette": 4.0,
        "sharpen": 8.0,
        "sharpen_radius": 1.3,
        "chromatic_aberration": 0.0,
        "grain": 0.0,

        # Modular overrides
        "slimming": 0.0,
        "blush": 8.0,
    },
    "aaa_photoreal_v2": {
        # Pushes aaa_photoreal_v1 to match a labeled reference edit brief:
        # brighter/cleaner catchlights + iris detail, cleaner hair
        # specular highlights, subtle satin (not heavy) lip gloss, and an
        # explicit blue-shadow/orange-highlight split-tone for the
        # "cinematic" separation/depth call-out — while keeping the same
        # texture_transplant-based skin (no plastic look) and no geometry
        # change. Still no eyelash-specific op in this engine; the "more
        # detailed eyes" ask is approximated via iris/catchlight strength.
        "extends": "aaa_photoreal_v1",
        "eyes": {"whites": 0.22, "teeth_whiten": 0.15, "iris": 0.40, "catchlight": 0.50},
        "lips": {"tint": None, "gloss": 0.20},
        "hair": {"shine": 0.55},
        # dodge_burn reverted 18.0 -> 12.0 (v1's value) — 18 produced
        # visibly harsh under-eye/cheek shadow sculpting on strong
        # side-lit photos (verified on DSCF7142), contradicting this
        # recipe's own "soft wraparound relight, no hard sculpt" intent.
        "dodge_burn": 12.0,
        "clarity": 14.0,
        "saturation": 6.0,
        "vibrance": 12.0,
        "shadow_hue": 220.0,
        "shadow_sat": 16.0,
        "midtone_hue": 30.0,
        "midtone_sat": 4.0,
        "highlight_hue": 45.0,
        "highlight_sat": 14.0,
        "bloom": {"opacity": 0.06, "threshold": 210.0},
        "vignette": 6.0,
        "sharpen": 12.0,

        # Modular overrides
        "slimming": 0.0,
        "blush": 10.0,
    },
    "zzz_anime_v1": {
        # Targets the "premium modern anime game render" brief (cel-shaded,
        # ZZZ/Genshin-adjacent aesthetic) without reproducing any specific
        # copyrighted character. Built on anime_cinematic_v1's rendered-
        # skin foundation but pushes iris/catchlight size, glossy lips,
        # and rim-light bloom harder for the "expressive eyes + soft bloom
        # + rim light" look; keeps porcelain moderate (not full flatten)
        # so it stays "stylized yet believable" rather than fully flat.
        "extends": "anime_cinematic_v1",
        "skin": {
            "porcelain": 0.45, "relight": 0.40,
            "shadow_lift": 0.35, "nose_restore": 0.40,
        },
        "relight_azimuth": 50.0,
        "relight_elevation": 30.0,
        "eyes": {"whites": 0.25, "teeth_whiten": 0.20, "iris": 0.50, "catchlight": 0.55, "dark_circles": 0.15},
        "lips": {"tint": "pink", "gloss": 0.40},
        "hair": {"shine": 0.85},
        # smooth/equalize raised (0.35/0.45 -> 0.55/0.65) so exposed
        # legs/arms/chest read closer to the porcelain face's retouch
        # level. relight/dodge_burn added (body_relight.py) as a landmark-
        # free analog of the face's own relight + dodge_burn passes.
        # shadow_lift addresses real local shade next to bright props/lights.
        "body_skin": {
            "smooth": 0.55, "equalize": 0.65, "match_face": 0.70,
            "relight": 0.40, "dodge_burn": 0.35, "shadow_lift": 0.35,
        },
        # Override anime_cinematic_v1's dodge_burn (18.0) down to 12.0 —
        # 18 produced visibly harsh under-eye/cheek shadow sculpting on
        # strong side-lit photos (verified on DSCF7142).
        "dodge_burn": 12.0,
        "contrast": 10.0,
        "clarity": 10.0,
        "saturation": 10.0,
        "vibrance": 16.0,
        "bloom": {"opacity": 0.22, "threshold": 185.0},
        "glow": 12.0,
        "vignette": 5.0,
        "sharpen": 14.0,
        "chromatic_aberration": 1.0,

        # Modular overrides
        "blush": 35.0,
        "nose_blush": True,
        "under_eye_blush": True,
    },
    "zzz_anime_v2": {
        # Moderate push past zzz_anime_v1: more rim-light bloom, bigger
        # eyes/catchlights, punchier color — still meant to read as a
        # photo of the real person in cosplay, not a full game render.
        # Porcelain/relight nudged up but stops short of anime_v2's
        # flatten/quantize (that would cel-shade and risk identity loss).
        "extends": "zzz_anime_v1",
        "skin": {"porcelain": 0.55, "relight": 0.55},
        "relight_azimuth": 55.0,
        "relight_elevation": 35.0,
        "eyes": {"whites": 0.30, "teeth_whiten": 0.22, "iris": 0.62, "catchlight": 0.68, "dark_circles": 0.18},
        "lips": {"tint": "pink", "gloss": 0.50},
        "hair": {"shine": 0.95},
        "contrast": 14.0,
        "clarity": 12.0,
        "saturation": 14.0,
        "vibrance": 22.0,
        # Bloom pulled back from 0.32/175 — at that strength it bled into
        # bright flyaway hair strands near the brow and softened them.
        # Higher threshold restricts bloom to true highlights; sharpen
        # raised to recover the strand-level crispness bloom was eating.
        "bloom": {"opacity": 0.22, "threshold": 195.0},
        "glow": 14.0,
        "vignette": 7.0,
        "sharpen": 22.0,
        "chromatic_aberration": 2.0,
        "specular_bloom": 45,
        "specular_bloom_tone": "rosy",

        # Modular overrides
        "blush": 40.0,
        "nose_blush": True,
        "under_eye_blush": True,
    },
}

RECIPES["soft"] = RECIPES["anime_cinematic_soft"]
RECIPES["action"] = RECIPES["anime_cinematic_action"]
RECIPES["fantasy"] = RECIPES["anime_cinematic_fantasy"]


# ---------------------------------------------------------------------------
# Phase 1.c — Official Fuji Film Simulations
# ---------------------------------------------------------------------------
# Three canonical Fuji film simulations. Each recipe combines the new
# Phase 1.a foundation primitives (tonal_curve_strength, skin_protect,
# highlight_rolloff, grain_strength) with the standard recipe sections.
# Source: docs/FUJI_COLOR_RESEARCH.md (Worker H, P0.9).
#
# For end-user consumption, prefer the JSON files in presets/ as the
# single source of truth for the descriptive metadata (curves, calibration,
# HSL adjustments). These in-Python entries are the engine-side execution
# form (flat scalars consumed by build_context).

RECIPES["provia"] = {
    "extends": "natural",
    "frequency": {"smooth": 0.30, "mid_reduction": 0.35},
    "skin": {"equalize": 0.20, "rosy": 0.15},
    "eyes": {"whites": 0.10, "teeth_whiten": 0.10, "iris": 0.10, "catchlight": 0.10},
    "lips": {"tint": None, "gloss": 0.10},
    "hair": {"shine": 0.10},
    "dodge_burn": {"amount": 0.05},
    "color_harmony": {"preset": "natural", "amount": 0.10},
    "bloom": {"opacity": 0.02},
    "texture": {"opacity": 0.95},
    "tonal_curve_strength": 0.40,
    "skin_protect": 0.20,
    "highlight_rolloff": 0.20,
    "grain_strength": 0.00,
    "saturation": 5.0,
    "contrast": 0.0,
    "highlights": 0.0,
    "shadows": 0.0,
    "sharpen": 40.0,
    "sharpen_radius": 1.2,
    "vignette": 0.0,
}

RECIPES["astia"] = {
    "extends": "natural",
    "frequency": {"smooth": 0.50, "mid_reduction": 0.45},
    "skin": {"equalize": 0.30, "rosy": 0.35},
    "eyes": {"whites": 0.15, "teeth_whiten": 0.15, "iris": 0.20, "catchlight": 0.15},
    "lips": {"tint": "pink", "gloss": 0.15},
    "hair": {"shine": 0.10},
    "dodge_burn": {"amount": 0.10},
    "color_harmony": {"preset": "natural", "amount": 0.20},
    "bloom": {"opacity": 0.04},
    "texture": {"opacity": 0.90},
    "tonal_curve_strength": 0.55,
    "skin_protect": 0.85,
    "highlight_rolloff": 0.50,
    "grain_strength": 0.00,
    "saturation": 2.0,
    "contrast": 8.0,
    "highlights": -3.0,
    "shadows": 0.0,
    "sharpen": 25.0,
    "sharpen_radius": 1.2,
    "vignette": 0.0,
    "midtone_hue": 35.0,
    "midtone_sat": 4.0,
    "shadow_hue": 30.0,
    "shadow_sat": 6.0,
    "highlight_hue": 40.0,
    "highlight_sat": 4.0,
}

RECIPES["classic_chrome"] = {
    "extends": "natural",
    "frequency": {"smooth": 0.45, "mid_reduction": 0.40},
    "skin": {"equalize": 0.25, "rosy": 0.20},
    "eyes": {"whites": 0.10, "teeth_whiten": 0.10, "iris": 0.15, "catchlight": 0.10},
    "lips": {"tint": None, "gloss": 0.10},
    "hair": {"shine": 0.10},
    "dodge_burn": {"amount": 0.15},
    "color_harmony": {"preset": "natural", "amount": 0.15},
    "bloom": {"opacity": 0.05},
    "texture": {"opacity": 0.90},
    "tonal_curve_strength": 0.85,
    "skin_protect": 0.30,
    "highlight_rolloff": 0.60,
    "grain_strength": 0.30,
    "saturation": -15.0,
    "contrast": 24.0,
    "highlights": -20.0,
    "shadows": -20.0,
    "sharpen": 20.0,
    "sharpen_radius": 1.0,
    "vignette": 6.0,
    "midtone_hue": 40.0,
    "midtone_sat": 4.0,
    "shadow_hue": 145.0,
    "shadow_sat": 14.0,
    "highlight_hue": 200.0,
    "highlight_sat": 6.0,
}


# Canonical name list for the three official Fuji film simulations.
# Used by the GUI dropdown, CLI helpers, and integration tests.
FUJI_SIM_NAMES: List[str] = ["classic_chrome", "astia", "provia"]


def list_fuji_sims() -> List[str]:
    """Return the names of the three official Fuji film simulations.

    These are the recipes that exist in :data:`RECIPES` and match the
    JSON files shipped in ``presets/``. Useful for populating a GUI
    dropdown or a CLI helper.
    """
    return sorted(name for name in FUJI_SIM_NAMES if name in RECIPES)

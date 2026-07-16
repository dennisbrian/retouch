"""Recipe cookbook for the Retouch Engine (T4).

A read-only browsable index of every recipe in :data:`retouch.recipes.RECIPES`,
grouped by category, with metadata (description, parent chain, key params).
This is the data layer behind the recipe-cookbook UI — it exposes no GUI
itself, only the query functions a UI (Gradio/CLI) would call.

Public API:
    list_recipes(category=None) -> List[RecipeInfo]
    get_recipe_info(name)       -> Optional[RecipeInfo]
    search_recipes(query)       -> List[RecipeInfo]
    list_categories()           -> List[str]
    RecipeInfo                  — dataclass describing one recipe

Categories:
    portrait, cosplay, outdoor, studio, convention, creative

Category assignment is keyword-driven via :data:`_CATEGORY_RULES`: a recipe
name or its parent chain matches a set of keywords. A recipe with no match
falls back to its parent's category, then to "creative" as the catch-all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .recipes import RECIPES


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

CATEGORIES: List[str] = [
    "portrait",
    "cosplay",
    "outdoor",
    "studio",
    "convention",
    "creative",
]

# Keyword -> category mapping, evaluated in order. First match wins.
# The keywords are matched against the recipe name (lowercased).
_CATEGORY_RULES: List[tuple] = [
    (("convention", "con_", "convention_repair"), "convention"),
    (("outdoor_",), "outdoor"),
    (("studio_",), "studio"),
    (("cosplay", "anime", "scifi", "fantasy", "idol", "game_character",
      "zzz_", "aaa_"), "cosplay"),
    (("portrait", "beauty", "natural", "porcelain", "milk_skin",
      "korean", "wedding", "wrinkle", "pore_realism", "body_match",
      "natural_polish", "full_showcase", "jp_transparent",
      "float32_beauty", "float32_glow", "float32_cinema"), "portrait"),
    (("pink_dream", "blue_dream", "xhs_", "fuji_", "xiaohongshu"), "creative"),
]

# Fuji film sims are categorized as creative (film-look emulation).
_FUJI_SIMS = {
    "provia", "astia", "classic_chrome",
    "velvia", "classic_neg", "nostalgic_neg",
    "pro_neg_hi", "pro_neg_std",
    "eterna", "eterna_bleach_bypass",
    "acros", "monochrome", "sepia",
    "reala_ace",
}


# ---------------------------------------------------------------------------
# Recipe metadata
# ---------------------------------------------------------------------------

# Curated one-line descriptions for the headline recipes. Recipes not listed
# here get a generated description ("Extends <parent>." or "Base recipe.").
_DESCRIPTIONS: Dict[str, str] = {
    "natural": "Baseline natural retouch — restrained smoothing and tone.",
    "portrait": "Classic portrait: moderate smoothing, eye/lip enhancement.",
    "cosplay": "Cosplay default: stronger skin evenness, vivid eyes, lip tint.",
    "cosplay_3d": "Cosplay with equalize disabled for a 3D sculpted read.",
    "cosplay_no_eq": "Cosplay variant with skin equalize off.",
    "xiaohongshu": "Xiaohongshu look: light sculpting, dreamy glow, warm/cool split.",
    "beauty": "Beauty glam: high gloss, saturated color harmony.",
    "korean_beauty": "Korean glass-skin: soft smoothing, pink lip, velvet finish.",
    "porcelain_unified_v1": "C1 hue-unified porcelain skin — flagship base.",
    "cosplay_sculpt_v1": "C2 structural sculpt over unified porcelain skin.",
    "convention_repair_v1": "Convention-hall shine removal + venue cast fix.",
    "milk_skin_v1": "Korean milk-skin: high L, low chroma variance, water glow.",
    "float32_beauty_v1": "Float32 fidelity showcase — smooth tonal beauty.",
    "float32_cinema_v1": "Float32 cinematic finish — fade toe, split tone, grain.",
    "float32_glow_v1": "Float32 glow showcase — large smooth gradients, bloom.",
    "natural_polish_v1": "Barely-retouched 'just better' restrained polish.",
    "wrinkle_free_glow_v1": "S5 wrinkle/line softening for close-up beauty.",
    "pore_realism_v1": "S6 texture transplant over heavy smoothing.",
    "body_match_v1": "S1 body-to-face tone match for full-body shots.",
    "full_showcase_v1": "Light-touch kitchen-sink across every stage.",
    "outdoor_harsh_sun_v1": "Midday sun: highlight recovery, sweat-shine removal.",
    "outdoor_golden_hour_v1": "Warm backlight: relight, halation, warm WB push.",
    "outdoor_overcast_v1": "Flat overcast: contrast/clarity lift, C2 sculpt.",
    "outdoor_backlit_v1": "Backlit subject: exposure lift, rim-light preserve.",
    "studio_hard_flash_v1": "Hard flash: specular removal, sculpt, vignette.",
    "studio_softbox_v1": "Softbox: clean color, moderate sculpt for dimension.",
    "studio_ringlight_v1": "Ring-light: boosted catchlight, even frontal fill.",
    "studio_gel_color_v1": "Gel wash: skin pulled neutral, background keeps cast.",
    "con_fluorescent_v1": "Fluorescent venue: green-cast fix, flat-shadow lift.",
    "con_mixed_temp_v1": "Mixed color temperatures: strong C1 unify + separation.",
    "con_crowd_bg_v1": "Crowded background: subject separation + airy haze.",
    "idol": "Idol look: vivid eyes, rose lips, moderate slimming.",
    "wedding": "Wedding: soft romantic grade, matte lips, subtle blush.",
    "anime_cosplay": "Anime cosplay: cosplay base with equalize off.",
    "scifi_cosplay": "Sci-fi cosplay: cyberpunk grade, heavy bloom, low texture.",
    "fantasy_goddess": "Fantasy goddess: high bloom, fantasy grade, soft contrast.",
    "pink_dream": "Pink dream: pink_dream grade, high equalize, glossy finish.",
    "blue_dream": "Blue dream: blue_dream grade, heavy bloom, dreamy finish.",
    "xhs_ultrasoft": "XHS ultra-soft: heavy smoothing, dreamy glow, velvet lips.",
    "xhs_soft_glow": "XHS soft glow: stronger relight, dreamy tonal compression.",
    "fuji_porcelain": "Fuji porcelain: heavy dodge/burn, specular bloom.",
    "anime_cinematic_v1": "Anime cinematic base: relight, bloom, chromatic AB.",
    "anime_cinematic_soft": "Softer anime cinematic: lower contrast, more bloom.",
    "anime_cinematic_action": "Action anime: high contrast/clarity, heavy sharpen.",
    "anime_crystal_void": "Crystal void: smoky desaturated matte-black backdrop.",
    "anime_cinematic_fantasy": "Fantasy anime: brighter, more bloom, glossy hair.",
    "anime_v2": "Anime v2: porcelain + flatten + quantize for cel-shade read.",
    "jp_transparent_v1": "Japanese transparent skin: airy haze, fade toe.",
    "game_character_v1": "AAA game character: directional relight, micro-contrast.",
    "game_character_v2": "Game character v2: subsurface-scatter skin, warm translucent shadows.",
    "cosplay_character_showcase_v1": "Character showreel: dewy skin, vivid eyes, wig-lace blend, no geometry edits.",
    "aaa_photoreal_v1": "AAA photoreal: pore transplant, soft relight, no geometry.",
    "aaa_photoreal_v2": "AAA photoreal v2: bigger eyes, cleaner hair, split tone.",
    "zzz_anime_v1": "ZZZ-style anime: expressive eyes, rim-light bloom, glossy lips.",
    "zzz_anime_v2": "ZZZ anime v2: pushed bloom, bigger eyes, punchier color.",
    "provia": "Fuji Provia: neutral standard film sim, clean color.",
    "astia": "Fuji Astia: soft portrait film sim, warm midtones.",
    "classic_chrome": "Fuji Classic Chrome: desaturated, high-contrast documentary.",
    "velvia": "Fuji Velvia: punchy saturated landscape film, deep greens/blues.",
    "classic_neg": "Fuji Classic Negative: Superia-style, saturated greens, warm yellows.",
    "nostalgic_neg": "Fuji Nostalgic Negative: lifted amber shadows, muted highlights.",
    "pro_neg_hi": "Fuji Pro Neg Hi: portrait negative, firmer contrast.",
    "pro_neg_std": "Fuji Pro Neg Std: portrait negative, softer/flatter contrast.",
    "eterna": "Fuji Eterna: cinema film stock, low contrast, flat log-style tones.",
    "eterna_bleach_bypass": "Fuji Eterna Bleach Bypass: high-contrast, near-desaturated, gritty.",
    "acros": "Fuji Acros: fine-grain B&W, smooth tonal transition.",
    "monochrome": "Fuji Monochrome: standard neutral B&W.",
    "sepia": "Fuji Sepia: B&W with warm amber split-tone.",
    "reala_ace": "Fuji Reala Ace: high-fidelity punchy-but-natural, deep blacks.",
}


# ---------------------------------------------------------------------------
# RecipeInfo dataclass
# ---------------------------------------------------------------------------

@dataclass
class RecipeInfo:
    """Metadata for one recipe, as exposed to the cookbook UI."""

    name: str
    category: str
    description: str
    extends: Optional[str]
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "extends": self.extends,
            "params": self.params,
        }


# ---------------------------------------------------------------------------
# Categorization helpers
# ---------------------------------------------------------------------------

def _categorize(name: str, recipe: Dict[str, Any]) -> str:
    """Return the category for *name* using keyword rules and parent fallback."""
    lower = name.lower()

    if name in _FUJI_SIMS:
        return "creative"

    for keywords, category in _CATEGORY_RULES:
        for kw in keywords:
            if kw in lower:
                return category

    parent = recipe.get("extends")
    if parent and parent in RECIPES:
        return _categorize(parent, RECIPES[parent])

    return "creative"


def _describe(name: str, recipe: Dict[str, Any]) -> str:
    """Return a human description for *name*."""
    if name in _DESCRIPTIONS:
        return _DESCRIPTIONS[name]
    parent = recipe.get("extends")
    if parent:
        return f"Extends '{parent}'."
    return "Base recipe."


def _key_params(recipe: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a small set of headline params for UI display."""
    out: Dict[str, Any] = {}
    freq = recipe.get("frequency")
    if isinstance(freq, dict) and "smooth" in freq:
        out["smooth"] = freq["smooth"]
    skin = recipe.get("skin")
    if isinstance(skin, dict):
        for k in ("equalize", "rosy", "hue_unify", "sculpt", "shine_removal"):
            if k in skin:
                out[f"skin.{k}"] = skin[k]
    eyes = recipe.get("eyes")
    if isinstance(eyes, dict) and "catchlight" in eyes:
        out["eyes.catchlight"] = eyes["catchlight"]
    bloom = recipe.get("bloom")
    if isinstance(bloom, dict) and "opacity" in bloom:
        out["bloom"] = bloom["opacity"]
    for k in ("slimming", "blush", "vignette", "grain_strength"):
        if k in recipe:
            out[k] = recipe[k]
    return out


# ---------------------------------------------------------------------------
# Cookbook API
# ---------------------------------------------------------------------------

def list_categories() -> List[str]:
    """Return the ordered list of recipe categories."""
    return list(CATEGORIES)


def _build_info(name: str) -> Optional[RecipeInfo]:
    recipe = RECIPES.get(name)
    if recipe is None:
        return None
    return RecipeInfo(
        name=name,
        category=_categorize(name, recipe),
        description=_describe(name, recipe),
        extends=recipe.get("extends"),
        params=_key_params(recipe),
    )


def list_recipes(category: Optional[str] = None) -> List[RecipeInfo]:
    """List all recipes, optionally filtered by *category*.

    If *category* is None, all recipes are returned sorted by name. If
    *category* is given (case-insensitive) and is not a known category,
    an empty list is returned.
    """
    if category is not None:
        category = category.lower()
        if category not in CATEGORIES:
            return []

    infos: List[RecipeInfo] = []
    for name in sorted(RECIPES.keys()):
        info = _build_info(name)
        if info is None:
            continue
        if category is not None and info.category != category:
            continue
        infos.append(info)
    return infos


def get_recipe_info(name: str) -> Optional[RecipeInfo]:
    """Return metadata for a single recipe, or None if not found.

    Lookup is case-insensitive on the recipe name.
    """
    if name is None:
        return None
    lower = name.lower()
    for key in RECIPES.keys():
        if key.lower() == lower:
            return _build_info(key)
    return None


def search_recipes(query: str) -> List[RecipeInfo]:
    """Search recipes by name or description (case-insensitive substring).

    Returns matches sorted by name. An empty query returns all recipes
    (same as ``list_recipes(None)``).
    """
    if not query or not query.strip():
        return list_recipes(None)

    q = query.lower().strip()
    matches: List[RecipeInfo] = []
    for name in sorted(RECIPES.keys()):
        info = _build_info(name)
        if info is None:
            continue
        if q in info.name.lower() or q in info.description.lower():
            matches.append(info)
    return matches


__all__ = [
    "RecipeInfo",
    "CATEGORIES",
    "list_categories",
    "list_recipes",
    "get_recipe_info",
    "search_recipes",
]

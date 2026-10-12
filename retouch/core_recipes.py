"""Small, correction-first recipe set used for release certification.

The full recipe catalog remains available.  This list is intentionally kept
separate from the catalog so certification has a stable scope and does not
silently expand when a new recipe is added.
"""

from __future__ import annotations

from typing import Dict, Tuple


CORE_RECIPE_NAMES: Tuple[str, ...] = (
    "natural",
    "portrait",
    "male",
    "female",
    "senior",
    "natural_polish_v1",
    "clear_skin_v1",
    "mole_safe_portrait_v1",
    "aniso_pore_real_v1",
    "tired_eye_rescue_v1",
    "wedding_timeless_v1",
    "cosplay_clear_v1",
)

# These labels drive the review worksheet; they are not claims that a recipe
# is safe for every member of a category.
CORE_RECIPE_REVIEW_DIMENSIONS: Dict[str, str] = {
    "natural": "baseline / mixed lighting",
    "portrait": "portrait baseline",
    "male": "facial hair / masculine presentation",
    "female": "beauty baseline",
    "senior": "mature skin and texture restraint",
    "natural_polish_v1": "restrained polish",
    "clear_skin_v1": "skin cleanup without plastic texture",
    "mole_safe_portrait_v1": "mark preservation",
    "aniso_pore_real_v1": "pores and high-frequency detail",
    "tired_eye_rescue_v1": "under-eye correction",
    "wedding_timeless_v1": "event / mixed light",
    "cosplay_clear_v1": "makeup, wigs and costume detail",
}

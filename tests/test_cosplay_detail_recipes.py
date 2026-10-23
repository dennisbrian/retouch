"""Scene-specific compositions remain explicit children of the natural candidate."""
import pytest
from retouch.engine import build_context
from retouch.params import recipe_to_params, resolve_recipe
from retouch.recipe_cookbook import get_recipe_info
from retouch.recipes import (RECIPES, COSPLAY_DETAIL_RECIPE_NAMES, CURATED_RECIPE_NAMES,
                            EXPERIMENTAL_RECIPE_NAMES, RECOMMENDED_RECIPE_NAMES)


@pytest.mark.parametrize('name', COSPLAY_DETAIL_RECIPE_NAMES)
def test_opt_in_identity_base_and_ui_engine_agreement(name):
    assert RECIPES[name]['extends'] == 'natural_v2'
    assert name in CURATED_RECIPE_NAMES and name in EXPERIMENTAL_RECIPE_NAMES
    assert name not in RECOMMENDED_RECIPE_NAMES
    ctx = build_context(name, resolve_recipe(name), {})
    gui = recipe_to_params(name)
    for key in ('whiten','blemish','slimming','reshape_chin_length','blush','nose_blush','under_eye_blush'):
        assert getattr(ctx, key) == gui[key] == 0
    assert ctx.mole_protect == 1.0
    assert ctx.fa02_texture_mode == 'legacy'
    assert ctx.makeup_coverage_even == ctx.makeup_cake_reduce == 0
    info = get_recipe_info(name)
    assert info.category == 'cosplay' and 'Experimental' in info.description


def test_paint_does_not_heal_painted_details():
    gui = recipe_to_params('cosplay_paint_detail_v1')
    assert gui['spot_heal'] == gui['blemish'] == gui['equalize'] == 0
    assert gui['body_paint'] == 45 and gui['mark_policy'] == 'preserve_all'


def test_wig_contact_units_and_no_extra_shine():
    name = 'cosplay_wig_contacts_v1'
    ctx, gui = build_context(name, resolve_recipe(name), {}), recipe_to_params(name)
    for key, value in {'hair_deglare':0,'iris_pop':25,'cosplay_wig_lace_blend':25,'hair_enhance':0}.items():
        assert getattr(ctx,key) == gui[key] == value


def test_stage_keeps_global_scene_colour():
    ctx = build_context('cosplay_stage_detail_v1', resolve_recipe('cosplay_stage_detail_v1'), {})
    assert ctx.face_shadow_fill == 35
    assert ctx.skin_hue_unify == 0 and ctx.saturation == 0
    assert resolve_recipe('cosplay_stage_detail_v1')['color_harmony']['amount'] == 0


def test_prosthetic_strength_is_bounded_candidate():
    assert recipe_to_params('cosplay_prosthetic_detail_v1')['prosthetic_blend'] == 40


@pytest.mark.parametrize('cast', [(0.5,0.8,1.25),(1.2,0.7,0.6)])
def test_stage_fill_keeps_coloured_light_in_linear_chromaticity(cast):
    import numpy as np
    from retouch import face_shadow_fill as fill
    from tests.test_face_shadow_fill import _face, _lin, _stops, IED
    img, regions = _face(gain=0.5)
    coloured = np.clip(_lin(img)*np.array(cast,np.float32),0,1)
    source = fill._linear_to_srgb(coloured)*255
    strength = recipe_to_params('cosplay_stage_detail_v1')['face_shadow_fill']
    out = fill.face_shadow_fill(source,strength,IED,regions['skin'])
    before, after = _lin(source), _lin(out)
    chroma = lambda x: x/(x.sum(axis=2,keepdims=True)+1e-8)
    assert np.abs(chroma(after)-chroma(before)).max() < 1e-5
    assert np.median((_stops(out)-_stops(source))[regions['shadow']]) > 0.1

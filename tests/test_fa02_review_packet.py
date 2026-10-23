"""Draft review inputs must not silently become accepted or change source identity."""
import copy
import pytest
from scripts.qa.fa02_review_packet import validate_draft, collect_cases


@pytest.fixture
def packet():
    case = dict(id='example', arrays_sha256='a'*64, source_sha256='b'*64,
                split='dev', crop_xywh=[10, 20, 64, 64], width=64, height=64)
    record = {k: case[k] for k in ('arrays_sha256', 'source_sha256', 'split', 'crop_xywh')}
    record.update(case_id='example', selections=[dict(id='hair_1', kind='hair_profile', p0=[20,20], p1=[30,20])])
    draft = dict(schema_version=1, status='draft_not_accepted', reviewer='reviewer',
                 coordinate_space='crop_native_pixels', cases=[record])
    return draft, [case]


def test_valid_profile_stays_unaccepted(packet):
    draft, cases = packet
    original = copy.deepcopy(draft)
    assert validate_draft(draft, cases) == dict(status='draft_not_accepted', cases=1,
                                               selections=1, qualification='not_scored_or_approved')
    assert draft == original


@pytest.mark.parametrize('key,value', [('arrays_sha256','stale'), ('source_sha256','stale'),
                                     ('split','locked_test'), ('crop_xywh',[0,0,64,64])])
def test_stale_source_or_coordinates_refused(packet, key, value):
    draft, cases = packet
    draft['cases'][0][key] = value
    with pytest.raises(ValueError, match='stale or changed'):
        validate_draft(draft, cases)


@pytest.mark.parametrize('selection', [
    dict(id='a',kind='hair_profile',p0=[20,20],p1=[20,20]),
    dict(id='a',kind='hair_profile',p0=[20,20],p1=[64,20]),
    dict(id='a',kind='pore',center=[1,1],radius=2),
    dict(id='a',kind='noise_region',xywh=[60,60,8,8]),
    dict(id='a',kind='unknown',xywh=[10,10,4,4]),
])
def test_bad_target_geometry_refused(packet, selection):
    draft, cases = packet
    draft['cases'][0]['selections'] = [selection]
    with pytest.raises(ValueError):
        validate_draft(draft, cases)


def test_locked_assets_are_not_opened_for_review(tmp_path):
    import json
    (tmp_path/'fa02_manifest.json').write_text(json.dumps(dict(
        purpose='development_pilot', cases=[dict(split='locked_test')])))
    with pytest.raises(ValueError, match='locked or unknown'):
        collect_cases([tmp_path])


def test_draft_cannot_claim_acceptance(packet):
    draft, cases = packet
    draft['status'] = 'owner_approved'
    with pytest.raises(ValueError, match='unaccepted'):
        validate_draft(draft, cases)

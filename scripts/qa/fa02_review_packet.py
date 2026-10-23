"""Build a self-contained native-pixel selection packet from accepted dev pilots.

Selections are draft scoring inputs, never accepted annotations. This tool does
not modify manifests, source pixels, masks, splits, or the frozen scoring lock.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fa02_texture_representation_experiment as experiment


def _png(array):
    ok, buffer = cv2.imencode('.png', array)
    if not ok:
        raise ValueError('Could not encode review display')
    return 'data:image/png;base64,' + base64.b64encode(buffer).decode('ascii')


def collect_cases(directories):
    """Validate provenance/checksums before exposing a development case."""
    cases, seen = [], set()
    for directory in directories:
        directory = Path(directory).resolve()
        path = directory / 'fa02_manifest.json'
        manifest = json.loads(path.read_text())
        if manifest.get('purpose') != 'development_pilot':
            raise ValueError(f'{path}: only development_pilot is accepted')
        if any(c.get('split') not in ('dev', 'calibration') for c in manifest['cases']):
            raise ValueError(f'{path}: locked or unknown splits cannot enter a review packet')
        experiment.validate_manifest(manifest, directory)
        for case in manifest['cases']:
            key = case['id']
            if key in seen:
                raise ValueError(f'Duplicate review case: {case["id"]}')
            seen.add(key)
            arrays = experiment.load_arrays(directory / case['arrays'])
            h, w = arrays['X'].shape[:2]
            # Display only; no quantized projection is ever a scoring target.
            images = {name: _png(np.rint(arrays[name]).astype(np.uint8)) for name in ('X', 'S')}
            overlay = np.zeros((h, w, 4), np.uint8)
            overlay[arrays['allow'] > 0] = (70, 180, 40, 45)
            overlay[arrays['corrected'] > 0] = (230, 60, 100, 125)
            overlay[arrays['protected'] > 0] = (40, 50, 240, 125)
            annotation = case.get('annotation', {})
            categories = ('pore_regions', 'fine_hair_regions', 'noise_regions',
                          'makeup_edge_regions', 'protected_identity_marks',
                          'corrected_defect_regions', 'uncertainty_regions')
            cases.append({
                'id': case['id'], 'manifest': str(path), 'arrays_sha256': case['arrays_sha256'],
                'source_sha256': case['source_sha256'], 'split': case['split'],
                'person_ids': case['person_ids'], 'crop_xywh': case['crop_xywh'],
                'width': w, 'height': h, 'images': images, 'mask_overlay': _png(overlay),
                'existing_regions': {k: annotation.get(k, []) for k in categories},
                'pores': case.get('pores', []), 'profiles': case.get('profiles', []),
            })
    if not cases:
        raise ValueError('No development cases supplied')
    return cases


def validate_draft(draft, cases):
    """Check a saved draft's identity/coordinates without accepting its labels."""
    if draft.get('schema_version') != 1 or draft.get('status') != 'draft_not_accepted':
        raise ValueError('Expected a version 1 unaccepted review draft')
    if not isinstance(draft.get('reviewer'), str) or not draft['reviewer'].strip():
        raise ValueError('Draft reviewer is required')
    if draft.get('coordinate_space') != 'crop_native_pixels':
        raise ValueError('Draft coordinates must be crop-native pixels')
    records = draft.get('cases')
    if not isinstance(records, list) or len(records) != len(cases):
        raise ValueError('Draft must retain every packet case')
    expected = {case['id']: case for case in cases}
    seen, count = set(), 0
    for record in records:
        case_id = record.get('case_id')
        if case_id not in expected or case_id in seen:
            raise ValueError('Draft has an unknown or duplicate case')
        seen.add(case_id)
        case = expected[case_id]
        for key in ('arrays_sha256', 'source_sha256', 'split', 'crop_xywh'):
            if record.get(key) != case[key]:
                raise ValueError(f'{case_id}: stale or changed {key}')
        def point(value):
            if (not isinstance(value, list) or len(value) != 2
                    or any(type(v) is not int for v in value)
                    or not 0 <= value[0] < case['width']
                    or not 0 <= value[1] < case['height']):
                raise ValueError(f'{case_id}: point outside native crop')
        selections = record.get('selections')
        if not isinstance(selections, list):
            raise ValueError('Selections must be a list')
        ids = set()
        for selection in selections:
            ident = selection.get('id')
            if not isinstance(ident, str) or not ident or ident in ids:
                raise ValueError('Selection IDs must be nonempty and unique per case')
            ids.add(ident)
            kind = selection.get('kind')
            if kind == 'hair_profile':
                point(selection.get('p0')); point(selection.get('p1'))
                if selection['p0'] == selection['p1']:
                    raise ValueError('Hair profile needs two different endpoints')
            elif kind == 'pore':
                point(selection.get('center'))
                radius = selection.get('radius')
                if type(radius) is not int or not 1 <= radius <= 20:
                    raise ValueError('Pore radius must be an integer from 1 to 20')
                x, y = selection['center']
                point([x-radius, y-radius]); point([x+radius, y+radius])
            elif kind in ('noise_region', 'corrected_region', 'protected_region', 'uncertain_region'):
                rect = selection.get('xywh')
                if (not isinstance(rect, list) or len(rect) != 4
                        or any(type(v) is not int for v in rect) or min(rect[2:]) <= 0):
                    raise ValueError('Region needs a positive integer rectangle')
                x, y, w, h = rect
                point([x, y]); point([x+w-1, y+h-1])
            else:
                raise ValueError('Unknown draft selection kind')
            count += 1
    return {'status': 'draft_not_accepted', 'cases': len(seen), 'selections': count,
            'qualification': 'not_scored_or_approved'}


def build_packet(directories, output):
    """Write only the requested HTML; all selections export as separate drafts."""
    output = Path(output).resolve()
    if output.suffix.lower() != '.html':
        raise ValueError('Review output must be an .html file')
    cases = collect_cases(directories)
    payload = json.dumps(cases, ensure_ascii=True).replace('<', '\\u003c')
    page = TEMPLATE.replace('__CASES__', payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding='utf-8')
    return {'html': str(output), 'cases': len(cases),
            'html_sha256': hashlib.sha256(page.encode()).hexdigest(), 'status': 'draft_review_only'}


TEMPLATE = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Texture scoring review</title>
<style>
body{font:16px system-ui;background:#151b22;color:#e7edf4;margin:24px}button,select,input{font:inherit;padding:8px;margin:4px;background:#25313e;color:inherit;border:1px solid #64748b;border-radius:5px}button{cursor:pointer}p{max-width:1000px;line-height:1.5}.views{display:flex;gap:24px;overflow:auto;padding:12px 0}canvas{display:block;image-rendering:pixelated;cursor:crosshair;border:1px solid #64748b}pre{white-space:pre-wrap;background:#202a35;padding:12px}label{white-space:nowrap}.muted{color:#aebdce}h2{font-size:18px}
</style>
<h1>Texture scoring review</h1>
<p>Review native crop pixels and select scoring inputs. Existing labels and supports are read-only. Every download is a <strong>draft</strong>: nothing is imported, accepted, or used to tune restoration automatically.</p>
<p class="muted">X is the saved input crop; S is its saved processed crop. Displays are rounded 8-bit PNG projections without resizing. Original floating-point arrays remain the scoring authority. Green support = allowed; blue = corrected; red = protected. A noisy region alone is not a paired clean/noisy control.</p>
<select id="cases" aria-label="Case"></select><label>Zoom <select id="zoom"><option>1</option><option>2</option><option>4</option></select>×</label>
<label><input id="masks" type="checkbox">Show support masks</label><label><input id="existing" type="checkbox" checked>Show existing labels</label>
<p id="identity"></p>
<select id="kind" aria-label="Selection type"><option value="hair_profile">Hair cross-section (two endpoints)</option><option value="pore">Pore center (one point)</option><option value="noise_region">Noise region (two corners)</option><option value="corrected_region">Corrected defect (two corners)</option><option value="protected_region">Protected detail (two corners)</option><option value="uncertain_region">Uncertain (two corners)</option></select>
<label>Pore radius <input id="radius" type="number" min="1" max="20" value="2" style="width:60px">px</label>
<input id="note" placeholder="What is visibly present?" aria-label="Selection note" size="32">
<button id="undo">Undo last selection</button><button id="cancel">Cancel pending point</button>
<p id="status">Choose a case. For hair, draw across one isolated strand, not along it.</p>
<div class="views"><section><h2>X — input crop</h2><canvas id="x"></canvas></section><section><h2>S — saved processed crop</h2><canvas id="s"></canvas></section></div>
<h2>Draft selections</h2><pre id="draft"></pre>
<label>Reviewer <input id="reviewer" placeholder="Your name" autocomplete="off"></label><button id="download">Download all draft selections</button>
<p class="muted">Export does not claim owner approval or biological pore truth. Keep uncertain detail labelled uncertain. Review strand ownership: wig strands crossing skin are not proof of fine skin hair. Calibration and locked-test subjects must be assigned separately before tuning.</p>
<script>
const data=__CASES__, $=id=>document.getElementById(id), drafts=data.map(()=>[]);
let selected=0, pending=null, generation=0, pictures={};
const canvases=[$('x'),$('s')];
for(let i=0;i<data.length;i++){let o=new Option(data[i].id,i);$('cases').add(o)}
function draw(){let c=data[selected];for(let [n,canvas] of canvases.entries()){
 canvas.width=c.width;canvas.height=c.height;canvas.style.width=c.width*Number($('zoom').value)+'px';canvas.style.height=c.height*Number($('zoom').value)+'px';let ctx=canvas.getContext('2d');let im=pictures[n?'S':'X'];if(im)ctx.drawImage(im,0,0);
 if($('masks').checked&&pictures.mask)ctx.drawImage(pictures.mask,0,0);
 if($('existing').checked){ctx.strokeStyle='#ffb347';ctx.lineWidth=1;for(let regions of Object.values(c.existing_regions))for(let r of regions){ctx.strokeRect(...r.xywh)}}
 ctx.strokeStyle='#00ffff';ctx.fillStyle='#00ffff';ctx.lineWidth=1;
 for(let r of drafts[selected]){ctx.beginPath();if(r.center){ctx.arc(...r.center,r.radius,0,Math.PI*2)}else if(r.p0){ctx.moveTo(...r.p0);ctx.lineTo(...r.p1)}else{ctx.rect(...r.xywh)}ctx.stroke()}
 if(pending){ctx.fillRect(pending[0]-2,pending[1]-2,4,4)}
 }$('draft').textContent=JSON.stringify(drafts[selected],null,2)}
async function load(){let token=++generation,c=data[selected];pending=null;pictures={};$('identity').textContent=`${c.id} | ${c.width} × ${c.height} native crop | ${c.split} | accepted pores: ${c.existing_regions.pore_regions.length}; hair: ${c.existing_regions.fine_hair_regions.length}; noise: ${c.existing_regions.noise_regions.length} | existing scoring profiles: ${c.profiles.length}`;
 draw();let entries=await Promise.all(Object.entries({...c.images,mask:c.mask_overlay}).map(([k,url])=>new Promise((resolve,reject)=>{let im=new Image();im.onload=()=>resolve([k,im]);im.onerror=reject;im.src=url})));if(token!==generation)return;pictures=Object.fromEntries(entries);draw()}
function click(e){if(!pictures.X)return;let c=data[selected],box=e.target.getBoundingClientRect(),p=[Math.floor((e.clientX-box.left-e.target.clientLeft)*c.width/e.target.clientWidth),Math.floor((e.clientY-box.top-e.target.clientTop)*c.height/e.target.clientHeight)];p=p.map((v,i)=>Math.max(0,Math.min(v,(i?c.height:c.width)-1)));let kind=$('kind').value,r={kind,note:$('note').value.trim()};
 if(kind==='pore'){let radius=Number($('radius').value);if(!Number.isInteger(radius)||radius<1||radius>20){$('status').textContent='Choose an integer radius from 1 to 20.';return}if(p[0]-radius<0||p[1]-radius<0||p[0]+radius>=c.width||p[1]+radius>=c.height){$('status').textContent='Pore neighborhood must fit inside the crop.';return}r.center=p;r.radius=radius}
 else if(!pending){pending=p;draw();$('status').textContent='Choose the second endpoint or corner.';return}
 else{let a=pending;if(a[0]===p[0]&&a[1]===p[1])return;if(kind==='hair_profile'){r.p0=a;r.p1=p}else{let w=Math.abs(a[0]-p[0]),h=Math.abs(a[1]-p[1]);if(!w||!h)return;r.xywh=[Math.min(a[0],p[0]),Math.min(a[1],p[1]),w,h]}pending=null}
 r.id=`draft_${selected}_${drafts[selected].length+1}`;drafts[selected].push(r);draw();$('status').textContent='Draft selection added. Review its meaning before accepting any scoring input.'}
for(let canvas of canvases)canvas.onclick=click;
$('cases').onchange=()=>{selected=Number($('cases').value);load()};$('zoom').onchange=draw;$('masks').onchange=draw;$('existing').onchange=draw;$('kind').onchange=()=>{pending=null;draw()};
$('undo').onclick=()=>{drafts[selected].pop();pending=null;draw()};$('cancel').onclick=()=>{pending=null;draw()};
$('download').onclick=()=>{if(pending){$('status').textContent='Finish or cancel the pending selection before exporting.';return}let reviewer=$('reviewer').value.trim();if(!reviewer){$('status').textContent='Enter the reviewer name before exporting.';return}let packet={schema_version:1,status:'draft_not_accepted',reviewer,created:new Date().toISOString(),coordinate_space:'crop_native_pixels',cases:data.map((c,i)=>({case_id:c.id,manifest:c.manifest,arrays_sha256:c.arrays_sha256,source_sha256:c.source_sha256,split:c.split,crop_xywh:c.crop_xywh,selections:drafts[i]}))};let url=URL.createObjectURL(new Blob([JSON.stringify(packet,null,2)],{type:'application/json'}));let a=document.createElement('a');a.href=url;a.download='texture-scoring-draft.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
load();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--html', type=Path)
    parser.add_argument('--draft', type=Path, help='Validate a downloaded draft; does not accept labels')
    args = parser.parse_args()
    if bool(args.html) == bool(args.draft):
        parser.error('Choose exactly one of --html or --draft')
    if args.draft:
        result = validate_draft(json.loads(args.draft.read_text()), collect_cases(args.directories))
    else:
        result = build_packet(args.directories, args.html)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

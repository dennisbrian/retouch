// Control-flow/API-shape test ONLY. This mock has no Photoshop pixel renderer.
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');
const code = fs.readFileSync(process.argv[2], 'utf8').replace(/^#target photoshop\s*/, '');
const files = {}, savedImages = [], docs = [];
function File(path) { this.fsName = path; this.name = path.split('/').pop(); }
File.prototype.copy = function (p) { files[p] = 'copied'; return true; };
File.prototype.open = function () { return true; };
File.prototype.write = function (s) { files[this.fsName] = s; };
File.prototype.close = function () {};
File.openDialog = () => new File('/fixture/rgb_atlas_8.png');
function Folder(path) { this.fsName = path; this.exists = false; }
Folder.prototype.create = () => true;
Folder.selectDialog = () => new Folder('/captures');
function doc(name) {
  const d = {name, fullName: new File('/fixture/' + name), mode: 'RGB', bitsPerChannel: 8,
    colorProfileName: 'sRGB IEC61966-2.1', width: {as: () => 1024}, height: {as: () => 160},
    layers: [{}], activeLayer: {duplicate: () => ({})},
    duplicate: n => doc(n), flatten() { this.flattened = true; },
    saveAs(file, options, copy, ext) { savedImages.push({name: file.name, options, copy, ext, top: {...this.activeLayer}, flattened: this.flattened}); },
    close() { const i = docs.indexOf(this); assert(i >= 0); docs.splice(i, 1); }};
  docs.push(d); return d;
}
const previous = doc('untouched_user_document.psd');
const app = {documents: docs, activeDocument: previous, version: 'MOCK ONLY', colorSettings: 'MOCK preset', displayDialogs: 'original', open: f => doc(f.name)};
const modes = ['NORMAL', 'MULTIPLY', 'SCREEN', 'OVERLAY', 'HARDLIGHT', 'SOFTLIGHT', 'COLORDODGE', 'COLORBURN', 'EXCLUSION', 'LINEARLIGHT', 'VIVIDLIGHT'];
const context = {File, Folder, app, TiffSaveOptions: function () {}, TIFFEncoding: {NONE: 'NONE'},
  Extension: {LOWERCASE: 'lower'}, SaveOptions: {DONOTSAVECHANGES: 'no'}, DialogModes: {ALL: 'all', NO: 'none'},
  DocumentMode: {RGB: 'RGB'}, BitsPerChannelType: {EIGHT: 8, SIXTEEN: 16},
  BlendMode: Object.fromEntries(modes.map(x => [x, x])), prompt: () => '', confirm: () => false, alert: () => {}};
vm.runInNewContext(code, context); // Parses and executes ordinary ES3 JSX body.
assert.equal(savedImages.length, 42);
for (const e of savedImages) {
  assert(e.flattened && e.copy && e.options.embedColorProfile);
  assert.equal(e.options.imageCompression, 'NONE');
  assert.equal(e.options.layers, false);
  assert.equal(e.options.transparency, false);
}
for (const e of savedImages.slice(1)) assert.equal(e.top.fillOpacity, 100);
for (const e of savedImages.slice(2)) assert([25,50,75,100].includes(e.top.opacity));
const meta = JSON.parse(Object.entries(files).find(([name]) => name.endsWith('capture_metadata.json'))[1]);
assert.equal(meta.capture_status, 'complete');
assert.equal(meta.cases.length, 40);
assert.equal(new Set(meta.cases.map(x => x.file)).size, 40);
assert.equal(meta.blend_gamma_setting.enabled, 'UNKNOWN');
assert.equal(meta.settings_confirmed, false);
assert.equal(meta.document_profile, 'sRGB IEC61966-2.1');
assert.equal(app.displayDialogs, 'original');
assert.equal(app.activeDocument, previous);
assert.deepEqual(docs, [previous]);
console.log('42 lossless exports; 40 cases; UNKNOWN gamma; cleanup OK');

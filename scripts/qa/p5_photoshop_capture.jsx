#target photoshop
/* P5 RESEARCH ONLY. Run File > Scripts > Browse in actual Photoshop.
 * No preferences are changed. No already-open document is edited or closed.
 * Uses the Photoshop JavaScript DOM (Adobe CC JavaScript Reference 2019).
 * Captures pixels; it does NOT itself certify parity. UNKNOWN settings remain
 * UNKNOWN. Run the Python comparison after capture and review its evidence gate.
 */
(function () {
    function quote(s) {
        return '"' + String(s).replace(/[\\"\x00-\x1f]/g, function (c) {
            if (c === '"') return '\\"';
            if (c === '\\') return '\\\\';
            return '\\u' + ('0000' + c.charCodeAt(0).toString(16)).slice(-4);
        }) + '"';
    }
    // ExtendScript installations do not all supply JSON.stringify.
    function json(v) {
        if (v === null) return 'null';
        if (typeof v === 'string') return quote(v);
        if (typeof v === 'number') return isFinite(v) ? String(v) : 'null';
        if (typeof v === 'boolean') return String(v);
        var parts = [], k;
        if (v instanceof Array) {
            for (k = 0; k < v.length; k++) parts.push(json(v[k]));
            return '[' + parts.join(',') + ']';
        }
        for (k in v) if (v.hasOwnProperty(k)) parts.push(quote(k) + ':' + json(v[k]));
        return '{' + parts.join(',') + '}';
    }
    function writeMetadata(folder, metadata) {
        var file = new File(folder.fsName + '/capture_metadata.json');
        file.encoding = 'UTF8';
        if (!file.open('w')) throw new Error('Cannot write capture metadata');
        file.write(json(metadata) + '\n');
        file.close();
    }
    function safeRead(object, key) {
        try { return object[key] === undefined || object[key] === null ? 'UNAVAILABLE' : (String(object[key]) || 'UNAVAILABLE'); }
        catch (e) { return 'UNAVAILABLE'; }
    }
    var input = File.openDialog('P5: select generated rgb_atlas_8.png or rgb_atlas_16.png');
    if (!input) return;
    if (!/\.(png|tif|tiff)$/i.test(input.name)) throw new Error('Use an opaque RGB PNG/TIFF fixture');
    var i;
    for (i = 0; i < app.documents.length; i++) {
        try {
            if (app.documents[i].fullName.fsName === input.fsName) {
                alert('Close the selected fixture first. P5 will not reuse an open document.');
                return;
            }
        } catch (ignored) { /* Unsaved unrelated document; leave it alone. */ }
    }
    var parent = Folder.selectDialog('P5: choose parent for a NEW timestamped capture directory');
    if (!parent) return;
    var out = new Folder(parent.fsName + '/p5_capture_' + new Date().getTime());
    if (out.exists || !out.create()) throw new Error('Refusing to overwrite a capture directory');
    var meta = {
        schema: 'p5_capture_v1', producer: 'adobe_photoshop', capture_status: 'incomplete',
        photoshop_version: safeRead(app, 'version'), document_profile: 'UNKNOWN', bits: 'UNKNOWN',
        color_settings_preset: safeRead(app, 'colorSettings'),
        blend_gamma_setting: {enabled: 'UNKNOWN', gamma: 'UNAVAILABLE', source: 'DOM access unavailable; optional operator attestation'},
        settings_confirmed: false, settings_evidence: null, operator: 'UNKNOWN',
        capture_notes: 'Duplicate opaque equal layers; layer Opacity, NOT Fill; no color conversion requested.',
        fill_opacity: 100, opaque_equal_layers: true, blend_if: 'disabled', layer_effects: false, masks: 'none',
        baseline: 'baseline.tif', normal_control: 'normal_100.tif', cases: [],
        source_fixture: 'input_fixture.' + input.name.split('.').pop().toLowerCase(),
        source_path: input.fsName, export: 'flattened TIFF, uncompressed, embedded document ICC',
        settings_access: 'Color Settings preset name is not proof of the blend gamma setting.',
        script: 'p5_photoshop_capture.jsx; static/mock-tested until run in Photoshop'
    };
    meta.operator = prompt('Optional capture operator/name (blank means UNKNOWN)', '') || 'UNKNOWN';
    // No guessed Action Manager keys and no silent preference changes.
    if (confirm('Have you inspected Edit > Color Settings > More Options and can attach evidence of Blend RGB Colors Using Gamma?')) {
        var setting = prompt('Enter OFF or the checked numeric gamma (for example 1.00). Leave blank if unknown.', '');
        var valid = false;
        if (setting && setting.toUpperCase() === 'OFF') {
            meta.blend_gamma_setting = {enabled: false, gamma: 'NOT_APPLICABLE', source: 'operator attestation'};
            valid = true;
        } else if (setting && isFinite(Number(setting)) && Number(setting) > 0) {
            meta.blend_gamma_setting = {enabled: true, gamma: Number(setting), source: 'operator attestation'};
            valid = true;
        }
        if (valid) {
            var proof = File.openDialog('Select saved settings screenshot/evidence (Cancel keeps capture unverified)');
            if (proof) {
                var proofName = 'settings_evidence.' + proof.name.split('.').pop().toLowerCase();
                if (!proof.copy(out.fsName + '/' + proofName)) throw new Error('Cannot copy settings evidence');
                meta.settings_evidence = proofName;
                meta.settings_confirmed = true;
            }
        }
    }
    writeMetadata(out, meta);
    var originalDialogs = app.displayDialogs, previous = app.documents.length ? app.activeDocument : null;
    var opened = null, baseline = null, working = null;
    function exportTiff(doc, name) {
        var options = new TiffSaveOptions();
        options.embedColorProfile = true;
        options.imageCompression = TIFFEncoding.NONE;
        options.layers = false;
        options.transparency = false;
        options.alphaChannels = false;
        doc.saveAs(new File(out.fsName + '/' + name), options, true, Extension.LOWERCASE);
    }
    function capture(name, blendMode, opacity) {
        working = baseline.duplicate('P5_' + name);
        app.activeDocument = working;
        var top = working.activeLayer.duplicate();
        working.activeLayer = top;
        top.blendMode = blendMode;
        top.opacity = opacity;
        top.fillOpacity = 100;
        if (top.blendMode !== blendMode || top.opacity !== opacity || top.fillOpacity !== 100)
            throw new Error('Requested layer settings did not stick: ' + name);
        working.flatten();
        exportTiff(working, name);
        working.close(SaveOptions.DONOTSAVECHANGES);
        working = null;
    }
    try {
        if (!input.copy(out.fsName + '/' + meta.source_fixture)) throw new Error('Cannot preserve source fixture');
        // Permit profile warnings on opening. Choose KEEP EMBEDDED PROFILE.
        app.displayDialogs = DialogModes.ALL;
        opened = app.open(input);
        if (opened.mode !== DocumentMode.RGB) throw new Error('Fixture must already be RGB; no conversion allowed');
        if (opened.bitsPerChannel === BitsPerChannelType.EIGHT) meta.bits = 8;
        else if (opened.bitsPerChannel === BitsPerChannelType.SIXTEEN) meta.bits = 16;
        else throw new Error('Only 8/16-bit captures are in scope');
        if (opened.layers.length !== 1) throw new Error('Fixture must be a single flat RGB image');
        meta.document_profile = safeRead(opened, 'colorProfileName');
        meta.width_pixels = Number(opened.width.as('px'));
        meta.height_pixels = Number(opened.height.as('px'));
        baseline = opened.duplicate('P5_baseline');
        opened.close(SaveOptions.DONOTSAVECHANGES);
        opened = null;
        app.displayDialogs = DialogModes.NO;
        baseline.flatten();
        exportTiff(baseline, meta.baseline);
        capture(meta.normal_control, BlendMode.NORMAL, 100);
        var modes = [
            ['multiply', BlendMode.MULTIPLY], ['screen', BlendMode.SCREEN],
            ['overlay', BlendMode.OVERLAY], ['hard_light', BlendMode.HARDLIGHT],
            ['soft_light', BlendMode.SOFTLIGHT], ['color_dodge', BlendMode.COLORDODGE],
            ['color_burn', BlendMode.COLORBURN], ['exclusion', BlendMode.EXCLUSION],
            ['linear_light', BlendMode.LINEARLIGHT], ['vivid_light', BlendMode.VIVIDLIGHT]
        ];
        var opacities = [25, 50, 75, 100], j, name;
        for (i = 0; i < modes.length; i++) for (j = 0; j < opacities.length; j++) {
            name = modes[i][0] + '_' + ('000' + opacities[j]).slice(-3) + '.tif';
            capture(name, modes[i][1], opacities[j]);
            meta.cases.push({mode: modes[i][0], opacity: opacities[j]/100, file: name});
        }
        meta.capture_status = 'complete';
        writeMetadata(out, meta);
        alert('P5 captured 40 cases + baseline + Normal control.\n' + out.fsName + '\nRun Python compare. Capture alone is NOT a parity claim.');
    } catch (error) {
        meta.capture_status = 'failed';
        meta.failure = String(error);
        writeMetadata(out, meta);
        alert('P5 capture incomplete: ' + error + '\nPartial data preserved in ' + out.fsName);
    } finally {
        if (working) working.close(SaveOptions.DONOTSAVECHANGES);
        if (baseline) baseline.close(SaveOptions.DONOTSAVECHANGES);
        if (opened) opened.close(SaveOptions.DONOTSAVECHANGES);
        app.displayDialogs = originalDialogs;
        if (previous) app.activeDocument = previous;
    }
})();

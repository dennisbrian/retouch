// Layer editor canvas: talks to retouch/editor/web.py. The server owns the
// document, history and pixels; this page draws the bounded preview, the
// brush cursor and the live stroke, and sends whole strokes on release.
(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const canvas = $('view');
  const ctx = canvas.getContext('2d');
  const stage = $('stage');
  const MIN_ZOOM = 0.01, MAX_ZOOM = 32;

  let sessionId = null;
  let state = { open: false };
  let previewBitmap = null;     // composite at preview scale
  let maskOverlay = null;       // red where the selected layer's mask hides it
  let view = { zoom: 1, x: 0, y: 0, fitted: true };   // screen px per document px, offset in CSS px
  let tool = 'hide';
  let stroke = null;            // { points: [[x, y], ...] } in document px
  let pan = null;
  let spaceDown = false;
  let pointer = null;           // last pointer position in CSS px
  let busy = 0;
  let queue = Promise.resolve();

  // ---- server ---------------------------------------------------------
  async function request(path, options) {
    options = options || {};
    options.headers = Object.assign({ 'X-Editor-Token': window.EDITOR_TOKEN }, options.headers || {});
    const response = await fetch(path, options);
    if (!response.ok) {
      let detail = {};
      try { detail = await response.json(); } catch (e) { /* not JSON */ }
      const error = new Error(detail.error || ('Request failed (' + response.status + ')'));
      error.detail = detail;
      error.status = response.status;
      throw error;
    }
    return response;
  }

  async function startSession() {
    let saved = null;
    try { saved = sessionStorage.getItem('retouch-editor-session'); } catch (e) { /* storage off */ }
    const response = await request('/editor/api/session', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: saved }),
    });
    const data = await response.json();
    sessionId = data.session_id;
    try { sessionStorage.setItem('retouch-editor-session', sessionId); } catch (e) { /* storage off */ }
    await applyState(data);
  }

  // Actions run one at a time, in order, so strokes never race each other.
  function act(action, args, message) {
    return actRaw(action, args, message).catch(async (error) => {
      if (error.detail && error.detail.expired) { await startSession().catch(() => {}); }
      reportError(error);
      return null;
    });
  }

  // Save and export ask before replacing an existing file.
  async function actConfirm(action, args, message) {
    try {
      return await actRaw(action, args, message);
    } catch (error) {
      if (error.status === 409 && error.detail && error.detail.exists) {
        if (!confirm(error.detail.exists + ' already exists. Replace it?')) { return null; }
        try {
          return await actRaw(action, Object.assign({}, args, { overwrite: true }), message);
        } catch (again) { reportError(again); return null; }
      }
      reportError(error);
      return null;
    }
  }

  function actRaw(action, args, message) {
    const run = async () => {
      setBusy(1);
      try {
        const response = await request('/editor/api/' + sessionId + '/action', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: action, args: args || {} }),
        });
        const data = await response.json();
        await applyState(data);
        setStatus(data.message || message || '');
        return data;
      } finally {
        setBusy(-1);
      }
    };
    const result = queue.then(run, run);
    queue = result.catch(() => {});
    return result;
  }

  async function applyState(next) {
    const previous = state;
    state = next;
    if (!state.open) { previewBitmap = null; maskOverlay = null; renderPanel(); draw(); return; }
    const resized = !previous.open || previous.width !== state.width || previous.height !== state.height;
    if (!previous.open || previous.revision !== state.revision || previous.title !== state.title || resized) {
      previewBitmap = await loadBitmap('/editor/api/' + sessionId + '/preview.png');
    }
    await refreshMask(previous);
    if (resized) {
      // Start a new canvas with a brush about 4% of its long side.
      $('size').value = clamp(Math.round(Math.max(state.width, state.height) * 0.04), 1, 2000);
      showValues();
    }
    if (resized || view.fitted) { fit(); }
    renderPanel();
    draw();
  }

  async function loadBitmap(path) {
    const response = await request(path);
    return createImageBitmap(await response.blob());
  }

  async function refreshMask(previous) {
    const layer = selectedLayer();
    if (!$('showMask').checked || !layer || !layer.has_mask) { maskOverlay = null; return; }
    if (maskOverlay && previous && previous.revision === state.revision &&
        maskOverlay.layer === layer.id) { return; }
    const bitmap = await loadBitmap('/editor/api/' + sessionId + '/mask.png?layer=' + encodeURIComponent(layer.id));
    const off = document.createElement('canvas');
    off.width = bitmap.width; off.height = bitmap.height;
    const octx = off.getContext('2d');
    octx.drawImage(bitmap, 0, 0);
    const image = octx.getImageData(0, 0, off.width, off.height);
    const data = image.data;
    for (let i = 0; i < data.length; i += 4) {
      const hidden = 255 - data[i];
      data[i] = 230; data[i + 1] = 30; data[i + 2] = 30; data[i + 3] = hidden * 0.55;
    }
    octx.putImageData(image, 0, 0);
    maskOverlay = { canvas: off, layer: layer.id, rect: layer.preview_rect };
  }

  // ---- drawing --------------------------------------------------------
  function resizeCanvas() {
    const ratio = window.devicePixelRatio || 1;
    const rect = stage.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(rect.width * ratio));
    canvas.height = Math.max(1, Math.round(rect.height * ratio));
    if (view.fitted) { fit(); }
    draw();
  }

  function fit() {
    if (!state.open) { return; }
    const rect = stage.getBoundingClientRect();
    const zoom = Math.min((rect.width - 24) / state.width, (rect.height - 24) / state.height);
    view.zoom = clamp(zoom, MIN_ZOOM, MAX_ZOOM);
    view.x = (rect.width - state.width * view.zoom) / 2;
    view.y = (rect.height - state.height * view.zoom) / 2;
    view.fitted = true;
  }

  function zoomAt(factor, cx, cy) {
    const zoom = clamp(view.zoom * factor, MIN_ZOOM, MAX_ZOOM);
    const docX = (cx - view.x) / view.zoom, docY = (cy - view.y) / view.zoom;
    view.zoom = zoom;
    view.x = cx - docX * zoom;
    view.y = cy - docY * zoom;
    view.fitted = false;
    draw();
  }

  function checker(w, h, x, y) {
    const size = 12;
    const css = getComputedStyle(document.documentElement);
    ctx.fillStyle = css.getPropertyValue('--check-a');
    ctx.fillRect(x, y, w, h);
    ctx.fillStyle = css.getPropertyValue('--check-b');
    ctx.save();
    ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip();
    for (let yy = 0; yy < h; yy += size) {
      for (let xx = ((yy / size) % 2) * size; xx < w; xx += size * 2) {
        ctx.fillRect(x + xx, y + yy, size, size);
      }
    }
    ctx.restore();
  }

  function draw() {
    const ratio = window.devicePixelRatio || 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    $('empty').style.display = state.open ? 'none' : 'flex';
    $('zoomV').textContent = Math.round(view.zoom * 100) + '%';
    if (!state.open) { return; }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const w = state.width * view.zoom, h = state.height * view.zoom;
    checker(w, h, view.x, view.y);
    ctx.imageSmoothingEnabled = view.zoom * state.preview_scale < 2;
    ctx.imageSmoothingQuality = 'high';
    if (previewBitmap) { ctx.drawImage(previewBitmap, view.x, view.y, w, h); }
    if (maskOverlay) {
      const s = view.zoom / state.preview_scale;
      const r = maskOverlay.rect;
      ctx.drawImage(maskOverlay.canvas, view.x + r[0] * s, view.y + r[1] * s, r[2] * s, r[3] * s);
    }
    if (stroke) { drawStroke(); }
    drawCursor();
  }

  function drawStroke() {
    const diameter = brushSettings().diameter * view.zoom;
    ctx.save();
    ctx.globalAlpha = 0.45 * brushSettings().opacity;
    ctx.strokeStyle = tool === 'hide' ? '#e11d48' : '#16a34a';
    ctx.fillStyle = ctx.strokeStyle;
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.lineWidth = diameter;
    const pts = stroke.points.map(([x, y]) => [view.x + x * view.zoom, view.y + y * view.zoom]);
    if (pts.length === 1) {
      ctx.beginPath(); ctx.arc(pts[0][0], pts[0][1], diameter / 2, 0, Math.PI * 2); ctx.fill();
    } else {
      ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
      for (const p of pts.slice(1)) { ctx.lineTo(p[0], p[1]); }
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawCursor() {
    if (!pointer || tool === 'pan' || spaceDown || pan) { return; }
    const radius = brushSettings().diameter * view.zoom / 2;
    ctx.save();
    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(0,0,0,.8)';
    ctx.beginPath(); ctx.arc(pointer.x, pointer.y, Math.max(radius, 1.5), 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = 'rgba(255,255,255,.9)';
    ctx.beginPath(); ctx.arc(pointer.x, pointer.y, Math.max(radius - 1, 0.5), 0, Math.PI * 2); ctx.stroke();
    ctx.restore();
  }

  // ---- panel ----------------------------------------------------------
  function selectedLayer() {
    if (!state.open) { return null; }
    return state.layers.find((layer) => layer.id === state.selected_id) || null;
  }

  function renderPanel() {
    const list = $('layers');
    list.textContent = '';
    const open = state.open;
    if (open) {
      for (const layer of state.layers.slice().reverse()) {
        const item = document.createElement('li');
        if (layer.id === state.selected_id) { item.className = 'selected'; }
        const eye = document.createElement('input');
        eye.type = 'checkbox'; eye.checked = layer.visible; eye.title = 'Show or hide this layer';
        eye.addEventListener('click', (event) => {
          event.stopPropagation();
          act('update_layer', { layer_id: layer.id, visible: eye.checked });
        });
        const name = document.createElement('span');
        name.className = 'name'; name.textContent = layer.name; name.title = layer.name;
        item.append(eye, name);
        if (layer.has_mask) {
          const tag = document.createElement('span');
          tag.className = 'tag' + (layer.mask_enabled ? '' : ' off');
          tag.textContent = 'mask';
          item.append(tag);
        }
        if (layer.opacity < 1) {
          const tag = document.createElement('span');
          tag.className = 'tag';
          tag.textContent = Math.round(layer.opacity * 100) + '%';
          item.append(tag);
        }
        item.addEventListener('click', () => { if (layer.id !== state.selected_id) { act('select', { layer_id: layer.id }); } });
        item.addEventListener('dblclick', () => renameLayer(layer));
        list.append(item);
      }
    }
    const layer = selectedLayer();
    const index = layer ? state.layers.indexOf(layer) : -1;
    $('undo').disabled = !open || !state.can_undo;
    $('redo').disabled = !open || !state.can_redo;
    $('undo').title = open && state.undo_label ? 'Undo ' + state.undo_label.toLowerCase() + ' (Ctrl+Z)' : 'Undo (Ctrl+Z)';
    $('redo').title = open && state.redo_label ? 'Redo ' + state.redo_label.toLowerCase() + ' (Ctrl+Shift+Z)' : 'Redo (Ctrl+Shift+Z)';
    for (const id of ['save', 'saveAs', 'export', 'addLayer', 'uploadLayer']) { $(id).disabled = !open; }
    $('save').textContent = open && state.dirty ? 'Save •' : 'Save';
    for (const id of ['up', 'down', 'rename', 'remove', 'layerOpacity']) { $(id).disabled = !layer; }
    $('up').disabled = !layer || index === state.layers.length - 1;
    $('down').disabled = !layer || index <= 0;
    $('maskAdd').disabled = $('maskHideAll').disabled = !layer || layer.has_mask || layer.blank;
    $('maskToggle').disabled = $('maskRemove').disabled = !layer || !layer.has_mask;
    $('maskToggle').textContent = layer && layer.has_mask && !layer.mask_enabled ? 'Mask on' : 'Mask off';
    if (layer && document.activeElement !== $('layerOpacity')) {
      $('layerOpacity').value = Math.round(layer.opacity * 100);
    }
    $('layerOpacityV').textContent = $('layerOpacity').value + '%';
    document.title = open ? (state.dirty ? '• ' : '') + state.title + ' — Layer Editor' : 'Layer Editor';
  }

  function renameLayer(layer) {
    const name = prompt('Layer name', layer.name);
    if (name !== null && name.trim() && name.trim() !== layer.name) {
      act('update_layer', { layer_id: layer.id, name: name.trim() });
    }
  }

  // ---- brush and pointer ----------------------------------------------
  function brushSettings() {
    return {
      diameter: Number($('size').value),
      hardness: Number($('hardness').value) / 100,
      opacity: Number($('brushOpacity').value) / 100,
    };
  }

  function toDocument(event) {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left, y = event.clientY - rect.top;
    return [(x - view.x) / view.zoom, (y - view.y) / view.zoom];
  }

  function setTool(name) {
    tool = name;
    $('toolHide').classList.toggle('on', name === 'hide');
    $('toolReveal').classList.toggle('on', name === 'reveal');
    $('toolPan').classList.toggle('on', name === 'pan');
    updateCursor();
    draw();
  }

  function updateCursor() {
    canvas.style.cursor = pan ? 'grabbing' : (tool === 'pan' || spaceDown) ? 'grab' : 'none';
  }

  canvas.addEventListener('pointerdown', (event) => {
    if (!state.open) { return; }
    canvas.setPointerCapture(event.pointerId);
    if (tool === 'pan' || spaceDown || event.button === 1) {
      pan = { x: event.clientX, y: event.clientY, vx: view.x, vy: view.y };
      updateCursor();
      return;
    }
    if (event.button !== 0) { return; }
    const layer = selectedLayer();
    if (!layer) { setStatus('Select a layer first.', true); return; }
    if (!layer.visible) { setStatus('Show the layer before painting its mask.', true); return; }
    if (layer.has_mask && !layer.mask_enabled) { setStatus('Turn the mask on before painting it.', true); return; }
    if (tool === 'reveal' && !layer.has_mask) { setStatus('Nothing is hidden on this layer yet.', true); return; }
    stroke = { points: [toDocument(event)] };
    draw();
  });

  canvas.addEventListener('pointermove', (event) => {
    const rect = canvas.getBoundingClientRect();
    pointer = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    if (pan) {
      view.x = pan.vx + event.clientX - pan.x;
      view.y = pan.vy + event.clientY - pan.y;
      view.fitted = false;
    } else if (stroke) {
      const events = event.getCoalescedEvents ? event.getCoalescedEvents() : [event];
      for (const e of (events.length ? events : [event])) { stroke.points.push(toDocument(e)); }
    }
    draw();
  });

  function endPointer(event) {
    if (pan) { pan = null; updateCursor(); return; }
    if (!stroke) { return; }
    const points = simplify(stroke.points);
    const layer = selectedLayer();
    const settings = brushSettings();
    act('brush', Object.assign({ layer_id: layer.id, points: points, reveal: tool === 'reveal' }, settings))
      .then(() => { stroke = null; draw(); });
    if (event && event.type === 'pointercancel') { stroke = null; draw(); }
  }
  canvas.addEventListener('pointerup', endPointer);
  canvas.addEventListener('pointercancel', endPointer);
  canvas.addEventListener('pointerleave', () => { pointer = null; draw(); });

  // Drop points closer than a quarter pixel: the server lays its own dabs.
  function simplify(points) {
    const out = [points[0]];
    for (const p of points.slice(1)) {
      const last = out[out.length - 1];
      if (Math.hypot(p[0] - last[0], p[1] - last[1]) >= 0.25) { out.push(p); }
    }
    return out.map(([x, y]) => [Math.round(x * 100) / 100, Math.round(y * 100) / 100]);
  }

  canvas.addEventListener('wheel', (event) => {
    if (!state.open) { return; }
    event.preventDefault();
    const rect = canvas.getBoundingClientRect();
    if (event.ctrlKey || event.metaKey || Math.abs(event.deltaY) >= 50 || event.deltaMode) {
      zoomAt(Math.exp(-event.deltaY * (event.deltaMode ? 0.05 : 0.002)), event.clientX - rect.left, event.clientY - rect.top);
    } else {
      view.x -= event.deltaX; view.y -= event.deltaY; view.fitted = false; draw();
    }
  }, { passive: false });

  // ---- commands -------------------------------------------------------
  function confirmDiscard() {
    return !state.open || !state.dirty ||
      confirm('“' + state.title + '” has unsaved changes. Discard them?');
  }

  function pathValue() { return $('path').value.trim(); }

  $('open').addEventListener('click', () => {
    if (!pathValue()) { setStatus('Type the full path of a photo or .comp project first.', true); return; }
    if (!confirmDiscard()) { return; }
    view.fitted = true;
    act('open', { path: pathValue() }, 'Opened.');
  });
  $('path').addEventListener('keydown', (event) => { if (event.key === 'Enter') { $('open').click(); } });
  $('addLayer').addEventListener('click', () => {
    if (!pathValue()) { setStatus('Type the full path of the photo to add first.', true); return; }
    act('add_layer', { path: pathValue() }, 'Layer added.');
  });
  let uploadMode = 'open';
  $('upload').addEventListener('click', () => { uploadMode = 'open'; $('file').click(); });
  $('uploadLayer').addEventListener('click', () => { uploadMode = 'layer'; $('file').click(); });
  $('file').addEventListener('change', async () => {
    const file = $('file').files[0];
    $('file').value = '';
    if (!file) { return; }
    if (uploadMode === 'open' && !confirmDiscard()) { return; }
    setBusy(1);
    try {
      const response = await request('/editor/api/' + sessionId + '/upload?mode=' + uploadMode +
        '&name=' + encodeURIComponent(file.name), { method: 'POST', body: file });
      view.fitted = view.fitted || uploadMode === 'open';
      await applyState(await response.json());
      setStatus(uploadMode === 'open' ? 'Opened ' + file.name + '. Use Save as to keep it as a project.' : 'Layer added.');
    } catch (error) { reportError(error); } finally { setBusy(-1); }
  });
  $('undo').addEventListener('click', () => act('undo'));
  $('redo').addEventListener('click', () => act('redo'));
  $('save').addEventListener('click', save);
  $('saveAs').addEventListener('click', saveAs);
  $('export').addEventListener('click', () => {
    let path = pathValue();
    if (!/\.png$/i.test(path)) { path = prompt('Export a PNG to (full path):', suggest('.png')); }
    if (path) { actConfirm('export', { path: path }); }
  });

  function suggest(suffix) {
    const base = state.project_path ? state.project_path.replace(/\.comp$/i, '') : '~/' + state.title;
    return base + suffix;
  }

  function save() {
    if (!state.open) { return; }
    if (!state.project_path) { saveAs(); return; }
    actConfirm('save', {}, 'Saved ' + state.project_path + '.');
  }

  function saveAs() {
    let path = pathValue();
    if (!/\.comp$/i.test(path)) { path = prompt('Save the project as (full path, .comp):', suggest('.comp')); }
    if (path) { actConfirm('save', { path: path }).then((data) => { if (data) { setStatus('Saved ' + data.project_path + '.'); } }); }
  }

  $('toolHide').addEventListener('click', () => setTool('hide'));
  $('toolReveal').addEventListener('click', () => setTool('reveal'));
  $('toolPan').addEventListener('click', () => setTool('pan'));
  $('zoomIn').addEventListener('click', () => zoomAtCentre(1.25));
  $('zoomOut').addEventListener('click', () => zoomAtCentre(0.8));
  $('fit').addEventListener('click', () => { fit(); draw(); });
  $('actual').addEventListener('click', () => zoomAtCentre(1 / view.zoom));
  function zoomAtCentre(factor) {
    const rect = stage.getBoundingClientRect();
    zoomAt(factor, rect.width / 2, rect.height / 2);
  }
  $('showMask').addEventListener('change', async () => { maskOverlay = null; await refreshMask(null); draw(); });
  for (const id of ['size', 'hardness', 'brushOpacity']) {
    $(id).addEventListener('input', showValues);
  }
  function showValues() {
    $('sizeV').textContent = $('size').value + ' px';
    $('hardnessV').textContent = $('hardness').value + '%';
    $('brushOpacityV').textContent = $('brushOpacity').value + '%';
    draw();
  }
  $('layerOpacity').addEventListener('input', () => { $('layerOpacityV').textContent = $('layerOpacity').value + '%'; });
  $('layerOpacity').addEventListener('change', () => {
    const layer = selectedLayer();
    if (layer) { act('update_layer', { layer_id: layer.id, opacity: Number($('layerOpacity').value) / 100 }); }
  });
  $('up').addEventListener('click', () => moveSelected(1));
  $('down').addEventListener('click', () => moveSelected(-1));
  function moveSelected(step) {
    const layer = selectedLayer();
    if (layer) { act('move_layer', { layer_id: layer.id, index: state.layers.indexOf(layer) + step }); }
  }
  $('rename').addEventListener('click', () => { const layer = selectedLayer(); if (layer) { renameLayer(layer); } });
  $('remove').addEventListener('click', () => {
    const layer = selectedLayer();
    if (layer && confirm('Delete the layer “' + layer.name + '”? Undo brings it back.')) {
      act('remove_layer', { layer_id: layer.id });
    }
  });
  $('maskAdd').addEventListener('click', () => withLayer((layer) => act('add_mask', { layer_id: layer.id, reveal: true })));
  $('maskHideAll').addEventListener('click', () => withLayer((layer) => act('add_mask', { layer_id: layer.id, reveal: false })));
  $('maskRemove').addEventListener('click', () => withLayer((layer) => act('remove_mask', { layer_id: layer.id })));
  $('maskToggle').addEventListener('click', () => withLayer((layer) => act('update_layer', { layer_id: layer.id, mask_enabled: !layer.mask_enabled })));
  function withLayer(fn) { const layer = selectedLayer(); if (layer) { fn(layer); } }

  document.addEventListener('keydown', (event) => {
    if (event.target instanceof HTMLInputElement && event.target.type === 'text') { return; }
    const mod = event.ctrlKey || event.metaKey;
    const key = event.key.toLowerCase();
    if (mod && key === 'z') { event.preventDefault(); act(event.shiftKey ? 'redo' : 'undo'); return; }
    if (mod && key === 'y') { event.preventDefault(); act('redo'); return; }
    if (mod && key === 's') { event.preventDefault(); if (event.shiftKey) { saveAs(); } else { save(); } return; }
    if (mod) { return; }
    if (event.key === ' ') { if (!spaceDown) { spaceDown = true; updateCursor(); draw(); } event.preventDefault(); return; }
    if (key === 'e') { setTool('hide'); }
    else if (key === 'b') { setTool('reveal'); }
    else if (key === 'h') { setTool('pan'); }
    else if (key === 'x') { setTool(tool === 'hide' ? 'reveal' : 'hide'); }
    else if (key === '[' || key === ']') {
      const size = Number($('size').value);
      $('size').value = Math.round(key === '[' ? size / 1.15 : size * 1.15 + 1);
      showValues();
    } else if (key === '0') { fit(); draw(); }
    else if (key === '1') { zoomAtCentre(1 / view.zoom); }
    else if (key === '+' || key === '=') { zoomAtCentre(1.25); }
    else if (key === '-') { zoomAtCentre(0.8); }
  });
  document.addEventListener('keyup', (event) => {
    if (event.key === ' ') { spaceDown = false; updateCursor(); draw(); }
  });

  window.addEventListener('beforeunload', (event) => {
    if (state.open && state.dirty) { event.preventDefault(); event.returnValue = ''; }
  });

  // ---- status ---------------------------------------------------------
  function setStatus(text, error) {
    $('status').textContent = text || '';
    $('status').className = error ? 'error' : '';
  }
  function reportError(error) { setStatus(error.message || String(error), true); }
  function setBusy(delta) {
    busy = Math.max(0, busy + delta);
    $('busy').style.display = busy ? 'block' : 'none';
  }
  function clamp(value, low, high) { return Math.min(high, Math.max(low, value)); }

  new ResizeObserver(resizeCanvas).observe(stage);
  showValues();
  setTool('hide');
  renderPanel();
  startSession().catch(reportError);
})();

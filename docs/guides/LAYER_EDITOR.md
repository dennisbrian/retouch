# Layer Editor

The **Layer Editor** tab in the app window stacks photos as layers and lets
you hide or reveal parts of each one with a brush. A typical use: open the
original photo, add Retouch's output for the same photo as a layer above it,
then brush away the retouch wherever it went too far. Projects are saved in
Compositor's `.comp` format, so they also open in the Compositor Mac app.

This is milestone 3 of the [Compositor port](../plans/COMPOSITOR_PYTHON_PORT.md):
an early editor. Retouch recipes do not run from inside it yet (milestone 4);
render the photo with the Single Photo Editor or `cli.py` first, then add the
result as a layer.

## Open, add, brush

1. Type the full path of a photo (JPEG, PNG, TIFF, WebP, BMP or RAW) or a
   `.comp` project in the box at the top and press **Open**. `~` works, for
   example `~/Pictures/DSCF3503.jpg`. **Upload photo…** picks a file instead.
2. To stack another photo, type its path and press **Add as layer**, or use
   **Upload layer…**. It goes above the selected layer, at the top-left corner
   of the canvas; use photos of the same size, such as an original and its
   retouched copy.
3. Pick the layer in the Layers panel, then paint on the canvas:
   - **Hide brush** (E) hides the layer where you paint, so the layer below
     shows. The first stroke adds a mask to the layer.
   - **Reveal brush** (B) brings hidden parts back.
   - **Size**, **Hardness** and **Opacity** work as in Photoshop. A stroke
     never goes past its opacity, even where it crosses itself; a soft brush
     builds up along the stroke.
   - **Show mask** tints hidden parts red.
4. Each stroke is one undo step. **Undo** (Ctrl/Cmd+Z) and **Redo**
   (Ctrl/Cmd+Shift+Z) cover strokes and every layer change.

Layers panel: tick a layer to show or hide it; double-click a name (or
**Rename**) to rename; **Up** / **Down** reorder; **Delete** removes the
layer (undo brings it back); the opacity slider sets the layer's opacity.
**Add mask** shows the whole layer and **Hide all** hides it, ready to
reveal with the brush; **Mask off** turns the mask off without losing it.

Moving around: scroll to pan, Ctrl/Cmd+scroll (or pinch) to zoom at the
pointer, **Hand** (H) or hold Space and drag, **Fit** (0), **100%** (1),
`+` / `-` to zoom, `[` / `]` to change the brush size, X to swap brushes.

## Save and export

- **Save as** writes a `.comp` project to the path in the box (or asks for
  one). **Save** (Ctrl/Cmd+S) then saves to the same project. A dot on the
  Save button means there are unsaved changes.
- **Export PNG** flattens every visible layer at full resolution into a PNG
  at the path in the box (or asks for one). Exporting does not count as
  saving the project.
- Neither replaces an existing file without asking, and Save never replaces
  a folder that is not a Compositor project.
- Opening another photo with unsaved changes asks first, and so does closing
  the desktop window. Reloading the page keeps the open document.

## What it does not do yet

- Only Normal blending is drawn. Projects made in Compositor with folders,
  adjustment layers, effects, clipping masks or rotated or scaled layers are
  refused with a message naming the feature; nothing is silently lost.
- No painting on layer pixels, healing, cloning, selections or transforms;
  the brush edits masks only.
- The canvas shows a preview at most 2048 px on its long side; strokes,
  saves and exports always use the full-size pixels. Zooming past that size
  enlarges the preview, so fine detail at 100% looks soft on screen even
  though the export is sharp.
- Saving a 26 MP two-layer project takes about 20 s (every layer is
  re-encoded as PNG), and exporting about 15 s.
- The editor only answers on this computer (`127.0.0.1` / `localhost`).
- Never checked against the Compositor Mac app's own rendering (needs a Mac).

Measured in the cloud sandbox on Alex's 26 MP DSCF3503 with its
`cosplay_clear_v1` render as a second layer (2026-10-08): open 3.7 s, add
layer 3.5 s, a 2,600 px soft stroke 0.1 s (250 px brush) to 0.3 s (1,000 px)
plus 0.25 s to redraw, undo 0.3 s. The exported PNG equals the render where
the mask is white and the original where it is black, and matches the
analytic blend within half a level in between.

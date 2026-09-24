# Paint X computer-use drawing experiment

Date: 2026-09-09

## Purpose and result

The user supplied a cosplay portrait and asked for a capability test: draw from the reference in Paint X using computer use. The drawing was made through native UI clicks, shape drags, palette selections, and keyboard commands. No image generation, imported tracing layer, or programmatically rendered artwork was used.

Saved artifact: [paintx-blue-cosplay-sketch.png](../paintx-blue-cosplay-sketch.png), verified as a 2759 × 1430 RGBA PNG. Paint X displayed the saved filename and path after saving.

The result is a basic cartoon with blue hair and buns, blue eyes, yellow/gold ornaments and tassels, a white costume, fastenings, and beads. It demonstrates successful UI operation and export. It does not demonstrate accurate portrait likeness or detailed illustration quality.

## What worked

- Open Paint X with `cua.getApp("Paint X")` and inspect its accessibility state and screenshot.
- Use accessibility controls for tool and palette selection; use screenshot-derived canvas coordinates for drags.
- Color1 controls shape outlines; Color2 controls shape fills. Enable Fill before drawing filled shapes.
- Build back to front: hair silhouette and buns, costume, face, bangs, eyes, facial details, ornaments, then accents.
- Use ovals for major masses and eyes, triangles for tassels, rounded rectangles for the collar, and lines for small accents.
- After each shape drag, press Escape and explicitly reselect the shape tool before the next shape. This sequence worked reliably in this session.
- Batch predictable actions, then inspect fresh accessibility state. Screenshots are essential because canvas geometry is not represented in the accessibility tree.
- Save with Command-S, enter a PNG filename, use Command-Shift-G to select the destination folder, and verify the resulting window title and path.

Accessibility indices and canvas coordinates are session-specific. Re-observe them on a new run; do not reuse this session's numbers as a fixed script.

## Problems observed and lessons

| Observation | Lesson |
| --- | --- |
| Early consecutive shape interactions left an unintended small blue dot and an absent intended bun. | Shape selection/commit state matters. Explicit Escape and tool reselection resolved the workflow; the precise cause of the initial behavior was not established. |
| Clicking Edit colors, including a coordinate click and double-click, did not expose a usable color dialog in the observed state. | Custom-color entry remains unverified. Do not claim it is unsupported; inspect available windows or controls before another attempt. |
| The built-in Salmon swatch produced an excessively pink face. | Palette approximation materially reduced fidelity. Establish a suitable skin color before constructing detailed facial features. |
| The upright, symmetrical cartoon lost the reference's head tilt, face proportions, and swept hair shape. | Plan pose and silhouette first. Those relationships matter more for likeness than adding decorative beads. |
| Large circles and triangles made the result look schematic. | Use more deliberate contours and smaller overlapping forms where useful; investigate curve-tool behavior before relying on it. |
| A stray mark was covered with a white filled shape. | This worked on the plain white canvas. Prefer immediate Undo when safe; white overpainting is unsuitable for a colored or textured background. |
| Paint X Lite showed a small watermark at the canvas corner. | Inspect and disclose application-added marks when delivering artwork. No watermark-removal purchase was made. |

## Improved workflow for the next attempt

1. Inspect the app and canvas, preserving any existing user drawing. Confirm the working canvas size and visible zoom.
2. Study the reference's head angle, facial centerline, eye line, hair silhouette, and shoulder direction. Translate those into a small set of screen-coordinate anchors.
3. Establish usable skin, hair, shadow, highlight, and gold colors. Resolve color selection early instead of accepting a visibly mismatched skin tone.
4. Block in the tilted silhouette and costume. Take a screenshot and compare proportions before adding details.
5. Construct the face and swept bangs, checking eye spacing and orientation against the reference.
6. Add the largest characteristic ornaments, then restrained highlights and costume details.
7. Inspect the full canvas for stray marks, overlaps, unsuitable colors, and application watermarks.
8. Save a separate PNG and verify the saved state. Describe the achieved fidelity honestly.

These improvements are a proposed workflow, not a second completed drawing or a tested custom-color/curve-tool procedure.

## Verification boundary

The original run visually inspected the final canvas and confirmed the saved document in Paint X. During documentation, the file's PNG format and dimensions were checked. No Retouch pipeline changes, automated image-quality measurements, or additional drawing edits were performed.

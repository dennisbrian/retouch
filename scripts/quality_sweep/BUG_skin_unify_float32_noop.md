# Bug: `skin_unify` is a silent no-op in the pipeline (float32 vs uint8 colour conversion)

**Severity:** medium — a documented, user-facing parameter has zero effect and fails silently.
**Found:** during Phase-2 parameter screening (`skin_unify` moved output by *exactly* 0.00000 at every value).
**Status:** root-caused and reproduced. **Not fixed** — audit/measurement task, no product code changed.

## Symptom

`engine.process(img, recipe=..., skin_unify=X)` produces byte-identical output for
every X (0.5, 0.9, 1.0, 2.0, 25.0, 60.0). The parameter appears fully wired: it shows
up in `safe_auto` `parameter_provenance.explicit_overrides`, reaches
`_process_face_core`'s `ctx` intact as `60.0`, and `SkinProcessor.unify_tone` **is
called once** with `strength=60` and a healthy mask (`mask.sum() ≈ 35_620`).
It just changes nothing.

## Root cause

`SkinProcessor.unify_tone` (`retouch/skin.py:1234`) is documented and implemented for
**uint8** BGR input:

```
img_bgr: (H, W, 3) uint8 BGR image.
```

The engine calls it from `retouch/perf_optimizations.py:658` with a **float32** canvas
in range [0, 255]. Per `_process_face_core`'s own docstring, "Stage E1: Canvas is
converted to float32 [0, 255] at the top and kept float throughout the skin operation
chain to eliminate quantization noise."

`bgr_to_lch` delegates to OpenCV, which interprets float input as **[0, 1]**, not
[0, 255]. With [0, 255] float32 the conversion saturates and returns garbage:

| input dtype | L range | C range | `skin_mask_lch` sum | `w_sum` |
|---|---|---|---|---|
| uint8 | 0.00 – 100.00 | 0.00 – 79.61 | 305 026.9 | 79 919.59 |
| float32 [0,255] | 0.00 – **39.22** | **181.02 – 226.29** | **0.0** | **0.000000** |

Chroma of 181–226 is far outside the real gamut, so `skin_mask_lch` classifies
**zero** pixels as skin. `unify_tone` then hits its own guard

```python
if w_sum < 1e-6:
    return img_bgr
```

and returns the input unchanged. The no-op is silent: no warning, no exception.

## Reproduction

```python
sp = RetouchEngine()._skin
sp.unify_tone(crop,                  mask, 60, target_hue=-1.0)  # uint8   -> meandiff 0.23659
sp.unify_tone(crop.astype(np.float32), mask, 60, target_hue=-1.0)  # float32 -> meandiff 0.00000
```

Identical mask, identical strength; only dtype differs.

## Blast radius (not yet fully verified)

`unify_tone` is one of several skin ops invoked on the float32 canvas in
`_process_face_core`. Any op in that chain that routes through `bgr_to_lch`/
`skin_mask_lch` (or otherwise assumes uint8) is suspect. In Phase-2 screening these
also showed *exactly* 0.00000 effect and are the natural next candidates to check:

- `skin_unify_hue`
- `glow`

The following were near-zero but not exactly zero, so they are probably scaled-down
rather than dead: `shine_removal` (0.018), `vibrance` (0.015), `blemish` (0.010),
`eye_enhance` (0.006), `lip_enhance` (0.005), `whiten` (0.002), `dark_circles` (0.001).

## Why the golden fixtures never caught it

`skin_unify` is 0 in all three snapshotted recipes, so the broken branch is never
entered. This is exactly the 2.9%-of-eligible-recipes coverage gap from the Phase-1
audit: the harness locks in current behaviour on 3 of 103 eligible recipes, and a
parameter that silently does nothing produces a perfectly stable hash.

## Suggested fix (not applied)

Make the dtype contract explicit at the boundary rather than at every call site —
normalize/convert inside `bgr_to_lch`, or have `unify_tone` reject non-uint8 input
loudly instead of silently returning. A cheap regression test: assert
`unify_tone(img, mask, 60)` changes the image for **both** dtypes.

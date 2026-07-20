# RESEARCH — Frontier AC-series: AB1 delivery fidelity reopened by `copy_exif`

**Date:** 2026-07-18 · **Author:** Fable research pass (read-only on production code)
**Owner ask:** continue the K/X/Y/Z/AA/BB research workstream.
**Method:** verified by reading + a real round-trip execution (not simulated —
see §1), at HEAD (`89404b4`).
**Rules inherited:** classical/deterministic, unit-testable, no learned models,
tone-fair, no absolute intensity thresholds, no identity-changing defaults.

**Fixed (2026-07-18, same-day):** AC1 is shipped in the working tree — EXIF
is now embedded in the original `write_image_with_icc` save (new `exif`
param + `read_exif_bytes`) at all three call sites instead of a second PIL
save; `copy_exif` is no longer called from any of them (retained as a
standalone, still-tested util). Verified against real rendered pixels
through the actual CLI binary, not just re-measured metrics — see
`TODO_WEEK_2026_07_20.md` Day 1 for the full verification record, including
two follow-up gaps closed in the same pass (`batch_processor.py` gained ICC
embedding it never had; the `cv2.imwrite`-fallback branch now warns instead
of silently dropping EXIF/ICC).

---

## 0. TL;DR

`89404b4` ("preserve delivery color fidelity") shipped AB1 (4:4:4 JPEG
sampling) and the K12 float-contract groundwork, verified in
`RESEARCH_FRONTIER_AB_2026_07_17.md`. This pass found that a sibling function
in the same file — `copy_exif` — silently undoes AB1 (and the ICC embed) at
every CLI/batch call site whenever EXIF copy is requested, which is the
default in normal usage. This is not a new algorithm axis; it is a delivery
regression in already-shipped work, found by exercising the actual code path
rather than reading it in isolation. Filed as **AC1**.

---

## 1. AC1 — `copy_exif` re-encodes the delivered JPEG at PIL defaults, destroying AB1 + ICC

### The bug

`copy_exif` (`io.py:509`) does:

```python
src_img = Image.open(src_path)
exif = src_img.getexif()
...
dst_img = Image.open(dst_path)
dst_img.save(dst_path, exif=exif.tobytes())
```

`dst_img.save()` is called with **no `quality`, no `subsampling`, no
`icc_profile`** kwarg. For JPEG, PIL's save defaults are quality **75** and
4:2:0 subsampling — regardless of what the file was encoded at moments
earlier. The call re-decodes and re-encodes the delivered JPEG a second time,
at those defaults, and drops the ICC profile the exporter just embedded
(`ImageCms`/`icc_profile` is never passed through).

### Verified by execution (not simulated)

Ran the actual export → `copy_exif` sequence end to end on a synthetic
gradient+noise image (512×512, texture chosen to be recompression-sensitive),
through `write_image_with_icc(..., bit_depth=8, quality=95)` followed by
`copy_exif`:

| Metric | After export (pre-`copy_exif`) | After `copy_exif` |
|---|---|---|
| File size | 197,714 B | 23,870 B (12%) |
| PSNR vs. source pixels | 36.03 dB | 32.62 dB |
| ICC profile embedded | Yes | **No** |
| Chroma subsampling (PIL `get_sampling`, 0=4:4:4) | **0** | **2** (4:2:0) |
| EXIF present | — | Yes (Make/Copyright/Orientation all correctly transferred) |

EXIF itself transfers correctly — that part of the function works. The
destructive side effect is the unguarded re-save. Confirmed separately that
**PNG output is unaffected**: `Image.open`/`.save()` infers PNG from the
extension both times and PNG's save path has no quality/subsampling knob to
reset (byte-identical round trip on a synthetic PNG). The corruption is
JPEG-format-specific — exactly the format AB1 targeted, which is why this
reopens that work rather than being an unrelated finding.

### Where this fires

Three call sites, all unconditional once EXIF-copy is requested (the default
in normal usage — `--no-exif` / `no_exif` must be explicitly passed to avoid
it):

| Site | Guard before calling | Format scope |
|---|---|---|
| `cli.py:284-285` (single-image path) | `copy_exif_flag and bit_depth == 8` | JPEG only affected (guard already excludes 16-bit, which routes through OpenCV, not PIL) |
| `cli.py:849-850` (second CLI code path — batch-via-args) | `not args.no_exif and args.bit_depth == 8` | same |
| `retouch/batch_processor.py:493-498` (GUI-driven batch) | **none** — always called after `cv2.imwrite` | Fires for **every** export format, not just JPEG; harmless on PNG/TIFF (verified above), destructive on JPEG |

All three write via `encode_write_params` (`io.py:483-502`, extended in
`89404b4`) or `write_image_with_icc`, both of which correctly apply AB1's
4:4:4 sampling on the **first** write — re-verified this pass, `fmt in
("jpg", "jpeg")` unconditionally appends
`IMWRITE_JPEG_SAMPLING_FACTOR_444` when the OpenCV build exposes it. So the
initial write is correct at all three sites; `copy_exif`'s second write is
the single common regression, not three separate bugs. `batch_processor.py`'s
only *additional* gap is that it never had an ICC profile to lose in the
first place (raw `cv2.imwrite` can't embed one) — not a second sampling gap.

**GUI direct-export path (`gui.py:559-589`) does not call `copy_exif` at
all** and is unaffected.

### Why this matters more than a normal quality regression

This is the exact defect class `RESEARCH_FRONTIER_AB_2026_07_17.md` §2 (K12)
and §3 (AB1) were written to close — chroma-subsampled, ICC-less delivery
JPEGs — reopened one commit after being closed, on the CLI/batch paths that
are the primary "ship to a client" surface (the GUI export button is
unaffected, but CLI batch runs are the higher-volume path). A user who ran
`89404b4`'s own regression tests (`tests/test_io_icc.py`, which exercise
`write_image_with_icc` and `encode_write_params` directly) would see AB1
passing, because those tests don't call `copy_exif` — the export function is
correct in isolation; the corruption is downstream of it.

### Fix spec (implementation-grade, not yet applied — read-only pass)

1. **Stop double-encoding.** `copy_exif` should not decode-then-resave the
   pixels at all — it only needs to write EXIF bytes into an existing file.
   Two options, in order of preference:
   - **Piggyback EXIF onto the original save call** at each of the three
     call sites (pass `exif=...` to `write_image_with_icc`/`cv2.imwrite`'s
     caller in the same save that already has the correct quality/sampling/
     ICC in scope) — zero extra encode, no regression possible by
     construction. Requires reading the source EXIF *before* the write, then
     passing it through, which changes `copy_exif`'s signature/call sites.
   - **If a standalone post-hoc function is kept:** re-encode with the same
     quality and 4:4:4 sampling the original write used (thread `quality`
     and re-apply `subsampling=0` / `encode_write_params` in the resave),
     and re-embed the ICC profile it already has access to via
     `read_icc_profile` at the call sites. This keeps `copy_exif`'s current
     shape but requires it to accept `quality` and `icc_profile` params
     instead of defaulting silently.
2. **Guard `batch_processor.py:497-498`** the same way the two `cli.py` sites
   already are (`bit_depth == 8` / format check) — even after fix (1), a
   format check avoids an unnecessary decode/re-encode cycle on PNG/TIFF.
3. **Acceptance:** extend `tests/test_io_icc.py` (or a new
   `tests/test_copy_exif.py`) with the exact round-trip this pass ran:
   export → `copy_exif` → assert **file size does not shrink by more than a
   quality-95→95 re-encode tolerance**, **`get_sampling` stays 0**, **ICC
   profile is still present**, **EXIF fields still transfer**. This is a
   direct regression test on the measured deltas above, not a new metric.

**Effort:** ~0.5–1 day (small function, three call sites, one shared test
pattern). **Risk:** low — the fix is strictly corrective; no output that is
today "acceptable" gets worse.

---

## 2. Recommended sequencing

This slots as a **direct continuation of the K12/AB1 delivery-fidelity PR**,
not a new backlog item — same subsystem, same commit's intent, closes the gap
the commit's own test suite didn't cover because the test suite (correctly)
tests the exporter, not the EXIF-copy step that runs after it. Recommend
landing AC1 before any of AB2–AB4/BB1/BB2 so the delivery-boundary slice is
actually closed end-to-end before new axes build on top of it.

## Bounds / NO-GO

- AC1 is metadata/IO plumbing only — no pixel-pipeline change, no new
  algorithm, no change to `write_image_with_icc`'s public contract beyond
  what's needed to pass quality/ICC through to the EXIF step.
- Fix must not regress the 16-bit path (already excluded from `copy_exif` by
  the existing `bit_depth == 8` guards at both `cli.py` sites — preserve
  that, and add the missing equivalent guard/format-check at
  `batch_processor.py`).

## Sources

- Pillow `JpegImagePlugin` save defaults (quality 75, `subsampling=-1`→auto
  4:2:0-equivalent when unspecified) — same family of default-behavior gap
  documented for the encode side in `RESEARCH_FRONTIER_AB_2026_07_17.md`
  AB1's Sources section (OpenCV #22052/#22064, Pillow `JpegPresets`).
- Verification is this pass's own executed round-trip (§1 table), not a
  secondary source — flagged per repo convention (`CLAUDE.md` Verification &
  Honesty) as an actually-run test, not narrated.

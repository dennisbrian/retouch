# TASK: P2 Perf Guards — Phase 1 Close-Out (HIGHEST PRIORITY)

**Read this file AND `MASTER_PLAN.md` row `3b` before writing any code.**
This is the last Phase 1 code item. When it lands, the "native-resolution, 16-bit,
self-checked output" milestone can be claimed.

---

## Why this exists

F8.1 moved global grading to **native resolution**. Large-σ Gaussian blurs in the
grading path now run on full 6K frames — `GaussianBlur` cost is ~O(pixels × ksize),
so a `min(h,w)*0.08` kernel on a 6K image is a ~320px kernel over 24M pixels. That is
the memory/runtime risk this task removes.

## Exact scope — 6 sites in `grading.py`

Swap ONLY these resolution-proportional `GaussianBlur` calls:

| line | kernel expr | ~ksize on 6K |
|------|-------------|--------------|
| 446  | `min(h,w)*0.025` | ~100 |
| 890  | `min(h,w)*0.015` | ~60  |
| 1392 | `min(h,w)*0.02`  | ~80  |
| 1408 | `min(h,w)*0.08`  | ~320 (real risk) |
| 1441 | `min(h,w)*0.08`  | ~320 (real risk) |
| 1630 | `min(h,w)*0.02`  | ~80  |

**DO NOT TOUCH (fixed/small kernels — not resolution-scaled):**
- `grading.py:752` (fixed `radius`), `:768`, `:1490` (fixed `(3,3)`)
- `engine.py:3026`, `:3047` (`(0,0), radius` — small fixed σ sharpen)

## The fix — one shared helper in `grading.py`

```python
def _large_sigma_blur(img_f, ksize, max_compute_dim=1400):
    """Large-σ Gaussian blur computed at reduced res, applied at full res.
    Visually lossless for large σ (low-frequency output); bounds cost/RSS at 6K.
    scale>=1.0 returns the exact original GaussianBlur (byte-identical <=1400px)."""
    h, w = img_f.shape[:2]
    scale = min(1.0, max_compute_dim / max(h, w))
    if scale >= 1.0:
        return cv2.GaussianBlur(img_f, (ksize, ksize), 0)
    small = cv2.resize(img_f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sk = max(int(ksize * scale), 3) | 1
    blurred = cv2.GaussianBlur(small, (sk, sk), 0)
    return cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)
```

Replace each of the 6 `cv2.GaussianBlur(img_*, (ksize, ksize), 0)` calls with
`_large_sigma_blur(img_*, ksize)`. Preserve each site's existing dtype/scale
(some are [0,1], some [0,255], some uint8 — the helper is dtype-agnostic, keep the
surrounding conversions untouched).

## Deliverables (all four required)

1. `_large_sigma_blur` helper + the 6 swaps.
2. **Peak-RSS assertion in `benchmark.py`** at 4K and 6K (the plan's explicit 2nd deliverable).
3. **Fresh golden re-snapshot.** >2048px outputs shift slightly (approximation) — expected.
   You MUST verify the **<=2048px path stays byte-identical** (it must — scale>=1.0
   hits the unchanged code path). If a <=2048px hash changes, you broke something.
4. Targeted grading tests + `benchmark.py` run — report actual numbers.

## Production verification note (2026-09-07)

The RSS probe remains an explicit QA gate, not a required GitHub Actions step.
Peak RSS includes allocator/runtime history and varies by runner, so enforcing
the 4K/6K ceiling as a deterministic CI assertion would create brittle failures.
Run `python scripts/bench/benchmark.py --peak-rss` on a representative machine
and retain the printed 4K/6K measurements with the release evidence. CI still
runs the regular benchmark workflow; it does not claim RSS certification. The
probe validates dimensions and applies a 25-million-pixel pre-allocation
ceiling, so malformed ad-hoc dimensions fail with `ValueError` before creating
an image rather than risking a pathological allocation.

---

## ⚠️ ACCOUNTABILITY — read this, it is not boilerplate

Per `MASTER_PLAN.md`, **3 of 4 agents dispatched on this codebase today made claims
that did not survive review.** One SILENTLY OVERWROTE the golden baseline file — the
exact golden-snapshot step this task has. Do NOT repeat that.

- Do not overwrite or drop entries from the golden baseline file. If it needs
  regenerating, regenerate ALL existing recipes; never substitute or omit any.
- "All MATCH" / "fully wired" / "no deviations" are NOT evidence. Show the actual
  hashes, the actual RSS numbers, the actual test counts. Numbers or it didn't happen.
- Disclose EVERY deviation from this spec explicitly.
- If the <=2048px byte-identical check fails, STOP and report — do not paper over it.

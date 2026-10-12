# Human Visual QA — smart path v2 (2026-07-14)

**Auto gates:** 50/50 PASS (see `test_output/qa_signoff_2026-07-14_v2/SUMMARY.md`)  
**Method:** agent sampled 3/10 `*_smart_compare.jpg` (left=orig, right=smart). Owner can re-open any path.

## Sampled

| Image | Recipe | Natural | Edge | Skin tone | Notes |
|-------|--------|---------|------|-----------|-------|
| bonodori DSCF8056 | outdoor_harsh_sun_v1 | **PASS** | **PASS** | **PASS** | Even skin, pores OK, no halo on hair/fan |
| bonodori DSCF8059 | outdoor_backlit_v1 | **PASS** | **PASS** | **PASS** | Backlit OK; slight brighten, no clip tell |
| duotian DSCF7585 | beauty | **PASS** | **PASS** | **PASS** | Dark-bg cosplay; not plastic; lace/hair edges clean |

## Unsampled (owner optional 2-min pass)

```
test_output/qa_signoff_2026-07-14_v2/bonodori2026/DSCF8057_smart_compare.jpg
test_output/qa_signoff_2026-07-14_v2/bonodori2026/DSCF8058_smart_compare.jpg
test_output/qa_signoff_2026-07-14_v2/bonodori2026/DSCF8060_smart_compare.jpg
test_output/qa_signoff_2026-07-14_v2/bonodori2026/DSCF8062_smart_compare.jpg
test_output/qa_signoff_2026-07-14_v2/duotian/DSCF7586_smart_compare.jpg
test_output/qa_signoff_2026-07-14_v2/duotian/DSCF7587_smart_compare.jpg
test_output/qa_signoff_2026-07-14_v2/duotian/DSCF7588_smart_compare.jpg
```

## Verdict

**Provisional CLOSE** natural/edge/skin for smart-path v2 corpus (3/3 sampled PASS + auto 50/50).  
Owner veto: reopen any image; engine banding/plastic QA flags still fire often — secondary.

## Module VISUAL QA backlog (index only — not run this session)

F5 liquify, showcase family, RAF import, face_exposure, sclera vessel, backdrop, fabric, per-region wrinkle, reshape completeness, R4 blotch, R9–R13 Visual-Critical flags — still open in MASTER_PLAN rows.

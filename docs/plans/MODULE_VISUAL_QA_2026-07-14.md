# Module Visual QA — human sample (2026-07-14)

**Harness:** `scripts/visual_qa_modules.py` → `test_output/visual_qa_modules/`  
**Source:** `test_output/masterwork_v1/DSCF8007.jpg` maxd=640 (left=base natural, right=feature ON)

## Sampled (3)

| Module | MAE | Natural | Edge | Notes |
|--------|----:|---------|------|-------|
| face_exposure | 0.54 | **PASS** | **PASS** | Subtle face lift; no halo on hair/hands |
| f5_liquify | 0.19 | **PASS** | **PASS** | Mild eye/jaw move; no tear on face oval |
| backdrop | 1.10 | **PASS** | **PASS** | BG cleanup mild at 640; subject intact |

## Unsampled — open montages in OS

```
test_output/visual_qa_modules/blotch_r4_montage.jpg   # MAE 0.03 weak
test_output/visual_qa_modules/sclera_montage.jpg      # MAE ~0 weak on this crop
test_output/visual_qa_modules/fabric_montage.jpg      # MAE 0.44
test_output/visual_qa_modules/wrinkle_montage.jpg     # MAE 0 weak
test_output/visual_qa_modules/makeup_p4_montage.jpg   # MAE 0.11
```

**Caveat:** single half-body cosplay crop; sclera/wrinkle/blotch need face-close / defect corpus for strong signal. Re-run with `qa_signoff` outdoor faces for F5/exposure stress.

## Verdict
Sampled 3/3 natural+edge **PASS**. Module backlog **partially closed** for face_exposure / F5 / backdrop on this corpus; rest PENDING human OS review.

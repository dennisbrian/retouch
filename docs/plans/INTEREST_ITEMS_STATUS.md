# Interest items — status (2026-07-14)

## 1. Module human visual QA
| Status | Detail |
|--------|--------|
| Harness | `scripts/visual_qa_modules.py` → MAE + montages |
| Sampled human | face_exposure / f5_liquify / backdrop → **PASS** natural+edge |
| Report | `docs/plans/MODULE_VISUAL_QA_2026-07-14.md` |
| Left | OS-review blotch/sclera/fabric/wrinkle/makeup_p4; stronger corpus if MAE~0 |

## 2. F6 / T4 rows (MASTER was stale)
| | Was | Now |
|--|-----|-----|
| F6 | SHIPPED-UNWIRED | ✅ GUI Extract Look + CLI `--extract-look` |
| T4 plugins | SHIPPED-UNWIRED | ✅ discover/init in engine |
| T4 cookbook | search only | ✅ category browse + search (GUI thickened) |
| MASTER | updated | RESUME + table rows fixed |

## 3. T5 full linear develop
| Layer | Status |
|-------|--------|
| 16-bit sRGB ingest | ✅ `imread_engine` GUI+CLI |
| `load_raw` gamma bug | ✅ fixed `gamma=(1,1)` |
| LinearGrader island | ✅ CLI `--linear-raw [--raw-exposure] [--raw-contrast]` |
| Develop UX (WB, exp_shift UI) | 📋 backlog Step 3 |

```bash
python3 cli.py photo.RAF -o out/ --linear-raw --raw-exposure 0.5 --recipe natural
```

## 4. Cookbook GUI
- Category dropdown + search + select → base recipe
- Not: full card grid / ratings / community — out of scope

## 5. Auto stray-hair #5
| Status | 🅿️ PARKED (low priority) |
|--------|-----------|
| Why | A4-gated; needs model; A1 competitor track deferred (licensing fees) |
| Action | Skip unless owner later funds A1/A4 or accepts classical-only gap |

## 6. Per-face auto Male/Female/Child
| Status | 📋 Slice 3 backlog |
|--------|---------------------|
| v1 | Manual face index + recipe (picker shipped) |
| Slice 3 | Heuristic size/ITA/landmarks → suggested `face_params` (no neural) |
| Not | Evoto-class auto video classification |

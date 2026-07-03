# Retouch Engine Pipeline Architecture / 引擎管线架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      RETOUCH ENGINE PIPELINE                     │
│                        引擎管线架构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  DSCF6102.jpg                                                    │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────┐    ┌──────────┐    ┌────────────┐                │
│  │  imread  │───▶│ Resize   │───▶│ Face Detect │  (MediaPipe)  │
│  │  + EXIF  │    │ (max_dim)│    │ (landmarks) │                │
│  │  读取    │    │  缩放    │    │  人脸检测   │                │
│  └──────────┘    └──────────┘    └──────┬───────┘               │
│                                         │                       │
│                    ┌────────────────────┼─────┐                 │
│                    ▼                    ▼     ▼                 │
│             ┌──────────┐       ┌────────────────────┐           │
│             │ Crop ROI │       │ Region Parsing     │  BiSeNet  │
│             │  裁剪ROI │       │ (skin/eyes/hair/   │           │
│             │   per face│      │  lips/neck/body)   │           │
│             │  每张脸  │       │  区域解析          │           │
│             └────┬─────┘       └─────────┬──────────┘           │
│                  │                       │                      │
│                  ▼                       ▼                      │
│         ┌──────────────────────────────────────────┐            │
│         │       PER-FACE PROCESSING (parallel)     │ ✅ FIXED   │
│         │       单脸处理 (并行多进程)               │ 修复       │
│         │  ┌──────────────────────────────────┐    │            │
│         │  │ Frequency Separation (频率分离)  │    │            │
│         │  │  ├─ Low freq 低频 (smooth/色调) │    │            │
│         │  │  └─ High freq 高频 (texture/孔) │    │            │
│         │  │                                    │    │            │
│         │  │ Skin ops:     Eyes/Teeth/Lips:    │    │            │
│         │  │  皮肤操作      眼睛/牙齿/嘴唇     │    │            │
│         │  │  ├ smooth平滑  ├ eye_enhance 眼睛 │    │            │
│         │  │  ├ flatten磨平 ├ catchlight 眼神光│    │            │
│         │  │  ├ whiten美白  ├ teeth_whiten 牙  │    │            │
│         │  │  ├ hue_unify   ├ lip_enhance 唇   │    │            │
│         │  │  │  肤色统一   ├ lip_tint 唇色    │    │            │
│         │  │  ├ equalize    │                  │    │            │
│         │  │  │  色调均衡   │ Hair/Body:       │    │            │
│         │  │  ├ shine_rem   │ ├ hair_enhance   │    │            │
│         │  │  │  去油光     │ │  头发增强      │    │            │
│         │  │  ├ micro_db    │ └ body_fix       │    │            │
│         │  │  │  微立体     │   身体修正       │    │            │
│         │  │  ├ redness_even│                  │    │            │
│         │  │  │  去红       │                  │    │            │
│         │  │  └ blemish_rm │                  │    │            │
│         │  │    去瑕疵      │                  │    │            │
│         │  │                                    │    │            │
│         │  │ ✅ 修复脸部蓝色偏色                │    │            │
│         │  │    blue face cast fix              │    │            │
│         │  └──────────────────────────────────┘    │            │
│         └──────────────┬───────────────────────────┘            │
│                        ▼                                        │
│         ┌──────────────────────────────┐                        │
│         │  Compose faces → full image  │                        │
│         │  合成人脸 → 全图             │                        │
│         └──────────────┬───────────────┘                        │
│                        ▼                                        │
│         ┌──────────────────────────────┐                        │
│         │  GLOBAL FINISH 全局后期      │                        │
│         │  ├─ Grading 调色(curves/LUT)│                        │
│         │  ├─ Bloom柔光 / Grain颗粒    │                        │
│         │  ├─ Vignette暗角/Sharpen锐化 │                        │
│         │  └─ ICC profile 色彩写入     │                        │
│         └──────────────┬───────────────┘                        │
│                        ▼                                        │
│              output_recipe.jpg                                  │
│              (× 50 recipes 配方 = sweep)                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Recent Improvements / 近期改进

| Area 模块 | What 内容 | Impact 影响 |
|-----------|-----------|-------------|
| ✅ **Skin 皮肤** | 修复脸部蓝色/青色偏色 (`color_science.py`, `skin.py`) — LAB 美白/色相路径修正 | 肤色自然不偏蓝 |
| ✅ **Perf 性能** | 修复多进程 ctx 序列化崩溃 (`perf_optimizations.py` — `SimpleNamespace` 适配器) | 多脸并行不再 AttributeError |
| ✅ **Engine 引擎** | 身体皮肤支持、relight v2、S4 光泽、频域管线升级 | 更完整的全身处理 |
| ✅ **Tests 测试** | 新增 `test_e1_float_parity`, `test_smooth_exposure_lock` | float32 一致性保障 |
| ✅ **Recipes 配方** | 50 个预设全部随引擎更新同步 | 全配方质量提升 |

---

*Generated 2026-07-04 / Pipeline architecture overview for developers*

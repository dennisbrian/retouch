# Project Structure

## Root Level (Entry Points)
```
├── cli.py              # Command-line interface
├── gui.py              # Gradio web interface
├── desktop.py          # Desktop app entry
├── README.md           # Installation & quick start
├── CLAUDE.md           # Development guidelines
└── test_visual_qa.py   # Visual QA testing
```

## Core Engine (`retouch/`)
```
retouch/
├── engine.py           # Main processing pipeline
├── params.py           # Parameter registry (single source of truth)
├── recipes.py          # Recipe loading & management
├── detection.py        # Face detection
├── segmentation.py     # Semantic segmentation
├── skin.py             # Skin processing
├── color.py            # Color grading
└── [~70 modules total]
```

## Documentation (`docs/`)
```
docs/
├── INDEX.md                    # Navigation guide
├── FUJI_SIMS_GUIDE.md          # Fuji film simulation recipes
├── guides/                     # User guides
│   ├── RECIPE_GUIDE.md
│   ├── GUI.md
│   ├── BATCH_GUIDE.md
│   └── TROUBLESHOOTING.md
├── architecture/               # Technical design
│   ├── ARCHITECTURE.md
│   ├── API.md
│   └── PIPELINE_FLOW.md
├── plans/                      # Roadmaps & initiatives
│   ├── MASTER_PLAN.md
│   └── PLAN_*.md
├── reference/                  # Implementation notes
├── review/                     # Audits & QA
├── improvements/               # Enhancement proposals
└── assets/                     # Images & diagrams
```

## Requirements (`requirements/`)
```
requirements/
├── base.txt          # Core engine dependencies
├── gui.txt           # Gradio web UI
├── dev.txt           # Testing & linting
├── raw.txt           # RAW image support
└── README.md         # Installation guide
```

## Scripts (`scripts/`)
```
scripts/
├── build/            # Build & packaging
│   ├── build_app.sh
│   └── *.spec
├── dev/              # Development utilities
│   ├── benchmark.py
│   └── benchmark_results.json
├── qa_*/             # QA & testing scripts
├── batch/            # Batch processing
├── bench/            # Benchmarking
└── README.md
```

## Data Folders
```
├── models/           # ONNX & ML models
├── presets/          # Recipe presets
├── luts/             # Color lookup tables
├── meta/             # Metadata & config
├── prompts/          # AI generation prompts
├── styles/           # Style definitions
├── tests/            # Test suite (3,736 tests collected 2026-07-15)
└── test_output/      # Test artifacts
```

## Quality Assurance
- **Test Coverage:** 89% on core modules (last measured 2026-07-01)
- **Test Count:** 3,736 collected (`pytest --collect-only`, 2026-07-15)
- **Performance:** 702ms per-face (400×400)
- **CI/CD:** Weekly syntax checks, monthly audits

See `docs/INDEX.md` for complete navigation.

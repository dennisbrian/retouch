# Scripts & Build Tools

## Build Scripts (`build/`)
- `build_app.sh` — Build executable bundle
- `*.spec` — PyInstaller configuration files

## Development Scripts (`dev/`)
- `benchmark.py` — Performance benchmarking
- `benchmark_results.json` — Benchmark results
- `test` — Hermetic pytest wrapper; disables user-site packages and redirects caches to a writable temporary root

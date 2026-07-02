# Contributing to retouch

Welcome! This guide walks new contributors through setup, workflow, and best practices.

---

## Setup (5 min)

### 1. Clone & Virtual Environment
```bash
git clone https://github.com/USERNAME/REPO.git
cd retouch
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements-gui.txt  # GUI + all extras
pip install pytest pytest-cov        # Testing
```

### 3. Download Model Assets
Models are large (~800 MB). They auto-download on first run:
```bash
python3 -c "from retouch import RetouchEngine; RetouchEngine()"
```

Or manually:
```bash
mkdir -p models
# Download from S3 or local mirror (see ARCHITECTURE.md for URLs)
```

### 4. Verify Setup
```bash
# Syntax check
for f in retouch/*.py gui.py cli.py; do python3 -m py_compile "$f"; done

# Run tests
python3 -m pytest tests/ -q
```

---

## Git Workflow

### Branch Naming
```
feature/some-feature       # New feature
fix/issue-name            # Bug fix
docs/section              # Documentation
refactor/module-name      # Code cleanup
test/test-name            # Test additions
perf/optimization-name    # Performance work
```

### Commit Message Format
```
<type>(<scope>): <subject>

<body (optional)>

<footer (optional)>
```

**Types:** `feat`, `fix`, `docs`, `refactor`, `test`, `perf`, `chore`  
**Scopes:** `engine`, `grading`, `gui`, `cli`, `parsing`, `recipes`, etc.

**Example:**
```
fix(engine): replace white_balance hardcoded literals with _DEFAULTS

Use _DEFAULTS["white_balance_kelvin"] instead of hardcoded 6500
to maintain single-source-of-truth with params.py registry.

Fixes #142
```

### Before Pushing

1. **Syntax check (all modules):**
   ```bash
   for f in retouch/*.py gui.py cli.py; do python3 -m py_compile "$f"; done
   ```

2. **Run full test suite:**
   ```bash
   python3 -m pytest tests/ -q
   ```

3. **Add tests for new features:**
   - Tests live in `tests/test_<module>.py`
   - Use `@pytest.mark.parametrize` for multiple scenarios
   - Mock external models with fixtures
   - Aim for 85%+ coverage on new code

4. **Check coverage (optional but recommended):**
   ```bash
   python3 -m pytest tests/ --cov=retouch --cov-report=term
   ```

5. **Push to your branch:**
   ```bash
   git push origin <your-branch>
   ```

6. **Create PR** with descriptive title and summary

---

## Code Style Quick Reference

### Naming Conventions
- `ProcessingContext` = typed parameter bag (replaces raw dicts)
- `ProcessingResult` = ndarray subclass with metadata
- `*Processor`, `*Enhancer`, `*Analyzer` = stage/component classes
- `_stage_*` = pipeline stage methods
- `_process_*`, `_apply_*` = internal helpers
- `_DEFAULTS` = immutable constants

### Structure
- No circular imports (broken via `style_transfer.py` re-export)
- All public functions/classes have docstrings
- Private methods prefixed with `_`
- No bare `except: pass` (log via `_logger` or `log_crash()`)

### Performance Checklist
- Avoid `**kwargs` in hot loops → use dataclass fields
- Pre-allocate large arrays → no dynamic growth
- Vectorize with NumPy → no Python loops over pixels
- Cache pre-computed LUTs/matrices at class level
- Use `cv2.LUT` for 1D LUT application
- ThreadPoolExecutor (max 4 workers) for multi-face

### Comments
- Default: **no comments** (well-named code is self-documenting)
- Add a comment ONLY when the "WHY" is non-obvious
  - Hidden constraints (e.g., "must match Fuji JPEG exactly")
  - Subtle invariants (e.g., "alpha channel is always premultiplied")
  - Workarounds for specific bugs
- Don't document WHAT the code does — that's what names are for

---

## Testing Checklist

### Before Submitting a PR

| Step | Command | What It Checks |
|------|---------|---|
| **Syntax** | `for f in retouch/*.py gui.py cli.py; do python3 -m py_compile "$f"; done` | Python parse errors |
| **Unit tests** | `pytest tests/ -q` | All 1,626 tests pass |
| **Coverage** | `pytest tests/ --cov=retouch --cov-report=term` | ≥85% on new code |
| **Integration** | `python3 -c "from retouch import RetouchEngine; engine = RetouchEngine(); print('OK')"` | Import + basic flow works |
| **GUI** | `python3 gui.py` | Opens http://127.0.0.1:7860 without error |
| **CLI** | `python3 cli.py --help` | Prints help, no import errors |

### Writing Tests

**Structure:**
```python
import pytest
from retouch.module import function

def test_basic_behavior():
    """One-line description of what you're testing."""
    result = function(input)
    assert result == expected

@pytest.mark.parametrize("input,expected", [
    ("case1", "output1"),
    ("case2", "output2"),
])
def test_multiple_cases(input, expected):
    """Parameterized test for multiple scenarios."""
    result = function(input)
    assert result == expected
```

**Best Practices:**
- One assertion per test when possible (easier to debug)
- Use descriptive test names: `test_<function>_<scenario>`
- Mock external models with fixtures (don't download at runtime)
- Use `pytest.mark.skipif` for platform-specific tests
- Verify algorithm correctness, not just syntax

---

## Common Gotchas

### ⚠️ Circular Imports
The codebase has one deliberate circular import pattern:
- `style_transfer.py` re-exports leaf modules to break cycles
- Don't add new circular imports — refactor instead

**How to check:**
```bash
python3 -c "import retouch; print('OK')"
```

### ⚠️ Dead Recipe Keys
If you add a new recipe, verify all keys map to registered parameters:
```bash
python3 -m pytest tests/test_recipe_validation.py -v
```

This catches silent no-op keys (like `anime_crystal_void`'s 7 dead keys).

### ⚠️ Model Dependency
Some tests require downloaded models (marked with `pytest.mark.skipif`). They auto-skip in CI. To run them locally, download models first:
```bash
python3 -c "from retouch import RetouchEngine; RetouchEngine()"
```

### ⚠️ Parameter Registry
All tunable parameters must be registered in `retouch/params.py` under `PROCESSING_PARAMS`. This maintains single-source-of-truth:
- GUI sliders auto-scale from registry
- CLI flags auto-wire from registry
- Engine defaults auto-pull from registry
- Recipes resolve against registry

**Never hardcode defaults in multiple places** — that's the white_balance_kelvin bug.

### ⚠️ Performance Tests
Benchmarks are optional in CI but useful locally:
```bash
python3 scripts/benchmark.py
```

If you touch detection, parsing, or grading, run this to catch regressions.

---

## Budget-Conscious Development

This project runs on a **$20/month Claude budget**. To stay sustainable:

- ✅ Use **native tools** (Read, Edit, Bash) over agents
- ✅ Use **Haiku model** for routine work (testing, documentation)
- ✅ Keep **thinking OFF** by default (prompt only when needed)
- ✅ Avoid **agent spawns for file reads** (use grep/find instead)

[Read CLAUDE.md § Budget & Sustainability for more]

---

## Getting Help

### Documentation Map
| File | Purpose |
|------|---------|
| `README.md` | Quick start & examples |
| `ARCHITECTURE.md` | Deep-dive pipeline & modules |
| `CLAUDE.md` | Contributor playbook |
| `API.md` | Python API reference |
| `PERFORMANCE_TUNING.md` | GUI/CLI speed/memory trade-offs |
| `AUDIT_REPORT.md` | Security & test coverage audit |

### Common Questions

**Q: How do I add a new Fuji film simulation?**  
A: See `FUJI_SIMS_GUIDE.md`. Copy an existing recipe, adjust LUT paths + grade values, test against reference Fuji JPEG.

**Q: How do I optimize something slow?**  
A: Profile with `scripts/benchmark.py` first. Common bottlenecks: detection (switch to faster fallback), frequency separation (vectorize loops), grading (cache LUTs).

**Q: Can I modify the 7-stage pipeline?**  
A: Probably not without breaking existing recipes. The pipeline is locked for v1. New stages planned for v2 (face editing).

**Q: What's the deal with `fast=True` vs `fast=False`?**  
A: See `PERFORMANCE_TUNING.md`. `fast=True` downscales to 800px for interactive GUI work (3-5× faster, softer details). `fast=False` uses full resolution for final exports.

---

## Pull Request Template

```markdown
## Summary
[1-3 sentence description of changes]

## Type
- [ ] Feature (new behavior)
- [ ] Fix (bug fix)
- [ ] Docs (documentation only)
- [ ] Refactor (no behavior change)
- [ ] Test (test additions only)
- [ ] Perf (performance improvement)

## Testing
- [ ] Added tests for new behavior
- [ ] All 1,626 tests pass locally
- [ ] Coverage ≥85% on new code
- [ ] Syntax check passes
- [ ] GUI/CLI tested manually

## Related
Fixes #XXX (if applicable)

## Notes
[Optional additional context]
```

---

## Code Review Expectations

**We look for:**
- ✅ Clear, descriptive commit messages
- ✅ Tests for all new behavior
- ✅ No performance regressions
- ✅ Adherence to code style
- ✅ Single-source-of-truth maintenance (params.py)

**We'll ask for changes if:**
- ❌ No tests or tests are incomplete
- ❌ Hardcoded values instead of params.py registry
- ❌ Comments that should be self-evident code
- ❌ Python loops over pixels (should be NumPy)
- ❌ Circular imports

---

**Welcome aboard! 🚀 Questions? Open a discussion or reach out to the maintainers.**

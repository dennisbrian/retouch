"""Mutation audit for the golden-fixture harnesses.

Applies a deliberate corruption to product code (or fixture code, where
noted), runs both golden harnesses, and records whether each harness
KILLED the mutant (test failed => fixture catches the regression) or the
mutant SURVIVED (tests passed => coverage gap).

Always reverts via `git checkout --` in a finally block.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = str(REPO / ".venv" / "bin" / "python")

BASE_HARNESS = "tests/test_golden_pipeline.py"
FACE_HARNESS = "tests/test_golden_pipeline_face.py"

# Each mutation: (id, file, old_substring, new_substring, rationale, expectation)
MUTATIONS = [
    (
        "M1_oklab_m1_subtle",
        "retouch/color_science.py",
        "[0.4121656, 0.5362752, 0.0514575],",
        "[0.4141656, 0.5362752, 0.0514575],",
        "OKLab M1 linear-RGB->LMS row0 +0.002 (0.5%) - live color matrix",
        "probe: is a small live-matrix error caught?",
    ),
    (
        "M2_oklab_m1_gross",
        "retouch/color_science.py",
        "[0.2118591, 0.6807189, 0.1074220],",
        "[0.2118591, 0.5807189, 0.1074220],",
        "OKLab M1 row1 green -0.10 (15% gross error) - live color matrix",
        "probe: is a GROSS live-matrix error caught?",
    ),
    (
        "M3_landmark_scaling",
        "tests/golden_face_fixture.py",
        "ied = _SRC_IED * ((scale_x + scale_y) / 2.0)",
        "ied = _SRC_IED * ((scale_x + scale_y) / 2.0) * 1.15",
        "inter-eye distance scaling inflated 15% - drives region mask sizing",
        "face harness should KILL; base harness unaffected",
    ),
    (
        "M4_deadcode_srgb_xyz",
        "retouch/color_space.py",
        "[0.2126729, 0.7151522, 0.0721750],",
        "[0.2126729, 0.6151522, 0.0721750],",
        "sRGB->XYZ luminance row -0.10; feeds bgr_to_prophoto, which has NO callers",
        "expected SURVIVE - documents dead code, not a fixture gap",
    ),
]



def run_harness(path: str) -> tuple[bool, str]:
    """Return (passed, tail_of_output)."""
    proc = subprocess.run(
        [PY, "-m", "pytest", path, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True,
    )
    tail = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    return proc.returncode == 0, (tail[-1] if tail else "<no output>")


def apply_mutation(file: str, old: str, new: str) -> None:
    p = REPO / file
    text = p.read_text()
    if text.count(old) != 1:
        raise SystemExit(f"anchor not unique in {file}: {text.count(old)} occurrences")
    p.write_text(text.replace(old, new))


def revert(file: str) -> None:
    subprocess.run(["git", "checkout", "--", file], cwd=REPO, check=True)


def main() -> None:
    results = []
    for mid, file, old, new, rationale, expectation in MUTATIONS:
        print(f"\n=== {mid} :: {file} ===")
        print(f"    {rationale}")
        try:
            apply_mutation(file, old, new)
            base_pass, base_tail = run_harness(BASE_HARNESS)
            face_pass, face_tail = run_harness(FACE_HARNESS)
        finally:
            revert(file)

        entry = {
            "id": mid, "file": file, "rationale": rationale,
            "expectation": expectation,
            "base_harness": "SURVIVED" if base_pass else "KILLED",
            "face_harness": "SURVIVED" if face_pass else "KILLED",
            "base_tail": base_tail, "face_tail": face_tail,
        }
        results.append(entry)
        print(f"    base: {entry['base_harness']:9s} ({base_tail})")
        print(f"    face: {entry['face_harness']:9s} ({face_tail})")

    out = Path(__file__).parent / "mutation_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")

    # Confirm tree is clean after all reverts.
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                           capture_output=True, text=True).stdout.strip()
    tracked = [l for l in dirty.splitlines() if not l.startswith("??")]
    print("tracked-file cleanliness after revert:", "CLEAN" if not tracked else tracked)


if __name__ == "__main__":
    main()

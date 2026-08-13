#!/usr/bin/env python3
"""Apply a completed human-review worksheet to an automatic Core matrix.

This intentionally does not render images. It updates only the independent
human-review and final-certification fields after a reviewer has completed
``human_review.json`` produced by ``core_recipe_certification.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retouch.certification import core_case_human_review_status, core_matrix_final_certified


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="Certification output directory containing matrix_manifest.json")
    parser.add_argument("--review", default=None, help="Human-review JSON; defaults to OUTPUT/human_review.json")
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    matrix_path = output / "matrix_manifest.json"
    review_path = Path(args.review).expanduser().resolve() if args.review else output / "human_review.json"
    if not matrix_path.is_file() or not review_path.is_file():
        parser.error("matrix_manifest.json and human_review.json must both exist")

    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    review_payload = json.loads(review_path.read_text(encoding="utf-8"))
    rows = list(matrix.get("cases") or [])
    human_rows = list(review_payload.get("rows") or [])
    for row in rows:
        row["human_review"] = core_case_human_review_status(
            str(row.get("case", "")), row.get("recipes") or [], human_rows,
        )
    matrix["final_certified"] = core_matrix_final_certified(
        rows,
        human_rows,
        int(matrix.get("recipe_count", 0)),
        corpus_complete=bool(matrix.get("corpus_complete")),
    )
    matrix["cases"] = rows
    matrix_path.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    print(f"final_certified={matrix['final_certified']}")
    return 0 if matrix["final_certified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Apply Certification Evidence v2 to an automatic Core matrix.

This intentionally does not render images. It updates only derived review and
final-certification fields after a v2 review payload has been completed. A
legacy v1 worksheet remains readable as a non-certifying input, but it cannot
make a matrix certified because it does not prove independent blinded review.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retouch.certification_review_v2 import finalize_matrix_review


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
    result = finalize_matrix_review(matrix, review_payload)
    finalized = result["matrix"]
    matrix_path.write_text(json.dumps(finalized, indent=2), encoding="utf-8")
    print(f"final_certified={finalized['final_certified']}")
    if result["errors"]:
        print(f"review_errors={len(result['errors'])}", file=sys.stderr)
    return 0 if finalized["final_certified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

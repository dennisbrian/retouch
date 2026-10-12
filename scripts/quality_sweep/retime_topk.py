"""Re-measure runtime for the top-K configs SERIALLY.

Runtime collected during the parallel sweep is contaminated: N agents doing
CPU-bound image work on one machine contend for cores, so wall-clock per
config reflects scheduler pressure, not the config. Quality metrics (SSIM,
dE) are deterministic and unaffected; only timing needs redoing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    lb = json.loads((HERE / "leaderboard.json").read_text())
    meta = lb["meta"]

    from sweep_runner import load_corpus  # noqa
    from retouch.engine import RetouchEngine

    corpus = load_corpus(Path(meta["corpus"]), meta["downscale"], None)
    engine = RetouchEngine()
    recipe = meta["recipe"]

    def timeit(**kw):
        best = None
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            for _n, img, _m in corpus:
                engine.process(img, recipe=recipe, **kw)
            dt = (time.perf_counter() - t0) / len(corpus)
            best = dt if best is None else min(best, dt)
        return round(best, 4)  # min of repeats = least contended sample

    print("warming up engine...")
    engine.process(corpus[0][1], recipe=recipe)

    base = timeit()
    print(f"baseline serial runtime/img: {base}s\n")

    out = []
    for r in lb["ranked"][:args.top]:
        rt = timeit(**r["params"])
        out.append({"rank": r["rank"], "config_id": r["config_id"],
                    "runtime_parallel_s": r.get("runtime_s_mean"),
                    "runtime_serial_s": rt,
                    "vs_baseline_pct": round((rt - base) / base * 100, 1)})
        print(f"#{r['rank']:>2} serial={rt:.4f}s "
              f"(parallel had {r.get('runtime_s_mean')}s) "
              f"{out[-1]['vs_baseline_pct']:+.1f}% vs baseline")

    lb["serial_timing"] = {"baseline_serial_s": base, "repeats": args.repeats,
                           "note": "min-of-repeats, measured with no parallel load",
                           "results": out}
    (HERE / "leaderboard.json").write_text(json.dumps(lb, indent=2))
    print(f"\nleaderboard updated with serial timings")


if __name__ == "__main__":
    main()

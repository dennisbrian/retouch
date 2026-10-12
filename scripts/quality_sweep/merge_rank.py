"""Merge sweep slice results into a persistent leaderboard and rank them.

Ranking objective
-----------------
    quality = normalized_effect * structure_retention

where effect  = delta_e_face_vs_baseline (clamped to an editorial band)
      retention = ssim_face_vs_baseline

*** KNOWN LIMITATION -- READ BEFORE USING THIS RANKING ***

This score is MONOTONE IN dE over the observed data, so it ranks configs by
"most changed from baseline", NOT by "best looking". Two reasons:

  1. The DE_HI clamp never binds. It is set to 6.0 but observed dE tops out
     around 2.6, so effect_score stays on its linear segment and never
     penalises an over-strong edit.
  2. ssim varies only ~0.987-0.995 (<1%), so the retention factor cannot
     meaningfully offset the effect term.

Net: quality_score ~= (dE - 0.5)/5.5 * ~0.99. A config that changes the face
more will outrank one that changes it less, regardless of whether the change
is an improvement. There is NO perceptual ground truth here -- the reference
is the baseline render, not a human-preferred target.

Treat the ranking as "magnitude of departure from baseline, structure-weighted".
To make it a genuine quality ranking you need either human preference labels
or a no-reference IQA model; neither is wired up.

Configs with any hard regression (crash / NaN / identity / degenerate) are
disqualified regardless of score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent
LEADERBOARD = HERE / "leaderboard.json"

# Editorial band for face-local dE vs baseline. Below LO the config is
# indistinguishable from baseline; above HI it is a heavy-handed edit.
DE_LO, DE_HI = 0.5, 6.0


def effect_score(de: float) -> float:
    if de is None:
        return 0.0
    if de <= DE_LO:
        return 0.0
    return min((de - DE_LO) / (DE_HI - DE_LO), 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    files = sorted(Path(args.results_dir).glob("*.json"))
    if not files:
        raise SystemExit(f"no slice results in {args.results_dir}")

    rows, gates, meta = [], [], {}
    for f in files:
        d = json.loads(f.read_text())
        gates.append((f.name, d.get("golden_gate", {}).get("ok")))
        meta.setdefault("recipe", d.get("recipe"))
        meta.setdefault("corpus", d.get("corpus"))
        meta.setdefault("downscale", d.get("downscale"))
        meta.setdefault("n_images", d.get("n_images"))
        meta.setdefault("baseline_runtime_s_mean", d.get("baseline_runtime_s_mean"))
        meta["n_configs_total"] = d.get("n_configs_total")
        rows.extend(d.get("results", []))

    # Deduplicate by config_id (slices are disjoint, but be defensive).
    seen, uniq = set(), []
    for r in rows:
        if r["config_id"] in seen:
            continue
        seen.add(r["config_id"])
        uniq.append(r)

    for r in uniq:
        r["disqualified"] = r["n_failures"] > 0
        ssim = r.get("ssim_face_vs_baseline")
        de = r.get("delta_e_face_vs_baseline")
        r["quality_score"] = (
            0.0 if (r["disqualified"] or ssim is None)
            else round(effect_score(de) * ssim, 6)
        )

    ranked = sorted(uniq, key=lambda r: -r["quality_score"])
    base_rt = meta.get("baseline_runtime_s_mean")
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
        rt = r.get("runtime_s_mean")
        r["runtime_vs_baseline_pct"] = (
            round((rt - base_rt) / base_rt * 100, 1)
            if rt and base_rt else None)

    payload = {
        "meta": meta,
        "golden_gate_per_slice": gates,
        "golden_gate_all_ok": all(g[1] for g in gates),
        "n_configs_scored": len(uniq),
        "n_disqualified": sum(1 for r in uniq if r["disqualified"]),
        "ranked": ranked,
    }
    LEADERBOARD.write_text(json.dumps(payload, indent=2))

    print(f"merged {len(files)} slices -> {len(uniq)} unique configs")
    print(f"golden gate all slices OK: {payload['golden_gate_all_ok']}")
    print(f"disqualified (hard regression): {payload['n_disqualified']}")
    print(f"baseline runtime/img: {base_rt}s\n")
    hdr = f"{'#':>2} {'score':>8} {'ssim':>8} {'dE':>7} {'rt_s':>6} {'d%':>7}  config"
    print(hdr); print("-" * len(hdr))
    for r in ranked[:args.top]:
        print(f"{r['rank']:>2} {r['quality_score']:>8.5f} "
              f"{(r.get('ssim_face_vs_baseline') or 0):>8.5f} "
              f"{(r.get('delta_e_face_vs_baseline') or 0):>7.3f} "
              f"{(r.get('runtime_s_mean') or 0):>6.3f} "
              f"{(r.get('runtime_vs_baseline_pct') or 0):>+7.1f}  {r['config_id']}")
    print(f"\nleaderboard -> {LEADERBOARD}")


if __name__ == "__main__":
    main()

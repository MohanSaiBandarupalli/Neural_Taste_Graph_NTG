"""Real MovieLens 32M benchmark, end to end, on your own machine.

1. Download ml-32m.zip from https://grouplens.org/datasets/movielens/ (licence: research use,
   no redistribution — that is why it is not in this repository) and unzip it so that
   data/external/movielens/ml-32m/ratings.csv exists.
2. python scripts/run_ml32m.py                 # ~10% user sample (fits in 16 GB RAM)
   python scripts/run_ml32m.py --user-sample 0.25
3. Results land in reports/v4/ml-32m/*.json; this script prints the headline table at the end.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

SRC = Path("data/external/movielens/ml-32m")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(SRC))
    ap.add_argument("--user-sample", type=float, default=0.1)
    ap.add_argument("--min-item-support", type=int, default=20)
    args = ap.parse_args()
    src = Path(args.source)
    missing = [f for f in ("ratings.csv", "movies.csv") if not (src / f).exists()]
    if missing:
        print(f"Missing {missing} in {src}. See the instructions at the top of this file.", file=sys.stderr)
        return 1
    size_gb = (src / "ratings.csv").stat().st_size / 1e9
    print(f"Found ML-32M at {src} (ratings.csv {size_gb:.2f} GB). Python {platform.python_version()}.")
    try:
        import torch  # noqa: F401
        print("PyTorch found: SASRec and two-tower channels will be trained and benchmarked.")
    except ImportError:
        print("PyTorch not installed (poetry install --with deep): deep models will be skipped.")
    from ntg.pipelines.v4 import Pipeline

    t0 = time.time()
    Pipeline("ml-32m").run(src, {"user_sample_frac": args.user_sample, "min_item_support": args.min_item_support})
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Reports: reports/v4/ml-32m/\n")
    b = json.loads(Path("reports/v4/ml-32m/benchmark.json").read_text())
    warm = b["results"]["warm"]
    print(f"Test window from {b['protocol']['t_test']}; warm users n={next(iter(warm.values()))['n_users']}")
    print(f"{'model':14s} {'NDCG@10':>9s} {'95% CI':>19s} {'Recall@20':>10s} {'Coverage':>9s}")
    for name, r in sorted(warm.items(), key=lambda kv: -kv[1]["metrics"]["ndcg@10"]):
        m, ci = r["metrics"], r["ci95"]["ndcg@10"]
        print(f"{name:14s} {m['ndcg@10']:9.4f} [{ci[0]:.4f}, {ci[1]:.4f}] {m['recall@20']:10.4f} {m['coverage@10']:9.1%}")
    print("\nShare reports/v4/ml-32m/ (small JSON files) to have the docs updated with real results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

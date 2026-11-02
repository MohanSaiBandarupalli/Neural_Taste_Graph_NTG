"""NTG command-line interface.

    ntg data synth   [--out data/external/synthetic] [--users 8000] [--items 3000]
    ntg build        --dataset synthetic [--source data/external/synthetic] [--quick]
    ntg build        --dataset ml-32m --source data/external/movielens/ml-32m --user-sample 0.1 --min-item-support 20
    ntg serve        [--host 127.0.0.1] [--port 8000]
    ntg legacy       (runs the original v1 DuckDB pipeline: build_dataset_duckdb -> graph -> ranker)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ntg", description="Neural Taste Graph v4")
    sub = ap.add_subparsers(dest="cmd", required=True)
    syn = sub.add_parser("data", help="dataset utilities")
    syn.add_argument("action", choices=["synth"])
    syn.add_argument("--out", default="data/external/synthetic")
    syn.add_argument("--users", type=int, default=8000)
    syn.add_argument("--items", type=int, default=3000)
    syn.add_argument("--seed", type=int, default=42)

    b = sub.add_parser("build", help="prepare + tune + train + evaluate + package")
    b.add_argument("--dataset", default="synthetic")
    b.add_argument("--source", default=None, help="directory with MovieLens-schema ratings.csv/movies.csv/tags.csv")
    b.add_argument("--quick", action="store_true", help="smaller grids / fewer users (CI)")
    b.add_argument("--user-sample", type=float, default=1.0)
    b.add_argument("--min-item-support", type=int, default=5)
    b.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("serve", help="run the API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    sub.add_parser("legacy", help="run the original v1 pipeline (requires ML-32M at data/external/movielens/ml-32m)")
    args = ap.parse_args(argv)

    if args.cmd == "data":
        from ntg.datasets.synthetic import SyntheticConfig, generate

        card = generate(args.out, SyntheticConfig(n_users=args.users, n_items=args.items, seed=args.seed))
        print(json.dumps({k: v for k, v in card.items() if k != "config"}, indent=2))
    elif args.cmd == "build":
        from ntg.pipelines.v4 import Pipeline

        source = Path(args.source) if args.source else (Path("data/external/synthetic") if args.dataset == "synthetic" else None)
        if source is not None and args.dataset == "synthetic" and not (source / "ratings.csv").exists():
            from ntg.datasets.synthetic import SyntheticConfig, generate

            generate(source, SyntheticConfig(n_users=3000 if args.quick else 8000, n_items=1500 if args.quick else 3000))
        pipe = Pipeline(args.dataset, quick=args.quick, seed=args.seed)
        summary = pipe.run(source, {"user_sample_frac": args.user_sample, "min_item_support": args.min_item_support})
        print(json.dumps(summary, indent=2))
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("ntg.api.app:create_app", factory=True, host=args.host, port=args.port)
    elif args.cmd == "legacy":
        from ntg.graph import build_graph
        from ntg.models import ranker
        from ntg.pipelines import build_dataset_duckdb

        build_dataset_duckdb.main()
        build_graph.main()
        ranker.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Dataset preparation: MovieLens-schema CSV -> leakage-safe Parquet benchmark.

Split protocol (v4)
-------------------
A *global* temporal split: two timestamp cut-offs ``t_val`` and ``t_test`` are
chosen as quantiles of all interaction timestamps. Every training event happens
strictly before every validation event, and every validation event strictly
before every test event. This is stricter than the legacy per-user split (which
let one user's validation events precede another user's training events, i.e.
cross-user temporal leakage) and mirrors how a model trained "today" is used
"tomorrow".

Works on the real MovieLens 32M files (DuckDB streams the 32M-row CSV; an
optional deterministic user hash-sample keeps memory bounded on laptops) and on
the synthetic generator's output.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PrepareConfig:
    source_dir: Path
    out_dir: Path
    dataset_name: str = "synthetic"
    val_quantile: float = 0.80     # events before this quantile -> train
    test_quantile: float = 0.90    # events in [val_q, test_q) -> val; rest -> test
    min_user_interactions: int = 3
    min_item_support: int = 5      # computed on the TRAIN window only
    user_sample_frac: float = 1.0  # deterministic hash sample of users (ML-32M on laptops)
    duckdb_memory_limit: str = "4GB"
    duckdb_threads: int = 4


_YEAR = re.compile(r"\((\d{4})\)\s*$")
_SEQUEL = re.compile(r"(\s+(\d+|II|III|IV|V|VI|Part\s+\w+|Chapter\s+\w+))$", re.I)


def franchise_key(title: str) -> str:
    """Heuristic franchise key: strip year, sequel numbering and subtitles.

    MovieLens has no franchise field; this approximation is used for duplicate
    / franchise-repetition suppression and is documented as a heuristic.
    """
    t = _YEAR.sub("", title).strip()
    t = t.split(":")[0].strip()
    t = _SEQUEL.sub("", t).strip()
    t = re.sub(r",\s*The$", "", t)
    return re.sub(r"^(The|A|An)\s+", "", t, flags=re.I).lower()


def _clean_title(title: str) -> str:
    t = _YEAR.sub("", title).strip()
    m = re.match(r"^(.*), (The|A|An)$", t)
    return f"{m.group(2)} {m.group(1)}" if m else t


def prepare(cfg: PrepareConfig) -> dict:
    t_start = time.perf_counter()
    src = Path(cfg.source_dir)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ratings = src / "ratings.csv"
    movies = src / "movies.csv"
    tags = src / "tags.csv"
    for p in (ratings, movies):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Place MovieLens-schema CSVs there or run `ntg data synth`.")

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{cfg.duckdb_memory_limit}'")
    con.execute(f"SET threads={int(cfg.duckdb_threads)}")
    con.execute("SET preserve_insertion_order=false")
    sample = ""
    if cfg.user_sample_frac < 1.0:
        sample = f"WHERE (hash(userId) % 10000) < {int(cfg.user_sample_frac * 10000)}"
    con.execute(
        f"""
        CREATE TABLE raw AS
        SELECT CAST(userId AS INTEGER) AS user_id, CAST(movieId AS INTEGER) AS item_id,
               CAST(rating AS REAL) AS rating, CAST(timestamp AS BIGINT) AS ts
        FROM read_csv_auto('{ratings.as_posix()}', header=true) {sample}
        """
    )
    t_val, t_test = con.execute(
        f"SELECT quantile_disc(ts, {cfg.val_quantile}), quantile_disc(ts, {cfg.test_quantile}) FROM raw"
    ).fetchone()
    # Item support and user activity filters use TRAIN-window statistics only.
    con.execute(
        f"""
        CREATE TABLE keep_items AS
        SELECT item_id FROM raw WHERE ts < {t_val} GROUP BY item_id HAVING COUNT(*) >= {cfg.min_item_support}
        """
    )
    con.execute(
        f"""
        CREATE TABLE inter AS
        SELECT r.*, CASE WHEN ts < {t_val} THEN 'train' WHEN ts < {t_test} THEN 'val' ELSE 'test' END AS split
        FROM raw r SEMI JOIN keep_items k USING (item_id)
        """
    )
    con.execute(
        f"""
        CREATE TABLE keep_users AS
        SELECT user_id FROM inter GROUP BY user_id HAVING COUNT(*) >= {cfg.min_user_interactions}
        """
    )
    con.execute("CREATE TABLE final AS SELECT * FROM inter SEMI JOIN keep_users USING (user_id)")
    inter = con.execute("SELECT user_id, item_id, rating, ts, split FROM final ORDER BY ts, user_id, item_id").df()
    con.close()

    # Hard leakage contract.
    tr_max = inter.loc[inter.split == "train", "ts"].max()
    va_min, va_max = inter.loc[inter.split == "val", "ts"].agg(["min", "max"])
    te_min = inter.loc[inter.split == "test", "ts"].min()
    if not (tr_max < va_min and va_max < te_min):
        raise RuntimeError("Temporal leakage contract violated")

    # ----------------------------------------------------------------- items
    mv = pd.read_csv(movies)
    mv = mv[mv["movieId"].isin(inter["item_id"].unique())].copy()
    mv["year"] = mv["title"].str.extract(_YEAR, expand=False).astype("float").astype("Int64")
    mv["genres_list"] = mv["genres"].fillna("").map(
        lambda s: [g for g in s.split("|") if g and g != "(no genres listed)"]
    )
    mv["display_title"] = mv["title"].map(_clean_title)
    mv["franchise"] = mv["title"].map(franchise_key)
    if tags.exists():
        tg = pd.read_csv(tags)
        tg = tg[tg["timestamp"] < t_val]  # tags from the training window only
        tg["tag"] = tg["tag"].astype(str).str.lower().str.strip()
        top = (
            tg.groupby(["movieId", "tag"]).size().rename("n").reset_index()
            .sort_values(["movieId", "n", "tag"], ascending=[True, False, True])
            .groupby("movieId").head(8)
            .groupby("movieId")["tag"].apply(list)
        )
        mv["tags"] = mv["movieId"].map(top)
    else:
        mv["tags"] = None
    mv["tags"] = mv["tags"].map(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else [])
    items = mv.rename(columns={"movieId": "item_id"})[
        ["item_id", "title", "display_title", "year", "genres_list", "franchise", "tags"]
    ].rename(columns={"genres_list": "genres"}).sort_values("item_id").reset_index(drop=True)

    inter.to_parquet(out / "interactions.parquet", index=False)
    items.to_parquet(out / "items.parquet", index=False)

    counts = inter["split"].value_counts().to_dict()
    users_by_split = inter.groupby("split")["user_id"].nunique().to_dict()
    data_card = {}
    if (src / "DATA_CARD.json").exists():
        data_card = json.loads((src / "DATA_CARD.json").read_text())
    meta = {
        "dataset": cfg.dataset_name,
        "is_synthetic": bool(data_card.get("is_synthetic", False)),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "t_val": int(t_val),
        "t_test": int(t_test),
        "t_val_iso": pd.Timestamp(int(t_val), unit="s").isoformat(),
        "t_test_iso": pd.Timestamp(int(t_test), unit="s").isoformat(),
        "n_interactions": int(len(inter)),
        "n_users": int(inter["user_id"].nunique()),
        "n_items": int(len(items)),
        "interactions_by_split": {k: int(v) for k, v in counts.items()},
        "users_by_split": {k: int(v) for k, v in users_by_split.items()},
        "leakage_contract": "max(train.ts) < min(val.ts) <= max(val.ts) < min(test.ts): verified",
        "seconds": round(time.perf_counter() - t_start, 2),
    }
    if (src / "ground_truth").exists():
        gt_out = out / "ground_truth"
        gt_out.mkdir(exist_ok=True)
        for f in (src / "ground_truth").glob("*.parquet"):
            (gt_out / f.name).write_bytes(f.read_bytes())
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta

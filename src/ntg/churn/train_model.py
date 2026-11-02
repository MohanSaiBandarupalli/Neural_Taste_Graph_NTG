# src/ntg/churn/train_model.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Config
# ============================================================
@dataclass(frozen=True)
class ChurnModelConfig:
    user_features_path: Path = Path("data/features/user_features.parquet")
    train_split_path: Path = Path("data/processed/splits/train.parquet")

    out_dir: Path = Path("outputs/risk")
    out_scores_path: Path = Path("outputs/risk/churn_scores.parquet")
    out_model_report_path: Path = Path("reports/churn_model_report.json")

    # Churn proxy: inactive for N days at end of TRAIN
    inactivity_days: int = 90
    fallback_churn_frac: float = 0.20  # if proxy collapses to 1 class

    # Model
    max_iter: int = 2000
    C: float = 1.0
    random_state: int = 42

    # DuckDB
    threads: int = 8
    tmp_dir: Path = Path("data/interim/duckdb_tmp")


# ============================================================
# Helpers
# ============================================================
def _log(msg: str) -> None:
    print(msg, flush=True)


# ============================================================
# Churn labels (DuckDB, leakage-safe)
# ============================================================
def _build_labels_inactivity_duckdb(cfg: ChurnModelConfig) -> pd.DataFrame:
    """
    Build churn labels using TRAIN only.

    churn = 1 if user inactive >= inactivity_days
    inactivity measured from last interaction to end of TRAIN window.

    Handles ANY timestamp type safely:
    TIMESTAMP / DATE / INT / FLOAT → epoch seconds (DOUBLE)
    """
    cfg.tmp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={cfg.threads};")
    con.execute(f"PRAGMA temp_directory='{cfg.tmp_dir.as_posix()}';")

    # Raw view
    con.execute(
        f"""
        CREATE OR REPLACE VIEW raw_tr AS
        SELECT
            CAST(user_id AS BIGINT) AS user_id,
            timestamp AS ts_raw
        FROM read_parquet('{cfg.train_split_path.as_posix()}');
        """
    )

    # Normalize timestamp → epoch seconds (DOUBLE)
    con.execute(
        """
        CREATE OR REPLACE VIEW tr AS
        SELECT
            user_id,
            CAST(
                CASE
                    WHEN typeof(ts_raw) LIKE 'TIMESTAMP%' THEN epoch(ts_raw)
                    WHEN typeof(ts_raw) = 'DATE' THEN epoch(CAST(ts_raw AS TIMESTAMP))
                    WHEN typeof(ts_raw) IN ('BIGINT','INTEGER','SMALLINT','TINYINT')
                        THEN CAST(ts_raw AS DOUBLE)
                    WHEN typeof(ts_raw) IN ('DOUBLE','FLOAT','REAL')
                        THEN CAST(ts_raw AS DOUBLE)
                    ELSE epoch(CAST(ts_raw AS TIMESTAMP))
                END
            AS DOUBLE) AS ts
        FROM raw_tr;
        """
    )

    df = con.execute(
        f"""
        WITH mx AS (SELECT MAX(ts) AS train_end_ts FROM tr),
        last AS (
            SELECT user_id, MAX(ts) AS last_ts
            FROM tr
            GROUP BY user_id
        )
        SELECT
            l.user_id,
            ((SELECT train_end_ts FROM mx) - l.last_ts) / 86400.0
                AS inactivity_days,
            CASE
                WHEN ((SELECT train_end_ts FROM mx) - l.last_ts) / 86400.0
                     >= {cfg.inactivity_days}
                THEN 1 ELSE 0
            END AS churn_label
        FROM last l;
        """
    ).df()

    return df


# ============================================================
# Feature handling
# ============================================================
def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    blacklist = {
        "user_id",
        "item_id",
        "timestamp",
        "ts",
        "churn",
        "churn_label",
        "label",
        "target",
        "inactivity_days",
    }
    cols: List[str] = []
    for c in df.columns:
        if c.lower() in blacklist or c.lower().endswith("_id"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _ensure_two_classes(df: pd.DataFrame, cfg: ChurnModelConfig) -> pd.DataFrame:
    y = df["churn_label"].astype(int)
    if y.nunique() >= 2:
        return df

    _log(
        f"[WARN] churn_label collapsed to single class. "
        f"Applying fallback: top {int(cfg.fallback_churn_frac*100)}% inactive users → churn=1"
    )

    cutoff = df["inactivity_days"].quantile(1.0 - cfg.fallback_churn_frac)
    df = df.copy()
    df["churn_label"] = (df["inactivity_days"] >= cutoff).astype(int)
    return df


# ============================================================
# Model training
# ============================================================
def _train_eval_model(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: List[str],
    cfg: ChurnModelConfig,
) -> Tuple[Pipeline, dict]:

    # deterministic split
    order = np.argsort(X["user_id"].to_numpy())
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)

    cut = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:cut], X.iloc[cut:]
    y_train, y_val = y.iloc[:cut], y.iloc[cut:]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_cols,
            )
        ]
    )

    clf = LogisticRegression(
        max_iter=cfg.max_iter,
        C=cfg.C,
        solver="lbfgs",
        class_weight="balanced",
        random_state=cfg.random_state,
    )

    model = Pipeline([("pre", pre), ("clf", clf)])
    validation_note = "deterministic 80/20 split by user_id"
    if y_train.nunique() < 2:
        # Too little data for a held-out fold with both classes (e.g. tiny test
        # fixtures). Fit on everything and report validation metrics as NaN
        # instead of silently inventing numbers.
        model.fit(X, y)
        X_val, y_val = X.iloc[0:0], y.iloc[0:0]
        validation_note = "insufficient data for held-out validation; metrics not computed"
    else:
        model.fit(X_train, y_train)

    if len(X_val) and y_val.nunique() > 1:
        p_val = model.predict_proba(X_val)[:, 1]
        roc = float(roc_auc_score(y_val, p_val))
        prec, rec, _ = precision_recall_curve(y_val, p_val)
        pr_auc = float(auc(rec, prec))
    else:
        roc = float("nan")
        pr_auc = float("nan")

    report = {
        "n_users": int(len(X)),
        "churn_rate": float(y.mean()),
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "n_features": len(feature_cols),
        "proxy": {
            "type": "train_inactivity_days",
            "days": cfg.inactivity_days,
            "fallback_frac": cfg.fallback_churn_frac,
        },
        "validation": validation_note,
        # NOTE: legacy v1 model. Labels (inactivity at end of TRAIN) are derived
        # from the same window as the features, so recency features leak the
        # label. Use ntg.enterprise.engagement_risk for forward-looking labels.
        "leakage_safe": False,
        "superseded_by": "ntg.enterprise.engagement_risk",
    }
    return model, report


# ============================================================
# Pipeline entry
# ============================================================
def train_and_score(cfg: ChurnModelConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    _log("=== Day-6: Churn model (leakage-safe proxy) ===")
    _log("[1/4] Building churn labels from TRAIN inactivity (DuckDB)")
    labels = _build_labels_inactivity_duckdb(cfg)

    _log("[2/4] Loading user_features")
    uf = pd.read_parquet(cfg.user_features_path)
    df = uf.merge(labels, on="user_id", how="inner")

    if df.empty:
        raise RuntimeError("No overlap between user_features and churn labels")

    df = _ensure_two_classes(df, cfg)

    feature_cols = _select_feature_columns(df)
    if len(feature_cols) < 3:
        raise RuntimeError("Too few numeric features for churn model")

    y = df["churn_label"].astype(int)
    X = df[["user_id"] + feature_cols]

    _log(f"[3/4] Training logistic model ({len(feature_cols)} features)")
    model, report = _train_eval_model(X, y, feature_cols, cfg)

    _log("[4/4] Scoring users")
    p = model.predict_proba(X)[:, 1]

    # ✅ tests expect churn_prob (not p_churn)
    out = pd.DataFrame(
        {
            "user_id": df["user_id"].astype("int64"),
            "churn_prob": pd.Series(p, dtype="float64").clip(0.0, 1.0).astype("float32"),
        }
    )
    out.to_parquet(cfg.out_scores_path, index=False)

    cfg.out_model_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _log(f"✅ Wrote: {cfg.out_scores_path}")
    _log(f"✅ Report: {cfg.out_model_report_path}")


if __name__ == "__main__":
    train_and_score(ChurnModelConfig())

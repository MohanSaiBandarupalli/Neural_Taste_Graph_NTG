"""Temporal-holdout evaluation of TasteTwin.

1. *Future preference prediction*: predict each user's genre distribution in
   the holdout window from fit-window data only. Compared against the global
   distribution (non-personalised), a static long-term profile, a short-term
   profile, and the TasteTwin blend (blend weight chosen on the validation
   window, then frozen for test). Metric: Jensen–Shannon divergence (base 2,
   lower is better) and top-3 genre recall.
2. *Drift detection* (synthetic data only, where ground truth exists): AUC of
   TasteTwin's instability score (1 - stability) for identifying users with a
   planted taste drift that started before the cutoff.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ntg.datasets.matrix import InteractionData
from ntg.recsys.evaluate import bootstrap_ci
from ntg.tastetwin.model import TasteTwin, preference_weight


def _true_share(tt: TasteTwin, holdout: pd.DataFrame, users: np.ndarray) -> np.ndarray:
    h = holdout[holdout["u"].isin(users)]
    w = preference_weight(h["rating"].to_numpy())
    S = np.zeros((tt.data.n_users, len(tt.genres)))
    np.add.at(S, h["u"].to_numpy(), w[:, None] * tt.G[h["i"].to_numpy()])
    S = S[users]
    return S / np.maximum(S.sum(1, keepdims=True), 1e-9)


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)

    def kl(a, b):
        return np.where(a > 0, a * np.log2(np.maximum(a, 1e-12) / np.maximum(b, 1e-12)), 0).sum(-1)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def top3_recall(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    tp = np.argsort(-pred, 1)[:, :3]
    tt = np.argsort(-true, 1)[:, :3]
    return np.array([len(set(a) & set(b)) / 3 for a, b in zip(tp, tt)])


def _eligible(fit_df, holdout, min_fit=5, min_hold=3):
    nf = fit_df.groupby("u").size()
    nh = holdout.groupby("u").size()
    return np.intersect1d(nf[nf >= min_fit].index.values, nh[nh >= min_hold].index.values)


def evaluate_tastetwin(data: InteractionData, bootstrap: int = 500) -> dict:
    """Select half-lives + blend on VAL, then report frozen results on TEST."""
    from ntg.tastetwin.model import TasteTwinConfig

    STATIC = TasteTwinConfig(half_life_long_days=1e9, half_life_short_days=1e9)
    # 1) model selection on the validation window
    wv = data.window("val")
    uv = _eligible(wv.fit, wv.holdout)
    val_grid = []
    for hl_long in (180.0, 365.0, 730.0):
        for hl_short in (20.0, 45.0, 90.0):
            cfg = TasteTwinConfig(half_life_long_days=hl_long, half_life_short_days=hl_short)
            tv = TasteTwin(data, cfg).fit(wv.fit, wv.cutoff_ts)
            true_v = _true_share(tv, wv.holdout, uv)
            for a in np.round(np.linspace(0, 1, 6), 2):
                js = float(js_divergence(tv.predict_future_share(uv, a), true_v).mean())
                val_grid.append({"half_life_long": hl_long, "half_life_short": hl_short, "blend": float(a), "js": js})
    best = min(val_grid, key=lambda r: r["js"])
    best_cfg = TasteTwinConfig(half_life_long_days=best["half_life_long"], half_life_short_days=best["half_life_short"])

    # 2) frozen evaluation on the test window
    wt = data.window("test")
    ut = _eligible(wt.fit, wt.holdout)
    tt = TasteTwin(data, best_cfg).fit(wt.fit, wt.cutoff_ts)
    ts_ = TasteTwin(data, STATIC).fit(wt.fit, wt.cutoff_ts)
    true_t = _true_share(tt, wt.holdout, ut)
    preds = {
        "global_popularity": np.broadcast_to(tt.global_share, true_t.shape),
        "static_profile_no_decay": ts_.predict_future_share(ut, 0.0),
        "tastetwin_long_only": tt.predict_future_share(ut, 0.0),
        "tastetwin_short_only": tt.predict_future_share(ut, 1.0),
        "tastetwin": tt.predict_future_share(ut, best["blend"]),
    }
    res = {}
    for name, p in preds.items():
        js = js_divergence(p, true_t)
        r3 = top3_recall(p, true_t)
        res[name] = {
            "js_divergence": float(js.mean()),
            "js_ci95": bootstrap_ci(js, bootstrap),
            "top3_genre_recall": float(r3.mean()),
            "top3_ci95": bootstrap_ci(r3, bootstrap),
        }
    diff = js_divergence(preds["static_profile_no_decay"], true_t) - js_divergence(preds["tastetwin"], true_t)
    out = {
        "protocol": "select half-lives/blend on VAL; fit on events < test cutoff; predict genre distribution of "
                    "test-window events (users with >=5 fit and >=3 holdout events)",
        "selected_on_val": best,
        "val_grid": val_grid,
        "n_users_test": int(len(ut)),
        "n_users_val": int(len(uv)),
        "test": res,
        "js_reduction_vs_static": float(diff.mean()),
        "js_reduction_vs_static_ci95": bootstrap_ci(diff, bootstrap),
    }

    gt_path = Path(data.root) / "ground_truth" / "users.parquet"
    if gt_path.exists():
        gt = pd.read_parquet(gt_path)
        gt = gt[gt["userId"].isin(data.user_index)]
        gt["u"] = gt["userId"].map(data.user_index)
        cutoff = wt.cutoff_ts
        tt = TasteTwin(data).fit(wt.fit, wt.cutoff_ts)  # a-priori default config for detection
        active = tt.n_events[gt["u"].to_numpy()] >= 10
        g = gt[active].copy()
        started = g["drift_start_ts"].fillna(np.inf).to_numpy() < cutoff - 60 * 86400
        g["label"] = started & g["drifts"].to_numpy()
        g = g[(g["label"]) | (~g["drifts"])]  # exclude drifters whose drift had not yet started
        u = g["u"].to_numpy()
        instab = 1 - tt.stability(u)
        mom = np.abs(tt.momentum(u)).max(1)
        if g["label"].nunique() == 2:
            out["drift_detection_synthetic"] = {
                "n_users": int(len(g)),
                "n_drifting": int(g["label"].sum()),
                "auc_instability": float(roc_auc_score(g["label"], instab)),
                "auc_max_abs_momentum": float(roc_auc_score(g["label"], mom)),
                "config": "default TasteTwinConfig (not tuned on this task)",
                "note": "Ground truth from the synthetic generator; not available for real MovieLens.",
            }
    return out

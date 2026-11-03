"""Engagement-risk (churn-proxy) model with forward-looking temporal labels.

MovieLens contains NO subscription or cancellation data. This module therefore
predicts an explicitly defined *engagement* proxy:

    population : users with >= 1 event in the 90 days before cutoff ``c``
    label = 1  : the user has NO event in [c, c + horizon)

Features use only events with ts < c. The model is trained at the validation
cutoff (labels from the validation window) and evaluated at the test cutoff
(labels from the test window) — a genuine out-of-time test. A calibrated
gradient-boosting model is compared with logistic-regression and
recency-only baselines; metrics are reported overall and by tenure segment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ntg.datasets.matrix import InteractionData
from ntg.discovery.features import ACTIVITY_FEATURES, FATIGUE_FEATURES, user_state_features

DAY = 86400.0
FEATURES = ACTIVITY_FEATURES + FATIGUE_FEATURES
FEATURE_TEXT = {
    "days_since_last": "days since last activity", "tenure_days": "tenure", "events_last_30d": "activity in last 30 days",
    "events_last_90d": "activity in last 90 days", "events_prev_90d": "activity in the 90 days before that",
    "activity_trend": "activity trend", "mean_rating": "average rating", "rating_std": "rating variability",
    "distinct_genres": "genre breadth", "recent_genre_hhi": "recent genre concentration",
    "hhi_delta": "change in genre concentration", "franchise_repeat_recent": "franchise repetition",
    "rating_trend": "recent rating trend", "novelty_shift": "novelty shift", "gap_trend": "session-gap trend",
    "log_n_events": "history length",
}


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def snapshot(data: InteractionData, cutoff: int, horizon_days: int = 90, active_days: int = 90,
             with_labels: bool = True) -> pd.DataFrame:
    df = data.interactions
    X = user_state_features(data, df, cutoff)
    X = X[X["days_since_last"] <= active_days].copy()
    if with_labels:
        end = cutoff + horizon_days * DAY
        if end > df["ts"].max() + 1:
            raise ValueError("label window extends beyond the data (right-censored)")
        fut = df[(df["ts"] >= cutoff) & (df["ts"] < end)]["u"].unique()
        X["label"] = (~X["u"].isin(fut)).astype(int)
    return X.reset_index(drop=True)


def tenure_segment(tenure_days: pd.Series) -> pd.Series:
    return pd.cut(tenure_days, [-1, 90, 365, 1e9], labels=["new (<90d)", "established (90-365d)", "loyal (>1y)"]).astype(str)


@dataclass
class EngagementRiskModel:
    horizon_days: int = 90
    seed: int = 0

    def fit(self, data: InteractionData, cutoff: int) -> dict:
        tr = snapshot(data, cutoff, self.horizon_days)
        X, y = tr[FEATURES].to_numpy(float), tr["label"].to_numpy()
        gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                             l2_regularization=1.0, random_state=self.seed)
        self.model = CalibratedClassifierCV(gbm, method="isotonic", cv=3).fit(X, y)
        self.logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)).fit(X, y)
        self.train_cutoff = cutoff
        self.mean_ = X.mean(0)
        return {"n_users": int(len(tr)), "positive_rate": float(y.mean()), "cutoff": int(cutoff)}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[FEATURES].to_numpy(float))[:, 1]

    def evaluate(self, data: InteractionData, cutoff: int) -> dict:
        te = snapshot(data, cutoff, self.horizon_days)
        X, y = te[FEATURES].to_numpy(float), te["label"].to_numpy()
        p = self.model.predict_proba(X)[:, 1]
        p_lr = self.logreg.predict_proba(X)[:, 1]
        rec = te["days_since_last"].to_numpy()

        def m(score, prob=None):
            out = {"auc": float(roc_auc_score(y, score)), "pr_auc": float(average_precision_score(y, score))}
            if prob is not None:
                out["brier"] = float(brier_score_loss(y, prob))
                out["ece"] = ece(y, prob)
            top = np.argsort(-score)[: max(1, len(score) // 10)]
            out["lift_top_decile"] = float(y[top].mean() / max(y.mean(), 1e-9))
            return out

        seg = tenure_segment(te["tenure_days"])
        by_seg = {}
        for s in sorted(seg.unique()):
            mask = (seg == s).to_numpy()
            if len(np.unique(y[mask])) == 2:
                by_seg[s] = {"n": int(mask.sum()), "positive_rate": float(y[mask].mean()),
                             "auc": float(roc_auc_score(y[mask], p[mask])), "ece": ece(y[mask], p[mask])}
        # calibration curve for the UI
        edges = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
        curve = [{"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": int((idx == b).sum()),
                  "predicted": float(p[idx == b].mean()), "observed": float(y[idx == b].mean())}
                 for b in range(10) if (idx == b).any()]
        return {
            "label_definition": f"no activity in the {self.horizon_days} days after cutoff, among users active in the 90 days before",
            "is_proxy": True,
            "proxy_note": "MovieLens has no subscription/cancellation data; this is an engagement proxy, not churn.",
            "train_cutoff_iso": pd.Timestamp(self.train_cutoff, unit="s").isoformat(),
            "test_cutoff_iso": pd.Timestamp(cutoff, unit="s").isoformat(),
            "n_users": int(len(te)),
            "positive_rate": float(y.mean()),
            "gbm_calibrated": m(p, p),
            "logistic_regression": m(p_lr, p_lr),
            "recency_only_baseline": m(rec),
            "by_tenure_segment": by_seg,
            "calibration_curve": curve,
        }

    def explain(self, row: pd.Series) -> list[dict]:
        """Local explanation via standardised logistic-regression contributions (transparent proxy)."""
        sc = self.logreg[0]
        lr = self.logreg[1]
        z = (row[FEATURES].to_numpy(float) - sc.mean_) / np.where(sc.scale_ > 0, sc.scale_, 1)
        c = z * lr.coef_[0]
        order = np.argsort(-c)
        return [{"feature": FEATURES[j], "text": FEATURE_TEXT[FEATURES[j]], "value": float(row[FEATURES[j]]),
                 "contribution": float(c[j])} for j in order[:4] if c[j] > 0]

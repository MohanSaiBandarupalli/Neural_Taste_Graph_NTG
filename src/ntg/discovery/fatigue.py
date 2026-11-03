"""Recommendation-fatigue risk.

Definition used here (stated explicitly because "fatigue" is easy to fake):

    A user is *fatigued* at cutoff ``c`` if, among users who REMAIN ACTIVE in the
    following window (>= ``min_future_events`` events), their mean rating in
    that window drops by more than ``drop`` stars versus their own history.

Restricting to users who stay active means inactivity is *never* used as
evidence of fatigue (disengagement is modelled separately in
:mod:`ntg.enterprise.engagement_risk`).

The model is a standardised logistic regression over interpretable signals
(genre concentration, concentration increase, franchise repetition, rating
trend, novelty shift, session-gap trend, activity volume) so every score
decomposes into per-signal contributions for the explanation UI.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ntg.datasets.matrix import InteractionData
from ntg.discovery.features import FATIGUE_FEATURES, user_state_features

SIGNAL_TEXT = {
    "recent_genre_hhi": "recent viewing concentrated in few genres",
    "hhi_delta": "viewing has become more concentrated than usual",
    "franchise_repeat_recent": "repeated franchise / sequel consumption",
    "rating_trend": "recent ratings vs. personal average",
    "novelty_shift": "shift towards less mainstream titles",
    "gap_trend": "time between sessions changing",
    "log_n_events": "amount of history",
}


def fatigue_labels(df_future: pd.DataFrame, hist: pd.DataFrame, min_future_events: int = 3, drop: float = 0.35) -> pd.Series:
    fut = df_future.groupby("u")["rating"].agg(["mean", "size"])
    fut = fut[fut["size"] >= min_future_events]
    base = hist.groupby("u")["rating"].mean()
    common = fut.index.intersection(base.index)
    return ((base.loc[common] - fut.loc[common, "mean"]) > drop).astype(int)


@dataclass
class FatigueModel:
    C: float = 0.5

    def fit(self, data: InteractionData, hist: pd.DataFrame, future: pd.DataFrame, cutoff: int) -> dict:
        X = user_state_features(data, hist, cutoff).set_index("u")
        y = fatigue_labels(future, hist)
        idx = X.index.intersection(y.index)
        Xf = X.loc[idx, FATIGUE_FEATURES].to_numpy(float)
        self.scaler = StandardScaler().fit(Xf)
        self.clf = LogisticRegression(C=self.C, max_iter=2000).fit(self.scaler.transform(Xf), y.loc[idx])
        p = self.clf.predict_proba(self.scaler.transform(Xf))[:, 1]
        return {"n_users": int(len(idx)), "positive_rate": float(y.loc[idx].mean()),
                "train_auc": float(roc_auc_score(y.loc[idx], p))}

    def evaluate(self, data, hist, future, cutoff) -> dict:
        X = user_state_features(data, hist, cutoff).set_index("u")
        y = fatigue_labels(future, hist)
        idx = X.index.intersection(y.index)
        p = self.predict(X.loc[idx])
        yy = y.loc[idx].to_numpy()
        # Baseline: a single naive signal (recent genre concentration)
        naive = X.loc[idx, "recent_genre_hhi"].to_numpy()
        return {
            "n_users": int(len(idx)),
            "positive_rate": float(yy.mean()),
            "auc": float(roc_auc_score(yy, p)),
            "pr_auc": float(average_precision_score(yy, p)),
            "baseline_recent_hhi_auc": float(roc_auc_score(yy, naive)),
            "coefficients": dict(zip(FATIGUE_FEATURES, map(float, self.clf.coef_[0]))),
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(self.scaler.transform(X[FATIGUE_FEATURES].to_numpy(float)))[:, 1]

    def explain(self, row: pd.Series) -> list[dict]:
        z = self.scaler.transform(row[FATIGUE_FEATURES].to_numpy(float)[None, :])[0]
        contrib = z * self.clf.coef_[0]
        order = np.argsort(-np.abs(contrib))
        return [{"signal": FATIGUE_FEATURES[j], "text": SIGNAL_TEXT[FATIGUE_FEATURES[j]],
                 "value": float(row[FATIGUE_FEATURES[j]]), "contribution": float(contrib[j])} for j in order[:4]]

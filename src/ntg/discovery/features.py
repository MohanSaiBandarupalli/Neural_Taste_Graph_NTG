"""User-state features as of a cutoff (shared by fatigue and engagement-risk models).

Every feature uses only events with ``ts < cutoff``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ntg.datasets.matrix import InteractionData

DAY = 86400.0

FATIGUE_FEATURES = [
    "recent_genre_hhi", "hhi_delta", "franchise_repeat_recent", "rating_trend",
    "novelty_shift", "gap_trend", "log_n_events",
]
ACTIVITY_FEATURES = [
    "days_since_last", "tenure_days", "events_last_30d", "events_last_90d", "events_prev_90d",
    "activity_trend", "mean_rating", "rating_std", "distinct_genres",
]


def user_state_features(data: InteractionData, df: pd.DataFrame, cutoff: int, recent_n: int = 15) -> pd.DataFrame:
    df = df[df["ts"] < cutoff].sort_values(["u", "ts", "i"])
    if df.empty:
        return pd.DataFrame(columns=["u", *FATIGUE_FEATURES, *ACTIVITY_FEATURES])
    G = data.genre_matrix
    primary = np.argmax(G, axis=1)
    pop = data.popularity(df)
    logpop = np.log1p(pop)
    df = df.assign(
        g=primary[df["i"].to_numpy()],
        fr=data.franchise_codes[df["i"].to_numpy()],
        lp=logpop[df["i"].to_numpy()],
    )
    df["rk_rev"] = df.groupby("u").cumcount(ascending=False)
    df["recent"] = df["rk_rev"] < recent_n
    # franchise repeat: item's franchise already seen earlier by the same user
    df["fr_repeat"] = df.duplicated(["u", "fr"], keep="first")

    def hhi(s: pd.Series) -> float:
        p = s.value_counts(normalize=True).to_numpy()
        return float((p ** 2).sum())

    rec = df[df["recent"]]
    old = df[~df["recent"]]
    out = pd.DataFrame(index=pd.Index(df["u"].unique(), name="u"))
    out["recent_genre_hhi"] = rec.groupby("u")["g"].apply(hhi)
    out["long_genre_hhi"] = df.groupby("u")["g"].apply(hhi)
    out["hhi_delta"] = out["recent_genre_hhi"] - out["long_genre_hhi"]
    out["franchise_repeat_recent"] = rec.groupby("u")["fr_repeat"].mean()
    out["rating_trend"] = rec.groupby("u")["rating"].mean() - old.groupby("u")["rating"].mean()
    out["novelty_shift"] = old.groupby("u")["lp"].mean() - rec.groupby("u")["lp"].mean()
    gaps = df.assign(gap=df.groupby("u")["ts"].diff() / DAY)
    out["gap_trend"] = np.log1p(gaps[gaps["recent"]].groupby("u")["gap"].median()) - np.log1p(gaps.groupby("u")["gap"].median())
    n = df.groupby("u").size()
    out["log_n_events"] = np.log1p(n)
    last = df.groupby("u")["ts"].max()
    first = df.groupby("u")["ts"].min()
    out["days_since_last"] = (cutoff - last) / DAY
    out["tenure_days"] = (cutoff - first) / DAY
    for name, lo, hi in (("events_last_30d", 30, 0), ("events_last_90d", 90, 0), ("events_prev_90d", 180, 90)):
        m = (df["ts"] >= cutoff - lo * DAY) & (df["ts"] < cutoff - hi * DAY)
        out[name] = df[m].groupby("u").size()
    out = out.fillna({"events_last_30d": 0, "events_last_90d": 0, "events_prev_90d": 0})
    out["activity_trend"] = np.log1p(out["events_last_90d"]) - np.log1p(out["events_prev_90d"])
    out["mean_rating"] = df.groupby("u")["rating"].mean()
    out["rating_std"] = df.groupby("u")["rating"].std().fillna(0)
    out["distinct_genres"] = df.groupby("u")["g"].nunique()
    out = out.fillna({"rating_trend": 0.0, "novelty_shift": 0.0, "gap_trend": 0.0, "franchise_repeat_recent": 0.0})
    return out.reset_index()

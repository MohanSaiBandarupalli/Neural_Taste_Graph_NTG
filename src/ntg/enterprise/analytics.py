"""Audience, content and business analytics computed from platform data.

Every number returned here is labelled with its evidence type:
``observed`` (counted from interaction data), ``modeled`` (model output) or
``simulated`` (scenario arithmetic on user-editable assumptions).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

DAY = 86400.0


# ------------------------------------------------------------------ audience segments
def audience_segments(cs, n_segments: int = 6, seed: int = 0) -> tuple[np.ndarray, list[dict]]:
    """K-means over TasteTwin long-term genre shares (+ exploration) of active viewers."""
    tt = cs.tt
    users = np.flatnonzero(tt.n_events >= 5)
    feats = np.hstack([tt.long_share(users), 0.5 * tt.exploration[users, None]])
    km = KMeans(n_clusters=n_segments, n_init=10, random_state=seed).fit(feats)
    labels = np.full(tt.data.n_users, -1)
    labels[users] = km.labels_
    segs = []
    for s in range(n_segments):
        m = users[km.labels_ == s]
        share = tt.long_share(m).mean(0)
        top = [tt.genres[g] for g in np.argsort(-share)[:3]]
        segs.append({
            "segment_id": int(s),
            "name": " / ".join(top[:2]) + " fans",
            "n_viewers": int(len(m)),
            "top_genres": [{"genre": tt.genres[g], "share": float(share[g])} for g in np.argsort(-share)[:5]],
            "avg_exploration": float(tt.exploration[m].mean()),
            "avg_stability": float(tt.stability(m).mean()),
            "avg_events": float(tt.n_events[m].mean()),
            "evidence_type": "modeled",
        })
    return labels, segs


# ------------------------------------------------------------------ content intelligence
def content_intelligence(cs, rec_items: np.ndarray | None = None) -> dict:
    """Catalogue coverage, long-tail exposure and under-discovered genres.

    ``rec_items``: (U, K) recommended item indices for a sample of users (to
    measure what the recommender actually exposes).
    """
    d = cs.data
    pop = cs.pop
    G = d.genre_matrix
    n_items = d.n_items
    consumed = pop > 0
    head = pop >= np.quantile(pop[consumed], 0.8) if consumed.any() else np.zeros(n_items, bool)
    R = d.csr(cs.fit_df, "rating")
    mean_rating = np.asarray(R.sum(0)).ravel() / np.maximum(np.asarray((R > 0).sum(0)).ravel(), 1)
    genres = []
    exposure = np.bincount(rec_items.ravel(), minlength=n_items) if rec_items is not None else None
    for k, g in enumerate(d.genre_vocab):
        m = G[:, k] > 0
        if not m.any():
            continue
        row = {
            "genre": g,
            "n_titles": int(m.sum()),
            "catalog_share": float(m.mean()),
            "consumption_share": float(pop[m].sum() / max(pop.sum(), 1)),
            "avg_rating": float(np.average(mean_rating[m], weights=np.maximum(pop[m], 1e-9))) if pop[m].sum() else None,
            "long_tail_titles": int((m & ~head).sum()),
        }
        if exposure is not None:
            row["recommendation_exposure_share"] = float(exposure[m].sum() / max(exposure.sum(), 1))
        genres.append(row)
    gdf = pd.DataFrame(genres)
    # Under-discovered: consumed less than catalogue share, but well rated
    gdf["discovery_gap"] = gdf["catalog_share"] - gdf["consumption_share"]
    under = gdf[(gdf["discovery_gap"] > 0) & (gdf["avg_rating"].fillna(0) >= gdf["avg_rating"].median())]
    # Hidden gems: high rating, low popularity, enough ratings to trust
    n_ratings = pop
    gem_mask = (n_ratings >= 10) & ~head
    gems = np.flatnonzero(gem_mask)
    gems = gems[np.argsort(-mean_rating[gems])][:20]
    out = {
        "evidence_type": "observed",
        "n_titles": int(n_items),
        "titles_with_any_interaction": int(consumed.sum()),
        "head_share_of_consumption": float(pop[head].sum() / max(pop.sum(), 1)),
        "genres": gdf.sort_values("consumption_share", ascending=False).to_dict("records"),
        "under_discovered_genres": under.sort_values("discovery_gap", ascending=False)[["genre", "catalog_share", "consumption_share", "avg_rating"]].to_dict("records"),
        "hidden_gems": [{"item_index": int(i), "title": d.item_title(int(i)), "genres": list(d.items.at[int(i), "genres"]),
                         "avg_rating": float(mean_rating[i]), "n_ratings": int(n_ratings[i])} for i in gems],
        "limitations": "No content cost, licensing or revenue data: commercial content performance cannot be assessed.",
    }
    if exposure is not None:
        out["recommendation_catalog_coverage"] = float((exposure > 0).mean())
        out["recommendation_long_tail_share"] = float(exposure[~head].sum() / max(exposure.sum(), 1))
    return out


# ------------------------------------------------------------------ business scenario simulator
@dataclass(frozen=True)
class ScenarioAssumptions:
    n_subscribers: int = 1_000_000
    monthly_price: float = 12.99
    baseline_monthly_churn: float = 0.035
    retention_improvement_rel: float = 0.05     # relative reduction of churn attributed to the intervention
    intervention_reach: float = 0.30            # share of subscribers targeted
    intervention_cost_per_user_month: float = 0.05
    horizon_months: int = 12
    gross_margin: float = 0.55
    uncertainty_rel: float = 0.5                # +/- band on the retention improvement for sensitivity


def business_scenario(a: ScenarioAssumptions) -> dict:
    """Transparent scenario arithmetic. All outputs are SIMULATED from the stated assumptions."""
    if not (0 <= a.baseline_monthly_churn < 1 and 0 <= a.intervention_reach <= 1):
        raise ValueError("rates must be in [0, 1)")

    def run(improvement: float) -> dict:
        months = np.arange(1, a.horizon_months + 1)
        churn_t = a.baseline_monthly_churn * (1 - improvement * a.intervention_reach)
        base_subs = a.n_subscribers * (1 - a.baseline_monthly_churn) ** months
        new_subs = a.n_subscribers * (1 - churn_t) ** months
        retained = new_subs - base_subs
        revenue = float((retained * a.monthly_price).sum())
        cost = float((new_subs * a.intervention_reach * a.intervention_cost_per_user_month).sum())
        return {
            "retained_subscribers_end": float(retained[-1]),
            "incremental_revenue": revenue,
            "incremental_gross_profit": revenue * a.gross_margin,
            "intervention_cost": cost,
            "net_impact": revenue * a.gross_margin - cost,
            "monthly": [{"month": int(m), "retained": float(r), "baseline_subscribers": float(b)}
                        for m, r, b in zip(months, retained, base_subs)],
        }

    central = run(a.retention_improvement_rel)
    low = run(a.retention_improvement_rel * (1 - a.uncertainty_rel))
    high = run(a.retention_improvement_rel * (1 + a.uncertainty_rel))
    breakeven = None
    lo_i, hi_i = 0.0, 1.0
    if run(hi_i)["net_impact"] > 0:
        for _ in range(40):
            mid = (lo_i + hi_i) / 2
            if run(mid)["net_impact"] > 0:
                hi_i = mid
            else:
                lo_i = mid
        breakeven = hi_i
    return {
        "evidence_type": "simulated",
        "assumptions": asdict(a),
        "central": central,
        "sensitivity": {"low": {k: v for k, v in low.items() if k != "monthly"},
                        "high": {k: v for k, v in high.items() if k != "monthly"}},
        "breakeven_retention_improvement_rel": breakeven,
        "disclaimer": "Hypothetical scenario computed from editable assumptions. Not company financial data and not a forecast.",
    }

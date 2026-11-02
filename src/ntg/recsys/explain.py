"""Grounded recommendation explanations.

Every reason is derived from a model quantity that actually influenced the
item's score — never from free-text generation:

* ``because_you_watched``: history items with the largest item-kNN cosine to the
  recommended item (the same similarity matrix used by the itemknn channel).
* ``retrieved_by``: channels whose top-N contained the item (provenance bits).
* ``taste_match``: overlap between the item's genres and the viewer's TasteTwin
  long-term / emerging genres, with the shares.
* ``ranker_drivers``: the largest positive TreeSHAP contributions from the
  learned ranker for this (viewer, item) pair.
"""
from __future__ import annotations

import numpy as np

_FEATURE_TEXT = {
    "tt_long_match": "matches your long-term taste profile",
    "tt_short_match": "matches what you've been watching recently",
    "tt_momentum_match": "fits an interest that's been growing lately",
    "item_log_pop": "widely watched",
    "item_recent_pop": "popular right now",
    "n_channels": "surfaced by several independent signals",
    "franchise_seen": "continues a franchise you've watched",
    "fatigue_genre_saturation": "genre saturation in your recent viewing",
    "tt_exploration": "your exploration willingness",
}


GROUNDING = "All reasons are computed from model internals (item-kNN similarity, TasteTwin state, TreeSHAP); none are generated text."


def channel_names(provenance: int, channels: list[str]) -> list[str]:
    return [c for b, c in enumerate(channels) if provenance >> b & 1]


def ranker_drivers(contrib: np.ndarray | None, feature_names: list[str], top: int = 4) -> list[dict]:
    if contrib is None:
        return []
    vals = np.asarray(contrib)[:-1]  # last column is the bias term
    return [{"feature": feature_names[j], "contribution": float(vals[j]),
             "text": _FEATURE_TEXT.get(feature_names[j], feature_names[j].replace("_", " "))}
            for j in np.argsort(-vals)[:top] if vals[j] > 0]


def explain(u: int, i: int, *, cs, provenance: int, contrib: np.ndarray | None, feature_names: list[str],
            hidden_genres: set[str] | None = None) -> dict:
    d = cs.data
    reasons: list[dict] = []
    # 1) anchors from item-kNN
    anchors = []
    if "itemknn" in cs.models:
        S = cs.models["itemknn"].S
        hist = cs.X[u].indices
        if hist.size:
            col = np.asarray(S[hist, i].todense()).ravel()
            for j in np.argsort(-col)[:3]:
                if col[j] > 0:
                    anchors.append({"item_index": int(hist[j]), "title": d.item_title(int(hist[j])), "similarity": float(col[j])})
    if anchors:
        reasons.append({"type": "because_you_watched", "text": "Because you watched " + ", ".join(a["title"] for a in anchors[:2]),
                        "evidence": anchors})
    # 2) taste match
    tt = cs.tt
    genres = list(d.items.at[i, "genres"])
    ls = dict(zip(tt.genres, tt.long_share([u])[0]))
    mom = dict(zip(tt.genres, tt.momentum([u])[0]))
    matched = sorted([(g, ls.get(g, 0.0)) for g in genres if ls.get(g, 0.0) > 0.08], key=lambda x: -x[1])
    if matched:
        reasons.append({"type": "taste_match", "text": f"{matched[0][0]} is {matched[0][1]:.0%} of your long-term viewing",
                        "evidence": [{"genre": g, "long_share": float(s)} for g, s in matched]})
    rising = [(g, mom[g]) for g in genres if mom.get(g, 0.0) > tt.cfg.emerging_threshold]
    if rising:
        g, m = max(rising, key=lambda x: x[1])
        reasons.append({"type": "emerging_interest", "text": f"Your interest in {g} has been rising (+{m:.0%} recently)",
                        "evidence": [{"genre": g, "momentum": float(m)}]})
    if hidden_genres and set(genres) & hidden_genres:
        hg = sorted(set(genres) & hidden_genres)
        reasons.append({"type": "hidden_interest", "text": f"A discovery pick: viewers like you enjoy {', '.join(hg)} more than your history shows",
                        "evidence": [{"genre": g} for g in hg]})
    drivers = ranker_drivers(contrib, feature_names)
    chans = channel_names(provenance, list(cs.models))
    return {
        "reasons": reasons,
        "retrieved_by": chans,
        "ranker_drivers": drivers,
        "grounding": GROUNDING,
    }

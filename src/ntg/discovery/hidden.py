"""Hidden-interest discovery.

For each viewer we look for genres that are *under-represented in their own
history* but *supported by two independent signals*:

1. Taste neighbours: the 50 most similar viewers in iALS embedding space and
   their long-term genre shares.
2. Collaborative mass: the genre distribution of the viewer's top-200 unseen
   items under the EASE item-item model.

hidden_score(g) = 0.5 * neighbour_share(g) + 0.5 * cf_mass(g) - own_share(g),
reported only where own_share(g) < ``max_own_share``. Evidence numbers are
returned with every suggestion.

Validation (see :func:`evaluate_hidden_interests`): on the temporal holdout, do
suggested genres anticipate genres the viewer newly adopts (own share < 5%
before the cutoff, >= 15% of holdout viewing)? Compared with a non-personal
baseline that suggests the globally most popular genres the viewer hasn't
explored.
"""
from __future__ import annotations

import numpy as np

from ntg.recsys.models import topk_rows


def hidden_interests(cs, users: np.ndarray, top: int = 3, max_own_share: float = 0.05,
                     n_neighbors: int = 50, weights: tuple[float, float, float] = (0.0, 0.5, 0.5),
                     min_score: float = 0.0) -> list[list[dict]]:
    tt = cs.tt
    users = np.asarray(users)
    own = tt.long_share(users)
    U = cs.models["ials"].U if "ials" in cs.models else None
    active = np.flatnonzero(tt.n_events >= 5)
    if U is not None and len(active) > n_neighbors:
        Ua = U[active] / (np.linalg.norm(U[active], axis=1, keepdims=True) + 1e-9)
        Uq = U[users] / (np.linalg.norm(U[users], axis=1, keepdims=True) + 1e-9)
        sims = Uq @ Ua.T
        sims[np.isin(active, users)[None, :] & (active[None, :] == users[:, None])] = -np.inf
        nb, _ = topk_rows(sims, n_neighbors)
        neigh = tt.long_share(active[nb.ravel()]).reshape(len(users), n_neighbors, -1).mean(1)
    else:
        neigh = np.broadcast_to(tt.global_share, own.shape)
    model = cs.models.get("ease") or next(iter(cs.models.values()))
    s = model.score(users).astype(np.float32)
    seen = cs.X[users]
    r, c = seen.nonzero()
    s[r, c] = -np.inf
    top_items, top_scores = topk_rows(s, 200)
    w = np.maximum(top_scores - np.nanmin(np.where(np.isfinite(top_scores), top_scores, np.nan), axis=1, keepdims=True), 0)
    cf = np.einsum("uk,ukg->ug", w, cs.Gn[top_items])
    cf = cf / np.maximum(cf.sum(1, keepdims=True), 1e-9)
    wn, wc, wg = weights
    score = wn * neigh + wc * cf + wg * tt.global_share[None, :] - own
    out = []
    for row in range(len(users)):
        sugg = []
        for g in np.argsort(-score[row]):
            if own[row, g] >= max_own_share or score[row, g] <= min_score:
                continue
            gname = tt.genres[g]
            in_g = [int(x) for x in top_items[row] if cs.data.genre_matrix[x, g] > 0][:5]
            sugg.append({
                "genre": gname,
                "score": float(score[row, g]),
                "evidence": {
                    "your_share": float(own[row, g]),
                    "similar_viewers_share": float(neigh[row, g]),
                    "collaborative_signal_share": float(cf[row, g]),
                },
                "example_item_indices": in_g,
                "explanation": (f"{gname} is {own[row, g]:.0%} of your viewing, but {neigh[row, g]:.0%} for the 50 viewers "
                                f"most like you, and {cf[row, g]:.0%} of what our model predicts you'd enjoy."),
            })
            if len(sugg) >= top:
                break
        out.append(sugg)
    return out


def evaluate_hidden_interests(cs, holdout, min_new_share: float = 0.15, max_own: float = 0.05,
                              max_users: int = 3000, seed: int = 0, weights=(0.0, 0.5, 0.5)) -> dict:
    tt = cs.tt
    h = holdout
    nh = h.groupby("u").size()
    users = nh[nh >= 3].index.values
    users = users[tt.n_events[users] >= 5]
    if len(users) > max_users:
        users = np.random.default_rng(seed).choice(users, max_users, replace=False)
    own = tt.long_share(users)
    S = np.zeros((cs.data.n_users, len(tt.genres)))
    np.add.at(S, h["u"].to_numpy(), cs.Gn[h["i"].to_numpy()])
    fut = S[users] / np.maximum(S[users].sum(1, keepdims=True), 1e-9)
    adopted = (own < max_own) & (fut >= min_new_share)
    has = adopted.any(1)
    sugg = hidden_interests(cs, users[has], top=3, max_own_share=max_own, weights=weights)
    gi = {g: k for k, g in enumerate(tt.genres)}
    hits = np.array([any(adopted[np.flatnonzero(has)[n], gi[s["genre"]]] for s in row) for n, row in enumerate(sugg)])
    prec = np.array([np.mean([adopted[np.flatnonzero(has)[n], gi[s["genre"]]] for s in row]) if row else 0.0
                     for n, row in enumerate(sugg)])
    # baseline: the 3 globally most popular genres the user has < max_own share of
    base_hits, base_prec = [], []
    order = np.argsort(-tt.global_share)
    for u_row in np.flatnonzero(has):
        cand = [g for g in order if own[u_row, g] < max_own][:3]
        base_hits.append(any(adopted[u_row, g] for g in cand))
        base_prec.append(np.mean([adopted[u_row, g] for g in cand]) if cand else 0.0)
    return {
        "protocol": f"users with >=3 holdout and >=5 fit events who newly adopt a genre (own share <{max_own:.0%} before, >={min_new_share:.0%} after)",
        "n_users_evaluated": int(len(users)),
        "n_users_adopting_new_genre": int(has.sum()),
        "hit_rate@3": float(hits.mean()) if len(hits) else float("nan"),
        "precision@3": float(prec.mean()) if len(prec) else float("nan"),
        "baseline_global_popular_hit_rate@3": float(np.mean(base_hits)) if base_hits else float("nan"),
        "baseline_global_popular_precision@3": float(np.mean(base_prec)) if base_prec else float("nan"),
    }

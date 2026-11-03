"""Offline evaluation: accuracy + beyond-accuracy metrics with bootstrap CIs.

Metric definitions (per user, then averaged over evaluated users):

* ``recall@K``     |rec_K ∩ rel| / min(|rel|, K)   (capped recall, standard for top-K)
* ``precision@K``  |rec_K ∩ rel| / K
* ``hit_rate@K``   1[|rec_K ∩ rel| > 0]
* ``ndcg@K``       binary-gain DCG / ideal DCG
* ``map@K``        mean of precision at each hit position / min(|rel|, K)
* ``coverage@K``   fraction of catalogue appearing in any user's top-K
* ``gini@K``       Gini coefficient of item exposure in top-K lists (0 = equal)
* ``ild@K``        intra-list diversity: 1 - mean pairwise cosine of genre vectors
* ``novelty@K``    mean self-information -log2(pop_share) of recommended items
* ``long_tail@K``  share of recommended items outside the popularity head (top 20% items by fit-window interactions)
* ``franchise_dup@K`` share of lists containing >1 item of the same franchise

Relevant items = holdout interactions of the user (``min_rating`` optionally
restricts to liked items). Items already seen in the fit window are excluded
from recommendations and from relevance.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ntg.datasets.matrix import InteractionData


@dataclass(frozen=True)
class EvalConfig:
    ks: tuple[int, ...] = (10, 20)
    min_rating: float | None = None
    max_users: int | None = 3000
    seed: int = 0
    bootstrap: int = 300
    batch: int = 256


def holdout_relevance(data: InteractionData, fit_df: pd.DataFrame, holdout: pd.DataFrame,
                      min_rating: float | None = None) -> sp.csr_matrix:
    h = holdout if min_rating is None else holdout[holdout["rating"] >= min_rating]
    rel = data.csr(h)
    seen = data.csr(fit_df)
    rel = rel - rel.multiply(seen)  # an item already consumed cannot be a new hit
    rel.eliminate_zeros()
    return sp.csr_matrix(rel)


def eval_users(data: InteractionData, fit_df: pd.DataFrame, rel: sp.csr_matrix, cfg: EvalConfig,
               segment: str = "warm") -> np.ndarray:
    has_rel = np.asarray(rel.sum(1)).ravel() > 0
    n_fit = np.bincount(fit_df["u"].to_numpy(), minlength=data.n_users)
    if segment == "warm":
        mask = has_rel & (n_fit >= 5)
    elif segment == "cold":
        mask = has_rel & (n_fit >= 1) & (n_fit < 5)
    elif segment == "new":
        mask = has_rel & (n_fit == 0)
    else:
        raise ValueError(segment)
    users = np.flatnonzero(mask)
    if cfg.max_users and len(users) > cfg.max_users:
        users = np.sort(np.random.default_rng(cfg.seed).choice(users, cfg.max_users, replace=False))
    return users


def _per_user_metrics(rec: np.ndarray, rel: sp.csr_matrix, users: np.ndarray, k: int) -> dict[str, np.ndarray]:
    R = rel[users]
    hits = np.zeros((len(users), k), dtype=np.float32)
    for r in range(len(users)):
        row = R.indices[R.indptr[r]:R.indptr[r + 1]]
        hits[r] = np.isin(rec[r, :k], row)
    n_rel = np.diff(R.indptr).astype(np.float32)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits * disc).sum(1)
    ideal = np.array([disc[: int(min(n, k))].sum() for n in n_rel])
    cum = np.cumsum(hits, 1)
    prec_at = cum / np.arange(1, k + 1)
    denom = np.minimum(n_rel, k)
    nh = hits.sum(1)
    return {
        f"recall@{k}": nh / denom,
        f"precision@{k}": nh / k,
        f"hit_rate@{k}": (nh > 0).astype(np.float32),
        f"ndcg@{k}": dcg / np.maximum(ideal, 1e-9),
        f"map@{k}": (prec_at * hits).sum(1) / denom,
    }


def beyond_accuracy(rec: np.ndarray, data: InteractionData, pop: np.ndarray, k: int) -> dict[str, float]:
    rk = rec[:, :k]
    flat = rk.ravel()
    exposure = np.bincount(flat, minlength=data.n_items).astype(np.float64)
    srt = np.sort(exposure)
    n = srt.size
    gini = float((np.arange(1, n + 1) * 2 - n - 1) @ srt / (n * srt.sum())) if srt.sum() else 0.0
    share = (pop + 1) / (pop.sum() + len(pop))
    novelty = float(np.mean(-np.log2(share[flat])))
    head_cut = np.quantile(pop, 0.8)
    long_tail = float(np.mean(pop[flat] < head_cut))
    G = data.genre_matrix
    Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
    V = Gn[rk]  # (U, k, g)
    sims = np.einsum("ukg,ulg->ukl", V, V)
    iu = np.triu_indices(k, 1)
    ild = float(1.0 - sims[:, iu[0], iu[1]].mean())
    fr = data.franchise_codes[rk]
    dup = np.array([len(np.unique(r)) < len(r) for r in fr])
    return {
        f"coverage@{k}": float((exposure > 0).mean()),
        f"gini@{k}": gini,
        f"ild@{k}": ild,
        f"novelty@{k}": novelty,
        f"long_tail@{k}": long_tail,
        f"franchise_dup@{k}": float(dup.mean()),
    }


def bootstrap_ci(x: np.ndarray, n: int, seed: int = 0) -> tuple[float, float]:
    if n <= 0 or len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate_recommender(recommend_fn, data: InteractionData, fit_df: pd.DataFrame, holdout: pd.DataFrame,
                         cfg: EvalConfig = EvalConfig(), segment: str = "warm",
                         users: np.ndarray | None = None) -> dict:
    """``recommend_fn(users, k) -> (B, k) item indices`` (seen items already excluded)."""
    rel = holdout_relevance(data, fit_df, holdout, cfg.min_rating)
    if users is None:
        users = eval_users(data, fit_df, rel, cfg, segment)
    if len(users) == 0:
        return {"n_users": 0, "segment": segment}
    kmax = max(cfg.ks)
    recs = []
    t0 = time.perf_counter()
    for s in range(0, len(users), cfg.batch):
        recs.append(recommend_fn(users[s:s + cfg.batch], kmax))
    elapsed = time.perf_counter() - t0
    rec = np.vstack(recs)
    pop = data.popularity(fit_df)
    out: dict = {
        "segment": segment,
        "n_users": int(len(users)),
        "n_relevant": int(rel[users].nnz),
        "latency_ms_per_user": 1000 * elapsed / len(users),
        "throughput_users_per_s": len(users) / max(elapsed, 1e-9),
        "metrics": {},
        "ci95": {},
    }
    for k in cfg.ks:
        pm = _per_user_metrics(rec, rel, users, k)
        for name, arr in pm.items():
            out["metrics"][name] = float(arr.mean())
            out["ci95"][name] = bootstrap_ci(arr, cfg.bootstrap, cfg.seed)
        out["metrics"].update(beyond_accuracy(rec, data, pop, k))
    return out

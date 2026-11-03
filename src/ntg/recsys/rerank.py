"""Controllable re-ranking: diversity, novelty, franchise/genre caps, exploration level.

Greedy slate construction. At each position the item maximising

    (1 - λ_div - λ_unexp) * relevance
    + λ_div   * (1 - max genre-similarity to already selected items)
    + λ_nov   * novelty            (normalised -log popularity)
    + λ_unexp * (1 - profile match) (distance from the viewer's TasteTwin profile)

is chosen, subject to: at most ``franchise_cap`` items per franchise, at most
``genre_cap`` share of the slate sharing a primary genre, (for the exploratory
modes) a relevance floor so that "unexpected" never means "irrelevant", and a
``tail_quota`` share of slots reserved for titles outside the popularity head. All terms are normalised to [0, 1] within the candidate set.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EXPLORATION_LEVELS = ("familiar", "balanced", "exploratory", "unexpected")


@dataclass(frozen=True)
class RerankConfig:
    lambda_div: float = 0.15
    lambda_nov: float = 0.05
    lambda_unexp: float = 0.0
    franchise_cap: int = 1
    genre_cap: float = 0.5
    relevance_floor_quantile: float = 0.0
    tail_quota: float = 0.0  # minimum share of the slate from outside the popularity head

    @staticmethod
    def for_level(level: str) -> "RerankConfig":
        return {
            "familiar": RerankConfig(0.0, 0.0, 0.0, franchise_cap=2, genre_cap=1.0),
            "balanced": RerankConfig(0.15, 0.05, 0.0, franchise_cap=1, genre_cap=0.5),
            "exploratory": RerankConfig(0.30, 0.15, 0.20, franchise_cap=1, genre_cap=0.4, relevance_floor_quantile=0.5,
                                        tail_quota=0.3),
            "unexpected": RerankConfig(0.25, 0.20, 0.45, franchise_cap=1, genre_cap=0.3, relevance_floor_quantile=0.7,
                                       tail_quota=0.5),
        }[level]


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = np.min(x), np.max(x)
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def rerank(items: np.ndarray, scores: np.ndarray, k: int, *, genre_matrix: np.ndarray, franchise: np.ndarray,
           log_pop: np.ndarray, profile: np.ndarray | None, cfg: RerankConfig) -> list[int]:
    """Return indices into ``items`` forming the final slate (length <= k)."""
    items = np.asarray(items)
    if len(items) == 0:
        return []
    rel = _minmax(np.asarray(scores, dtype=np.float64))
    G = genre_matrix[items]
    Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
    nov = _minmax(-log_pop[items])
    if profile is not None and cfg.lambda_unexp > 0:
        p = profile / (np.linalg.norm(profile) + 1e-9)
        unexp = 1 - _minmax(Gn @ p)
    else:
        unexp = np.zeros(len(items))
    eligible = np.ones(len(items), dtype=bool)
    if cfg.relevance_floor_quantile > 0:
        # exploratory modes still draw from the most relevant part of the candidate pool
        eligible = rel >= np.quantile(rel, cfg.relevance_floor_quantile) if len(rel) > k * 2 else eligible
    primary = np.argmax(G, axis=1)
    # long-tail quota: titles outside the popularity head (top 20% of the catalogue by log-popularity)
    nz = log_pop[log_pop > 0]
    head_cut = np.quantile(nz, 0.8) if len(nz) else np.inf
    is_tail = log_pop[items] < head_cut
    need_tail = int(np.ceil(cfg.tail_quota * min(k, len(items)))) if cfg.tail_quota > 0 else 0
    tail_ok = eligible & is_tail
    if need_tail and not tail_ok.any():
        tail_ok = is_tail  # no tail item clears the relevance floor: take the most relevant tail items
    w_rel = max(0.0, 1.0 - cfg.lambda_div - cfg.lambda_unexp)
    chosen: list[int] = []
    fr_count: dict[int, int] = {}
    g_count: dict[int, int] = {}
    max_sim = np.zeros(len(items))
    for _ in range(min(k, len(items))):
        obj = w_rel * rel + cfg.lambda_div * (1 - max_sim) + cfg.lambda_nov * nov + cfg.lambda_unexp * unexp
        n_tail = int(is_tail[chosen].sum()) if chosen else 0
        if need_tail and tail_ok.any() and (min(k, len(items)) - len(chosen)) <= (need_tail - n_tail):
            obj = np.where(tail_ok, obj, -np.inf)  # remaining slots are reserved for the long tail
        else:
            obj = np.where(eligible, obj, -np.inf)
        if chosen:
            obj[chosen] = -np.inf
        for cand in np.argsort(-obj):
            if not np.isfinite(obj[cand]):
                cand = None
                break
            f, g = int(franchise[items[cand]]), int(primary[cand])
            if fr_count.get(f, 0) >= cfg.franchise_cap:
                continue
            if (g_count.get(g, 0) + 1) / k > cfg.genre_cap and len(chosen) >= 2:
                continue
            break
        else:
            cand = None
        if cand is None:
            # constraints exhausted: relax caps rather than return a short slate
            rest = [j for j in np.argsort(-rel) if j not in chosen and eligible[j]] or [j for j in np.argsort(-rel) if j not in chosen]
            if not rest:
                break
            cand = rest[0]
        chosen.append(int(cand))
        f, g = int(franchise[items[cand]]), int(primary[cand])
        fr_count[f] = fr_count.get(f, 0) + 1
        g_count[g] = g_count.get(g, 0) + 1
        max_sim = np.maximum(max_sim, Gn @ Gn[cand])
    return chosen

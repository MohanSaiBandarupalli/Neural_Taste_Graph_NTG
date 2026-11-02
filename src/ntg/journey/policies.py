"""Slate policies for multi-session journeys.

Policies only see: the candidate pool, the past-only relevance scores, the
viewer's TasteTwin (past), and in-journey accept/reject feedback. They never
see the simulator's oracle preferences or fatigue parameters.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _norm(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


class Greedy:
    """Top-k by relevance every session (re-shows unaccepted items)."""

    name = "greedy_relevance"

    def __init__(self, world, row, k=5):
        self.rel, self.k = world.rel[row], k

    def select(self, t, consumed):
        s = np.where(consumed, -np.inf, self.rel)
        return np.argsort(-s)[: self.k]

    def feedback(self, t, slate, accepted):
        pass


class GreedyFresh(Greedy):
    """Strong baseline: greedy relevance, but never re-show an item from the last W sessions."""

    name = "greedy_fresh"

    def __init__(self, world, row, k=5, window=3):
        super().__init__(world, row, k)
        self.window = window
        self.last: dict[int, int] = {}

    def select(self, t, consumed):
        s = np.where(consumed, -np.inf, self.rel).copy()
        for j, ts in self.last.items():
            if t - ts <= self.window:
                s[j] = -np.inf
        return np.argsort(-s)[: self.k]

    def feedback(self, t, slate, accepted):
        for j in slate:
            self.last[int(j)] = t


class MMRDiversity(Greedy):
    """Static diversity re-ranking (genre MMR), no memory across sessions."""

    name = "mmr_diversity"

    def __init__(self, world, row, k=5, lam=0.3):
        super().__init__(world, row, k)
        self.g, self.lam = world.primary[row], lam

    def select(self, t, consumed):
        rel = _norm(np.where(consumed, self.rel.min(), self.rel))
        chosen, used = [], set()
        for _ in range(self.k):
            obj = (1 - self.lam) * rel + self.lam * np.array([g not in used for g in self.g], float)
            obj[consumed] = -np.inf
            obj[chosen] = -np.inf
            j = int(np.argmax(obj))
            chosen.append(j)
            used.add(self.g[j])
        return np.array(chosen)


class EpsilonGreedy(Greedy):
    name = "epsilon_greedy"

    def __init__(self, world, row, k=5, eps=0.2, seed=0):
        super().__init__(world, row, k)
        self.eps = eps
        self.rng = np.random.default_rng(seed + row)

    def select(self, t, consumed):
        base = list(super().select(t, consumed))
        pool = np.flatnonzero(~consumed)
        for pos in range(len(base)):
            if self.rng.random() < self.eps:
                base[pos] = int(self.rng.choice(pool))
        return np.array(list(dict.fromkeys(base)))


class GenreThompson(Greedy):
    """Thompson sampling over genre arms (Beta posterior on accept rate, prior from TasteTwin)."""

    name = "genre_thompson"

    def __init__(self, world, row, k=5, prior=4.0, seed=0):
        super().__init__(world, row, k)
        self.g = world.primary[row]
        share = world.own_share[row]
        self.a = 0.2 + prior * share * len(share) * 0.1
        self.b = np.full_like(self.a, 1.0 + prior * 0.5)
        self.rng = np.random.default_rng(seed + row)
        self.reln = _norm(self.rel)

    def select(self, t, consumed):
        theta = self.rng.beta(self.a, self.b)
        obj = 0.5 * self.reln + 0.5 * theta[self.g]
        obj = np.where(consumed, -np.inf, obj)
        return np.argsort(-obj)[: self.k]

    def feedback(self, t, slate, accepted):
        for j in slate:
            g = self.g[j]
            if accepted is not None and j == accepted:
                self.a[g] += 1
            else:
                self.b[g] += 0.25  # examined-not-chosen is weak negative evidence


@dataclass(frozen=True)
class OrchestratorConfig:
    lambda_fatigue: float = 0.8
    lambda_diversity: float = 0.1
    lambda_explore: float = 0.1
    memory: float = 0.7
    repeat_window: int = 3
    saturation_threshold: float = 0.35
    prior: float = 2.0
    adaptive: bool = True


class JourneyOrchestrator(Greedy):
    """Adaptive fatigue-aware journey policy.

    Objective for candidate j (genre g) at session t:

        rel(j)
        - λ_f · F_u · saturation_t(g)     saturation from what THIS viewer consumed in-journey
        + λ_d · F_u · 1[g not yet in slate]
        + λ_e · F_u · UCB(g)              optimistic bonus for under-explored genres
        hard constraint: no item shown in the last W sessions and not accepted.

    ``F_u`` is the policy's *learned* belief that this viewer fatigues:

        F_u = clip((p_fresh - p_saturated) / p_fresh, 0, 1)

    where p_fresh / p_saturated are Beta-posterior accept rates of shown items
    from unsaturated vs saturated genres (prior centred on the TasteTwin
    exploration willingness). A viewer who keeps happily accepting the same
    genre drives F_u -> 0 and the policy converges to fresh greedy relevance;
    a viewer who rejects saturated genres drives F_u up. The policy never sees
    the simulator's fatigue parameter.
    """

    name = "journey_orchestrator"

    def __init__(self, world, row, k=5, cfg: OrchestratorConfig = OrchestratorConfig()):
        super().__init__(world, row, k)
        self.cfg = cfg
        self.g = world.primary[row]
        n_g = world.own_share.shape[1]
        self.mem = np.zeros(n_g)
        self.shows = np.zeros(n_g)
        self.accepts = np.zeros(n_g)
        self.last_shown: dict[int, int] = {}
        self.reln = _norm(self.rel)
        f0 = float(np.clip(world.exploration[row], 0.05, 0.95))
        # Beta pseudo-counts: [accepted, shown] for fresh and saturated genres
        self.fresh = np.array([cfg.prior * 0.3, cfg.prior])
        self.satc = np.array([cfg.prior * 0.3 * (1 - f0), cfg.prior])

    def fatigue_belief(self) -> float:
        if not self.cfg.adaptive:
            return 1.0
        pf = self.fresh[0] / self.fresh[1]
        ps = self.satc[0] / self.satc[1]
        return float(np.clip((pf - ps) / max(pf, 1e-6), 0.0, 1.0))

    def _sat(self):
        return self.mem / max(self.mem.sum(), 1e-9) if self.mem.sum() else np.zeros_like(self.mem)

    def select(self, t, consumed):
        c = self.cfg
        F = self.fatigue_belief()
        sat = self._sat()
        ucb = np.sqrt(np.log(t + 2) / (1 + self.shows)) * (self.accepts + 1) / (self.shows + 2)
        base = self.reln - c.lambda_fatigue * F * sat[self.g] + c.lambda_explore * F * ucb[self.g]
        base = np.where(consumed, -np.inf, base)
        for j, ts in self.last_shown.items():
            if t - ts <= c.repeat_window:
                base[j] = -np.inf
        chosen, used = [], set()
        for _ in range(self.k):
            obj = base + c.lambda_diversity * F * np.array([g not in used for g in self.g], float)
            obj[chosen] = -np.inf
            j = int(np.argmax(obj))
            if not np.isfinite(obj[j]):
                break
            chosen.append(j)
            used.add(self.g[j])
        return np.array(chosen)

    def feedback(self, t, slate, accepted):
        sat = self._sat()
        for j in slate:
            g = self.g[j]
            bucket = self.satc if sat[g] >= self.cfg.saturation_threshold else self.fresh
            bucket[1] += 1
            if accepted is not None and j == accepted:
                bucket[0] += 1
            self.shows[g] += 1
            if accepted is None or j != accepted:
                self.last_shown[int(j)] = t
        self.mem *= self.cfg.memory
        if accepted is not None:
            self.mem[self.g[accepted]] += 1
            self.accepts[self.g[accepted]] += 1


POLICIES = {
    "greedy_relevance": Greedy,
    "greedy_fresh": GreedyFresh,
    "mmr_diversity": MMRDiversity,
    "epsilon_greedy": EpsilonGreedy,
    "genre_thompson": GenreThompson,
    "journey_orchestrator": JourneyOrchestrator,
}

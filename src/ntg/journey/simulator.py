"""Multi-session viewing-journey simulator (offline, assumption-explicit).

IMPORTANT: results from this module are *simulated*. They quantify how
policies behave under the stated user model; they are not causal estimates of
real-world impact. Every assumption is a field of :class:`WorldConfig` and the
counterfactual module sweeps them.

World model
-----------
* Preference: ``p_like(u, i)`` comes from an *oracle* ensemble (EASE + RP3beta)
  fitted on ALL interactions, including the future window that policies never
  see, converted to a probability with a per-user rank transform. Policies only
  see models fitted on the past.
* Fatigue: the viewer keeps a decayed memory of consumed primary genres. The
  acceptance probability of an item is multiplied by
  ``(1 - fatigue_sensitivity * saturation(genre))``.
* Repetition: an item shown in the last ``repeat_window`` steps and not
  accepted has its acceptance probability multiplied by ``repeat_penalty``.
* Discovery: an accepted item in a genre the viewer had < 5% of before gives a
  satisfaction bonus ``discovery_bonus * exploration_willingness``.
* Leaving: after each session the viewer leaves for good with probability
  ``leave_base + leave_fatigue * saturation_of_last_accept + leave_reject * 1[no accept]``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ntg.recsys.models import EASE, RP3Beta, topk_rows


@dataclass(frozen=True)
class WorldConfig:
    steps: int = 25
    slate_size: int = 5
    fatigue_sensitivity: float = 1.0
    fatigue_memory: float = 0.7          # per-step decay of the genre-consumption memory
    repeat_window: int = 3
    repeat_penalty: float = 0.4
    discovery_bonus: float = 0.5
    leave_base: float = 0.02
    leave_fatigue: float = 0.15
    leave_reject: float = 0.10
    like_scale: float = 0.6              # max acceptance probability of the best item
    position_decay: float = 0.85         # examination probability multiplier per slot
    seed: int = 0


@dataclass
class JourneyWorld:
    """Pre-computed per-user candidate pools, policy-visible scores and oracle preferences."""

    cs: object                      # ChannelSet fitted on the past (policy-visible)
    users: np.ndarray
    cand: np.ndarray                # (U, C) item indices
    rel: np.ndarray                 # (U, C) policy-visible relevance (past-only model)
    p_like: np.ndarray              # (U, C) oracle acceptance propensity (hidden from policies)
    primary: np.ndarray             # (U, C) primary genre of each candidate
    own_share: np.ndarray           # (U, n_genres) TasteTwin long-term share (past)
    exploration: np.ndarray         # (U,) TasteTwin exploration willingness (past)
    info: dict = field(default_factory=dict)


def build_world(data, cs, full_df, users: np.ndarray, n_candidates: int = 300, rel_model: str = "ease") -> JourneyWorld:
    """``cs`` is fitted on past data; ``full_df`` (all data) is used only for the oracle."""
    users = np.asarray(users)
    base = cs.models[rel_model]
    s = base.score(users).astype(np.float32)
    seen = cs.X[users]
    r, c = seen.nonzero()
    s[r, c] = -np.inf
    cand, rel = topk_rows(s, n_candidates)
    oracle_a = EASE().fit(data, full_df)
    oracle_b = RP3Beta().fit(data, full_df)
    oa = np.take_along_axis(oracle_a.score(users), cand, 1)
    ob = np.take_along_axis(oracle_b.score(users), cand, 1)

    def pct(x):  # per-user percentile rank in [0, 1]
        return np.argsort(np.argsort(x, 1), 1) / max(x.shape[1] - 1, 1)

    p_like = (0.5 * pct(oa) + 0.5 * pct(ob)) ** 3  # convex: only the oracle's top items are likely accepts
    primary = np.argmax(data.genre_matrix[cand], axis=2)
    return JourneyWorld(cs=cs, users=users, cand=cand, rel=rel, p_like=p_like, primary=primary,
                        own_share=cs.tt.long_share(users), exploration=cs.tt.exploration[users],
                        info={"oracle": "EASE+RP3beta fitted on all data (incl. future)", "policy_model": rel_model,
                              "n_candidates": int(n_candidates)})


def simulate(world: JourneyWorld, policy_factory, cfg: WorldConfig = WorldConfig()) -> dict:
    rng = np.random.default_rng(cfg.seed)
    n_g = world.own_share.shape[1]
    per_user = []
    first_accept = []
    for row in range(len(world.users)):
        policy = policy_factory(world, row)
        mem = np.zeros(n_g)
        consumed = np.zeros(world.cand.shape[1], dtype=bool)
        last_shown: dict[int, int] = {}
        accepts, satisfaction, steps, discovered, repeats, prev_g = 0, 0.0, 0, set(), 0, None
        shown_total, unique_shown = 0, set()
        explore_w = world.exploration[row]
        for t in range(cfg.steps):
            steps += 1
            slate = policy.select(t, consumed)
            shown_total += len(slate)
            unique_shown.update(int(x) for x in slate)
            sat = mem / max(mem.sum(), 1e-9) if mem.sum() > 0 else np.zeros(n_g)
            accepted = None
            exam = 1.0
            for pos, j in enumerate(slate):
                if consumed[j]:
                    continue
                g = world.primary[row, j]
                p = cfg.like_scale * world.p_like[row, j] * max(0.0, 1 - cfg.fatigue_sensitivity * sat[g])
                if j in last_shown and t - last_shown[j] <= cfg.repeat_window:
                    p *= cfg.repeat_penalty
                if rng.random() < exam * p:
                    accepted = j
                    break
                exam *= cfg.position_decay
            for j in slate:
                last_shown[int(j)] = t
            policy.feedback(t, slate, accepted)
            if accepted is not None:
                g = world.primary[row, accepted]
                consumed[accepted] = True
                accepts += 1
                new_genre = world.own_share[row, g] < 0.05 and g not in discovered
                if world.own_share[row, g] < 0.05:
                    discovered.add(int(g))
                satisfaction += world.p_like[row, accepted] + (cfg.discovery_bonus * explore_w if new_genre else 0.0)
                repeats += int(prev_g == g)
                prev_g = g
                fat = sat[g]
                mem = cfg.fatigue_memory * mem
                mem[g] += 1.0
            else:
                fat = 0.0
                mem = cfg.fatigue_memory * mem
            if t == 0:
                first_accept.append(accepted is not None)
            leave = cfg.leave_base + cfg.leave_fatigue * fat * cfg.fatigue_sensitivity + cfg.leave_reject * (accepted is None)
            if rng.random() < leave:
                break
        per_user.append((accepts, satisfaction, steps, len(discovered), repeats / max(accepts - 1, 1),
                         len(unique_shown) / max(shown_total, 1)))
    a = np.array(per_user, dtype=float)

    def ci(x):
        boot = x[rng.integers(0, len(x), (300, len(x)))].mean(1)
        return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]

    return {
        "n_users": int(len(a)),
        "immediate_accept_rate": float(np.mean(first_accept)),
        "accepts_per_journey": float(a[:, 0].mean()),
        "accepts_ci95": ci(a[:, 0]),
        "satisfaction_per_journey": float(a[:, 1].mean()),
        "satisfaction_ci95": ci(a[:, 1]),
        "sessions_before_leaving": float(a[:, 2].mean()),
        "genres_discovered": float(a[:, 3].mean()),
        "consecutive_same_genre_rate": float(a[:, 4].mean()),
        "slate_freshness": float(a[:, 5].mean()),
        "_per_user_satisfaction": a[:, 1],
    }

"""Counterfactual decision simulation + off-policy evaluation.

``compare_policies``      : all policies under one set of world assumptions.
``sensitivity``           : how the policy ranking changes as world assumptions
                            change (e.g. fatigue sensitivity 0 -> 1.5). If the
                            orchestrator only wins when fatigue matters, that is
                            reported — it is the honest answer.
``off_policy_evaluation`` : validates IPS / SNIPS estimators on simulated logs
                            whose true target-policy value is known, which is
                            the machinery needed before trusting OPE on real
                            logged data.

All outputs carry ``"evidence_type": "simulated"``.
"""
from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np

from ntg.journey.policies import POLICIES
from ntg.journey.simulator import JourneyWorld, WorldConfig, simulate


def compare_policies(world: JourneyWorld, cfg: WorldConfig = WorldConfig(), policies=None, policy_kwargs=None) -> dict:
    policies = policies or list(POLICIES)
    policy_kwargs = policy_kwargs or {}
    out = {}
    for name in policies:
        cls = POLICIES[name]
        kw = policy_kwargs.get(name, {})
        r = simulate(world, lambda w, row, cls=cls, kw=kw: cls(w, row, k=cfg.slate_size, **kw), cfg)
        out[name] = r
    for ref, key in (("greedy_relevance", "greedy"), ("greedy_fresh", "greedy_fresh")):
        base = out.get(ref)
        if base is None:
            continue
        for name, r in out.items():
            diff = r["_per_user_satisfaction"] - base["_per_user_satisfaction"]  # paired: same users, same seed
            rng = np.random.default_rng(1)
            boot = diff[rng.integers(0, len(diff), (500, len(diff)))].mean(1)
            r[f"satisfaction_lift_vs_{key}"] = float(diff.mean())
            r[f"satisfaction_lift_vs_{key}_ci95"] = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
            if key == "greedy":
                r["satisfaction_lift_ci95"] = r["satisfaction_lift_vs_greedy_ci95"]
    for r in out.values():
        r.pop("_per_user_satisfaction", None)
    return {"evidence_type": "simulated", "world": asdict(cfg), "world_model": world.info, "policies": out}


def sensitivity(world: JourneyWorld, param: str, values, base: WorldConfig = WorldConfig(),
                policies=("greedy_relevance", "greedy_fresh", "mmr_diversity", "journey_orchestrator")) -> dict:
    rows = []
    for v in values:
        cfg = replace(base, **{param: v})
        res = compare_policies(world, cfg, list(policies))["policies"]
        rows.append({param: v, **{p: {"satisfaction": res[p]["satisfaction_per_journey"],
                                      "accepts": res[p]["accepts_per_journey"],
                                      "sessions": res[p]["sessions_before_leaving"],
                                      "lift_vs_greedy": res[p].get("satisfaction_lift_vs_greedy"),
                                      "lift_ci95": res[p].get("satisfaction_lift_ci95"),
                                      "lift_vs_greedy_fresh": res[p].get("satisfaction_lift_vs_greedy_fresh"),
                                      "lift_vs_greedy_fresh_ci95": res[p].get("satisfaction_lift_vs_greedy_fresh_ci95")}
                                  for p in policies}})
    return {"evidence_type": "simulated", "parameter": param, "rows": rows}


def off_policy_evaluation(world: JourneyWorld, eps: float = 0.3, n_logs: int = 20000, seed: int = 0,
                          like_scale: float = 0.6) -> dict:
    """Single-slot, first-session OPE. Logging policy: ε-greedy over the top-20 pool.

    Target policy: deterministic greedy top-1. True value is computed exactly
    from the oracle; IPS and SNIPS are estimated from the logs alone.
    """
    rng = np.random.default_rng(seed)
    U = len(world.users)
    pool = 20
    order = np.argsort(-world.rel, axis=1)[:, :pool]
    target = order[:, 0]
    true_value = float((like_scale * world.p_like[np.arange(U), target]).mean())
    rows = rng.integers(0, U, n_logs)
    explore = rng.random(n_logs) < eps
    pick_pos = np.where(explore, rng.integers(0, pool, n_logs), 0)
    actions = order[rows, pick_pos]
    prop = np.where(pick_pos == 0, (1 - eps) + eps / pool, eps / pool)
    reward = (rng.random(n_logs) < like_scale * world.p_like[rows, actions]).astype(float)
    match = (actions == target[rows]).astype(float)
    w = match / prop
    ips = float((w * reward).mean())
    snips = float((w * reward).sum() / max(w.sum(), 1e-9))
    boot = rng.integers(0, n_logs, (300, n_logs))
    ips_ci = np.quantile((w[boot] * reward[boot]).mean(1), [0.025, 0.975]).tolist()
    snips_ci = np.quantile((w[boot] * reward[boot]).sum(1) / w[boot].sum(1), [0.025, 0.975]).tolist()
    naive = float(reward.mean())
    return {
        "evidence_type": "simulated",
        "protocol": "logging = ε-greedy over top-20 (known propensities); target = greedy top-1; first session, single slot",
        "epsilon": eps, "n_logged": n_logs,
        "true_target_value": true_value,
        "ips_estimate": ips, "ips_ci95": ips_ci,
        "snips_estimate": snips, "snips_ci95": snips_ci,
        "naive_logged_average": naive,
    }

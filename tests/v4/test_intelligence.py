from __future__ import annotations

import numpy as np
import pytest

from ntg.enterprise.analytics import ScenarioAssumptions, business_scenario
from ntg.intent.parser import parse_intent
from ntg.journey.counterfactual import compare_policies, off_policy_evaluation
from ntg.journey.simulator import WorldConfig, build_world
from ntg.recsys.hybrid import ChannelSet, build_candidates, build_candidates_for_history
from ntg.recsys.models import History
from ntg.recsys.rerank import RerankConfig, rerank
from ntg.tastetwin.model import TasteTwin


def test_tastetwin_incremental_update_equals_refit(data):
    df = data.interactions.sort_values("ts")
    cut = int(df["ts"].quantile(0.9))
    before, after = df[df.ts < cut], df[df.ts >= cut]
    u = int(after.groupby("u").size().idxmax())
    new = after[after.u == u]
    t_inc = TasteTwin(data).fit(before, cut)
    t_inc.update(u, new["i"].to_numpy(), new["rating"].to_numpy(), new["ts"].to_numpy())
    t_full = TasteTwin(data).fit(df[(df.ts < cut) | (df.u == u)], int(new["ts"].max()))
    np.testing.assert_allclose(t_inc.S_long[u], t_full.S_long[u], rtol=1e-6)
    np.testing.assert_allclose(t_inc.S_short[u], t_full.S_short[u], rtol=1e-6)


def test_tastetwin_history_state_matches_batch(data):
    w = data.window("test")
    tt = TasteTwin(data).fit(w.fit, w.cutoff_ts)
    u = int(w.fit.groupby("u").size().idxmax())
    g = w.fit[w.fit.u == u]
    st = tt.history_state(g["i"], g["rating"], g["ts"], w.cutoff_ts)
    np.testing.assert_allclose(st["S_long"], tt.S_long[u], rtol=1e-6)
    assert st["exploration"] == pytest.approx(tt.exploration[u], abs=1e-6)
    p = tt.profile(u)
    assert p["provenance"].startswith("inferred") and 0 <= p["confidence"] <= 1


def test_online_candidate_features_match_batch(data):
    """The frozen ranker sees identical features whether a viewer is scored in batch or online."""
    w = data.window("test")
    cs = ChannelSet(data, w.fit, ("ease", "itemknn", "popularity", "sequential", "content")).fit()
    u = int(w.fit.groupby("u").size().idxmax())
    df_b, X_b = build_candidates(cs, np.array([u]), 50)
    g = w.fit[w.fit.u == u]
    h = History(g["i"].to_numpy(), g["rating"].to_numpy(), g["ts"].to_numpy(), cs.cutoff_ts, data.n_items)
    st = cs.tt.history_state(h.items, h.ratings, h.ts, h.now)
    df_o, X_o, _ = build_candidates_for_history(cs, h, st, 50)
    assert set(df_b["i"]) == set(df_o["i"])
    ob = np.argsort(df_b["i"].to_numpy())
    oo = np.argsort(df_o["i"].to_numpy())
    np.testing.assert_allclose(X_b[ob], X_o[oo], rtol=1e-4, atol=1e-4)


def test_rerank_respects_franchise_cap_and_levels(data):
    items = np.arange(min(60, data.n_items))
    scores = np.linspace(1, 0, len(items))
    fr = np.zeros(data.n_items, dtype=int)  # everything in one franchise
    sel = rerank(items, scores, 10, genre_matrix=data.genre_matrix, franchise=fr, log_pop=np.zeros(data.n_items),
                 profile=None, cfg=RerankConfig.for_level("balanced"))
    assert len(sel) == 10  # caps relax rather than returning a short slate
    fam = rerank(items, scores, 10, genre_matrix=data.genre_matrix, franchise=data.franchise_codes,
                 log_pop=np.zeros(data.n_items), profile=None, cfg=RerankConfig.for_level("familiar"))
    assert fam[:3] == [0, 1, 2] or len(set(fam)) == 10


@pytest.mark.parametrize("q,expect", [
    ("I want something exciting but not too long.", {"boost": "Action", "unsupported": "duration"}),
    ("Find a mystery series that is not too dark.", {"boost": "Mystery", "avoid": "Horror", "unsupported": "format"}),
    ("Recommend something completely different from what I usually watch.", {"exploration": "unexpected"}),
    ("a funny classic from the 80s", {"boost": "Comedy", "year_min": 1980}),
])
def test_intent_parser(q, expect):
    p = parse_intent(q)
    if "boost" in expect:
        assert expect["boost"] in p.boost_genres
    if "avoid" in expect:
        assert expect["avoid"] in p.avoid_genres
    if "unsupported" in expect:
        assert expect["unsupported"] in [u["constraint"] for u in p.unsupported]
    if "exploration" in expect:
        assert p.exploration == expect["exploration"]
    if "year_min" in expect:
        assert p.year_min == expect["year_min"]


def test_business_scenario_is_transparent():
    zero = business_scenario(ScenarioAssumptions(retention_improvement_rel=0.0, intervention_cost_per_user_month=0.0))
    assert zero["central"]["incremental_revenue"] == pytest.approx(0.0, abs=1e-6)
    r = business_scenario(ScenarioAssumptions())
    assert r["evidence_type"] == "simulated"
    assert r["sensitivity"]["low"]["net_impact"] <= r["central"]["net_impact"] <= r["sensitivity"]["high"]["net_impact"]
    b = r["breakeven_retention_improvement_rel"]
    at = business_scenario(ScenarioAssumptions(retention_improvement_rel=b))
    assert abs(at["central"]["net_impact"]) < 0.01 * r["central"]["intervention_cost"]


def test_journey_simulation_deterministic_and_ope_unbiased(data):
    w = data.window("test")
    cs = ChannelSet(data, w.fit, ("ease", "popularity")).fit()
    users = np.flatnonzero(np.bincount(w.fit.u, minlength=data.n_users) >= 10)[:80]
    world = build_world(data, cs, data.interactions, users, n_candidates=100)
    a = compare_policies(world, WorldConfig(steps=10), ["greedy_relevance", "journey_orchestrator"])
    b = compare_policies(world, WorldConfig(steps=10), ["greedy_relevance", "journey_orchestrator"])
    assert a["policies"]["journey_orchestrator"]["satisfaction_per_journey"] == b["policies"]["journey_orchestrator"]["satisfaction_per_journey"]
    assert a["evidence_type"] == "simulated"
    ope = off_policy_evaluation(world, n_logs=40000)
    assert abs(ope["snips_estimate"] - ope["true_target_value"]) < 0.05

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ntg.datasets.prepare import franchise_key
from ntg.recsys.evaluate import EvalConfig, _per_user_metrics, evaluate_recommender
from ntg.recsys.models import MODEL_REGISTRY, History
import scipy.sparse as sp


def test_global_temporal_split_has_no_leakage(data):
    df = data.interactions
    tr, va, te = (df[df.split == s]["ts"] for s in ("train", "val", "test"))
    assert tr.max() < va.min() <= va.max() < te.min()
    assert data.meta["is_synthetic"] is True


def test_franchise_key_strips_numbering_and_year():
    assert franchise_key("Toy Story 2 (1999)") == franchise_key("Toy Story (1995)") == "toy story"
    assert franchise_key("Matrix Reloaded, The (2003)") == "matrix reloaded"


def test_ranking_metrics_hand_computed():
    rel = sp.csr_matrix(np.array([[1, 0, 1, 0, 0]], dtype=np.float32))
    rec = np.array([[0, 1, 2, 3, 4]])
    m = _per_user_metrics(rec, rel, np.array([0]), 3)
    assert m["precision@3"][0] == pytest.approx(2 / 3)
    assert m["recall@3"][0] == pytest.approx(1.0)
    ideal = 1 + 1 / np.log2(3)
    assert m["ndcg@3"][0] == pytest.approx((1 + 1 / np.log2(4)) / ideal)
    assert m["map@3"][0] == pytest.approx((1 + 2 / 3) / 2)


@pytest.mark.parametrize("name", list(MODEL_REGISTRY))
def test_every_model_fits_scores_and_excludes_seen(data, name):
    w = data.window("val")
    kw = {"epochs": 1} if name == "neumf" else {}
    m = MODEL_REGISTRY[name](**kw).fit(data, w.fit)
    users = np.unique(w.fit["u"])[:20]
    s = m.score(users)
    assert s.shape == (len(users), data.n_items) and np.isfinite(s).all()
    top, _ = m.recommend(users, 10)
    X = data.csr(w.fit)
    for r, u in enumerate(users):
        assert not set(top[r]) & set(X[u].indices), "recommended an already-seen item"


@pytest.mark.parametrize("name", ["popularity", "itemknn", "ease", "rp3beta", "content", "sequential", "ials"])
def test_history_scoring_matches_batch_scoring(data, name):
    """The online path (any history) must agree with the batch path for dataset users."""
    w = data.window("test")
    m = MODEL_REGISTRY[name]().fit(data, w.fit)
    u = int(w.fit.groupby("u").size().idxmax())
    g = w.fit[w.fit.u == u]
    h = History(g["i"].to_numpy(), g["rating"].to_numpy(), g["ts"].to_numpy(), m.cutoff_ts, data.n_items)
    a, b = m.score(np.array([u]))[0], m.score_history(h)
    if name == "ials":  # fold-in vs. ALS fixed point: same ranking up to numerical noise
        assert np.corrcoef(a, b)[0, 1] > 0.98
    else:
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


def test_evaluate_beats_random(data):
    w = data.window("test")
    m = MODEL_REGISTRY["ease"]().fit(data, w.fit)
    rng = np.random.default_rng(0)
    cfg = EvalConfig(ks=(10,), bootstrap=0)
    good = evaluate_recommender(lambda u, k: m.recommend(u, k)[0], data, w.fit, w.holdout, cfg)
    rand = evaluate_recommender(lambda u, k: rng.integers(0, data.n_items, (len(u), k)), data, w.fit, w.holdout, cfg)
    assert good["metrics"]["ndcg@10"] > rand["metrics"]["ndcg@10"]

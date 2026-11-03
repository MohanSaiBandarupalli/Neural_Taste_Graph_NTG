"""Deep models (optional `deep` group). Skipped automatically when PyTorch is not installed."""
from __future__ import annotations

import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ntg.recsys.deep import ANNIndex, SASRec, TwoTower  # noqa: E402
from ntg.recsys.models import History  # noqa: E402


@pytest.fixture(scope="module")
def fitted(data):
    w = data.window("val")
    return w, {
        "sasrec": SASRec(max_epochs=2, patience=1, dim=32).fit(data, w.fit),
        "twotower": TwoTower(max_epochs=2, patience=1, dim=32).fit(data, w.fit),
    }


def test_sasrec_left_padding_gives_finite_scores(data):
    net = SASRec(dim=32)._build(data.n_items).eval()
    x = torch.tensor([[0, 0, 0, 0, 3, 7], [0, 0, 0, 0, 0, 1]])
    assert torch.isfinite(net.logits(net(x)[:, -1])).all()  # regression: NaN from fully masked rows


@pytest.mark.parametrize("name", ["sasrec", "twotower"])
def test_online_equals_batch_and_pickle_roundtrip(data, fitted, name):
    w, models = fitted
    m = models[name]
    u = int(w.fit.groupby("u").size().idxmax())
    g = w.fit[w.fit.u == u]
    h = History(g["i"].to_numpy(), g["rating"].to_numpy(), g["ts"].to_numpy(), m.cutoff_ts, data.n_items)
    a = m.score(np.array([u]))[0]
    np.testing.assert_allclose(a, m.score_history(h), rtol=1e-5, atol=1e-5)
    m2 = pickle.loads(pickle.dumps(m))  # bundles store weights, not modules
    np.testing.assert_allclose(m2.score(np.array([u]))[0], a, rtol=1e-5, atol=1e-5)
    assert np.isfinite(a).all() and m.epochs_trained >= 1


def test_new_viewer_without_retraining(fitted, data):
    _, models = fitted
    tt = models["twotower"]
    h = History(np.array([1, 5, 9]), np.array([5.0, 4.0, 4.5]), np.array([1, 2, 3]) + tt.cutoff_ts, tt.cutoff_ts + 10,
                data.n_items)
    s = tt.score_history(h)
    assert np.isfinite(s).all() and s.std() > 0  # inductive user tower: a never-seen viewer gets a real ranking


def test_hnsw_ann_recall_vs_exact(fitted):
    _, models = fitted
    V = models["twotower"].item_vecs
    q = V[:50] + 0.01
    exact, _ = ANNIndex(V, "flat").search(q, 20)
    approx, _ = ANNIndex(V, "hnsw").search(q, 20)
    recall = np.mean([len(set(a) & set(b)) / 20 for a, b in zip(exact, approx)])
    assert recall >= 0.95

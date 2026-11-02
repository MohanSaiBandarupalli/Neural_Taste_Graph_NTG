"""Candidate-generation / scoring models.

Every model implements the same tiny interface::

    model.fit(data, fit_df)           # fit_df: interactions in the fit window only
    model.score(user_idx)  -> (B, n_items) float32 scores (higher is better)

``recommend`` masks items already seen in the fit window and returns top-k.
All models are pure NumPy/SciPy so the system runs on a laptop and in CI with
no GPU and no network. Choices are documented in MODEL_CARD.md.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from ntg.datasets.matrix import InteractionData

DAY = 86400.0


def topk_rows(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise top-k (indices sorted by descending score)."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-vals, axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1), np.take_along_axis(vals, order, axis=1)


def _sparse_topk_cols(m: sp.csr_matrix | np.ndarray, k: int) -> sp.csr_matrix:
    """Keep the top-k entries of every row of a (dense or sparse) square matrix."""
    dense = m.toarray() if sp.issparse(m) else np.asarray(m)
    n = dense.shape[0]
    k = min(k, n - 1) if n > 1 else 1
    idx = np.argpartition(-dense, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(dense, idx, axis=1)
    rows = np.repeat(np.arange(n), k)
    out = sp.csr_matrix((vals.ravel(), (rows, idx.ravel())), shape=dense.shape, dtype=np.float32)
    out.eliminate_zeros()
    return out


@dataclass
class History:
    """A single viewer's history: item indices, ratings and timestamps (any order)."""

    items: np.ndarray
    ratings: np.ndarray
    ts: np.ndarray
    now: int
    n_items: int

    def __post_init__(self):
        order = np.argsort(self.ts, kind="stable")
        self.items = np.asarray(self.items, dtype=np.int64)[order]
        self.ratings = np.asarray(self.ratings, dtype=np.float64)[order]
        self.ts = np.asarray(self.ts, dtype=np.int64)[order]

    @property
    def unique_items(self) -> np.ndarray:
        return np.unique(self.items)

    def binary_row(self) -> sp.csr_matrix:
        u = self.unique_items
        return sp.csr_matrix((np.ones(len(u), np.float32), (np.zeros(len(u), int), u)), shape=(1, self.n_items))


@dataclass
class Recommender:
    name: str = "base"
    fit_seconds: float = field(default=0.0, init=False)

    def fit(self, data: InteractionData, fit_df: pd.DataFrame) -> "Recommender":
        t = time.perf_counter()
        self.data = data
        self.X = data.csr(fit_df)
        self.cutoff_ts = int(fit_df["ts"].max()) + 1 if len(fit_df) else 0
        self._fit(data, fit_df)
        self.fit_seconds = time.perf_counter() - t
        return self

    def _fit(self, data: InteractionData, fit_df: pd.DataFrame) -> None:  # pragma: no cover
        raise NotImplementedError

    def score(self, users: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def score_history(self, h: "History") -> np.ndarray:
        """Score the catalogue for an arbitrary interaction history (online / new users)."""
        raise NotImplementedError

    def recommend(self, users: np.ndarray, k: int = 10, exclude_seen: bool = True) -> tuple[np.ndarray, np.ndarray]:
        s = self.score(np.asarray(users)).astype(np.float32, copy=True)
        if exclude_seen:
            sub = self.X[np.asarray(users)]
            r, c = sub.nonzero()
            s[r, c] = -np.inf
        return topk_rows(s, k)


# ---------------------------------------------------------------- baselines
@dataclass
class Popularity(Recommender):
    name: str = "popularity"
    half_life_days: float | None = 180.0

    def _fit(self, data, fit_df):
        if self.half_life_days:
            age = (self.cutoff_ts - fit_df["ts"].to_numpy()) / DAY
            w = 0.5 ** (age / self.half_life_days)
        else:
            w = np.ones(len(fit_df))
        self.pop = np.bincount(fit_df["i"].to_numpy(), weights=w, minlength=data.n_items).astype(np.float32)

    def score(self, users):
        return np.broadcast_to(self.pop, (len(users), self.pop.size)).copy()

    def score_history(self, h):
        return self.pop.copy()


@dataclass
class ItemKNN(Recommender):
    """Item-item cosine CF with shrinkage (generalises the legacy co-occurrence graph)."""

    name: str = "itemknn"
    k: int = 20
    shrink: float = 10.0

    def _fit(self, data, fit_df):
        X = self.X
        co = (X.T @ X).toarray().astype(np.float32)
        n = np.sqrt(np.diag(co)).astype(np.float32)
        sim = co / (np.outer(n, n) + self.shrink + 1e-9)
        np.fill_diagonal(sim, 0.0)
        self.S = _sparse_topk_cols(sim, self.k)  # row i: neighbours of item i

    def score(self, users):
        return np.asarray((self.X[users] @ self.S).todense(), dtype=np.float32)

    def score_history(self, h):
        return np.asarray((h.binary_row() @ self.S).todense(), dtype=np.float32).ravel()


@dataclass
class EASE(Recommender):
    """Embarrassingly Shallow Autoencoder (Steck, 2019): closed-form item-item model."""

    name: str = "ease"
    l2: float = 1500.0
    max_items: int = 25_000

    def _fit(self, data, fit_df):
        if data.n_items > self.max_items:
            raise MemoryError(f"EASE needs a dense {data.n_items}^2 matrix; increase min_item_support")
        G = (self.X.T @ self.X).toarray().astype(np.float64)
        G[np.diag_indices_from(G)] += self.l2
        P = np.linalg.inv(G)
        B = -P / np.diag(P)
        np.fill_diagonal(B, 0.0)
        self.B = B.astype(np.float32)

    def score(self, users):
        return np.asarray(self.X[users] @ self.B, dtype=np.float32)

    def score_history(self, h):
        return np.asarray(h.binary_row() @ self.B, dtype=np.float32).ravel()


@dataclass
class RP3Beta(Recommender):
    """Graph random walk user->item->user->item with popularity penalisation (Paudel et al.)."""

    name: str = "rp3beta"
    alpha: float = 0.8
    beta: float = 0.6
    k: int = 100

    def _fit(self, data, fit_df):
        X = self.X.astype(np.float32)
        Pui = sp.diags(1.0 / np.maximum(np.asarray(X.sum(1)).ravel(), 1)) @ X
        Piu = sp.diags(1.0 / np.maximum(np.asarray(X.sum(0)).ravel(), 1)) @ X.T
        if self.alpha != 1.0:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)
        W = (Piu @ Pui).toarray()
        pop = np.maximum(np.asarray(X.sum(0)).ravel(), 1)
        W = W / (pop[None, :] ** self.beta)
        np.fill_diagonal(W, 0)
        self.W = _sparse_topk_cols(W, self.k)
        self.Pui = sp.csr_matrix(Pui)

    def score(self, users):
        return np.asarray((self.Pui[users] @ self.W).todense(), dtype=np.float32)

    def score_history(self, h):
        x = h.binary_row()
        x = x / max(x.sum(), 1)
        if self.alpha != 1.0:
            x = x.power(self.alpha)
        return np.asarray((sp.csr_matrix(x) @ self.W).todense(), dtype=np.float32).ravel()


# ---------------------------------------------------------------- latent factor
@dataclass
class IALS(Recommender):
    """Implicit ALS (Hu, Koren & Volinsky 2008) with confidence c = 1 + alpha * x."""

    name: str = "ials"
    factors: int = 16
    reg: float = 1.0
    alpha: float = 5.0
    iterations: int = 8
    seed: int = 0

    def _solve(self, Y: np.ndarray, M: sp.csr_matrix) -> np.ndarray:
        YtY = Y.T @ Y
        lam = self.reg * np.eye(Y.shape[1])
        out = np.zeros((M.shape[0], Y.shape[1]), dtype=np.float64)
        indptr, indices = M.indptr, M.indices
        for u in range(M.shape[0]):
            idx = indices[indptr[u]:indptr[u + 1]]
            if idx.size == 0:
                continue
            Yu = Y[idx]
            A = YtY + self.alpha * (Yu.T @ Yu) + lam
            b = (1.0 + self.alpha) * Yu.sum(0)
            out[u] = np.linalg.solve(A, b)
        return out

    def _fit(self, data, fit_df):
        rng = np.random.default_rng(self.seed)
        X = self.X
        Xt = sp.csr_matrix(X.T)
        self.V = rng.normal(0, 0.01, (data.n_items, self.factors))
        for _ in range(self.iterations):
            self.U = self._solve(self.V, X)
            self.V = self._solve(self.U, Xt)
        self.U = self.U.astype(np.float32)
        self.V = self.V.astype(np.float32)

    def score(self, users):
        return self.U[users] @ self.V.T

    def score_history(self, h):
        if len(h.items) == 0:
            return np.zeros(self.V.shape[0], np.float32)
        return self.fold_in(h.unique_items) @ self.V.T

    def fold_in(self, item_idx: np.ndarray) -> np.ndarray:
        """Real-time user vector for an arbitrary history (no retraining)."""
        Y = self.V.astype(np.float64)
        A = Y.T @ Y + self.alpha * (Y[item_idx].T @ Y[item_idx]) + self.reg * np.eye(Y.shape[1])
        b = (1.0 + self.alpha) * Y[item_idx].sum(0)
        return np.linalg.solve(A, b).astype(np.float32)


@dataclass
class NeuMF(Recommender):
    """Neural collaborative filtering (He et al. 2017: GMF + MLP) trained with BPR loss.

    Implemented in NumPy with manual gradients and Adam so it runs without a
    deep-learning framework. Small by design; see MODEL_CARD.md for results.
    """

    name: str = "neumf"
    dim: int = 32
    hidden: int = 32
    epochs: int = 8
    batch: int = 2048
    lr: float = 0.01
    l2: float = 1e-5
    seed: int = 0

    def _fit(self, data, fit_df):
        rng = np.random.default_rng(self.seed)
        nu, ni, d, h = data.n_users, data.n_items, self.dim, self.hidden
        p = {
            "Pg": rng.normal(0, 0.05, (nu, d)), "Qg": rng.normal(0, 0.05, (ni, d)),
            "Pm": rng.normal(0, 0.05, (nu, d)), "Qm": rng.normal(0, 0.05, (ni, d)),
            "W1u": rng.normal(0, 1 / np.sqrt(2 * d), (d, h)), "W1i": rng.normal(0, 1 / np.sqrt(2 * d), (d, h)),
            "b1": np.zeros(h), "w2": rng.normal(0, 1 / np.sqrt(h), h), "wg": np.ones(d),
        }
        m = {k: np.zeros_like(v) for k, v in p.items()}
        v = {k: np.zeros_like(v) for k, v in p.items()}
        X = self.X.tocoo()
        pu, pi = X.row, X.col
        t = 0
        self.loss_history = []
        for _ in range(self.epochs):
            perm = rng.permutation(len(pu))
            ep_loss = 0.0
            for s in range(0, len(perm), self.batch):
                b = perm[s:s + self.batch]
                u, i = pu[b], pi[b]
                j = rng.integers(0, ni, len(b))
                g, loss = self._grads(p, u, i, j)
                ep_loss += loss * len(b)
                t += 1
                for k in p:
                    if k not in g:
                        continue
                    gk = g[k]
                    if isinstance(gk, tuple):  # sparse row gradient (idx, rows)
                        idx, rows = gk
                        dense = np.zeros_like(p[k])
                        np.add.at(dense, idx, rows)
                        uniq = np.unique(idx)
                        gk = dense[uniq] + self.l2 * p[k][uniq]
                        m[k][uniq] = 0.9 * m[k][uniq] + 0.1 * gk
                        v[k][uniq] = 0.999 * v[k][uniq] + 0.001 * gk ** 2
                        p[k][uniq] -= self.lr * (m[k][uniq] / (1 - 0.9 ** t)) / (np.sqrt(v[k][uniq] / (1 - 0.999 ** t)) + 1e-8)
                    else:
                        m[k] = 0.9 * m[k] + 0.1 * gk
                        v[k] = 0.999 * v[k] + 0.001 * gk ** 2
                        p[k] -= self.lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
            self.loss_history.append(ep_loss / len(perm))
        self.p = {k: val.astype(np.float32) for k, val in p.items()}

    @staticmethod
    def _forward(p, u, i):
        gmf = p["Pg"][u] * p["Qg"][i]
        z = p["Pm"][u] @ p["W1u"] + p["Qm"][i] @ p["W1i"] + p["b1"]
        a = np.maximum(z, 0)
        return gmf @ p["wg"] + a @ p["w2"], (gmf, z, a)

    def _grads(self, p, u, i, j):
        si, ci = self._forward(p, u, i)
        sj, cj = self._forward(p, u, j)
        x = si - sj
        sig = 1 / (1 + np.exp(-np.clip(x, -30, 30)))
        loss = float(np.mean(-np.log(sig + 1e-12)))
        dx = -(1 - sig) / len(u)  # d loss / d x
        g: dict = {}
        g["wg"] = (dx[:, None] * (ci[0] - cj[0])).sum(0)
        g["w2"] = (dx[:, None] * (ci[2] - cj[2])).sum(0)
        dzi = (dx[:, None] * p["w2"]) * (ci[1] > 0)
        dzj = (-dx[:, None] * p["w2"]) * (cj[1] > 0)
        g["b1"] = (dzi + dzj).sum(0)
        g["W1u"] = p["Pm"][u].T @ (dzi + dzj)
        g["W1i"] = p["Qm"][i].T @ dzi + p["Qm"][j].T @ dzj
        dgi = dx[:, None] * p["wg"]
        g["Pg"] = (u, dgi * (p["Qg"][i] - p["Qg"][j]))
        g["Qg"] = (np.concatenate([i, j]), np.concatenate([dgi * p["Pg"][u], -dgi * p["Pg"][u]]))
        g["Pm"] = (u, (dzi + dzj) @ p["W1u"].T)
        g["Qm"] = (np.concatenate([i, j]), np.concatenate([dzi @ p["W1i"].T, dzj @ p["W1i"].T]))
        return g, loss

    def score(self, users):
        p = self.p
        users = np.asarray(users)
        out = np.empty((len(users), self.data.n_items), dtype=np.float32)
        C = p["Qm"] @ p["W1i"]  # (I, h)
        for s in range(0, len(users), 32):
            u = users[s:s + 32]
            gmf = (p["Pg"][u] * p["wg"]) @ p["Qg"].T
            A = p["Pm"][u] @ p["W1u"] + p["b1"]
            mlp = np.maximum(A[:, None, :] + C[None, :, :], 0) @ p["w2"]
            out[s:s + 32] = gmf + mlp
        return out


# ---------------------------------------------------------------- content / semantic
def _item_documents(data: InteractionData) -> list[str]:
    docs = []
    for _, r in data.items.iterrows():
        toks = [f"genre_{g.lower().replace('-', '')}" for g in r["genres"]]
        toks += [f"tag_{t.replace(' ', '_')}" for t in r["tags"]]
        if pd.notna(r["year"]):
            toks.append(f"decade_{int(r['year']) // 10 * 10}")
        docs.append(" ".join(toks))
    return docs


def _user_profiles(X_rating: sp.csr_matrix, ts: sp.csr_matrix | None, cutoff: int, half_life_days: float,
                   users: np.ndarray, item_vecs: np.ndarray) -> np.ndarray:
    R = X_rating[users]
    W = R.copy()
    W.data = W.data - 3.0  # centred rating: dislikes push the profile away
    if ts is not None:
        T = ts[users]
        W.data = W.data * (0.5 ** (((cutoff - T.data) / DAY) / half_life_days))
    return np.asarray(W @ item_vecs)


@dataclass
class ContentTFIDF(Recommender):
    """Content-based retrieval over genres, tags and decade (works for cold items)."""

    name: str = "content"
    half_life_days: float = 365.0

    def _fit(self, data, fit_df):
        self.vec = TfidfVectorizer(token_pattern=r"[^ ]+", sublinear_tf=True)
        M = self.vec.fit_transform(_item_documents(data)).astype(np.float32)
        self.item_vecs = M.toarray()
        self._build_user_mats(data, fit_df)

    def _build_user_mats(self, data, fit_df):
        last = fit_df.drop_duplicates(["u", "i"], keep="last")
        shape = (data.n_users, data.n_items)
        self.R = sp.csr_matrix((last["rating"].to_numpy(np.float32), (last["u"], last["i"])), shape=shape)
        self.T = sp.csr_matrix((last["ts"].to_numpy(np.float64), (last["u"], last["i"])), shape=shape)

    def score(self, users):
        prof = _user_profiles(self.R, self.T, self.cutoff_ts, self.half_life_days, np.asarray(users), self.item_vecs)
        prof /= np.linalg.norm(prof, axis=1, keepdims=True) + 1e-9
        return (prof @ self.item_vecs.T).astype(np.float32)

    def score_history(self, h):
        if len(h.items) == 0:
            return np.zeros(self.item_vecs.shape[0], np.float32)
        # last rating per item, decayed to the same reference time as batch scoring
        last = {}
        for i, r, t in zip(h.items, h.ratings, h.ts):
            last[int(i)] = (r, t)
        idx = np.array(list(last))
        r = np.array([v[0] for v in last.values()]) - 3.0
        t = np.array([v[1] for v in last.values()], dtype=np.float64)
        w = r * 0.5 ** (((self.cutoff_ts - t) / DAY) / self.half_life_days)
        prof = w @ self.item_vecs[idx]
        prof = prof / (np.linalg.norm(prof) + 1e-9)
        return (prof @ self.item_vecs.T).astype(np.float32)


@dataclass
class SemanticLSA(ContentTFIDF):
    """Latent semantic retrieval: TF-IDF(title words + genres + tags) -> truncated SVD.

    A lightweight, local stand-in for neural sentence embeddings (which need a
    model download). Also powers natural-language search via :meth:`embed_text`.
    """

    name: str = "semantic"
    dim: int = 48

    def _fit(self, data, fit_df):
        docs = [
            f"{d} " + " ".join(w.lower() for w in str(t).split() if len(w) > 2)
            for d, t in zip(_item_documents(data), data.items["display_title"])
        ]
        self.vec = TfidfVectorizer(token_pattern=r"[^ ]+", sublinear_tf=True, min_df=1)
        M = self.vec.fit_transform(docs)
        dim = max(2, min(self.dim, M.shape[1] - 1, M.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=dim, random_state=0).fit(M)
        E = self.svd.transform(M).astype(np.float32)
        self.item_vecs = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        self._build_user_mats(data, fit_df)

    def embed_text(self, tokens: list[str]) -> np.ndarray:
        v = self.svd.transform(self.vec.transform([" ".join(tokens)]))[0].astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)


# ---------------------------------------------------------------- sequential
@dataclass
class SequentialMarkov(Recommender):
    """Time-aware item-to-next-item transition model.

    Transitions are counted between each event and the next ``window`` events
    of the same user (weights 1, 1/2, 1/4, ...), skipping gaps > ``max_gap_days``.
    A user's score is a recency-weighted sum of transition rows of their last
    ``last_n`` items. Captures short-term sequential intent that static CF misses.
    """

    name: str = "sequential"
    window: int = 3
    last_n: int = 5
    max_gap_days: float = 60.0
    smoothing: float = 1.0

    def _fit(self, data, fit_df):
        df = fit_df.sort_values(["u", "ts", "i"])
        u = df["u"].to_numpy()
        i = df["i"].to_numpy()
        t = df["ts"].to_numpy()
        rows, cols, vals = [], [], []
        for lag in range(1, self.window + 1):
            same = (u[lag:] == u[:-lag]) & ((t[lag:] - t[:-lag]) <= self.max_gap_days * DAY)
            rows.append(i[:-lag][same])
            cols.append(i[lag:][same])
            vals.append(np.full(same.sum(), 0.5 ** (lag - 1), dtype=np.float32))
        T = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(data.n_items, data.n_items))
        T.sum_duplicates()
        T = sp.diags(1.0 / (np.asarray(T.sum(1)).ravel() + self.smoothing)) @ T
        self.T = sp.csr_matrix(T, dtype=np.float32)
        # last-n items per user (most recent first)
        tail = df.groupby("u").tail(self.last_n)
        tail = tail.assign(rk=tail.groupby("u").cumcount(ascending=False))
        w = 0.6 ** tail["rk"].to_numpy()
        self.L = sp.csr_matrix((w.astype(np.float32), (tail["u"], tail["i"])), shape=(data.n_users, data.n_items))

    def score(self, users):
        return np.asarray((self.L[users] @ self.T).todense(), dtype=np.float32)

    def score_history(self, h):
        tail = h.items[-self.last_n:][::-1]
        x = np.zeros(self.T.shape[0], np.float32)
        for rk, i in enumerate(tail):
            x[i] += 0.6 ** rk
        return np.asarray((sp.csr_matrix(x) @ self.T).todense(), dtype=np.float32).ravel()


MODEL_REGISTRY = {
    "popularity": Popularity,
    "itemknn": ItemKNN,
    "ease": EASE,
    "rp3beta": RP3Beta,
    "ials": IALS,
    "neumf": NeuMF,
    "content": ContentTFIDF,
    "semantic": SemanticLSA,
    "sequential": SequentialMarkov,
}

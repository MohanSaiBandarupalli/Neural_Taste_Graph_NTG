"""Deep recommenders (PyTorch, optional `deep` dependency group).

* :class:`SASRec`   — self-attentive sequential recommendation (Kang & McAuley, 2018), trained with
  full softmax cross-entropy over the catalogue, which is markedly stronger than the original
  binary-negative loss (Klenitskiy & Vasilev, RecSys 2023).
* :class:`TwoTower` — inductive two-tower retrieval: the user tower is a function of the
  viewer's history (so new viewers need no retraining), the item tower combines an ID embedding
  with genre/tag content features (so items with little data still get a usable vector). Trained
  with in-batch sampled softmax and logQ popularity correction (Yi et al., RecSys 2019).

Both implement the common :class:`~ntg.recsys.models.Recommender` interface, including
``score_history`` for arbitrary histories, and both early-stop on a leave-last-out split *inside*
the fit window, so evaluation windows are never touched during training.

torch is imported lazily: the rest of NTG works without the deep group installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ntg.recsys.models import DAY, History, Recommender


def _torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover - exercised only without the deep group
        raise ImportError("Deep models need the optional 'deep' group: poetry install --with deep") from e
    return torch


def _sequences(fit_df: pd.DataFrame, n_users: int) -> list[np.ndarray]:
    """Per-user item sequences (0-based item indices) ordered by time."""
    df = fit_df.sort_values(["u", "ts", "i"], kind="stable")
    seqs = [np.zeros(0, dtype=np.int64) for _ in range(n_users)]
    for u, g in df.groupby("u", sort=False)["i"]:
        seqs[int(u)] = g.to_numpy(np.int64)
    return seqs


def _leave_last_out_ndcg(scores: np.ndarray, targets: np.ndarray, k: int = 10) -> float:
    """NDCG@k of a single held-out target per row (scores already exclude history)."""
    if not np.isfinite(scores[np.arange(len(targets)), targets]).all():
        raise FloatingPointError("non-finite target scores: model produced NaN/inf")
    target_score = scores[np.arange(len(targets)), targets][:, None]
    rank = (scores > target_score).sum(1)
    return float(np.where(rank < k, 1.0 / np.log2(rank + 2), 0.0).mean())


# ======================================================================================== SASRec
@dataclass
class SASRec(Recommender):
    name: str = "sasrec"
    maxlen: int = 50
    dim: int = 64
    heads: int = 2
    layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    batch: int = 128
    max_epochs: int = 60
    patience: int = 6
    seed: int = 0
    threads: int = 4
    history: list = field(default_factory=list, init=False, repr=False)

    def _build(self, n_items: int):
        torch = _torch()
        nn = torch.nn
        cfg = self

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.item = nn.Embedding(n_items + 1, cfg.dim, padding_idx=0)  # 0 = padding, item i -> i+1
                self.pos = nn.Embedding(cfg.maxlen, cfg.dim)
                layer = nn.TransformerEncoderLayer(cfg.dim, cfg.heads, cfg.dim * 2, cfg.dropout, batch_first=True,
                                                   norm_first=True, activation="gelu")
                self.enc = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
                self.norm = nn.LayerNorm(cfg.dim)
                self.drop = nn.Dropout(cfg.dropout)
                causal = torch.triu(torch.ones(cfg.maxlen, cfg.maxlen, dtype=torch.bool), 1)
                self.register_buffer("causal", causal)

            def forward(self, seq):  # seq: (B, L) item ids (+1), left-padded with 0
                B, L = seq.shape
                x = self.item(seq) * (cfg.dim ** 0.5) + self.pos(torch.arange(L, device=seq.device))[None]
                x = self.drop(x)
                # One combined mask: causal, and real positions never attend to padding. Every position may
                # attend to itself, so no attention row is fully masked. (A fully masked row yields NaN, and
                # NaN values at padded positions then poison all real positions in the next layer.)
                mask = self.causal[:L, :L][None] | (seq == 0)[:, None, :]
                ar = torch.arange(L, device=seq.device)
                mask[:, ar, ar] = False
                h = self.enc(x, mask=mask.repeat_interleave(cfg.heads, dim=0))
                return self.norm(h)

            def logits(self, h):  # tied output embeddings over real items (drop padding row)
                return h @ self.item.weight[1:].T

        return Net()

    def _pad(self, seqs: list[np.ndarray]) -> np.ndarray:
        out = np.zeros((len(seqs), self.maxlen), dtype=np.int64)
        for r, s in enumerate(seqs):
            s = s[-self.maxlen:] + 1
            if len(s):
                out[r, -len(s):] = s
        return out

    def _fit(self, data, fit_df):
        torch = _torch()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.set_num_threads(self.threads)
        n_items = data.n_items
        self.seqs = _sequences(fit_df, data.n_users)
        # leave-last-out inside the fit window for early stopping
        train_seqs, val_in, val_tgt = [], [], []
        for s in self.seqs:
            if len(s) >= 3:
                train_seqs.append(s[:-1])
                val_in.append(s[:-1])
                val_tgt.append(s[-1])
            elif len(s) >= 2:
                train_seqs.append(s)
        rng = np.random.default_rng(self.seed)
        if len(val_tgt) > 2000:
            pick = rng.choice(len(val_tgt), 2000, replace=False)
            val_in = [val_in[k] for k in pick]
            val_tgt = [val_tgt[k] for k in pick]
        val_X = torch.from_numpy(self._pad(val_in))
        val_y = np.asarray(val_tgt)
        # training windows: every user contributes its last maxlen+1 items (longer users: extra chunks)
        windows = []
        for s in train_seqs:
            for end in range(len(s), 1, -self.maxlen):
                windows.append(s[max(0, end - self.maxlen - 1):end])
        W = np.zeros((len(windows), self.maxlen + 1), dtype=np.int64)
        for r, s in enumerate(windows):
            s = s + 1
            W[r, -len(s):] = s
        W = torch.from_numpy(W)
        net = self._build(n_items)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-4)
        best, best_state, bad = -1.0, None, 0
        self.history = []
        for epoch in range(self.max_epochs):
            net.train()
            perm = torch.randperm(len(W))
            tot, n = 0.0, 0
            for b in range(0, len(W), self.batch):
                xb = W[perm[b:b + self.batch]]
                inp, tgt = xb[:, :-1], xb[:, 1:]
                h = net(inp)
                mask = tgt > 0
                loss = torch.nn.functional.cross_entropy(net.logits(h[mask]), tgt[mask] - 1)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                tot += loss.item() * int(mask.sum())
                n += int(mask.sum())
            ndcg = self._val_ndcg(net, val_X, val_y, val_in)
            self.history.append({"epoch": epoch + 1, "train_loss": tot / max(n, 1), "val_ndcg@10": ndcg})
            if ndcg > best + 1e-4:
                best, bad = ndcg, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        self.net = net
        self.best_val_ndcg = best
        self.epochs_trained = len(self.history)

    def _val_ndcg(self, net, X, y, hist) -> float:
        torch = _torch()
        net.eval()
        with torch.no_grad():
            s = net.logits(net(X)[:, -1]).numpy()
        for r, hs in enumerate(hist):
            s[r, hs] = -np.inf  # history excluded, as at serving time
        return _leave_last_out_ndcg(s, y)

    def _score_padded(self, P: np.ndarray) -> np.ndarray:
        torch = _torch()
        out = np.zeros((len(P), self.data.n_items), dtype=np.float32)
        with torch.no_grad():
            for b in range(0, len(P), 512):
                x = torch.from_numpy(P[b:b + 512])
                out[b:b + 512] = self.net.logits(self.net(x)[:, -1]).numpy()
        empty = P[:, -1] == 0
        out[empty] = 0.0  # no history: no sequential signal
        return out

    def score(self, users):
        return self._score_padded(self._pad([self.seqs[int(u)] for u in users]))

    def score_history(self, h: History):
        return self._score_padded(self._pad([h.items]))[0]

    def __getstate__(self):
        # Store weights, not the nn.Module, so bundles unpickle without the class closure.
        st = self.__dict__.copy()
        if "net" in st:
            st["net_state"] = {k: v.cpu() for k, v in st.pop("net").state_dict().items()}
        return st

    def __setstate__(self, st):
        state = st.pop("net_state", None)
        self.__dict__.update(st)
        if state is not None:
            net = self._build(self.data.n_items)
            net.load_state_dict(state)
            net.eval()
            self.net = net


# ===================================================================================== Two-tower
@dataclass
class TwoTower(Recommender):
    name: str = "twotower"
    dim: int = 64
    hist_len: int = 50
    half_life_days: float = 120.0
    temperature: float = 0.07
    lr: float = 2e-3
    batch: int = 512
    max_epochs: int = 40
    patience: int = 5
    seed: int = 0
    threads: int = 4
    n_tags: int = 100
    history: list = field(default_factory=list, init=False, repr=False)

    def _features(self, data):
        G = data.genre_matrix
        vocab = pd.Series([t for ts in data.items["tags"] for t in ts]).value_counts().index[: self.n_tags]
        tidx = {t: k for k, t in enumerate(vocab)}
        T = np.zeros((data.n_items, len(tidx)), dtype=np.float32)
        for r, ts in enumerate(data.items["tags"]):
            for t in ts:
                if t in tidx:
                    T[r, tidx[t]] = 1.0
        dec = ((np.clip(data.item_years, 1900, 2030) - 1900) // 10).astype(np.int64)
        D = np.eye(14, dtype=np.float32)[np.clip(dec, 0, 13)]
        return np.hstack([G, T, D]).astype(np.float32)

    def _build(self, n_items: int, n_feat: int):
        torch = _torch()
        nn = torch.nn
        d = self.dim

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.item_id = nn.Embedding(n_items, d)
                self.item_feat = nn.Linear(n_feat, d)
                self.item_mlp = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
                self.hist_id = nn.Embedding(n_items + 1, d, padding_idx=0)
                self.user_mlp = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
                nn.init.normal_(self.item_id.weight, std=0.02)
                nn.init.normal_(self.hist_id.weight, std=0.02)

            def items(self, feats):  # (I, F) -> (I, d) unit vectors
                x = torch.cat([self.item_id.weight, self.item_feat(feats)], 1)
                return nn.functional.normalize(self.item_mlp(x), dim=-1)

            def users(self, hist, w, feats):  # hist (B, L) ids+1, w (B, L) recency weights
                e = self.hist_id(hist)
                f = self.item_feat(feats[(hist - 1).clamp(min=0)]) * (hist > 0)[..., None]
                wn = w / w.sum(1, keepdim=True).clamp(min=1e-6)
                pooled = torch.cat([(wn[..., None] * e).sum(1), (wn[..., None] * f).sum(1)], 1)
                return nn.functional.normalize(self.user_mlp(pooled), dim=-1)

        return Net()

    def _hist_arrays(self, items: np.ndarray, ts: np.ndarray, ref: float):
        it = items[-self.hist_len:]
        tt = ts[-self.hist_len:]
        H = np.zeros(self.hist_len, dtype=np.int64)
        Wt = np.zeros(self.hist_len, dtype=np.float32)
        if len(it):
            H[-len(it):] = it + 1
            Wt[-len(it):] = 0.5 ** (np.maximum(ref - tt, 0) / DAY / self.half_life_days)
        return H, Wt

    def _fit(self, data, fit_df):
        torch = _torch()
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        rng = np.random.default_rng(self.seed)
        self.F = torch.from_numpy(self._features(data))
        df = fit_df.sort_values(["u", "ts", "i"], kind="stable")
        self.seq_items = _sequences(fit_df, data.n_users)
        self.seq_ts = [np.zeros(0, dtype=np.int64) for _ in range(data.n_users)]
        for u, g in df.groupby("u", sort=False)["ts"]:
            self.seq_ts[int(u)] = g.to_numpy(np.int64)
        # training examples: (history prefix -> next item); leave-last-out per user for early stopping
        ex_h, ex_w, ex_y, va_h, va_w, va_y, va_u = [], [], [], [], [], [], []
        for u in range(data.n_users):
            s, t = self.seq_items[u], self.seq_ts[u]
            n = len(s)
            if n < 2:
                continue
            last = n - 1 if n >= 3 else n
            pos = np.arange(1, last)
            if len(pos) > 40:
                pos = rng.choice(pos, 40, replace=False)
            for p in pos:
                H, W = self._hist_arrays(s[:p], t[:p], float(t[p]))
                ex_h.append(H), ex_w.append(W), ex_y.append(s[p])
            if n >= 3:
                H, W = self._hist_arrays(s[:-1], t[:-1], float(t[-1]))
                va_h.append(H), va_w.append(W), va_y.append(s[-1]), va_u.append(u)
        Xh, Xw, Y = (torch.from_numpy(np.asarray(a)) for a in (ex_h, ex_w, ex_y))
        if len(va_y) > 2000:
            pick = rng.choice(len(va_y), 2000, replace=False)
            va_h, va_w, va_y, va_u = ([a[k] for k in pick] for a in (va_h, va_w, va_y, va_u))
        Vh, Vw = torch.from_numpy(np.asarray(va_h)), torch.from_numpy(np.asarray(va_w))
        Vy = np.asarray(va_y)
        pop = np.bincount(fit_df["i"].to_numpy(), minlength=data.n_items) + 1.0
        logq = torch.from_numpy(np.log(pop / pop.sum()).astype(np.float32))
        net = self._build(data.n_items, self.F.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-5)
        best, best_state, bad = -1.0, None, 0
        self.history = []
        for epoch in range(self.max_epochs):
            net.train()
            perm = torch.randperm(len(Y))
            tot = 0.0
            for b in range(0, len(Y), self.batch):
                idx = perm[b:b + self.batch]
                y = Y[idx]
                u = net.users(Xh[idx], Xw[idx], self.F)
                v = net.items(self.F)[y]
                logits = (u @ v.T) / self.temperature - logq[y][None, :]  # logQ correction
                same = y[None, :] == y[:, None]
                logits = logits.masked_fill(same & ~torch.eye(len(y), dtype=torch.bool), -1e9)
                loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(y)))
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(y)
            net.eval()
            with torch.no_grad():
                s = (net.users(Vh, Vw, self.F) @ net.items(self.F).T).numpy()
            for r, h in enumerate(va_h):
                s[r, h[h > 0] - 1] = -np.inf
            ndcg = _leave_last_out_ndcg(s, Vy)
            self.history.append({"epoch": epoch + 1, "train_loss": tot / len(Y), "val_ndcg@10": ndcg})
            if ndcg > best + 1e-4:
                best, bad = ndcg, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        self.net = net
        self.best_val_ndcg = best
        self.epochs_trained = len(self.history)
        with torch.no_grad():
            self.item_vecs = net.items(self.F).numpy().astype(np.float32)

    def user_vectors(self, hists: list[tuple[np.ndarray, np.ndarray]], ref: float) -> np.ndarray:
        torch = _torch()
        H, W = zip(*(self._hist_arrays(i, t, ref) for i, t in hists)) if hists else ((), ())
        with torch.no_grad():
            return self.net.users(torch.from_numpy(np.asarray(H)), torch.from_numpy(np.asarray(W)), self.F).numpy()

    def score(self, users):
        hs = [(self.seq_items[int(u)], self.seq_ts[int(u)]) for u in users]
        U = self.user_vectors(hs, float(self.cutoff_ts))
        out = (U @ self.item_vecs.T).astype(np.float32)
        empty = np.array([len(h[0]) == 0 for h in hs])
        out[empty] = 0.0
        return out

    def score_history(self, h: History):
        if len(h.items) == 0:
            return np.zeros(self.data.n_items, np.float32)
        # same reference time as batch scoring when the history ends inside the fit window
        ref = float(max(self.cutoff_ts, h.ts.max() + 1)) if h.ts.max() >= self.cutoff_ts else float(self.cutoff_ts)
        return (self.user_vectors([(h.items, h.ts)], ref) @ self.item_vecs.T)[0].astype(np.float32)

    def __getstate__(self):
        st = self.__dict__.copy()
        if "net" in st:
            st["net_state"] = {k: v.cpu() for k, v in st.pop("net").state_dict().items()}
        return st

    def __setstate__(self, st):
        state = st.pop("net_state", None)
        self.__dict__.update(st)
        if state is not None:
            net = self._build(self.data.n_items, self.F.shape[1])
            net.load_state_dict(state)
            net.eval()
            self.net = net


# ===================================================================================== ANN index
class ANNIndex:
    """faiss inner-product index over item vectors (exact Flat or approximate HNSW)."""

    def __init__(self, item_vecs: np.ndarray, kind: str = "hnsw", m: int = 32, ef_search: int = 128):
        import faiss

        self.kind = kind
        d = item_vecs.shape[1]
        if kind == "flat":
            self.index = faiss.IndexFlatIP(d)
        else:
            self.index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = ef_search
        self.index.add(np.ascontiguousarray(item_vecs, dtype=np.float32))

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores, ids = self.index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        return ids, scores

"""Hybrid two-stage recommender: multi-channel retrieval -> learned ranking.

Stage 1 (candidate generation): each channel model scores the catalogue and
contributes its top-``n_per_channel`` unseen items. The union is the candidate
set (the channel that retrieved an item is kept for provenance).

Stage 2 (ranking): a LightGBM LambdaMART ranker over features that combine all
channel scores/ranks with item, user, TasteTwin and fatigue features.

Training protocol (no leakage): channels + TasteTwin are fit on the TRAIN
window; candidate labels come from the VALIDATION window; ranker
hyper-parameters use early stopping on a held-out subset of validation users.
For test evaluation the channels are re-fit on TRAIN+VAL and the frozen ranker
is applied; labels from the TEST window are used only for scoring.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp

from ntg.datasets.matrix import InteractionData
from ntg.recsys.models import MODEL_REGISTRY, Recommender, topk_rows
from ntg.tastetwin.model import TasteTwin

DEFAULT_CHANNELS = ("ease", "rp3beta", "itemknn", "ials", "sequential", "content", "semantic", "popularity")


@dataclass
class ChannelSet:
    """A set of fitted channel models + TasteTwin for one time window."""

    data: InteractionData
    fit_df: pd.DataFrame
    channels: tuple[str, ...] = DEFAULT_CHANNELS
    params: dict = field(default_factory=dict)

    def fit(self) -> "ChannelSet":
        self.models: dict[str, Recommender] = {}
        self.fit_seconds: dict[str, float] = {}
        for name in self.channels:
            m = MODEL_REGISTRY[name](**self.params.get(name, {})).fit(self.data, self.fit_df)
            self.models[name] = m
            self.fit_seconds[name] = m.fit_seconds
        any_m = next(iter(self.models.values()))
        self.X = any_m.X
        self.cutoff_ts = any_m.cutoff_ts
        emb = self.models["ials"].U if "ials" in self.models else None
        self.tt = TasteTwin(self.data).fit(self.fit_df, self.cutoff_ts, embedding=emb)
        d = self.data
        self.pop = d.popularity(self.fit_df)
        self.log_pop = np.log1p(self.pop).astype(np.float32)
        self.recent_pop = np.log1p(self.models["popularity"].pop) if "popularity" in self.models else self.log_pop
        cutoff_year = pd.Timestamp(self.cutoff_ts, unit="s").year
        self.item_age = (cutoff_year - d.item_years).astype(np.float32)
        Gn = d.genre_matrix / np.maximum(d.genre_matrix.sum(1, keepdims=True), 1)
        self.Gn = Gn.astype(np.float32)
        # recent-genre saturation (last 12 events)
        recent = self.fit_df.sort_values("ts").groupby("u").tail(12)
        R = sp.csr_matrix((np.ones(len(recent), np.float32), (recent["u"], recent["i"])), shape=(d.n_users, d.n_items))
        rg = np.asarray(R @ Gn)
        self.recent_genre_share = rg / np.maximum(rg.sum(1, keepdims=True), 1e-9)
        # franchise exposure
        fr = d.franchise_codes
        self.F = sp.csr_matrix((np.ones(self.X.nnz, np.float32), (self.X.nonzero()[0], fr[self.X.nonzero()[1]])),
                               shape=(d.n_users, fr.max() + 1))
        n = np.bincount(self.fit_df["u"].to_numpy(), minlength=d.n_users)
        self.user_log_n = np.log1p(n).astype(np.float32)
        last = np.full(d.n_users, -1, np.int64)
        np.maximum.at(last, self.fit_df["u"].to_numpy(), self.fit_df["ts"].to_numpy())
        self.user_days_since = np.where(last >= 0, (self.cutoff_ts - last) / 86400.0, 9999).astype(np.float32)
        return self


FEATURE_NAMES: list[str] = []


def build_candidates(cs: ChannelSet, users: np.ndarray, n_per_channel: int = 100) -> tuple[pd.DataFrame, np.ndarray]:
    """Return a long dataframe of (user, item) candidates and the feature matrix."""
    d = cs.data
    users = np.asarray(users)
    seen = cs.X[users]
    sr, sc = seen.nonzero()
    names = list(cs.models)
    dense = {}
    ranks = {}
    cand = np.zeros((len(users), d.n_items), dtype=bool)
    provenance = np.zeros((len(users), d.n_items), dtype=np.int32)
    for bit, name in enumerate(names):
        s = cs.models[name].score(users).astype(np.float32)
        s[sr, sc] = -np.inf
        top, _ = topk_rows(s, n_per_channel)
        rows = np.repeat(np.arange(len(users)), top.shape[1])
        cand[rows, top.ravel()] = True
        provenance[rows, top.ravel()] |= 1 << bit
        order = np.argsort(-s, axis=1, kind="stable")
        rk = np.empty_like(order)
        np.put_along_axis(rk, order, np.arange(d.n_items)[None, :].repeat(len(users), 0), axis=1)
        s[~np.isfinite(s)] = np.nan
        dense[name] = s
        ranks[name] = rk
    r, c = np.nonzero(cand)
    u = users[r]
    feats = []
    fnames = []
    for name in names:
        feats.append(np.nan_to_num(dense[name][r, c], nan=0.0))
        fnames.append(f"{name}_score")
        feats.append(np.log1p(ranks[name][r, c]).astype(np.float32))
        fnames.append(f"{name}_logrank")
        # per-user standardised score (makes channel scores comparable across users)
        mu = np.nanmean(dense[name], axis=1)
        sd = np.nanstd(dense[name], axis=1) + 1e-6
        feats.append(np.nan_to_num((dense[name][r, c] - mu[r]) / sd[r], nan=0.0).astype(np.float32))
        fnames.append(f"{name}_z")
    tt = cs.tt
    long_s = tt.long_share(u)
    short_s = tt.short_share(u)
    Gi = cs.Gn[c]
    feats += [
        cs.log_pop[c], cs.recent_pop[c].astype(np.float32), cs.item_age[c],
        cs.user_log_n[u], cs.user_days_since[u],
        tt.exploration[u].astype(np.float32), tt.confidence(u).astype(np.float32), tt.stability(u).astype(np.float32),
        (long_s * Gi).sum(1).astype(np.float32), (short_s * Gi).sum(1).astype(np.float32),
        ((short_s - long_s) * Gi).sum(1).astype(np.float32),
        (cs.recent_genre_share[u] * Gi).sum(1).astype(np.float32),
        np.asarray(cs.F[u, d.franchise_codes[c]]).ravel().astype(np.float32),
        np.array([bin(x).count("1") for x in provenance[r, c]], dtype=np.float32),
    ]
    fnames += [
        "item_log_pop", "item_recent_pop", "item_age_years", "user_log_n", "user_days_since_last",
        "tt_exploration", "tt_confidence", "tt_stability", "tt_long_match", "tt_short_match",
        "tt_momentum_match", "fatigue_genre_saturation", "franchise_seen", "n_channels",
    ]
    global FEATURE_NAMES
    FEATURE_NAMES = fnames
    df = pd.DataFrame({"u": u, "i": c, "provenance": provenance[r, c]})
    return df, np.vstack(feats).T.astype(np.float32)


@dataclass
class HybridRanker:
    n_per_channel: int = 100
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 40
    seed: int = 0

    def train(self, cs: ChannelSet, holdout: pd.DataFrame, users: np.ndarray) -> dict:
        t0 = time.perf_counter()
        frames, Xs = [], []
        for s in range(0, len(users), 256):
            df, X = build_candidates(cs, users[s:s + 256], self.n_per_channel)
            frames.append(df)
            Xs.append(X)
        df = pd.concat(frames, ignore_index=True)
        X = np.vstack(Xs)
        rel = set(zip(holdout["u"].to_numpy(), holdout["i"].to_numpy()))
        y = np.fromiter(((a, b) in rel for a, b in zip(df["u"].to_numpy(), df["i"].to_numpy())), dtype=np.int32, count=len(df))
        # candidate recall: fraction of holdout items (for these users, unseen) that are in the candidate set
        hu = holdout[holdout["u"].isin(users)]
        seen = cs.X
        unseen_rel = sum(1 for a, b in zip(hu["u"], hu["i"]) if seen[a, b] == 0)
        cand_recall = float(y.sum() / max(unseen_rel, 1))
        # keep only users with at least one positive (lambdarank needs signal)
        pos_users = df.loc[y == 1, "u"].unique()
        rng = np.random.default_rng(self.seed)
        es_users = set(rng.choice(pos_users, max(1, len(pos_users) // 7), replace=False))
        keep = df["u"].isin(pos_users).to_numpy()
        is_es = df["u"].isin(es_users).to_numpy()
        tr = keep & ~is_es
        va = keep & is_es

        uu = df["u"].to_numpy()

        def groups(order):
            return np.unique(uu[order], return_counts=True)[1]  # rows are sorted by u

        # LightGBM requires rows sorted by group
        order_tr = np.flatnonzero(tr)[np.argsort(df.loc[tr, "u"].to_numpy(), kind="stable")]
        order_va = np.flatnonzero(va)[np.argsort(df.loc[va, "u"].to_numpy(), kind="stable")]
        self.model = lgb.LGBMRanker(
            objective="lambdarank", n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            num_leaves=self.num_leaves, min_child_samples=self.min_child_samples, random_state=self.seed,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, lambdarank_truncation_level=20,
        )
        self.model.fit(
            X[order_tr], y[order_tr], group=groups(order_tr),
            eval_set=[(X[order_va], y[order_va])],
            eval_group=[groups(order_va)],
            eval_at=[10], callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        self.feature_names = list(FEATURE_NAMES)
        imp = self.model.booster_.feature_importance("gain")
        self.importance = dict(sorted(zip(self.feature_names, map(float, imp)), key=lambda kv: -kv[1]))
        return {
            "n_train_users": int(len(set(df.loc[tr, 'u']))),
            "n_es_users": int(len(es_users)),
            "n_rows": int(tr.sum()),
            "positive_rate": float(y[tr].mean()),
            "candidate_recall": cand_recall,
            "best_iteration": int(self.model.best_iteration_ or self.n_estimators),
            "seconds": time.perf_counter() - t0,
            "feature_importance_gain": self.importance,
        }

    def score_candidates(self, cs: ChannelSet, users: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
        df, X = build_candidates(cs, users, self.n_per_channel)
        df["score"] = self.model.predict(X, num_iteration=self.model.best_iteration_)
        return df, X

    def recommend(self, cs: ChannelSet, users: np.ndarray, k: int) -> np.ndarray:
        df, _ = self.score_candidates(cs, users)
        df = df.sort_values(["u", "score"], ascending=[True, False])
        top = df.groupby("u", sort=False).head(k)
        out = np.zeros((len(users), k), dtype=np.int64)
        pos = {u: n for n, u in enumerate(users)}
        for u, grp in top.groupby("u", sort=False):
            items = grp["i"].to_numpy()
            out[pos[u], :len(items)] = items
            if len(items) < k:  # pad with popularity (never seen)
                pad = [x for x in np.argsort(-cs.pop) if cs.X[u, x] == 0 and x not in set(items)][: k - len(items)]
                out[pos[u], len(items):len(items) + len(pad)] = pad
        return out

    def contributions(self, X: np.ndarray) -> np.ndarray:
        """Per-feature additive contributions (TreeSHAP) for explanations."""
        return self.model.predict(X, num_iteration=self.model.best_iteration_, pred_contrib=True)

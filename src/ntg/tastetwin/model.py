"""TasteTwin: an evolving, uncertainty-aware representation of each viewer.

State per user (all computed from the fit window only, as of ``cutoff_ts``):

* ``long``  : genre affinity with a long half-life (default 365 days)
* ``short`` : genre affinity with a short half-life (default 45 days)
* ``momentum`` = short_share - long_share  ->  emerging (>0) / declining (<0) interests
* ``stability`` = cosine(long, short) in [0, 1]
* ``exploration`` : normalised genre entropy of recent events blended with long-tail share
* ``n_eff`` : effective (decayed) sample size; ``confidence`` = n_eff / (n_eff + k0)
* per-genre ``uncertainty`` : posterior std of a Beta(share) with decayed counts
* ``theme`` affinities over item tags (where tags exist)
* optional latent ``embedding`` (iALS user factors)

The decayed sums are *sufficient statistics*: :meth:`TasteTwin.update` applies
new interactions in O(new events) without recomputing history, which is the
basis for real-time preference updates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ntg.datasets.matrix import InteractionData

DAY = 86400.0


def preference_weight(rating: np.ndarray) -> np.ndarray:
    """Interaction strength: any rating is a weak positive; high ratings stronger."""
    return np.clip((np.asarray(rating, dtype=np.float64) - 1.5) / 3.5, 0.05, 1.0)


@dataclass(frozen=True)
class TasteTwinConfig:
    half_life_long_days: float = 180.0
    half_life_short_days: float = 45.0
    recent_events: int = 20
    confidence_k0: float = 8.0
    emerging_threshold: float = 0.06
    prior_strength: float = 2.0


class TasteTwin:
    def __init__(self, data: InteractionData, cfg: TasteTwinConfig = TasteTwinConfig()):
        self.data = data
        self.cfg = cfg
        self.genres = data.genre_vocab
        self.G = data.genre_matrix / np.maximum(data.genre_matrix.sum(1, keepdims=True), 1)  # split mass across genres
        self.embedding: np.ndarray | None = None

    # ------------------------------------------------------------------ build
    def fit(self, fit_df: pd.DataFrame, cutoff_ts: int | None = None, embedding: np.ndarray | None = None) -> "TasteTwin":
        d, c = self.data, self.cfg
        self.cutoff_ts = int(cutoff_ts if cutoff_ts is not None else fit_df["ts"].max() + 1)
        age = (self.cutoff_ts - fit_df["ts"].to_numpy()) / DAY
        w = preference_weight(fit_df["rating"].to_numpy())
        u, i = fit_df["u"].to_numpy(), fit_df["i"].to_numpy()
        shape = (d.n_users, d.n_items)
        W_long = sp.csr_matrix((w * 0.5 ** (age / c.half_life_long_days), (u, i)), shape=shape)
        W_short = sp.csr_matrix((w * 0.5 ** (age / c.half_life_short_days), (u, i)), shape=shape)
        # Sufficient statistics (decayed to cutoff_ts)
        self.S_long = np.asarray(W_long @ self.G)
        self.S_short = np.asarray(W_short @ self.G)
        self.last_ts = np.full(d.n_users, -1, dtype=np.int64)
        np.maximum.at(self.last_ts, u, fit_df["ts"].to_numpy())
        self.n_events = np.bincount(u, minlength=d.n_users)
        self.global_share = self.S_long.sum(0) / max(self.S_long.sum(), 1e-9)

        # Exploration: recent genre entropy + long-tail share
        recent = fit_df.sort_values("ts").groupby("u").tail(c.recent_events)
        R = sp.csr_matrix((np.ones(len(recent)), (recent["u"], recent["i"])), shape=shape)
        Rg = np.asarray(R @ self.G)
        p = Rg / np.maximum(Rg.sum(1, keepdims=True), 1e-9)
        ent = -(np.where(p > 0, p * np.log(p + 1e-12), 0)).sum(1) / np.log(len(self.genres))
        pop = d.popularity(fit_df)
        tail = pop < np.quantile(pop[pop > 0], 0.8) if (pop > 0).any() else np.zeros_like(pop, bool)
        tail_share = np.asarray(R @ tail.astype(np.float64)).ravel() / np.maximum(np.asarray(R.sum(1)).ravel(), 1)
        self.exploration = np.clip(0.7 * ent + 0.3 * tail_share, 0, 1)

        # Themes (tags) — only meaningful where tags exist
        vocab = sorted({t for ts in d.items["tags"] for t in ts})
        self.theme_vocab = vocab[:200]
        if self.theme_vocab:
            tidx = {t: k for k, t in enumerate(self.theme_vocab)}
            T = np.zeros((d.n_items, len(tidx)), dtype=np.float32)
            for r, ts in enumerate(d.items["tags"]):
                for t in ts:
                    if t in tidx:
                        T[r, tidx[t]] = 1.0
            self.S_theme = np.asarray(W_long @ T)
        else:
            self.S_theme = np.zeros((d.n_users, 0))
        self.embedding = embedding
        self._history_df = fit_df[["u", "i", "rating", "ts"]]
        return self

    # ------------------------------------------------------------------ online update
    def update(self, u: int, items: np.ndarray, ratings: np.ndarray, ts: np.ndarray) -> None:
        """Apply new events to user ``u`` incrementally (decay-then-add)."""
        c = self.cfg
        order = np.argsort(ts)
        for i, r, t in zip(np.asarray(items)[order], np.asarray(ratings)[order], np.asarray(ts)[order]):
            ref = max(self.cutoff_ts, int(self.last_ts[u]))
            dt = max(0.0, (t - ref) / DAY)
            # Bring the user's state to time t (state is stored as-of max(cutoff, last_ts)).
            self.S_long[u] *= 0.5 ** (dt / c.half_life_long_days)
            self.S_short[u] *= 0.5 ** (dt / c.half_life_short_days)
            w = float(preference_weight(np.array([r]))[0])
            self.S_long[u] += w * self.G[int(i)]
            self.S_short[u] += w * self.G[int(i)]
            self.last_ts[u] = max(int(self.last_ts[u]), int(t))
            self.n_events[u] += 1

    # ------------------------------------------------------------------ derived views
    @staticmethod
    def _share(S: np.ndarray) -> np.ndarray:
        return S / np.maximum(S.sum(-1, keepdims=True), 1e-9)

    def long_share(self, users=None) -> np.ndarray:
        S = self.S_long if users is None else self.S_long[users]
        return self._share(S)

    def short_share(self, users=None) -> np.ndarray:
        S = self.S_short if users is None else self.S_short[users]
        return self._share(S)

    def momentum(self, users=None) -> np.ndarray:
        return self.short_share(users) - self.long_share(users)

    def stability(self, users=None) -> np.ndarray:
        a, b = self.long_share(users), self.short_share(users)
        num = (a * b).sum(-1)
        den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
        return np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)

    def n_eff(self, users=None) -> np.ndarray:
        S = self.S_long if users is None else self.S_long[users]
        return S.sum(-1)

    def confidence(self, users=None) -> np.ndarray:
        n = self.n_eff(users)
        return n / (n + self.cfg.confidence_k0)

    def uncertainty(self, users=None) -> np.ndarray:
        """Posterior std of each genre share under a Beta prior centred on the global share."""
        S = self.S_long if users is None else self.S_long[users]
        n = S.sum(-1, keepdims=True)
        a = S + self.cfg.prior_strength * self.global_share
        b = (n - S) + self.cfg.prior_strength * (1 - self.global_share)
        return np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))

    def predict_future_share(self, users=None, blend: float = 0.5, shrink: float = 0.05) -> np.ndarray:
        """Forecast of the next-period genre distribution."""
        p = blend * self.short_share(users) + (1 - blend) * self.long_share(users)
        return (1 - shrink) * p + shrink * self.global_share

    # ------------------------------------------------------------------ profile
    def profile(self, u: int, top: int = 6) -> dict:
        long_s, short_s = self.long_share([u])[0], self.short_share([u])[0]
        mom = short_s - long_s
        unc = self.uncertainty([u])[0]
        g = self.genres
        order = np.argsort(-long_s)
        emerging = [
            {"genre": g[k], "momentum": float(mom[k]), "short_share": float(short_s[k])}
            for k in np.argsort(-mom) if mom[k] > self.cfg.emerging_threshold
        ][:3]
        declining = [
            {"genre": g[k], "momentum": float(mom[k]), "long_share": float(long_s[k])}
            for k in np.argsort(mom) if mom[k] < -self.cfg.emerging_threshold
        ][:3]
        themes = []
        if self.S_theme.shape[1]:
            st = self.S_theme[u]
            themes = [{"theme": self.theme_vocab[k], "weight": float(st[k] / max(st.sum(), 1e-9))}
                      for k in np.argsort(-st)[:5] if st[k] > 0]
        conf = float(self.confidence([u])[0])
        return {
            "user_index": int(u),
            "as_of_ts": int(max(self.cutoff_ts, self.last_ts[u])),
            "n_events": int(self.n_events[u]),
            "effective_sample_size": float(self.n_eff([u])[0]),
            "confidence": conf,
            "confidence_label": "high" if conf > 0.8 else "medium" if conf > 0.5 else "low",
            "stability": float(self.stability([u])[0]),
            "exploration_willingness": float(self.exploration[u]),
            "long_term": [{"genre": g[k], "share": float(long_s[k]), "uncertainty": float(unc[k])}
                          for k in order[:top] if long_s[k] > 0],
            "short_term": [{"genre": g[k], "share": float(short_s[k])}
                           for k in np.argsort(-short_s)[:top] if short_s[k] > 0],
            "emerging": emerging,
            "declining": declining,
            "themes": themes,
            "unavailable_dimensions": {
                "content_format": "dataset contains films only (no series/episode data)",
                "language": "dataset has no language metadata",
            },
            "provenance": "inferred from viewing history (time-decayed); not stated by the viewer",
        }

    def evolution(self, u: int, freq_days: int = 90, max_points: int = 16) -> list[dict]:
        """Genre-share snapshots over time for the taste-evolution chart."""
        h = self._history_df[self._history_df.u == u]
        if h.empty:
            return []
        t_end = int(max(self.cutoff_ts, self.last_ts[u]))
        t_start = int(h["ts"].min())
        step = freq_days * DAY
        n = int(min(max_points, max(1, np.ceil((t_end - t_start) / step))))
        points = []
        for k in range(n):
            t = t_end - (n - 1 - k) * step
            hh = h[h.ts < t]
            if hh.empty:
                continue
            w = preference_weight(hh["rating"].to_numpy()) * 0.5 ** (((t - hh["ts"].to_numpy()) / DAY) / self.cfg.half_life_short_days * 2)
            s = (w[:, None] * self.G[hh["i"].to_numpy()]).sum(0)
            s = s / max(s.sum(), 1e-9)
            points.append({"ts": int(t), "shares": {self.genres[j]: float(s[j]) for j in np.argsort(-s)[:6] if s[j] > 0.01}})
        return points

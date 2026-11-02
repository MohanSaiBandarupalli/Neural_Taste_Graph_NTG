"""MovieLens-schema synthetic data generator with known ground truth.

Why this exists
---------------
The real MovieLens 32M archive cannot be redistributed in this repository and
may not be downloadable in every environment (CI, sandboxes). This generator
produces files with the *exact* MovieLens schema (``ratings.csv``,
``movies.csv``, ``tags.csv``) so every pipeline runs end-to-end anywhere.

It is *not* a substitute for real data when reporting benchmark results. All
reports produced from it are labelled ``dataset=synthetic``.

What it simulates (and therefore what has ground truth)
-------------------------------------------------------
* Items with genres, latent "themes" (also emitted as tags), quality, a heavy
  tailed popularity prior, release years and franchises (sequels).
* Users with a long-term taste vector, an exploration propensity and, for a
  subset of users, a *planted taste drift* towards an emerging theme starting
  at a known time. ``ground_truth/users.parquet`` records who drifts, when and
  towards what, so TasteTwin's emerging-interest detection can be validated.
* Session-based consumption, popularity bias, rating noise.
* Recommendation fatigue: consuming many items of the same primary genre in a
  row lowers ratings and raises the probability that the user disengages
  permanently. This creates a real (if simulated) signal for engagement-risk
  and fatigue models; the hazard function is recorded in the data card.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]
# Genre base rates roughly following MovieLens.
GENRE_WEIGHTS = np.array([
    0.10, 0.06, 0.03, 0.03, 0.14, 0.05, 0.03, 0.18, 0.03, 0.01, 0.06, 0.005,
    0.02, 0.03, 0.07, 0.04, 0.07, 0.02, 0.015,
])
THEMES = [
    "dark", "feel-good", "cerebral", "action-packed", "romantic", "family",
    "epic", "quirky", "gritty", "nostalgic", "suspenseful", "inspiring",
]
# How strongly each genre loads on each theme (hand-specified, plausible).
_GT = {
    "Action": {"action-packed": 1.0, "epic": 0.4, "gritty": 0.3},
    "Adventure": {"epic": 0.8, "action-packed": 0.5, "family": 0.3},
    "Animation": {"family": 1.0, "quirky": 0.5, "feel-good": 0.5},
    "Children": {"family": 1.0, "feel-good": 0.6},
    "Comedy": {"feel-good": 0.8, "quirky": 0.7},
    "Crime": {"gritty": 0.9, "dark": 0.5, "suspenseful": 0.5},
    "Documentary": {"cerebral": 0.9, "inspiring": 0.6},
    "Drama": {"cerebral": 0.4, "inspiring": 0.4, "romantic": 0.2, "dark": 0.2},
    "Fantasy": {"epic": 0.8, "family": 0.3, "quirky": 0.3},
    "Film-Noir": {"dark": 0.9, "nostalgic": 0.7, "gritty": 0.4},
    "Horror": {"dark": 1.0, "suspenseful": 0.7},
    "IMAX": {"epic": 0.7, "action-packed": 0.4},
    "Musical": {"feel-good": 0.8, "nostalgic": 0.6, "romantic": 0.3},
    "Mystery": {"suspenseful": 0.9, "cerebral": 0.5, "dark": 0.3},
    "Romance": {"romantic": 1.0, "feel-good": 0.4},
    "Sci-Fi": {"cerebral": 0.7, "epic": 0.5, "action-packed": 0.3},
    "Thriller": {"suspenseful": 1.0, "dark": 0.4, "gritty": 0.3},
    "War": {"gritty": 0.8, "epic": 0.6, "inspiring": 0.3},
    "Western": {"nostalgic": 0.8, "gritty": 0.6},
}
_ADJ = [
    "Silent", "Crimson", "Hidden", "Last", "Golden", "Broken", "Midnight", "Electric",
    "Distant", "Burning", "Frozen", "Hollow", "Wild", "Velvet", "Iron", "Paper",
    "Shattered", "Lucky", "Secret", "Endless", "Northern", "Quiet", "Restless", "Stolen",
]
_NOUN = [
    "Harbor", "Garden", "Signal", "Empire", "Witness", "Horizon", "Frontier", "Machine",
    "Orchard", "Kingdom", "Detective", "Summer", "Voyage", "Lantern", "Republic", "Circus",
    "Archive", "Canyon", "Echo", "Monarch", "Station", "Island", "Protocol", "Serenade",
]


@dataclass(frozen=True)
class SyntheticConfig:
    n_users: int = 8000
    n_items: int = 3000
    seed: int = 42
    start: str = "2019-01-01"
    end: str = "2024-12-31"
    median_user_events: float = 70.0
    drift_frac: float = 0.4               # fraction of users with planted taste drift
    franchise_frac: float = 0.08          # fraction of items that are sequels
    fatigue_rating_penalty: float = 0.9   # rating drop at full genre saturation
    base_disengage_hazard: float = 0.012  # per-session hazard of leaving for good
    fatigue_disengage_hazard: float = 0.06
    tag_prob: float = 0.04


def _theme_matrix() -> np.ndarray:
    m = np.zeros((len(GENRES), len(THEMES)))
    for g, loads in _GT.items():
        for t, v in loads.items():
            m[GENRES.index(g), THEMES.index(t)] = v
    return m


def _unit(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=axis, keepdims=True), 1e-9, None)


def generate(out_dir: str | Path, cfg: SyntheticConfig = SyntheticConfig()) -> dict:
    rng = np.random.default_rng(cfg.seed)
    out = Path(out_dir)
    (out / "ground_truth").mkdir(parents=True, exist_ok=True)
    gt_m = _theme_matrix()
    n_g, n_t = gt_m.shape

    # ------------------------------------------------------------------ items
    gw = GENRE_WEIGHTS / GENRE_WEIGHTS.sum()
    primary = rng.choice(n_g, size=cfg.n_items, p=gw)
    genre_sets: list[list[int]] = []
    for p in primary:
        gs = [int(p)]
        n_extra = rng.choice([0, 1, 2], p=[0.35, 0.45, 0.20])
        # secondary genres correlated with primary via theme overlap
        aff = gt_m @ gt_m[p] + 0.05
        aff[p] = 0
        aff = aff * gw
        for g in rng.choice(n_g, size=n_extra, replace=False, p=aff / aff.sum()):
            gs.append(int(g))
        genre_sets.append(sorted(set(gs)))
    item_theme = np.zeros((cfg.n_items, n_t))
    for i, gs in enumerate(genre_sets):
        w = np.array([1.0 if g == primary[i] else 0.5 for g in gs])
        item_theme[i] = (w[:, None] * gt_m[gs]).sum(0)
    item_theme = _unit(item_theme + rng.normal(0, 0.25, item_theme.shape).clip(0, None))
    quality = rng.normal(0, 0.45, cfg.n_items)
    pop_prior = rng.lognormal(0, 1.5, cfg.n_items)
    years = np.clip((2024 - rng.gamma(2.0, 9.0, cfg.n_items)).round(), 1950, 2024).astype(int)

    titles = [f"The {rng.choice(_ADJ)} {rng.choice(_NOUN)}" for _ in range(cfg.n_items)]
    # franchises: sequels copy genres/themes of a parent
    franchise_of = np.full(cfg.n_items, -1)
    n_seq = int(cfg.franchise_frac * cfg.n_items)
    seq_ids = rng.choice(np.arange(cfg.n_items // 2, cfg.n_items), size=n_seq, replace=False)
    for s in seq_ids:
        parent = int(rng.integers(0, cfg.n_items // 2))
        root = parent if franchise_of[parent] < 0 else franchise_of[parent]
        franchise_of[s] = root
        genre_sets[s] = list(genre_sets[root])
        primary[s] = primary[root]
        item_theme[s] = _unit(item_theme[root] + rng.normal(0, 0.05, n_t).clip(0, None))
        n_prev = int((franchise_of == root).sum())
        titles[s] = f"{titles[root]} {n_prev + 1}"
        years[s] = min(2024, years[root] + 2 * n_prev)
        pop_prior[s] = pop_prior[root] * 0.7
    # disambiguate duplicate non-franchise titles with the year (MovieLens style)
    movies = pd.DataFrame({
        "movieId": np.arange(1, cfg.n_items + 1),
        "title": [f"{t} ({y})" for t, y in zip(titles, years)],
        "genres": ["|".join(GENRES[g] for g in gs) for gs in genre_sets],
    })

    # ------------------------------------------------------------------ users
    t0 = pd.Timestamp(cfg.start).value // 10**9
    t1 = pd.Timestamp(cfg.end).value // 10**9
    span = t1 - t0
    alpha = np.full(n_t, 0.35)
    u_long = _unit(rng.dirichlet(alpha, cfg.n_users))
    explore = rng.beta(2.0, 5.0, cfg.n_users)
    drifts = rng.random(cfg.n_users) < cfg.drift_frac
    # emerging theme: one the user currently has low affinity to
    emerging = np.array([int(np.argsort(u)[rng.integers(0, n_t // 2)]) for u in u_long])
    n_events = np.clip(rng.lognormal(np.log(cfg.median_user_events), 0.9, cfg.n_users), 8, 2500).astype(int)
    start = t0 + (rng.random(cfg.n_users) * span * 0.92).astype(np.int64)
    year_of_item_ts = (pd.to_datetime(years.astype(str) + "-01-01").values.astype("datetime64[s]").astype(np.int64))

    log_pop = np.log(pop_prior)
    rows_u, rows_i, rows_r, rows_t = [], [], [], []
    tag_rows = []
    user_meta = []
    for u in range(cfg.n_users):
        seen = np.zeros(cfg.n_items, dtype=bool)
        t = float(start[u])
        mean_gap = 86400 * rng.uniform(4, 30)
        drift_start = t + rng.uniform(0.25, 0.6) * (n_events[u] / 3.0) * mean_gap if drifts[u] else np.inf
        drift_len = 86400 * rng.uniform(60, 240)
        target = u_long[u].copy()
        e_vec = np.zeros(n_t)
        e_vec[emerging[u]] = 1.0
        recent_genres: list[int] = []
        produced = 0
        disengaged_at = None
        while produced < n_events[u] and t < t1:
            # taste at time t
            w = 0.0 if not np.isfinite(drift_start) else float(np.clip((t - drift_start) / drift_len, 0, 0.75))
            taste = _unit((1 - w) * target + w * e_vec)
            # fatigue: concentration of the last 12 primary genres
            if recent_genres:
                _, cnt = np.unique(recent_genres[-12:], return_counts=True)
                fatigue = float((cnt.max() / len(recent_genres[-12:])) * min(1.0, len(recent_genres) / 12))
            else:
                fatigue = 0.0
            sess_len = int(rng.integers(1, 6))
            # availability: items not released yet are invisible
            avail = (year_of_item_ts <= t) & ~seen
            if avail.sum() < sess_len:
                break
            affinity = item_theme @ taste
            # fatigued users prefer items outside the saturated genre
            sat_pen = np.zeros(cfg.n_items)
            if recent_genres and fatigue > 0.5:
                dom = max(set(recent_genres[-12:]), key=recent_genres[-12:].count)
                sat_pen = (primary == dom) * fatigue * 1.5 * explore[u] * 2
            logits = 9.0 * affinity + 0.9 * log_pop + 0.6 * quality - sat_pen
            logits = logits + rng.gumbel(0, 1.0 + 2.5 * explore[u], cfg.n_items)
            logits[~avail] = -np.inf
            picks = np.argpartition(-logits, sess_len)[:sess_len]
            for i in picks:
                seen[i] = True
                same = recent_genres[-12:].count(int(primary[i])) / 12 if recent_genres else 0.0
                r = 3.3 + 3.2 * (affinity[i] - 0.45) + quality[i] - cfg.fatigue_rating_penalty * same * (0.5 + explore[u])
                r = r + rng.normal(0, 0.55)
                r = float(np.clip(np.round(r * 2) / 2, 0.5, 5.0))
                ts = int(t + rng.integers(0, 4 * 3600))
                rows_u.append(u + 1)
                rows_i.append(int(i) + 1)
                rows_r.append(r)
                rows_t.append(ts)
                recent_genres.append(int(primary[i]))
                produced += 1
                if rng.random() < cfg.tag_prob:
                    th = int(np.argmax(item_theme[i] + rng.normal(0, 0.1, n_t)))
                    tag_rows.append((u + 1, int(i) + 1, THEMES[th], ts + 60))
            # disengagement hazard grows with fatigue
            hazard = cfg.base_disengage_hazard + cfg.fatigue_disengage_hazard * fatigue * (0.5 + explore[u])
            if rng.random() < hazard:
                disengaged_at = int(t)
                break
            t += rng.exponential(mean_gap)
        user_meta.append({
            "userId": u + 1,
            "drifts": bool(drifts[u]),
            "drift_start_ts": int(drift_start) if np.isfinite(drift_start) else None,
            "emerging_theme": THEMES[emerging[u]] if drifts[u] else None,
            "dominant_theme": THEMES[int(np.argmax(u_long[u]))],
            "exploration_propensity": float(explore[u]),
            "disengaged_ts": disengaged_at,
        })

    ratings = pd.DataFrame({"userId": rows_u, "movieId": rows_i, "rating": rows_r, "timestamp": rows_t})
    ratings = ratings.sort_values(["userId", "timestamp", "movieId"]).reset_index(drop=True)
    tags = pd.DataFrame(tag_rows, columns=["userId", "movieId", "tag", "timestamp"])

    ratings.to_csv(out / "ratings.csv", index=False)
    movies.to_csv(out / "movies.csv", index=False)
    tags.to_csv(out / "tags.csv", index=False)
    pd.DataFrame(user_meta).to_parquet(out / "ground_truth" / "users.parquet", index=False)
    pd.DataFrame({
        "movieId": movies["movieId"],
        "franchise_root": np.where(franchise_of >= 0, franchise_of + 1, movies["movieId"]),
        "quality": quality,
        **{f"theme_{t}": item_theme[:, k] for k, t in enumerate(THEMES)},
    }).to_parquet(out / "ground_truth" / "items.parquet", index=False)

    card = {
        "name": "ntg-synthetic-movielens-schema",
        "is_synthetic": True,
        "config": asdict(cfg),
        "n_ratings": int(len(ratings)),
        "n_users_with_ratings": int(ratings["userId"].nunique()),
        "n_items_rated": int(ratings["movieId"].nunique()),
        "n_tags": int(len(tags)),
        "mechanisms": {
            "taste_drift": "planted for drift_frac of users; linear ramp to 75% weight on an emerging theme",
            "fatigue": "concentration of last 12 primary genres; lowers ratings and raises disengagement hazard",
            "popularity_bias": "log-normal popularity prior in choice logits",
            "disengage_hazard": "base + fatigue_coef * fatigue * (0.5 + exploration_propensity) per session",
        },
        "warning": "Synthetic data. Do not report results from it as MovieLens or real-world performance.",
    }
    (out / "DATA_CARD.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    return card


if __name__ == "__main__":  # pragma: no cover
    import sys

    print(json.dumps(generate(sys.argv[1] if len(sys.argv) > 1 else "data/external/synthetic"), indent=2))

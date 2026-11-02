"""Serving engine: one object that answers every product/API question.

Loads a versioned bundle produced by :mod:`ntg.pipelines.v4` and supports:
* dataset viewers (history from the dataset) and brand-new viewers (history
  only from in-app events), with live events applied without retraining;
* hybrid recommendations + intent + exploration level + grounded explanations;
* TasteTwin profiles/evolution, hidden interests, fatigue risk, discovery
  journeys; similar items, search; enterprise analytics.

SECURITY NOTE: bundles are pickles — only load artifacts produced by this
pipeline from a trusted artifact store (the manifest records a SHA-256 that is
verified at load time).
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ntg.discovery.features import FATIGUE_FEATURES, user_state_features
from ntg.intent.parser import ParsedIntent, parse_intent
from ntg.journey.policies import JourneyOrchestrator, OrchestratorConfig
from ntg.recsys.explain import GROUNDING, channel_names, ranker_drivers
from ntg.recsys.hybrid import build_candidates_for_history
from ntg.recsys.models import History, topk_rows
from ntg.recsys.rerank import EXPLORATION_LEVELS, RerankConfig, rerank

DAY = 86400.0


class UnknownItem(KeyError):
    pass


class Engine:
    def __init__(self, bundle: dict, manifest: dict, report_dir: Path | None = None):
        self.bundle = bundle
        self.manifest = manifest
        self.cs = bundle["cs"]
        self.data = self.cs.data
        self.ranker = bundle["ranker"]
        self.fatigue = bundle["fatigue"]
        self.risk = bundle["risk"]
        self.report_dir = report_dir
        self.live: dict[str, list[tuple[int, float, int]]] = {}  # viewer key -> [(item_idx, rating, ts)]
        self._lock = threading.Lock()
        d = self.data
        self.now_ts = int(d.interactions["ts"].max()) + 1
        self._by_user = {int(u): g for u, g in d.interactions.groupby("u", sort=False)[["i", "rating", "ts"]]}
        self.titles_lower = d.items["display_title"].str.lower().to_numpy()
        self.genre_index = {g: k for k, g in enumerate(d.genre_vocab)}
        self.mean_rating, self.n_ratings = self._item_rating_stats()

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, artifact_dir: str | Path, version: str | None = None, report_root: Path = Path("reports/v4")) -> "Engine":
        artifact_dir = Path(artifact_dir)
        version = version or (artifact_dir / "CURRENT").read_text().strip()
        vdir = artifact_dir / version
        manifest = json.loads((vdir / "manifest.json").read_text())
        raw = (vdir / "bundle.pkl").read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["bundle_sha256"]:
            raise RuntimeError("bundle checksum mismatch — refusing to load")
        bundle = pickle.loads(raw)  # trusted: produced by our pipeline, checksum verified
        return cls(bundle, manifest, report_root / manifest["dataset"])

    def _item_rating_stats(self):
        df = self.data.interactions
        g = df.groupby("i")["rating"]
        m = np.zeros(self.data.n_items)
        n = np.zeros(self.data.n_items, dtype=int)
        m[g.mean().index] = g.mean().to_numpy()
        n[g.size().index] = g.size().to_numpy()
        return m, n

    # ------------------------------------------------------------------ viewers & items
    def item_index(self, item_id: int) -> int:
        try:
            return self.data.item_index[int(item_id)]
        except KeyError as e:
            raise UnknownItem(item_id) from e

    def item_card(self, i: int) -> dict:
        r = self.data.items.iloc[int(i)]
        return {
            "item_id": int(r["item_id"]),
            "title": str(r["display_title"]),
            "year": int(r["year"]) if pd.notna(r["year"]) else None,
            "genres": list(r["genres"]),
            "tags": list(r["tags"])[:6],
            "avg_rating": round(float(self.mean_rating[i]), 2) if self.n_ratings[i] else None,
            "n_ratings": int(self.n_ratings[i]),
            "popularity_percentile": float((self.cs.pop < self.cs.pop[i]).mean()),
        }

    def history(self, viewer: str, dataset_user_id: int | None) -> History:
        items, ratings, ts = [], [], []
        if dataset_user_id is not None and int(dataset_user_id) in self.data.user_index:
            g = self._by_user.get(self.data.user_index[int(dataset_user_id)])
            if g is not None:
                items, ratings, ts = list(g["i"]), list(g["rating"]), list(g["ts"])
        for i, r, t in self.live.get(viewer, []):
            items.append(i)
            ratings.append(r)
            ts.append(t)
        now = max(self.now_ts, max(ts) + 1 if ts else self.now_ts)
        return History(np.array(items, dtype=np.int64), np.array(ratings, dtype=float), np.array(ts, dtype=np.int64),
                       now, self.data.n_items)

    def record_event(self, viewer: str, item_id: int, rating: float, ts: int | None = None) -> None:
        i = self.item_index(item_id)
        rating = float(np.clip(rating, 0.5, 5.0))
        ts = int(ts if ts is not None else max(self.now_ts, int(time.time())))
        with self._lock:
            self.live.setdefault(viewer, []).append((i, rating, ts))

    def forget_viewer(self, viewer: str) -> None:
        with self._lock:
            self.live.pop(viewer, None)

    # ------------------------------------------------------------------ TasteTwin
    def tastetwin(self, viewer: str, dataset_user_id: int | None) -> dict:
        h = self.history(viewer, dataset_user_id)
        st = self.cs.tt.history_state(h.items, h.ratings, h.ts, h.now)
        prof = self.cs.tt.profile_from_state(st)
        prof["evolution"] = self._evolution(h)
        prof["live_events_applied"] = len(self.live.get(viewer, []))
        prof["fatigue"] = self.fatigue_risk(h)
        return prof

    def _evolution(self, h: History, step_days: int = 90, points: int = 12) -> list[dict]:
        if not h.items.size:
            return []
        tt = self.cs.tt
        out = []
        t_end = h.now
        span = max(h.now - h.ts.min(), step_days * DAY)
        n = int(min(points, max(2, np.ceil(span / (step_days * DAY)))))
        for k in range(n):
            t = t_end - (n - 1 - k) * step_days * DAY
            m = h.ts < t
            if not m.any():
                continue
            st = tt.history_state(h.items[m], h.ratings[m], h.ts[m], int(t))
            sh = tt._share(st["S_short"] * 0.5 + st["S_long"] * 0.5)
            out.append({"ts": int(t), "date": pd.Timestamp(int(t), unit="s").strftime("%Y-%m"),
                        "shares": {tt.genres[j]: round(float(sh[j]), 4) for j in np.argsort(-sh)[:6] if sh[j] > 0.01}})
        return out

    def fatigue_risk(self, h: History) -> dict:
        if h.items.size < 5:
            return {"available": False, "reason": "needs at least 5 interactions"}
        df = pd.DataFrame({"u": 0, "i": h.items, "rating": h.ratings, "ts": h.ts})
        feats = user_state_features(self.data, df, h.now)
        row = feats.iloc[0]
        p = float(self.fatigue.predict(feats)[0])
        return {"available": True, "risk": p, "label": "elevated" if p > 0.3 else "moderate" if p > 0.18 else "low",
                "drivers": self.fatigue.explain(row),
                "definition": "probability that satisfaction (ratings) drops while the viewer stays active; inactivity is not used",
                "evidence_type": "modeled"}

    # ------------------------------------------------------------------ recommendations
    def _intent_items(self, it: ParsedIntent) -> np.ndarray:
        d = self.data
        mask = np.ones(d.n_items, dtype=bool)
        if it.boost_genres:
            gi = [self.genre_index[g] for g in it.boost_genres if g in self.genre_index]
            mask &= d.genre_matrix[:, gi].sum(1) > 0
        if it.year_min:
            mask &= d.item_years >= it.year_min
        if it.year_max:
            mask &= d.item_years <= it.year_max
        idx = np.flatnonzero(mask)
        return idx[np.argsort(-self.cs.pop[idx])][:300]

    def find_title(self, title: str) -> int | None:
        t = title.lower().strip()
        exact = np.flatnonzero(self.titles_lower == t)
        if exact.size:
            return int(exact[np.argmax(self.cs.pop[exact])])
        sub = np.flatnonzero(np.char.find(self.titles_lower.astype(str), t) >= 0)
        if sub.size:
            return int(sub[np.argmax(self.cs.pop[sub])])
        return None

    def recommend(self, dataset_user_id: int | None = None, viewer: str | None = None, k: int = 20,
                  level: str = "balanced", intent: str | None = None, explain: bool = True,
                  exclude_item_ids: list[int] | None = None) -> dict:
        t0 = time.perf_counter()
        viewer = viewer or f"dataset:{dataset_user_id}"
        if level not in EXPLORATION_LEVELS:
            raise ValueError(f"level must be one of {EXPLORATION_LEVELS}")
        h = self.history(viewer, dataset_user_id)
        tt = self.cs.tt
        st = tt.history_state(h.items, h.ratings, h.ts, h.now)
        parsed = parse_intent(intent, current_year=pd.Timestamp(h.now, unit="s").year) if intent else None
        if parsed and parsed.exploration:
            level = parsed.exploration
        exclude = np.array([self.item_index(x) for x in (exclude_item_ids or []) if int(x) in self.data.item_index], dtype=np.int64)
        include, anchor = None, None
        if parsed:
            include = self._intent_items(parsed)
            if parsed.anchor_title:
                anchor = self.find_title(parsed.anchor_title)
                if anchor is not None and "itemknn" in self.cs.models:
                    row = self.cs.models["itemknn"].S[anchor]
                    include = np.concatenate([include, row.indices])
        cold = h.items.size == 0
        df, X, dense = build_candidates_for_history(self.cs, h, st, self.ranker.n_per_channel, exclude, include)
        score = self.ranker.model.predict(X, num_iteration=self.ranker.model.best_iteration_) if len(df) else np.array([])
        if cold:
            # No history: the ranker's user features are uninformative; lean on (decayed) popularity.
            score = 0.3 * score + 0.7 * (dense["popularity"][df["i"].to_numpy()] / max(self.cs.models["popularity"].pop.max(), 1e-9)) * 3
        df["score"] = score
        df["row"] = np.arange(len(df))  # index into X for explanations
        applied = []
        if parsed:
            G = self.data.genre_matrix[df["i"].to_numpy()]
            if parsed.avoid_genres:
                gi = [self.genre_index[g] for g in parsed.avoid_genres if g in self.genre_index]
                bad = G[:, gi].sum(1) > 0
                df = df[~bad]
                G = G[~bad]
                applied.append(f"excluded genres: {', '.join(parsed.avoid_genres)}")
            if parsed.boost_genres:
                gi = [self.genre_index[g] for g in parsed.boost_genres if g in self.genre_index]
                match = G[:, gi].sum(1) > 0
                if match.sum() >= k:
                    df = df[match]
                    applied.append(f"restricted to: {', '.join(parsed.boost_genres)}")
                else:
                    df.loc[match, "score"] += 2.0
                    applied.append(f"boosted: {', '.join(parsed.boost_genres)} (too few strict matches)")
            years = self.data.item_years[df["i"].to_numpy()]
            if parsed.year_min:
                df = df[years >= parsed.year_min]
                years = self.data.item_years[df["i"].to_numpy()]
                applied.append(f"released {parsed.year_min} or later")
            if parsed.year_max:
                df = df[years <= parsed.year_max]
                applied.append(f"released {parsed.year_max} or earlier")
            if parsed.avoid_tags:
                tags = self.data.items["tags"].to_numpy()
                keep = [not (set(tags[i]) & set(parsed.avoid_tags)) for i in df["i"]]
                df = df[np.array(keep, dtype=bool)] if len(df) else df
            if anchor is not None:
                df = df[df["i"] != anchor]
                sim = np.asarray(self.cs.models["itemknn"].S[anchor].todense()).ravel()
                df = df.assign(score=df["score"] + 3.0 * sim[df["i"].to_numpy()] / max(sim.max(), 1e-9))
                applied.append(f"similar to: {self.data.item_title(anchor)}")
        df = df.sort_values("score", ascending=False).head(300)
        rc = RerankConfig.for_level(level)
        items = df["i"].to_numpy()
        sel = rerank(items, df["score"].to_numpy(), k, genre_matrix=self.data.genre_matrix,
                     franchise=self.data.franchise_codes, log_pop=self.cs.log_pop, profile=tt._share(st["S_long"]), cfg=rc)
        chosen = df.iloc[sel]
        hidden = {x["genre"] for x in self.hidden_interests_for(h, st, top=3)} if explain and not cold else set()
        out_items = []
        if explain and len(chosen):
            contrib = self.ranker.contributions(X[chosen["row"].to_numpy()])
        for n, (_, r) in enumerate(chosen.iterrows()):
            card = self.item_card(int(r["i"]))
            card["rank"] = n + 1
            card["score"] = float(r["score"])
            if explain:
                card["explanation"] = explain_for_history(self, h, st, int(r["i"]), int(r["provenance"]), contrib[n], hidden)
            out_items.append(card)
        return {
            "viewer": viewer,
            "dataset_user_id": dataset_user_id,
            "cold_start": bool(cold),
            "exploration_level": level,
            "intent": parsed.as_dict() if parsed else None,
            "intent_applied": applied,
            "inferred": {
                "source": "TasteTwin (inferred from viewing history, not stated by the viewer)",
                "top_genres": [g["genre"] for g in tt.profile_from_state(st, top=3)["long_term"]],
            } if not cold else None,
            "items": out_items,
            "model_version": self.manifest["version"],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    # ------------------------------------------------------------------ discovery
    def hidden_interests_for(self, h: History, st: dict, top: int = 3, max_own: float = 0.05) -> list[dict]:
        cs, tt = self.cs, self.cs.tt
        if h.items.size == 0:
            return []
        own = tt._share(st["S_long"])
        s = cs.models["ease"].score_history(h).copy()
        s[h.unique_items] = -np.inf
        top_items, top_scores = topk_rows(s[None, :], 200)
        top_items, top_scores = top_items[0], top_scores[0]
        w = np.maximum(top_scores - top_scores[np.isfinite(top_scores)].min(), 0)
        cf = w @ cs.Gn[top_items]
        cf = cf / max(cf.sum(), 1e-9)
        score = 0.5 * cf + 0.5 * tt.global_share - own
        out = []
        for g in np.argsort(-score):
            if own[g] >= max_own or score[g] <= 0:
                continue
            ex = [int(x) for x in top_items if self.data.genre_matrix[x, g] > 0][:6]
            out.append({
                "genre": tt.genres[g], "score": float(score[g]),
                "evidence": {"your_share": float(own[g]), "collaborative_signal_share": float(cf[g]),
                             "catalog_wide_share": float(tt.global_share[g])},
                "explanation": (f"{tt.genres[g]} is {own[g]:.0%} of your viewing, but {cf[g]:.0%} of what the collaborative "
                                f"model predicts for you and {tt.global_share[g]:.0%} of all viewing."),
                "examples": [self.item_card(x) for x in ex],
                "evidence_type": "modeled",
            })
            if len(out) >= top:
                break
        return out

    def hidden_interests(self, viewer: str, dataset_user_id: int | None) -> list[dict]:
        h = self.history(viewer, dataset_user_id)
        st = self.cs.tt.history_state(h.items, h.ratings, h.ts, h.now)
        return self.hidden_interests_for(h, st)

    def journey_plan(self, viewer: str, dataset_user_id: int | None, sessions: int = 5, slate: int = 4) -> dict:
        """A model-based multi-session discovery plan (expected feedback = model relevance; not a promise)."""
        res = self.recommend(dataset_user_id, viewer, k=120, level="familiar", explain=False)
        items = [c["item_id"] for c in res["items"]]
        if not items:
            return {"sessions": []}
        idx = np.array([self.item_index(x) for x in items])
        rel = np.array([c["score"] for c in res["items"]])
        h = self.history(viewer, dataset_user_id)
        st = self.cs.tt.history_state(h.items, h.ratings, h.ts, h.now)

        class _W:  # minimal world view for the policy (no oracle)
            pass

        w = _W()
        w.rel = rel[None, :]
        w.primary = np.argmax(self.data.genre_matrix[idx], axis=1)[None, :]
        w.own_share = self.cs.tt._share(st["S_long"])[None, :]
        w.exploration = np.array([st["exploration"]])
        pol = JourneyOrchestrator(w, 0, k=slate, cfg=OrchestratorConfig())
        consumed = np.zeros(len(idx), dtype=bool)
        relz = (rel - rel.min()) / max(rel.max() - rel.min(), 1e-9)
        plan = []
        for t in range(sessions):
            sl = pol.select(t, consumed)
            # expected pick = highest predicted relevance in slate (planning assumption, stated in output)
            pick = int(sl[np.argmax(relz[sl])]) if len(sl) else None
            pol.feedback(t, sl, pick)
            if pick is not None:
                consumed[pick] = True
            plan.append({"session": t + 1,
                         "theme": self.data.genre_vocab[int(w.primary[0, sl[0]])] if len(sl) else None,
                         "slate": [self.item_card(int(idx[j])) for j in sl],
                         "assumed_pick_item_id": int(items[pick]) if pick is not None else None})
        return {"sessions": plan, "policy": "journey_orchestrator", "fatigue_belief_end": pol.fatigue_belief(),
                "evidence_type": "modeled",
                "assumption": "Plan assumes you pick the highest-relevance title each session; real choices update it."}

    # ------------------------------------------------------------------ catalogue
    def similar(self, item_id: int, k: int = 12) -> dict:
        i = self.item_index(item_id)
        out = {"item": self.item_card(i)}
        if "itemknn" in self.cs.models:
            row = self.cs.models["itemknn"].S[i]
            order = np.argsort(-row.data)[:k]
            out["collaborative"] = [{**self.item_card(int(row.indices[j])), "similarity": float(row.data[j])} for j in order]
        if "semantic" in self.cs.models:
            E = self.cs.models["semantic"].item_vecs
            sims = E @ E[i]
            sims[i] = -np.inf
            top = np.argsort(-sims)[:k]
            out["semantic"] = [{**self.item_card(int(j)), "similarity": float(sims[j])} for j in top]
        return out

    def search(self, q: str, k: int = 20) -> dict:
        q = (q or "").strip()[:200]
        if not q:
            return {"query": q, "items": []}
        ql = q.lower()
        title_hits = np.flatnonzero(np.char.find(self.titles_lower.astype(str), ql) >= 0)
        title_hits = title_hits[np.argsort(-self.cs.pop[title_hits])][:k]
        parsed = parse_intent(q)
        sem = []
        model = self.cs.models.get("semantic")
        if model is not None:
            toks = [f"genre_{g.lower().replace('-', '')}" for g in parsed.boost_genres]
            toks += [f"tag_{t.replace(' ', '_')}" for t in parsed.tags]
            toks += [w for w in re.findall(r"[a-z]{3,}", ql)]
            v = model.embed_text(toks)
            sims = model.item_vecs @ v
            sims = sims + 0.02 * self.cs.log_pop / max(self.cs.log_pop.max(), 1e-9)
            if parsed.avoid_genres:
                gi = [self.genre_index[g] for g in parsed.avoid_genres if g in self.genre_index]
                sims[self.data.genre_matrix[:, gi].sum(1) > 0] = -np.inf
            sem = [int(j) for j in np.argsort(-sims)[: k * 2] if j not in set(title_hits)][: k]
        return {"query": q, "title_matches": [self.item_card(int(i)) for i in title_hits],
                "semantic_matches": [self.item_card(i) for i in sem], "parsed": parsed.as_dict(),
                "semantic_method": "LSA over genres/tags/title words (local; no external API)"}

    def browse(self, genre: str | None = None, k: int = 30, offset: int = 0) -> list[dict]:
        idx = np.arange(self.data.n_items)
        if genre:
            if genre not in self.genre_index:
                return []
            idx = idx[self.data.genre_matrix[:, self.genre_index[genre]] > 0]
        idx = idx[np.argsort(-self.cs.models["popularity"].pop[idx])]
        return [self.item_card(int(i)) for i in idx[offset:offset + k]]

    # ------------------------------------------------------------------ reports
    def report(self, name: str) -> dict | None:
        if self.report_dir is None:
            return None
        p = self.report_dir / f"{name}.json"
        if not re.fullmatch(r"[a-z_]+", name) or not p.exists():
            return None
        return json.loads(p.read_text())


def explain_for_history(eng: Engine, h: History, st: dict, i: int, provenance: int, contrib: np.ndarray, hidden: set) -> dict:
    """History-based variant of :func:`ntg.recsys.explain.explain` (works for live/new viewers)."""
    cs, d, tt = eng.cs, eng.data, eng.cs.tt
    reasons = []
    anchors = []
    if "itemknn" in cs.models and h.items.size:
        S = cs.models["itemknn"].S
        hist = h.unique_items
        col = np.asarray(S[hist, i].todense()).ravel()
        for j in np.argsort(-col)[:3]:
            if col[j] > 0:
                anchors.append({"item_id": int(d.item_ids[hist[j]]), "title": d.item_title(int(hist[j])), "similarity": float(col[j])})
    if anchors:
        reasons.append({"type": "because_you_watched", "text": "Because you watched " + " and ".join(a["title"] for a in anchors[:2]),
                        "evidence": anchors})
    genres = list(d.items.at[i, "genres"])
    ls = dict(zip(tt.genres, tt._share(st["S_long"])))
    mom = dict(zip(tt.genres, tt._share(st["S_short"]) - tt._share(st["S_long"])))
    matched = sorted([(g, ls.get(g, 0.0)) for g in genres if ls.get(g, 0.0) > 0.08], key=lambda x: -x[1])
    if matched:
        reasons.append({"type": "taste_match", "text": f"{matched[0][0]} makes up {matched[0][1]:.0%} of your viewing",
                        "evidence": [{"genre": g, "long_share": float(s)} for g, s in matched]})
    rising = [(g, mom[g]) for g in genres if mom.get(g, 0.0) > tt.cfg.emerging_threshold]
    if rising:
        g, m = max(rising, key=lambda x: x[1])
        reasons.append({"type": "emerging_interest", "text": f"Your interest in {g} is rising (+{m:.0%} recently)",
                        "evidence": [{"genre": g, "momentum": float(m)}]})
    if hidden and set(genres) & hidden:
        hg = sorted(set(genres) & hidden)
        reasons.append({"type": "hidden_interest", "text": f"Discovery pick: {', '.join(hg)} may be an untapped interest",
                        "evidence": [{"genre": g} for g in hg]})
    if not reasons:
        reasons.append({"type": "popular", "text": "Popular with viewers right now",
                        "evidence": [{"popularity_percentile": eng.item_card(i)["popularity_percentile"]}]})
    return {"reasons": reasons,
            "retrieved_by": channel_names(provenance, list(cs.models)),
            "ranker_drivers": ranker_drivers(contrib, eng.ranker.feature_names),
            "grounding": GROUNDING}

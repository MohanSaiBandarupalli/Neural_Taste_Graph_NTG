"""NTG v4 end-to-end pipeline: prepare -> tune -> train -> evaluate -> package.

    python -m ntg.cli build --dataset synthetic          # everything, reproducible
    python -m ntg.cli build --dataset ml-32m --source data/external/movielens/ml-32m --user-sample 0.1

Outputs
-------
reports/v4/<dataset>/*.json   evaluation evidence (every number in the docs comes from here)
artifacts/<dataset>/          versioned serving bundle + manifest.json
"""
from __future__ import annotations

import hashlib
import json
import pickle
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from ntg.core.logging import get_logger
from ntg.datasets.matrix import InteractionData
from ntg.datasets.prepare import PrepareConfig, prepare
from ntg.discovery.fatigue import FatigueModel
from ntg.discovery.hidden import evaluate_hidden_interests
from ntg.enterprise.analytics import audience_segments, content_intelligence
from ntg.enterprise.engagement_risk import EngagementRiskModel
from ntg.journey.counterfactual import compare_policies, off_policy_evaluation, sensitivity
from ntg.journey.simulator import WorldConfig, build_world
from ntg.recsys.evaluate import EvalConfig, evaluate_recommender, holdout_relevance
from ntg.recsys.hybrid import DEFAULT_CHANNELS, ChannelSet, HybridRanker
from ntg.recsys.models import MODEL_REGISTRY
from ntg.recsys.rerank import EXPLORATION_LEVELS, RerankConfig, rerank
from ntg.tastetwin.evaluate import evaluate_tastetwin

log = get_logger("ntg.pipeline")

TUNING_GRID = {
    "ease": [{"l2": v} for v in (300.0, 800.0, 1500.0, 3000.0)],
    "rp3beta": [{"alpha": a, "beta": b} for a in (0.8, 1.0) for b in (0.4, 0.6)],
    "itemknn": [{"k": k, "shrink": 10.0} for k in (20, 50, 100)],
    "ials": [{"factors": f, "reg": 1.0, "alpha": a} for f in (16, 32) for a in (5.0, 10.0)],
    "popularity": [{"half_life_days": h} for h in (30.0, 180.0, None)],
}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=2), encoding="utf-8")


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _rss_mb() -> float:
    """Peak resident memory in MB. `resource` is POSIX-only, so Windows reports NaN instead of crashing."""
    try:
        import resource
    except ImportError:  # Windows
        return float("nan")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if platform.system() == "Darwin" else peak / 1024.0  # macOS reports bytes, Linux KB


class Pipeline:
    def __init__(self, dataset: str, data_root: Path = Path("data/v4"), report_root: Path = Path("reports/v4"),
                 artifact_root: Path = Path("artifacts"), quick: bool = False, seed: int = 0):
        self.dataset = dataset
        self.data_dir = data_root / dataset
        self.report_dir = report_root / dataset
        self.artifact_dir = artifact_root / dataset
        self.quick = quick
        self.seed = seed
        self.timings: dict[str, float] = {}

    def _timed(self, name, fn, *a, **kw):
        t = time.perf_counter()
        out = fn(*a, **kw)
        self.timings[name] = round(time.perf_counter() - t, 3)
        log.info(f"{name} done in {self.timings[name]}s (peak RSS {_rss_mb():.0f} MB)")
        return out

    # ------------------------------------------------------------------ steps
    def prepare(self, source: Path, **kw) -> dict:
        return self._timed("prepare", prepare, PrepareConfig(source_dir=source, out_dir=self.data_dir, dataset_name=self.dataset, **kw))

    def tune(self, d: InteractionData) -> dict:
        w = d.window("val")
        cfg = EvalConfig(ks=(10,), bootstrap=0, max_users=1500 if self.quick else 3000, seed=self.seed)
        chosen, trace = {}, {}
        for name, grid in TUNING_GRID.items():
            grid = grid[:2] if self.quick else grid
            res = []
            for params in grid:
                m = MODEL_REGISTRY[name](**params).fit(d, w.fit)
                r = evaluate_recommender(lambda u, k, m=m: m.recommend(u, k)[0], d, w.fit, w.holdout, cfg)
                res.append({"params": params, "ndcg@10": r["metrics"]["ndcg@10"]})
            best = max(res, key=lambda x: x["ndcg@10"])
            chosen[name] = best["params"]
            trace[name] = res
        return {"selected": chosen, "trace": trace, "selection_metric": "ndcg@10 on VAL window"}

    def run(self, source: Path | None = None, prepare_kw: dict | None = None) -> dict:
        t_all = time.perf_counter()
        if source is not None:
            meta = self.prepare(source, **(prepare_kw or {}))
        d = InteractionData(self.data_dir)
        meta = d.meta
        log.info(f"dataset {self.dataset}: {meta['n_interactions']} interactions, {meta['n_users']} users, {meta['n_items']} items")

        tuning = self._timed("tune", self.tune, d)
        _write(self.report_dir / "tuning.json", tuning)
        params = dict(tuning["selected"])
        if self.quick:  # CI / smoke runs: bounded deep-model training (no effect without the deep group)
            params.setdefault("sasrec", {"max_epochs": 3, "patience": 1})
            params.setdefault("twotower", {"max_epochs": 3, "patience": 1})

        # ---------------- ranker training on VAL labels (channels fit on TRAIN)
        wv = d.window("val")
        cs_val = self._timed("channels_fit_train", ChannelSet(d, wv.fit, DEFAULT_CHANNELS, params).fit)
        rel_v = holdout_relevance(d, wv.fit, wv.holdout)
        users_v = np.flatnonzero(np.asarray(rel_v.sum(1)).ravel() > 0)
        users_v = users_v[np.bincount(wv.fit["u"], minlength=d.n_users)[users_v] >= 1]
        ranker = HybridRanker(seed=self.seed)
        ranker_info = self._timed("ranker_train", ranker.train, cs_val, wv.holdout, users_v)

        fatigue = FatigueModel()
        fat_train = fatigue.fit(d, wv.fit, wv.holdout, wv.cutoff_ts)

        # ---------------- test evaluation (channels re-fit on TRAIN+VAL)
        wt = d.window("test")
        cs_test = self._timed("channels_fit_trainval", ChannelSet(d, wt.fit, DEFAULT_CHANNELS, params).fit)
        bench = self._timed("benchmark", self.benchmark, d, wt, cs_test, ranker, params)
        bench["ranker_training"] = ranker_info
        _write(self.report_dir / "benchmark.json", bench)

        tt_report = self._timed("tastetwin_eval", evaluate_tastetwin, d)
        _write(self.report_dir / "tastetwin.json", tt_report)

        fat_eval = fatigue.evaluate(d, wt.fit, wt.holdout, wt.cutoff_ts)
        _write(self.report_dir / "fatigue.json", {"train": fat_train, "test": fat_eval,
                                                  "label": "rating drop > 0.35 stars among users who stay active (>=3 events)"})
        hidden_val = evaluate_hidden_interests(cs_val, wv.holdout)
        hidden_test = evaluate_hidden_interests(cs_test, wt.holdout)
        _write(self.report_dir / "hidden_interests.json", {"val": hidden_val, "test": hidden_test,
                                                           "weights": "(neighbours, collaborative, global prior) = (0, 0.5, 0.5), selected on VAL"})

        risk = EngagementRiskModel(seed=self.seed)
        risk_train = risk.fit(d, int(meta["t_val"]))
        risk_eval = risk.evaluate(d, int(meta["t_test"]))
        risk_eval["train"] = risk_train
        _write(self.report_dir / "engagement_risk.json", risk_eval)

        journey = self._timed("journey", self.journey, d, cs_test)
        _write(self.report_dir / "journey.json", journey)

        # ---------------- serving bundle (fit on ALL data)
        bundle = self._timed("serving_fit", self.serving_bundle, d, params, ranker, fatigue, risk)
        version = self.save_bundle(bundle, d, tuning, bench, ranker_info)
        perf = self._timed("performance", self.performance, version)
        perf["pipeline_timings_s"] = self.timings
        perf["total_pipeline_s"] = round(time.perf_counter() - t_all, 2)
        _write(self.report_dir / "performance.json", perf)
        summary = {"dataset": self.dataset, "version": version, "is_synthetic": meta.get("is_synthetic", False),
                   "reports": sorted(p.name for p in self.report_dir.glob("*.json"))}
        _write(self.report_dir / "summary.json", summary)
        return summary

    # ------------------------------------------------------------------ benchmark
    def benchmark(self, d, wt, cs, ranker, params) -> dict:
        cfg = EvalConfig(ks=(10, 20), bootstrap=200 if self.quick else 500, max_users=2000 if self.quick else 5000, seed=self.seed)
        models = {}
        names = ["popularity", "itemknn", "ease", "rp3beta", "ials", "neumf", "content", "semantic", "sequential", "longtail"]
        names += [n for n in ("sasrec", "twotower") if n in cs.models]
        for name in names:
            m = cs.models.get(name) or MODEL_REGISTRY[name](**params.get(name, {})).fit(d, wt.fit)
            models[name] = (lambda u, k, m=m: m.recommend(u, k)[0], m.fit_seconds)
        models["hybrid_ltr"] = (lambda u, k: ranker.recommend(cs, u, k), None)
        results = {"warm": {}, "cold": {}}
        for seg in ("warm", "cold"):
            for name, (fn, fit_s) in models.items():
                r = evaluate_recommender(fn, d, wt.fit, wt.holdout, cfg, segment=seg)
                r["fit_seconds"] = fit_s
                results[seg][name] = r
        liked = {}
        lcfg = EvalConfig(ks=(10,), bootstrap=0, max_users=cfg.max_users, min_rating=4.0, seed=self.seed)
        for name in ("popularity", "rp3beta", "ease", "hybrid_ltr"):
            r = evaluate_recommender(models[name][0], d, wt.fit, wt.holdout, lcfg)
            liked[name] = {"n_users": r["n_users"], "ndcg@10": r["metrics"].get("ndcg@10"), "recall@10": r["metrics"].get("recall@10")}
        rerank_modes = {}
        for level in EXPLORATION_LEVELS:
            rc = RerankConfig.for_level(level)

            def fn(users, k, rc=rc):
                df, _ = ranker.score_candidates(cs, users)
                out = np.zeros((len(users), k), dtype=np.int64)
                prof = cs.tt.long_share(users)  # genre space == d.genre_vocab == genre_matrix columns
                pos = {u: n for n, u in enumerate(users)}
                for u, grp in df.groupby("u", sort=False):
                    grp = grp.nlargest(200, "score")
                    items = grp["i"].to_numpy()
                    sel = rerank(items, grp["score"].to_numpy(), k, genre_matrix=d.genre_matrix, franchise=d.franchise_codes,
                                 log_pop=cs.log_pop, profile=prof[pos[u]], cfg=rc)
                    chosen = items[sel]
                    out[pos[u], :len(chosen)] = chosen
                return out

            r = evaluate_recommender(fn, d, wt.fit, wt.holdout, EvalConfig(ks=(10,), bootstrap=0, max_users=1500, seed=self.seed))
            rerank_modes[level] = {"n_users": r["n_users"], **r["metrics"], "latency_ms_per_user": r["latency_ms_per_user"]}
        best_single = max((n for n in results["warm"] if n != "hybrid_ltr"), key=lambda n: results["warm"][n]["metrics"]["ndcg@10"])
        return {
            "dataset": self.dataset,
            "is_synthetic": d.meta.get("is_synthetic", False),
            "protocol": {
                "split": "global temporal: train < t_val <= val < t_test <= test",
                "t_val": d.meta["t_val_iso"], "t_test": d.meta["t_test_iso"],
                "fit_window": "train+val (models); ranker trained on val labels with channels fit on train",
                "relevance": "any test-window interaction with an item unseen in the fit window",
                "warm_users": ">=5 fit-window interactions", "cold_users": "1-4 fit-window interactions",
                "hyperparameters": "selected on VAL (see tuning.json)",
            },
            "results": results,
            "liked_only_relevance": liked,
            "rerank_exploration_levels": rerank_modes,
            "best_single_model_warm": best_single,
        }

    # ------------------------------------------------------------------ journey
    def journey(self, d, cs) -> dict:
        n = np.bincount(cs.fit_df["u"], minlength=d.n_users)
        users = np.flatnonzero(n >= 10)
        rng = np.random.default_rng(self.seed)
        users = rng.choice(users, min(len(users), 300 if self.quick else 600), replace=False)
        world = build_world(d, cs, d.interactions, users)
        base = compare_policies(world, WorldConfig(seed=self.seed))
        sens_f = sensitivity(world, "fatigue_sensitivity", [0.0, 0.5, 1.0, 1.5], WorldConfig(seed=self.seed))
        sens_r = sensitivity(world, "repeat_penalty", [1.0, 0.7, 0.4], WorldConfig(seed=self.seed))
        ope = off_policy_evaluation(world, seed=self.seed)
        world.cs = None  # keep the bundle small; the world arrays are self-contained
        self.world = world
        return {"evidence_type": "simulated", "default_world": base, "sensitivity_fatigue": sens_f,
                "sensitivity_repetition": sens_r, "off_policy_evaluation": ope}

    # ------------------------------------------------------------------ serving
    def serving_bundle(self, d, params, ranker, fatigue, risk) -> dict:
        wf = d.window("full")
        cs = ChannelSet(d, wf.fit, DEFAULT_CHANNELS, params).fit()
        labels, segments = audience_segments(cs, seed=self.seed)
        sample = np.flatnonzero(cs.tt.n_events >= 5)
        sample = np.random.default_rng(self.seed).choice(sample, min(800, len(sample)), replace=False)
        recs = ranker.recommend(cs, sample, 10)
        content = content_intelligence(cs, recs)
        return {"cs": cs, "ranker": ranker, "fatigue": fatigue, "risk": risk, "segments": segments,
                "segment_labels": labels, "content": content, "journey_world": getattr(self, "world", None)}

    def save_bundle(self, bundle, d, tuning, bench, ranker_info) -> str:
        h = hashlib.sha256(json.dumps(d.meta, sort_keys=True).encode()).hexdigest()[:8]
        version = f"{self.dataset}-{time.strftime('%Y%m%d%H%M%S')}-{h}"
        vdir = self.artifact_dir / version
        vdir.mkdir(parents=True, exist_ok=True)
        with open(vdir / "bundle.pkl", "wb") as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
        warm = bench["results"]["warm"]
        manifest = {
            "version": version,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_sha": _git_sha(),
            "dataset": self.dataset,
            "is_synthetic": d.meta.get("is_synthetic", False),
            "data_meta": d.meta,
            "hyperparameters": tuning["selected"],
            "channels": list(DEFAULT_CHANNELS),
            "ranker": {"type": "LightGBM LambdaMART", "best_iteration": ranker_info["best_iteration"]},
            "test_metrics_warm": {n: {k: warm[n]["metrics"][k] for k in ("ndcg@10", "recall@20", "hit_rate@10", "coverage@10")} for n in warm},
            "bundle_sha256": hashlib.sha256((vdir / "bundle.pkl").read_bytes()).hexdigest(),
            "python": platform.python_version(),
        }
        _write(vdir / "manifest.json", manifest)
        (self.artifact_dir / "CURRENT").write_text(version)
        return version

    @staticmethod
    def _ann_benchmark(eng) -> dict | None:
        """HNSW vs exact inner-product search over two-tower item vectors (recall@100, query latency)."""
        tt = eng.cs.models.get("twotower")
        if tt is None:
            return None
        from ntg.recsys.deep import ANNIndex

        V = tt.item_vecs
        rng = np.random.default_rng(0)
        users = rng.choice(np.flatnonzero(eng.cs.tt.n_events >= 5), 500, replace=False)
        Q = tt.user_vectors([(tt.seq_items[int(u)], tt.seq_ts[int(u)]) for u in users], float(tt.cutoff_ts))
        out = {"n_items": int(V.shape[0]), "dim": int(V.shape[1]), "n_queries": int(len(Q)), "k": 100}
        exact_ids = None
        for kind in ("flat", "hnsw"):
            idx = ANNIndex(V, kind)
            t = time.perf_counter()
            ids, _ = idx.search(Q, 100)
            ms = (time.perf_counter() - t) * 1000 / len(Q)
            if kind == "flat":
                exact_ids = ids
                out["flat_ms_per_query"] = round(ms, 4)
            else:
                out["hnsw_ms_per_query"] = round(ms, 4)
                out["hnsw_recall@100"] = float(np.mean([len(set(a) & set(b)) / 100 for a, b in zip(exact_ids, ids)]))
        out["note"] = "At this catalogue size exact search is already fast; HNSW matters at millions of items."
        return out

    def performance(self, version: str) -> dict:
        from ntg.platform.engine import Engine

        t = time.perf_counter()
        eng = Engine.load(self.artifact_dir, version)
        load_s = time.perf_counter() - t
        rng = np.random.default_rng(self.seed)
        users = rng.choice(np.flatnonzero(eng.cs.tt.n_events >= 5), 200, replace=False)
        lat = []
        for u in users:
            t = time.perf_counter()
            eng.recommend(int(eng.data.user_ids[u]), k=10, explain=False)
            lat.append((time.perf_counter() - t) * 1000)
        lat_e = []
        for u in users[:50]:
            t = time.perf_counter()
            eng.recommend(int(eng.data.user_ids[u]), k=10, explain=True)
            lat_e.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        eng.ranker.recommend(eng.cs, users, 10)
        batch_s = time.perf_counter() - t
        return {
            "evidence_type": "observed",
            "hardware": {"platform": platform.platform(), "processor": platform.processor() or platform.machine(),
                         "python": platform.python_version()},
            "bundle_load_seconds": round(load_s, 3),
            "bundle_size_mb": round((self.artifact_dir / version / "bundle.pkl").stat().st_size / 1e6, 1),
            "single_request_latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                                          "p99": float(np.percentile(lat, 99)), "n": len(lat)},
            "single_request_with_explanations_ms": {"p50": float(np.percentile(lat_e, 50)), "p95": float(np.percentile(lat_e, 95)), "n": len(lat_e)},
            "batch_throughput_users_per_s": round(len(users) / batch_s, 1),
            "peak_rss_mb": round(_rss_mb(), 1),
            "ann": self._ann_benchmark(eng),
            "note": "Measured in-process on the machine above; not a claim about production capacity.",
        }

"""AI Business Analyst: grounded question answering over platform evidence.

Design (deliberately conservative):

* A question is routed to one or more whitelisted *tools*; each tool reads a
  computed artifact (evaluation reports, model manifest, live aggregates) and
  returns facts with a ``source`` citation. The answer is assembled only from
  those facts — the analyst cannot produce a number that is not in a source.
* Prompt-injection defence: user text is treated as a query, never as
  instructions. Requests to ignore rules, reveal prompts/secrets, run code, or
  access per-person data are refused. Retrieved report text is never executed
  or followed.
* Questions about other companies' confidential data are declined.
* Every fact is labelled observed / modeled / simulated.

An LLM can later be layered on top to *phrase* answers from the same cited
facts (local or approved provider only); retrieval and grounding stay here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

INJECTION = re.compile(
    r"(ignore (all|any|the|previous|prior)|disregard|system prompt|developer message|you are now|jailbreak|"
    r"reveal|print (the|your) (prompt|instructions)|api[_ -]?key|password|secret|token|exec\(|import os|rm -rf|"
    r"drop table|<script)", re.I)
EXTERNAL = re.compile(r"\b(netflix|disney|hulu|amazon|prime video|hbo|max|apple tv|paramount|peacock)('s)?\b.*"
                      r"\b(internal|confidential|subscribers?|revenue|churn|data|numbers)\b", re.I)
PERSONAL = re.compile(r"\b(email|address|phone|name of|who is user|identify user|personal data of)\b", re.I)


@dataclass
class Fact:
    text: str
    source: str
    evidence_type: str
    value: float | str | None = None


@dataclass
class Answer:
    question: str
    answer: str
    facts: list[Fact] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    refused: bool = False

    def as_dict(self) -> dict:
        return {"question": self.question, "answer": self.answer, "refused": self.refused, "topics": self.topics,
                "facts": [f.__dict__ for f in self.facts],
                "grounding": "Answer assembled only from the cited platform artifacts; no external data."}


def _pct(x):
    return f"{100 * x:.1f}%"


class BusinessAnalyst:
    TOPICS = {
        "fatigue": ("fatigue", "tired", "repetit", "saturat", "bored"),
        "strategy": ("strategy", "policy", "policies", "journey", "orchestrator", "offline result", "strongest"),
        "content": ("under-discovered", "underdiscovered", "content group", "catalog", "catalogue", "long tail",
                    "long-tail", "hidden gem", "genre coverage"),
        "exploration": ("exploration", "explore", "diversity", "what happens if", "what if", "serendip", "novelty"),
        "model": ("model version", "deployed", "which model", "version", "manifest"),
        "evaluation": ("evaluation", "metric", "ndcg", "recall", "accuracy", "benchmark", "baseline", "latest result"),
        "risk": ("churn", "disengage", "risk", "retention", "inactive"),
        "segments": ("segment", "audience", "cluster", "viewer group"),
        "tastetwin": ("tastetwin", "taste twin", "preference prediction", "drift", "taste evolution"),
        "performance": ("latency", "throughput", "performance", "speed", "memory"),
    }

    def __init__(self, engine, aggregates: dict):
        self.e = engine
        self.agg = aggregates

    def ask(self, question: str) -> dict:
        q = (question or "").strip()[:600]
        if not q:
            return Answer(q, "Please ask a question about the platform's viewers, models, content or experiments.").as_dict()
        if INJECTION.search(q):
            return Answer(q, "I can only answer analytical questions from platform data. I don't follow instructions "
                             "embedded in questions, and I never reveal prompts, credentials or system internals.",
                          refused=True).as_dict()
        if EXTERNAL.search(q):
            return Answer(q, "I don't have access to other companies' confidential or internal data, and I won't estimate "
                             "it. I can answer questions about this platform's own data and models.", refused=True).as_dict()
        if PERSONAL.search(q):
            return Answer(q, "I don't disclose information that identifies individual viewers. I can describe segments "
                             "and aggregate patterns instead.", refused=True).as_dict()
        ql = q.lower()
        topics = [t for t, kws in self.TOPICS.items() if any(k in ql for k in kws)]
        if not topics:
            return Answer(q, "I couldn't match that to the data I can access. I can answer about: recommendation fatigue by "
                             "segment, recommendation strategies (offline/simulated results), under-discovered content, "
                             "effects of more exploration, the deployed model version, evaluation results, engagement "
                             "risk, audience segments, TasteTwin accuracy, and system performance.").as_dict()
        facts: list[Fact] = []
        for t in topics[:3]:
            facts += getattr(self, f"_{t}")(ql)
        facts = [f for f in facts if f is not None]
        if not facts:
            return Answer(q, "The relevant report has not been generated yet. Run `ntg build` to produce it.", topics=topics).as_dict()
        body = " ".join(f.text for f in facts)
        return Answer(q, body, facts, topics).as_dict()

    # ------------------------------------------------------------------ tools
    def _fatigue(self, ql):
        seg = self.agg.get("fatigue_by_segment", [])
        out = []
        if seg:
            top = sorted(seg, key=lambda s: -s["mean_fatigue_risk"])
            s0 = top[0]
            out.append(Fact(f"The segment with the highest modeled fatigue risk is '{s0['name']}' "
                            f"(mean risk {_pct(s0['mean_fatigue_risk'])}, n={s0['n_viewers']} sampled viewers), versus "
                            f"'{top[-1]['name']}' at {_pct(top[-1]['mean_fatigue_risk'])}.",
                            "live aggregate: fatigue model over a viewer sample, by audience segment", "modeled",
                            s0["mean_fatigue_risk"]))
        rep = self.e.report("fatigue")
        if rep:
            te = rep["test"]
            out.append(Fact(f"On the out-of-time test, the fatigue model has AUC {te['auc']:.3f} (the single-signal "
                            f"genre-concentration baseline scores {te['baseline_recent_hhi_auc']:.3f}); fatigue is defined "
                            f"as a rating drop among viewers who stay active, so inactivity is not counted as fatigue.",
                            "reports/fatigue.json", "observed", te["auc"]))
        return out

    def _strategy(self, ql):
        rep = self.e.report("journey")
        if not rep:
            return []
        pol = rep["default_world"]["policies"]
        best = max(pol, key=lambda p: pol[p]["satisfaction_per_journey"])
        b = pol[best]
        out = [Fact(f"In the journey simulator (default assumptions), '{best}' has the highest simulated satisfaction per "
                    f"journey ({b['satisfaction_per_journey']:.2f}, 95% CI {b['satisfaction_ci95'][0]:.2f}–"
                    f"{b['satisfaction_ci95'][1]:.2f}) vs greedy relevance at "
                    f"{pol['greedy_relevance']['satisfaction_per_journey']:.2f}.", "reports/journey.json", "simulated",
                    b["satisfaction_per_journey"])]
        if "greedy_fresh" in pol and best != "greedy_fresh":
            lift = b.get("satisfaction_lift_vs_greedy_fresh")
            ci = b.get("satisfaction_lift_vs_greedy_fresh_ci95")
            if lift is not None:
                out.append(Fact(f"Against the stronger 'greedy without re-showing' baseline the lift is {lift:+.2f} "
                                f"(95% CI {ci[0]:+.2f} to {ci[1]:+.2f}). These are simulation results under stated "
                                f"assumptions, not measured causal effects.", "reports/journey.json", "simulated", lift))
        bench = self.e.report("benchmark")
        if bench:
            w = bench["results"]["warm"]
            bm = max(w, key=lambda m: w[m]["metrics"]["ndcg@10"])
            out.append(Fact(f"On the offline temporal test set, the best ranking model by NDCG@10 is '{bm}' "
                            f"({w[bm]['metrics']['ndcg@10']:.4f}).", "reports/benchmark.json", "observed",
                            w[bm]["metrics"]["ndcg@10"]))
        return out

    def _content(self, ql):
        c = self.e.bundle.get("content")
        if not c:
            return []
        out = []
        ud = c.get("under_discovered_genres", [])[:3]
        if ud:
            out.append(Fact("Under-discovered genres (larger share of the catalogue than of viewing, with above-median "
                            "ratings): " + "; ".join(f"{g['genre']} ({_pct(g['catalog_share'])} of titles vs "
                                                    f"{_pct(g['consumption_share'])} of viewing)" for g in ud) + ".",
                            "content intelligence (serving bundle)", "observed"))
        out.append(Fact(f"The most popular 20% of titles account for {_pct(c['head_share_of_consumption'])} of viewing; "
                        f"recommendations cover {_pct(c.get('recommendation_catalog_coverage', 0))} of the catalogue for a "
                        f"sample of viewers.", "content intelligence (serving bundle)", "observed",
                        c["head_share_of_consumption"]))
        return out

    def _exploration(self, ql):
        bench = self.e.report("benchmark")
        out = []
        if bench and bench.get("rerank_exploration_levels"):
            lv = bench["rerank_exploration_levels"]
            f, x = lv["familiar"], lv["exploratory"]
            out.append(Fact(f"Offline, moving from 'familiar' to 'exploratory' changes NDCG@10 from {f['ndcg@10']:.4f} to "
                            f"{x['ndcg@10']:.4f}, intra-list diversity from {f['ild@10']:.3f} to {x['ild@10']:.3f}, and "
                            f"long-tail share from {_pct(f['long_tail@10'])} to {_pct(x['long_tail@10'])}.",
                            "reports/benchmark.json (rerank_exploration_levels)", "observed", x["ndcg@10"] - f["ndcg@10"]))
        rep = self.e.report("journey")
        if rep:
            rows = rep["sensitivity_fatigue"]["rows"]
            txt = "; ".join(f"fatigue sensitivity {r['fatigue_sensitivity']}: orchestrator vs greedy-fresh "
                            f"{r['journey_orchestrator']['lift_vs_greedy_fresh']:+.2f}" for r in rows
                            if r["journey_orchestrator"].get("lift_vs_greedy_fresh") is not None)
            out.append(Fact(f"In simulation, the value of adaptive exploration depends on how strongly viewers fatigue — {txt}.",
                            "reports/journey.json (sensitivity_fatigue)", "simulated"))
        return out

    def _model(self, ql):
        m = self.e.manifest
        return [Fact(f"Deployed model version is {m['version']} (git {m['git_sha']}, created {m['created_at']}), trained on "
                     f"dataset '{m['dataset']}'{' (synthetic)' if m.get('is_synthetic') else ''} with channels "
                     f"{', '.join(m['channels'])} and a {m['ranker']['type']} ranker.", "artifacts manifest.json", "observed",
                     m["version"])]

    def _evaluation(self, ql):
        bench = self.e.report("benchmark")
        if not bench:
            return []
        w = bench["results"]["warm"]
        h, p = w["hybrid_ltr"], w["popularity"]
        bm = bench["best_single_model_warm"]
        return [Fact(f"Latest offline test ({bench['protocol']['t_test']} onwards, {h['n_users']} warm users): hybrid "
                     f"NDCG@10 {h['metrics']['ndcg@10']:.4f} (95% CI {h['ci95']['ndcg@10'][0]:.4f}–{h['ci95']['ndcg@10'][1]:.4f}), "
                     f"best single model '{bm}' {w[bm]['metrics']['ndcg@10']:.4f}, popularity baseline "
                     f"{p['metrics']['ndcg@10']:.4f}.", "reports/benchmark.json", "observed", h["metrics"]["ndcg@10"])]

    def _risk(self, ql):
        r = self.e.report("engagement_risk")
        out = []
        if r:
            g = r["gbm_calibrated"]
            out.append(Fact(f"Engagement-risk proxy (no activity in 90 days; MovieLens has no subscription data): calibrated "
                            f"model AUC {g['auc']:.3f}, ECE {g['ece']:.3f}; recency-only baseline AUC "
                            f"{r['recency_only_baseline']['auc']:.3f}; base rate {_pct(r['positive_rate'])}.",
                            "reports/engagement_risk.json", "observed", g["auc"]))
        seg = self.agg.get("risk_by_segment", [])
        if seg:
            s0 = max(seg, key=lambda s: s["mean_risk"])
            out.append(Fact(f"Highest modeled engagement risk among recently active viewers: '{s0['name']}' "
                            f"({_pct(s0['mean_risk'])}).", "live aggregate: engagement-risk model by segment", "modeled",
                            s0["mean_risk"]))
        return out

    def _segments(self, ql):
        segs = self.e.bundle.get("segments", [])
        if not segs:
            return []
        top = sorted(segs, key=lambda s: -s["n_viewers"])[:4]
        return [Fact("Largest audience segments: " + "; ".join(f"{s['name']} ({s['n_viewers']} viewers, exploration "
                                                             f"{s['avg_exploration']:.2f})" for s in top) + ".",
                     "TasteTwin k-means segments (serving bundle)", "modeled")]

    def _tastetwin(self, ql):
        r = self.e.report("tastetwin")
        if not r:
            return []
        t = r["test"]
        return [Fact(f"TasteTwin forecasts next-period genre mix with JS divergence {t['tastetwin']['js_divergence']:.4f} vs "
                     f"{t['static_profile_no_decay']['js_divergence']:.4f} for a static profile (reduction "
                     f"{r['js_reduction_vs_static']:.4f}, 95% CI {r['js_reduction_vs_static_ci95'][0]:.4f}–"
                     f"{r['js_reduction_vs_static_ci95'][1]:.4f}; {r['n_users_test']} users).", "reports/tastetwin.json",
                     "observed", r["js_reduction_vs_static"])]

    def _performance(self, ql):
        p = self.e.report("performance")
        if not p:
            return []
        lat = p["single_request_latency_ms"]
        return [Fact(f"Measured on {p['hardware']['processor']}: single-request recommendation latency p50 "
                     f"{lat['p50']:.1f} ms, p95 {lat['p95']:.1f} ms; batch throughput {p['batch_throughput_users_per_s']} "
                     f"users/s. Local measurements, not production capacity.", "reports/performance.json", "observed",
                     lat["p50"])]

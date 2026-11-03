"""Paired bootstrap significance on the TEST window: same users, per-user NDCG@10 differences.

python scripts/paired_significance.py > reports/v4/synthetic/significance.json
Re-fits the channels on TRAIN+VAL exactly as the pipeline does and uses the frozen ranker from the
current bundle (artifacts/<ds>/CURRENT), so numbers match benchmark.json.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from ntg.datasets.matrix import InteractionData
from ntg.recsys.evaluate import EvalConfig, _per_user_metrics, eval_users, holdout_relevance
from ntg.recsys.hybrid import DEFAULT_CHANNELS, ChannelSet

ds = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
d = InteractionData(f"data/v4/{ds}")
art = Path(f"artifacts/{ds}")
bundle = pickle.loads((art / (art / "CURRENT").read_text().strip() / "bundle.pkl").read_bytes())
ranker = bundle["ranker"]
params = json.loads(Path(f"reports/v4/{ds}/tuning.json").read_text())["selected"]
wt = d.window("test")
cs = ChannelSet(d, wt.fit, DEFAULT_CHANNELS, params).fit()
cfg = EvalConfig(ks=(10,), bootstrap=0, max_users=5000, seed=0)
rel = holdout_relevance(d, wt.fit, wt.holdout)
users = eval_users(d, wt.fit, rel, cfg, "warm")


def per_user(fn):
    recs = np.vstack([fn(users[s:s + 256], 10) for s in range(0, len(users), 256)])
    return _per_user_metrics(recs, rel, users, 10)["ndcg@10"]


scores = {"hybrid_ltr": per_user(lambda u, k: ranker.recommend(cs, u, k))}
for name in ("twotower", "rp3beta", "ease", "popularity"):
    if name in cs.models:
        m = cs.models[name]
        scores[name] = per_user(lambda u, k, m=m: m.recommend(u, k)[0])
rng = np.random.default_rng(0)
idx = rng.integers(0, len(users), (5000, len(users)))
out = {"evidence_type": "observed", "dataset": ds, "n_users": int(len(users)), "metric": "ndcg@10",
       "method": "paired bootstrap over users (5000 resamples); two-sided p = 2*min(P(diff<=0), P(diff>=0))",
       "mean": {k: float(v.mean()) for k, v in scores.items()}, "comparisons": {}}
for other in [k for k in scores if k != "hybrid_ltr"]:
    diff = scores["hybrid_ltr"] - scores[other]
    boot = diff[idx].mean(1)
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    out["comparisons"][f"hybrid_ltr - {other}"] = {
        "mean_diff": float(diff.mean()), "relative": float(diff.mean() / max(scores[other].mean(), 1e-12)),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))], "p_value": float(min(1.0, p))}
print(json.dumps(out, indent=1))

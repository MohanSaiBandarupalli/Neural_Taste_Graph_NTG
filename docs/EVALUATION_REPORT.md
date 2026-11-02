# Evaluation report (v4)

All numbers in this report come from `reports/v4/synthetic/*.json`, produced by:

```bash
python -m ntg.cli build --dataset synthetic        # seed 0, ~2.5 min on 4 vCPU
```

## 0. Dataset and the most important caveat

| | |
|---|---|
| Dataset | **Synthetic, MovieLens schema** (`ntg.datasets.synthetic`, seed 42). It is not MovieLens. |
| Why | `files.grouplens.org` is blocked by the build environment's network policy (HTTP 403). |
| Size | 421,127 interactions · 7,956 users · 2,979 titles · 2019-01 to 2024-12 |
| Planted mechanisms | latent themes, popularity bias, franchises, taste drift (41% of users, start time known), genre fatigue that lowers ratings and raises the risk of leaving |
| Real-data path | `make build-ml32m` runs the same pipeline on ML-32M. Those results do not exist yet. |

Synthetic data is good for checking correctness: leakage, the metric code, whether a method can recover a
signal that was planted. It is weak evidence of real-world value. Methods tuned to its mechanisms (fatigue,
drift) will look better here than they may on real data. Section 7 lists where this matters.

## 1. Protocol

* **Split:** a global temporal split. Every training event comes before every validation event, which
  comes before every test event (cut-offs at the 80th and 90th percentiles of timestamps:
  t_val = 2023-10-25, t_test = 2024-04-27). The contract is asserted in code. This replaces v1's per-user
  split, which leaked across users in time.
* **Model selection:** every grid is evaluated on VAL (`tuning.json`). Models are then refit on TRAIN+VAL
  and scored once on TEST.
* **Ranker:** channels fit on TRAIN, labels taken from VAL, early stopping on 1/7 of the VAL users. The
  frozen ranker is then applied to channels refit on TRAIN+VAL.
* **Relevance:** any test-window interaction with an item the user had not seen in the fit window. A second
  "liked-only" variant (rating ≥ 4) is also reported.
* **Users:** warm = at least 5 fit-window events (n = 1,129); cold = 1–4 events (n = 79). Users with no
  history are served by popularity and are not scored.
* **CIs:** 500-sample bootstrap over users, 95% percentile intervals.

## 2. Ranking: warm users (n = 1,129)

| Model | NDCG@10 [95% CI] | Recall@20 | HR@10 | MAP@10 | Coverage@10 | ILD@10 | Long-tail@10 | Franchise-dup |
|---|---|---|---|---|---|---|---|---|
| **hybrid_ltr** | **0.1397** [0.1312, 0.1493] | **0.1241** | **0.639** | 0.0645 | 12.9% | 0.577 | 0.1% | 31.8% |
| rp3beta | 0.1388 [0.1293, 0.1490] | 0.1182 | 0.628 | 0.0647 | 17.3% | 0.614 | 3.1% | 42.1% |
| ease | 0.1379 [0.1288, 0.1474] | 0.1190 | 0.631 | 0.0641 | 11.1% | 0.659 | 0.2% | 34.9% |
| itemknn | 0.1372 [0.1277, 0.1473] | 0.1110 | 0.625 | 0.0633 | 7.9% | 0.665 | 0.5% | 44.6% |
| ials | 0.1354 [0.1268, 0.1458] | 0.1198 | 0.609 | 0.0633 | 17.1% | 0.613 | 0.6% | 27.5% |
| neumf | 0.0843 [0.0773, 0.0907] | 0.0788 | 0.482 | 0.0342 | 55.9% | 0.622 | 30.4% | 14.9% |
| popularity | 0.0774 [0.0705, 0.0840] | 0.0653 | 0.450 | 0.0313 | 0.9% | 0.839 | 0.0% | 74.3% |
| sequential | 0.0633 [0.0572, 0.0693] | 0.0550 | 0.384 | 0.0253 | 47.1% | 0.796 | 12.2% | 15.1% |
| content | 0.0444 [0.0396, 0.0489] | 0.0445 | 0.319 | 0.0162 | 65.7% | 0.147 | 74.8% | 27.4% |
| semantic (LSA) | 0.0396 [0.0358, 0.0439] | 0.0426 | 0.303 | 0.0134 | 67.1% | 0.313 | 73.2% | 50.0% |

**Reading:**

* Every collaborative model beats popularity by about 1.8×.
* The hybrid is numerically best on NDCG, Recall and HR, but its CI overlaps RP3β, EASE, ItemKNN and iALS.
  **We do not claim the hybrid is better than a well-tuned single model for warm users on this data.**
* Why the hybrid has little to add here: the generator is a low-rank latent model plus popularity, which
  a tuned item-item model already captures. Candidate recall is only 43.9% (union of top-100 per channel),
  so more than half of the relevant items never reach the ranker.
* NeuMF (NumPy, 8 epochs) is clearly under-performing. It is reported as is, not hidden.
* Liked-only relevance (rating ≥ 4, n = 1,103): hybrid NDCG@10 0.1085, RP3β 0.1064, EASE 0.1043,
  popularity 0.0416. Same conclusion.

## 3. Ranking: cold-start users (1–4 events, n = 79)

| Model | NDCG@10 [95% CI] | HR@10 |
|---|---|---|
| **hybrid_ltr** | **0.1280** [0.0980, 0.1521] | 0.658 |
| ials | 0.0997 [0.0738, 0.1222] | 0.595 |
| popularity | 0.0986 [0.0732, 0.1264] | 0.544 |
| itemknn | 0.0983 [0.0729, 0.1227] | 0.570 |
| ease | 0.0905 [0.0682, 0.1120] | 0.582 |

This is the hybrid's largest gain (+28% relative over the best single model). It is plausible: the ranker
learns to lean on popularity and TasteTwin features when history is thin. But n = 79 and the CIs overlap,
so it is a lead to confirm on real data, not a result.

## 4. Exploration levels (same ranker, different re-ranking; n = 1,500)

| Level | NDCG@10 | ILD@10 | Coverage@10 | Long-tail@10 | Franchise-dup |
|---|---|---|---|---|---|
| familiar | 0.1399 | 0.578 | 12.9% | 0.1% | 31.8% |
| **balanced** (default) | 0.1332 | 0.685 | 13.1% | 0.1% | **0.0%** |
| exploratory | 0.1006 | 0.880 | 12.9% | 0.2% | 0.0% |
| unexpected | 0.0760 | 0.894 | 14.3% | 0.7% | 0.1% |

* The default "balanced" mode removes franchise repetition entirely and raises diversity by 19%, at a cost
  of 4.8% NDCG. That trade-off is chosen on purpose, and we state it.
* **Gap:** the exploratory modes barely raise long-tail share or coverage. They re-rank candidates that
  head-heavy channels retrieved. A long-tail retrieval channel for these modes is the next fix (ROADMAP).

## 5. TasteTwin

### 5a. Future-preference forecast (`tastetwin.json`)

Task: predict each user's genre distribution in the test window from fit-window data only. There are
1,101 users with at least 5 fit events and at least 3 holdout events. Half-lives and the blend weight were
selected on VAL (long 180 days, short 20 days, blend 0.0).

| Predictor | JS divergence ↓ [95% CI] | Top-3 genre recall ↑ |
|---|---|---|
| global popularity | 0.2497 [0.2432, 0.2565] | 0.411 |
| static profile (no decay) | 0.1909 [0.1837, 0.1983] | 0.579 |
| short-term only | 0.2360 [0.2281, 0.2439] | 0.536 |
| **TasteTwin** | **0.1868** [0.1798, 0.1940] | **0.591** |

Paired JS reduction against the static profile: **0.0041 [0.0027, 0.0056]**. The CI excludes 0.

The improvement comes from **time decay**. The selected blend puts zero weight on the short-term profile:
on this data, short-term state is useful for *detecting change* (5b), not for *forecasting the mix*.

### 5b. Drift detection against ground truth (synthetic only)

Task: separate the 2,488 users with planted drift from 4,231 non-drifting users, using the a-priori default
config (not tuned for this task). AUC is 0.602 for the instability score and 0.609 for the largest
absolute momentum. Above chance, but modest. Drift is planted in *theme* space and observed through
genres, which lowers the achievable signal.

### 5c. Online update correctness

A unit test proves that the O(new events) incremental update equals a full refit to 1e-6
(`tests/v4/test_intelligence.py::test_tastetwin_incremental_update_equals_refit`).

## 6. Discovery and risk models

| Model | Metric (test) | Baseline | Verdict |
|---|---|---|---|
| Fatigue (logistic, 7 signals; label = rating drop > 0.35★ among users who stay active, n = 1,180, base rate 16.8%) | AUC 0.627, PR-AUC 0.250 | genre concentration alone: AUC 0.634 | **No gain over the one-signal baseline.** Kept only for per-signal explanations. |
| Hidden interests (users who newly adopt a genre, n = 203) | hit@3 0.586, precision@3 0.204 | popular genres you haven't explored: 0.640 / 0.227 | **Below baseline.** In the generator, emerging themes are random, so there is no personal signal to find. Needs real data. |
| Engagement-risk proxy (no activity in 90 days; trained at t_val, tested at t_test; n = 1,572, base rate 25.6%) | calibrated GBM AUC 0.798, PR-AUC 0.658, Brier 0.139, ECE 0.036, top-decile lift 3.2× | recency only: AUC 0.790. Logistic: AUC 0.804 | Mostly a recency signal. Logistic slightly beats the GBM. |

Engagement-risk AUC by tenure: new (< 90 days) 0.776 · established 0.821 · loyal (> 1 year) 0.744.

## 7. Journey Orchestrator (simulated; `journey.json`)

**Evidence type: simulated.** These are not causal estimates.

* World model: an oracle (EASE + RP3β) fitted on *all* data, including the future, gives acceptance
  propensities. Policies see only past-only models and in-journey feedback.
* Fatigue, repetition and leaving rules are explicit `WorldConfig` fields.
* 600 users, 25 sessions, slates of 5. Lifts are paired bootstrap over the same users with the same seed.

| Policy | Session-1 accept | Satisfaction / journey [95% CI] | Sessions kept | New genres | Lift vs greedy-fresh |
|---|---|---|---|---|---|
| greedy relevance (re-shows) | 0.940 | 6.33 [6.00, 6.68] | 12.1 | 1.08 | −3.49 |
| **greedy-fresh** (strong baseline: no re-show) | 0.957 | 9.82 [9.36, 10.26] | 13.3 | 1.76 | 0 |
| MMR diversity | 0.942 | 7.00 [6.61, 7.30] | 12.4 | 1.56 | −2.82 |
| ε-greedy | 0.892 | 5.67 [5.34, 6.04] | 11.5 | 1.07 | −4.15 |
| genre Thompson sampling | 0.950 | 6.76 [6.44, 7.11] | 12.1 | 0.93 | −3.06 |
| **journey orchestrator** | 0.955 | **11.71** [11.20, 12.30] | **15.4** | **2.56** | **+1.89** [1.18, 2.51] |

Sensitivity to the world's fatigue assumption (orchestrator vs greedy-fresh):

| fatigue_sensitivity | 0.0 | 0.5 | 1.0 | 1.5 |
|---|---|---|---|---|
| lift [95% CI] | −0.12 [−0.80, 0.63] | +0.99 [0.22, 1.75] | +1.89 [1.18, 2.51] | +1.61 [0.96, 2.26] |

Sensitivity to the repetition penalty: when re-shown items carry no penalty (1.0), MMR (+2.26) beats the
orchestrator (+1.89).

**Honest reading:**

1. Most of the gain over naive greedy (about 3.5 of 5.4) comes from simply **not re-showing rejected
   items**. Greedy-fresh captures that, which is why it is the reference baseline.
2. On top of that, the adaptive orchestrator adds about +1–2 satisfaction **only when viewers fatigue**.
   With no fatigue it is statistically tied. That is by design: it learns a per-viewer fatigue belief from
   accept/reject feedback, and an earlier non-adaptive version lost 1.6 here.
3. The synthetic generator plants fatigue, so real-data validation means estimating fatigue sensitivity
   from logs, or an online A/B test (an experiment template is seeded in the app).

Development history, kept for honesty: the first version reported +5.3 over naive greedy. Stress tests
showed most of that was the repetition rule, which led to the greedy-fresh baseline and the adaptive policy.

### Off-policy evaluation check

The logging policy is ε-greedy (ε = 0.3) over the top 20, with known propensities, on 20,000 logged
impressions. The target policy is greedy top-1.

| | value |
|---|---|
| true target value (from the oracle) | 0.5949 |
| IPS | 0.5976 [0.5885, 0.6071] |
| SNIPS | 0.5919 [0.5838, 0.6004] |
| naive logged average | 0.5736 (biased) |

Both estimators recover the truth, and the naive average does not. This validates the OPE machinery before
it is used on real logs.

## 8. Reproducing

```bash
python -m ntg.cli build --dataset synthetic           # all of the above
pytest tests/v4                                       # correctness tests (leakage, metrics, online==batch, OPE)
```

Seeds: generator 42, pipeline 0. Runtimes are in `performance.json`.

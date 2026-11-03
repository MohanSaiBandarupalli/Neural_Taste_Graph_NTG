# Model card: NTG v4

**Version:** set at build time (`artifacts/<dataset>/CURRENT`). The current local build is
`synthetic-20261102182943-b2e056e0`.
**Owner:** NTG project. **Status:** research and demo. Not validated on real viewers.

## Intended use

* Personalised title recommendations with controllable exploration and grounded explanations.
* Viewer-level taste summaries (TasteTwin), shown to the viewer about themselves.
* Aggregate, pseudonymous analytics for a catalogue operator: segments, risk and fatigue proxies,
  content discovery gaps, policy simulation.

**Out of scope:** decisions about individuals beyond recommendations (pricing, eligibility, credit);
predicting real subscription cancellation from MovieLens-style data; any claim about another company's
users or systems.

## Components

| Component | Purpose | Training | Inference | Artifact |
|---|---|---|---|---|
| Popularity | baseline and cold start | decayed counts (half-life tuned on VAL) | O(I) | bundle |
| ItemKNN | "because you watched" anchors and a channel | cosine with shrinkage, top-k = 20 | sparse x·S | bundle |
| EASE | strong item-item channel | closed form, λ = 1500 (VAL) | x·B (dense I×I) | bundle |
| RP3β | graph-walk channel | α = 0.8, β = 0.6 (VAL) | sparse | bundle |
| iALS | latent factors, user embedding for neighbours | 16 factors, 8 ALS sweeps | U·Vᵀ; fold-in for new users | bundle |
| NeuMF | neural CF (GMF + MLP, BPR loss) | NumPy, Adam, 8 epochs | per-user MLP | bundle (benchmark only) |
| Content TF-IDF | cold items, long tail | genres, tags, decade | profile cosine | bundle |
| Semantic LSA | NL search and a channel | TF-IDF → 48-d SVD | cosine | bundle |
| Sequential | short-term intent | lag-1..3 transitions within 60 days | last-5 items · T | bundle |
| Hybrid ranker | final ranking | LightGBM LambdaMART on 38 features; VAL labels; early stopping | TreeSHAP for explanations | bundle |
| TasteTwin | dynamic viewer state | decayed sufficient statistics | O(1) online update | bundle |
| Fatigue model | fatigue risk and its drivers | logistic, 7 standardised signals | per viewer | bundle |
| Engagement-risk proxy | disengagement risk | isotonic-calibrated HistGBM (plus logistic for explanations) | per viewer | bundle |

### Why there are no PyTorch models

The PyTorch wheel index and the CUDA PyPI builds were impractical in the build environment, and a CPU-only
laptop target was a design goal. NeuMF is implemented in NumPy with manual gradients. A transformer
sequential model (SASRec) and a two-tower retriever are on the roadmap, to be added only if they beat the
tuned baselines on real data.

## Training data

Synthetic data with the MovieLens schema (see EVALUATION_REPORT §0). With real MovieLens: ratings from
users who opted into a research dataset. That data is not representative of any streaming service's
users. It skews toward older, frequent raters, has only English titles, and contains only films.

## Evaluation summary

See `docs/EVALUATION_REPORT.md`. Key points:

* Hybrid ≈ best single model for warm users. It looks better for cold-start users, but n is small.
* TasteTwin gives a small, significant forecasting gain from time decay.
* The fatigue model and hidden interests do not beat their baselines on synthetic data.
* The journey-policy gains are simulated and depend on whether viewers fatigue.

## Limitations and risks

* **Synthetic evidence only.** All metrics are from generated data.
* **Popularity bias:** collaborative channels concentrate exposure (Gini about 0.95). The exploratory
  modes do not yet fix long-tail exposure.
* **Explanations** are faithful to model internals but simplified. The SHAP drivers describe the ranker,
  not the candidate generators.
* **Fatigue is a proxy.** It is a rating drop among viewers who stay active. Inactivity is never used as
  evidence.
* **Engagement risk is a proxy** for disengagement, not cancellation. Using it to target retention offers
  needs a causal (uplift) design, not just risk scores.
* **Feedback loops:** online use would bias future training data toward what was recommended.
  Mitigations: exploration modes, logged propensities (the OPE machinery exists), and a holdout policy.
* **Fairness:** no protected attributes exist in the data. Item-side fairness is monitored only through
  coverage and long-tail share.

## Versioning and reproducibility

Every build writes `manifest.json` with the version, git SHA, data meta, selected hyper-parameters, test
metrics and the bundle SHA-256. The engine refuses to load a bundle whose checksum does not match.
Reproduce with `python -m ntg.cli build --dataset synthetic` (seeds are fixed).

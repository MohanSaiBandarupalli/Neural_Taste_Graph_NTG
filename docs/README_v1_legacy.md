# 🎬 Neural Taste Graph (NTG)
### A Production Science–Style Machine Learning Platform for Personalization, Retention & Decision Intelligence

---

##  Executive Summary

**Neural Taste Graph (NTG)** is an end-to-end, production-inspired machine learning system designed to demonstrate ** ML engineering rigor**.

Unlike typical recommender demos, NTG is built as a **decision-support platform**, reflecting how ML is actually used inside organizations like Netflix—to inform **personalization, retention, experimentation, and financial planning**.

The system emphasizes:
- Leakage-safe data pipelines
- Scalable analytics (DuckDB + Parquet)
- Interpretable representations before complex models
- Reproducibility, observability, and evaluation
- End-to-end ownership from raw data to decision artifacts

---

##  Problem Framing (Production Science Perspective)

Most ML projects ask:
> *“How do we build the best model?”*

NTG asks:
> *“How do we design ML systems whose outputs decision-makers can trust?”*

This framing mirrors **Production Science teams**, where:
- ML augments human decisions
- Offline correctness matters more than optimistic metrics
- Evaluation, calibration, and interpretability are first-class concerns



##  What NTG Does (End-to-End)

NTG builds an **offline ML decision pipeline** that transforms raw interaction data into:

- Personalized ranking scores
- User taste representations
- Churn risk probabilities
- Revenue / retention risk signals
- Evaluation & calibration reports

These outputs are structured to plug into:
- Experimentation frameworks
- Analyst workflows
- Planning and prioritization processes

---

##  System Architecture

### High-Level Pipeline
```text
Raw Interaction Events
        │
        ▼
DuckDB Ingestion
(SQL + Parquet, scalable analytics)
        │
        ▼
Chronological Train / Validation / Test Splits
(Leakage-safe by construction)
        │
        ├──► Feature Engineering
        │        • User aggregates
        │        • Recency & frequency signals
        │        • Item statistics
        │
        ├──► Item–Item Similarity Graph
        │        • Co-occurrence modeling
        │        • Scale guardrails (power-user caps, top-K pruning)
        │
        ├──► Taste Representation
        │        • Interpretable taste axes
        │        • Optional latent embeddings
        │
        ├──► Personalized Ranking
        │
        ├──► Churn Risk Modeling
        │
        ▼
Decision Artifacts
        • Ranked content candidates
        • Churn probabilities
        • Calibration curves & metrics

```



##  Leakage Safety (Critical Design Principle)

All modeling decisions in NTG enforce **strict temporal correctness**:

- Chronological splits are performed once and reused everywhere
- **Only TRAIN data** is used to:
  - Build features
  - Construct graphs
  - Learn embeddings
  - Define churn labels
- Validation and test data are **never** used for feature generation

This mirrors real Production Science review standards.

---

##  Key Components

### 1️ Data Ingestion & Splitting
- DuckDB used as the analytical engine
- Parquet as the storage format
- Deterministic, time-aware splits
- Scales beyond pandas-only workflows

**Location**
src/ntg/pipelines/build_dataset_duckdb.py
data/processed/splits/


---

### 2️ Feature Engineering
- User-level aggregates (activity, recency, variance)
- Item-level statistics
- Interaction-level signals
- Feature manifest with metadata

**Location**
src/ntg/features/
data/features/




### 3️ Item–Item Similarity Graph
- Co-occurrence-based similarity
- Power-user guardrails
- Top-K pruning to avoid quadratic blowups

**Location**
src/ntg/graph/
outputs/graph/




### 4️ Taste Representation
- Interpretable “taste axes” derived from behavior
- Optional latent embeddings (SVD-based)
- Designed for explainability before complexity

**Location**
src/ntg/features/taste_axes.py
src/ntg/embeddings/
outputs/embeddings/




### 5️ Personalized Ranking
- Candidate scoring using learned representations
- Deterministic outputs
- Designed for offline evaluation & experimentation

**Location**
src/ntg/ranking/
outputs/ranking/




### 6️ Churn Risk Modeling
- Inactivity-based churn labeling
- Supervised churn probability estimation
- Evaluation & calibration reports

**Location**
src/ntg/churn/
outputs/churn/



### 7️ Evaluation & Calibration
- Metrics reported as versioned JSON artifacts
- Calibration curves & Expected Calibration Error (ECE)
- Schema validation for downstream consumers

**Location**
src/ntg/evaluation/
outputs/reports/

---

##  Configuration & Reproducibility

NTG is **fully config-driven**.

configs/
├── base.yaml
├── dev.yaml
└── prod.yaml


- Environment-specific overrides
- Deterministic reruns
- CI uses dev config for stability

---

##  Command-Line Interface

A real CLI is provided to mirror internal tooling.

### Run everything
```bash
poetry run ntg run-all
Run step-by-step
poetry run ntg build-dataset
poetry run ntg build-features
poetry run ntg build-graph
poetry run ntg rank
poetry run ntg score-users
Use a specific config
poetry run ntg --config configs/dev.yaml run-all
Testing & CI
Test Coverage
Unit tests: schemas, metrics, leakage checks

Integration tests: pipeline smoke tests

poetry run pytest
CI Signals
Fast unit tests on PRs

Nightly end-to-end pipeline on synthetic data

Deterministic failures for regressions

Repository Structure
.
├── configs/
├── data/
│   ├── external/
│   ├── processed/
│   └── features/
├── outputs/
│   ├── graph/
│   ├── embeddings/
│   ├── churn/
│   └── reports/
├── reports/figures/
├── src/ntg/
│   ├── cli.py
│   ├── settings.py
│   ├── logging.py
│   ├── pipelines/
│   ├── features/
│   ├── graph/
│   ├── embeddings/
│   ├── churn/
│   └── evaluation/
├── tests/
│   ├── unit/
│   └── integration/
├── Dockerfile
├── Makefile
└── README.md

``` 

Outputs

After a successful run:

data/features/*.parquet – engineered features

outputs/graph/*.parquet – similarity graph

outputs/embeddings/*.parquet – embeddings

outputs/churn/*.parquet – churn scores

outputs/reports/*.json – metrics & calibration

reports/figures/ – diagnostic plots

--- 

## Dataset Note
MovieLens is used only as a public proxy.

This project is not about movie ratings.

It is about:

Personalization system design

Retention modeling patterns

Decision-oriented ML pipelines

All architectural choices generalize directly to real production event data.

---

# Conclusion & Positioning

Neural Taste Graph (NTG) was intentionally designed not as a single recommender model, but as a Production Science–style machine learning platform.

The project demonstrates how modern ML systems should be built when the goal is decision support, not leaderboard performance:

Models are leakage-safe by construction

Pipelines are reproducible, observable, and testable

Outputs are structured to support experimentation, retention strategy, and planning

Interpretability and calibration are treated as first-class requirements

This mirrors how ML is applied in real-world environments
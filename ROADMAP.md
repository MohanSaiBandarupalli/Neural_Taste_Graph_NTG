# Roadmap

## Completed (v4, verified in this repository)

- [x] Phase 0: audit, baseline tests (82 passed, 11 failed, 15 skipped before any change), branch
- [x] Phase 1: fixed the test sandbox (tests wrote into the repo), corrupted YAML configs, and a churn
      crash on a single class (the v1 churn model's label leakage is flagged in its report)
- [x] Phase 2: global temporal split, 9 retrieval models, hybrid LambdaMART, re-ranking, grounded
      explanations, evaluation with CIs and beyond-accuracy metrics
- [x] Phase 3: TasteTwin (online update, uncertainty, evolution) with temporal and ground-truth evaluation
- [x] Phase 4: journey simulator, 6 policies, adaptive orchestrator, sensitivity analysis, IPS / SNIPS
- [x] Phase 5: engagement-risk proxy (out-of-time, calibrated), fatigue, hidden interests, segments,
      content intelligence, business simulator, grounded analyst
- [x] Phase 6: FastAPI platform (auth, RBAC, tenancy, API keys, rate limit, consent, export/delete, audit,
      usage)
- [x] Phase 7–8: Next.js consumer (12 pages) and enterprise (15 pages) apps, all wired to the API,
      Playwright-verified with zero console errors
- [x] Phase 9: Docker (both images built and run), compose, Alembic, CI, ruff, versioned artifacts with
      checksums, a performance pass (3× faster explained recommendations)
- [x] Phase 10: documentation set, interview guide, startup strategy

## Completed in v5 (branch `claude/ntg-v5-netflix-grade`)

- [x] SASRec and an inductive two-tower retriever (PyTorch, optional group), faiss HNSW ANN index
- [x] Hybrid ranker trained on two time windows: warm NDCG@10 0.1495, +7.7% over RP3β (paired p < 0.001)
- [x] Long-tail channel and reserved long-tail slots: the exploratory modes reach 26–37% long tail
- [x] Candidate recall 43.9% → 49.0%
- [x] Streaming TasteTwin slice (event log → idempotent consumer → state store), tested and benchmarked
- [x] Security: httpOnly cookies + CSRF, rotating refresh tokens with reuse detection, logout-everywhere,
      Redis-shared rate limiting, login brute-force limit (browser-verified)
- [x] One-command ML-32M kit (`scripts/run_ml32m.py`) and paired significance script

## Next (ordered by expected value)

1. **Real benchmark:** run `python scripts/run_ml32m.py` and publish `reports/v4/ml-32m` next to the synthetic
   results. This is the most important missing evidence.
2. **Cache user vectors** for known viewers (refreshed by the streaming consumer) to win back the v5
   latency (p50 7 → 15 ms).
3. **Raise candidate recall** above 49%: learned per-channel quotas, more two-tower candidates.
4. **Judge SASRec on ML-32M**, where sequential signal exists; drop it from the channel set if it doesn't
   beat the Markov baseline there.
5. **Fatigue from real logs:** estimate the acceptance decline after same-genre streaks to calibrate the
   simulator's `fatigue_sensitivity`.
6. **Security:** password reset and email verification, per-plan quotas, dependency scanning in CI.
7. **Per-tenant ingestion and training jobs** (upload → validate → build → evaluation gate → promote).
8. **LLM phrasing layer** for the analyst over the same cited facts. Local or approved provider only.
9. **Scale-out** components from ARCHITECTURE §6, built only when a customer workload needs them.

## Known limitations

- All evidence is from synthetic data.
- The hybrid ties the two-tower model alone for warm users (it beats every classical model).
- The fatigue and hidden-interest models do not beat their baselines on synthetic data.
- Journey results are simulated.
- The dense EASE / item-item models need item filtering above about 25k items.

## IP

The technical invention dossier was written to `docs/confidential/TECHNICAL_INVENTION.md`, which is
**git-ignored** because this repository is public. Note that the source code in a public repository
already discloses the implementation. Decide on an IP strategy with a qualified patent attorney before
pushing, or make the repository private. No novelty or patentability is claimed.

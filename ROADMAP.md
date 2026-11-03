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

## Next (ordered by expected value)

1. **Real benchmark:** run `make build-ml32m` and publish `reports/v4/ml-32m` next to the synthetic
   results. This is the most important missing evidence.
2. **Long-tail candidate channel** for the exploratory modes (they currently do not raise long-tail
   share).
3. **Raise candidate recall** above 44%: more candidates per channel, a two-tower ANN channel, and
   learned channel quotas.
4. **SASRec / GRU4Rec baseline** (requires PyTorch). Keep it only if it beats the tuned baselines on
   ML-32M.
5. **Fatigue from real logs:** estimate the acceptance decline after same-genre streaks to calibrate the
   simulator's `fatigue_sensitivity`.
6. **Security hardening:** httpOnly cookies and CSRF, refresh and revocation, Redis rate limiting,
   dependency scanning in CI.
7. **Per-tenant ingestion and training jobs** (upload → validate → build → evaluation gate → promote).
8. **LLM phrasing layer** for the analyst over the same cited facts. Local or approved provider only.
9. **Scale-out** components from ARCHITECTURE §6, built only when a customer workload needs them.

## Known limitations

- All evidence is from synthetic data.
- The hybrid ties the best single model for warm users.
- The fatigue and hidden-interest models do not beat their baselines on synthetic data.
- Journey results are simulated.
- The JWT is stored in localStorage (demo only).
- The dense EASE / item-item models need item filtering above about 25k items.

## IP

The technical invention dossier was written to `docs/confidential/TECHNICAL_INVENTION.md`, which is
**git-ignored** because this repository is public. Note that the source code in a public repository
already discloses the implementation. Decide on an IP strategy with a qualified patent attorney before
pushing, or make the repository private. No novelty or patentability is claimed.

# Progress log (resume point)

**Branch:** `claude/ntg-v4-entertainment-intelligence-5xuk0q`. The session's designated branch was used
instead of `feature/ntg-v4-entertainment-intelligence`.
**State:** v4 (phases 0–10) is in PR #1. **v5** is on `claude/ntg-v5-netflix-grade`: deep models, two-window
ranker, long-tail channel, streaming TasteTwin, and security hardening (see ROADMAP "Completed in v5").
Tests: 157 passed locally with the deep group installed; the deep tests skip without PyTorch.

## Blockers

1. ~~`git push` returned 403~~: resolved once GitHub access was granted; both branches are pushed.
2. **The MovieLens download host was blocked** in the build environment, so all results are synthetic. To
   produce real results, place ML-32M at `data/external/movielens/ml-32m/` and run `make build-ml32m`.

## How to resume

```bash
poetry install --with dev && (cd frontend && npm ci)
make build                       # regenerates data/v4, artifacts/, reports/v4/synthetic
make test                        # 157 passed, 13 skipped with the deep group (deep tests skip without it)
make api & make web              # http://localhost:3000
```

## Next commands (top of ROADMAP)

```bash
make build-ml32m                                  # real benchmark
python -m ntg.cli build --dataset ml-32m ...      # see DEPLOYMENT.md §4
```

## Verification record (this session)

| Check | Result |
|---|---|
| Baseline before changes | 82 passed, 11 failed, 15 skipped |
| v4 test suite | 139 passed, 13 skipped (`pytest -q -W ignore`) |
| v5 test suite | 157 passed, 13 skipped with PyTorch installed (v4 suite subset 62/62 under two-window training) |
| v5 browser check | cookie session, CSRF writes, silent refresh after access expiry, logout clears all cookies; 0 console errors |
| v5 significance | hybrid vs RP3β +7.7% NDCG@10, paired bootstrap p < 0.001 (`significance.json`) |
| ruff (v4 code) | clean |
| Frontend `tsc --noEmit` and `next build` | clean, 29 routes |
| Playwright walk-through of every page | 0 console errors; screenshots in docs/screenshots |
| Docker API image | built, ran as non-root, served /health and the analyst |
| Docker web image | built, served, proxied to the API |
| Alembic | upgrade → downgrade → upgrade OK |
| Full pipeline | 136.9 s; reports in reports/v4/synthetic |
| API load test | 62 req/s at p50 16 ms (1 worker, concurrency 1), 0 errors |

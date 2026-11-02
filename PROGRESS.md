# Progress log (resume point)

**Branch:** `claude/ntg-v4-entertainment-intelligence-5xuk0q`. The session's designated branch was used
instead of `feature/ntg-v4-entertainment-intelligence`.
**State:** all phases 0–10 complete, with all tests passing (139 passed, 13 skipped: legacy contract tests
that need ML-32M artifacts).

## Blockers

1. **`git push` returns 403.** The Claude GitHub App is not installed or authorised for
   `MohanSaiBandarupalli/Neural_Taste_Graph_NTG`. Every commit is local only until access is fixed:
   reconnect at https://claude.ai/connect-github or install the app, then
   `git push -u origin claude/ntg-v4-entertainment-intelligence-5xuk0q`.
2. **The MovieLens download host was blocked** in the build environment, so all results are synthetic. To
   produce real results, place ML-32M at `data/external/movielens/ml-32m/` and run `make build-ml32m`.

## How to resume

```bash
poetry install --with dev && (cd frontend && npm ci)
make build                       # regenerates data/v4, artifacts/, reports/v4/synthetic
make test                        # 139 passed, 13 skipped
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
| Final test suite | 139 passed, 13 skipped (`pytest -q -W ignore`) |
| ruff (v4 code) | clean |
| Frontend `tsc --noEmit` and `next build` | clean, 29 routes |
| Playwright walk-through of every page | 0 console errors; screenshots in docs/screenshots |
| Docker API image | built, ran as non-root, served /health and the analyst |
| Docker web image | built, served, proxied to the API |
| Alembic | upgrade → downgrade → upgrade OK |
| Full pipeline | 136.9 s; reports in reports/v4/synthetic |
| API load test | 62 req/s at p50 16 ms (1 worker, concurrency 1), 0 errors |

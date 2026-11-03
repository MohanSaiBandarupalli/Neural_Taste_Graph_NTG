# Contributing

## Workflow

1. Branch from `main` (`feature/<topic>`).
2. `make install`, then `make build-quick` (about 1 minute) to get a local bundle.
3. Write the change **and a test**. v4 tests live in `tests/v4`. The session fixture builds a tiny
   synthetic bundle, so no download is needed.
4. `make lint && make test` must pass. CI runs ruff, pytest, a migration round-trip, the frontend
   typecheck and build, and the Docker builds.
5. Keep commits small, with a message that says *why*.

## Rules for results

* Every number in documentation must come from a file in `reports/v4/` produced by `ntg build`. Include
  the command.
* Select on VAL and report on TEST, once. No test-set tuning.
* Report the baseline, the CI and `n`. Negative results stay in the report.
* Label every analytic output `observed`, `modeled` or `simulated`.
* Never present synthetic-data results as MovieLens or real-world results.

## Adding a model

1. Subclass `Recommender` in `ntg/recsys/models.py` and implement `_fit`, `score` and `score_history`.
2. Register it in `MODEL_REGISTRY`, add a tuning grid in `pipelines/v4.py`, and add it to the benchmark.
3. Add it to the parametrised tests in `tests/v4/test_data_and_models.py`. If it can be a channel, include
   the online-equals-batch test.

## Schema changes

Edit `ntg/api/db.py`, then run
`NTG_DATABASE_URL=sqlite:///tmp.db alembic revision --autogenerate -m "..."`, review the migration, and
commit it.

## Style

Python: ruff (config in pyproject). Type hints on public functions. TypeScript: strict mode; no `any` in
new shared types.

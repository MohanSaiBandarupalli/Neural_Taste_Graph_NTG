# Deployment

**Status:** NTG has been run locally and in Docker inside the build environment. It has **not** been
deployed to any cloud environment.

## 1. Local (no Docker)

Requirements: Python 3.11, Poetry 2.x, Node 20+.

```bash
make install          # poetry install --with dev ; cd frontend && npm ci
make build            # ~2.5 min: synthetic data -> models -> reports -> artifacts/synthetic
make migrate          # optional: Alembic schema (the API also creates tables on first start)
make api              # :8000  (OpenAPI at /docs)
make web              # :3000  (proxies /api -> :8000)
```

Demo accounts (development only): viewer@demo.ntg, analyst@demo.ntg, admin@demo.ntg, all with password
`demo-password-123`.

## 2. Docker Compose

```bash
docker compose --profile build run --rm build-models   # writes ./data, ./artifacts, ./reports
docker compose up -d api web                           # http://localhost:3000
```

* The web image bakes in the API origin at build time (`NTG_API_URL` build argument), because Next.js
  resolves `rewrites()` during the build.
* PostgreSQL: `NTG_DATABASE_URL=postgresql+psycopg://ntg:ntg@postgres:5432/ntg docker compose --profile postgres up -d`,
  then `alembic upgrade head` in the api container.
* In the build environment, both images were built and run: the API answered `/health` and analyst
  queries as non-root, and the web app served and proxied. Note that environment needed proxy CA
  injection, which is not part of the repository.

## 3. Configuration

| Variable | Default | Notes |
|---|---|---|
| `NTG_ENV` | development | `production` disables demo seeding and requires a secret |
| `NTG_SECRET_KEY` | (ephemeral in dev) | **required** in production; 32+ random bytes |
| `NTG_DATASET` | synthetic | selects `artifacts/<dataset>` |
| `NTG_ARTIFACT_ROOT` | artifacts | |
| `NTG_DATABASE_URL` | sqlite:///data/v4/ntg_app.db | Postgres recommended for production |
| `NTG_RATE_LIMIT_PER_MINUTE` | 240 | per principal |
| `NTG_CORS_ORIGINS` | localhost:3000 | comma-separated |
| `NTG_SEED_DEMO` | 1 | ignored in production |

## 4. Real MovieLens 32M

The dataset licence forbids redistribution, so download it yourself from GroupLens.

```bash
unzip ml-32m.zip -d data/external/movielens/
python -m ntg.cli build --dataset ml-32m --source data/external/movielens/ml-32m --user-sample 0.1 --min-item-support 20
NTG_DATASET=ml-32m make api
```

## 5. Suggested cloud path (not executed; needs your approval for any paid resources)

* **Smallest credible setup:** one container service (Cloud Run, ECS Fargate or Fly) for the API with
  2 GB RAM and 2 workers, a static or Node host for the web app, managed Postgres, and artifacts in object
  storage pulled at startup with the checksum verified.
* **Retraining:** a scheduled job running `ntg build`, which uploads the artifacts and manifest to object
  storage and flips `CURRENT` after an evaluation gate (for example, hybrid NDCG@10 must not regress
  beyond its CI).
* **Before going public:** complete the SECURITY.md gaps 1–3 and 6–8.
* **Scaling beyond one node:** see ARCHITECTURE §6.

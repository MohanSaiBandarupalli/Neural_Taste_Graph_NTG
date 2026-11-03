# API reference (v1)

Base URL in local development is `http://127.0.0.1:8000`. Interactive OpenAPI docs are at `/docs`, and the
schema is at `/openapi.json`. The web app reaches the API through the same-origin proxy `/api/*`.

**Auth**

* Viewer, analyst and admin routes take `Authorization: Bearer <JWT>`, issued by `/v1/auth/*`. Tokens are
  HS256 with an expiry and are signed with `NTG_SECRET_KEY`.
* B2B routes take `X-API-Key: ntg_…`. Keys are created by an admin and only their SHA-256 hash is stored.

**Errors** are `{"detail": "..."}` with the usual codes: 401 for unauthenticated, 403 for the wrong role,
404 for not found *or not yours* (tenant isolation never confirms that a resource exists), 409 for a
conflict, 422 for validation errors, and 429 for rate limiting (with a `Retry-After` header).

**Evidence labels.** Analytic responses carry `evidence_type`: `observed`, `modeled` or `simulated`.

## Public

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | status, model version, uptime |
| GET | `/v1/meta` | dataset, genres, exploration levels, data notice |
| POST | `/v1/auth/signup` | `{email, password(≥8), display_name?, consent_personalization, consent_analytics}` → `{token, user}`. Creates a viewer in the consumer org with one profile. |
| POST | `/v1/auth/login` | `{email, password}` → `{token, user}` |
| GET | `/v1/catalog/genres` | |
| GET | `/v1/catalog/browse?genre=&k=&offset=` | popularity-ordered |
| GET | `/v1/catalog/search?q=` | title matches plus semantic (LSA) matches; `parsed` shows how the query was read |
| GET | `/v1/catalog/items/{item_id}` | item card plus `collaborative` and `semantic` neighbours |

## Viewer (JWT; a viewer can only reach their own profiles)

| Method | Path | Notes |
|---|---|---|
| GET | `/v1/me` | user plus organisation |
| PATCH | `/v1/me/consent` | `{consent_personalization?, consent_analytics?}`. When personalisation is off, recommendations ignore all history. |
| GET | `/v1/me/export` | every profile and event (JSON) |
| DELETE | `/v1/me` | deletes the account, its profiles and events, and the engine's in-memory state |
| GET/POST | `/v1/profiles` | POST `{name, color, favorite_genres[]}`. Favourite genres become explicit "like" signals. Maximum 5 profiles. |
| DELETE | `/v1/profiles/{pid}` | |
| GET | `/v1/profiles/{pid}/home?level=` | personalised rows: top picks, because-you-watched, explore, hidden interests, emerging interest, trending |
| POST | `/v1/profiles/{pid}/recommendations` | `{k, level: familiar\|balanced\|exploratory\|unexpected, intent?: str, explain}` |
| GET | `/v1/profiles/{pid}/tastetwin` | profile, evolution, fatigue risk and its drivers |
| GET | `/v1/profiles/{pid}/hidden-interests` | genres, evidence numbers and example titles |
| GET | `/v1/profiles/{pid}/journey?sessions=` | multi-session discovery plan |
| GET/POST | `/v1/profiles/{pid}/events` | POST `{item_id, kind: watched\|rated\|like\|dislike\|not_interested, rating?}`. Applied to TasteTwin immediately. |

Example recommendation response (trimmed):

```json
{
  "exploration_level": "balanced",
  "intent": {"explicit": {"boost_genres": ["Mystery","Thriller"], "avoid_genres": ["Crime","Film-Noir","Horror"]},
             "unsupported": [{"constraint": "format", "phrase": "series", "reason": "the catalogue contains films only"}]},
  "inferred": {"source": "TasteTwin (inferred from viewing history, not stated by the viewer)", "top_genres": ["Comedy","Romance","Drama"]},
  "items": [{"item_id": 812, "title": "The Distant Island", "genres": ["Mystery"], "score": 1.93,
             "explanation": {"reasons": [{"type": "because_you_watched", "text": "Because you watched …", "evidence": [{"item_id": 77, "similarity": 0.25}]}],
                             "retrieved_by": ["ease","itemknn"], "ranker_drivers": [{"feature": "n_channels", "contribution": 0.41}]}}],
  "model_version": "synthetic-20261102182943-b2e056e0", "latency_ms": 11.2
}
```

## B2B (X-API-Key)

| Method | Path | Notes |
|---|---|---|
| POST | `/v1/b2b/recommend` | `{user_id?` (known to the model), `external_user?`, `history?: [{item_id, rating, ts?}]`, `k`, `level`, `intent?}`. Stateless: an explicit history is used for this request only. Metered in `usage_daily`. |

## Enterprise (JWT, role analyst or admin)

| Method | Path | Notes |
|---|---|---|
| GET | `/v1/enterprise/overview` | KPIs, segments, risk and fatigue summaries, journey policy results |
| GET | `/v1/enterprise/audience` | segments plus risk and fatigue by segment |
| GET | `/v1/enterprise/tastetwin` | population distributions plus forecast evaluation |
| GET | `/v1/enterprise/reports/{name}` | `benchmark`, `tastetwin`, `fatigue`, `hidden_interests`, `engagement_risk`, `journey`, `performance`, `tuning` |
| GET | `/v1/enterprise/models` | manifest, tuning traces, ranker feature importance, versions |
| POST | `/v1/enterprise/journey/simulate` | world assumptions plus a policy list → simulated comparison with paired CIs |
| GET | `/v1/enterprise/risk` | out-of-time evaluation plus the highest-risk pseudonymous viewers and their drivers |
| GET | `/v1/enterprise/fatigue` | |
| GET/POST | `/v1/enterprise/scenario[/defaults]` | transparent business scenario (simulated) |
| GET | `/v1/enterprise/content` | coverage, under-discovered genres, hidden gems |
| GET | `/v1/enterprise/monitoring` | API latency and errors, model age, drift PSI, live-event counts |
| GET | `/v1/enterprise/data` | dataset lineage plus contract checks |
| POST | `/v1/enterprise/data/validate` | multipart CSV (20 MB maximum). Validation only; nothing is stored. |
| GET/POST | `/v1/enterprise/experiments` | define policy experiments; arm shares must sum to 1 |
| GET | `/v1/enterprise/experiments/{key}/assignment?user_id=` | deterministic SHA-256 bucketing |
| POST | `/v1/enterprise/experiments/sample-size` | two-sided z-test sample size |
| POST | `/v1/enterprise/analyst` | `{question}` → grounded answer with cited facts, or a refusal |

## Admin (JWT, role admin)

| Method | Path |
|---|---|
| GET/POST | `/v1/admin/users` |
| GET/POST | `/v1/admin/api-keys` (POST returns the plaintext key once) |
| DELETE | `/v1/admin/api-keys/{kid}` (revokes the key) |
| GET | `/v1/admin/audit?limit=` |
| GET | `/v1/admin/usage` |

## Limits

* Rate limit: `NTG_RATE_LIMIT_PER_MINUTE` (default 240) per principal (API key prefix, token or IP), as an
  in-process sliding window.
* Bounded inputs: intent ≤ 500 characters, search ≤ 200, analyst question ≤ 600, B2B history ≤ 500 events,
  simulation ≤ 600 users and 40 sessions.

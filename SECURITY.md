# Security and privacy

Viewing behaviour is sensitive personal data. This file describes what is **implemented**, and then what
remains before a production launch.

## Implemented

| Control | Implementation |
|---|---|
| Password storage | scrypt (N = 2¹⁴, r = 8, p = 1) with a 16-byte random salt per user; constant-time compare (`ntg/api/security.py`) |
| Sessions | HS256 JWT with `exp`, `iat`, `iss` and the role. The secret comes from `NTG_SECRET_KEY`, and **startup fails in production if it is unset**. In development an ephemeral random key is used. |
| API keys | `ntg_` plus 32 random bytes, shown once. Only the SHA-256 hash and a 10-character prefix are stored. Keys can be revoked, and last use is tracked. |
| Authorisation | Roles viewer / analyst / admin, enforced per route. Every tenant row carries `org_id`, and every query filters on it. A viewer can reach only their own profiles. Unauthorised access returns 404, which does not reveal whether the resource exists. |
| Input validation | Pydantic schemas with bounds (lengths, numeric ranges, enum patterns). CSV uploads are capped at 20 MB and validated without being stored. |
| Rate limiting | Per-principal sliding window, returning 429 with `Retry-After`. It is in-process, so a multi-instance deployment needs Redis (see below). |
| Transport headers | `X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy`, `Cache-Control: no-store` on the API; the same set on the web app. CORS is an explicit allow-list. |
| Secret hygiene | No secrets in source. `.env*` is git-ignored. Logs scrub `password` / `token` / `secret` / `api_key` / `authorization` values. |
| Consent | `consent_personalization` (default on at signup, with an explicit checkbox). When it is off, recommendations use **no** history. `consent_analytics` defaults off. |
| Data export | `GET /v1/me/export` returns every profile and event as JSON. |
| Data deletion | `DELETE /v1/me` cascades to profiles and events in the database and drops in-memory engine state. The audit entry records only the user ID. |
| Audit | Signup, login, consent changes, exports, deletions, key creation and revocation, user creation, experiments, and analyst questions (truncated). |
| AI analyst | No LLM and no external calls. Answers come only from whitelisted tools over the platform's own reports. Prompt-injection, competitor-data and PII requests are refused (with tests). Retrieved text is never executed. |
| Pseudonymisation | Enterprise views show `v-000123`-style references and aggregates, never emails. |
| Artifact integrity | The model bundle's SHA-256 is checked against its manifest before unpickling. |
| Containers | The API runs as non-root UID 10001. Dev-only demo accounts are not seeded when `NTG_ENV=production`. |

## Known gaps (to do before production)

1. **Token storage in the browser:** the demo keeps the JWT in `localStorage`. Production should use an
   httpOnly, Secure, SameSite cookie set by the API, plus CSRF protection.
2. **Distributed rate limiting and quotas** in Redis. Brute-force lockout and a login rate limit are not
   yet separate from the global limit.
3. **Refresh tokens and revocation** (a deny-list), plus password reset and email verification.
4. **SSO / SAML / OIDC** for enterprise tenants.
5. **Pickle** is safe here only because the checksum and producer are trusted. Production should move to a
   signed artifact registry, or to non-executable formats (npz / LightGBM text).
6. **Encryption at rest** (database and artifacts) and a secrets manager (KMS / Vault).
7. **Retention policies**, a DPIA, GDPR / CCPA records of processing, and a DPA for B2B customers.
   Children's-data rules apply if any viewers are minors. None of this compliance work is done.
8. **Dependency and container scanning** in CI (pip-audit, npm audit, Trivy) and SAST.
9. **Differential privacy or k-anonymity thresholds** on aggregate analytics. Segments are currently large
   (100+ viewers) but have no formal guarantee.

## Reporting a vulnerability

Please do not open a public issue. Email the maintainer (see the repository owner profile) with the
details and steps to reproduce.

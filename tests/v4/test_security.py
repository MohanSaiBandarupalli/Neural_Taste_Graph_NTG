"""Browser-session security: httpOnly cookies, CSRF, refresh rotation + reuse detection,
global logout, login brute-force limit, distributed (Redis) rate limiter."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ntg.api.security import RateLimiter, RedisRateLimiter
from ntg.core.config import Settings

LOGIN = {"email": "viewer@demo.ntg", "password": "demo-password-123"}


@pytest.fixture()
def app_factory(engine, tmp_path):
    from ntg.api.app import create_app

    def make(**kw):
        s = Settings(database_url=f"sqlite:///{tmp_path}/sec_{len(kw)}_{id(kw)}.db", secret_key="s" * 40,
                     rate_limit_per_minute=10_000, **kw)
        return create_app(s, engine)

    return make


def test_login_sets_httponly_cookies_and_cookie_auth_works(app_factory):
    with TestClient(app_factory()) as c:
        r = c.post("/v1/auth/login", json=LOGIN)
        assert r.status_code == 200
        set_cookie = r.headers.get_list("set-cookie")
        access = next(h for h in set_cookie if h.startswith("ntg_access="))
        refresh = next(h for h in set_cookie if h.startswith("ntg_refresh="))
        csrf = next(h for h in set_cookie if h.startswith("ntg_csrf="))
        assert "HttpOnly" in access and "HttpOnly" in refresh and "HttpOnly" not in csrf
        assert "samesite=lax" in access.lower()
        assert c.get("/v1/me").json()["email"] == LOGIN["email"]  # cookie only, no Authorization header


def test_csrf_required_for_cookie_authenticated_writes(app_factory):
    with TestClient(app_factory()) as c:
        csrf = c.post("/v1/auth/login", json=LOGIN).json()["csrf_token"]
        assert c.patch("/v1/me/consent", json={"consent_analytics": True}).status_code == 403
        bad = c.patch("/v1/me/consent", json={"consent_analytics": True}, headers={"X-CSRF-Token": "forged"})
        assert bad.status_code == 403
        ok = c.patch("/v1/me/consent", json={"consent_analytics": True}, headers={"X-CSRF-Token": csrf})
        assert ok.status_code == 200


def test_bearer_clients_do_not_need_csrf(app_factory):
    with TestClient(app_factory()) as c:
        tok = c.post("/v1/auth/login", json=LOGIN).json()["token"]
        c.cookies.clear()
        r = c.patch("/v1/me/consent", json={"consent_analytics": True}, headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200


def test_refresh_rotates_and_reuse_revokes_family(app_factory):
    with TestClient(app_factory()) as c:
        first = c.post("/v1/auth/login", json=LOGIN).json()["refresh_token"]
        c.cookies.clear()
        second = c.post("/v1/auth/refresh", json={"refresh_token": first})
        assert second.status_code == 200
        new_rt = second.json()["refresh_token"]
        assert new_rt != first
        c.cookies.clear()
        # replaying the rotated token = theft signal: rejected, and the whole family is revoked
        assert c.post("/v1/auth/refresh", json={"refresh_token": first}).status_code == 401
        c.cookies.clear()
        assert c.post("/v1/auth/refresh", json={"refresh_token": new_rt}).status_code == 401


def test_logout_all_invalidates_existing_access_tokens(app_factory):
    with TestClient(app_factory()) as c:
        tok = c.post("/v1/auth/login", json=LOGIN).json()["token"]
        h = {"Authorization": f"Bearer {tok}"}
        assert c.get("/v1/me", headers=h).status_code == 200
        assert c.post("/v1/auth/logout-all", headers=h).status_code == 204
        c.cookies.clear()
        assert c.get("/v1/me", headers=h).status_code == 401


def test_logout_clears_cookies_and_revokes_refresh(app_factory):
    with TestClient(app_factory()) as c:
        rt = c.post("/v1/auth/login", json=LOGIN).json()["refresh_token"]
        assert c.post("/v1/auth/logout").status_code == 204
        assert c.get("/v1/me").status_code == 401
        assert c.post("/v1/auth/refresh", json={"refresh_token": rt}).status_code == 401


def test_login_bruteforce_limit(app_factory):
    with TestClient(app_factory(login_attempts_per_15min=3)) as c:
        codes = [c.post("/v1/auth/login", json={**LOGIN, "password": "wrong-password"}).status_code for _ in range(4)]
        assert codes == [401, 401, 401, 429]


class FakeRedis:
    def __init__(self, fail=False):
        self.store, self.fail = {}, fail

    def incr(self, k):
        if self.fail:
            raise ConnectionError("redis down")
        self.store[k] = self.store.get(k, 0) + 1
        return self.store[k]

    def expire(self, k, s):
        pass


def test_redis_rate_limiter_shared_and_fails_open():
    shared = FakeRedis()
    a, b = RedisRateLimiter(shared, 3), RedisRateLimiter(shared, 3)  # two API instances
    results = [a.allow("k")[0], b.allow("k")[0], a.allow("k")[0], b.allow("k")[0]]
    assert results == [True, True, True, False]  # the limit is global, not per instance
    down = RedisRateLimiter(FakeRedis(fail=True), 2)
    assert [down.allow("k")[0] for _ in range(3)] == [True, True, False] and down.degraded
    assert isinstance(down.fallback, RateLimiter)

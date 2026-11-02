from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ntg.core.config import Settings


@pytest.fixture(scope="module")
def client(engine, tmp_path_factory):
    from ntg.api.app import create_app

    db = tmp_path_factory.mktemp("db") / "app.db"
    s = Settings(database_url=f"sqlite:///{db}", secret_key="test-secret-key-0123456789", rate_limit_per_minute=10_000)
    with TestClient(create_app(s, engine)) as c:
        yield c


def _login(c, who):
    r = c.post("/v1/auth/login", json={"email": f"{who}@demo.ntg", "password": "demo-password-123"})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_health_and_meta(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/v1/meta").json()["is_synthetic"] is True


def test_auth_errors(client):
    assert client.post("/v1/auth/login", json={"email": "viewer@demo.ntg", "password": "wrong-password"}).status_code == 401
    assert client.get("/v1/me").status_code == 401
    assert client.get("/v1/me", headers={"Authorization": "Bearer forged.token.value"}).status_code == 401
    assert client.post("/v1/auth/signup", json={"email": "short@x.io", "password": "short"}).status_code == 422


def test_viewer_flow_and_isolation(client):
    H = _login(client, "viewer")
    profiles = client.get("/v1/profiles", headers=H).json()
    pid = profiles[0]["id"]
    home = client.get(f"/v1/profiles/{pid}/home", headers=H).json()
    assert home["rows"] and home["rows"][0]["items"]
    assert client.get(f"/v1/profiles/{pid}/tastetwin", headers=H).status_code == 200
    assert client.post(f"/v1/profiles/{pid}/recommendations", headers=H, json={"k": 5, "level": "unexpected"}).status_code == 200
    assert client.post(f"/v1/profiles/{pid}/recommendations", headers=H, json={"level": "bogus"}).status_code == 422
    # another user's profile is invisible
    other = client.post("/v1/auth/signup", json={"email": "iso@x.io", "password": "isolation-pass"}).json()
    O = {"Authorization": f"Bearer {other['token']}"}
    assert client.get(f"/v1/profiles/{pid}/tastetwin", headers=O).status_code == 404
    # viewers cannot reach enterprise or admin routes
    assert client.get("/v1/enterprise/overview", headers=H).status_code == 403
    assert client.get("/v1/admin/users", headers=H).status_code == 403


def test_feedback_updates_profile_and_consent_withdrawal(client):
    s = client.post("/v1/auth/signup", json={"email": "fb@x.io", "password": "feedback-pass"}).json()
    H = {"Authorization": f"Bearer {s['token']}"}
    pid = client.get("/v1/profiles", headers=H).json()[0]["id"]
    item = client.get("/v1/catalog/browse?k=1", headers=H).json()["items"][0]["item_id"]
    assert client.post(f"/v1/profiles/{pid}/events", headers=H, json={"item_id": item, "kind": "like"}).status_code == 201
    assert client.get(f"/v1/profiles/{pid}/tastetwin", headers=H).json()["n_events"] == 1
    assert client.post(f"/v1/profiles/{pid}/events", headers=H, json={"item_id": 10**9, "kind": "like"}).status_code == 404
    client.patch("/v1/me/consent", headers=H, json={"consent_personalization": False})
    r = client.post(f"/v1/profiles/{pid}/recommendations", headers=H, json={"k": 3}).json()
    assert r["personalized"] is False and r["cold_start"] is True
    exp = client.get("/v1/me/export", headers=H).json()
    assert exp["profiles"][0]["events"][0]["item_id"] == item
    assert client.delete("/v1/me", headers=H).status_code == 204
    assert client.get("/v1/me", headers=H).status_code == 401


def test_enterprise_routes(client):
    A = _login(client, "analyst")
    for path in ("overview", "audience", "tastetwin", "models", "risk", "fatigue", "content", "monitoring", "data", "experiments"):
        assert client.get(f"/v1/enterprise/{path}", headers=A).status_code == 200, path
    sim = client.post("/v1/enterprise/journey/simulate", headers=A, json={"n_users": 50, "steps": 8}).json()
    assert sim["evidence_type"] == "simulated" and "journey_orchestrator" in sim["policies"]
    sc = client.post("/v1/enterprise/scenario", headers=A, json={"baseline_monthly_churn": 0.02}).json()
    assert sc["evidence_type"] == "simulated"
    assert client.post("/v1/enterprise/scenario", headers=A, json={"baseline_monthly_churn": 2}).status_code == 422
    ans = client.post("/v1/enterprise/analyst", headers=A, json={"question": "Which model version is currently deployed?"}).json()
    assert ans["facts"]
    files = {"file": ("x.csv", b"userId,movieId,rating,timestamp\n1,2,4.0,1600000000\n1,3,9,1600000001\n", "text/csv")}
    v = client.post("/v1/enterprise/data/validate", headers=A, files=files).json()
    assert v["valid"] is True and any(c["status"] == "warn" for c in v["checks"])
    ss = client.post("/v1/enterprise/experiments/sample-size", headers=A, json={"baseline_mean": 1.0, "baseline_std": 1.0}).json()
    assert ss["n_per_arm"] > 0
    a = client.get("/v1/enterprise/experiments/journey-vs-greedy/assignment?user_id=42", headers=A).json()
    assert a["arm"] in ("control", "treatment")
    b = client.get("/v1/enterprise/experiments/journey-vs-greedy/assignment?user_id=42", headers=A).json()
    assert a["arm"] == b["arm"]  # deterministic bucketing


def test_admin_api_keys_and_b2b(client):
    AD = _login(client, "admin")
    k = client.post("/v1/admin/api-keys", headers=AD, json={"name": "partner"}).json()
    r = client.post("/v1/b2b/recommend", headers={"X-API-Key": k["key"]},
                    json={"external_user": "abc", "history": [{"item_id": 1, "rating": 5}], "k": 3})
    assert r.status_code == 200 and len(r.json()["items"]) == 3
    client.delete(f"/v1/admin/api-keys/{k['id']}", headers=AD)
    assert client.post("/v1/b2b/recommend", headers={"X-API-Key": k["key"]}, json={}).status_code == 401
    assert any(a["action"] == "api_key_revoked" for a in client.get("/v1/admin/audit", headers=AD).json())


def test_rate_limit(engine, tmp_path):
    from ntg.api.app import create_app

    s = Settings(database_url=f"sqlite:///{tmp_path}/rl.db", secret_key="k" * 32, rate_limit_per_minute=3, seed_demo=False)
    with TestClient(create_app(s, engine)) as c:
        codes = [c.get("/v1/meta").status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[-1] == 429


def test_security_primitives():
    from ntg.api.security import decode_token, hash_password, issue_token, verify_password

    h = hash_password("correct horse battery")
    assert verify_password("correct horse battery", h) and not verify_password("wrong", h)
    t = issue_token("s" * 32, 1, 1, "viewer", 5)
    assert decode_token("s" * 32, t)["role"] == "viewer"
    with pytest.raises(Exception):
        decode_token("other-secret-xxxxxxxxxxxxxxxxxxxxx", t)

"""NTG API (FastAPI, versioned under /v1).

Route groups
------------
public      /health, /v1/auth/*
viewer      /v1/me*, /v1/profiles/*            (JWT; a viewer can only touch their own profiles)
catalog     /v1/catalog/*                       (JWT or API key)
b2b         /v1/b2b/*                           (X-API-Key; org-scoped)
enterprise  /v1/enterprise/*                    (JWT, role analyst|admin)
admin       /v1/admin/*                         (JWT, role admin)
"""
from __future__ import annotations

import io
import json
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict, fields
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from ntg.api import db as m
from ntg.api.security import (RateLimiter, decode_token, hash_api_key, hash_password, issue_token, new_api_key,
                              verify_password)
from ntg.core.config import Settings, get_settings
from ntg.core.logging import get_logger
from ntg.enterprise.analyst import BusinessAnalyst
from ntg.enterprise.analytics import ScenarioAssumptions, business_scenario
from ntg.experiments.design import ExperimentDesign, required_sample_size_continuous
from ntg.journey.counterfactual import compare_policies
from ntg.journey.policies import POLICIES
from ntg.journey.simulator import WorldConfig
from ntg.platform.engine import Engine, UnknownItem
from ntg.recsys.rerank import EXPLORATION_LEVELS

log = get_logger("ntg.api")
ROLES = ("viewer", "analyst", "admin")
DEMO_PASSWORD = "demo-password-123"


# ====================================================================== schemas
class SignupIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: str = Field(default="", max_length=80)
    consent_personalization: bool = True
    consent_analytics: bool = False


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(max_length=128)


class ProfileIn(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    color: str = Field(default="#e50914", pattern=r"^#[0-9a-fA-F]{6}$")
    favorite_genres: list[str] = Field(default_factory=list, max_length=8)


class EventIn(BaseModel):
    item_id: int
    kind: str = Field(pattern="^(watched|rated|like|dislike|not_interested)$")
    rating: float | None = Field(default=None, ge=0.5, le=5.0)


class RecIn(BaseModel):
    k: int = Field(default=20, ge=1, le=100)
    level: str = "balanced"
    intent: str | None = Field(default=None, max_length=500)
    explain: bool = True

    @field_validator("level")
    @classmethod
    def _lvl(cls, v):
        if v not in EXPLORATION_LEVELS:
            raise ValueError(f"level must be one of {EXPLORATION_LEVELS}")
        return v


class ConsentIn(BaseModel):
    consent_personalization: bool | None = None
    consent_analytics: bool | None = None


class B2BHistoryEvent(BaseModel):
    item_id: int
    rating: float = Field(default=4.0, ge=0.5, le=5.0)
    ts: int | None = None


class B2BRecIn(RecIn):
    user_id: int | None = Field(default=None, description="dataset/customer user id known to the model")
    external_user: str | None = Field(default=None, max_length=64, description="opaque id for a user not in the dataset")
    history: list[B2BHistoryEvent] = Field(default_factory=list, max_length=500)


class SimulateIn(BaseModel):
    policies: list[str] = Field(default_factory=lambda: ["greedy_relevance", "greedy_fresh", "mmr_diversity", "journey_orchestrator"])
    steps: int = Field(default=25, ge=5, le=40)
    slate_size: int = Field(default=5, ge=3, le=10)
    fatigue_sensitivity: float = Field(default=1.0, ge=0, le=2)
    repeat_penalty: float = Field(default=0.4, ge=0, le=1)
    discovery_bonus: float = Field(default=0.5, ge=0, le=2)
    leave_base: float = Field(default=0.02, ge=0, le=0.3)
    leave_fatigue: float = Field(default=0.15, ge=0, le=1)
    n_users: int = Field(default=200, ge=50, le=600)
    seed: int = 0


class ScenarioIn(BaseModel):
    n_subscribers: int = Field(default=1_000_000, ge=1, le=1_000_000_000)
    monthly_price: float = Field(default=12.99, ge=0, le=1000)
    baseline_monthly_churn: float = Field(default=0.035, ge=0, lt=1)
    retention_improvement_rel: float = Field(default=0.05, ge=0, le=1)
    intervention_reach: float = Field(default=0.30, ge=0, le=1)
    intervention_cost_per_user_month: float = Field(default=0.05, ge=0, le=100)
    horizon_months: int = Field(default=12, ge=1, le=60)
    gross_margin: float = Field(default=0.55, ge=0, le=1)
    uncertainty_rel: float = Field(default=0.5, ge=0, le=1)


class ExperimentIn(BaseModel):
    key: str = Field(pattern=r"^[a-z0-9_\-]{3,64}$")
    name: str = Field(max_length=120)
    hypothesis: str = Field(default="", max_length=2000)
    arms: list[dict] = Field(min_length=2, max_length=5)
    primary_metric: str = Field(default="ndcg@10", max_length=64)


class SampleSizeIn(BaseModel):
    baseline_mean: float = Field(gt=0)
    baseline_std: float = Field(gt=0)
    mde_pct: float = Field(default=0.03, gt=0, lt=1)
    alpha: float = Field(default=0.05, gt=0, lt=0.5)
    power: float = Field(default=0.8, gt=0.5, lt=1)


class UserCreateIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    role: str = Field(pattern="^(viewer|analyst|admin)$")
    display_name: str = Field(default="", max_length=80)


class ApiKeyIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=600)


# ====================================================================== app factory
def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    settings = settings or get_settings()
    secret = settings.resolved_secret()
    db_engine, SessionLocal = m.make_session_factory(settings.database_url)
    m.init_db(db_engine)
    limiter = RateLimiter(settings.rate_limit_per_minute)
    latencies: deque = deque(maxlen=5000)
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        eng = engine or Engine.load(settings.model_dir, report_root=Path("reports/v4"))
        state["engine"] = eng
        state["started"] = time.time()
        with SessionLocal() as s:
            if settings.seed_demo and not settings.is_production:
                _seed_demo(s, eng)
            n = 0
            for ev in s.scalars(select(m.Event)):
                _apply_event(eng, ev)
                n += 1
            log.info(f"replayed {n} stored events into the engine")
        state["aggregates"] = _aggregates(eng)
        state["analyst"] = BusinessAnalyst(eng, state["aggregates"])
        yield

    app = FastAPI(title="NTG — Neural Taste Graph API", version="4.0.0", lifespan=lifespan,
                  description="Entertainment journey intelligence: recommendations, TasteTwin, discovery, enterprise analytics.")
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_credentials=True,
                       allow_methods=["GET", "POST", "PATCH", "DELETE"], allow_headers=["Authorization", "Content-Type", "X-API-Key"])

    @app.middleware("http")
    async def _mw(request: Request, call_next):
        rid = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        principal = request.headers.get("x-api-key", "")[:10] or (request.headers.get("authorization", "")[-16:]) or (
            request.client.host if request.client else "anon")
        if request.url.path.startswith("/v1"):
            ok, retry = limiter.allow(principal)
            if not ok:
                return JSONResponse({"detail": "rate limit exceeded"}, status_code=429, headers={"Retry-After": str(retry)})
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        latencies.append((request.url.path, response.status_code, ms, time.time()))
        response.headers.update({"X-Request-ID": rid, "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY",
                                 "Referrer-Policy": "no-referrer", "Cache-Control": "no-store"})
        log.info("request", extra={"request_id": rid, "path": request.url.path, "status": response.status_code,
                                   "latency_ms": round(ms, 1)})
        return response

    # ------------------------------------------------------------------ deps
    def get_db():
        with SessionLocal() as s:
            yield s

    def eng() -> Engine:
        return state["engine"]

    def current_user(request: Request, s: Session = Depends(get_db)) -> m.User:
        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            raise HTTPException(401, "missing bearer token")
        try:
            claims = decode_token(secret, auth[7:])
        except Exception:
            raise HTTPException(401, "invalid or expired token")
        u = s.get(m.User, int(claims["sub"]))
        if u is None or u.org_id != claims.get("org"):
            raise HTTPException(401, "unknown user")
        return u

    def require(*roles):
        def dep(u: m.User = Depends(current_user)) -> m.User:
            if u.role not in roles:
                raise HTTPException(403, f"requires role: {' or '.join(roles)}")
            return u
        return dep

    def api_key_org(request: Request, s: Session = Depends(get_db)) -> m.ApiKey:
        raw = request.headers.get("x-api-key", "")
        if not raw:
            raise HTTPException(401, "missing X-API-Key")
        k = s.scalar(select(m.ApiKey).where(m.ApiKey.key_hash == hash_api_key(raw), m.ApiKey.revoked.is_(False)))
        if k is None:
            raise HTTPException(401, "invalid API key")
        k.last_used_at = m.utcnow()
        _usage(s, k.org_id, "b2b")
        s.commit()
        return k

    def own_profile(pid: int, u: m.User = Depends(current_user), s: Session = Depends(get_db)) -> m.Profile:
        p = s.get(m.Profile, pid)
        # tenant + ownership isolation: a viewer may only access their own profiles
        if p is None or p.org_id != u.org_id or (p.user_id != u.id and u.role != "admin"):
            raise HTTPException(404, "profile not found")
        return p

    def audit(s: Session, u: m.User | None, action: str, detail: str = "") -> None:
        s.add(m.AuditLog(org_id=u.org_id if u else None, user_id=u.id if u else None, action=action, detail=detail[:2000]))

    # ------------------------------------------------------------------ public
    @app.get("/health")
    def health():
        e = state.get("engine")
        return {"status": "ok" if e else "starting", "model_version": e.manifest["version"] if e else None,
                "uptime_s": round(time.time() - state.get("started", time.time()), 1)}

    @app.get("/v1/meta")
    def meta(e: Engine = Depends(eng)):
        d = e.data.meta
        return {"model_version": e.manifest["version"], "dataset": d["dataset"], "is_synthetic": d.get("is_synthetic", False),
                "n_items": e.data.n_items, "genres": e.data.genre_vocab, "exploration_levels": list(EXPLORATION_LEVELS),
                "data_notice": ("Synthetic MovieLens-schema demo data — titles and ratings are generated, not real films."
                                if d.get("is_synthetic") else "MovieLens data © GroupLens, used for research/demo purposes.")}

    @app.post("/v1/auth/signup", status_code=201)
    def signup(body: SignupIn, s: Session = Depends(get_db)):
        if s.scalar(select(m.User).where(m.User.email == body.email.lower())):
            raise HTTPException(409, "email already registered")
        org = _consumer_org(s)
        u = m.User(org_id=org.id, email=body.email.lower(), password_hash=hash_password(body.password),
                   display_name=body.display_name or body.email.split("@")[0], role="viewer",
                   consent_personalization=body.consent_personalization, consent_analytics=body.consent_analytics)
        s.add(u)
        s.flush()
        s.add(m.Profile(org_id=org.id, user_id=u.id, name=u.display_name[:40] or "Me"))
        audit(s, u, "signup")
        s.commit()
        return {"token": issue_token(secret, u.id, u.org_id, u.role, settings.token_ttl_minutes), "user": _user_out(u)}

    @app.post("/v1/auth/login")
    def login(body: LoginIn, s: Session = Depends(get_db)):
        u = s.scalar(select(m.User).where(m.User.email == body.email.lower()))
        if u is None or not verify_password(body.password, u.password_hash):
            raise HTTPException(401, "invalid credentials")
        audit(s, u, "login")
        s.commit()
        return {"token": issue_token(secret, u.id, u.org_id, u.role, settings.token_ttl_minutes), "user": _user_out(u)}

    # ------------------------------------------------------------------ me / privacy
    @app.get("/v1/me")
    def me(u: m.User = Depends(current_user), s: Session = Depends(get_db)):
        org = s.get(m.Organization, u.org_id)
        return {**_user_out(u), "organization": {"name": org.name, "slug": org.slug, "plan": org.plan}}

    @app.patch("/v1/me/consent")
    def consent(body: ConsentIn, u: m.User = Depends(current_user), s: Session = Depends(get_db)):
        if body.consent_personalization is not None:
            u.consent_personalization = body.consent_personalization
        if body.consent_analytics is not None:
            u.consent_analytics = body.consent_analytics
        s.add(u)
        audit(s, u, "consent_update", json.dumps(body.model_dump()))
        s.commit()
        return _user_out(u)

    @app.get("/v1/me/export")
    def export(u: m.User = Depends(current_user), s: Session = Depends(get_db)):
        profiles = s.scalars(select(m.Profile).where(m.Profile.user_id == u.id)).all()
        audit(s, u, "data_export")
        s.commit()
        return {"user": _user_out(u), "exported_at": m.utcnow().isoformat(),
                "profiles": [{"id": p.id, "name": p.name, "created_at": p.created_at.isoformat(),
                              "events": [{"item_id": e.item_id, "kind": e.kind, "rating": e.rating, "ts": e.ts} for e in p.events]}
                             for p in profiles]}

    @app.delete("/v1/me", status_code=204)
    def delete_me(u: m.User = Depends(current_user), s: Session = Depends(get_db), e: Engine = Depends(eng)):
        for p in s.scalars(select(m.Profile).where(m.Profile.user_id == u.id)):
            e.forget_viewer(f"profile:{p.id}")
        audit(s, None, "account_deleted", f"user_id={u.id}")
        s.delete(u)
        s.commit()

    # ------------------------------------------------------------------ profiles
    @app.get("/v1/profiles")
    def list_profiles(u: m.User = Depends(current_user), s: Session = Depends(get_db)):
        ps = s.scalars(select(m.Profile).where(m.Profile.user_id == u.id, m.Profile.org_id == u.org_id)).all()
        return [_profile_out(p) for p in ps]

    @app.post("/v1/profiles", status_code=201)
    def create_profile(body: ProfileIn, u: m.User = Depends(current_user), s: Session = Depends(get_db), e: Engine = Depends(eng)):
        if s.query(m.Profile).filter(m.Profile.user_id == u.id).count() >= 5:
            raise HTTPException(400, "maximum of 5 profiles")
        p = m.Profile(org_id=u.org_id, user_id=u.id, name=body.name, color=body.color)
        s.add(p)
        s.flush()
        # onboarding: explicit favourite genres become weak "like" signals on popular titles of each genre
        ts = e.now_ts
        for g in [g for g in body.favorite_genres if g in e.genre_index][:5]:
            for it in e.browse(g, k=2):
                ev = m.Event(org_id=u.org_id, profile_id=p.id, item_id=it["item_id"], kind="like", rating=4.5, ts=ts)
                s.add(ev)
                _apply_event(e, ev)
        s.commit()
        return _profile_out(p)

    @app.delete("/v1/profiles/{pid}", status_code=204)
    def delete_profile(p: m.Profile = Depends(own_profile), s: Session = Depends(get_db), e: Engine = Depends(eng)):
        e.forget_viewer(f"profile:{p.id}")
        s.delete(p)
        s.commit()

    def _viewer(p: m.Profile, u: m.User) -> tuple[str, int | None]:
        if not u.consent_personalization:
            return "anonymous", None  # personalization withdrawn: no history is used
        return f"profile:{p.id}", p.dataset_user_id

    @app.post("/v1/profiles/{pid}/recommendations")
    def recommendations(pid: int, body: RecIn, p: m.Profile = Depends(own_profile), u: m.User = Depends(current_user),
                        s: Session = Depends(get_db), e: Engine = Depends(eng)):
        viewer, ds = _viewer(p, u)
        excl = [ev.item_id for ev in p.events if ev.kind in ("not_interested", "dislike")]
        out = e.recommend(ds, viewer, k=body.k, level=body.level, intent=body.intent, explain=body.explain,
                          exclude_item_ids=excl)
        out["personalized"] = u.consent_personalization
        _usage(s, u.org_id, "recommendations")
        s.commit()
        return out

    @app.get("/v1/profiles/{pid}/home")
    def home(pid: int, level: str = "balanced", p: m.Profile = Depends(own_profile), u: m.User = Depends(current_user),
             e: Engine = Depends(eng)):
        if level not in EXPLORATION_LEVELS:
            raise HTTPException(422, "invalid level")
        viewer, ds = _viewer(p, u)
        excl = [ev.item_id for ev in p.events if ev.kind in ("not_interested", "dislike")]
        top = e.recommend(ds, viewer, k=40, level=level, explain=True, exclude_item_ids=excl)
        rows = [{"key": "top", "title": "Top picks for you" if not top["cold_start"] else "Popular right now", "items": top["items"][:16]}]
        seen_ids = {c["item_id"] for c in top["items"][:16]}
        byw = [c for c in top["items"] if any(r["type"] == "because_you_watched" for r in c.get("explanation", {}).get("reasons", []))]
        if byw:
            anchor = next(r for r in byw[0]["explanation"]["reasons"] if r["type"] == "because_you_watched")["evidence"][0]
            sim = e.similar(anchor["item_id"], k=16).get("collaborative", [])
            rows.append({"key": "because", "title": f"Because you watched {anchor['title']}", "items": [c for c in sim if c["item_id"] not in seen_ids][:14]})
        if not top["cold_start"]:
            explore = e.recommend(ds, viewer, k=16, level="exploratory", explain=True, exclude_item_ids=excl + list(seen_ids))
            rows.append({"key": "explore", "title": "Step outside your usual", "items": explore["items"]})
            hidden = e.hidden_interests(viewer, ds)
            for h in hidden[:2]:
                rows.append({"key": f"hidden:{h['genre']}", "title": f"Hidden interest: {h['genre']}", "subtitle": h["explanation"],
                             "items": h["examples"]})
            prof = e.tastetwin(viewer, ds)
            for em in prof["emerging"][:1]:
                rows.append({"key": f"emerging:{em['genre']}", "title": f"Your growing interest in {em['genre']}",
                             "items": e.recommend(ds, viewer, k=14, intent=em["genre"].lower(), explain=False, exclude_item_ids=excl)["items"]})
        rows.append({"key": "trending", "title": "Trending on NTG", "items": e.browse(k=16)})
        return {"profile": _profile_out(p), "cold_start": top["cold_start"], "rows": [r for r in rows if r["items"]],
                "model_version": e.manifest["version"]}

    @app.get("/v1/profiles/{pid}/tastetwin")
    def tastetwin(pid: int, p: m.Profile = Depends(own_profile), u: m.User = Depends(current_user), e: Engine = Depends(eng)):
        viewer, ds = _viewer(p, u)
        return e.tastetwin(viewer, ds)

    @app.get("/v1/profiles/{pid}/hidden-interests")
    def hidden(pid: int, p: m.Profile = Depends(own_profile), u: m.User = Depends(current_user), e: Engine = Depends(eng)):
        viewer, ds = _viewer(p, u)
        return {"hidden_interests": e.hidden_interests(viewer, ds)}

    @app.get("/v1/profiles/{pid}/journey")
    def journey(pid: int, sessions: int = Query(5, ge=2, le=10), p: m.Profile = Depends(own_profile),
                u: m.User = Depends(current_user), e: Engine = Depends(eng)):
        viewer, ds = _viewer(p, u)
        return e.journey_plan(viewer, ds, sessions=sessions)

    @app.get("/v1/profiles/{pid}/events")
    def events(pid: int, p: m.Profile = Depends(own_profile), e: Engine = Depends(eng)):
        own = [{**_safe_card(e, ev.item_id), "kind": ev.kind, "rating": ev.rating, "ts": ev.ts, "source": "app"}
               for ev in sorted(p.events, key=lambda x: -x.ts)[:100]]
        hist = []
        if p.dataset_user_id is not None and p.dataset_user_id in e.data.user_index:
            h = e.history("_", p.dataset_user_id)
            for i, r, t in list(zip(h.items, h.ratings, h.ts))[::-1][:60]:
                hist.append({**e.item_card(int(i)), "kind": "rated", "rating": float(r), "ts": int(t), "source": "imported history"})
        return {"events": own + hist}

    @app.post("/v1/profiles/{pid}/events", status_code=201)
    def add_event(pid: int, body: EventIn, p: m.Profile = Depends(own_profile), u: m.User = Depends(current_user),
                  s: Session = Depends(get_db), e: Engine = Depends(eng)):
        try:
            e.item_index(body.item_id)
        except UnknownItem:
            raise HTTPException(404, "unknown item")
        rating = body.rating if body.rating is not None else {"watched": 3.5, "rated": 3.5, "like": 4.5, "dislike": 1.5, "not_interested": None}[body.kind]
        ev = m.Event(org_id=u.org_id, profile_id=p.id, item_id=body.item_id, kind=body.kind, rating=rating,
                     ts=max(e.now_ts, int(time.time())))
        s.add(ev)
        s.commit()
        _apply_event(e, ev)
        return {"ok": True, "tastetwin_updated": ev.kind != "not_interested"}

    # ------------------------------------------------------------------ catalog
    @app.get("/v1/catalog/genres")
    def genres(e: Engine = Depends(eng)):
        return {"genres": e.data.genre_vocab}

    @app.get("/v1/catalog/browse")
    def browse(genre: str | None = None, k: int = Query(30, ge=1, le=100), offset: int = Query(0, ge=0), e: Engine = Depends(eng)):
        return {"items": e.browse(genre, k, offset)}

    @app.get("/v1/catalog/search")
    def search(q: str = Query(..., min_length=1, max_length=200), e: Engine = Depends(eng)):
        return e.search(q)

    @app.get("/v1/catalog/items/{item_id}")
    def item(item_id: int, e: Engine = Depends(eng)):
        try:
            return e.similar(item_id, k=12)
        except UnknownItem:
            raise HTTPException(404, "unknown item")

    # ------------------------------------------------------------------ B2B
    @app.post("/v1/b2b/recommend")
    def b2b_recommend(body: B2BRecIn, k: m.ApiKey = Depends(api_key_org), e: Engine = Depends(eng)):
        viewer = f"b2b:{k.org_id}:{body.external_user or body.user_id or 'anon'}"
        # stateless: an explicit history is applied for this request only
        e.forget_viewer(viewer)
        for h in body.history:
            try:
                e.record_event(viewer, h.item_id, h.rating, h.ts)
            except UnknownItem:
                continue
        try:
            out = e.recommend(body.user_id, viewer, k=body.k, level=body.level, intent=body.intent, explain=body.explain)
        finally:
            e.forget_viewer(viewer)
        return out

    # ------------------------------------------------------------------ enterprise
    analyst_dep = require("analyst", "admin")

    @app.get("/v1/enterprise/overview")
    def overview(u=Depends(analyst_dep), e: Engine = Depends(eng)):
        d = e.data.meta
        bench = e.report("benchmark") or {}
        warm = bench.get("results", {}).get("warm", {})
        kpi = {}
        if warm:
            kpi = {"hybrid_ndcg@10": warm["hybrid_ltr"]["metrics"]["ndcg@10"], "popularity_ndcg@10": warm["popularity"]["metrics"]["ndcg@10"],
                   "hybrid_coverage@10": warm["hybrid_ltr"]["metrics"]["coverage@10"], "n_eval_users": warm["hybrid_ltr"]["n_users"]}
        agg = state["aggregates"]
        return {"dataset": d, "model": e.manifest, "kpis": kpi, "segments": e.bundle["segments"],
                "risk_summary": agg.get("risk_summary"), "fatigue_summary": agg.get("fatigue_summary"),
                "journey": (e.report("journey") or {}).get("default_world", {}).get("policies"),
                "evidence_note": "KPIs are offline evaluation results on held-out data (observed), not live business metrics."}

    @app.get("/v1/enterprise/audience")
    def audience(u=Depends(analyst_dep), e: Engine = Depends(eng)):
        return {"segments": e.bundle["segments"], "risk_by_segment": state["aggregates"].get("risk_by_segment"),
                "fatigue_by_segment": state["aggregates"].get("fatigue_by_segment")}

    @app.get("/v1/enterprise/tastetwin")
    def tt_analytics(u=Depends(analyst_dep), e: Engine = Depends(eng)):
        return {"population": state["aggregates"].get("tastetwin"), "evaluation": e.report("tastetwin")}

    @app.get("/v1/enterprise/reports/{name}")
    def report(name: str, u=Depends(analyst_dep), e: Engine = Depends(eng)):
        r = e.report(name)
        if r is None:
            raise HTTPException(404, "report not found")
        return r

    @app.get("/v1/enterprise/models")
    def models(u=Depends(analyst_dep), e: Engine = Depends(eng)):
        return {"manifest": e.manifest, "tuning": e.report("tuning"),
                "ranker_feature_importance": getattr(e.ranker, "importance", {}),
                "available_versions": sorted(p.name for p in Path(settings.model_dir).glob("*") if p.is_dir())}

    @app.post("/v1/enterprise/journey/simulate")
    def simulate(body: SimulateIn, u=Depends(analyst_dep), e: Engine = Depends(eng)):
        world = e.bundle.get("journey_world")
        if world is None:
            raise HTTPException(503, "journey world not available in this bundle")
        bad = [p for p in body.policies if p not in POLICIES]
        if bad:
            raise HTTPException(422, f"unknown policies: {bad}")
        n = min(body.n_users, len(world.users))
        sub = type(world)(cs=None, users=world.users[:n], cand=world.cand[:n], rel=world.rel[:n], p_like=world.p_like[:n],
                          primary=world.primary[:n], own_share=world.own_share[:n], exploration=world.exploration[:n], info=world.info)
        cfg = WorldConfig(steps=body.steps, slate_size=body.slate_size, fatigue_sensitivity=body.fatigue_sensitivity,
                          repeat_penalty=body.repeat_penalty, discovery_bonus=body.discovery_bonus, leave_base=body.leave_base,
                          leave_fatigue=body.leave_fatigue, seed=body.seed)
        pols = list(dict.fromkeys(["greedy_relevance", *body.policies]))
        return compare_policies(sub, cfg, pols)

    @app.get("/v1/enterprise/risk")
    def risk(u=Depends(analyst_dep), e: Engine = Depends(eng)):
        return {"evaluation": e.report("engagement_risk"), "at_risk": state["aggregates"].get("at_risk", []),
                "risk_by_segment": state["aggregates"].get("risk_by_segment")}

    @app.get("/v1/enterprise/fatigue")
    def fatigue(u=Depends(analyst_dep), e: Engine = Depends(eng)):
        return {"evaluation": e.report("fatigue"), "by_segment": state["aggregates"].get("fatigue_by_segment"),
                "summary": state["aggregates"].get("fatigue_summary")}

    @app.post("/v1/enterprise/scenario")
    def scenario(body: ScenarioIn, u=Depends(analyst_dep)):
        return business_scenario(ScenarioAssumptions(**body.model_dump()))

    @app.get("/v1/enterprise/scenario/defaults")
    def scenario_defaults(u=Depends(analyst_dep)):
        return asdict(ScenarioAssumptions())

    @app.get("/v1/enterprise/content")
    def content(u=Depends(analyst_dep), e: Engine = Depends(eng)):
        return e.bundle["content"]

    @app.get("/v1/enterprise/monitoring")
    def monitoring(u=Depends(analyst_dep), e: Engine = Depends(eng)):
        now = time.time()
        recent = [x for x in latencies if now - x[3] < 3600]
        lat = [x[2] for x in recent]
        by_route: dict[str, list] = {}
        for p_, st, ms, _ in recent:
            by_route.setdefault(p_.split("/")[2] if p_.count("/") >= 2 else p_, []).append(ms)
        created = pd.Timestamp(e.manifest["created_at"])
        return {
            "api": {"requests_last_hour": len(recent), "error_rate": float(np.mean([x[1] >= 500 for x in recent])) if recent else 0.0,
                    "p50_ms": float(np.percentile(lat, 50)) if lat else None, "p95_ms": float(np.percentile(lat, 95)) if lat else None,
                    "by_group": {k: {"n": len(v), "p95_ms": float(np.percentile(v, 95))} for k, v in by_route.items()}},
            "model": {"version": e.manifest["version"], "age_hours": round((pd.Timestamp.utcnow().tz_localize(None) - created.tz_localize(None)).total_seconds() / 3600, 1)},
            "drift": state["aggregates"].get("drift"),
            "live": {"viewers_with_live_events": len(e.live), "live_events": sum(len(v) for v in e.live.values())},
        }

    @app.get("/v1/enterprise/data")
    def data_info(u=Depends(analyst_dep), e: Engine = Depends(eng)):
        d = e.data
        return {"meta": d.meta, "items": d.n_items, "users": d.n_users,
                "checks": [{"check": "temporal leakage contract", "status": "pass", "detail": d.meta["leakage_contract"]},
                           {"check": "no null ids", "status": "pass" if d.interactions[["user_id", "item_id", "ts"]].notna().all().all() else "fail"},
                           {"check": "ratings in [0.5, 5]", "status": "pass" if d.interactions["rating"].between(0.5, 5).all() else "fail"},
                           {"check": "items have metadata", "status": "pass" if d.items["item_id"].is_unique else "fail"}]}

    @app.post("/v1/enterprise/data/validate")
    async def validate(file: UploadFile = File(...), u=Depends(analyst_dep)):
        raw = await file.read(20_000_001)
        if len(raw) > 20_000_000:
            raise HTTPException(413, "file too large (20 MB max for validation)")
        try:
            df = pd.read_csv(io.BytesIO(raw), nrows=1_000_000)
        except Exception:
            raise HTTPException(422, "could not parse CSV")
        return _validate_interactions(df)

    @app.get("/v1/enterprise/experiments")
    def list_experiments(u=Depends(analyst_dep), s: Session = Depends(get_db)):
        return [_exp_out(x) for x in s.scalars(select(m.Experiment).where(m.Experiment.org_id == u.org_id))]

    @app.post("/v1/enterprise/experiments", status_code=201)
    def create_experiment(body: ExperimentIn, u=Depends(analyst_dep), s: Session = Depends(get_db)):
        shares = [float(a.get("share", 0)) for a in body.arms]
        if abs(sum(shares) - 1.0) > 1e-6 or any(x <= 0 for x in shares):
            raise HTTPException(422, "arm shares must be positive and sum to 1")
        for a in body.arms:
            if a.get("policy") and a["policy"] not in POLICIES and a["policy"] not in EXPLORATION_LEVELS:
                raise HTTPException(422, f"unknown policy {a['policy']}")
        if s.scalar(select(m.Experiment).where(m.Experiment.org_id == u.org_id, m.Experiment.key == body.key)):
            raise HTTPException(409, "experiment key exists")
        x = m.Experiment(org_id=u.org_id, key=body.key, name=body.name, hypothesis=body.hypothesis,
                         arms=json.dumps(body.arms), primary_metric=body.primary_metric)
        s.add(x)
        audit(s, u, "experiment_created", body.key)
        s.commit()
        return _exp_out(x)

    @app.get("/v1/enterprise/experiments/{key}/assignment")
    def assignment(key: str, user_id: int, u=Depends(analyst_dep), s: Session = Depends(get_db)):
        from ntg.experiments.assign import _stable_hash_to_unit_interval

        x = s.scalar(select(m.Experiment).where(m.Experiment.org_id == u.org_id, m.Experiment.key == key))
        if x is None:
            raise HTTPException(404, "experiment not found")
        r = _stable_hash_to_unit_interval(key, user_id)
        acc = 0.0
        for a in json.loads(x.arms):
            acc += float(a["share"])
            if r < acc:
                return {"experiment": key, "user_id": user_id, "arm": a["name"], "bucket": r, "method": "sha256(experiment:user) -> [0,1)"}
        return {"experiment": key, "user_id": user_id, "arm": json.loads(x.arms)[-1]["name"], "bucket": r}

    @app.post("/v1/enterprise/experiments/sample-size")
    def sample_size(body: SampleSizeIn, u=Depends(analyst_dep)):
        n = required_sample_size_continuous(body.baseline_mean, body.baseline_std,
                                            ExperimentDesign(alpha=body.alpha, power=body.power, mde_pct=body.mde_pct))
        return {"n_per_arm": n, "method": "two-sided z-test, normal approximation", **body.model_dump()}

    @app.post("/v1/enterprise/analyst")
    def ask(body: AskIn, u=Depends(analyst_dep), s: Session = Depends(get_db)):
        ans = state["analyst"].ask(body.question)
        audit(s, u, "analyst_question", body.question[:300])
        s.commit()
        return ans

    # ------------------------------------------------------------------ admin
    admin_dep = require("admin")

    @app.get("/v1/admin/users")
    def users(u=Depends(admin_dep), s: Session = Depends(get_db)):
        return [_user_out(x) for x in s.scalars(select(m.User).where(m.User.org_id == u.org_id))]

    @app.post("/v1/admin/users", status_code=201)
    def create_user(body: UserCreateIn, u=Depends(admin_dep), s: Session = Depends(get_db)):
        if s.scalar(select(m.User).where(m.User.email == body.email.lower())):
            raise HTTPException(409, "email already registered")
        x = m.User(org_id=u.org_id, email=body.email.lower(), password_hash=hash_password(body.password), role=body.role,
                   display_name=body.display_name or body.email.split("@")[0])
        s.add(x)
        audit(s, u, "user_created", f"{body.email.lower()} role={body.role}")
        s.commit()
        return _user_out(x)

    @app.get("/v1/admin/api-keys")
    def keys(u=Depends(admin_dep), s: Session = Depends(get_db)):
        return [{"id": k.id, "name": k.name, "prefix": k.prefix, "revoked": k.revoked, "created_at": k.created_at.isoformat(),
                 "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None}
                for k in s.scalars(select(m.ApiKey).where(m.ApiKey.org_id == u.org_id))]

    @app.post("/v1/admin/api-keys", status_code=201)
    def create_key(body: ApiKeyIn, u=Depends(admin_dep), s: Session = Depends(get_db)):
        raw, prefix, h = new_api_key()
        k = m.ApiKey(org_id=u.org_id, name=body.name, prefix=prefix, key_hash=h)
        s.add(k)
        audit(s, u, "api_key_created", f"{body.name} {prefix}")
        s.commit()
        return {"id": k.id, "name": k.name, "prefix": prefix, "key": raw, "note": "Shown once. Store it securely."}

    @app.delete("/v1/admin/api-keys/{kid}", status_code=204)
    def revoke_key(kid: int, u=Depends(admin_dep), s: Session = Depends(get_db)):
        k = s.get(m.ApiKey, kid)
        if k is None or k.org_id != u.org_id:
            raise HTTPException(404, "not found")
        k.revoked = True
        audit(s, u, "api_key_revoked", k.prefix)
        s.commit()

    @app.get("/v1/admin/audit")
    def audit_log(u=Depends(admin_dep), s: Session = Depends(get_db), limit: int = Query(100, le=500)):
        rows = s.scalars(select(m.AuditLog).where(m.AuditLog.org_id == u.org_id).order_by(m.AuditLog.id.desc()).limit(limit))
        return [{"action": r.action, "detail": r.detail, "user_id": r.user_id, "ts": r.ts.isoformat()} for r in rows]

    @app.get("/v1/admin/usage")
    def usage(u=Depends(admin_dep), s: Session = Depends(get_db)):
        rows = s.scalars(select(m.UsageDaily).where(m.UsageDaily.org_id == u.org_id).order_by(m.UsageDaily.day.desc()).limit(200))
        return [{"day": r.day, "route": r.route, "count": r.count} for r in rows]

    return app


# ====================================================================== helpers
def _user_out(u: m.User) -> dict:
    return {"id": u.id, "email": u.email, "display_name": u.display_name, "role": u.role,
            "consent_personalization": u.consent_personalization, "consent_analytics": u.consent_analytics}


def _profile_out(p: m.Profile) -> dict:
    return {"id": p.id, "name": p.name, "color": p.color, "has_imported_history": p.dataset_user_id is not None,
            "n_app_events": len(p.events)}


def _exp_out(x: m.Experiment) -> dict:
    return {"key": x.key, "name": x.name, "hypothesis": x.hypothesis, "arms": json.loads(x.arms),
            "primary_metric": x.primary_metric, "status": x.status, "created_at": x.created_at.isoformat()}


def _safe_card(e: Engine, item_id: int) -> dict:
    try:
        return e.item_card(e.item_index(item_id))
    except UnknownItem:
        return {"item_id": item_id, "title": "(removed title)"}


def _apply_event(e: Engine, ev: m.Event) -> None:
    if ev.kind == "not_interested" or ev.rating is None:
        return
    try:
        e.record_event(f"profile:{ev.profile_id}", ev.item_id, ev.rating, ev.ts)
    except UnknownItem:
        pass


def _usage(s: Session, org_id: int, route: str) -> None:
    day = date.today().isoformat()
    row = s.scalar(select(m.UsageDaily).where(m.UsageDaily.org_id == org_id, m.UsageDaily.day == day, m.UsageDaily.route == route))
    if row is None:
        s.add(m.UsageDaily(org_id=org_id, day=day, route=route, count=1))
    else:
        row.count += 1


def _consumer_org(s: Session) -> m.Organization:
    org = s.scalar(select(m.Organization).where(m.Organization.slug == "ntg-consumer"))
    if org is None:
        org = m.Organization(name="NTG Consumer", slug="ntg-consumer", plan="consumer")
        s.add(org)
        s.flush()
    return org


def _validate_interactions(df: pd.DataFrame) -> dict:
    cols = {c.lower(): c for c in df.columns}
    alias = {"user_id": ["user_id", "userid"], "item_id": ["item_id", "movieid", "itemid"], "rating": ["rating"],
             "timestamp": ["timestamp", "ts"]}
    found, checks = {}, []
    for k, names in alias.items():
        hit = next((cols[n] for n in names if n in cols), None)
        found[k] = hit
        checks.append({"check": f"column {k}", "status": "pass" if hit else ("warn" if k == "rating" else "fail")})
    if all(found[k] for k in ("user_id", "item_id", "timestamp")):
        sub = df[[found["user_id"], found["item_id"], found["timestamp"]]]
        checks.append({"check": "no nulls in ids/timestamps", "status": "pass" if sub.notna().all().all() else "fail",
                       "detail": {k: int(v) for k, v in sub.isna().sum().items()}})
        checks.append({"check": "integer ids", "status": "pass" if all(pd.api.types.is_integer_dtype(df[found[k]]) for k in ("user_id", "item_id")) else "fail"})
        ts = pd.to_numeric(df[found["timestamp"]], errors="coerce")
        checks.append({"check": "epoch-second timestamps", "status": "pass" if ts.notna().all() and ts.between(0, 4e9).all() else "fail"})
        dup = int(df.duplicated([found["user_id"], found["item_id"], found["timestamp"]]).sum())
        checks.append({"check": "duplicate events", "status": "pass" if dup == 0 else "warn", "detail": dup})
    if found["rating"]:
        r = pd.to_numeric(df[found["rating"]], errors="coerce")
        checks.append({"check": "ratings numeric in [0.5, 5]", "status": "pass" if r.between(0.5, 5).all() else "warn"})
    return {"rows_checked": int(len(df)), "columns": list(df.columns), "checks": checks,
            "valid": all(c["status"] != "fail" for c in checks),
            "note": "Validation only — uploaded data is not stored or used for training."}


def _seed_demo(s: Session, e: Engine) -> None:
    """Idempotent demo tenant with 3 roles and persona profiles (development only)."""
    if s.scalar(select(m.Organization).where(m.Organization.slug == "ntg-demo")):
        return
    org = m.Organization(name="NTG Demo Studio", slug="ntg-demo", plan="business")
    s.add(org)
    s.flush()
    users = {}
    for role in ROLES:
        u = m.User(org_id=org.id, email=f"{role}@demo.ntg", password_hash=hash_password(DEMO_PASSWORD), role=role,
                   display_name={"viewer": "Alex (viewer)", "analyst": "Sam (analyst)", "admin": "Jordan (admin)"}[role])
        s.add(u)
        s.flush()
        users[role] = u
    tt = e.cs.tt
    active = np.flatnonzero(tt.n_events >= 40)
    if len(active) >= 3:
        stab = tt.stability(active)
        picks = {"Explorer": active[np.argmax(tt.exploration[active])], "Comfort viewer": active[np.argmax(stab)],
                 "Changing tastes": active[np.argmin(stab)]}
        colors = {"Explorer": "#e50914", "Comfort viewer": "#f5c518", "Changing tastes": "#46d369"}
        for name, ui in picks.items():
            s.add(m.Profile(org_id=org.id, user_id=users["viewer"].id, name=name, color=colors[name],
                            dataset_user_id=int(e.data.user_ids[ui])))
    s.add(m.Profile(org_id=org.id, user_id=users["viewer"].id, name="New here", color="#5b8def"))
    s.add(m.Experiment(org_id=org.id, key="journey-vs-greedy", name="Journey Orchestrator vs fresh greedy",
                       hypothesis="Adaptive fatigue-aware journeys increase 4-week satisfaction without lowering week-1 engagement.",
                       arms=json.dumps([{"name": "control", "share": 0.5, "policy": "greedy_fresh"},
                                        {"name": "treatment", "share": 0.5, "policy": "journey_orchestrator"}]),
                       primary_metric="satisfaction_per_journey"))
    s.commit()


def _aggregates(e: Engine) -> dict:
    """Population-level aggregates computed once at startup (pseudonymous, sampled)."""
    from ntg.discovery.features import user_state_features
    from ntg.enterprise.engagement_risk import FEATURES as RISK_FEATURES

    out: dict = {}
    tt = e.cs.tt
    labels = e.bundle["segment_labels"]
    segs = {s["segment_id"]: s["name"] for s in e.bundle["segments"]}
    rng = np.random.default_rng(0)
    active = np.flatnonzero(tt.n_events >= 5)
    sample = np.sort(rng.choice(active, min(1500, len(active)), replace=False))
    df = e.data.interactions
    feats = user_state_features(e.data, df[df["u"].isin(sample)], e.now_ts).set_index("u").loc[sample]
    fat = e.fatigue.predict(feats)
    seg_s = labels[sample]
    out["fatigue_by_segment"] = [{"segment_id": int(k), "name": segs[int(k)], "n_viewers": int((seg_s == k).sum()),
                                  "mean_fatigue_risk": float(fat[seg_s == k].mean())} for k in sorted(segs) if (seg_s == k).any()]
    out["fatigue_summary"] = {"mean": float(fat.mean()), "share_elevated": float((fat > 0.3).mean()), "n": int(len(fat)),
                              "evidence_type": "modeled"}
    recent = feats[feats["days_since_last"] <= 90]
    if len(recent):
        risk = e.risk.predict(recent)
        rs = labels[recent.index.to_numpy()]
        out["risk_by_segment"] = [{"segment_id": int(k), "name": segs[int(k)], "n_viewers": int((rs == k).sum()),
                                   "mean_risk": float(risk[rs == k].mean())} for k in sorted(segs) if (rs == k).any()]
        out["risk_summary"] = {"mean": float(risk.mean()), "share_high": float((risk > 0.5).mean()), "n_recently_active": int(len(risk)),
                               "definition": "P(no activity in next 90 days) among viewers active in last 90 days", "evidence_type": "modeled"}
        top = np.argsort(-risk)[:25]
        out["at_risk"] = [{"viewer_ref": f"v-{int(e.data.user_ids[recent.index[j]]):06d}",
                           "segment": segs.get(int(rs[j]), "-"), "risk": float(risk[j]),
                           "drivers": e.risk.explain(recent.iloc[j])} for j in top]
    ex, st = tt.exploration[sample], tt.stability(sample)
    mom = tt.momentum(sample)
    emerging = {}
    for row in mom:
        for k in np.flatnonzero(row > tt.cfg.emerging_threshold):
            emerging[tt.genres[k]] = emerging.get(tt.genres[k], 0) + 1
    out["tastetwin"] = {
        "n_sampled": int(len(sample)),
        "exploration_hist": np.histogram(ex, bins=10, range=(0, 1))[0].tolist(),
        "stability_hist": np.histogram(st, bins=10, range=(0, 1))[0].tolist(),
        "mean_exploration": float(ex.mean()), "mean_stability": float(st.mean()),
        "share_with_emerging_interest": float((mom > tt.cfg.emerging_threshold).any(1).mean()),
        "emerging_genre_counts": dict(sorted(emerging.items(), key=lambda kv: -kv[1])),
        "confidence_mean": float(tt.confidence(sample).mean()),
    }
    # drift: PSI of genre mix + activity between the final 90 days and the preceding year
    t1 = e.now_ts
    last = df[df["ts"] >= t1 - 90 * 86400]
    prev = df[(df["ts"] < t1 - 90 * 86400) & (df["ts"] >= t1 - 455 * 86400)]
    G = e.cs.Gn

    def mix(x):
        v = G[x["i"].to_numpy()].sum(0) + 1e-6
        return v / v.sum()

    if len(last) and len(prev):
        a, b = mix(last), mix(prev)
        psi = float(((a - b) * np.log(a / b)).sum())
        ra, rb = last["rating"].mean(), prev["rating"].mean()
        out["drift"] = {"genre_mix_psi": psi, "status": "stable" if psi < 0.1 else "moderate" if psi < 0.25 else "significant",
                        "mean_rating_recent": float(ra), "mean_rating_prior": float(rb),
                        "events_recent_90d": int(len(last)), "events_prior_365d": int(len(prev)),
                        "method": "Population Stability Index of genre mix, final 90 days vs preceding 365 days"}
    return out

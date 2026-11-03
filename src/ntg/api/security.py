"""Password hashing, tokens, API keys and rate limiting.

* Passwords: scrypt (memory-hard, stdlib) with per-user random salt; constant-time compare.
* Sessions: short-lived HS256 access JWTs (default 15 min) carrying user id, org id, role and a
  session version; rotating opaque refresh tokens (hashed at rest) with reuse detection.
* API keys: random 32-byte tokens shown once; only a SHA-256 hash is stored.
* Rate limiting: Redis fixed window shared across instances when NTG_REDIS_URL is set,
  otherwise a per-process sliding window.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import threading
import time
from collections import deque

import jwt

_SCRYPT = {"n": 2**14, "r": 8, "p": 1, "dklen": 32}


def hash_password(password: str) -> str:
    if len(password) < 8:
        raise ValueError("password must be at least 8 characters")
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(dk).decode()


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, salt_b64, dk_b64 = stored.split("$")
        if algo != "scrypt":
            return False
        dk = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt_b64), **_SCRYPT)
        return hmac.compare_digest(dk, base64.b64decode(dk_b64))
    except Exception:
        return False


def issue_token(secret: str, user_id: int, org_id: int, role: str, ttl_minutes: int, session_version: int = 0) -> str:
    """Short-lived access token. ``sv`` must match the user's current session_version."""
    now = int(time.time())
    return jwt.encode({"sub": str(user_id), "org": org_id, "role": role, "sv": session_version, "iat": now,
                       "exp": now + ttl_minutes * 60, "iss": "ntg", "jti": secrets.token_hex(8)},
                      secret, algorithm="HS256")


def new_refresh_token() -> tuple[str, str]:
    """Returns (plaintext opaque token, sha256 hash). Only the hash is stored."""
    raw = secrets.token_urlsafe(48)
    return raw, hashlib.sha256(raw.encode()).hexdigest()


def hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def new_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def decode_token(secret: str, token: str) -> dict:
    return jwt.decode(token, secret, algorithms=["HS256"], issuer="ntg", options={"require": ["exp", "sub", "iss"]})


def new_api_key() -> tuple[str, str, str]:
    """Returns (plaintext key, display prefix, sha256 hash)."""
    raw = "ntg_" + secrets.token_urlsafe(32)
    return raw, raw[:10], hashlib.sha256(raw.encode()).hexdigest()


def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


class RateLimiter:
    """Sliding-window limiter (in-process). One instance per worker process."""

    def __init__(self, per_minute: int, window_s: int = 60):
        self.per_minute = per_minute
        self.window_s = window_s
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] > self.window_s:
                q.popleft()
            if len(q) >= self.per_minute:
                return False, int(self.window_s - (now - q[0])) + 1
            q.append(now)
            return True, 0


class RedisRateLimiter:
    """Fixed-window limiter shared by every worker and instance (INCR + EXPIRE on one key per window).

    Fails OPEN to the in-process limiter if Redis is unreachable, and logs it, so a Redis outage
    degrades limits per instance instead of taking the API down.
    """

    def __init__(self, client, per_minute: int, window_s: int = 60, prefix: str = "ntg:rl"):
        self.client = client
        self.per_minute = per_minute
        self.window_s = window_s
        self.prefix = prefix
        self.fallback = RateLimiter(per_minute, window_s)
        self.degraded = False

    def allow(self, key: str) -> tuple[bool, int]:
        now = int(time.time())
        bucket = now // self.window_s
        k = f"{self.prefix}:{key}:{bucket}"
        try:
            n = self.client.incr(k)
            if n == 1:
                self.client.expire(k, self.window_s + 1)
            self.degraded = False
        except Exception:
            self.degraded = True
            return self.fallback.allow(key)
        if n > self.per_minute:
            return False, self.window_s - (now % self.window_s)
        return True, 0


def make_rate_limiter(redis_url: str, per_minute: int, window_s: int = 60, prefix: str = "ntg:rl"):
    if redis_url:
        import redis  # optional dependency

        return RedisRateLimiter(redis.Redis.from_url(redis_url, socket_timeout=0.2), per_minute, window_s, prefix)
    return RateLimiter(per_minute, window_s)

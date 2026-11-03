"""Password hashing, tokens, API keys and rate limiting.

* Passwords: scrypt (memory-hard, stdlib) with per-user random salt; constant-time compare.
* Sessions: short-lived HS256 JWTs carrying user id, org id and role.
* API keys: random 32-byte tokens shown once; only a SHA-256 hash is stored.
* Rate limiting: per-principal sliding window (in-process). For multi-instance
  deployments put the counter in Redis (documented in DEPLOYMENT.md).
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


def issue_token(secret: str, user_id: int, org_id: int, role: str, ttl_minutes: int) -> str:
    now = int(time.time())
    return jwt.encode({"sub": str(user_id), "org": org_id, "role": role, "iat": now, "exp": now + ttl_minutes * 60,
                       "iss": "ntg"}, secret, algorithm="HS256")


def decode_token(secret: str, token: str) -> dict:
    return jwt.decode(token, secret, algorithms=["HS256"], issuer="ntg", options={"require": ["exp", "sub", "iss"]})


def new_api_key() -> tuple[str, str, str]:
    """Returns (plaintext key, display prefix, sha256 hash)."""
    raw = "ntg_" + secrets.token_urlsafe(32)
    return raw, raw[:10], hashlib.sha256(raw.encode()).hexdigest()


def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False, int(60 - (now - q[0])) + 1
            q.append(now)
            return True, 0

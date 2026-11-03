"""Runtime configuration for NTG v4.

All settings come from environment variables (12-factor style) with safe
development defaults. Secrets are never hard-coded: in production the process
refuses to start without an explicit ``NTG_SECRET_KEY``.
"""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    env: str = field(default_factory=lambda: _env("NTG_ENV", "development"))
    data_root: Path = field(default_factory=lambda: Path(_env("NTG_DATA_ROOT", "data/v4")))
    artifact_root: Path = field(default_factory=lambda: Path(_env("NTG_ARTIFACT_ROOT", "artifacts")))
    dataset: str = field(default_factory=lambda: _env("NTG_DATASET", "synthetic"))
    database_url: str = field(default_factory=lambda: _env("NTG_DATABASE_URL", "sqlite:///data/v4/ntg_app.db"))
    secret_key: str = field(default_factory=lambda: _env("NTG_SECRET_KEY", ""))
    token_ttl_minutes: int = field(default_factory=lambda: int(_env("NTG_TOKEN_TTL_MINUTES", "15")))
    refresh_ttl_days: int = field(default_factory=lambda: int(_env("NTG_REFRESH_TTL_DAYS", "14")))
    login_attempts_per_15min: int = field(default_factory=lambda: int(_env("NTG_LOGIN_ATTEMPTS_PER_15MIN", "10")))
    rate_limit_per_minute: int = field(default_factory=lambda: int(_env("NTG_RATE_LIMIT_PER_MINUTE", "240")))
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            o.strip() for o in _env("NTG_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if o.strip()
        )
    )
    redis_url: str = field(default_factory=lambda: _env("NTG_REDIS_URL", ""))
    seed_demo: bool = field(default_factory=lambda: _env("NTG_SEED_DEMO", "1") == "1")

    @property
    def is_production(self) -> bool:
        return self.env.lower() in {"prod", "production"}

    def resolved_secret(self) -> str:
        if self.secret_key:
            return self.secret_key
        if self.is_production:
            raise RuntimeError("NTG_SECRET_KEY must be set in production")
        # Development: ephemeral per-process key (tokens invalidate on restart).
        return _DEV_SECRET

    @property
    def dataset_dir(self) -> Path:
        return self.data_root / self.dataset

    @property
    def model_dir(self) -> Path:
        return self.artifact_root / self.dataset


_DEV_SECRET = secrets.token_urlsafe(48)


def get_settings() -> Settings:
    return Settings()

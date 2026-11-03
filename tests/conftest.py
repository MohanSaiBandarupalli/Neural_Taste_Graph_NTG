# tests/conftest.py
from __future__ import annotations

from pathlib import Path
import os

import pandas as pd
import pytest


@pytest.fixture()
def sandbox(tmp_path: Path, monkeypatch) -> Path:
    """Per-test filesystem sandbox.

    The working directory is switched to ``tmp_path`` so that pipelines using
    relative paths (data/..., outputs/..., reports/...) never touch the repo.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture()
def project_root() -> Path:
    """Repo root (where pyproject.toml lives)."""
    # tests/ -> repo root
    return Path(__file__).resolve().parents[1]


@pytest.fixture()
def small_interactions_df() -> pd.DataFrame:
    """Tiny deterministic interactions dataset (3 users)."""
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
            "item_id": [10, 11, 12, 10, 13, 15, 16, 11, 12, 14],
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-04",
                    "2020-01-02",
                    "2020-01-05",
                    "2020-01-06",
                ]
            ),
            "rating": [4.0, 5.0, 3.0, 4.5, 2.0, 3.5, 4.0, 4.0, 5.0, 4.0],
        }
    )


@pytest.fixture()
def write_small_splits(sandbox: Path, small_interactions_df: pd.DataFrame):
    """
    Writes train/val/test parquet files into a directory that is NOT the same as
    the path tests later copy into (data/processed/splits under chdir_sandbox).

    Returns dict with:
      - "dir": directory containing the parquet files
      - "train"/"val"/"test": dataframes
    """
    # IMPORTANT: write somewhere else so shutil.copy() does not copy onto itself
    base = sandbox / "_generated_splits"
    base.mkdir(parents=True, exist_ok=True)

    df = small_interactions_df.sort_values(["user_id", "timestamp", "item_id"]).copy()

    # IMPORTANT: do NOT reset index before using it to drop from df.
    train = df.groupby("user_id").head(2)
    tail = df.drop(index=train.index)

    val = tail.groupby("user_id").head(1)
    test = tail.drop(index=val.index)

    # Now reset for writing
    train_out = train.reset_index(drop=True)
    val_out = val.reset_index(drop=True)
    test_out = test.reset_index(drop=True)

    canonical = sandbox / "data" / "processed" / "splits"
    canonical.mkdir(parents=True, exist_ok=True)
    for d in (base, canonical):
        train_out.to_parquet(d / "train.parquet", index=False)
        val_out.to_parquet(d / "val.parquet", index=False)
        test_out.to_parquet(d / "test.parquet", index=False)

    return {"dir": base, "train": train_out, "val": val_out, "test": test_out}


@pytest.fixture()
def chdir_sandbox(monkeypatch, sandbox: Path):
    """
    Run code as-if repo root is sandbox so all relative paths
    like data/... outputs/... resolve inside sandbox.
    """
    monkeypatch.chdir(sandbox)
    return sandbox


@pytest.fixture()
def has_movielens_data() -> bool:
    """
    Integration smoke tests may check real MovieLens presence.
    Keep this false by default for unit suite.
    """
    return os.getenv("HAS_MOVIELENS", "0") == "1"

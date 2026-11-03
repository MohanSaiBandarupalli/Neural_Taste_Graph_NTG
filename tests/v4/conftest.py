"""Session fixtures: a tiny synthetic dataset + a full v4 bundle built once (CI-safe, no downloads)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from ntg.datasets.synthetic import SyntheticConfig, generate


@pytest.fixture(scope="session")
def v4_workspace(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("ntg_v4")
    cwd = os.getcwd()
    os.chdir(root)
    try:
        generate(root / "data/external/tiny", SyntheticConfig(n_users=900, n_items=450, median_user_events=45, seed=7))
        from ntg.pipelines.v4 import Pipeline

        Pipeline("tiny", quick=True).run(root / "data/external/tiny", {"min_item_support": 3})
    finally:
        os.chdir(cwd)
    return root


@pytest.fixture(scope="session")
def data(v4_workspace):
    from ntg.datasets.matrix import InteractionData

    return InteractionData(v4_workspace / "data/v4/tiny")


@pytest.fixture(scope="session")
def engine(v4_workspace):
    from ntg.platform.engine import Engine

    return Engine.load(v4_workspace / "artifacts/tiny", report_root=v4_workspace / "reports/v4")

"""In-memory interaction data with contiguous indices and sparse matrices."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class Window:
    """A fit/holdout pair. ``fit`` events strictly precede ``holdout`` events."""

    name: str
    fit: pd.DataFrame
    holdout: pd.DataFrame
    cutoff_ts: int


class InteractionData:
    """Loads a prepared dataset directory (see :mod:`ntg.datasets.prepare`)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        inter = pd.read_parquet(self.root / "interactions.parquet")
        self.items = pd.read_parquet(self.root / "items.parquet").reset_index(drop=True)
        self.item_ids = self.items["item_id"].to_numpy()
        self.item_index = {int(i): k for k, i in enumerate(self.item_ids)}
        self.user_ids = np.sort(inter["user_id"].unique())
        self.user_index = {int(u): k for k, u in enumerate(self.user_ids)}
        inter = inter[inter["item_id"].isin(self.item_index)].copy()
        inter["u"] = inter["user_id"].map(self.user_index).astype(np.int32)
        inter["i"] = inter["item_id"].map(self.item_index).astype(np.int32)
        self.interactions = inter.sort_values(["ts", "u", "i"]).reset_index(drop=True)
        self.n_users = len(self.user_ids)
        self.n_items = len(self.item_ids)

    # ------------------------------------------------------------ windows
    def window(self, name: str) -> Window:
        df = self.interactions
        if name == "val":
            return Window("val", df[df.split == "train"], df[df.split == "val"], int(self.meta["t_val"]))
        if name == "test":
            return Window("test", df[df.split != "test"], df[df.split == "test"], int(self.meta["t_test"]))
        if name == "full":
            return Window("full", df, df.iloc[0:0], int(df["ts"].max()) + 1)
        raise ValueError(name)

    # ------------------------------------------------------------ matrices
    def csr(self, df: pd.DataFrame, value: str = "binary") -> sp.csr_matrix:
        if value == "binary":
            vals = np.ones(len(df), dtype=np.float32)
        elif value == "rating":
            vals = df["rating"].to_numpy(np.float32)
        else:
            raise ValueError(value)
        m = sp.csr_matrix((vals, (df["u"].to_numpy(), df["i"].to_numpy())), shape=(self.n_users, self.n_items))
        if value == "binary":
            m.data[:] = 1.0  # collapse duplicates
        m.sum_duplicates()
        return m

    # ------------------------------------------------------------ item metadata
    @cached_property
    def genre_vocab(self) -> list[str]:
        return sorted({g for gs in self.items["genres"] for g in gs})

    @cached_property
    def genre_matrix(self) -> np.ndarray:
        """Multi-hot (n_items, n_genres) float32."""
        idx = {g: k for k, g in enumerate(self.genre_vocab)}
        m = np.zeros((self.n_items, len(idx)), dtype=np.float32)
        for r, gs in enumerate(self.items["genres"]):
            for g in gs:
                m[r, idx[g]] = 1.0
        return m

    @cached_property
    def franchise_codes(self) -> np.ndarray:
        return pd.factorize(self.items["franchise"])[0].astype(np.int32)

    @cached_property
    def item_years(self) -> np.ndarray:
        y = self.items["year"].astype("float").to_numpy()
        return np.where(np.isnan(y), np.nanmedian(y), y)

    def popularity(self, df: pd.DataFrame) -> np.ndarray:
        return np.bincount(df["i"].to_numpy(), minlength=self.n_items).astype(np.float64)

    def item_title(self, i: int) -> str:
        return str(self.items.at[i, "display_title"])

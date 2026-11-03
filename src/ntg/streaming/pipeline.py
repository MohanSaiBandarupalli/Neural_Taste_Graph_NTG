"""A working slice of the real-time path in ARCHITECTURE §6.

    producer --(append)--> EventLog (P partitions, JSONL segments, offsets)
                                |
                        TasteTwinConsumer (one per partition)
                                |  state update + offset commit in ONE transaction
                                v
                         StateStore (SQLite KV: viewer -> TasteTwin sufficient statistics)

Correctness properties (all covered by tests/v4/test_streaming.py):

* **Order independence.** Viewer state is the decayed sum ``S(T) = Σ w_i · 2^(-(T - t_i)/h)``,
  stored together with its reference time ``T`` (= max event time). A newer event first decays the
  state forward and then adds itself; an older event adds itself already decayed. Both give the same
  sum, so out-of-order delivery produces exactly the batch result.
* **Exactly-once effect under at-least-once delivery.** The consumer writes the new states and the
  partition offset in a single SQLite transaction. A crash before commit rolls back both, and the
  replay starts from the last committed offset.
* **Producer retries.** Every event carries an ``event_id``. Applied ids are recorded in the same
  transaction, and duplicates are skipped.

Local stand-ins: the JSONL log plays the role of Kafka/Kinesis and SQLite plays the role of an
online KV store (Redis/DynamoDB/Cassandra). The interfaces are deliberately small so each can be
swapped without touching the update logic.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ntg.tastetwin.model import TasteTwinConfig, preference_weight

DAY = 86400.0


# ========================================================================= event log
class EventLog:
    """Partitioned, append-only event log with integer offsets (one JSONL file per partition)."""

    def __init__(self, root: str | Path, partitions: int = 4):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.partitions = partitions

    def partition_of(self, viewer: str) -> int:
        return int(hashlib.sha1(viewer.encode()).hexdigest(), 16) % self.partitions

    def _path(self, p: int) -> Path:
        return self.root / f"partition-{p:03d}.jsonl"

    def append(self, viewer: str, item: int, rating: float, ts: int, event_id: str | None = None) -> tuple[int, str]:
        ev = {"event_id": event_id or uuid.uuid4().hex, "viewer": viewer, "item": int(item),
              "rating": float(rating), "ts": int(ts)}
        p = self.partition_of(viewer)
        with open(self._path(p), "a", encoding="utf-8") as f:
            f.write(json.dumps(ev) + "\n")
        return p, ev["event_id"]

    def read(self, partition: int, offset: int, max_events: int) -> list[dict]:
        path = self._path(partition)
        if not path.exists():
            return []
        out = []
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f):
                if n < offset:
                    continue
                if len(out) >= max_events:
                    break
                out.append(json.loads(line))
        return out


# ========================================================================= state store
class StateStore:
    """Viewer -> TasteTwin sufficient statistics, plus consumer offsets and applied event ids."""

    def __init__(self, path: str | Path, n_genres: int):
        self.path = str(path)
        self.n_genres = n_genres
        self.con = sqlite3.connect(self.path, isolation_level=None)  # explicit transactions
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=NORMAL")
        self.con.executescript(
            """
            CREATE TABLE IF NOT EXISTS viewer_state (
                viewer TEXT PRIMARY KEY, s_long BLOB, s_short BLOB, ref_ts INTEGER, n_events INTEGER, updated_at REAL);
            CREATE TABLE IF NOT EXISTS consumer_offsets (partition INTEGER PRIMARY KEY, next_offset INTEGER);
            CREATE TABLE IF NOT EXISTS applied_events (event_id TEXT PRIMARY KEY);
            """
        )

    def offset(self, partition: int) -> int:
        row = self.con.execute("SELECT next_offset FROM consumer_offsets WHERE partition=?", (partition,)).fetchone()
        return int(row[0]) if row else 0

    def get(self, viewer: str) -> dict | None:
        row = self.con.execute("SELECT s_long, s_short, ref_ts, n_events FROM viewer_state WHERE viewer=?",
                               (viewer,)).fetchone()
        if row is None:
            return None
        return {"S_long": np.frombuffer(row[0], dtype=np.float64).copy(),
                "S_short": np.frombuffer(row[1], dtype=np.float64).copy(), "ref_ts": int(row[2]), "n_events": int(row[3])}

    def close(self) -> None:
        self.con.close()


# ========================================================================= consumer
@dataclass
class TasteTwinConsumer:
    log: EventLog
    store: StateStore
    genre_weights: np.ndarray            # (n_items, n_genres): TasteTwin's per-item genre mass
    partition: int
    cfg: TasteTwinConfig = TasteTwinConfig()
    batch_size: int = 500

    @staticmethod
    def apply(state: dict | None, g: np.ndarray, rating: float, ts: int, cfg: TasteTwinConfig) -> dict:
        """Order-independent decay-and-add on one viewer's state."""
        w = float(preference_weight(np.array([rating]))[0])
        if state is None:
            return {"S_long": w * g, "S_short": w * g, "ref_ts": int(ts), "n_events": 1}
        ref = state["ref_ts"]
        if ts >= ref:  # newer event: bring the state forward, then add
            dt = (ts - ref) / DAY
            s_long = state["S_long"] * 0.5 ** (dt / cfg.half_life_long_days) + w * g
            s_short = state["S_short"] * 0.5 ** (dt / cfg.half_life_short_days) + w * g
            ref = ts
        else:  # late event: add it already decayed to the state's reference time
            dt = (ref - ts) / DAY
            s_long = state["S_long"] + w * g * 0.5 ** (dt / cfg.half_life_long_days)
            s_short = state["S_short"] + w * g * 0.5 ** (dt / cfg.half_life_short_days)
        return {"S_long": s_long, "S_short": s_short, "ref_ts": int(ref), "n_events": state["n_events"] + 1}

    def poll_once(self, fail_before_commit: bool = False) -> int:
        """Process up to one batch. Returns the number of events consumed (incl. skipped duplicates)."""
        start = self.store.offset(self.partition)
        events = self.log.read(self.partition, start, self.batch_size)
        if not events:
            return 0
        con = self.store.con
        con.execute("BEGIN")
        try:
            cache: dict[str, dict | None] = {}
            for ev in events:
                seen = con.execute("SELECT 1 FROM applied_events WHERE event_id=?", (ev["event_id"],)).fetchone()
                if seen:
                    continue  # producer retry / duplicate delivery
                v = ev["viewer"]
                if v not in cache:
                    cache[v] = self.store.get(v)
                cache[v] = self.apply(cache[v], self.genre_weights[ev["item"]], ev["rating"], ev["ts"], self.cfg)
                con.execute("INSERT INTO applied_events(event_id) VALUES (?)", (ev["event_id"],))
            now = time.time()
            for v, st in cache.items():
                con.execute(
                    "INSERT INTO viewer_state VALUES (?,?,?,?,?,?) ON CONFLICT(viewer) DO UPDATE SET "
                    "s_long=excluded.s_long, s_short=excluded.s_short, ref_ts=excluded.ref_ts, "
                    "n_events=excluded.n_events, updated_at=excluded.updated_at",
                    (v, st["S_long"].astype(np.float64).tobytes(), st["S_short"].astype(np.float64).tobytes(),
                     st["ref_ts"], st["n_events"], now))
            if fail_before_commit:
                raise RuntimeError("simulated crash before commit")
            con.execute("INSERT INTO consumer_offsets VALUES (?, ?) ON CONFLICT(partition) DO UPDATE SET "
                        "next_offset=excluded.next_offset", (self.partition, start + len(events)))
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise
        return len(events)

    def run_until_caught_up(self) -> int:
        total = 0
        while True:
            n = self.poll_once()
            if n == 0:
                return total
            total += n


def state_as_of(state: dict, t: int, cfg: TasteTwinConfig = TasteTwinConfig()) -> dict:
    """Decay a stored state to time ``t`` (>= its reference time) for serving."""
    dt = max(0, t - state["ref_ts"]) / DAY
    return {"S_long": state["S_long"] * 0.5 ** (dt / cfg.half_life_long_days),
            "S_short": state["S_short"] * 0.5 ** (dt / cfg.half_life_short_days), "ref_ts": int(t),
            "n_events": state["n_events"]}

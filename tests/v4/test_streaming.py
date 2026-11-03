"""Streaming TasteTwin: shuffled, duplicated, crash-interrupted delivery must equal the batch state."""
from __future__ import annotations

import numpy as np
import pytest

from ntg.streaming.pipeline import EventLog, StateStore, TasteTwinConsumer, state_as_of
from ntg.tastetwin.model import TasteTwin


@pytest.fixture()
def setup(data, tmp_path):
    tt = TasteTwin(data).fit(data.interactions.iloc[:0], 0)
    df = data.interactions
    users = df.groupby("u").size().sort_values(ascending=False).index[:40]
    ev = df[df.u.isin(users)].sample(frac=1.0, random_state=0)  # shuffled = out of order
    log = EventLog(tmp_path / "log", partitions=3)
    store = StateStore(tmp_path / "state.db", len(tt.genres))
    return tt, ev, log, store


def _batch_state(tt, ev, viewer_u, now):
    g = ev[ev.u == viewer_u]
    return tt.history_state(g["i"].to_numpy(), g["rating"].to_numpy(), g["ts"].to_numpy(), now)


def test_out_of_order_duplicates_and_crash_equal_batch(setup):
    tt, ev, log, store = setup
    ids = []
    for r in ev.itertuples():
        _, eid = log.append(f"u{r.u}", r.i, r.rating, r.ts)
        ids.append((r, eid))
    for r, eid in ids[:200]:  # producer retries: same event_id delivered twice
        log.append(f"u{r.u}", r.i, r.rating, r.ts, event_id=eid)
    consumers = [TasteTwinConsumer(log, store, tt.G, p, batch_size=300) for p in range(log.partitions)]
    with pytest.raises(RuntimeError):
        consumers[0].poll_once(fail_before_commit=True)  # crash mid-batch
    assert store.offset(0) == 0  # nothing committed
    for c in consumers:
        c.run_until_caught_up()
    now = int(ev["ts"].max()) + 1
    for u in ev["u"].unique():
        got = state_as_of(store.get(f"u{u}"), now, tt.cfg)
        want = _batch_state(tt, ev, u, now)
        # identical math; differences are float32 rounding of the genre matrix summed in another order
        np.testing.assert_allclose(got["S_long"], want["S_long"], rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(got["S_short"], want["S_short"], rtol=1e-6, atol=1e-12)
        assert got["n_events"] == int((ev.u == u).sum())  # duplicates not double-counted


def test_replay_after_restart_is_idempotent(setup):
    tt, ev, log, store = setup
    for r in ev.head(500).itertuples():
        log.append(f"u{r.u}", r.i, r.rating, r.ts)
    c = TasteTwinConsumer(log, store, tt.G, 0)
    c.run_until_caught_up()
    snap = {v: store.get(v) for (v,) in store.con.execute("SELECT viewer FROM viewer_state")}
    c2 = TasteTwinConsumer(log, StateStore(store.path, len(tt.genres)), tt.G, 0)  # "restarted" process
    assert c2.run_until_caught_up() == 0  # resumes from the committed offset
    for v, st in snap.items():
        np.testing.assert_array_equal(store.get(v)["S_long"], st["S_long"])

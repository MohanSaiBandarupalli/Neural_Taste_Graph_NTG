"""Streaming TasteTwin throughput: python scripts/streaming_benchmark.py [dataset_dir] > reports/v4/<ds>/streaming.json"""
import json
import sys
import tempfile
import time
from pathlib import Path

from ntg.datasets.matrix import InteractionData
from ntg.streaming.pipeline import EventLog, StateStore, TasteTwinConsumer
from ntg.tastetwin.model import TasteTwin

d = InteractionData(sys.argv[1] if len(sys.argv) > 1 else "data/v4/synthetic")
tt = TasteTwin(d).fit(d.interactions.iloc[:0], 0)
ev = d.interactions
with tempfile.TemporaryDirectory() as tmp:
    log = EventLog(Path(tmp) / "log", partitions=4)
    t0 = time.perf_counter()
    for r in ev.itertuples():
        log.append(f"u{r.u}", r.i, r.rating, r.ts)
    produce_s = time.perf_counter() - t0
    store = StateStore(Path(tmp) / "state.db", len(tt.genres))
    t0 = time.perf_counter()
    n = sum(TasteTwinConsumer(log, store, tt.G, p, batch_size=2000).run_until_caught_up() for p in range(4))
    consume_s = time.perf_counter() - t0
print(json.dumps({
    "evidence_type": "observed", "events": int(n), "viewers": int(ev["u"].nunique()), "partitions": 4,
    "produce_events_per_s": round(len(ev) / produce_s), "consume_events_per_s": round(n / consume_s),
    "consumer": "single process, partitions consumed sequentially; SQLite WAL state store, offset+state committed per batch of 2000",
    "note": "Local stand-ins for Kafka and an online KV store; one 4-vCPU container.",
}, indent=1))

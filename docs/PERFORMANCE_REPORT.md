# Performance report (measured)

**Machine:** one Linux container with 4 vCPU, 15 GB RAM, Python 3.11.18, no GPU.
**Data:** synthetic, 421k interactions, 7,956 users, 2,979 titles.
Sources: `reports/v4/synthetic/performance.json` and `api_load_test.json`. These are local measurements and
**say nothing about production capacity at scale**. See ARCHITECTURE §6 for the scale-out design, which is
not implemented.

## v5 update (11 channels incl. SASRec + two-tower, two-window ranker)

| Measure | v4 | v5 |
|---|---|---|
| Engine, one request, no explanations | p50 7.4 ms, p95 8.3 ms | p50 **14.8 ms**, p95 19.0 ms, p99 20.6 ms |
| Engine, one request, with explanations | p50 10.2 ms | p50 18.5 ms, p95 20.0 ms |
| Engine batch scoring | 282 users/s | 201 users/s |
| Bundle size / load | 137 MB / 0.40 s | 171 MB / 0.72 s |
| Peak RSS during build | 1.2 GB | 2.3 GB |
| Full pipeline | 2.3 min | 13.1 min measured for the single-window v5 run; the two-window pipeline adds one channel fit (~3–4 min), so **~16–17 min uncontended (estimate)**. Its measured run took 36 min because the test suite ran on the same 4 CPUs. |

Latency doubled because every request now runs a transformer forward pass (SASRec) and a two-tower user
encoding. That is the price of the +7.7% NDCG gain. The obvious next optimisation is to precompute and
cache user vectors for known viewers, refreshed by the streaming consumer, so only new events trigger an
encode. Training cost: SASRec about 130 s and two-tower about 35 s per fit on CPU; each is fit four
times per build.

Streaming TasteTwin (`streaming.json`): 21,941 events/s consumed and 36,706 events/s produced, on one
process.

ANN (`performance.json` → `ann`): HNSW recall@100 0.995 vs exact; 0.015 ms per query (exact 0.012 ms at
3k titles).

## v4 pipeline (`ntg build`, full grids): 136.9 s total

| Stage | Seconds | Peak RSS |
|---|---|---|
| prepare (DuckDB CSV → split Parquet) | 1.0 | 397 MB |
| tune (5 model families × grid on VAL) | 17.3 | 715 MB |
| fit 8 channels (train) | 5.8 | 715 MB |
| LambdaMART ranker training (385k rows, 38 features) | 5.7 | 817 MB |
| fit 8 channels (train+val) | 5.2 | 908 MB |
| test benchmark (10 models × 2 segments + 4 exploration levels, bootstrap) | 44.0 | 908 MB |
| TasteTwin eval (27-config VAL grid) | 1.1 | 908 MB |
| journey simulation (6 policies, 2 sensitivity sweeps, OPE) | 33.1 | 943 MB |
| serving bundle fit (all data) | 8.3 | 1,228 MB |

The slowest single model to fit is NeuMF at 19.2 s for 8 epochs in NumPy. The fastest is RP3β at 0.3 s.

## Serving

| Measure | Value |
|---|---|
| Bundle size / load time | 137 MB / 0.40 s (SHA-256 verified) |
| Engine, one request, k = 10, no explanations | p50 **7.4 ms**, p95 8.3 ms, p99 9.0 ms (n = 200) |
| Engine, one request, with explanations | p50 **10.2 ms**, p95 11.4 ms (n = 50) |
| Engine batch scoring | 282 users/s (one thread) |

### API end to end (`POST /v1/profiles/{id}/recommendations`, k = 20, half with explanations, 1 uvicorn worker, keep-alive client)

| Concurrency | Throughput | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|
| 1 | 62.1 req/s | 16.4 ms | 20.8 ms | 24.0 ms | 0 |
| 4 | 41.7 req/s | 98.7 ms | 129.6 ms | 135.7 ms | 0 |
| 8 | 37.5 req/s | 212.4 ms | 279.3 ms | 332.8 ms | 0 |

**Reading:** a single worker is CPU-bound and GIL-bound. Concurrency inside one process adds queueing, not
throughput, so throughput drops slightly from 62 to 38 req/s. Throughput scales by adding worker processes
(the Dockerfile runs 2), and each holds about 0.5–1 GB of model state. Above that, the design precomputes
candidates offline (ARCHITECTURE §6).

## Bottlenecks found and fixed

| Issue | Evidence | Fix | Result |
|---|---|---|---|
| pandas row access on the request path (item cards, titles) | cProfile: about a third of request time in `item_card` / `item_title` (pandas `_ixs`) | precompute item cards and genre lists at load time | explained recs p50 35 ms → 10.2 ms |
| sparse column slicing for "because you watched" | cProfile: about 12% in `csr __getitem__` | a CSC copy of the item-kNN matrix | included above |
| load-test client opening a connection per request | server-side p50 16 ms vs client-side 50 ms | keep-alive pool in `scripts/api_load_test.py` | measurement fixed (server time was already 16 ms) |
| SQLite commit per request (usage metering) | suspected | WAL plus `synchronous=NORMAL` | negligible change; not the bottleneck |
| v1 legacy: `build_graph` self-join on ML-32M | historical memory issues | v4 caps items with `min_item_support` and samples users by hash | ML-32M at 10% users fits in about 4 GB (estimate, not measured here) |

## Memory budget for real MovieLens 32M (estimate, not measured)

EASE and the dense item-item matrices are O(I²). With `--min-item-support 20` and a 10% user sample, the
catalogue is roughly 10–15k items. A float32 EASE matrix is 0.4–0.9 GB, and the inversion in float64
transiently needs about 2× that. That is feasible on a 16 GB laptop. A full 87k-item catalogue would need
sparse or approximate item models (ROADMAP).

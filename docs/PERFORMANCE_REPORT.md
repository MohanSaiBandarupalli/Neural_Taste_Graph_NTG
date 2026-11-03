# Performance report (measured)

**Machine:** one Linux container with 4 vCPU, 15 GB RAM, Python 3.11.18, no GPU.
**Data:** synthetic, 421k interactions, 7,956 users, 2,979 titles.
Sources: `reports/v4/synthetic/performance.json` and `api_load_test.json`. These are local measurements and
**say nothing about production capacity at scale**. See ARCHITECTURE §6 for the scale-out design, which is
not implemented.

## Pipeline (`ntg build`, full grids): 136.9 s total

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

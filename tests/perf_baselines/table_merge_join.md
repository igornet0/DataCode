# Table merge / join benchmarks

Commands:

```bash
cargo test --test table_tests -- --nocapture test_merge_tables test_table_alias
cargo test --release --test profiling_benchmarks join -- --nocapture
```

## Before (commit `a73fe46`, 2026-08-27)

Release, Apple Silicon, `profiling_benchmarks`:

| Benchmark | Input | Result rows | Wall time |
|-----------|--------|-------------|-----------|
| `bench_merge_tables_large` | 3 × 50_000, same schema | 150_000 | 41.930834ms |
| `bench_merge_tables_outer_mismatch` | 2 × 50_000, extra column | 100_000 | 34.028583ms |
| `bench_inner_join_large` | 50_000 × 50_000, 1:1 on `id` | 50_000 | 45.06575ms |

Notes:

- `native_merge_tables` materializes `Vec<Vec<Value>>` per input and does `headers().position` per cell.
- `Value::Table` clone deep-copies the flat buffer; join natives call `t.borrow().clone()` on inputs.
- `t2 = t1; t2.add_row(...)` currently aliases (`len(t1)` becomes 2): assignment copies the heap `ValueId`.

## After (this change)

Release, same machine and commands:

| Benchmark | Input | Result rows | Wall time | vs before |
|-----------|--------|-------------|-----------|-----------|
| `bench_merge_tables_large` | 3 × 50_000, same schema | 150_000 | 7.900959ms | ~5.3× faster |
| `bench_merge_tables_outer_mismatch` | 2 × 50_000, extra column | 100_000 | 4.957542ms | ~6.9× faster |
| `bench_inner_join_large` | 50_000 × 50_000, 1:1 on `id` | 50_000 | 34.406709ms | ~1.3× faster |

What changed:

- `Value::Table` clone shares `Rc` instead of deep-copying the flat buffer.
- `t2 = t1` forks a heap handle; `add_row` / `push` clone-on-write so `t1` stays unchanged.
- `merge_tables` same-schema uses `append_flat_chunk`; mismatched columns remap via a column index map (no per-cell `headers().position`).
- Join natives no longer deep-clone Owned inputs.

`cargo test --release --test table_tests`: 252 passed.

## Join variants baseline (2026-08-28)

Release, Apple Silicon, `cargo test --release --test profiling_benchmarks join -- --nocapture`:

| Benchmark | Input | Result rows | Wall time |
|-----------|--------|-------------|-----------|
| `bench_inner_join_large` | 50_000 × 50_000, 1:1 on `id` | 50_000 | 71.6ms |
| `bench_left_join_large` | 50_000 × 50_000, 1:1 on `id` | 50_000 | 74.5ms |
| `bench_right_join_large` | 50_000 × 50_000, 1:1 on `id` | 50_000 | 91.4ms |
| `bench_full_join_large` | 50_000 × 50_000, 1:1 on `id` | 50_000 | 88.7ms |
| `bench_semi_join_large` | 50_000 × 50_000, 1:1 on `id` | 50_000 | 46.4ms |
| `bench_anti_join_large` | 50_000 × 50_000, full key overlap | 0 | 36.2ms |
| `bench_zip_join_large` | 50_000 × 50_000 positional | 50_000 | 21.9ms |
| `bench_cross_join_1k` | 1_000 × 1_000 cartesian | 1_000_000 | 236.1ms |
| `bench_asof_join_large` | 50_000 × 50_000, `by=group`, `backward` | 50_000 | 120.2ms |
| `bench_join_on_equi_medium` | 2_000 × 2_000, `id == id` (nested loop) | 2_000 | 48.3ms |
| `bench_apply_join_medium` | 5_000 left rows, 1 row / call | 5_000 | 42.3ms |
| `bench_inner_join_then_suffixes_large` | inner 50k + suffixes | 50_000 | join 75.5ms + suffixes 24.2ms |

Notes:

- Equi-join family shares the same hash-join core as `inner_join`; timings are regression guards, not merge-scale wins.
- `join_on` is intentionally smaller (O(n²) nested loop).
- `apply_join` includes VM + user-function overhead per left row (5k rows).
- `cross_join` kept at 1k×1k (1M output rows).

## Join optimizations (2026-08-28)

Release, same machine, `cargo test --release --test profiling_benchmarks join -- --nocapture`:

| Benchmark | Before | After | Speedup |
|-----------|--------|-------|---------|
| `bench_inner_join_large` | 64.0ms | 18.3ms | ~3.5× |
| `bench_left_join_large` | 46.2ms | 23.5ms | ~2.0× |
| `bench_right_join_large` | 77.3ms | 26.1ms | ~3.0× |
| `bench_full_join_large` | 62.9ms | 39.0ms | ~1.6× |
| `bench_semi_join_large` | 59.2ms | 6.7ms | ~8.8× |
| `bench_anti_join_large` | 41.2ms | 18.4ms | ~2.2× |
| `bench_zip_join_large` | 21.4ms | 14.7ms | ~1.5× |
| `bench_cross_join_1k` | 215.0ms | 90.8ms | ~2.4× |
| `bench_asof_join_large` | 118.3ms | 54.3ms | ~2.2× |
| `bench_inner_join_then_suffixes_large` | join 58.6ms + suffixes 35.9ms | join 23.3ms + suffixes 8.6µs | suffixes ~4000× |

What changed:

- Equi-join output via flat buffer (`from_flat_owned`) instead of `Vec<Vec<Value>>` per row.
- Single-column numeric join keys use `HashMap<u64, _>` (no `Value` clone per key).
- Join key column indices precomputed once (`JoinKeyPlan`).
- `right_join` native path (no left-join + row shuffle).
- `table_suffixes` renames headers in-place (no table rebuild).
- `asof_join` uses binary search on sorted right times (numeric columns).

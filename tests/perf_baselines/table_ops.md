# Table ops performance baselines (50k rows, release)

Run wave 1:

```bash
cargo test --release --test profiling_benchmarks bench_table_ -- --nocapture
cargo test --release --test profiling_benchmarks bench_view_table -- --nocapture
```

Run wave 2:

```bash
cargo test --release --test profiling_benchmarks bench_table_value_map_large -- --nocapture
cargo test --release --test profiling_benchmarks bench_table_map_abs_large -- --nocapture
cargo test --release --test profiling_benchmarks bench_table_split_column_delim_large -- --nocapture
cargo test --release --test profiling_benchmarks bench_table_join_columns_large -- --nocapture
cargo test --release --test profiling_benchmarks bench_table_aggregate_large -- --nocapture
cargo test --release --test profiling_benchmarks bench_table_read_csv_large -- --nocapture
cargo test --release --test profiling_benchmarks bench_table_save_csv_large -- --nocapture
cargo test --release --test profiling_benchmarks bench_relate_primary_key_large -- --nocapture
```

Environment: local dev machine, `cargo test --release`, Aug 2026.

---

## Wave 1 — filter / slice / transform (flat-path)

### Before (`get_row().to_vec()` + `Table::from_data`)

| Benchmark | Rows out | Time |
|-----------|----------|------|
| `bench_table_head_large` | 5_000 | 27.5ms |
| `bench_table_tail_large` | 5_000 | 14.0ms |
| `bench_table_select_large` | 50_000 | 43.2ms |
| `bench_table_where_large` | 24_999 | 16.8ms |
| `bench_table_sort_large` | 50_000 | 45.9ms |
| `bench_table_drop_nulls_large` | 45_000 | 45.1ms |
| `bench_table_distinct_large` | 50_000 | 65.9ms |
| `bench_table_rename_large` | 50_000 | 45.4ms |
| `bench_table_drop_column_large` | 50_000 | 43.3ms |
| `bench_table_row_number_large` | 50_000 | 54.6ms |
| `bench_table_add_column_large` | 50_000 | 58.5ms |

### After (`slice_rows_owned`, `gather_rows_owned`, `select_columns_owned`, …)

| Benchmark | Rows out | Time | Speedup |
|-----------|----------|------|---------|
| `bench_table_head_large` | 5_000 | 8.6ms | ~3.2× |
| `bench_table_tail_large` | 5_000 | 9.5ms | ~1.5× |
| `bench_table_select_large` | 50_000 | 4–14ms | ~3–10× |
| `bench_table_where_large` | 24_999 | 10.0ms | ~1.7× |
| `bench_table_sort_large` | 50_000 | 11.6ms | ~4× |
| `bench_table_drop_nulls_large` | 45_000 | 16.9ms | ~2.7× |
| `bench_table_distinct_large` | 50_000 | 36.0ms | ~1.8× |
| `bench_table_rename_large` | 50_000 | 19.6ms | ~2.3× |
| `bench_table_drop_column_large` | 50_000 | 14.9ms | ~2.9× |
| `bench_table_row_number_large` | 50_000 | 20.1ms | ~2.7× |
| `bench_table_add_column_large` | 50_000 | 18.9ms | ~3.1× |
| `bench_table_aggregate_group_large` | 50_000 groups | 28.5ms | (new) |
| `bench_view_table_merge_push_large` | 10k view chain | 5.5ms | (new) |

---

## Wave 2 — map / column ops / aggregate / I/O / relations

### Before (row materialization + multi-pass column append)

| Benchmark | Rows out | Time |
|-----------|----------|------|
| `bench_table_value_map_large` | 50_000 | 9.2ms |
| `bench_table_map_abs_large` | 50_000 | (script; ~7–15ms est. row path) |
| `bench_table_split_column_delim_large` | 50_000 | 21.0ms |
| `bench_table_join_columns_large` | 50_000 | 15.8ms |
| `bench_table_aggregate_large` | 1 | 6.6ms |
| `bench_table_read_csv_large` | 50_000 | 9.6ms |
| `bench_table_save_csv_large` | 50_000 | broken¹ |
| `bench_relate_primary_key_large` | 100× on 50k | 64.7µs |

¹ `materialize_table` always required VM context even for owned tables.

### After (flat in-place / single-pass append / no owned clone on save)

| Benchmark | Rows out | Time | Speedup |
|-----------|----------|------|---------|
| `bench_table_value_map_large` | 50_000 | 8.9ms | ~1.0× |
| `bench_table_map_abs_large` | 50_000 | 7.4ms | ~flat path |
| `bench_table_split_column_delim_large` | 50_000 | 11.4ms | ~1.8× |
| `bench_table_join_columns_large` | 50_000 | 11.8ms | ~1.3× |
| `bench_table_aggregate_large` | 1 | 4.3ms | ~1.5× |
| `bench_table_read_csv_large` | 50_000 | 8.7ms | ~1.1× |
| `bench_table_save_csv_large` | 50_000 | 8.2ms | new (no full clone) |
| `bench_relate_primary_key_large` | 100× on 50k | 41µs | ~1.6× |

---

## Changes

### Wave 1
- `src/common/table.rs`: `slice_rows_owned`, `gather_rows_owned`, `select_columns_owned`, `clone_flat_with_headers`, `append_column_owned`.
- `src/vm/natives/table.rs`: owned fast paths for head/tail/select/where/sort/drop_nulls/distinct/rename/add_column; `column_view` for filter/sort.
- `src/vm/table_ops.rs`: `build_view_flat_from_row_slots` (column-oriented + `range()` iterables).
- `src/vm/runtime/call_engine/native_call/fast_paths.rs`, `execute.rs`: view fast path.

### Wave 2
- `src/common/table.rs`: `map_column_owned`, `append_columns_owned`, `clone_flat_owned`.
- `src/vm/natives/table.rs`: owned paths for `value_map`, `map`, `split_column` (delim), `join_columns`; `apply_aggregate_op_on_table` without row materialization for owned tables + `aggregate_group` bucket scan on flat.
- `src/vm/natives/table_save.rs`: owned tables exported in-place (`write_table_csv` / `export_single_table` borrow); view-only materialization needs VM context.
- `tests/profiling_benchmarks.rs`: wave-2 benchmarks added.

## Still copy-heavy (next wave)

- `table_map` / `column_map` on View tables (still materialize)
- `table_split_column` with VM callback (`iter_fn`)
- `read` / large XLSX paths
- `save_tables_sqlite` multi-table export
- `relate` / `primary_key` correctness under CoW/share (metadata only today; binding tests TBD)

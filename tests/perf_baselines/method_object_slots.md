# Method-call receiver temp slot strategies

Compares three compiler strategies for `__method_object_*` locals used during method-call compilation.

| Strategy | Cargo feature | Behavior |
|----------|---------------|----------|
| **depth_stack** (default) | *(none)* | One slot per nesting depth (`__method_object_0`, `_1`, …); reset per function |
| **per_call** (legacy) | `method_object_per_call` | New local on every method call site (bloats frame size) |
| **shared** (broken) | `method_object_shared` | Single reused slot — corrupts nested calls in arguments |

Run:

```bash
cargo test --release --test profiling_benchmarks bench_method_object -- --nocapture
cargo test --release --features method_object_per_call --test profiling_benchmarks bench_method_object -- --nocapture
cargo test --release --features method_object_shared --test profiling_benchmarks bench_method_object -- --nocapture
```

Environment: release build, local run (Aug 2026).

## Results

### Frame size (`max_locals` from bytecode) — critical for functions with many call sites

| Benchmark | depth_stack | per_call | shared |
|-----------|-------------|----------|--------|
| `many_sites_locals` (2000 sequential `a.push(i)`) | **3** | **2002** | 3 |
| `flat_1m` loop body (100k pushes) | 5 | 5 | 5 |
| `nested_1m` loop body (100k, nested `_capacity()` in index) | 6 | 6 | 6 |

**Takeaway:** Loop bodies compile once, so flat/nested loops use O(1) temp slots regardless of strategy. Functions with **many distinct call sites** (generated code, large DSL emitters) benefit most: depth_stack keeps frames tiny (3 vs 2002 locals).

### Runtime (100k iterations; 1M native table op separate)

| Benchmark | depth_stack | per_call | shared |
|-----------|-------------|----------|--------|
| `many_sites_locals` | 39.97 ms | 37.77 ms | 35.86 ms |
| `flat_1m` (100k pushes) | **90.60 s** | 112.52 s | 105.20 s |
| `nested_1m` (100k deque-like) | 788.28 s | 773.98 s | **FAIL** (wrong result) |
| `table_where_1m` (1M rows, native) | 6.34 ms | 5.98 ms | 4.05 ms |

**Takeaway:**

- **Correctness:** only `shared` breaks nested method calls (`nested_1m` assertion failure). depth_stack and per_call both pass `example_deque_dc_runs`.
- **Large row data (1M+):** table natives (`table_where_1m`) are unaffected by slot strategy (~6 ms). Method-slot optimization is not the lever for bulk table throughput.
- **Method-heavy loops:** depth_stack is ~19% faster than per_call on flat 100k pushes (smaller `CallFrame.slots`). Nested path is dominated by double method dispatch per row (~8× flat cost); slot strategy difference is within noise except shared failure.
- **Many call sites:** runtime similar (~36–40 ms for 2000 pushes); memory / frame footprint is the differentiator (3 vs 2002 locals).

### Full 1M push loop (manual, slow)

```bash
cargo test --release --test profiling_benchmarks bench_method_object_flat_1m_full -- --ignored --nocapture
```

Estimated ~30–40 min per run. Not executed in this baseline batch.

## Recommendation

**Use depth_stack (default).** It matches per_call correctness, avoids shared-slot corruption on nested calls (e.g. `result.push(… this._capacity() …)`), shrinks frames for multi-site functions, and slightly improves method-loop runtime. Requires `reset_method_object_temps()` whenever `local_count` is reset for a new function/method body.

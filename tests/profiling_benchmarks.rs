// Profiling benchmarks for GIL-bottleneck analysis.
//
// Run with: cargo test profiling_benchmarks --release -- --nocapture
//
// GIL experiment (run manually, may take tens of seconds):
//   cargo test test_gil_concurrent --release -- --ignored --nocapture
//
// Interpretation: T8 ≈ T1 → no serialization (no GIL). T8 ≈ 8*T1 → full GIL.
// T8 ≈ 2–3*T1 → partial serialization. CPU monitor is secondary; wall-clock scaling is primary.
//
// To profile allocations/locks:
//   - Allocations: cargo test bench_native_calls_heavy --release -- --nocapture
//     then run with heaptrack, or RUSTFLAGS="-g" cargo test ... and use valgrind --tool=massif
//   - Locks: use perf record -g on the test binary, or a contention profiler
//   - Duration baseline: the printed timings give a baseline for before/after optimizations

#[cfg(test)]
mod tests {
    use data_code::run;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::thread;
    use std::time::Instant;

    /// Mode A: pure arithmetic, minimal allocations (executor + value_store).
    fn script_mode_a(n: u32) -> String {
        format!(
            r#"
let acc = 0
for i in range({n}) {{
  let acc = acc + i
}}
acc
"#,
            n = n
        )
    }

    /// Mode B: object creation in loop (allocation + object creation).
    fn script_mode_b(n: u32) -> String {
        format!(
            r#"
for i in range({n}) {{
  let obj = {{"a": i, "b": i+1}}
}}
0
"#,
            n = n
        )
    }

    /// Mode C: class method calls in loop (method lookup, class metadata).
    fn script_mode_c(n: u32) -> String {
        format!(
            r#"
cls A {{
  new A() {{}}
  fn inc(x) {{
    return x + 1
  }}
}}
for i in range({n}) {{
  A.inc(i)
}}
0
"#,
            n = n
        )
    }

    /// Hot path: many native calls (arithmetic, len, range) to stress load_value/store_value and args buffer.
    #[test]
    fn bench_native_calls_heavy() {
        let n: u32 = 10_000;
        let source = format!(
            r#"
let s = 0
for i in range({n}) {{
  let s = s + 1
  let _ = len("x")
}}
s
"#,
            n = n
        );
        let start = Instant::now();
        let r = run(&source);
        let elapsed = start.elapsed();
        assert!(r.is_ok(), "run failed: {:?}", r);
        if let Ok(data_code::Value::Number(v)) = r {
            assert_eq!(v, n as f64, "expected sum {}", n);
        }
        println!(
            "bench_native_calls_heavy: {} iterations in {:?}",
            n, elapsed
        );
    }

    /// Hot path: table operations (native table(), iteration) to stress Value/HeavyStore and relations.
    #[test]
    fn bench_table_ops_heavy() {
        let rows: u32 = 5_000;
        let source = format!(
            r#"
let t = table([["a","b"],[1,2],[3,4]])
for i in range({rows}) {{
  let _ = table([["x","y"],[i, i+1]])
}}
len(t)
"#,
            rows = rows
        );
        let start = Instant::now();
        let r = run(&source);
        let elapsed = start.elapsed();
        assert!(r.is_ok(), "run failed: {:?}", r);
        println!(
            "bench_table_ops_heavy: {} table creations in {:?}",
            rows, elapsed
        );
    }

    /// Mixed: arithmetic + array push (mutability write-back path).
    #[test]
    fn bench_arithmetic_and_arrays() {
        let n: u32 = 5_000;
        let source = format!(
            r#"
let acc = 0
let a = []
for i in range({n}) {{
  push(a, i)
  let acc = acc + 1
}}
acc
"#,
            n = n
        );
        let start = Instant::now();
        let r = run(&source);
        let elapsed = start.elapsed();
        assert!(r.is_ok(), "run failed: {:?}", r);
        if let Ok(data_code::Value::Number(v)) = r {
            assert_eq!(v, n as f64);
        }
        println!("bench_arithmetic_and_arrays: {} iters in {:?}", n, elapsed);
    }

    /// GIL experiment: 1 thread vs 8 threads, same workload. T8 ≈ T1 → no serialization; T8 ≈ 8*T1 → GIL.
    /// Run: cargo test test_gil_concurrent --release -- --ignored --nocapture
    #[test]
    #[ignore = "GIL experiment: run with --ignored --nocapture; takes 1-2 min"]
    fn test_gil_concurrent() {
        const NUM_THREADS: usize = 8;
        const K_ITERATIONS: u32 = 10;

        // n chosen so one run() is ~1-2 s (tune per machine)
        const N_MODE_A: u32 = 600_000;
        const N_MODE_B: u32 = 120_000;
        const N_MODE_C: u32 = 120_000;

        let run_one_thread = |script: &str, k: u32| -> std::time::Duration {
            let start = Instant::now();
            for _ in 0..k {
                let r = run(script);
                assert!(r.is_ok(), "run failed: {:?}", r);
            }
            start.elapsed()
        };

        let run_many_threads =
            |script: Arc<String>, k: u32, num_threads: usize| -> std::time::Duration {
                let start = Instant::now();
                let handles: Vec<_> = (0..num_threads)
                    .map(|_| {
                        let script = Arc::clone(&script);
                        thread::spawn(move || {
                            for _ in 0..k {
                                let r = run(script.as_str());
                                assert!(r.is_ok(), "run failed: {:?}", r);
                            }
                        })
                    })
                    .collect();
                for h in handles {
                    h.join().expect("thread panicked");
                }
                start.elapsed()
            };

        println!(
            "--- GIL experiment: {} threads, {} iterations per thread ---",
            NUM_THREADS, K_ITERATIONS
        );

        // Mode A: arithmetic
        let script_a = script_mode_a(N_MODE_A);
        let t1_a = run_one_thread(&script_a, K_ITERATIONS);
        let t8_a = run_many_threads(Arc::new(script_a.clone()), K_ITERATIONS, NUM_THREADS);
        let ratio_a = t8_a.as_secs_f64() / t1_a.as_secs_f64();
        println!(
            "Mode A (arithmetic):  T1 = {:?}, T8 = {:?}, T8/T1 = {:.2}",
            t1_a, t8_a, ratio_a
        );
        assert!(ratio_a < 3.0, "Mode A ratio {:.2} suggests serialization; expect < 3 (2-3 = partial contention, 8 = GIL)", ratio_a);

        // Mode B: objects
        let script_b = script_mode_b(N_MODE_B);
        let t1_b = run_one_thread(&script_b, K_ITERATIONS);
        let t8_b = run_many_threads(Arc::new(script_b.clone()), K_ITERATIONS, NUM_THREADS);
        let ratio_b = t8_b.as_secs_f64() / t1_b.as_secs_f64();
        println!(
            "Mode B (objects):    T1 = {:?}, T8 = {:?}, T8/T1 = {:.2}",
            t1_b, t8_b, ratio_b
        );
        assert!(ratio_b < 3.0, "Mode B ratio {:.2} suggests serialization; expect < 3 (2-3 = partial contention, 8 = GIL)", ratio_b);

        // Mode C: class methods
        let script_c = script_mode_c(N_MODE_C);
        let t1_c = run_one_thread(&script_c, K_ITERATIONS);
        let t8_c = run_many_threads(Arc::new(script_c.clone()), K_ITERATIONS, NUM_THREADS);
        let ratio_c = t8_c.as_secs_f64() / t1_c.as_secs_f64();
        println!(
            "Mode C (class):      T1 = {:?}, T8 = {:?}, T8/T1 = {:.2}",
            t1_c, t8_c, ratio_c
        );
        assert!(ratio_c < 3.0, "Mode C ratio {:.2} suggests serialization; expect < 3 (2-3 = partial contention, 8 = GIL)", ratio_c);

        println!("--- ratio < 2 => architecture clean; ratio 2-3 => partial contention; T8/T1 ~ 8 => full serialization ---");
    }

    /// GIL experiment with 16 threads: check scaling when thread count exceeds typical core count.
    /// Run: cargo test test_gil_concurrent_16_threads --release -- --ignored --nocapture
    #[test]
    #[ignore = "GIL experiment 16 threads: run with --ignored --nocapture; takes 1-2 min"]
    fn test_gil_concurrent_16_threads() {
        const NUM_THREADS: usize = 16;
        const K_ITERATIONS: u32 = 5;

        const N_MODE_A: u32 = 600_000;
        const N_MODE_B: u32 = 120_000;
        const N_MODE_C: u32 = 120_000;

        let run_one_thread = |script: &str, k: u32| -> std::time::Duration {
            let start = Instant::now();
            for _ in 0..k {
                let r = run(script);
                assert!(r.is_ok(), "run failed: {:?}", r);
            }
            start.elapsed()
        };

        let run_many_threads =
            |script: Arc<String>, k: u32, num_threads: usize| -> std::time::Duration {
                let start = Instant::now();
                let handles: Vec<_> = (0..num_threads)
                    .map(|_| {
                        let script = Arc::clone(&script);
                        thread::spawn(move || {
                            for _ in 0..k {
                                let r = run(script.as_str());
                                assert!(r.is_ok(), "run failed: {:?}", r);
                            }
                        })
                    })
                    .collect();
                for h in handles {
                    h.join().expect("thread panicked");
                }
                start.elapsed()
            };

        println!(
            "--- GIL experiment: {} threads, {} iterations per thread ---",
            NUM_THREADS, K_ITERATIONS
        );

        let script_a = script_mode_a(N_MODE_A);
        let t1_a = run_one_thread(&script_a, K_ITERATIONS);
        let t16_a = run_many_threads(Arc::new(script_a.clone()), K_ITERATIONS, NUM_THREADS);
        let ratio_a = t16_a.as_secs_f64() / t1_a.as_secs_f64();
        println!(
            "Mode A (arithmetic):  T1 = {:?}, T16 = {:?}, T16/T1 = {:.2}",
            t1_a, t16_a, ratio_a
        );
        assert!(ratio_a < 5.0, "Mode A (16 threads) ratio {:.2} suggests serialization; expect < 5 (allocator contention ok)", ratio_a);

        let script_b = script_mode_b(N_MODE_B);
        let t1_b = run_one_thread(&script_b, K_ITERATIONS);
        let t16_b = run_many_threads(Arc::new(script_b.clone()), K_ITERATIONS, NUM_THREADS);
        let ratio_b = t16_b.as_secs_f64() / t1_b.as_secs_f64();
        println!(
            "Mode B (objects):    T1 = {:?}, T16 = {:?}, T16/T1 = {:.2}",
            t1_b, t16_b, ratio_b
        );
        assert!(
            ratio_b < 5.0,
            "Mode B (16 threads) ratio {:.2} suggests serialization; expect < 5",
            ratio_b
        );

        let script_c = script_mode_c(N_MODE_C);
        let t1_c = run_one_thread(&script_c, K_ITERATIONS);
        let t16_c = run_many_threads(Arc::new(script_c.clone()), K_ITERATIONS, NUM_THREADS);
        let ratio_c = t16_c.as_secs_f64() / t1_c.as_secs_f64();
        println!(
            "Mode C (class):      T1 = {:?}, T16 = {:?}, T16/T1 = {:.2}",
            t1_c, t16_c, ratio_c
        );
        assert!(
            ratio_c < 5.0,
            "Mode C (16 threads) ratio {:.2} suggests serialization; expect < 5",
            ratio_c
        );

        println!("--- 16 threads: ratio < 5 => no GIL; T16/T1 ~ 16 => full serialization ---");
    }

    /// Allocator comparison: same workload for 1, 8, and 16 threads; outputs T1, T8, T16 and ratios.
    /// Run: cargo test test_allocator_comparison --release --features allocator_jemalloc -- --ignored --nocapture
    /// Peak memory: measure with /usr/bin/time -v (Linux) or Instruments (macOS) or heaptrack.
    #[test]
    #[ignore = "Allocator comparison: run with --ignored --nocapture; use scripts/compare_allocators.sh"]
    fn test_allocator_comparison() {
        const K_ITERATIONS: u32 = 5;
        const N_MODE_A: u32 = 600_000;
        const N_MODE_B: u32 = 120_000;
        const N_MODE_C: u32 = 120_000;

        let run_one_thread = |script: &str, k: u32| -> std::time::Duration {
            let start = Instant::now();
            for _ in 0..k {
                let r = run(script);
                assert!(r.is_ok(), "run failed: {:?}", r);
            }
            start.elapsed()
        };

        let run_many_threads =
            |script: Arc<String>, k: u32, num_threads: usize| -> std::time::Duration {
                let start = Instant::now();
                let handles: Vec<_> = (0..num_threads)
                    .map(|_| {
                        let script = Arc::clone(&script);
                        thread::spawn(move || {
                            for _ in 0..k {
                                let r = run(script.as_str());
                                assert!(r.is_ok(), "run failed: {:?}", r);
                            }
                        })
                    })
                    .collect();
                for h in handles {
                    h.join().expect("thread panicked");
                }
                start.elapsed()
            };

        fn report_mode(
            name: &str,
            t1: std::time::Duration,
            t8: std::time::Duration,
            t16: std::time::Duration,
        ) {
            let s1 = t1.as_secs_f64();
            let s8 = t8.as_secs_f64();
            let s16 = t16.as_secs_f64();
            let r8 = s8 / s1;
            let r16 = s16 / s1;
            println!(
                "Mode {}: T1 = {:.2}s, T8 = {:.2}s, T16 = {:.2}s, T8/T1 = {:.2}, T16/T1 = {:.2}, peak_memory = (use time -v or heaptrack)",
                name, s1, s8, s16, r8, r16
            );
            println!(
                "ALLOC_BENCH Mode={} T1={:.3} T8={:.3} T16={:.3} T8/T1={:.3} T16/T1={:.3}",
                name, s1, s8, s16, r8, r16
            );
        }

        println!(
            "--- Allocator comparison: 1 / 8 / 16 threads, {} iterations per thread ---",
            K_ITERATIONS
        );

        let script_a = script_mode_a(N_MODE_A);
        let t1_a = run_one_thread(&script_a, K_ITERATIONS);
        let t8_a = run_many_threads(Arc::new(script_a.clone()), K_ITERATIONS, 8);
        let t16_a = run_many_threads(Arc::new(script_a), K_ITERATIONS, 16);
        report_mode("A", t1_a, t8_a, t16_a);

        let script_b = script_mode_b(N_MODE_B);
        let t1_b = run_one_thread(&script_b, K_ITERATIONS);
        let t8_b = run_many_threads(Arc::new(script_b.clone()), K_ITERATIONS, 8);
        let t16_b = run_many_threads(Arc::new(script_b), K_ITERATIONS, 16);
        report_mode("B", t1_b, t8_b, t16_b);

        let script_c = script_mode_c(N_MODE_C);
        let t1_c = run_one_thread(&script_c, K_ITERATIONS);
        let t8_c = run_many_threads(Arc::new(script_c.clone()), K_ITERATIONS, 8);
        let t16_c = run_many_threads(Arc::new(script_c), K_ITERATIONS, 16);
        report_mode("C", t1_c, t8_c, t16_c);

        println!(
            "--- Peak memory: run with /usr/bin/time -v or heaptrack and fill table manually ---"
        );
    }

    /// Same blocked layout as Python `stress_test` (seed 42, 10_000 cells, start/goal cleared).
    fn python_stress_blocked_literal(rows: u32, cols: u32, goal_r: u32, goal_c: u32) -> String {
        let mut rng = StdRng::seed_from_u64(42);
        let mut blocked: HashSet<(u32, u32)> = HashSet::new();
        for _ in 0..10_000 {
            blocked.insert((
                rng.gen_range(0..rows),
                rng.gen_range(0..cols),
            ));
        }
        blocked.remove(&(0, 0));
        blocked.remove(&(goal_r, goal_c));
        let mut pairs: Vec<(u32, u32)> = blocked.into_iter().collect();
        pairs.sort_unstable();
        let cells: String = pairs
            .iter()
            .map(|(r, c)| format!("({}, {})", r, c))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            r#"
import heapq

fn a_star(rows, cols, start, goal, blocked) {{
    start_id = start[0] * cols + start[1]
    goal_id = goal[0] * cols + goal[1]
    goal_r, goal_c = goal
    blocked_ids = {{r * cols + c for r, c in blocked}}
    if start_id in blocked_ids or goal_id in blocked_ids: return null
    g_score = {{start_id: 0}}
    f_score = {{start_id: abs(start[0] - goal_r) + abs(start[1] - goal_c)}}
    closed_set = set()
    open_heap = [(f_score[start_id], start_id)]
    open_set = {{start_id}}
    neighbors_delta = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while open_heap {{
        current_f, current = heapq.heappop(open_heap)
        if current_f != f_score.get(current): continue
        open_set.discard(current)
        if current in closed_set: continue
        if current == goal_id {{ return 1 }}
        closed_set.add(current)
        r, c = divmod(current, cols)
        for dr, dc in neighbors_delta {{
            nr, nc = r + dr, c + dc
            if !(0 <= nr < rows and 0 <= nc < cols): continue
            neighbor = nr * cols + nc
            if neighbor in blocked_ids or neighbor in closed_set: continue
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, float(inf)) {{
                g_score[neighbor] = tentative_g
                h = abs(nr - goal_r) + abs(nc - goal_c)
                f = tentative_g + h
                f_score[neighbor] = f
                if !neighbor in open_set {{
                    heapq.heappush(open_heap, (f, neighbor))
                    open_set.add(neighbor)
                }}
            }}
        }}
    }}
    return null
}}

let blocked = set([{cells}])
a_star({rows}, {cols}, (0, 0), ({goal_r}, {goal_c}), blocked)
"#,
            cells = cells,
            rows = rows,
            cols = cols,
            goal_r = goal_r,
            goal_c = goal_c
        )
    }

    fn astar_grid_script(rows: u32, cols: u32, goal_r: u32, goal_c: u32) -> String {
        format!(
            r#"
import heapq

fn a_star(rows, cols, start, goal, blocked) {{
    start_id = start[0] * cols + start[1]
    goal_id = goal[0] * cols + goal[1]
    goal_r, goal_c = goal
    blocked_ids = {{r * cols + c for r, c in blocked}}
    if start_id in blocked_ids or goal_id in blocked_ids: return null
    g_score = {{start_id: 0}}
    f_score = {{start_id: abs(start[0] - goal_r) + abs(start[1] - goal_c)}}
    closed_set = set()
    open_heap = [(f_score[start_id], start_id)]
    open_set = {{start_id}}
    neighbors_delta = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while open_heap {{
        current_f, current = heapq.heappop(open_heap)
        if current_f != f_score.get(current): continue
        open_set.discard(current)
        if current in closed_set: continue
        if current == goal_id {{ return 1 }}
        closed_set.add(current)
        r, c = divmod(current, cols)
        for dr, dc in neighbors_delta {{
            nr, nc = r + dr, c + dc
            if !(0 <= nr < rows and 0 <= nc < cols): continue
            neighbor = nr * cols + nc
            if neighbor in blocked_ids or neighbor in closed_set: continue
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, float(inf)) {{
                g_score[neighbor] = tentative_g
                h = abs(nr - goal_r) + abs(nc - goal_c)
                f = tentative_g + h
                f_score[neighbor] = f
                if !neighbor in open_set {{
                    heapq.heappush(open_heap, (f, neighbor))
                    open_set.add(neighbor)
                }}
            }}
        }}
    }}
    return null
}}

a_star({rows}, {cols}, (0, 0), ({goal_r}, {goal_c}), set())
"#,
            rows = rows,
            cols = cols,
            goal_r = goal_r,
            goal_c = goal_c
        )
    }

    #[test]
    fn bench_astar_mini_20x20() {
        let source = astar_grid_script(20, 20, 5, 7);
        let start = Instant::now();
        assert!(run(&source).is_ok());
        println!("bench_astar_mini_20x20: elapsed={:?}", start.elapsed());
    }

    #[test]
    fn bench_astar_small_50x50() {
        let source = astar_grid_script(50, 50, 12, 34);
        let start = Instant::now();
        assert!(run(&source).is_ok());
        println!("bench_astar_small_50x50: elapsed={:?}", start.elapsed());
    }

    #[test]
    fn bench_astar_medium_200x200() {
        let source = astar_grid_script(200, 200, 50, 150);
        let start = Instant::now();
        assert!(run(&source).is_ok());
        println!("bench_astar_medium_200x200: elapsed={:?}", start.elapsed());
    }

    /// Baseline (release, same script): ~74–87 s wall, ~3–5 GiB peak RSS.
    /// Measure: `/usr/bin/time -l cargo test bench_astar_adv_2_single --release -- --ignored --nocapture --test-threads=1`
    #[test]
    #[ignore = "1000x5000 A*; run: cargo test bench_astar_adv_2_single --release -- --ignored --nocapture --test-threads=1 (wall limit ~300s via /usr/bin/time -l)"]
    fn bench_astar_adv_2_single() {
        let source = astar_grid_script(1000, 5000, 559, 1234);
        let start = Instant::now();
        assert!(run(&source).is_ok());
        println!("bench_astar_adv_2_single: elapsed={:?}", start.elapsed());
    }

    #[test]
    #[ignore = "1000x5000 A* with 10k blocked (Python stress layout); cargo test bench_astar_adv_2_blocked_single --release -- --ignored --nocapture --test-threads=1 (wall limit ~600s via /usr/bin/time -l)"]
    fn bench_astar_adv_2_blocked_single() {
        let source = python_stress_blocked_literal(1000, 5000, 559, 1234);
        let start = Instant::now();
        assert!(run(&source).is_ok());
        println!(
            "bench_astar_adv_2_blocked_single: elapsed={:?}",
            start.elapsed()
        );
    }

    #[test]
    #[ignore = "10 sequential A* runs; cargo test bench_astar_stress_10_sequential --release -- --ignored --nocapture --test-threads=1 (wall limit ~3600s via /usr/bin/time -l)"]
    fn bench_astar_stress_10_sequential() {
        const RUNS: u32 = 10;
        let source = astar_grid_script(1000, 5000, 559, 1234);
        let start = Instant::now();
        for i in 0..RUNS {
            assert!(run(&source).is_ok(), "run {} failed", i + 1);
        }
        let total = start.elapsed();
        println!(
            "bench_astar_stress_10_sequential: total={:?}, avg={:?}",
            total,
            total / RUNS
        );
    }

    #[test]
    #[ignore = "10 parallel A* runs (one Vm per thread); cargo test bench_astar_stress_10_parallel --release -- --ignored --nocapture (wall limit ~3600s via /usr/bin/time -l)"]
    fn bench_astar_stress_10_parallel() {
        const RUNS: usize = 10;
        let source = Arc::new(astar_grid_script(1000, 5000, 559, 1234));
        let start = Instant::now();
        let handles: Vec<_> = (0..RUNS)
            .map(|_| {
                let script = Arc::clone(&source);
                thread::spawn(move || {
                    assert!(run(script.as_str()).is_ok());
                })
            })
            .collect();
        for h in handles {
            h.join().expect("thread panicked");
        }
        let total = start.elapsed();
        println!(
            "bench_astar_stress_10_parallel: total={:?}, avg={:?}",
            total,
            total / RUNS as u32
        );
    }

    fn bench_make_table(n: usize, id_offset: f64) -> data_code::Value {
        use data_code::common::table::Table;
        use data_code::Value;
        use std::cell::RefCell;
        use std::rc::Rc;

        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            let id = id_offset + i as f64;
            rows.push(vec![
                Value::Number(id),
                Value::Number(id * 2.0),
                Value::String("x".to_string()),
            ]);
        }
        Value::Table(Rc::new(RefCell::new(Table::from_data(
            rows,
            Some(vec!["id".into(), "val".into(), "tag".into()]),
        ))))
    }

    fn bench_table_len(v: &data_code::Value) -> usize {
        match v {
            data_code::Value::Table(t) => t.borrow().len(),
            _ => 0,
        }
    }

    fn bench_assert_table(v: &data_code::Value) -> usize {
        match v {
            data_code::Value::Table(t) => t.borrow().len(),
            _ => panic!("expected table, got {:?}", v),
        }
    }

    fn bench_equi_join_pair(n: usize) -> [data_code::Value; 4] {
        use data_code::Value;
        let left = bench_make_table(n, 0.0);
        let right = bench_make_table(n, 0.0);
        [
            left,
            right,
            Value::String("id".to_string()),
            Value::String("id".to_string()),
        ]
    }

    fn bench_make_asof_table(n: usize, time_bias: f64) -> data_code::Value {
        use data_code::common::table::Table;
        use data_code::Value;
        use std::cell::RefCell;
        use std::rc::Rc;

        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            let group = (i % 100) as f64;
            let time = i as f64 + time_bias;
            rows.push(vec![
                Value::Number(group),
                Value::Number(time),
                Value::String("x".to_string()),
            ]);
        }
        Value::Table(Rc::new(RefCell::new(Table::from_data(
            rows,
            Some(vec!["group".into(), "time".into(), "tag".into()]),
        ))))
    }

    macro_rules! bench_equi_join {
        ($name:ident, $native:path, $n:expr, $expected:expr, $label:expr) => {
            #[test]
            fn $name() {
                const N: usize = $n;
                let args = bench_equi_join_pair(N);
                let start = Instant::now();
                let joined = $native(&args);
                let elapsed = start.elapsed();
                let rows = bench_assert_table(&joined);
                assert_eq!(rows, $expected, "unexpected row count for {}", $label);
                println!(
                    "{}: {} x {} -> {} rows in {:?}",
                    stringify!($name),
                    N,
                    N,
                    rows,
                    elapsed
                );
            }
        };
    }

    bench_equi_join!(
        bench_left_join_large,
        data_code::vm::natives::native_left_join,
        50_000,
        50_000,
        "left_join"
    );
    bench_equi_join!(
        bench_right_join_large,
        data_code::vm::natives::native_right_join,
        50_000,
        50_000,
        "right_join"
    );
    bench_equi_join!(
        bench_full_join_large,
        data_code::vm::natives::native_full_join,
        50_000,
        50_000,
        "full_join"
    );
    bench_equi_join!(
        bench_semi_join_large,
        data_code::vm::natives::native_semi_join,
        50_000,
        50_000,
        "semi_join"
    );

    /// Anti join on identical keys: no left-only rows.
    #[test]
    fn bench_anti_join_large() {
        use data_code::vm::natives::native_anti_join;

        const N: usize = 50_000;
        let args = bench_equi_join_pair(N);
        let start = Instant::now();
        let joined = native_anti_join(&args);
        let elapsed = start.elapsed();
        let rows = bench_table_len(&joined);
        assert_eq!(rows, 0, "anti_join on full key overlap should be empty");
        println!(
            "bench_anti_join_large: {} x {} -> {} rows in {:?}",
            N, N, rows, elapsed
        );
    }

    /// Positional zip: min(left, right) rows.
    #[test]
    fn bench_zip_join_large() {
        use data_code::vm::natives::native_zip_join;

        const N: usize = 50_000;
        let left = bench_make_table(N, 0.0);
        let right = bench_make_table(N, 0.0);
        let args = [left, right];

        let start = Instant::now();
        let joined = native_zip_join(&args);
        let elapsed = start.elapsed();
        let rows = bench_assert_table(&joined);
        assert_eq!(rows, N);
        println!(
            "bench_zip_join_large: {} x {} -> {} rows in {:?}",
            N, N, rows, elapsed
        );
    }

    /// Cartesian product — keep modest (1M output rows).
    #[test]
    fn bench_cross_join_1k() {
        use data_code::vm::natives::native_cross_join;

        const N: usize = 1_000;
        let left = bench_make_table(N, 0.0);
        let right = bench_make_table(N, 0.0);
        let args = [left, right];

        let start = Instant::now();
        let joined = native_cross_join(&args);
        let elapsed = start.elapsed();
        let rows = bench_assert_table(&joined);
        assert_eq!(rows, N * N);
        println!(
            "bench_cross_join_1k: {} x {} -> {} rows in {:?}",
            N, N, rows, elapsed
        );
    }

    /// ASOF with `by` + `direction=backward` on 50k rows (100 groups).
    #[test]
    fn bench_asof_join_large() {
        use data_code::vm::natives::native_asof_join;
        use data_code::Value;

        const N: usize = 50_000;
        let left = bench_make_asof_table(N, 0.0);
        let right = bench_make_asof_table(N, -0.5);
        let args = [
            left,
            right,
            Value::String("time".to_string()),
            Value::String("group".to_string()),
            Value::String("backward".to_string()),
        ];

        let start = Instant::now();
        let joined = native_asof_join(&args);
        let elapsed = start.elapsed();
        let rows = bench_assert_table(&joined);
        assert_eq!(rows, N);
        println!(
            "bench_asof_join_large: {} x {}, by=group, direction=backward -> {} rows in {:?}",
            N, N, rows, elapsed
        );
    }

    /// Non-equi nested-loop join (`id == id`). Smaller N — O(n²).
    #[test]
    fn bench_join_on_equi_medium() {
        use data_code::vm::natives::native_join_on;
        use data_code::Value;

        const N: usize = 2_000;
        let left = bench_make_table(N, 0.0);
        let right = bench_make_table(N, 0.0);
        let args = [
            left,
            right,
            Value::String("id == id".to_string()),
            Value::String("inner".to_string()),
        ];

        let start = Instant::now();
        let joined = native_join_on(&args);
        let elapsed = start.elapsed();
        let rows = bench_assert_table(&joined);
        assert_eq!(rows, N);
        println!(
            "bench_join_on_equi_medium: {} x {}, id == id -> {} rows in {:?}",
            N, N, rows, elapsed
        );
    }

    /// Lateral apply: one output row per left row (VM + user function).
    #[test]
    fn bench_apply_join_medium() {
        const N: u32 = 5_000;
        let source = format!(
            r#"
fn one_row(row) {{
    return table([[row[0]]], ["out"])
}}
let rows = []
for i in range({N}) {{
    push(rows, [i, "x"])
}}
let left = table(rows, ["id", "tag"])
len(apply_join(left, one_row))
"#
        );

        let start = Instant::now();
        let result = run(&source).expect("bench_apply_join_medium");
        let elapsed = start.elapsed();
        match result {
            data_code::Value::Number(n) => {
                assert_eq!(n as u32, N, "apply_join row count");
                println!(
                    "bench_apply_join_medium: {} left rows -> {} rows in {:?}",
                    N, n, elapsed
                );
            }
            v => panic!("expected row count, got {:?}", v),
        }
    }

    /// Inner join then rename overlapping columns via suffixes.
    #[test]
    fn bench_inner_join_then_suffixes_large() {
        use data_code::vm::natives::{native_inner_join, native_table_suffixes};
        use data_code::Value;

        const N: usize = 50_000;
        let args = bench_equi_join_pair(N);

        let join_start = Instant::now();
        let joined = native_inner_join(&args);
        let join_elapsed = join_start.elapsed();
        let joined_rows = bench_assert_table(&joined);
        assert_eq!(joined_rows, N);

        let suffix_args = [
            joined,
            Value::String("_l".to_string()),
            Value::String("_r".to_string()),
        ];
        let suffix_start = Instant::now();
        let suffixed = native_table_suffixes(&suffix_args);
        let suffix_elapsed = suffix_start.elapsed();
        let _ = bench_assert_table(&suffixed);

        println!(
            "bench_inner_join_then_suffixes_large: join {} rows in {:?}, suffixes in {:?}, total {:?}",
            joined_rows,
            join_elapsed,
            suffix_elapsed,
            join_elapsed + suffix_elapsed
        );
    }

    /// Same-schema vertical concat: 3 tables × 50k rows.
    #[test]
    fn bench_merge_tables_large() {
        use data_code::vm::natives::native_merge_tables;
        use data_code::Value;
        use std::cell::RefCell;
        use std::rc::Rc;

        const N: usize = 50_000;
        let t1 = bench_make_table(N, 0.0);
        let t2 = bench_make_table(N, N as f64);
        let t3 = bench_make_table(N, (2 * N) as f64);
        let args = [Value::Array(Rc::new(RefCell::new(vec![t1, t2, t3])))];

        let start = Instant::now();
        let merged = native_merge_tables(&args);
        let elapsed = start.elapsed();
        let rows = bench_table_len(&merged);
        assert_eq!(rows, N * 3, "expected {} merged rows", N * 3);
        println!(
            "bench_merge_tables_large: {} rows x 3 tables -> {} rows in {:?}",
            N, rows, elapsed
        );
    }

    /// Outer merge with almost-identical schemas (id,val,tag vs id,val,extra).
    #[test]
    fn bench_merge_tables_outer_mismatch() {
        use data_code::common::table::Table;
        use data_code::vm::natives::native_merge_tables;
        use data_code::Value;
        use std::cell::RefCell;
        use std::rc::Rc;

        const N: usize = 50_000;
        let t1 = bench_make_table(N, 0.0);
        let mut rows2 = Vec::with_capacity(N);
        for i in 0..N {
            let id = N as f64 + i as f64;
            rows2.push(vec![
                Value::Number(id),
                Value::Number(id * 2.0),
                Value::String("y".to_string()),
            ]);
        }
        let t2 = Value::Table(Rc::new(RefCell::new(Table::from_data(
            rows2,
            Some(vec!["id".into(), "val".into(), "extra".into()]),
        ))));
        let args = [
            Value::Array(Rc::new(RefCell::new(vec![t1, t2]))),
            Value::String("outer".to_string()),
        ];

        let start = Instant::now();
        let merged = native_merge_tables(&args);
        let elapsed = start.elapsed();
        let rows = bench_table_len(&merged);
        assert_eq!(rows, N * 2, "expected {} merged rows", N * 2);
        println!(
            "bench_merge_tables_outer_mismatch: {}+{} rows -> {} rows in {:?}",
            N, N, rows, elapsed
        );
    }

    /// Equi inner join on `id`: 50k × 50k, 1:1 matches.
    #[test]
    fn bench_inner_join_large() {
        use data_code::vm::natives::native_inner_join;
        use data_code::Value;

        const N: usize = 50_000;
        let left = bench_make_table(N, 0.0);
        let right = bench_make_table(N, 0.0);
        let args = [
            left,
            right,
            Value::String("id".to_string()),
            Value::String("id".to_string()),
        ];

        let start = Instant::now();
        let joined = native_inner_join(&args);
        let elapsed = start.elapsed();
        let rows = bench_table_len(&joined);
        assert_eq!(rows, N, "expected {} joined rows", N);
        println!(
            "bench_inner_join_large: {} x {} -> {} rows in {:?}",
            N, N, rows, elapsed
        );
    }

    // --- Table ops (filter / slice / transform) benchmarks ---
    // Run: cargo test --release profiling_benchmarks::tests::bench_table_ -- --nocapture

    const TABLE_OPS_N: usize = 50_000;

    fn bench_make_table_with_nulls(n: usize) -> data_code::Value {
        use data_code::common::table::Table;
        use data_code::Value;
        use std::cell::RefCell;
        use std::rc::Rc;

        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            let id = i as f64;
            let val = if i % 10 == 0 {
                Value::Null
            } else {
                Value::Number(id * 2.0)
            };
            rows.push(vec![Value::Number(id), val, Value::String("x".to_string())]);
        }
        Value::Table(Rc::new(RefCell::new(Table::from_data(
            rows,
            Some(vec!["id".into(), "val".into(), "tag".into()]),
        ))))
    }

    macro_rules! bench_table_op {
        ($name:ident, $setup:expr, $body:expr, $label:expr) => {
            #[test]
            fn $name() {
                const N: usize = TABLE_OPS_N;
                let setup = $setup;
                let start = Instant::now();
                let result = $body;
                let elapsed = start.elapsed();
                let rows = bench_assert_table(&result);
                println!("{}: {} rows in {:?} ({})", stringify!($name), rows, elapsed, $label);
                let _ = (N, setup);
            }
        };
    }

    bench_table_op!(
        bench_table_head_large,
        bench_make_table(TABLE_OPS_N, 0.0),
        {
            use data_code::vm::natives::native_table_head;
            use data_code::Value;
            let table = bench_make_table(TABLE_OPS_N, 0.0);
            native_table_head(&[table, Value::Number(5_000.0)])
        },
        "head 5000 of 50k"
    );

    bench_table_op!(
        bench_table_tail_large,
        (),
        {
            use data_code::vm::natives::native_table_tail;
            use data_code::Value;
            let table = bench_make_table(TABLE_OPS_N, 0.0);
            native_table_tail(&[table, Value::Number(5_000.0)])
        },
        "tail 5000 of 50k"
    );

    bench_table_op!(
        bench_table_select_large,
        (),
        {
            use data_code::vm::natives::native_table_select;
            use data_code::Value;
            use std::cell::RefCell;
            use std::rc::Rc;
            let table = bench_make_table(TABLE_OPS_N, 0.0);
            let cols = Value::Array(Rc::new(RefCell::new(vec![
                Value::String("id".into()),
                Value::String("val".into()),
            ])));
            native_table_select(&[table, cols])
        },
        "select 2 of 3 cols"
    );

    bench_table_op!(
        bench_table_where_large,
        (),
        {
            use data_code::vm::natives::native_table_where;
            use data_code::Value;
            let table = bench_make_table(TABLE_OPS_N, 0.0);
            native_table_where(&[
                table,
                Value::String("id".into()),
                Value::String(">".into()),
                Value::Number(25_000.0),
            ])
        },
        "where id > 25000"
    );

    bench_table_op!(
        bench_table_sort_large,
        (),
        {
            use data_code::vm::natives::native_table_sort;
            use data_code::Value;
            let table = bench_make_table(TABLE_OPS_N, 0.0);
            native_table_sort(&[table, Value::String("id".into()), Value::Bool(true)])
        },
        "sort by id asc"
    );

    bench_table_op!(
        bench_table_drop_nulls_large,
        (),
        {
            use data_code::vm::natives::native_table_drop_nulls;
            let table = bench_make_table_with_nulls(TABLE_OPS_N);
            native_table_drop_nulls(&[table])
        },
        "drop_nulls all columns"
    );

    bench_table_op!(
        bench_table_distinct_large,
        (),
        {
            use data_code::vm::natives::native_table_distinct;
            let table = bench_make_table(TABLE_OPS_N, 0.0);
            native_table_distinct(&[table])
        },
        "distinct all columns"
    );

    bench_table_op!(
        bench_table_rename_large,
        (),
        {
            use data_code::vm::natives::native_table_rename;
            use data_code::Value;
            let table = bench_make_table(TABLE_OPS_N, 0.0);
            native_table_rename(&[
                table,
                Value::String("id".into()),
                Value::String("identifier".into()),
            ])
        },
        "rename single column"
    );

    bench_table_op!(
        bench_table_drop_column_large,
        (),
        {
            use data_code::vm::natives::native_table_drop_column;
            use data_code::Value;
            let table = bench_make_table(TABLE_OPS_N, 0.0);
            native_table_drop_column(&[table, Value::String("tag".into())])
        },
        "drop one column"
    );

    bench_table_op!(
        bench_table_row_number_large,
        (),
        {
            use data_code::vm::natives::native_table_row_number;
            let table = bench_make_table(TABLE_OPS_N, 0.0);
            native_table_row_number(&[table])
        },
        "row_number"
    );

    bench_table_op!(
        bench_table_add_column_large,
        (),
        {
            use data_code::vm::natives::native_table_add_column;
            use data_code::Value;
            let table = bench_make_table(TABLE_OPS_N, 0.0);
            native_table_add_column(&[
                table,
                Value::String("idx".into()),
                Value::Number(0.0),
            ])
        },
        "add scalar column"
    );

    /// View table from arrays → merge → push chain (script path).
    #[test]
    fn bench_view_table_merge_push_large() {
        const N: u32 = 10_000;
        let source = format!(
            r#"
let n = {n}
let t = table([range(n), range(n)], ["id", "val"])
let f = table_where(t, "id", ">", n / 2)
let s = table_select(f, ["id", "val"])
let _ = s.push([n, n])
len(s)
"#,
            n = N
        );
        let start = Instant::now();
        let r = run(&source);
        let elapsed = start.elapsed();
        assert!(r.is_ok(), "run failed: {:?}", r);
        if let Ok(data_code::Value::Number(v)) = r {
            // range(n) is 0..n-1; id > n/2 yields (n/2 - 1) rows, +1 push
            let expected = (N / 2) as u32;
            assert_eq!(v as u32, expected, "expected {} rows", expected);
        }
        println!(
            "bench_view_table_merge_push_large: view where→select→push in {:?}",
            elapsed
        );
    }

    /// aggregate_group via native (count by id).
    #[test]
    fn bench_table_aggregate_group_large() {
        use data_code::vm::natives::native_table_aggregate_group;
        use data_code::Value;
        use std::collections::HashMap;

        const N: usize = TABLE_OPS_N;
        let table = bench_make_table(N, 0.0);
        let mut spec = HashMap::new();
        spec.insert("group".to_string(), Value::String("id".into()));
        spec.insert("cnt".to_string(), Value::String("count".into()));
        let spec_val = Value::legacy_object(spec);

        let start = Instant::now();
        let result = native_table_aggregate_group(&[table, spec_val]);
        let elapsed = start.elapsed();
        let rows = bench_assert_table(&result);
        assert_eq!(rows, N, "expected one row per unique id");
        println!(
            "bench_table_aggregate_group_large: {} groups in {:?}",
            rows, elapsed
        );
    }

    // --- Wave 2: map / transform / aggregate / I/O / relations ---

    fn bench_make_table_splittable(n: usize) -> data_code::Value {
        use data_code::common::table::Table;
        use data_code::Value;
        use std::cell::RefCell;
        use std::rc::Rc;

        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            let id = i as f64;
            rows.push(vec![
                Value::Number(id),
                Value::Number(id * 2.0),
                Value::String(format!("a|{}", i % 100)),
            ]);
        }
        Value::Table(Rc::new(RefCell::new(Table::from_data(
            rows,
            Some(vec!["id".into(), "val".into(), "tag".into()]),
        ))))
    }

    fn bench_write_csv_fixture(n: usize) -> std::path::PathBuf {
        use data_code::common::table_csv_export::write_table_csv;
        let table_val = bench_make_table(n, 0.0);
        let data_code::Value::Table(t) = table_val else {
            panic!("expected table");
        };
        let path = std::env::temp_dir().join(format!("dc_bench_read_{}_{}.csv", n, std::process::id()));
        write_table_csv(&t.borrow(), &path).expect("write fixture csv");
        path
    }

    bench_table_op!(
        bench_table_value_map_large,
        (),
        {
            use data_code::vm::natives::native_table_value_map;
            use data_code::Value;
            use std::collections::HashMap;
            let table = bench_make_table(TABLE_OPS_N, 0.0);
            let mut mappings = HashMap::new();
            mappings.insert("x".to_string(), Value::String("X".into()));
            let spec = Value::legacy_object(mappings);
            native_table_value_map(&[table, Value::String("tag".into()), spec])
        },
        "value_map tag x->X"
    );

    #[test]
    fn bench_table_map_abs_large() {
        use data_code::run;
        use data_code::Value;

        let source = format!(
            r#"
let n = {n}
let ids = range(n)
let vals = range(n)
let t = table([ids, vals], ["id", "val"])
let m = table_map(t, "val", abs)
len(m)
"#,
            n = TABLE_OPS_N
        );
        let start = Instant::now();
        let r = run(&source);
        let elapsed = start.elapsed();
        assert!(r.is_ok(), "run failed: {:?}", r);
        if let Ok(Value::Number(v)) = r {
            assert_eq!(v as usize, TABLE_OPS_N);
        }
        println!(
            "bench_table_map_abs_large: {} rows in {:?} (map abs(val))",
            TABLE_OPS_N,
            elapsed
        );
    }

    bench_table_op!(
        bench_table_split_column_delim_large,
        (),
        {
            use data_code::vm::natives::native_table_split_column;
            use data_code::Value;
            use std::cell::RefCell;
            use std::rc::Rc;
            let table = bench_make_table_splittable(TABLE_OPS_N);
            let new_cols = Value::Array(Rc::new(RefCell::new(vec![
                Value::String("part_a".into()),
                Value::String("part_b".into()),
            ])));
            native_table_split_column(&[
                table,
                Value::String("tag".into()),
                Value::String("|".into()),
                new_cols,
            ])
        },
        "split_column tag by |"
    );

    bench_table_op!(
        bench_table_join_columns_large,
        (),
        {
            use data_code::vm::natives::native_table_join_columns;
            use data_code::Value;
            use std::cell::RefCell;
            use std::rc::Rc;
            let table = bench_make_table(TABLE_OPS_N, 0.0);
            native_table_join_columns(&[
                table,
                Value::Array(Rc::new(RefCell::new(vec![
                    Value::String("id".into()),
                    Value::String("val".into()),
                ]))),
                Value::String("joined".into()),
                Value::String(",".into()),
            ])
        },
        "join_columns id+val"
    );

    bench_table_op!(
        bench_table_aggregate_large,
        (),
        {
            use data_code::vm::natives::native_table_aggregate;
            use data_code::Value;
            use std::collections::HashMap;
            let table = bench_make_table(TABLE_OPS_N, 0.0);
            let mut spec = HashMap::new();
            let mut sum_spec = HashMap::new();
            sum_spec.insert("op".to_string(), Value::String("sum".into()));
            sum_spec.insert("column".to_string(), Value::String("val".into()));
            spec.insert("total".to_string(), Value::legacy_object(sum_spec));
            let spec_val = Value::legacy_object(spec);
            native_table_aggregate(&[table, spec_val])
        },
        "aggregate sum(val)"
    );

    #[test]
    fn bench_table_read_csv_large() {
        use data_code::file_io::read_value;
        use data_code::Value;
        use std::fs;

        const N: usize = TABLE_OPS_N;
        let path = bench_write_csv_fixture(N);
        let args = [Value::String(path.to_string_lossy().to_string())];
        let start = Instant::now();
        let result = read_value(&args);
        let elapsed = start.elapsed();
        let _ = fs::remove_file(&path);
        let rows = match result {
            Ok(Value::Table(t)) => t.borrow().len(),
            other => panic!("expected table from read, got {:?}", other),
        };
        assert_eq!(rows, N);
        println!(
            "bench_table_read_csv_large: {} rows in {:?}",
            rows, elapsed
        );
    }

    #[test]
    fn bench_table_save_csv_large() {
        use data_code::vm::natives::table_save::native_table_save_csv;
        use data_code::Value;
        use std::fs;

        const N: usize = TABLE_OPS_N;
        let table = bench_make_table(N, 0.0);
        let path = std::env::temp_dir().join(format!(
            "dc_bench_save_{}_{}.csv",
            N,
            std::process::id()
        ));
        let start = Instant::now();
        let result = native_table_save_csv(&[table, Value::String(path.to_string_lossy().to_string())]);
        let elapsed = start.elapsed();
        let _ = fs::remove_file(&path);
        assert!(matches!(result, Value::String(_)), "save failed: {:?}", result);
        println!(
            "bench_table_save_csv_large: {} rows in {:?}",
            N, elapsed
        );
    }

    #[test]
    fn bench_relate_primary_key_large() {
        use data_code::vm::natives::{native_primary_key, native_relate};
        use data_code::Value;
        use std::rc::Rc;

        const N: usize = TABLE_OPS_N;
        let left = bench_make_table(N, 0.0);
        let right = bench_make_table(N, 0.0);
        let pk_col = Value::ColumnReference {
            table: match &left {
                data_code::Value::Table(t) => Rc::clone(t),
                _ => panic!("table"),
            },
            column_name: "id".into(),
        };
        let fk_col = Value::ColumnReference {
            table: match &right {
                data_code::Value::Table(t) => Rc::clone(t),
                _ => panic!("table"),
            },
            column_name: "id".into(),
        };

        let start = Instant::now();
        for _ in 0..100 {
            let _ = native_primary_key(&[pk_col.clone()]);
            let _ = native_relate(&[pk_col.clone(), fk_col.clone()]);
        }
        let elapsed = start.elapsed();
        println!(
            "bench_relate_primary_key_large: 100x relate+pk on {} rows in {:?}",
            N, elapsed
        );
    }

    // --- Method-call receiver temp slot strategy benchmarks ---
    // Run: cargo test --release profiling_benchmarks::tests::bench_method_object -- --nocapture
    // Compare strategies:
    //   default (depth stack)
    //   --features method_object_per_call
    //   --features method_object_shared (incorrect for nested calls)

    const METHOD_OBJECT_N: u32 = 1_000_000;
    const METHOD_OBJECT_LOOP: u32 = 100_000;
    const METHOD_OBJECT_NESTED: u32 = 100_000;

    fn max_local_index_in_chunk(chunk: &data_code::Chunk) -> usize {
        use data_code::bytecode::OpCode;
        let mut max_idx = 0usize;
        for op in &chunk.code {
            match op {
                OpCode::LoadLocal(i) | OpCode::StoreLocal(i) => {
                    max_idx = max_idx.max(*i);
                }
                _ => {}
            }
        }
        max_idx + 1
    }

    fn max_locals_in_compiled(source: &str) -> usize {
        let (_main, functions) = data_code::compile(source).expect("compile");
        functions
            .iter()
            .map(|f| max_local_index_in_chunk(&f.chunk))
            .max()
            .unwrap_or(0)
    }

    fn script_method_object_nested_loop(n: u32) -> String {
        format!(
            r#"
cls Ring {{
    new Ring(cap: int) {{
        this.buf = []
        for _ in range(cap) {{ this.buf.push(null) }}
        this.head = 0
        this.count = 0
    }}
    fn _capacity() -> int {{ return len(this.buf) }}
    fn to_list() -> list {{
        result = []
        for i in range(this.count) {{
            result.push(this.buf[(this.head + i) % this._capacity()])
        }}
        return result
    }}
    fn fill(n: int) {{
        for i in range(n) {{
            this.buf[this.head] = i
            this.head = (this.head + 1) % this._capacity()
            this.count = this.count + 1
        }}
    }}
}}
r = Ring({n})
r.fill({n})
len(r.to_list())
"#,
            n = n
        )
    }

    fn script_method_object_flat_loop(n: u32) -> String {
        format!(
            r#"
cls Acc {{
    new Acc() {{ this.buf = [] }}
    fn push_many(n: int) {{
        for i in range(n) {{ this.buf.push(i) }}
    }}
    fn len() {{ return len(this.buf) }}
}}
a = Acc()
a.push_many({n})
a.len()
"#,
            n = n
        )
    }

    fn script_method_object_many_sites(sites: usize) -> String {
        let mut body = String::from("fn many_pushes() {\n    a = []\n");
        for i in 0..sites {
            body.push_str(&format!("    a.push({})\n", i));
        }
        body.push_str("    return len(a)\n}\nmany_pushes()\n");
        body
    }

    #[test]
    #[ignore]
    fn bench_method_object_nested_1m() {
        let n = METHOD_OBJECT_NESTED;
        let source = script_method_object_nested_loop(n);
        let locals = max_locals_in_compiled(&source);
        let start = Instant::now();
        let r = run(&source);
        let elapsed = start.elapsed();
        assert!(r.is_ok(), "run failed: {:?}", r);
        if let Ok(data_code::Value::Number(v)) = r {
            assert_eq!(v, n as f64);
        }
        println!(
            "bench_method_object_nested_1m: {} rows (nested _capacity in index), max_locals={}, {:?}",
            n, locals, elapsed
        );
    }

    #[test]
    #[ignore]
    fn bench_method_object_flat_1m() {
        let n = METHOD_OBJECT_LOOP;
        let source = script_method_object_flat_loop(n);
        let locals = max_locals_in_compiled(&source);
        let start = Instant::now();
        let r = run(&source);
        let elapsed = start.elapsed();
        assert!(r.is_ok(), "run failed: {:?}", r);
        if let Ok(data_code::Value::Number(v)) = r {
            assert_eq!(v, n as f64);
        }
        println!(
            "bench_method_object_flat_1m: {} pushes, max_locals={}, {:?}",
            n, locals, elapsed
        );
    }

    /// Full 1M push loop — slow (~30–40 min release). Run manually for large-scale timing.
    #[test]
    #[ignore]
    fn bench_method_object_flat_1m_full() {
        let n = METHOD_OBJECT_N;
        let source = script_method_object_flat_loop(n);
        let start = Instant::now();
        let r = run(&source);
        let elapsed = start.elapsed();
        assert!(r.is_ok(), "run failed: {:?}", r);
        println!(
            "bench_method_object_flat_1m_full: {} pushes in {:?}",
            n, elapsed
        );
    }

    #[test]
    fn bench_method_object_many_sites_locals() {
        const SITES: usize = 2_000;
        let source = script_method_object_many_sites(SITES);
        let locals = max_locals_in_compiled(&source);
        let start = Instant::now();
        let r = run(&source);
        let elapsed = start.elapsed();
        assert!(r.is_ok(), "run failed: {:?}", r);
        if let Ok(data_code::Value::Number(v)) = r {
            assert_eq!(v, SITES as f64);
        }
        println!(
            "bench_method_object_many_sites_locals: {} call sites, max_locals={}, {:?}",
            SITES, locals, elapsed
        );
    }

    #[test]
    fn bench_method_object_table_where_1m() {
        use data_code::vm::natives::native_table_where;
        use data_code::Value;
        const N: usize = 1_000_000;
        let mut rows = Vec::with_capacity(N);
        for i in 0..N {
            rows.push(vec![Value::Number(i as f64), Value::Number((i % 2) as f64)]);
        }
        use data_code::common::table::Table;
        use std::cell::RefCell;
        use std::rc::Rc;
        let table = Value::Table(Rc::new(RefCell::new(Table::from_data(
            rows,
            Some(vec!["id".into(), "parity".into()]),
        ))));
        let pred = Value::String("parity == 0".into());
        let start = Instant::now();
        let out = native_table_where(&[table, pred]);
        let elapsed = start.elapsed();
        let kept = bench_table_len(&out);
        println!(
            "bench_method_object_table_where_1m: {} rows -> {} kept in {:?} (native path, slots N/A)",
            N, kept, elapsed
        );
    }
}

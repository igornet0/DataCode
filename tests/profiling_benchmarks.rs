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
        println!("bench_native_calls_heavy: {} iterations in {:?}", n, elapsed);
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
        println!("bench_table_ops_heavy: {} table creations in {:?}", rows, elapsed);
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

        let run_many_threads = |script: Arc<String>, k: u32, num_threads: usize| -> std::time::Duration {
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

        println!("--- GIL experiment: {} threads, {} iterations per thread ---", NUM_THREADS, K_ITERATIONS);

        // Mode A: arithmetic
        let script_a = script_mode_a(N_MODE_A);
        let t1_a = run_one_thread(&script_a, K_ITERATIONS);
        let t8_a = run_many_threads(Arc::new(script_a.clone()), K_ITERATIONS, NUM_THREADS);
        let ratio_a = t8_a.as_secs_f64() / t1_a.as_secs_f64();
        println!("Mode A (arithmetic):  T1 = {:?}, T8 = {:?}, T8/T1 = {:.2}", t1_a, t8_a, ratio_a);
        assert!(ratio_a < 3.0, "Mode A ratio {:.2} suggests serialization; expect < 3 (2-3 = partial contention, 8 = GIL)", ratio_a);

        // Mode B: objects
        let script_b = script_mode_b(N_MODE_B);
        let t1_b = run_one_thread(&script_b, K_ITERATIONS);
        let t8_b = run_many_threads(Arc::new(script_b.clone()), K_ITERATIONS, NUM_THREADS);
        let ratio_b = t8_b.as_secs_f64() / t1_b.as_secs_f64();
        println!("Mode B (objects):    T1 = {:?}, T8 = {:?}, T8/T1 = {:.2}", t1_b, t8_b, ratio_b);
        assert!(ratio_b < 3.0, "Mode B ratio {:.2} suggests serialization; expect < 3 (2-3 = partial contention, 8 = GIL)", ratio_b);

        // Mode C: class methods
        let script_c = script_mode_c(N_MODE_C);
        let t1_c = run_one_thread(&script_c, K_ITERATIONS);
        let t8_c = run_many_threads(Arc::new(script_c.clone()), K_ITERATIONS, NUM_THREADS);
        let ratio_c = t8_c.as_secs_f64() / t1_c.as_secs_f64();
        println!("Mode C (class):      T1 = {:?}, T8 = {:?}, T8/T1 = {:.2}", t1_c, t8_c, ratio_c);
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

        let run_many_threads = |script: Arc<String>, k: u32, num_threads: usize| -> std::time::Duration {
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

        println!("--- GIL experiment: {} threads, {} iterations per thread ---", NUM_THREADS, K_ITERATIONS);

        let script_a = script_mode_a(N_MODE_A);
        let t1_a = run_one_thread(&script_a, K_ITERATIONS);
        let t16_a = run_many_threads(Arc::new(script_a.clone()), K_ITERATIONS, NUM_THREADS);
        let ratio_a = t16_a.as_secs_f64() / t1_a.as_secs_f64();
        println!("Mode A (arithmetic):  T1 = {:?}, T16 = {:?}, T16/T1 = {:.2}", t1_a, t16_a, ratio_a);
        assert!(ratio_a < 5.0, "Mode A (16 threads) ratio {:.2} suggests serialization; expect < 5 (allocator contention ok)", ratio_a);

        let script_b = script_mode_b(N_MODE_B);
        let t1_b = run_one_thread(&script_b, K_ITERATIONS);
        let t16_b = run_many_threads(Arc::new(script_b.clone()), K_ITERATIONS, NUM_THREADS);
        let ratio_b = t16_b.as_secs_f64() / t1_b.as_secs_f64();
        println!("Mode B (objects):    T1 = {:?}, T16 = {:?}, T16/T1 = {:.2}", t1_b, t16_b, ratio_b);
        assert!(ratio_b < 5.0, "Mode B (16 threads) ratio {:.2} suggests serialization; expect < 5", ratio_b);

        let script_c = script_mode_c(N_MODE_C);
        let t1_c = run_one_thread(&script_c, K_ITERATIONS);
        let t16_c = run_many_threads(Arc::new(script_c.clone()), K_ITERATIONS, NUM_THREADS);
        let ratio_c = t16_c.as_secs_f64() / t1_c.as_secs_f64();
        println!("Mode C (class):      T1 = {:?}, T16 = {:?}, T16/T1 = {:.2}", t1_c, t16_c, ratio_c);
        assert!(ratio_c < 5.0, "Mode C (16 threads) ratio {:.2} suggests serialization; expect < 5", ratio_c);

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

        let run_many_threads = |script: Arc<String>, k: u32, num_threads: usize| -> std::time::Duration {
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
            println!("ALLOC_BENCH Mode={} T1={:.3} T8={:.3} T16={:.3} T8/T1={:.3} T16/T1={:.3}", name, s1, s8, s16, r8, r16);
        }

        println!("--- Allocator comparison: 1 / 8 / 16 threads, {} iterations per thread ---", K_ITERATIONS);

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

        println!("--- Peak memory: run with /usr/bin/time -v or heaptrack and fill table manually ---");
    }
}

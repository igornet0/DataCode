//! Regression: `push` on a growing array must stay ~linear (not O(n²) via full load/store per call).
//! See `tests/performance_tests/LARGE_DATASET_10K_LOG.md` and `docs/vm_mutating_natives_audit.md`.

use data_code::{run, Value};
use std::time::Instant;

fn run_push_only_loop(n: u32) -> std::time::Duration {
    let src = format!(
        "let a = array_with_capacity({n})\nfor i in range({n}) {{ push(a, i) }}\nlen(a)",
        n = n
    );
    let t0 = Instant::now();
    let v = run(&src).expect("run");
    let elapsed = t0.elapsed();
    match v {
        Value::Number(x) => assert!((x - n as f64).abs() < 0.01, "expected len {} got {}", n, x),
        other => panic!("expected Number, got {:?}", other),
    }
    elapsed
}

/// If push regressed to O(n²), t4000/t2000 would approach 4×; linear growth stays near 2× (margin for CI noise).
#[test]
fn push_grow_loop_scaling_not_quadratic() {
    let t2000 = run_push_only_loop(2000);
    let t4000 = run_push_only_loop(4000);
    let r = if t2000.as_secs_f64() > 1e-9 {
        t4000.as_secs_f64() / t2000.as_secs_f64()
    } else {
        1.0
    };
    assert!(
        r < 3.6,
        "push loop time ratio t4000/t2000 = {:.2} (t2000={:?} t4000={:?}); expected < ~2.5 for linear, ~4 indicates n²",
        r,
        t2000,
        t4000
    );
}

/// Larger stress; ignored on purpose for slow debug builds / optional local profiling.
#[test]
#[ignore = "Heavy push stress (~15k iterations); run: cargo test push_grow_loop_15k_stress -- --ignored --release --nocapture"]
fn push_grow_loop_15k_stress() {
    let t0 = Instant::now();
    let _ = run_push_only_loop(15_000);
    let elapsed = t0.elapsed();
    assert!(
        elapsed.as_secs() < 120,
        "15k push loop took {:?} (expected < 120s even in debug)",
        elapsed
    );
}

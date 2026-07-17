//! Grid buffer module and native A* tests.

use data_code::{run, run_with_vm, Value};
use std::time::{Duration, Instant};

#[test]
fn grid_alloc_and_bitmap() {
    let source = r#"
import grid
n = 100
blocked = grid.alloc_u8(grid.bitmap_bytes(n), 0)
grid.set_blocked(blocked, 5)
grid.set_blocked(blocked, 99)
grid.test_blocked(blocked, 5) and grid.test_blocked(blocked, 99) and !grid.test_blocked(blocked, 0)
"#;
    let v = run(source).expect("run");
    assert_eq!(v, Value::Bool(true));
}

#[test]
fn grid_astar_small() {
    let source = r#"
import grid
rows, cols = 10, 10
n = rows * cols
blocked = grid.alloc_u8(grid.bitmap_bytes(n), 0)
grid.set_blocked(blocked, 55)
g = grid.alloc_i32(n, 2000000000)
f = grid.alloc_i32(n, 0)
p = grid.alloc_i32(n, -1)
c = grid.alloc_u8(n, 0)
path = grid.astar(rows, cols, (0, 0), (9, 9), blocked, g, f, p, c)
len(path)
"#;
    let v = run(source).expect("run");
    assert!(matches!(v, Value::Number(n) if n >= 18.0));
}

#[test]
fn grid_astar_from_set_matches_astar_grid() {
    let source = r#"
import grid
import pathfind
rows, cols = 20, 20
blocked = set()
blocked.add((5, 5))
p1 = grid.astar_from_set(rows, cols, (0, 0), (10, 10), blocked)
p2 = pathfind.astar_grid(rows, cols, (0, 0), (10, 10), blocked)
len(p1) == len(p2)
"#;
    let v = run(source).expect("run");
    assert_eq!(v, Value::Bool(true));
}

/// CI gate: 100×100 native grid.astar, bounded store and wall time (release: <0.5s).
#[test]
fn bench_astar_100x100_ci() {
    let source = r#"
import grid
rows, cols = 100, 100
n = rows * cols
blocked = grid.alloc_u8(grid.bitmap_bytes(n), 0)
g = grid.alloc_i32(n, 2000000000)
f = grid.alloc_i32(n, 0)
p = grid.alloc_i32(n, -1)
c = grid.alloc_u8(n, 0)
path = grid.astar(rows, cols, (0, 0), (99, 99), blocked, g, f, p, c)
len(path)
"#;
    let start = Instant::now();
    let (v, vm) = run_with_vm(source).expect("run");
    let elapsed = start.elapsed();
    println!(
        "bench_astar_100x100_ci: path_len={:?}, store_len={}, elapsed={:?}",
        v,
        vm.value_store_len(),
        elapsed
    );
    assert!(matches!(v, Value::Number(n) if n >= 100.0));
    assert!(
        vm.value_store_len() < 50_000,
        "store_len={}",
        vm.value_store_len()
    );
    #[cfg(not(debug_assertions))]
    assert!(
        elapsed < Duration::from_millis(500),
        "too slow in release: {:?}",
        elapsed
    );
    #[cfg(debug_assertions)]
    assert!(
        elapsed < Duration::from_secs(2),
        "too slow in debug: {:?}",
        elapsed
    );
}

#[test]
fn bench_astar_vm_store_len_per_run() {
    let source = r#"
import grid
rows, cols = 80, 80
n = rows * cols
blocked = grid.alloc_u8(grid.bitmap_bytes(n), 0)
g = grid.alloc_i32(n, 2000000000)
f = grid.alloc_i32(n, 0)
p = grid.alloc_i32(n, -1)
c = grid.alloc_u8(n, 0)
lens = []
for _ in range(10) {
    path = grid.astar(rows, cols, (0, 0), (79, 79), blocked, g, f, p, c)
    lens.push(len(path))
    lens.push(grid.store_len())
}
lens
"#;
    let (v, vm) = run_with_vm(source).expect("run");
    let arr = match v {
        Value::Array(a) => a.borrow().clone(),
        other => panic!("expected array, got {:?}", other),
    };
    assert_eq!(arr.len(), 20);
    for i in (1..arr.len()).step_by(2) {
        if let Value::Number(n) = arr[i] {
            assert!(
                n < 50_000.0,
                "store_len grew on run {}: {}",
                i / 2,
                n
            );
        }
    }
    println!(
        "bench_astar_vm_store_len_per_run: final_store_len={}",
        vm.value_store_len()
    );
}

//! Multi-run memory / time regression tests for A* workloads.
//!
//! Run: `cargo test --test memory_regression_tests -q`
//! Release (ignored large): `cargo test --test memory_regression_tests --release -- --ignored --nocapture`

use data_code::{run, run_with_vm, Value};
use std::time::Instant;

const MINI_ASTAR: &str = r#"
import heapq
from system import time
fn a_star(rows, cols, start, goal, blocked_ids,
    g_score, f_score, came_from, closed_set, open_heap, open_set) {
    g_score.clear()
    f_score.clear()
    came_from.clear()
    closed_set.clear()
    open_set.clear()
    heapq.heap_clear(open_heap)
    start_id = start[0] * cols + start[1]
    goal_id = goal[0] * cols + goal[1]
    goal_r, goal_c = goal
    if start_id in blocked_ids or goal_id in blocked_ids: return null
    g_score[start_id] = 0
    f_score[start_id] = abs(start[0] - goal_r) + abs(start[1] - goal_c)
    open_heap = [(f_score[start_id], start_id)]
    open_set.add(start_id)
    neighbors_delta = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while open_heap {
        current_f, current = heapq.heappop(open_heap)
        if current_f != f_score.get(current): continue
        open_set.discard(current)
        if current in closed_set: continue
        if current == goal_id { return 1 }
        closed_set.add(current)
        r, c = divmod(current, cols)
        for dr, dc in neighbors_delta {
            nr, nc = r + dr, c + dc
            if !(0 <= nr < rows and 0 <= nc < cols): continue
            neighbor = nr * cols + nc
            if neighbor in blocked_ids or neighbor in closed_set: continue
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, float(inf)) {
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + abs(nr - goal_r) + abs(nc - goal_c)
                f_score[neighbor] = f
                if !neighbor in open_set {
                    heapq.heappush(open_heap, (f, neighbor))
                    open_set.add(neighbor)
                }
            }
        }
    }
    return null
}
fn bench_runs(n) {
    g_score = {}
    f_score = {}
    came_from = {}
    closed_set = set()
    open_heap = []
    open_set = set()
    blocked_ids = {}
    t0 = 0.0
    t_last = 0.0
    for i in range(n) {
        start = time.perf_counter()
        a_star(50, 50, (0, 0), (10, 10), blocked_ids,
            g_score, f_score, came_from, closed_set, open_heap, open_set)
        elapsed = time.perf_counter() - start
        if i == 0 { t0 = elapsed }
        if i == n - 1 { t_last = elapsed }
    }
    if t0 <= 0.0 { return -1.0 }
    return t_last / t0
}
bench_runs(10)
"#;

fn assert_number(source: &str, check: impl Fn(f64) -> bool) {
    match run(source) {
        Ok(Value::Number(n)) => assert!(check(n), "check failed for ratio {}", n),
        Ok(v) => panic!("expected Number, got {:?}", v),
        Err(e) => panic!("{:#?}", e),
    }
}

/// Ten A* runs in one VM must not slow down by more than 4× (was ~50× before ephemeral arena).
#[test]
fn degradation_same_vm_10_mini_astar() {
    match run(MINI_ASTAR) {
        Ok(Value::Number(ratio)) => {
            println!("degradation_same_vm_10_mini_astar: time_ratio(last/first)={}", ratio);
            assert!(
                ratio < 4.0,
                "expected last/first time ratio < 4 (was ~50× before ephemeral arena), got {}",
                ratio
            );
        }
        Ok(v) => panic!("expected Number, got {:?}", v),
        Err(e) => panic!("{:#?}", e),
    }
}

/// Ten separate VMs should keep similar per-run time (no cross-run store growth).
#[test]
fn stable_fresh_vm_10_runs() {
    let script = r#"
import heapq
fn a_star(rows, cols, start, goal, blocked) {
    start_id = start[0] * cols + start[1]
    goal_id = goal[0] * cols + goal[1]
    goal_r, goal_c = goal
    blocked_ids = {r * cols + c for r, c in blocked}
    g_score = {start_id: 0}
    f_score = {start_id: abs(start[0] - goal_r) + abs(start[1] - goal_c)}
    closed_set = set()
    open_heap = [(f_score[start_id], start_id)]
    open_set = {start_id}
    neighbors_delta = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while open_heap {
        current_f, current = heapq.heappop(open_heap)
        if current_f != f_score.get(current): continue
        open_set.discard(current)
        if current in closed_set: continue
        if current == goal_id { return 1 }
        closed_set.add(current)
        r, c = divmod(current, cols)
        for dr, dc in neighbors_delta {
            nr, nc = r + dr, c + dc
            if !(0 <= nr < rows and 0 <= nc < cols): continue
            neighbor = nr * cols + nc
            if neighbor in blocked_ids or neighbor in closed_set: continue
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, float(inf)) {
                g_score[neighbor] = tentative_g
                f = tentative_g + abs(nr - goal_r) + abs(nc - goal_c)
                f_score[neighbor] = f
                if !neighbor in open_set {
                    heapq.heappush(open_heap, (f, neighbor))
                    open_set.add(neighbor)
                }
            }
        }
    }
    return null
}
a_star(50, 50, (0, 0), (10, 10), set())
"#;
    let mut first = None;
    let mut last = None;
    for i in 0..10 {
        let t0 = Instant::now();
        assert!(run(script).is_ok(), "run {} failed", i + 1);
        let elapsed = t0.elapsed();
        if i == 0 {
            first = Some(elapsed);
        }
        if i == 9 {
            last = Some(elapsed);
        }
    }
    let ratio = last.unwrap().as_secs_f64() / first.unwrap().as_secs_f64();
    println!("stable_fresh_vm_10_runs: ratio={}", ratio);
    assert!(ratio < 2.5, "fresh VM runs degraded: ratio {}", ratio);
}

/// `dict.clear()` on a function parameter (stress-test reuse path).
#[test]
fn dict_clear_on_param() {
    assert_number(
        r#"
fn f(d) {
    d.clear()
    d[1] = 2
    return d.get(1)
}
f({})
"#,
        |n| (n - 2.0).abs() < 1e-9,
    );
}

/// `dict.clear()` resets integral side table.
#[test]
fn dict_clear_reuse() {
    assert_number(
        r#"
d = {1: 2, 2: 3}
d.clear()
d[1] = 10
d.get(1)
"#,
        |n| (n - 10.0).abs() < 1e-9,
    );
}

/// Store growth: five came_from-heavy runs in one VM (reuse + clear).
#[test]
fn value_store_growth_with_reuse_and_clear() {
    let source = r#"
import heapq
fn a_star(rows, cols, g_score, f_score, came_from, closed_set, open_heap, open_set) {
    g_score.clear()
    f_score.clear()
    came_from.clear()
    closed_set.clear()
    open_set.clear()
    heapq.heap_clear(open_heap)
    start_id = 0
    goal_id = cols * (rows - 1) + (cols - 1)
    goal_r, goal_c = rows - 1, cols - 1
    g_score[0] = 0
    f_score[0] = goal_r + goal_c
    open_heap = [(f_score[0], 0)]
    open_set.add(0)
    neighbors_delta = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while open_heap {
        current_f, current = heapq.heappop(open_heap)
        if current_f != f_score.get(current): continue
        open_set.discard(current)
        if current in closed_set: continue
        if current == goal_id { return 1 }
        closed_set.add(current)
        r, c = divmod(current, cols)
        for dr, dc in neighbors_delta {
            nr, nc = r + dr, c + dc
            if !(0 <= nr < rows and 0 <= nc < cols): continue
            neighbor = nr * cols + nc
            if neighbor in closed_set: continue
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, float(inf)) {
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + abs(nr - goal_r) + abs(nc - goal_c)
                f_score[neighbor] = f
                if !neighbor in open_set {
                    heapq.heappush(open_heap, (f, neighbor))
                    open_set.add(neighbor)
                }
            }
        }
    }
    return null
}
g_score = {}
f_score = {}
came_from = {}
closed_set = set()
open_heap = []
open_set = set()
for _ in range(5) {
    a_star(80, 80, g_score, f_score, came_from, closed_set, open_heap, open_set)
}
1
"#;
    let (v, vm) = run_with_vm(source).expect("run");
    assert!(matches!(v, Value::Number(n) if (n - 1.0).abs() < 1e-9));
    let len = vm.value_store_len();
    println!("value_store_growth_with_reuse_and_clear: store_len={}", len);
    assert!(
        len < 500_000,
        "ValueStore grew too large with reuse+clear: {}",
        len
    );
}

/// `heapq.heap_clear` reuse must not allocate a pair per old entry (unlike heappop-clear loop).
#[test]
fn heap_clear_reuse_bounded_store() {
    let source = r#"
import heapq
h = []
for run in range(3) {
    for i in range(5000) {
        heapq.heappush(h, (i, i))
    }
    heapq.heap_clear(h)
}
len(h)
"#;
    let (v, vm) = run_with_vm(source).expect("run");
    assert!(matches!(v, Value::Number(n) if (n - 0.0).abs() < 1e-9));
    let len = vm.value_store_len();
    println!("heap_clear_reuse_bounded_store: store_len={}", len);
    assert!(
        len < 50_000,
        "heap_clear reuse should not allocate per cleared entry, store_len={}",
        len
    );
}

#[test]
fn grid_buffer_reuse_bounded_store() {
    let source = r#"
import grid
rows, cols = 80, 80
n = rows * cols
blocked = grid.alloc_u8(grid.bitmap_bytes(n), 0)
g = grid.alloc_i32(n, 2000000000)
f = grid.alloc_i32(n, 0)
p = grid.alloc_i32(n, -1)
c = grid.alloc_u8(n, 0)
for _ in range(5) {
    grid.astar(rows, cols, (0, 0), (79, 79), blocked, g, f, p, c)
}
grid.store_len()
"#;
    let (v, vm) = run_with_vm(source).expect("run");
    if let Value::Number(n) = v {
        assert!(n < 50_000.0, "store_len={}", n);
    }
    println!(
        "grid_buffer_reuse_bounded_store: store_len={}",
        vm.value_store_len()
    );
}

/// Without reuse/clear, retained top-level heap objects grow the main ValueStore (one cell per object).
/// Graph dict *entries* live inside plain ObjectMaps and do not inflate `value_store.len()`.
/// Dicts created inside a user call use the call arena and are freed on return (see
/// `value_store_growth_with_reuse_and_clear`).
#[test]
fn value_store_growth_without_reuse() {
    let source = r#"
import heapq
rows, cols = 60, 60
neighbors_delta = ((1, 0), (0, 1))
leaked = []
for _ in range(5) {
    g_score = {}
    f_score = {}
    came_from = {}
    closed_set = set()
    open_heap = [(1, 0)]
    open_set = {0}
    steps = 0
    while open_heap and steps < 2000 {
        current_f, current = heapq.heappop(open_heap)
        steps = steps + 1
        closed_set.add(current)
        r, c = divmod(current, cols)
        for dr, dc in neighbors_delta {
            nr, nc = r + dr, c + dc
            if !(0 <= nr < rows and 0 <= nc < cols): continue
            neighbor = nr * cols + nc
            came_from[neighbor] = current
            g_score[neighbor] = steps
            f_score[neighbor] = steps
            heapq.heappush(open_heap, (steps, neighbor))
            open_set.add(neighbor)
        }
    }
    leaked.push((g_score, f_score, came_from, closed_set, open_heap, open_set))
}
for i in range(1200) {
    leaked.push({i: i})
}
len(leaked)
"#;
    let (v, vm) = run_with_vm(source).expect("run");
    assert!(matches!(v, Value::Number(n) if (n - 1205.0).abs() < 1e-9));
    let len = vm.value_store_len();
    println!("value_store_growth_without_reuse: store_len={}", len);
    assert!(
        len > 1000,
        "expected measurable main-store growth when heap objects are retained (got {})",
        len
    );
}

/// A* scratch dicts inside a function must not accumulate in the main store across calls.
#[test]
fn value_store_ephemeral_frees_function_locals() {
    let source = r#"
import heapq
fn a_star(rows, cols) {
    g_score = {}
    f_score = {}
    came_from = {}
    closed_set = set()
    open_heap = [(1, 0)]
    open_set = {0}
    neighbors_delta = ((1, 0), (0, 1))
    steps = 0
    while open_heap and steps < 2000 {
        current_f, current = heapq.heappop(open_heap)
        steps = steps + 1
        closed_set.add(current)
        r, c = divmod(current, cols)
        for dr, dc in neighbors_delta {
            nr, nc = r + dr, c + dc
            if !(0 <= nr < rows and 0 <= nc < cols): continue
            neighbor = nr * cols + nc
            came_from[neighbor] = current
            g_score[neighbor] = steps
            f_score[neighbor] = steps
            heapq.heappush(open_heap, (steps, neighbor))
            open_set.add(neighbor)
        }
    }
    return steps
}
let s = 0
for _ in range(5) {
    s = s + a_star(60, 60)
}
s
"#;
    let (v, vm) = run_with_vm(source).expect("run");
    let _ = v;
    let len = vm.value_store_len();
    println!("value_store_ephemeral_frees_function_locals: store_len={}", len);
    assert!(
        len < 1000,
        "call-arena dicts should be freed on return; main store stayed large (len={})",
        len
    );
}

#[test]
#[ignore = "release timing: cargo test stress_loop_dc_only --test memory_regression_tests --release -- --ignored --nocapture"]
fn stress_loop_dc_only() {
    let t0 = Instant::now();
    assert!(run(MINI_ASTAR).is_ok());
    println!("stress_loop_dc_only: {:?}", t0.elapsed());
}

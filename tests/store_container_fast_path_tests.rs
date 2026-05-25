//! Store-backed dict/set fast paths: no whole-container clone on hot ops (A* RAM fix).

use data_code::{run, Value};
use std::time::Instant;

fn assert_number(source: &str, expected: f64) {
    match run(source) {
        Ok(Value::Number(n)) => assert!(
            (n - expected).abs() < 1e-9,
            "expected {}, got {}\n{}",
            expected,
            n,
            source
        ),
        Ok(v) => panic!("expected Number, got {:?}\n{}", v, source),
        Err(e) => panic!("error {:?}\n{}", e, source),
    }
}

/// `closed_set.add(i)` must not clone the whole set on each `.add` property resolve.
#[test]
fn set_add_integral_many_entries() {
    assert_number(
        r#"
s = set()
for i in range(2000) {
    s.add(i)
}
len(s)
"#,
        2000.0,
    );
}

#[test]
fn set_discard_integral_many_entries() {
    assert_number(
        r#"
s = set()
for i in range(1500) {
    s.add(i)
}
for i in range(500) {
    s.discard(i)
}
len(s)
"#,
        1000.0,
    );
}

/// Mini A* profile (50×50) — regression for dict/set/heapq fast paths, low RAM.
#[test]
fn astar_50x50_terminates() {
    let source = r#"
import heapq

fn a_star(rows, cols, start, goal, blocked) {
    start_id = start[0] * cols + start[1]
    goal_id = goal[0] * cols + goal[1]
    goal_r, goal_c = goal
    blocked_ids = {r * cols + c for r, c in blocked}
    if start_id in blocked_ids or goal_id in blocked_ids: return null
    g_score = {start_id: 0}
    f_score = {start_id: abs(start[0] - goal_r) + abs(start[1] - goal_c)}
    came_from = {}
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
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h = abs(nr - goal_r) + abs(nc - goal_c)
                f = tentative_g + h
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

a_star(50, 50, (0, 0), (12, 34), set())
"#;
    assert_number(source, 1.0);
}

#[test]
#[ignore = "timing: cargo test bench_set_add_integral_loop --release -- --ignored --nocapture"]
fn bench_set_add_integral_loop() {
    let n = 20_000;
    let source = format!(
        r#"
s = set()
for i in range({n}) {{
    s.add(i)
}}
len(s)
"#,
        n = n
    );
    let start = Instant::now();
    assert_number(&source, n as f64);
    println!("bench_set_add_integral_loop: n={} elapsed={:?}", n, start.elapsed());
}

//! Isolate memory-hot VM paths (A* building blocks). Run: cargo test memory_path_isolation --release

use data_code::{run, Value};
use std::time::Instant;

fn assert_number(source: &str, expected: f64) {
    match run(source) {
        Ok(v) => {
            let n = v.as_ieee_f64().unwrap_or_else(|| panic!("expected number, got {:?}\n{}", v, source));
            assert!(
                (n - expected).abs() < 1e-6,
                "expected {}, got {}\n{}",
                expected,
                n,
                source
            );
        }
        Err(e) => panic!("{:#?}\n{}", e, source),
    }
}

/// `set.add` via native call must not rely on cloning the whole set (SET_ADD fast path).
#[test]
fn part_set_add_integral_3000() {
    assert_number(
        r#"
s = set()
for i in range(3000) {
    s.add(i)
}
len(s)
"#,
        3000.0,
    );
}

/// `set.discard` on integral keys (open_set.discard in A*).
#[test]
fn part_set_discard_integral_2000() {
    assert_number(
        r#"
s = set()
for i in range(2000) {
    s.add(i)
}
for i in range(1000) {
    s.discard(i)
}
len(s)
"#,
        1000.0,
    );
}

/// Rewriting the same integral keys must not allocate a new cell per assignment (A* g_score leak).
#[test]
fn part_dict_integral_rewrite_same_keys() {
    assert_number(
        r#"
d = {}
for i in range(5000) {
    d[i] = 0
}
for i in range(5000) {
    d[i] = i
}
len(d)
"#,
        5000.0,
    );
}

/// A* pattern: many cells, each updated a few times (g_score / f_score churn).
#[test]
fn part_dict_astar_score_churn_500x500() {
    assert_number(
        r#"
import heapq
fn churn(rows, cols) {
    g_score = {}
    f_score = {}
    g_score[0] = 0
    f_score[0] = 1
    open_heap = [(1, 0)]
    neighbors_delta = ((1, 0), (0, 1))
    steps = 0
    while open_heap and steps < 8000 {
        current_f, current = heapq.heappop(open_heap)
        if current_f != f_score.get(current): continue
        steps = steps + 1
        r, c = divmod(current, cols)
        for dr, dc in neighbors_delta {
            nr, nc = r + dr, c + dc
            if !(0 <= nr < rows and 0 <= nc < cols): continue
            neighbor = nr * cols + nc
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, float(inf)) {
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + 1
                heapq.heappush(open_heap, (f_score[neighbor], neighbor))
            }
        }
    }
    return steps
}
churn(500, 500)
"#,
        8000.0,
    );
}

/// `dict.get(k, float(inf))` on missing keys must not allocate a new `inf` cell per call.
#[test]
fn part_dict_get_inf_default_churn() {
    assert_number(
        r#"
d = {0: 1}
let n = 0
for i in range(1, 5001) {
    if d.get(i, float(inf)) == float(inf) {
        n = n + 1
    }
}
n
"#,
        5000.0,
    );
}

/// Many heappop+unpack cycles should not grow ValueStore without bound (scratch pair cell).
#[test]
fn part_heappop_unpack_scratch_50k() {
    assert_number(
        r#"
import heapq
h = []
for i in range(100) {
    heapq.heappush(h, (i, i))
}
let n = 0
for k in range(500) {
    while h {
        f, x = heapq.heappop(h)
        n = n + 1
    }
    for i in range(100) {
        heapq.heappush(h, (i + k, i))
    }
}
n
"#,
        50_000.0,
    );
}

/// A* unpack pattern: heappop + tuple assign recycles 2-slot heap pair shells via StoreLocal.
#[test]
fn part_heapq_unpack_turnover_10k() {
    assert_number(
        r#"
import heapq
h = []
for i in range(10000) {
    heapq.heappush(h, (i, i))
}
let n = 0
while h {
    current_f, current = heapq.heappop(h)
    n = n + current
}
n
"#,
        49_995_000.0,
    );
}

/// `while open_heap` must not materialize the whole heap each iteration (JumpIfFalse fast path).
#[test]
fn part_while_nonempty_heap_5000() {
    assert_number(
        r#"
import heapq
h = []
for i in range(5000) {
    heapq.heappush(h, (i, i))
}
n = 0
while h {
    heapq.heappop(h)
    n = n + 1
}
n
"#,
        5000.0,
    );
}

/// heapq push/pop with tuple items — recycles 2-slot heap pair shells.
#[test]
fn part_heapq_tuple_churn_2000() {
    assert_number(
        r#"
import heapq
h = []
for i in range(2000) {
    heapq.heappush(h, (i, i))
}
n = 0
while h {
    f, x = heapq.heappop(h)
    n = n + 1
}
n
"#,
        2000.0,
    );
}

/// Dict integral writes without whole-map take (plain_object_upsert_integral).
#[test]
fn part_dict_write_integral_2000() {
    assert_number(
        r#"
d = {}
for i in range(2000) {
    d[i] = i
}
len(d)
"#,
        2000.0,
    );
}

/// Dict integral reads via bracket + .get (no load_value whole Object).
#[test]
fn part_dict_read_integral_2000() {
    assert_number(
        r#"
d = {}
for i in range(2000) {
    d[i] = i * 2
}
let s = 0
for i in range(2000) {
    s = s + d.get(i, 0)
}
s
"#,
        3_998_000.0,
    );
}

/// object_get fast path for stale-check pattern in A*.
#[test]
fn part_object_get_integral_stale_check() {
    assert_number(
        r#"
f_score = {1: 10}
if 10.0 != f_score.get(2) { 1 } else { 0 }
"#,
        1.0,
    );
}

#[allow(dead_code)]
fn assert_bool(source: &str, expected: bool) {
    match run(source) {
        Ok(Value::Bool(b)) => assert_eq!(b, expected, "\n{}", source),
        Ok(v) => panic!("expected Bool, got {:?}\n{}", v, source),
        Err(e) => panic!("{:#?}\n{}", e, source),
    }
}

/// heapq push/pop loop (open_heap growth pattern).
#[test]
fn part_heapq_push_pop_5000() {
    assert_number(
        r#"
import heapq
h = []
for i in range(5000) {
    heapq.heappush(h, (i, i))
}
let n = 0
while h {
    f, x = heapq.heappop(h)
    n = n + 1
}
n
"#,
        5000.0,
    );
}

/// Core A* loop body without huge grid (validates combined paths).
#[test]
fn part_astar_loop_30x30() {
    assert_number(
        r#"
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
a_star(30, 30, (0, 0), (10, 10), set())
"#,
        1.0,
    );
}

/// A* without `came_from` must reach goal (200×500); pop count printed for profiling.
#[test]
fn part_astar_200x500_no_came_from() {
    let source = r#"
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
    let pops = 0
    while open_heap {
        current_f, current = heapq.heappop(open_heap)
        pops = pops + 1
        if current_f != f_score.get(current): continue
        open_set.discard(current)
        if current in closed_set: continue
        if current == goal_id { return pops }
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
    return pops
}
a_star(200, 500, (0, 0), (50, 250), set())
"#;
    match run(source) {
        Ok(v) => {
            let n = v
                .as_ieee_f64()
                .unwrap_or_else(|| panic!("expected number, got {:?}", v));
            assert!(n > 0.0, "expected positive pop count, got {}", n);
            println!("part_astar_200x500_no_came_from: pops={}", n);
        }
        Err(e) => panic!("{:#?}", e),
    }
}

/// Full grid goal check (same script as `bench_astar_adv_2_single`, no `came_from`).
#[test]
#[ignore = "memory+time: cargo test part_astar_1000x5000_goal --release -- --ignored --nocapture"]
fn part_astar_1000x5000_goal() {
    let t0 = Instant::now();
    assert_number(
        r#"
import heapq
fn a_star(rows, cols, start, goal, blocked) {
    start_id = start[0] * cols + start[1]
    goal_id = goal[0] * cols + goal[1]
    goal_r, goal_c = goal
    blocked_ids = {r * cols + c for r, c in blocked}
    if start_id in blocked_ids or goal_id in blocked_ids: return null
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
a_star(1000, 5000, (0, 0), (559, 1234), set())
"#,
        1.0,
    );
    println!("part_astar_1000x5000_goal: {:?}", t0.elapsed());
}

/// Subset of 1000×5000 grid (should finish in <30s, RSS ≪ full bench).
#[test]
#[ignore = "memory: cargo test part_astar_200x1000 --release -- --ignored --nocapture"]
fn part_astar_200x1000_completes() {
    let source = r#"
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
a_star(200, 1000, (0, 0), (50, 500), set())
"#;
    assert_number(source, 1.0);
}

#[test]
#[ignore = "timing: cargo test part_timing_set_add --release -- --ignored --nocapture"]
fn part_timing_set_add_20k() {
    let n = 20_000;
    let src = format!(
        "s = set()\nfor i in range({n}) {{ s.add(i) }}\nlen(s)\n",
        n = n
    );
    let t0 = Instant::now();
    assert_number(&src, n as f64);
    println!("part_timing_set_add_20k: {:?}", t0.elapsed());
}

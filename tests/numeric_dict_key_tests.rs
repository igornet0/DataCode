//! Dict/set numeric keys: `int` and whole `number`/`float` must share one hash bucket (Python `hash(1)==hash(1.0)`).

use data_code::{run, Value};

fn assert_bool(source: &str, expected: bool) {
    match run(source) {
        Ok(Value::Bool(b)) => assert_eq!(b, expected, "expr:\n{}", source),
        Ok(v) => panic!("expected Bool, got {:?}\n{}", v, source),
        Err(e) => panic!("error {:?}\n{}", e, source),
    }
}

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

fn assert_string(source: &str, expected: &str) {
    match run(source) {
        Ok(Value::String(s)) => assert_eq!(s, expected, "expr:\n{}", source),
        Ok(v) => panic!("expected String, got {:?}\n{}", v, source),
        Err(e) => panic!("error {:?}\n{}", e, source),
    }
}

#[test]
fn numeric_dict_key_get_int_lookup_with_number() {
    assert_string(
        r#"
d = {}
d[1] = "a"
d.get(1.0)
"#,
        "a",
    );
}

#[test]
fn numeric_dict_key_number_then_int_get() {
    assert_number(
        r#"
d = {}
d[1.0] = 42
d.get(1)
"#,
        42.0,
    );
}

#[test]
fn numeric_dict_key_int_and_number_same_slot() {
    assert_number(
        r#"
d = {}
d[1] = 10
d[1.0] = 20
d.get(1)
"#,
        20.0,
    );
}

#[test]
fn numeric_dict_key_set_contains_number_after_int() {
    assert_bool(
        r#"
s = set()
s.add(1)
1.0 in s
"#,
        true,
    );
}

#[test]
fn numeric_dict_key_astar_stale_check_not_falsy() {
    // A*: `current_f != f_score.get(current)` must be false when f matches (int key vs number lookup).
    assert_bool(
        r#"
f_score = {1: 10}
!(10.0 != f_score.get(1))
"#,
        true,
    );
    assert_bool(
        r#"
f_score = {}
f_score[1] = 10
!(10 != f_score.get(1.0))
"#,
        true,
    );
    assert_bool(
        r#"
f_score = {1: 10}
10.0 != f_score.get(2)
"#,
        true,
    );
}

#[test]
fn numeric_dict_key_set_int_number_dedup() {
    assert_number(
        r#"
s = set()
s.add(1)
s.add(1.0)
len(s)
"#,
        1.0,
    );
}

#[test]
fn numeric_dict_key_mini_astar_terminates() {
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

path_ok = a_star(20, 20, (0, 0), (5, 7), set())
path_ok
"#;
    assert_number(source, 1.0);
}

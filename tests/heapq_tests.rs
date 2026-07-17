//! Built-in [`heapq`] module (min-heap on arrays).

#[cfg(test)]
mod tests {
    use data_code::common::numeric::IntValue;
    use data_code::{run, Value};

    fn assert_number(source: &str, expected: f64) {
        let result = run(source);
        let n = match result {
            Ok(Value::Number(n)) => n,
            Ok(Value::Int(IntValue::Finite(i))) => i as f64,
            Ok(v) => panic!("expected Number({}), got {:?}\n{}", expected, v, source),
            Err(e) => panic!("error {:?}\n{}", e, source),
        };
        assert!(
            (n - expected).abs() < 1e-9,
            "expected {}, got {}\nexpr:\n{}",
            expected,
            n,
            source
        );
    }

    fn assert_bool(source: &str, expected: bool) {
        let result = run(source);
        match result {
            Ok(Value::Bool(b)) => assert_eq!(b, expected, "expr:\n{}", source),
            Ok(v) => panic!("expected Bool, got {:?}\n{}", v, source),
            Err(e) => panic!("error {:?}\n{}", e, source),
        }
    }

    fn assert_string(source: &str, expected: &str) {
        let result = run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(s, expected, "expr:\n{}", source),
            Ok(v) => panic!("expected String, got {:?}\n{}", v, source),
            Err(e) => panic!("error {:?}\n{}", e, source),
        }
    }

    #[test]
    fn heappush_heappop_ordering() {
        let source = r#"
import heapq
heap = []
heapq.heappush(heap, 5)
heapq.heappush(heap, 1)
heapq.heappush(heap, 3)
heapq.heappop(heap) * 100 + heapq.heappop(heap) * 10 + heapq.heappop(heap)
"#;
        assert_number(source, 135.0);
    }

    #[test]
    fn tuple_primary_order() {
        let source = r#"
import heapq
heap = []
heapq.heappush(heap, (10, "A"))
heapq.heappush(heap, (5, "B"))
heapq.heappush(heap, (7, "C"))
heapq.heappop(heap)[0]
"#;
        assert_number(source, 5.0);
    }

    #[test]
    fn duplicate_tuples_lexicographic_tiebreak() {
        let source = r#"
import heapq
heap = []
heapq.heappush(heap, (5, "B"))
heapq.heappush(heap, (5, "A"))
heapq.heappop(heap)[1]
"#;
        assert_string(source, "A");
    }

    #[test]
    fn heapify_makes_min_root() {
        let source = r#"
import heapq
arr = [9, 4, 7, 1, 0, 3]
heapq.heapify(arr)
arr[0]
"#;
        assert_number(source, 0.0);
    }

    #[test]
    fn astar_one_neighbor_smoke() {
        let source = r#"
import heapq
fn test(rows, cols, goal_r, goal_c) {
    start_id = 0
    goal_id = goal_r * cols + goal_c
    g_score = {start_id: 0}
    f_score = {start_id: abs(0 - goal_r) + abs(0 - goal_c)}
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
            if neighbor in closed_set: continue
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
test(20, 20, 5, 7)
"#;
        assert_number(source, 1.0);
    }

    #[test]
    fn heappush_expr_tuple_roundtrip_after_pop() {
        let source = r#"
import heapq
fn test() {
    h = [(10, 1)]
    cf, c = heapq.heappop(h)
    heapq.heappush(h, (cf + 1, c + 1))
    cf2, c2 = heapq.heappop(h)
    return c2
}
test()
"#;
        assert_number(source, 2.0);
    }

    #[test]
    fn heappush_vars_roundtrip_after_pop() {
        let source = r#"
import heapq
fn test() {
    h = [(10, 1)]
    cf, c = heapq.heappop(h)
    f = cf + 1
    n = c + 1
    heapq.heappush(h, (f, n))
    cf2, c2 = heapq.heappop(h)
    return c2
}
test()
"#;
        assert_number(source, 2.0);
    }

    #[test]
    fn heappush_flat_while_second_iter() {
        let source = r#"
import heapq
fn test() {
    h = [(10, 1)]
    it = 0
    while h {
        cf, c = heapq.heappop(h)
        if it > 0 { return c }
        f = cf + 1
        n = c + 1
        heapq.heappush(h, (f, n))
        it = it + 1
    }
    return 0
}
test()
"#;
        assert_number(source, 2.0);
    }

    /// Regression: two `HeappushFlat` calls per loop when priority is a heap-stored int (e.g. `steps + 1`).
    #[test]
    fn heappush_flat_twice_per_iter_heap_int_priority() {
        let source = r#"
import heapq
fn test() {
    h = [(1, 0)]
    steps = 0
    while h and steps < 2 {
        cf, c = heapq.heappop(h)
        steps = steps + 1
        heapq.heappush(h, (steps, c + 1))
        heapq.heappush(h, (steps, c + 2))
    }
    return steps
}
test()
"#;
        assert_number(source, 2.0);
    }

    #[test]
    fn astar_loop_head_smoke() {
        let source = r#"
import heapq
fn test(rows, cols) {
    start_id = 0
    g_score = {start_id: 0}
    f_score = {start_id: 100}
    closed_set = set()
    open_heap = [(f_score[start_id], start_id)]
    open_set = {start_id}
    while open_heap {
        current_f, current = heapq.heappop(open_heap)
        if current_f != f_score.get(current): continue
        open_set.discard(current)
        if current in closed_set: continue
        return 1
    }
    return null
}
test(20, 20)
"#;
        assert_number(source, 1.0);
    }

    #[test]
    fn heappop_unpack2_heappush_flat_with_divmod() {
        let source = r#"
import heapq
h = [(10, 1)]
a, b = heapq.heappop(h)
r, c = divmod(b, 5)
heapq.heappush(h, (5, 2))
len(h) + r + c
"#;
        assert_number(source, 2.0);
    }

    #[test]
    fn heappop_unpack2_heappush_flat_roundtrip() {
        let source = r#"
import heapq
h = [(10, 1)]
a, b = heapq.heappop(h)
heapq.heappush(h, (5, 2))
len(h)
"#;
        assert_number(source, 1.0);
    }

    #[test]
    fn heappop_unpack2_twice_after_flat_push() {
        let source = r#"
import heapq
h = [(10, 1)]
a, b = heapq.heappop(h)
heapq.heappush(h, (5, 2))
c, d = heapq.heappop(h)
d
"#;
        assert_number(source, 2.0);
    }

    #[test]
    fn heappop_unpack2_twice_after_expr_push() {
        let source = r#"
import heapq
h = [(10, 1)]
cf, c = heapq.heappop(h)
heapq.heappush(h, (cf + 1, c + 1))
cf2, c2 = heapq.heappop(h)
c2
"#;
        assert_number(source, 2.0);
    }

    #[test]
    fn heappop_unpack2_twice_in_function() {
        let source = r#"
import heapq
fn test() {
    h = [(10, 1)]
    cf, c = heapq.heappop(h)
    heapq.heappush(h, (cf + 1, c + 1))
    cf2, c2 = heapq.heappop(h)
    return c2
}
test()
"#;
        assert_number(source, 2.0);
    }

    #[test]
    fn priority_loop_astar_like() {
        let source = r#"
import heapq
open_heap = []
heapq.heappush(open_heap, (10, 1))
heapq.heappush(open_heap, (5, 7))
heapq.heappush(open_heap, (7, 3))
a = heapq.heappop(open_heap)
b = heapq.heappop(open_heap)
c = heapq.heappop(open_heap)
a[0] == 5 and a[1] == 7 and b[0] == 7 and b[1] == 3 and c[0] == 10 and c[1] == 1
"#;
        assert_bool(source, true);
    }

    #[test]
    fn heappop_empty_index_error() {
        let source = r#"
import heapq
heap = []
ok = false
try {
    heapq.heappop(heap)
} catch e {
    ok = true
}
ok
"#;
        assert_bool(source, true);
    }

    #[test]
    fn duplicate_integers_stable_heap() {
        let source = r#"
import heapq
heap = []
heapq.heappush(heap, 1)
heapq.heappush(heap, 1)
heapq.heappush(heap, 1)
heapq.heappush(heap, 1)
heapq.heappop(heap) + heapq.heappop(heap) + heapq.heappop(heap) + heapq.heappop(heap)
"#;
        assert_number(source, 4.0);
    }

    #[test]
    fn mixed_types_heappush_typeerror() {
        let source = r#"
import heapq
heap = []
heapq.heappush(heap, 1)
heapq.heappush(heap, 2)
ok = false
try {
    heapq.heappush(heap, "a")
} catch e {
    ok = true
}
ok
"#;
        assert_bool(source, true);
    }

    #[test]
    fn heappeek_empty_index_error() {
        let source = r#"
import heapq
heap = []
ok = false
try {
    heapq.heappeek(heap)
} catch e {
    ok = true
}
ok
"#;
        assert_bool(source, true);
    }

    #[test]
    fn heapreplace_empty_index_error() {
        let source = r#"
import heapq
heap = []
ok = false
try {
    heapq.heapreplace(heap, 1)
} catch e {
    ok = true
}
ok
"#;
        assert_bool(source, true);
    }

    #[test]
    fn heapreplace_returns_old_root() {
        let source = r#"
import heapq
heap = [5, 9, 7]
heapq.heapify(heap)
old = heapq.heapreplace(heap, 10)
peek = heapq.heappeek(heap)
old == 5 and peek == 7
"#;
        assert_bool(source, true);
    }
}

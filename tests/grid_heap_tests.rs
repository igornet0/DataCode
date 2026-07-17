//! Grid index-heap tests.

use data_code::{run, Value};

#[test]
fn grid_heap_push_pop_order() {
    let source = r#"
import grid
f = grid.alloc_i32(5, 0)
grid.set_i32(f, 0, 10)
grid.set_i32(f, 1, 5)
grid.set_i32(f, 2, 8)
grid.set_i32(f, 3, 3)
grid.set_i32(f, 4, 2)
h = grid.heap_alloc()
for n in range(5) {
    grid.heap_push(h, n, f)
}
out = []
while grid.heap_len(h) > 0 {
    _, node = grid.heap_pop(h)
    out.push(node)
}
out
"#;
    let v = run(source).expect("run");
    assert_eq!(
        v,
        Value::Array(std::rc::Rc::new(std::cell::RefCell::new(vec![
            Value::Number(4.0),
            Value::Number(3.0),
            Value::Number(1.0),
            Value::Number(2.0),
            Value::Number(0.0),
        ])))
    );
}

#[test]
fn grid_heap_stale_pop() {
    let source = r#"
import grid
f = grid.alloc_i32(3, 0)
grid.set_i32(f, 0, 10)
grid.set_i32(f, 1, 5)
grid.set_i32(f, 2, 8)
h = grid.heap_alloc()
grid.heap_push(h, 0, f)
grid.set_i32(f, 0, 1)
grid.heap_push(h, 0, f)
a, n0 = grid.heap_pop(h)
b, n1 = grid.heap_pop(h)
a == 1 and n0 == 0 and b == 10 and n1 == 0
"#;
    let v = run(source).expect("run");
    assert_eq!(v, Value::Bool(true));
}

#[test]
fn grid_heap_clear_and_len() {
    let source = r#"
import grid
f = grid.alloc_i32(1, 0)
grid.set_i32(f, 0, 1)
h = grid.heap_alloc()
grid.heap_push(h, 0, f)
grid.heap_len(h) == 1 and grid.heap_clear(h) or true and grid.heap_len(h) == 0
"#;
    let v = run(source).expect("run");
    assert_eq!(v, Value::Bool(true));
}

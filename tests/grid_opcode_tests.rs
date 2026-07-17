//! Grid opcode fast-path tests (compile + run).

use data_code::{run, Value};

#[test]
fn grid_opcode_get_set_i32_in_function() {
    let source = r#"
import grid
fn f(buf, i, v) {
    grid.set_i32(buf, i, v)
    return grid.get_i32(buf, i)
}
buf = grid.alloc_i32(4, 0)
f(buf, 2, 42)
"#;
    let v = run(source).expect("run");
    assert_eq!(v, Value::Number(42.0));
}

#[test]
fn grid_opcode_test_blocked() {
    let source = r#"
import grid
bmp = grid.alloc_u8(grid.bitmap_bytes(16), 0)
grid.set_blocked(bmp, 3)
grid.test_blocked(bmp, 3) and !grid.test_blocked(bmp, 0)
"#;
    let v = run(source).expect("run");
    assert_eq!(v, Value::Bool(true));
}

#[test]
fn grid_opcode_heap_push_pop_with_f_buf() {
    let source = r#"
import grid
f = grid.alloc_i32(4, 0)
grid.set_i32(f, 0, 3)
grid.set_i32(f, 1, 1)
h = grid.heap_alloc()
grid.heap_push(h, 0, f)
grid.heap_push(h, 1, f)
a, n0 = grid.heap_pop(h)
b, n1 = grid.heap_pop(h)
a == 1 and n0 == 1 and b == 3 and n1 == 0
"#;
    let v = run(source).expect("run");
    assert_eq!(v, Value::Bool(true));
}

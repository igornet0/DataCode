//! Typed constructor overload resolution (`new Foo(n: int)` vs `new Foo(arr: array[int])`).

use data_code::{run, Value};

fn assert_ok(source: &str) {
    match run(source) {
        Ok(Value::Null) | Ok(Value::Object(_)) => {}
        Ok(v) => {
            let _ = v;
        }
        Err(e) => panic!("expected success, got {:?}\n{}", e, source),
    }
}

#[test]
fn typed_constructor_int_literal() {
    assert_ok(
        r#"
cls FenwickTree {
    new FenwickTree(n: int) { this.n = n }
    new FenwickTree(arr: array[int]) { this.n = len(arr) }
}
fn __main__() {
    FenwickTree(5)
}
"#,
    );
}

#[test]
fn typed_constructor_array_literal() {
    assert_ok(
        r#"
cls FenwickTree {
    new FenwickTree(n: int) { this.n = n }
    new FenwickTree(arr: array[int]) { this.n = len(arr) }
}
fn __main__() {
    FenwickTree([1, 2, 3])
}
"#,
    );
}

#[test]
fn typed_constructor_int_variable_runtime() {
    assert_ok(
        r#"
cls FenwickTree {
    new FenwickTree(n: int) { this.n = n }
    new FenwickTree(arr: array[int]) { this.n = len(arr) }
}
fn __main__() {
    r = 1
    FenwickTree(r)
}
"#,
    );
}

#[test]
fn typed_constructor_array_variable_runtime() {
    assert_ok(
        r#"
cls FenwickTree {
    new FenwickTree(n: int) { this.n = n }
    new FenwickTree(arr: array[int]) { this.n = len(arr) }
}
fn __main__() {
    data = [1, 2, 3]
    FenwickTree(data)
}
"#,
    );
}

#[test]
fn this_method_call_from_constructor_body() {
    assert_ok(
        r#"
cls Calc {
    new Calc(n: int) { this.n = n }
    fn inc() { this.n = this.n + 1 }
    new Calc(start: int, bump: int) {
        this.n = start
        this.inc()
    }
}
fn __main__() {
    Calc(1, 1)
}
"#,
    );
}

#[test]
fn same_class_delegate_empty_body() {
    assert_ok(
        r#"
cls Pair {
    new Pair(x: int) { this.x = x; this.y = 0 }
    new Pair(x: int, y: int) : this(x) { this.y = y }
}
fn __main__() {
    p = Pair(1, 2)
}
"#,
    );
}

#[test]
fn same_class_delegate_with_body() {
    assert_ok(
        r#"
cls FenwickTree {
    new FenwickTree(n: int) {
        this.n = n
        this.bit = []
        for _ in range(n + 1) { this.bit.push(0) }
    }
    fn add(i: int, delta: int) {
        j = i + 1
        while j <= this.n {
            this.bit[j] = this.bit[j] + delta
            j = j + 1
        }
    }
    new FenwickTree(arr: array[int]) : this(len(arr)) {
        for i in range(len(arr)) {
            this.add(i, arr[i])
        }
    }
}
fn __main__() {
    ft = FenwickTree([1, 2, 3])
}
"#,
    );
}

#[test]
fn fenwick_tree_integration() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/ru/09-продвинутые/структуры данных/деревья/fenwick_tree.dc"
    );
    let source = std::fs::read_to_string(path).expect("fenwick_tree.dc");
    match run(&source) {
        Ok(_) => {}
        Err(e) => panic!("fenwick_tree.dc failed: {:?}", e),
    }
}

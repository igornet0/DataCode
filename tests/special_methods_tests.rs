//! Class special methods (@add, @string, @len, …).

#[cfg(test)]
mod tests {
    use data_code::common::numeric::IntValue;
    use data_code::{run, Value};

    fn run_ok(source: &str) -> Value {
        run(source).unwrap_or_else(|e| panic!("unexpected error: {:?}", e))
    }

    fn assert_int(source: &str, expected: i64) {
        match run_ok(source) {
            Value::Int(IntValue::Finite(n)) => assert_eq!(n, expected),
            Value::Number(n) if n == expected as f64 => {}
            other => panic!("expected int {}, got {:?}", expected, other),
        }
    }

    fn assert_bool(source: &str, expected: bool) {
        match run_ok(source) {
            Value::Bool(b) => assert_eq!(b, expected),
            other => panic!("expected bool {}, got {:?}", expected, other),
        }
    }

    fn assert_err_contains(source: &str, substring: &str) {
        let err = run(source).unwrap_err();
        let msg = format!("{:?}", err);
        assert!(
            msg.contains(substring),
            "expected {:?} in error: {}",
            substring,
            msg
        );
    }

    const VECTOR_CLS: &str = r#"
cls Vector {
    x: float
    y: float
    new Vector(x: float, y: float) {
        this.x = x
        this.y = y
    }
    fn @add(other: Vector) -> Vector {
        return Vector(this.x + other.x, this.y + other.y)
    }
    fn @string() -> str {
        return "Vector(" + str(this.x) + ", " + str(this.y) + ")"
    }
    fn @len() -> int {
        return 2
    }
}
"#;

    #[test]
    fn vector_add_and_string_and_len() {
        let src = format!(
            r#"{VECTOR_CLS}
v1 = Vector(1, 2)
v2 = Vector(3, 4)
v3 = v1 + v2
len(v3)
"#
        );
        assert_int(&src, 2);
    }

    #[test]
    fn vector_print_uses_string() {
        let src = format!(
            r#"{VECTOR_CLS}
print(Vector(1, 2))
0
"#
        );
        assert_int(&src, 0);
    }

    #[test]
    fn direct_special_method_call_forbidden() {
        let src = format!(
            r#"{VECTOR_CLS}
v = Vector(1, 2)
v.@string()
"#
        );
        assert_err_contains(&src, "cannot be called directly");
    }

    #[test]
    fn special_method_wrong_signature() {
        let src = r#"
cls Bad {
    new Bad() {}
    fn @len(x: int) -> int { return 0 }
}
"#;
        assert_err_contains(src, "@len` must take 0 arguments");
    }

    #[test]
    fn contains_via_in() {
        let src = r#"
cls Bag {
    data: array
    new Bag() { this.data = [1, 2, 3] }
    fn @contains(x) -> bool {
        for v in this.data {
            if v == x { return true }
        }
        return false
    }
}
b = Bag()
2 in b
"#;
        assert_bool(src, true);
    }

    #[test]
    fn iter_next_for_in() {
        let src = r#"
cls Counter {
    n: int
    new Counter(n: int) { this.n = n }
    fn @iter() -> Counter { return this }
    fn @next() -> int {
        if this.n <= 0 { return null }
        this.n = this.n - 1
        return this.n + 1
    }
}
s = 0
for x in Counter(3) { s = s + x }
s
"#;
        assert_int(src, 6);
    }

    #[test]
    fn call_operator() {
        let src = r#"
cls Adder {
    base: int
    new Adder(b: int) { this.base = b }
    fn @call(x: int) -> int { return this.base + x }
}
Adder(10)(5)
"#;
        assert_int(src, 15);
    }

    #[test]
    fn clone_special() {
        let src = r#"
cls Box {
    v: int
    new Box(v: int) { this.v = v }
    fn @clone() -> Box { return Box(this.v) }
}
b1 = Box(7)
b2 = b1.clone()
b2.v = 99
b1.v
"#;
        assert_int(src, 7);
    }

    #[test]
    fn hash_for_set() {
        let src = r#"
cls Key {
    k: int
    new Key(k: int) { this.k = k }
    fn @hash() -> int { return this.k }
    fn @eq(other: Key) -> bool { return this.k == other.k }
}
s = set()
s.add(Key(1))
s.add(Key(1))
len(s)
"#;
        assert_int(src, 1);
    }

    #[test]
    fn init_after_constructor() {
        let src = r#"
cls Widget {
    ready: bool
    new Widget() { this.ready = false }
    fn @init() { this.ready = true }
}
w = Widget()
if w.ready { 1 } else { 0 }
"#;
        assert_int(src, 1);
    }

    #[test]
    fn get_set_indexing() {
        let src = r#"
cls Map2 {
    a: int
    b: int
    new Map2(a: int, b: int) { this.a = a; this.b = b }
    fn @get(k: str) -> int {
        if k == "a" { return this.a }
        return this.b
    }
    fn @set(k: str, v: int) {
        if k == "a" { this.a = v } else { this.b = v }
    }
}
m = Map2(1, 2)
m["b"] = 5
m["b"]
"#;
        assert_int(src, 5);
    }

    #[test]
    fn special_method_outside_class_forbidden() {
        assert_err_contains("fn @add(x) { return x }", "Expect function name");
    }

    #[test]
    fn duplicate_special_method_in_class() {
        let src = r#"
cls Dup {
    new Dup() {}
    fn @len() -> int { return 0 }
    fn @len() -> int { return 1 }
}
"#;
        assert_err_contains(src, "Duplicate method");
    }

    #[test]
    fn drop_reserved_not_implemented() {
        let src = r#"
cls D {
    new D() {}
    fn @drop() {}
}
"#;
        assert_err_contains(src, "not implemented");
    }

    #[test]
    fn add_without_special_method_type_error() {
        let src = r#"
cls Plain {
    new Plain() {}
}
p = Plain()
p + 1
"#;
        assert_err_contains(src, "Operands must be numbers or strings");
    }
}

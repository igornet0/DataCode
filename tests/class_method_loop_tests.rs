//! Class methods and constructors compiled into nested chunks must finalize jump labels / ForRange
//! so main chunk bytecode is not corrupted (regression for Stack underflow with `for` in methods).

use data_code::{run, Value};
use std::path::Path;

fn deque_example_path(filename: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples/ru/04-продвинутые/структуры данных/")
        .join(filename)
}

#[test]
fn class_method_with_for_loop_and_push_returns_len() {
    let src = r#"
cls C {
    private:
        buf: array
    public:
    new C() {
        this.buf = []
    }
    fn fill() {
        for i in range(0, 3) {
            this.buf = push(this.buf, i)
        }
    }
    fn buflen() {
        return len(this.buf)
    }
}
c = C()
c.fill()
c.buflen()
"#;
    match run(src) {
        Ok(Value::Number(n)) => assert_eq!(n, 3.0),
        Ok(v) => panic!("Expected Number(3), got {:?}", v),
        Err(e) => panic!("{}", e),
    }
}

fn fn_before_class_source() -> &'static str {
    r#"
fn _prepend(items: array, item) {
    rest = [item]
    for i in range(0, len(items)) {
        rest = push(rest, items[i])
    }
    return rest
}

cls DD {
    private:
        xs: array
    public:
    new DD() {
        this.xs = []
    }
    fn poke() {
        this.xs = push(this.xs, 1)
        n = len(this.xs)
        if n >= 2 {
            return 2
        }
        return n
    }
}
d = DD()
d.poke()
"#
}

#[test]
fn fn_before_class_methods_with_conditionals_runs() {
    match run(fn_before_class_source()) {
        Ok(Value::Number(n)) => assert_eq!(n, 1.0),
        Ok(v) => panic!("Expected Number(1), got {:?}", v),
        Err(e) => panic!("{:?}", e),
    }
}

#[test]
fn example_deque_dc_runs() {
    let p = deque_example_path("deque.dc");
    let src = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {}", p.display(), e));
    assert!(
        run(&src).is_ok(),
        "examples deque.dc must compile & run — regression for class-method jump patching"
    );
}
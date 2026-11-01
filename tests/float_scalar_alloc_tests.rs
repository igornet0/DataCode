//! Этап 26: быстрые пути `int|number <op> float` и сравнения float не меняют семантику.
//! Ожидаемые значения сняты с версии до этапа 26.

use data_code::run;

fn eval_str(prelude: &str, expr: &str) -> String {
    let src = format!("{}\nlet r = {}\ntypeof(r) + \" \" + str(r)", prelude, expr);
    match run(&src).unwrap_or_else(|e| panic!("run failed: {:?}\n{}", e, src)) {
        data_code::Value::String(s) => s.to_string(),
        v => panic!("expected string, got {:?} for {}", v, src),
    }
}

fn check(cases: &[(&str, &str, &str)]) {
    for (prelude, expr, expected) in cases {
        assert_eq!(eval_str(prelude, expr), *expected, "{} ; {}", prelude, expr);
    }
}

#[test]
fn int_float_add_sub_mul() {
    check(&[
        ("let f = 2.5", "1 + f", "float 3.5"),
        ("let f = 2.5", "f + 1", "float 3.5"),
        ("let f = 2.5", "f - 2.5", "int 0"),
        ("let f = 2.5", "1 - f", "float -1.5"),
        ("let f = 2.5", "0 * f", "int 0"),
        ("let f = 2.5", "-1 * f", "float -2.5"),
        ("let f = 2.5", "3 * f", "float 7.5"),
        ("let w = 3.0", "1 + w", "int 4"),
        ("let w = 3.0", "w * 2", "int 6"),
        ("let f = 0.1", "f + 0.2", "float 0.30000000000000004"),
        ("let f = 2.5", "1000000000000 + f", "float 1000000000002.5"),
        ("let f = 0.5", "9007199254740993 * f", "int 4503599627370496"),
        ("let f = 2.5", "f / 2", "float 1.25"),
    ]);
}

#[test]
fn float_zero_overflow_inf_nan() {
    check(&[
        ("let f = 0.0", "-1 * f", "int 0"),
        ("let f = -0.0", "f + 0", "int 0"),
        ("let f = 1e308", "f * 10.0", "int 9223372036854776000"),
        ("let f = 1e308", "-f * 10.0", "int -9223372036854776000"),
        ("let f = float(inf)", "f + 1", "float inf"),
        ("let f = float(inf)", "f - f", "float NaN"),
        ("let f = float(nan)", "f + 1", "float NaN"),
    ]);
}

#[test]
fn float_comparisons() {
    check(&[
        ("let f = 2.5", "f < 3", "bool true"),
        ("let f = 2.5", "f > 2", "bool true"),
        ("let f = 2.5", "2 < f", "bool true"),
        ("let f = float(nan)", "f < 1", "bool false"),
        ("let f = float(nan)", "f > 1", "bool true"),
    ]);
}

#[test]
fn float_through_functions_containers_globals() {
    check(&[
        ("fn h(x) { return x + 1.5 }", "h(1)", "float 2.5"),
        ("fn h(x) { return x * 2.0 }", "h(2)", "int 4"),
        ("fn h(x) { let c = 2.5\n return x - c }", "h(1)", "float -1.5"),
        ("let g = 0\nfor i in range(10) { g = g + 0.25 }", "g", "float 2.5"),
        ("let a = [1 + 2.5, 2.5 * 3]", "a[0] + a[1]", "int 11"),
        ("let d = {\"k\": 2.5 - 1}", "d[\"k\"]", "float 1.5"),
        ("let s = 0.0\nfor i in range(4) { let c = i + 2.5\n s = s + c }", "s", "int 16"),
    ]);
}

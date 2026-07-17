use data_code::{run, Value};

fn assert_bool_result(source: &str, expected: bool) {
    match run(source) {
        Ok(Value::Bool(b)) => assert_eq!(b, expected, "source: {}", source),
        Ok(v) => panic!("Expected Bool({}), got {:?}", expected, v),
        Err(e) => panic!("Error: {:?}\nsource: {}", e, source),
    }
}

/// Regression: list with `(0.0, 0.0)` interns `0` as `int`; `[0]` reuses that constant.
/// Tuple indexing must accept whole `int` indices, not only `number`.
#[test]
fn tuple_index_with_reused_int_zero_constant() {
    let source = r#"
        tests = [((2.0, 2.0), true), ((0.0, 0.0), true)]
        t = tests[0]
        t[0][0] == 2.0
    "#;
    assert_bool_result(source, true);
}

#[test]
fn point_in_polygon_example_runs() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples/ru/09-продвинутые/алгоритмы/геометрия/point_in_polygon.dc");
    let source = std::fs::read_to_string(&path).expect("read example");
    let base = path.parent().expect("example dir");
    data_code::run_with_base_path(&source, base).expect("point_in_polygon.dc");
}

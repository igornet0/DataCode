use data_code::{run, Value};

fn assert_bool_result(source: &str, expected: bool) {
    match run(source) {
        Ok(Value::Bool(b)) => assert_eq!(b, expected, "source: {}", source),
        Ok(v) => panic!("Expected Bool({}), got {:?}", expected, v),
        Err(e) => panic!("Error: {:?}\nsource: {}", e, source),
    }
}

fn assert_number_near(source: &str, expected: f64) {
    match run(source) {
        Ok(Value::Number(n)) => assert!(
            (n - expected).abs() < expected.abs().max(1.0) * 1e-12,
            "expected ~{}, got {}",
            expected,
            n
        ),
        Ok(v) => panic!("Expected Number, got {:?}", v),
        Err(e) => panic!("Error: {:?}\nsource: {}", e, source),
    }
}

#[test]
fn literal_1e_minus_9() {
    assert_number_near("1e-9", 1e-9);
}

#[test]
fn comparison_with_epsilon() {
    assert_bool_result("abs(0.0) < 1e-9", true);
    assert_bool_result("1e-10 < 1e-9", true);
    assert_bool_result("1e-8 < 1e-9", false);
}

#[test]
fn geometry_style_epsilon() {
    let source = r#"
        val = (1.0 - 1.0) * (2.0 - 1.0) - (0.0 - 0.0) * (1.0 - 0.0)
        abs(val) < 1e-9
    "#;
    assert_bool_result(source, true);
}

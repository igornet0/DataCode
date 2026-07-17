use data_code::{run, run_with_base_path, Value};
use std::path::PathBuf;

fn assert_string_result(source: &str, expected: &str) {
    match run(source) {
        Ok(Value::String(s)) => assert_eq!(s, expected, "source: {}", source),
        Ok(v) => panic!("Expected String({:?}), got {:?}", expected, v),
        Err(e) => panic!("Error: {:?}\nsource: {}", e, source),
    }
}

fn assert_error_contains(source: &str, substr: &str) {
    match run(source) {
        Ok(v) => panic!("Expected error containing {:?}, got {:?}", substr, v),
        Err(e) => {
            let msg = format!("{:?}", e);
            assert!(
                msg.contains(substr),
                "Expected error containing {:?}, got {:?}",
                substr,
                msg
            );
        }
    }
}

#[test]
fn basic_substring() {
    assert_string_result(r#""babad"[0:3]"#, "bab");
}

#[test]
fn negative_and_full() {
    assert_string_result(r#""hello"[-2:]"#, "lo");
    assert_string_result(r#""hello"[:]"#, "hello");
}

#[test]
fn step_and_reverse() {
    assert_string_result(r#""abcdef"[::2]"#, "ace");
    assert_string_result(r#""abc"[::-1]"#, "cba");
}

#[test]
fn unicode() {
    assert_string_result(r#""Привет"[1:3]"#, "ри");
    assert_string_result(r#""a🙂b🙂c"[1:4]"#, "🙂b🙂");
}

#[test]
fn empty_string() {
    assert_string_result(
        r#"
        s = ""
        s[0:0]
    "#,
        "",
    );
}

#[test]
fn array_slice_still_works() {
    let source = r#"
        arr = [10, 20, 30]
        arr[0:2][0]
    "#;
    match run(source) {
        Ok(Value::Number(n)) => assert_eq!(n, 10.0),
        other => panic!("expected Number(10), got {:?}", other),
    }
}

#[test]
fn type_error_on_number_container() {
    assert_error_contains("42[0:1]", "GetArraySlice requires an array or string");
}

#[test]
fn palindrome_example_runs() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples/ru/09-продвинутые/dp/longest_palindrome_substring.dc");
    let source = std::fs::read_to_string(&path).expect("read example");
    let base = path.parent().expect("example dir");
    run_with_base_path(&source, base).expect("longest_palindrome_substring.dc");
}

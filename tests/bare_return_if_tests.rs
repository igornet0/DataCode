//! Bare return after single-line if regression.

use data_code::{run, Value};

#[test]
fn bare_return_after_single_line_if() {
    let source = r#"
        fn test() {
            if false: return
            return 2
        }
        test()
    "#;
    match run(source) {
        Ok(Value::Number(n)) => assert_eq!(n, 2.0),
        Ok(v) => panic!("expected 2, got {:?}", v),
        Err(e) => panic!("error: {:?}", e),
    }
}

#[test]
fn assign_after_bare_return_if() {
    let source = r#"
        fn test() {
            if false: return
            x = 42
            return x
        }
        test()
    "#;
    match run(source) {
        Ok(Value::Number(n)) => assert_eq!(n, 42.0),
        Ok(v) => panic!("expected 42, got {:?}", v),
        Err(e) => panic!("error: {:?}", e),
    }
}

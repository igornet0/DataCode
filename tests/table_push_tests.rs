//! Integration tests for `table.push(data [, ignore=false])`.

#[cfg(test)]
mod tests {
    use data_code::{run, Value};

    fn run_ok(source: &str) -> Value {
        run(source).unwrap_or_else(|e| panic!("execution failed: {:?}", e))
    }

    fn run_err(source: &str) -> String {
        match run(source) {
            Ok(v) => panic!("expected error, got {:?}", v),
            Err(e) => format!("{:?}", e),
        }
    }

    #[test]
    fn push_object_adds_row() {
        let src = r#"
            users = table([[1, "Ann"]], ["id", "name"])
            n = users.push({id: 2, name: "Bob"})
            print(n)
            print(len(users))
            print(users[1]["name"])
        "#;
        run_ok(src);
    }

    #[test]
    fn push_object_missing_column_errors() {
        let err = run_err(
            r#"
            users = table([[1, "Ann"]], ["id", "name"])
            users.push({id: 2, name: "Bob", age: 25})
        "#,
        );
        assert!(
            err.contains("Column \"age\" does not exist") || err.contains("age"),
            "got: {}",
            err
        );
    }

    #[test]
    fn push_object_ignore_extends_schema() {
        let src = r#"
            users = table([[1, "Ann"]], ["id", "name"])
            users.push({id: 2, name: "Bob", age: 30}, true)
            print(len(users["columns"]))
            print(users[0]["age"])
            print(users[1]["age"])
        "#;
        run_ok(src);
    }

    #[test]
    fn push_array_objects_batch() {
        let src = r#"
            users = table([], ["id", "name"])
            n = users.push([
                {id: 1, name: "Alex"},
                {id: 2, name: "Kate"},
                {id: 3, name: "Bob"},
            ])
            print(n)
            print(len(users))
        "#;
        let v = run_ok(src);
        if let Value::Number(n) = v {
            assert_eq!(n, 0.0); // last print is len(users)=3 but run returns last expr - actually print returns null
        }
        // Re-run with explicit return
        let src2 = r#"
            users = table([], ["id", "name"])
            n = users.push([
                {id: 1, name: "Alex"},
                {id: 2, name: "Kate"},
                {id: 3, name: "Bob"},
            ])
            n
        "#;
        match run_ok(src2) {
            Value::Number(n) => assert_eq!(n, 3.0),
            v => panic!("expected 3, got {:?}", v),
        }
    }

    #[test]
    fn push_positional_array_row() {
        let src = r#"
            users = table([], ["id", "name", "age"])
            users.push([10, "John", 25])
            users[0]["name"]
        "#;
        match run_ok(src) {
            Value::String(s) => assert_eq!(s, "John"),
            v => panic!("expected John, got {:?}", v),
        }
    }

    #[test]
    fn push_positional_wrong_length_errors() {
        let err = run_err(
            r#"
            users = table([], ["id", "name", "age"])
            users.push([10, "John"])
        "#,
        );
        assert!(
            err.contains("Invalid row length") || err.contains("Expected 3"),
            "got: {}",
            err
        );
    }

    #[test]
    fn push_table_same_schema() {
        let src = r#"
            a = table([[1, "x"], [2, "y"]], ["id", "name"])
            b = table([[3, "z"]], ["id", "name"])
            n = a.push(b)
            n + len(a)
        "#;
        match run_ok(src) {
            Value::Number(n) => assert_eq!(n, 4.0),
            v => panic!("expected 4, got {:?}", v),
        }
    }

    #[test]
    fn push_table_extra_column_errors() {
        let err = run_err(
            r#"
            a = table([[1, "x"]], ["id", "name"])
            b = table([[2, "y", 25]], ["id", "name", "age"])
            a.push(b)
        "#,
        );
        assert!(
            err.contains("Unknown column") || err.contains("age"),
            "got: {}",
            err
        );
    }

    #[test]
    fn push_table_ignore_extends_schema() {
        let src = r#"
            a = table([[1, "x"]], ["id", "name"])
            b = table([[2, "y", 25]], ["id", "name", "age"])
            n = a.push(b, true)
            len(a["columns"]) + n
        "#;
        match run_ok(src) {
            Value::Number(n) => assert_eq!(n, 4.0),
            v => panic!("expected 4, got {:?}", v),
        }
    }

    #[test]
    fn push_type_error_string_to_int_column() {
        let err = run_err(
            r#"
            t = table([[1]], ["age"])
            t.push({age: "hello"})
        "#,
        );
        assert!(
            err.contains("Cannot convert string to int") || err.contains("convert"),
            "got: {}",
            err
        );
    }

    #[test]
    fn push_string_numeric_to_int_column_ok() {
        let src = r#"
            t = table([[1]], ["age"])
            t.push({age: "25"})
            t[1]["age"]
        "#;
        match run_ok(src) {
            Value::Int(i) => {
                use data_code::common::numeric::IntValue;
                assert_eq!(i, IntValue::Finite(25));
            }
            Value::Number(n) => assert_eq!(n, 25.0),
            v => panic!("expected 25, got {:?}", v),
        }
    }

    #[test]
    fn push_returns_inserted_count() {
        let src = r#"
            users = table([], ["id", "name"])
            users.push({id: 1, name: "A"})
        "#;
        match run_ok(src) {
            Value::Number(n) => assert_eq!(n, 1.0),
            v => panic!("expected 1, got {:?}", v),
        }
    }

    #[test]
    fn push_on_table_fast_path_view() {
        let src = r#"
            a = table([[1, "a"], [2, "b"]], ["id", "name"])
            b = table([[3, "c"]], ["id", "name"])
            a.push(b)
            len(a)
        "#;
        match run_ok(src) {
            Value::Number(n) => assert_eq!(n, 3.0),
            v => panic!("expected 3, got {:?}", v),
        }
    }

    #[test]
    fn chained_push() {
        let src = r#"
            t = table([], ["id", "name"])
            t.push({id: 1, name: "A"})
            t.push([{id: 2, name: "B"}, {id: 3, name: "C"}])
            len(t)
        "#;
        match run_ok(src) {
            Value::Number(n) => assert_eq!(n, 3.0),
            v => panic!("expected 3, got {:?}", v),
        }
    }

    #[test]
    fn push_named_item_and_ignore() {
        let src = r#"
            users = table([[1, "Ann"]], ["id", "name"])
            users.push(item={id: 2, name: "Bob", age: 30}, ignore=true)
            len(users["columns"])
        "#;
        match run_ok(src) {
            Value::Number(n) => assert_eq!(n, 3.0),
            v => panic!("expected 3 columns, got {:?}", v),
        }
    }

    #[test]
    fn push_named_data_alias_for_item() {
        let src = r#"
            users = table([], ["id"])
            users.push(data={id: 1}, ignore=false)
            len(users)
        "#;
        match run_ok(src) {
            Value::Number(n) => assert_eq!(n, 1.0),
            v => panic!("expected 1 row, got {:?}", v),
        }
    }

    #[test]
    fn push_positional_with_named_ignore() {
        let src = r#"
            users = table([[1, "Ann"]], ["id", "name"])
            users.push({id: 2, name: "Bob", age: 25}, ignore=true)
            len(users["columns"])
        "#;
        match run_ok(src) {
            Value::Number(n) => assert_eq!(n, 3.0),
            v => panic!("expected 3 columns, got {:?}", v),
        }
    }
}

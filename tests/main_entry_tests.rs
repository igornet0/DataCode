// Тесты для fn __main__: get_main_entry_params и вызов с argv

#[cfg(test)]
mod tests {
    use data_code::{get_main_entry_params, run_with_vm_with_args, Value};

    // ========== get_main_entry_params ==========

    #[test]
    fn test_get_main_entry_params_with_defaults() {
        let source = r#"
            fn main(env) { print("main env - ${env}") }
            fn __main__(env: "dev" | "prod" = "dev", count = 5) {
                print("env - ${env}")
                print("count - ${count}")
                main(env)
            }
        "#;
        let params = get_main_entry_params(source).expect("__main__ should be found");
        assert_eq!(params.len(), 2, "two params: env, count");
        assert_eq!(params[0].0, "env");
        assert_eq!(params[0].1.as_ref(), Some(&Value::String("dev".to_string())));
        assert_eq!(params[1].0, "count");
        assert_eq!(params[1].1.as_ref(), Some(&Value::Number(5.0)));
    }

    #[test]
    fn test_get_main_entry_params_no_main() {
        let source = r#"
            fn foo() { return 1 }
            foo()
        "#;
        let params = get_main_entry_params(source);
        assert!(params.is_none());
    }

    #[test]
    fn test_get_main_entry_params_no_params() {
        let source = r#"
            fn __main__() {
                return 42
            }
        "#;
        let params = get_main_entry_params(source).expect("__main__ should be found");
        assert!(params.is_empty());
    }

    #[test]
    fn test_get_main_entry_params_single_param_no_default() {
        let source = r#"
            fn __main__(x) {
                return x
            }
        "#;
        let params = get_main_entry_params(source).expect("__main__ should be found");
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].0, "x");
        assert!(params[0].1.is_none());
    }

    // ========== run_with_vm_with_args: вызов __main__ с дефолтами и с argv ==========

    #[test]
    fn test_main_defaults_no_args() {
        // При отсутствии args __main__ получает значения по умолчанию
        let source = r#"
            fn __main__(env: "dev" | "prod" = "dev", count = 5) {
                return env + ":" + str(count)
            }
        "#;
        let result = run_with_vm_with_args(source, None);
        let (value, _vm) = result.expect("run should succeed");
        match &value {
            Value::String(s) => assert_eq!(s.as_str(), "dev:5", "expected default env and count"),
            v => panic!("expected String, got {:?}", v),
        }
    }

    #[test]
    fn test_main_defaults_empty_args() {
        let source = r#"
            fn __main__(env: "dev" | "prod" = "dev", count = 5) {
                return env + ":" + str(count)
            }
        "#;
        let result = run_with_vm_with_args(source, Some(vec![]));
        let (value, _vm) = result.expect("run should succeed");
        match &value {
            Value::String(s) => assert_eq!(s.as_str(), "dev:5"),
            v => panic!("expected String, got {:?}", v),
        }
    }

    #[test]
    fn test_main_with_args() {
        // argv передаётся как массив строк: [env, count]
        let source = r#"
            fn __main__(env: "dev" | "prod" = "dev", count = 5) {
                return env + ":" + str(count)
            }
        "#;
        let result = run_with_vm_with_args(
            source,
            Some(vec!["prod".to_string(), "10".to_string()]),
        );
        let (value, _vm) = result.expect("run should succeed");
        match &value {
            Value::String(s) => assert_eq!(s.as_str(), "prod:10"),
            v => panic!("expected String, got {:?}", v),
        }
    }

    #[test]
    fn test_main_partial_args_uses_defaults() {
        // Передаём только первый аргумент — второй берётся по умолчанию
        let source = r#"
            fn __main__(env: "dev" | "prod" = "dev", count = 5) {
                return env + ":" + str(count)
            }
        "#;
        let result = run_with_vm_with_args(source, Some(vec!["prod".to_string()]));
        let (value, _vm) = result.expect("run should succeed");
        match &value {
            Value::String(s) => assert_eq!(s.as_str(), "prod:5"),
            v => panic!("expected String, got {:?}", v),
        }
    }

    #[test]
    fn test_main_no_params_returns_value() {
        let source = r#"
            fn __main__() {
                return 100
            }
        "#;
        let result = run_with_vm_with_args(source, None);
        let (value, _vm) = result.expect("run should succeed");
        match value {
            Value::Number(n) => assert_eq!(n, 100.0),
            v => panic!("expected Number(100), got {:?}", v),
        }
    }

    #[test]
    fn test_main_calls_main() {
        // __main__ вызывает main(env) — проверяем, что цепочка работает и возврат от main
        let source = r#"
            fn main(env) {
                return "main got " + env
            }
            fn __main__(env: "dev" | "prod" = "dev", count = 5) {
                return main(env)
            }
        "#;
        let result = run_with_vm_with_args(source, Some(vec!["prod".to_string()]));
        let (value, _vm) = result.expect("run should succeed");
        match &value {
            Value::String(s) => assert_eq!(s.as_str(), "main got prod"),
            v => panic!("expected String, got {:?}", v),
        }
    }

    // ========== Доступ к глобальному argv ==========

    #[test]
    fn test_main_accesses_argv_global() {
        let source = r#"
            fn __main__() {
                return len(argv)
            }
        "#;
        let result = run_with_vm_with_args(source, Some(vec!["a".into(), "b".into(), "c".into()]));
        let (value, _vm) = result.expect("run should succeed");
        match value {
            Value::Number(n) => assert_eq!(n, 3.0, "len(argv) = 3"),
            v => panic!("expected Number(3), got {:?}", v),
        }
    }

    #[test]
    fn test_main_argv_empty() {
        let source = r#"
            fn __main__() {
                return len(argv)
            }
        "#;
        let result = run_with_vm_with_args(source, Some(vec![]));
        let (value, _vm) = result.expect("run should succeed");
        match value {
            Value::Number(n) => assert_eq!(n, 0.0),
            v => panic!("expected Number(0), got {:?}", v),
        }
    }

    #[test]
    fn test_main_argv_element_access() {
        let source = r#"
            fn __main__() {
                return argv[0] + "-" + argv[1]
            }
        "#;
        let result = run_with_vm_with_args(source, Some(vec!["foo".into(), "bar".into()]));
        let (value, _vm) = result.expect("run should succeed");
        match &value {
            Value::String(s) => assert_eq!(s.as_str(), "foo-bar"),
            v => panic!("expected String, got {:?}", v),
        }
    }

    #[test]
    fn test_main_params_and_argv_together() {
        // Параметры first, second получают значения из argv[0], argv[1]
        let source = r#"
            fn __main__(first, second) {
                return first + "|" + second
            }
        "#;
        let result = run_with_vm_with_args(source, Some(vec!["x".into(), "y".into()]));
        let (value, _vm) = result.expect("run should succeed");
        match &value {
            Value::String(s) => assert_eq!(s.as_str(), "x|y"),
            v => panic!("expected String, got {:?}", v),
        }
    }

    // ========== Три и более параметров ==========

    #[test]
    fn test_main_three_params() {
        let source = r#"
            fn __main__(a, b, c) {
                return a + b + c
            }
        "#;
        let result = run_with_vm_with_args(
            source,
            Some(vec!["1".into(), "2".into(), "3".into()]),
        );
        let (value, _vm) = result.expect("run should succeed");
        match &value {
            Value::String(s) => assert_eq!(s.as_str(), "123"),
            v => panic!("expected String, got {:?}", v),
        }
    }

    #[test]
    fn test_main_three_params_with_defaults() {
        let source = r#"
            fn __main__(a = "x", b = "y", c = "z") {
                return a + ":" + b + ":" + c
            }
        "#;
        let result = run_with_vm_with_args(source, Some(vec!["A".into()]));
        let (value, _vm) = result.expect("run should succeed");
        match &value {
            Value::String(s) => assert_eq!(s.as_str(), "A:y:z"),
            v => panic!("expected String, got {:?}", v),
        }
    }

    // ========== Арифметика и преобразование типов ==========

    #[test]
    fn test_main_int_from_argv() {
        let source = r#"
            fn __main__(count) {
                return int(count) * 2
            }
        "#;
        let result = run_with_vm_with_args(source, Some(vec!["21".into()]));
        let (value, _vm) = result.expect("run should succeed");
        match value {
            Value::Number(n) => assert_eq!(n, 42.0),
            v => panic!("expected Number(42), got {:?}", v),
        }
    }

    #[test]
    fn test_main_sum_argv_numbers() {
        let source = r#"
            fn __main__(a, b, c) {
                return int(a) + int(b) + int(c)
            }
        "#;
        let result = run_with_vm_with_args(
            source,
            Some(vec!["10".into(), "20".into(), "12".into()]),
        );
        let (value, _vm) = result.expect("run should succeed");
        match value {
            Value::Number(n) => assert_eq!(n, 42.0),
            v => panic!("expected Number(42), got {:?}", v),
        }
    }

    // ========== Рекурсия и циклы ==========

    #[test]
    fn test_main_recursion() {
        let source = r#"
            fn __main__(n) {
                fn fac(x) {
                    if x <= 1 { return 1 }
                    return x * fac(x - 1)
                }
                return fac(int(n))
            }
        "#;
        let result = run_with_vm_with_args(source, Some(vec!["5".into()]));
        let (value, _vm) = result.expect("run should succeed");
        match value {
            Value::Number(n) => assert_eq!(n, 120.0),
            v => panic!("expected Number(120), got {:?}", v),
        }
    }

    #[test]
    fn test_main_for_loop_over_argv() {
        let source = r#"
            fn __main__() {
                let sum = 0
                for s in argv {
                    sum = sum + int(s)
                }
                return sum
            }
        "#;
        let result = run_with_vm_with_args(
            source,
            Some(vec!["1".into(), "2".into(), "3".into()]),
        );
        let (value, _vm) = result.expect("run should succeed");
        match value {
            Value::Number(n) => assert_eq!(n, 6.0),
            v => panic!("expected Number(6), got {:?}", v),
        }
    }

    #[test]
    fn test_main_while_loop() {
        let source = r#"
            fn __main__(n) {
                let x = int(n)
                let sum = 0
                while x > 0 {
                    sum = sum + x
                    x = x - 1
                }
                return sum
            }
        "#;
        let result = run_with_vm_with_args(source, Some(vec!["4".into()]));
        let (value, _vm) = result.expect("run should succeed");
        match value {
            Value::Number(n) => assert_eq!(n, 10.0), // 4+3+2+1
            v => panic!("expected Number(10), got {:?}", v),
        }
    }

    // ========== try/catch ==========

    #[test]
    fn test_main_try_catch_success() {
        let source = r#"
            fn __main__(a, b) {
                try {
                    return int(a) + int(b)
                } catch {
                    return -1
                }
            }
        "#;
        let result = run_with_vm_with_args(source, Some(vec!["10".into(), "32".into()]));
        let (value, _vm) = result.expect("run should succeed");
        match value {
            Value::Number(n) => assert_eq!(n, 42.0),
            v => panic!("expected Number(42), got {:?}", v),
        }
    }

    #[test]
    fn test_main_try_catch_error() {
        let source = r#"
            fn __main__(a, b) {
                try {
                    return int(a) / int(b)
                } catch {
                    return 999
                }
            }
        "#;
        let result = run_with_vm_with_args(source, Some(vec!["10".into(), "0".into()]));
        let (value, _vm) = result.expect("run should succeed");
        match value {
            Value::Number(n) => assert_eq!(n, 999.0, "division by zero caught"),
            v => panic!("expected Number(999), got {:?}", v),
        }
    }

    // ========== Возврат массивов и объектов ==========

    #[test]
    fn test_main_returns_array() {
        let source = r#"
            fn __main__() {
                return argv
            }
        "#;
        let result = run_with_vm_with_args(source, Some(vec!["x".into(), "y".into()]));
        let (value, _vm) = result.expect("run should succeed");
        match &value {
            Value::Array(arr) => {
                let a = arr.borrow();
                assert_eq!(a.len(), 2);
                assert_eq!(a[0], Value::String("x".into()));
                assert_eq!(a[1], Value::String("y".into()));
            }
            v => panic!("expected Array, got {:?}", v),
        }
    }

    #[test]
    fn test_main_returns_object() {
        let source = r#"
            fn __main__(env, count) {
                return { "env": env, "count": int(count) }
            }
        "#;
        let result = run_with_vm_with_args(source, Some(vec!["prod".into(), "42".into()]));
        let (value, _vm) = result.expect("run should succeed");
        match &value {
            Value::Object(rc) => {
                let o = rc.borrow();
                assert_eq!(o.get("env"), Some(&Value::String("prod".into())));
                assert_eq!(o.get("count"), Some(&Value::Number(42.0)));
            }
            v => panic!("expected Object, got {:?}", v),
        }
    }

    // ========== Глобальные переменные ==========

    #[test]
    fn test_main_global_variable() {
        let source = r#"
            global prefix = "> "
            fn __main__(msg) {
                return prefix + msg
            }
        "#;
        let result = run_with_vm_with_args(source, Some(vec!["hello".into()]));
        let (value, _vm) = result.expect("run should succeed");
        match &value {
            Value::String(s) => assert_eq!(s.as_str(), "> hello"),
            v => panic!("expected String, got {:?}", v),
        }
    }

    // ========== break/continue в цикле ==========

    #[test]
    fn test_main_break_in_loop() {
        let source = r#"
            fn __main__(limit) {
                let sum = 0
                let i = 0
                while i < 100 {
                    i = i + 1
                    if i > int(limit) { break }
                    sum = sum + i
                }
                return sum
            }
        "#;
        let result = run_with_vm_with_args(source, Some(vec!["5".into()]));
        let (value, _vm) = result.expect("run should succeed");
        match value {
            Value::Number(n) => assert_eq!(n, 15.0), // 1+2+3+4+5
            v => panic!("expected Number(15), got {:?}", v),
        }
    }

    #[test]
    fn test_main_continue_in_loop() {
        let source = r#"
            fn __main__() {
                let sum = 0
                for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {
                    if i % 2 == 0 { continue }
                    sum = sum + i
                }
                return sum
            }
        "#;
        let result = run_with_vm_with_args(source, None);
        let (value, _vm) = result.expect("run should succeed");
        match value {
            Value::Number(n) => assert_eq!(n, 25.0), // 1+3+5+7+9
            v => panic!("expected Number(25), got {:?}", v),
        }
    }

    // ========== Логика and/or ==========

    #[test]
    fn test_main_and_condition() {
        let source = r#"
            fn __main__(a, b) {
                if int(a) > 0 and int(b) > 0 {
                    return 1
                }
                return 0
            }
        "#;
        let result = run_with_vm_with_args(source, Some(vec!["1".into(), "2".into()]));
        let (value, _vm) = result.expect("run should succeed");
        match value {
            Value::Number(n) => assert_eq!(n, 1.0),
            v => panic!("expected Number(1), got {:?}", v),
        }
        let result2 = run_with_vm_with_args(source, Some(vec!["0".into(), "2".into()]));
        let (v2, _) = result2.expect("run should succeed");
        match v2 {
            Value::Number(n) => assert_eq!(n, 0.0),
            v => panic!("expected Number(0), got {:?}", v),
        }
    }

    #[test]
    fn test_main_or_condition() {
        let source = r#"
            fn __main__(mode) {
                if mode == "dev" or mode == "prod" {
                    return 1
                }
                return 0
            }
        "#;
        let result = run_with_vm_with_args(source, Some(vec!["prod".into()]));
        let (value, _vm) = result.expect("run should succeed");
        match value {
            Value::Number(n) => assert_eq!(n, 1.0),
            v => panic!("expected Number(1), got {:?}", v),
        }
    }

    // ========== Ошибка в __main__ ==========

    #[test]
    fn test_main_throws_propagates() {
        let source = r#"
            fn __main__(x) {
                if int(x) == 0 {
                    throw "zero not allowed"
                }
                return int(x)
            }
        "#;
        let result = run_with_vm_with_args(source, Some(vec!["0".into()]));
        assert!(result.is_err(), "expected error when __main__ throws");
    }

    #[test]
    fn test_main_undefined_in_subcall() {
        let source = r#"
            fn helper() {
                return undefined_var
            }
            fn __main__() {
                return helper()
            }
        "#;
        let result = run_with_vm_with_args(source, None);
        assert!(result.is_err(), "expected error for undefined variable");
    }

    // ========== null и пустой возврат ==========

    #[test]
    fn test_main_returns_null() {
        let source = r#"
            fn __main__() {
                return null
            }
        "#;
        let result = run_with_vm_with_args(source, None);
        let (value, _vm) = result.expect("run should succeed");
        assert!(matches!(value, Value::Null));
    }
}

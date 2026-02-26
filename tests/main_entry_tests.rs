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
}

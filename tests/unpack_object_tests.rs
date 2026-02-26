// Тесты для распаковки объекта (**obj): object literal spread и call kwargs (CallWithUnpack).

#[cfg(test)]
mod tests {
    use data_code::{run, Value};
    use std::collections::HashMap;

    fn run_ok(source: &str) -> Value {
        run(source).expect("expected Ok")
    }

    fn run_err(source: &str) -> Result<Value, data_code::LangError> {
        run(source)
    }

    fn assert_runtime_error_contains(source: &str, substring: &str) {
        let result = run_err(source);
        assert!(result.is_err(), "expected Err, got Ok");
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains(substring),
            "expected error containing {:?}, got: {}",
            substring,
            err_msg
        );
    }

    /// Получить объект как HashMap (только простые ключи без __meta и т.д.).
    fn object_map(v: &Value) -> Option<std::cell::Ref<HashMap<String, Value>>> {
        match v {
            Value::Object(rc) => Some(rc.borrow()),
            _ => None,
        }
    }

    fn object_has_key_with_number(obj: &Value, key: &str, expected: f64) -> bool {
        if let Some(map) = object_map(obj) {
            if let Some(Value::Number(n)) = map.get(key) {
                return (n - expected).abs() < 1e-10;
            }
        }
        false
    }

    // ========== Object literal spread: { **x, "c": 3 } ==========

    #[test]
    fn test_object_literal_spread_merge() {
        // y = { **x, "c": 3 } → объект с ключами a, b, c
        let source = r#"
            x = { "a": 1, "b": 2 }
            y = { **x, "c": 3 }
            y
        "#;
        let v = run_ok(source);
        let map = object_map(&v).expect("expected Object");
        assert_eq!(map.len(), 3, "expected 3 keys");
        assert!(object_has_key_with_number(&v, "a", 1.0));
        assert!(object_has_key_with_number(&v, "b", 2.0));
        assert!(object_has_key_with_number(&v, "c", 3.0));
    }

    #[test]
    fn test_unpack_literal_then_call_kwargs() {
        // Объект y = { **x, "c": 3 } с ключами a, b, c передаём в f(**y); функция должна принимать те же ключи (a, b, c).
        let source = r#"
            x = { "a": 1, "b": 2 }
            y = { **x, "c": 3 }
            fn f(a, b, c) { return { "a": a, "b": b, "c": c } }
            z = f(**y)
            z
        "#;
        let v = run_ok(source);
        let map = object_map(&v).expect("expected Object (result)");
        assert!(map.contains_key("a"));
        assert!(map.contains_key("b"));
        assert!(map.contains_key("c"));
        assert!(object_has_key_with_number(&v, "a", 1.0));
        assert!(object_has_key_with_number(&v, "b", 2.0));
        assert!(object_has_key_with_number(&v, "c", 3.0));
    }

    // ========== Call with **obj: ключи должны совпадать с параметрами функции ==========

    #[test]
    fn test_call_with_unpack_wrong_keys_error() {
        // test_simple_call.dc: _config = { "a": 1, "b": 2 }, функция принимает opts → ошибка
        let source = r#"
            _config = { "a": 1, "b": 2 }
            fn SettingsConfigDict(opts) { return opts }
            model_config = SettingsConfigDict(**_config)
        "#;
        assert_runtime_error_contains(source, "Object keys must match function parameters");
    }

    #[test]
    fn test_call_with_unpack_correct_keys_ok() {
        // test_simple_call_ok.dc: _config = { "opts": { "a": 1, "b": 2 } } → успех
        let source = r#"
            _config = { "opts": { "a": 1, "b": 2 } }
            fn SettingsConfigDict(opts) { return opts }
            model_config = SettingsConfigDict(**_config)
            model_config
        "#;
        let v = run_ok(source);
        let map = object_map(&v).expect("expected Object (model_config)");
        assert!(map.contains_key("a"));
        assert!(map.contains_key("b"));
        assert!(object_has_key_with_number(&v, "a", 1.0));
        assert!(object_has_key_with_number(&v, "b", 2.0));
    }

    // ========== Class with object spread in class variable (implicit new_0) ==========

    #[test]
    fn test_class_with_object_spread_class_var() {
        // test_class_obj.dc: cls C { model_config = { **_base, "b": 2 } }, x = C(), x.model_config
        let source = r#"
            _base = { "a": 1 }
            cls C {
                model_config = { **_base, "b": 2 }
            }
            x = C()
            x.model_config
        "#;
        let v = run_ok(source);
        let map = object_map(&v).expect("expected Object (model_config)");
        assert!(map.contains_key("a"));
        assert!(map.contains_key("b"));
        assert!(object_has_key_with_number(&v, "a", 1.0));
        assert!(object_has_key_with_number(&v, "b", 2.0));
    }

    #[test]
    fn test_class_with_object_spread_class_var_typed() {
        // test_class_spread_only.dc: model_config: object = { **_base, "b": 2 }; класс с аннотацией типа компилируется и создаётся.
        let source = r#"
            _base = { "a": 1 }
            cls C {
                model_config: object = { **_base, "b": 2 }
            }
            x = C()
            x.model_config
        "#;
        let _v = run_ok(source);
        // Проверяем только отсутствие ошибки (типизированное поле может иметь иное представление в runtime).
    }

    // ========== Class with SettingsConfigDict(**_config) in class var: wrong keys → error ==========

    #[test]
    fn test_class_unpack_call_wrong_keys_error() {
        // test_class_unpack.dc / test_class_unpack2.dc: cls DevSettings { model_config = SettingsConfigDict(**_config) }, _config = { "a": 1, "b": 2 } → ошибка
        let source = r#"
            _config = { "a": 1, "b": 2 }
            fn SettingsConfigDict(opts) { return opts }
            cls DevSettings {
                model_config = SettingsConfigDict(**_config)
            }
            x = DevSettings()
        "#;
        assert_runtime_error_contains(source, "Object keys must match function parameters");
    }

    #[test]
    fn test_class_unpack_call_wrong_keys_error_no_ctor_args() {
        // То же без скобок у класса: cls DevSettings { ... }
        let source = r#"
            _config = { "a": 1, "b": 2 }
            fn SettingsConfigDict(opts) { return opts }
            cls DevSettings {
                model_config = SettingsConfigDict(**_config)
            }
            x = DevSettings()
        "#;
        assert_runtime_error_contains(source, "Object keys must match function parameters");
    }
}

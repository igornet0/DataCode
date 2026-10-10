// Тесты вызова функций импортированного модуля с аргументами.
// Фикстуры: tests/import_fixtures/modcalls.
//
// Баг 1: `import m` + `m.f(x)` падал с «Expected 1 arguments but got 2»: компилятор передаёт
// получатель (сам модуль) первым аргументом, а VM отбрасывал его только у функций без параметров.
//
// Баг 2: если модуль сам делал `import sub`, функции `sub` внутри его пространства имён хранили
// индексы таблицы функций VM модуля. При импорте в другой скрипт они не пересчитывались:
// `sub.f(x)` из функции модуля вызывал чужую функцию или падал с «Can only call functions».
//
// Баг 3: именованные аргументы импортированной функции раскладывались не по именам параметров
// (`m.f(b = 1)` — по позиции, `from m import f; f(b = 1)` — по алфавиту).
//
// Баг 4: `*args` у импортированной функции не упаковывались («Expected 1 arguments but got 3»).

#[cfg(test)]
mod tests {
    use data_code::common::numeric::IntValue;
    use data_code::{run_with_base_path, Value};
    use std::path::PathBuf;

    fn fixtures_dir() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tests");
        path.push("import_fixtures");
        path.push("modcalls");
        path
    }

    fn run(source: &str) -> Result<Value, String> {
        run_with_base_path(source, fixtures_dir().as_path()).map_err(|e| format!("{:?}", e))
    }

    fn assert_number(source: &str, expected: f64) {
        match run(source) {
            Ok(Value::Int(IntValue::Finite(i))) => assert_eq!(i as f64, expected),
            Ok(Value::Float(f)) => assert_eq!(f.as_raw_f64(), expected),
            Ok(Value::Number(n)) => assert!(
                (n - expected).abs() < 1e-10,
                "expected {}, got {}",
                expected,
                n
            ),
            other => panic!("expected {}, got {:?}", expected, other),
        }
    }

    // ========== Баг 1: m.f(x) ==========

    #[test]
    fn test_namespace_call_one_arg() {
        assert_number("import mathx\nmathx.inc(1)", 2.0);
    }

    #[test]
    fn test_namespace_call_two_args() {
        assert_number("import mathx\nmathx.mul(3, 4)", 12.0);
    }

    #[test]
    fn test_namespace_call_zero_args() {
        assert_number("import mathx\nmathx.zero()", 5.0);
    }

    #[test]
    fn test_namespace_call_default_used() {
        assert_number("import mathx\nmathx.with_default(1)", 12.0);
    }

    #[test]
    fn test_namespace_call_default_overridden() {
        assert_number("import mathx\nmathx.with_default(1, 5)", 15.0);
    }

    #[test]
    fn test_namespace_call_from_function() {
        assert_number(
            "import mathx\nfn go(v) { return mathx.mul(v, 3) }\ngo(7)",
            21.0,
        );
    }

    #[test]
    fn test_namespace_in_variable() {
        assert_number("import mathx\nm = mathx\nm.mul(2, 5)", 10.0);
    }

    #[test]
    fn test_plain_dict_method_still_gets_receiver() {
        // Обычный словарь с функцией по-прежнему передаёт себя первым аргументом.
        assert_number(
            "fn pick(self, k) { return self[k] }\nobj = {\"g\": pick, \"v\": 42}\nobj.g(\"v\")",
            42.0,
        );
    }

    #[test]
    fn test_module_passed_as_plain_argument_is_kept() {
        // Получатель отбрасывается только у функций самого модуля, не у show(m).
        assert_number(
            "import mathx\nfn show(m) { return m.mul(2, 3) }\nshow(mathx)",
            6.0,
        );
    }

    // ========== Баг 2: модуль вызывает функции своего `import sub` ==========

    #[test]
    fn test_nested_namespace_via_import() {
        // Функция `caller` сдвигает таблицу функций хоста: индексы wrapper/mathx не совпадают.
        assert_number(
            "fn caller() { return 0 }\nimport wrapper\nwrapper.wrap_inc(4)",
            500.0,
        );
    }

    #[test]
    fn test_nested_namespace_via_from_import() {
        assert_number("from wrapper import wrap_inc\nwrap_inc(4)", 500.0);
    }

    #[test]
    fn test_nested_namespace_zero_arg_via_from_import() {
        assert_number("from wrapper import wrap_zero\nwrap_zero()", 5.0);
    }

    #[test]
    fn test_nested_namespace_top_level_value() {
        assert_number("import wrapper\nwrapper.TOP", 11.0);
    }

    #[test]
    fn test_two_levels_of_nesting() {
        assert_number(
            "fn a() { return 0 }\nfn b() { return 0 }\nimport outer\nouter.outer_inc(1)",
            201.0,
        );
    }

    #[test]
    fn test_two_levels_of_nesting_via_from_import() {
        assert_number("from outer import outer_inc\nouter_inc(1)", 201.0);
    }

    // ========== Баг 3: именованные аргументы ==========

    #[test]
    fn test_namespace_call_named_args_by_name() {
        // sub(z, a): порядок параметров не алфавитный.
        assert_number("import mathx\nmathx.sub(a = 1, z = 10)", 9.0);
    }

    #[test]
    fn test_from_import_named_args_by_name() {
        assert_number("from mathx import sub\nsub(a = 1, z = 10)", 9.0);
    }

    #[test]
    fn test_namespace_call_mixed_positional_and_named() {
        assert_number("import mathx\nmathx.with_default(3, y = 4)", 34.0);
    }

    #[test]
    fn test_from_import_mixed_positional_and_named() {
        assert_number("from mathx import with_default\nwith_default(3, y = 4)", 34.0);
    }

    #[test]
    fn test_namespace_call_named_with_default() {
        assert_number("import mathx\nmathx.with_default(x = 2)", 22.0);
    }

    #[test]
    fn test_namespace_call_unknown_named_arg_is_error() {
        let err = run("import mathx\nmathx.inc(q = 1)").expect_err("unknown kwarg must fail");
        assert!(err.contains("unexpected keyword argument 'q'"), "{}", err);
    }

    #[test]
    fn test_namespace_call_binding_error_is_catchable() {
        assert_number(
            "import mathx\nr = 0\ntry {\n    mathx.inc(q = 1)\n} catch e {\n    r = 7\n}\nr",
            7.0,
        );
    }

    // ========== Баг 4: *args ==========

    #[test]
    fn test_from_import_variadic() {
        assert_number("from mathx import count_all\ncount_all(1, 2, 3)", 3.0);
    }

    #[test]
    fn test_from_import_variadic_single_arg_is_packed() {
        assert_number("from mathx import count_all\ncount_all(7)", 1.0);
    }

    #[test]
    fn test_namespace_call_variadic() {
        assert_number("import mathx\nmathx.count_all(1, 2, 3, 4)", 4.0);
    }

    #[test]
    fn test_namespace_call_fixed_and_variadic() {
        assert_number("import mathx\nmathx.head_and_rest(2, 9, 9)", 202.0);
    }
}

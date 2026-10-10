// Модули программы: все .dc-модули запуска живут в одной VM со своими таблицами глобалов
// (src/vm/program_modules.rs). Фикстуры: tests/import_fixtures/progmod.

#[cfg(test)]
mod tests {
    use data_code::common::numeric::IntValue;
    use data_code::{run_with_base_path, run_with_vm_with_args_and_lib, Value};
    use std::path::PathBuf;

    fn base() -> PathBuf {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("tests");
        p.push("import_fixtures");
        p.push("progmod");
        p
    }

    fn num(v: Value) -> f64 {
        match v {
            Value::Int(IntValue::Finite(i)) => i as f64,
            Value::Float(f) => f.as_raw_f64(),
            Value::Number(n) => n,
            v => panic!("expected number, got {:?}", v),
        }
    }

    fn run(src: &str) -> Result<Value, String> {
        run_with_base_path(src, base().as_path()).map_err(|e| format!("{:?}", e))
    }

    #[test]
    fn diamond_import_shares_one_module() {
        // main -> left -> shared и main -> right -> shared: один экземпляр shared.
        let v = run("from left import left_add\nimport right\nimport shared\nleft_add(1)\nright.right_add(2)\nshared.add(3)").unwrap();
        assert_eq!(num(v), 3.0);
    }

    #[test]
    fn diamond_import_other_order() {
        let v = run("import shared\nshared.add(0)\nfrom left import left_add\nleft_add(1)").unwrap();
        assert_eq!(num(v), 2.0);
    }

    #[test]
    fn import_cycle_binds_partial_module() {
        let v = run("import cyc_a\nimport cyc_b\ncyc_a.read_b() * 10 + cyc_b.read_a()").unwrap();
        assert_eq!(num(v), 21.0);
    }

    #[test]
    fn failed_module_init_runs_again_on_next_import() {
        let v = run(
            "r = 0\ntry {\n    import flaky\n} catch e {\n    r = 1\n}\nimport flaky\nif flaky.READY {\n    r = r + 10\n}\nr",
        )
        .unwrap();
        assert_eq!(num(v), 11.0);
    }

    #[test]
    fn module_does_not_see_main_script_names() {
        let err = run("HOST_ONLY = 5\nfrom leak import read_host\nread_host()").unwrap_err();
        assert!(err.contains("Undefined variable: HOST_ONLY"), "{}", err);
    }

    #[test]
    fn script_next_to_lib_keeps_its_own_functions() {
        // __lib__.dc в папке скрипта больше не выполняется неявно и не сдвигает функции скрипта.
        let mut dir = base();
        dir.push("libdir");
        let v = run_with_base_path(
            "fn go() {\n    return 41\n}\nfrom helper import helper\ngo() + helper() - 7 + 1",
            dir.as_path(),
        )
        .unwrap();
        assert_eq!(num(v), 42.0);
    }

    #[test]
    fn module_state_survives_stateless_reset() {
        // HTTP/WebSocket-обработчики сбрасывают хранилище между запросами; данные модулей остаются.
        let (_, mut vm) = run_with_vm_with_args_and_lib(
            "from shared import add\nadd(1)\nadd(2)",
            None,
            None,
            Some(base().as_path()),
            None,
        )
        .unwrap();
        vm.reset_stores_and_globals_for_stateless();
        let add = vm
            .get_functions()
            .iter()
            .position(|f| f.name == "add")
            .expect("add");
        let v = vm.call_function_by_index(add, &[Value::Int(IntValue::Finite(3))]).unwrap();
        assert_eq!(num(v), 3.0);
    }
}

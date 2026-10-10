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

    // --- #18: m.Class(...), m.X = v, чтение до присваивания, имена встроенных ---

    #[test]
    fn module_function_named_like_builtin_is_exported() {
        let v = run("import nsattr\nfrom nsattr import count\nnsattr.count() * 10 + count()").unwrap();
        assert_eq!(num(v), 0.0);
        let v = run("import nsattr\nnsattr.bump()\nnsattr.count()").unwrap();
        assert_eq!(num(v), 1.0);
    }

    #[test]
    fn class_called_through_namespace() {
        let v = run("import nsattr\nnsattr.Box().area() * 100 + nsattr.Box(3).area()").unwrap();
        assert_eq!(num(v), 109.0);
        let v = run("import nsattr\nnsattr.Pt(1).s() * 100 + nsattr.Pt(1, 2).s()").unwrap();
        assert_eq!(num(v), 1103.0);
        let v = run("import nsattr\nlet B = nsattr.Box\nB(4).area()").unwrap();
        assert_eq!(num(v), 16.0);
        let v = run("import nsattr\nfn make() {\n    return nsattr.Box(5)\n}\nmake().area()").unwrap();
        assert_eq!(num(v), 25.0);
    }

    #[test]
    fn namespace_attribute_assignment_reaches_module() {
        // Запись снаружи видна функциям модуля и последующему from-импорту.
        let v = run(
            "import nsattr\nnsattr.ITEMS = [1, 2, 3]\nnsattr.COUNT = 7\nfrom nsattr import ITEMS, COUNT\nnsattr.items_len() * 1000 + nsattr.count() * 100 + len(ITEMS) * 10 + COUNT",
        )
        .unwrap();
        assert_eq!(num(v), 3737.0);
    }

    #[test]
    fn namespace_attribute_assignment_from_function_alias_and_module() {
        let v = run(
            "import nsattr\nfrom nsattr_setter import set_count\nfn setf(v) {\n    nsattr.COUNT = v\n}\nsetf(11)\nlet a = nsattr.count()\nlet alias = nsattr\nalias.COUNT = 12\nlet b = nsattr.count()\nset_count(13)\nnsattr.bump()\na * 10000 + b * 100 + nsattr.count()",
        )
        .unwrap();
        assert_eq!(num(v), 111214.0);
    }

    #[test]
    fn module_store_is_visible_through_namespace() {
        let v = run("import nsattr\nnsattr.COUNT = 7\nnsattr.bump()\nnsattr.COUNT").unwrap();
        assert_eq!(num(v), 8.0);
    }

    #[test]
    fn global_read_before_assignment_is_an_error() {
        let err = run("fn f() {\n    return B\n}\nf()\nB = 5").unwrap_err();
        assert!(err.contains("Undefined variable: B"), "{}", err);
        let v = run(
            "fn f() {\n    return B\n}\nr = 0\ntry {\n    f()\n} catch e {\n    r = 1\n}\nB = 5\nr * 10 + f()",
        )
        .unwrap();
        assert_eq!(num(v), 15.0);
    }

    #[test]
    fn global_assigned_null_or_in_function_reads_normally() {
        let v = run("fn setg() {\n    global G = 3\n}\nsetg()\nX = null\nif X == null {\n    G = G + 1\n}\nG").unwrap();
        assert_eq!(num(v), 4.0);
    }
}

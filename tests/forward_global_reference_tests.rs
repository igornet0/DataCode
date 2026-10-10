// Функция читает глобальную переменную, объявленную ниже неё.
//
// Баг: компилятор давал неизвестному имени индекс-заглушку usize::MAX и клал его в таблицу
// глобалов. Следующее присваивание `B = 5` эмитировало StoreGlobal(usize::MAX), и VM падала
// с переполнением. Кроме того, все неизвестные имена одной функции делили ключ usize::MAX
// в chunk.global_names: при двух таких именах одно молча подменялось другим.

#[cfg(test)]
mod tests {
    use data_code::{run, run_with_base_path, Value};
    use std::path::PathBuf;

    fn num(source: &str) -> f64 {
        match run(source) {
            Ok(v) => v
                .as_finite_f64()
                .unwrap_or_else(|| panic!("expected number, got {:?}", v)),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    #[test]
    fn read_global_declared_below() {
        assert_eq!(num("fn f() { return B }\nB = 5\nf()"), 5.0);
    }

    #[test]
    fn two_globals_declared_below() {
        assert_eq!(num("fn f() { return B * 10 + C }\nB = 5\nC = 7\nf()"), 57.0);
    }

    #[test]
    fn same_name_used_twice_in_function() {
        assert_eq!(num("fn f() { return B + B }\nB = 4\nf()"), 8.0);
    }

    #[test]
    fn mutate_array_declared_below() {
        assert_eq!(
            num("fn f() { push(L, 1) return len(L) }\nL = []\nf()\nf()"),
            2.0
        );
    }

    #[test]
    fn global_keyword_with_global_declared_below() {
        assert_eq!(
            num("fn get() { return N }\nfn bump() { global N = N + 1 }\nN = 0\nbump()\nbump()\nget()"),
            2.0
        );
    }

    #[test]
    fn nested_function_reads_global_declared_below() {
        assert_eq!(
            num("fn outer() {\n    fn inner() { return K }\n    return inner()\n}\nK = 9\nouter()"),
            9.0
        );
    }

    #[test]
    fn never_defined_name_is_catchable_error() {
        let v = run("fn f() { return Q }\nr = 0\ntry {\n    f()\n} catch e {\n    r = 1\n}\nr")
            .expect("ok");
        assert_eq!(v.as_finite_f64(), Some(1.0));
    }

    #[test]
    fn never_defined_name_in_main_is_error() {
        let err = run("print(ZZZ)").expect_err("undefined");
        assert!(err.to_string().contains("Undefined variable: ZZZ"), "{}", err);
    }

    #[test]
    fn module_data_declared_below_functions() {
        let mut base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        base.push("tests");
        base.push("import_fixtures");
        base.push("modcalls");
        let v = run_with_base_path(
            "from late_data import add, count\nadd(\"a\")\nadd(\"b\")\ncount()",
            base.as_path(),
        )
        .expect("ok");
        assert_eq!(v.as_finite_f64(), Some(3.0));
        let v = run_with_base_path("import late_data\nlate_data.total()", base.as_path())
            .expect("ok");
        assert_eq!(v.as_finite_f64(), Some(30.0));
        assert!(!matches!(v, Value::Null));
    }
}

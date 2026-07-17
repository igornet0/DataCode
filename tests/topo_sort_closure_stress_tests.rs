//! Стресс-тест вложенного `dfs` + `graph[node]` + `graph.keys` (регрессия флака inline-кэша GetArrayElement).

#[cfg(test)]
mod tests {
    use data_code::run;

    fn topo_sort_source() -> &'static str {
        include_str!("../examples/ru/09-продвинутые/структуры данных/графы/topological_sort.dc")
    }

    #[test]
    fn topo_sort_nested_dfs_many_runs() {
        let src = topo_sort_source();
        for i in 0..800 {
            let r = run(src);
            assert!(
                r.is_ok(),
                "run {} failed: {:?}\n---\n{}\n---",
                i,
                r.as_ref().err(),
                src
            );
        }
    }

    /// Присваивание захваченному имени родителя из вложенной функции — ошибка компиляции (immutable closures).
    #[test]
    fn nested_fn_assign_to_outer_is_compile_error() {
        let src = r#"
            fn outer() {
                acc = []
                fn inner() {
                    acc = push(acc, 10)
                }
                inner()
                return acc
            }
            outer()
        "#;
        match data_code::compile(src) {
            Err(data_code::LangError::ParseError { message, .. }) => {
                assert!(
                    message.contains("cannot assign to outer variable"),
                    "unexpected message: {}",
                    message
                );
            }
            Ok(_) => panic!("expected compile error"),
            Err(e) => panic!("expected ParseError, got {:?}", e),
        }
    }
}

//! Колоночное представление таблиц: добавление колонки разделяет буферы, но не меняет
//! ни исходную таблицу, ни её алиасы.

#[cfg(test)]
mod tests {
    use data_code::{run, Value};

    fn num(source: &str) -> f64 {
        match run(source).unwrap_or_else(|e| panic!("execution failed: {:?}", e)) {
            Value::Number(n) => n,
            v => panic!("expected number, got {:?}", v),
        }
    }

    const BASE: &str = r#"
        n = table([[1, 2.0, "x"], [3, 4.0, "y"]], ["a", "b", "s"])
        m = n
    "#;

    #[test]
    fn add_column_does_not_change_alias() {
        assert_eq!(num(&format!("{BASE} n![\"c\"] = n[\"a\"] * n[\"b\"]\n len(m.columns)")), 3.0);
        assert_eq!(num(&format!("{BASE} n![\"c\"] = n[\"a\"] * n[\"b\"]\n len(n.columns)")), 4.0);
    }

    #[test]
    fn chained_columns_read_shared_buffers() {
        let src = format!(
            "{BASE} n![\"c\"] = n[\"a\"] * n[\"b\"]\n n![\"d\"] = n[\"c\"] + 1\n n![\"e\"] = n[\"d\"]\n n[\"e\"][1] + n[\"c\"][0]"
        );
        assert_eq!(num(&src), 15.0);
    }

    #[test]
    fn alias_of_derived_table_is_independent() {
        let src = format!(
            "{BASE} n![\"c\"] = n[\"a\"] * n[\"b\"]\n k = n\n k![\"e\"] = k[\"c\"] * 2\n len(n.columns) * 10 + len(k.columns)"
        );
        assert_eq!(num(&src), 45.0);
    }

    #[test]
    fn push_after_add_column_keeps_other_tables() {
        let src = format!(
            "{BASE} n![\"c\"] = n[\"a\"] * n[\"b\"]\n k = n\n n.push({{a: 5, b: 1.0, s: \"z\", c: 5.0}})\n len(n) * 100 + len(m) * 10 + n[\"c\"][2]"
        );
        assert_eq!(num(&src), 325.0);
    }

    #[test]
    fn rows_of_columnar_table_match_columns() {
        let src = format!(
            "{BASE} t = table_add_column(m, \"z\", [7, 8])\n r = t[1]\n if r[\"s\"] == \"y\" and r[\"a\"] == 3 {{ r[\"z\"] * 10 + t[0][\"z\"] }} else {{ 0 }}"
        );
        assert_eq!(num(&src), 87.0);
    }

    #[test]
    fn duplicate_and_length_errors_unchanged() {
        let dup = run(&format!("{BASE} n![\"c\"] = n[\"a\"]\n table_add_column(n, \"c\", [1, 2])")).unwrap_err();
        assert!(format!("{:?}", dup).contains("already exists"), "{:?}", dup);
        let len = run(&format!("{BASE} table_add_column(n, \"z\", [1, 2, 3])")).unwrap_err();
        assert!(format!("{:?}", len).contains("does not match"), "{:?}", len);
    }

    // Шаг 2: арифметика по колонкам пишет F64/I64; значения и печать — как раньше.
    mod typed {
        use data_code::common::numeric::{FloatValue, IntValue};
        use data_code::{run, Value};

        fn eval(src: &str) -> Value {
            run(src).unwrap_or_else(|e| panic!("run failed: {:?}\n{}", e, src))
        }

        fn cells(src: &str, expr: &str, n: usize) -> Vec<String> {
            (0..n).map(|i| format!("{:?}", eval(&format!("{src}\n {expr}[{i}]")))).collect()
        }

        fn fl(x: f64) -> String {
            format!("{:?}", Value::Float(FloatValue::classify_f64(x)))
        }

        fn int(x: i64) -> String {
            format!("{:?}", Value::Int(IntValue::Finite(x)))
        }

        const T: &str = r#"
            t = table([[1, 1.5, 2], [2, null, 3], [3, 2.5, null]], ["i", "f", "j"])
        "#;

        #[test]
        fn float_column_keeps_float_and_nulls() {
            let src = format!("{T} t![\"c\"] = t[\"f\"] * 2");
            assert_eq!(cells(&src, "t[\"c\"]", 3), vec![fl(3.0), "Null".into(), fl(5.0)]);
            let src = format!("{T} t![\"c\"] = t[\"f\"] + t[\"j\"]");
            assert_eq!(cells(&src, "t[\"c\"]", 3), vec![fl(3.5), "Null".into(), "Null".into()]);
        }

        // Литерал `1` — это Number; Int получается из `int("…")`.
        const TI: &str = r#"
            t = table([[int("1"), 1.5], [int("2"), null], [int("3"), 2.5]], ["i", "f"])
        "#;

        #[test]
        fn number_literals_stay_number() {
            let src = "t = table([[1], [2]], [\"a\"])\n t![\"c\"] = t[\"a\"] * 3";
            assert_eq!(
                cells(src, "t[\"c\"]", 2),
                vec![format!("{:?}", Value::Number(3.0)), format!("{:?}", Value::Number(6.0))]
            );
        }

        #[test]
        fn int_column_stays_int_and_overflow_falls_back() {
            let src = format!("{TI} t![\"c\"] = t[\"i\"] * t[\"i\"]\n t![\"k\"] = t[\"i\"] * 2 + t[\"i\"]");
            assert_eq!(cells(&src, "t[\"c\"]", 3), vec![int(1), int(4), int(9)]);
            // `2` — Number, поэтому Int * Number даёт Number, как и вне таблиц.
            let num = |x: f64| format!("{:?}", Value::Number(x));
            assert_eq!(cells(&src, "t[\"k\"]", 3), vec![num(3.0), num(6.0), num(9.0)]);
            let src = "t = table([[int(\"9223372036854775807\")], [int(\"1\")]], [\"a\"])\n t![\"c\"] = t[\"a\"] + int(\"1\")";
            let c = cells(src, "t[\"c\"]", 2);
            // Переполнение в колонке откатывается на f64 (как до шага 2), а не заворачивается.
            assert_eq!(c, vec![format!("{:?}", Value::Number(i64::MAX as f64 + 1.0)), int(2)]);
        }

        #[test]
        fn mixed_int_float_column() {
            // int и float вперемешку: int-ячейка остаётся int, float — float.
            let src = "t = table([[int(\"1\")], [2.5], [null]], [\"a\"])\n t![\"c\"] = t[\"a\"] * int(\"2\")";
            assert_eq!(cells(src, "t[\"c\"]", 3), vec![int(2), fl(5.0), "Null".into()]);
        }

        #[test]
        fn chained_arithmetic_on_typed_columns() {
            let src = format!(
                "{TI} t![\"c\"] = t[\"f\"] * 2\n t![\"d\"] = t[\"c\"] + t[\"i\"]\n t![\"e\"] = t[\"d\"] - t[\"c\"]\n t![\"g\"] = t[\"e\"] / 2"
            );
            assert_eq!(cells(&src, "t[\"d\"]", 3), vec![fl(4.0), "Null".into(), fl(8.0)]);
            assert_eq!(cells(&src, "t[\"e\"]", 3), vec![fl(1.0), "Null".into(), fl(3.0)]);
            let scalar: Vec<String> = ["1.0 / 2", "null", "3.0 / 2"]
                .iter()
                .map(|e| format!("{:?}", eval(e)))
                .collect();
            assert_eq!(cells(&src, "t[\"g\"]", 3), scalar);
        }

        #[test]
        fn non_finite_floats() {
            let src = "t = table([[1.0], [float(inf)], [float(nan)]], [\"a\"])\n t![\"c\"] = t[\"a\"] * 2\n t![\"d\"] = t[\"c\"] - t[\"c\"]";
            assert_eq!(
                cells(src, "t[\"c\"]", 3),
                vec![fl(2.0), fl(f64::INFINITY), fl(f64::NAN)]
            );
            assert_eq!(cells(src, "t[\"d\"]", 3), vec![fl(0.0), fl(f64::NAN), fl(f64::NAN)]);
        }

        #[test]
        fn all_null_and_div_by_zero() {
            let src = "t = table([[null, 1.0], [null, 0.0]], [\"n\", \"f\"])\n t![\"c\"] = t[\"n\"] + 1\n t![\"d\"] = t[\"f\"] / 0.0";
            assert_eq!(cells(src, "t[\"c\"]", 2), vec!["Null".to_string(), "Null".into()]);
            let typed = cells(src, "t[\"d\"]", 2);
            let scalar = vec![
                format!("{:?}", eval("1.0 / 0.0")),
                format!("{:?}", eval("0.0 / 0.0")),
            ];
            assert_eq!(typed, scalar);
        }

        #[test]
        fn number_columns_match_scalar_arithmetic() {
            // Number-колонки хранятся как F64 с меткой Number. Без Float-операндов каждая ячейка
            // совпадает со скалярной операцией; с Float-операндом колонка, как и раньше, даёт Float.
            let src = format!(
                "{TI} t![\"n\"] = t[\"i\"] * 2\n t![\"a\"] = t[\"n\"] + t[\"n\"]\n t![\"b\"] = t[\"n\"] - t[\"i\"]\n t![\"c\"] = t[\"n\"] * t[\"f\"]\n t![\"d\"] = 1 - t[\"n\"]\n t![\"e\"] = t[\"n\"] + 1.5"
            );
            let rows = ["int(\"1\")", "int(\"2\")", "int(\"3\")"];
            let exprs = [
                ("n", "{i} * 2"),
                ("a", "{i} * 2 + {i} * 2"),
                ("b", "{i} * 2 - {i}"),
                ("d", "1 - {i} * 2"),
            ];
            for (col, e) in exprs {
                let want: Vec<String> = rows
                    .iter()
                    .map(|i| format!("{:?}", eval(&e.replace("{i}", i))))
                    .collect();
                assert_eq!(cells(&src, &format!("t[\"{col}\"]"), 3), want, "column {col}");
            }
            let num = |x: f64| format!("{:?}", Value::Number(x));
            assert_eq!(cells(&src, "t[\"n\"]", 3), vec![num(2.0), num(4.0), num(6.0)]);
            assert_eq!(cells(&src, "t[\"c\"]", 3), vec![fl(3.0), "Null".into(), fl(15.0)]);
            assert_eq!(cells(&src, "t[\"e\"]", 3), vec![fl(3.5), fl(5.5), fl(7.5)]);
        }

        #[test]
        fn number_column_with_nulls_and_kind_switch() {
            // null в Number-колонке, затем строка, где результат становится Int (int×int).
            let src = "t = table([[1, int(\"2\")], [null, int(\"3\")], [int(\"4\"), int(\"5\")]], [\"a\", \"b\"])\n t![\"c\"] = t[\"a\"] * t[\"b\"]";
            let num = |x: f64| format!("{:?}", Value::Number(x));
            assert_eq!(cells(src, "t[\"c\"]", 3), vec![num(2.0), "Null".into(), int(20)]);
            // Ошибка типа сохраняет прежний текст.
            let err = run("t = table([[1], [\"x\"]], [\"a\"])\n t![\"c\"] = t[\"a\"] * 2").unwrap_err();
            assert!(format!("{:?}", err).contains("Column arithmetic needs numeric values"), "{:?}", err);
        }

        #[test]
        fn add_column_from_arithmetic_and_print() {
            // Таблица с типизированными колонками печатается так же, как построенная из литералов.
            let typed = eval(&format!(
                "{T} t![\"c\"] = t[\"f\"] * 2\n t![\"k\"] = t[\"i\"] + 1\n u = table_add_column(t, \"m\", t[\"c\"])\n str(u)"
            ));
            let literal = eval(
                "u = table([[1, 1.5, 2, 3.0, 2, 3.0], [2, null, 3, null, 3, null], [3, 2.5, null, 5.0, 4, 5.0]], [\"i\", \"f\", \"j\", \"c\", \"k\", \"m\"])\n str(u)",
            );
            assert_eq!(typed, literal);
        }

        #[test]
        fn compound_filter_on_columnar_table() {
            // Фильтр читает и базовые, и производные (типизированные) колонки; чужая колонка — false.
            let src = format!(
                "{T} t![\"c\"] = t[\"f\"] * 2\n u = t[\"c\" > 4 or \"i\" == 1 and \"j\" >= 2]\n v = t[\"zz\" > 0 or \"i\" > 2]\n len(u) * 100 + u[\"i\"][1] * 10 + len(v)"
            );
            assert_eq!(format!("{:?}", eval(&src)), format!("{:?}", Value::Number(231.0)));
        }
    }

    mod csv_typed {
        use data_code::run;

        // CSV грузится в типизированные колонки; колонка `m` смешанная (Mixed).
        fn eval_csv(name: &str, expr: &str) -> f64 {
            eval_csv_text(name, "a,b,s,m\n1,1.5,x,1\n2,,y,z\n3,2.5,,\n4,4.5,w,2\n", expr)
        }

        fn eval_csv_text(name: &str, text: &str, expr: &str) -> f64 {
            let path = std::env::temp_dir().join(format!("dc_columnar_{name}_{}.csv", std::process::id()));
            std::fs::write(&path, text).unwrap();
            let src = format!("t = read({:?})\n{expr}", path.to_str().unwrap());
            let out = run(&src).unwrap_or_else(|e| panic!("run failed: {:?}\n{}", e, src));
            let _ = std::fs::remove_file(&path);
            // Большие целые приходят как Int/Float, а не Number.
            out.as_finite_f64().unwrap_or_else(|| panic!("expected number, got {:?}", out))
        }

        #[test]
        fn drop_nulls_on_typed_columns() {
            assert_eq!(eval_csv("dn_all", "u = t.drop_nulls()\n len(u) * 10 + u[\"a\"][1]"), 24.0);
            assert_eq!(eval_csv("dn_b", "len(table_drop_nulls(t, \"b\"))"), 3.0);
            assert_eq!(eval_csv("dn_sm", "u = table_drop_nulls(t, [\"s\", \"m\"])\n len(u) * 10 + u[\"a\"][2]"), 34.0);
            // Исходная таблица не меняется.
            assert_eq!(eval_csv("dn_cow", "u = t.drop_nulls()\n len(t)"), 4.0);
        }

        #[test]
        fn filter_on_typed_columns() {
            assert_eq!(eval_csv("f_or", "u = t[\"b\" > 2 or \"s\" == \"y\"]\n len(u) * 10 + u[\"a\"][0]"), 32.0);
            assert_eq!(eval_csv("f_mixed", "u = t[\"m\" == \"z\" or \"a\" > 3 and \"b\" < 5]\n len(u) * 10 + u[\"a\"][1]"), 24.0);
            assert_eq!(eval_csv("f_none", "u = t[\"a\" > 10 and \"b\" > 0]\n len(u) * 10 + len(u.columns)"), 4.0);
        }

        // `k` — словарная строковая колонка с пустым полем (null) и текстом "null";
        // `g` — целые с пропусками; `d` — даты вперемешку со строкой и числами.
        const DICT: &str = "k,g,v,d\nx,1,10,2024-01-02\ny,,20,2024-01-03\nnull,1,30,x\nx,2,40,7\n,,50,2024-01-02\nx,2,60,3.5\n";

        #[test]
        fn dict_column_drop_nulls_and_filter() {
            // Пустое поле — null, текст "null" — обычная строка.
            assert_eq!(eval_csv_text("dict_dn", DICT, "len(t.drop_nulls(\"k\"))"), 5.0);
            assert_eq!(eval_csv_text("dict_dn_g", DICT, "u = t.drop_nulls(\"g\")\n len(u) * 100 + u[\"v\"][1]"), 430.0);
            let src = "f1 = t[\"k\" == \"x\"]\n f2 = t[\"k\" in [\"y\", \"null\"]]\n f3 = t[\"k\" != \"x\"]\n f4 = t[\"k\" > \"x\"]\n len(f1) * 1000 + len(f2) * 100 + len(f3) * 10 + len(f4)";
            assert_eq!(eval_csv_text("dict_f", DICT, src), 3231.0);
            assert_eq!(eval_csv_text("dict_f_and", DICT, "u = t[\"k\" == \"x\" and \"g\" == 2]\n len(u) * 100 + u[\"v\"][0]"), 240.0);
        }

        #[test]
        fn groupby_dict_key_with_null_and_null_text() {
            // Как и раньше, ключ null и строка "null" попадают в одну группу.
            let src = "g = t.aggregate_group({\"group\": \"k\", \"n\": \"count\", \"s\": {\"op\": \"sum\", \"column\": \"v\"}})\n s = 0\n for i in range(len(g)) { s = s * 1000 + g[\"n\"][i] * 100 + g[\"s\"][i] / 10 }\n len(g) * 1000000000 + s";
            assert_eq!(eval_csv_text("gb_dict", DICT, src), 3_311_102_208.0);
        }

        #[test]
        fn groupby_int_key_with_nulls() {
            // Группы в порядке первого появления: 1, null, 2.
            let src = "g = t.aggregate_group({\"group\": \"g\", \"n\": \"count\", \"s\": {\"op\": \"sum\", \"column\": \"v\"}})\n s = 0\n for i in range(len(g)) { s = s * 1000 + g[\"n\"][i] * 100 + g[\"s\"][i] / 10 }\n z = 0\n if g[\"g\"][1] == null { z = 1 }\n len(g) * 10000000000 + z * 1000000000 + s";
            assert_eq!(eval_csv_text("gb_int", DICT, src), 31_204_207_210.0);
        }

        #[test]
        fn repeated_strings_keep_their_types() {
            // Повторный текст берётся из словаря колонки и остаётся тем же значением.
            let src = "a = 0\n if t[\"d\"][0] == t[\"d\"][4] { a = 1 }\n b = 0\n if t[\"k\"][0] == t[\"k\"][3] { b = 1 }\n c = 0\n if typeof(t[\"d\"][0]) == \"date\" and typeof(t[\"d\"][3]) == \"int\" and typeof(t[\"d\"][2]) == \"string\" { c = 1 }\n a * 100 + b * 10 + c";
            assert_eq!(eval_csv_text("dict_rep", DICT, src), 111.0);
            assert_eq!(eval_csv_text("dict_num", DICT, "t[\"d\"][3] + t[\"d\"][5]"), 10.5);
        }

        #[test]
        fn csv_fields_are_trimmed() {
            // Пробелы и табы вокруг полей и заголовков обрезаются; поле из одних пробелов — null.
            let text = " a , b ,c\n 1 ,\tx\t, 2024-01-02 \n2,   ,  y z \n\t3\t, x ,\n";
            let src = "u = t[\"b\" == \"x\"]\n n = 0\n if t[\"c\"][1] == \"y z\" and t.columns[1] == \"b\" { n = 1 }\n (t[\"a\"][0] + t[\"a\"][2]) * 1000 + len(t.drop_nulls(\"b\")) * 100 + len(u) * 10 + n";
            assert_eq!(eval_csv_text("trim", text, src), 4221.0);
        }
    }
}

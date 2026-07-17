//! Tests for parse_date / format_date and date.format().

#[cfg(test)]
mod tests {
    use data_code::{run, Value};

    fn run_ok(source: &str) -> Value {
        run(source).unwrap_or_else(|e| panic!("execution failed: {:?}", e))
    }

    fn run_err(source: &str) -> data_code::LangError {
        match run(source) {
            Ok(v) => panic!("expected error, got {:?}", v),
            Err(e) => e,
        }
    }

    fn assert_string(v: &Value, expected: &str) {
        match v {
            Value::String(s) => assert_eq!(s.as_str(), expected),
            other => panic!("expected string '{}', got {:?}", expected, other),
        }
    }

    fn assert_number(v: &Value, expected: f64) {
        match v {
            Value::Number(n) => assert!((n - expected).abs() < 1e-9),
            other => {
                let s = other.to_string();
                let n: f64 = s.parse().unwrap_or(f64::NAN);
                assert!((n - expected).abs() < 1e-9, "expected {}, got {:?}", expected, other);
            }
        }
    }

    #[test]
    fn parse_date_european_format() {
        let src = r#"
            d = parse_date("15.03.2024", "%d.%m.%Y")
            typeof(d)
        "#;
        assert_string(&run_ok(src), "date");
    }

    #[test]
    fn parse_date_day_field() {
        let src = r#"
            d = parse_date("15.03.2024", "%d.%m.%Y")
            d.day
        "#;
        assert_number(&run_ok(src), 15.0);
    }

    #[test]
    fn parse_date_with_time() {
        let src = r#"
            d = parse_date("15.03.2024 13:45", "%d.%m.%Y %H:%M")
            d.hour * 100 + d.minute
        "#;
        assert_number(&run_ok(src), 1345.0);
    }

    #[test]
    fn format_date_round_trip() {
        let src = r#"
            d = parse_date("15.03.2024", "%d.%m.%Y")
            format_date(d, "%d.%m.%Y")
        "#;
        assert_string(&run_ok(src), "15.03.2024");
    }

    #[test]
    fn date_format_method() {
        let src = r#"
            d = parse_date("15.03.2024", "%d.%m.%Y")
            d.format("%Y-%m-%d")
        "#;
        assert_string(&run_ok(src), "2024-03-15");
    }

    #[test]
    fn parse_date_invalid_returns_null() {
        let src = r#"
            d = parse_date("not-a-date", "%d.%m.%Y")
            d
        "#;
        assert!(matches!(run_ok(src), Value::Null));
    }

    #[test]
    fn format_date_wrong_type_errors() {
        let _ = run_err(r#"format_date(42, "%Y-%m-%d")"#);
    }

    #[test]
    fn date_auto_parse_regression() {
        let src = r#"
            typeof(date("2024-01-15"))
        "#;
        assert_string(&run_ok(src), "date");
    }

    #[test]
    fn parse_date_us_slash_format() {
        let src = r#"
            d = parse_date("12/10/2020", "%m/%d/%Y")
            d.year
        "#;
        assert_number(&run_ok(src), 2020.0);
    }

    #[test]
    fn date_weekday_field() {
        let src = r#"
            d = date("2024-03-15")
            d.weekday
        "#;
        // 2024-03-15 — пятница, ISO weekday 5
        assert_number(&run_ok(src), 5.0);
    }

    #[test]
    fn date_quarter_field_values() {
        assert_number(&run_ok(r#"date("2024-01-15").quarter"#), 1.0);
        assert_number(&run_ok(r#"date("2024-04-01").quarter"#), 2.0);
        assert_number(&run_ok(r#"date("2024-07-01").quarter"#), 3.0);
        assert_number(&run_ok(r#"date("2024-10-01").quarter"#), 4.0);
    }

    #[test]
    fn date_quarter_method_equals_field() {
        let src = r#"
            d = date("2024-11-10")
            d.quarter == d.quarter()
        "#;
        assert!(matches!(run_ok(src), Value::Bool(true)));
    }

    #[test]
    fn table_map_with_date_native() {
        let src = r#"
            data = table([["2024-03-15"], ["2024-07-01"]], ["OrderDate"])
            mapped = data.map("OrderDate", date)
            typeof(mapped["OrderDate"][0])
        "#;
        assert_string(&run_ok(src), "date");
    }
}

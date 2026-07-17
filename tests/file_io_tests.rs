//! Integration tests for universal `read()` and `save()`.

#[cfg(test)]
mod tests {
    use data_code::common::numeric::IntValue;
    use data_code::{run, run_with_base_path, Value};
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn run_ok(source: &str) -> Value {
        run(source).unwrap_or_else(|e| panic!("execution failed: {:?}", e))
    }

    fn run_err(source: &str) -> String {
        match run(source) {
            Ok(v) => panic!("expected error, got {:?}", v),
            Err(e) => format!("{:?}", e),
        }
    }

    fn escape_path(p: &std::path::Path) -> String {
        p.to_string_lossy().replace('\\', "\\\\")
    }

    fn test_data_path(name: &str) -> String {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("tests/test_data");
        p.push(name);
        escape_path(&p)
    }

    #[test]
    fn read_csv_returns_table() {
        let csv = test_data_path("sample.csv");
        let src = format!(
            r#"
            t = read("{}")
            typeof(t)
        "#,
            csv
        );
        assert_string(&run_ok(&src), "table");
    }

    #[test]
    fn read_json_returns_object() {
        let dir = TempDir::new().expect("tempdir");
        let json_path = dir.path().join("cfg.json");
        fs::write(&json_path, r#"{"name": "test", "count": 3}"#).unwrap();
        let p = escape_path(&json_path);
        let src = format!(
            r#"
            o = read("{}")
            typeof(o)
        "#,
            p
        );
        assert_string(&run_ok(&src), "object");
    }

    #[test]
    fn read_toml_and_yaml_round_trip() {
        let dir = TempDir::new().expect("tempdir");
        let toml_path = dir.path().join("app.toml");
        fs::write(&toml_path, "title = \"demo\"\nversion = 1\n").unwrap();
        let yaml_path = dir.path().join("app.yaml");
        fs::write(&yaml_path, "title: demo\nversion: 1\n").unwrap();

        let toml_src = format!(
            r#"typeof(read("{}"))"#,
            escape_path(&toml_path)
        );
        let yaml_src = format!(
            r#"typeof(read("{}"))"#,
            escape_path(&yaml_path)
        );
        assert_string(&run_ok(&toml_src), "object");
        assert_string(&run_ok(&yaml_src), "object");
    }

    #[test]
    fn read_xml_returns_object_and_save_round_trip() {
        let dir = TempDir::new().expect("tempdir");
        let xml_path = dir.path().join("app.xml");
        fs::write(
            &xml_path,
            r#"<?xml version="1.0"?><app><title>demo</title><count>2</count></app>"#,
        )
        .unwrap();
        let out_path = escape_path(&dir.path().join("out.xml"));
        let src = format!(
            r#"
            o = read("{}")
            typeof(o)
        "#,
            escape_path(&xml_path)
        );
        assert_string(&run_ok(&src), "object");
        let save_src = format!(
            r#"
            o = read("{}")
            save(o, "{}")
            "ok"
        "#,
            escape_path(&xml_path),
            out_path
        );
        assert_string(&run_ok(&save_src), "ok");
        assert!(dir.path().join("out.xml").exists());
    }

    #[test]
    fn read_txt_returns_string() {
        let txt = test_data_path("sample.txt");
        let src = format!(
            r#"
            s = read("{}")
            typeof(s)
        "#,
            txt
        );
        assert_string(&run_ok(&src), "string");
    }

    #[test]
    fn read_bin_returns_bytebuffer() {
        let bin = test_data_path("read_file_bin_sample.bin");
        let src = format!(
            r#"
            b = read("{}")
            typeof(b)
        "#,
            bin
        );
        let ty = match run_ok(&src) {
            Value::String(s) => s,
            other => panic!("expected typeof string, got {:?}", other),
        };
        assert_eq!(ty, "array", "read(.bin) returns ByteBuffer (typeof array)");
    }

    #[test]
    fn read_extension_case_insensitive() {
        let dir = TempDir::new().expect("tempdir");
        let upper = dir.path().join("DATA.CSV");
        fs::copy(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test_data/sample.csv"),
            &upper,
        )
        .unwrap();
        let src = format!(r#"typeof(read("{}"))"#, escape_path(&upper));
        assert_string(&run_ok(&src), "table");
    }

    #[test]
    fn read_nonexistent_includes_path_in_error() {
        let err = run_err(r#"read("definitely_missing_file_io_test.csv")"#);
        assert!(
            err.contains("definitely_missing_file_io_test.csv") || err.contains("does not exist"),
            "error should mention path: {}",
            err
        );
    }

    #[test]
    fn read_invalid_json_error() {
        let dir = TempDir::new().expect("tempdir");
        let bad = dir.path().join("bad.json");
        fs::write(&bad, "{not json").unwrap();
        let err = run_err(&format!(r#"read("{}")"#, escape_path(&bad)));
        assert!(
            err.to_lowercase().contains("json") || err.contains("bad.json"),
            "expected JSON parse error: {}",
            err
        );
    }

    #[test]
    fn save_table_csv_and_json() {
        let dir = TempDir::new().expect("tempdir");
        let out_csv = escape_path(&dir.path().join("out.csv"));
        let out_json = escape_path(&dir.path().join("out.json"));
        let src = format!(
            r#"
            t = table([[1, "a"]], ["id", "name"])
            p1 = save(t, "{}")
            p2 = save(t, "{}")
            p1 + "|" + p2
        "#,
            out_csv, out_json
        );
        let combined = match run_ok(&src) {
            Value::String(s) => s,
            other => panic!("expected string, got {:?}", other),
        };
        assert!(combined.contains("out.csv"));
        assert!(combined.contains("out.json"));
        assert!(dir.path().join("out.csv").exists());
        assert!(dir.path().join("out.json").exists());
    }

    #[test]
    fn save_object_json_toml() {
        let dir = TempDir::new().expect("tempdir");
        let out_json = escape_path(&dir.path().join("cfg.json"));
        let out_toml = escape_path(&dir.path().join("cfg.toml"));
        let src = format!(
            r#"
            o = {{"a": 1, "b": "x"}}
            save(o, "{}")
            save(o, "{}")
            "ok"
        "#,
            out_json, out_toml
        );
        assert_string(&run_ok(&src), "ok");
        assert!(dir.path().join("cfg.json").exists());
        assert!(dir.path().join("cfg.toml").exists());
    }

    #[test]
    fn save_string_txt() {
        let dir = TempDir::new().expect("tempdir");
        let out = escape_path(&dir.path().join("note.txt"));
        let src = format!(
            r#"
            save("hello file io", "{}")
        "#,
            out
        );
        run_ok(&src);
        let content = fs::read_to_string(dir.path().join("note.txt")).unwrap();
        assert_eq!(content, "hello file io");
    }

    #[test]
    fn save_incompatible_type_extension() {
        let dir = TempDir::new().expect("tempdir");
        let out = escape_path(&dir.path().join("img.png"));
        let err = run_err(&format!(
            r#"
            t = table([[1]], ["x"])
            save(t, "{}")
        "#,
            out
        ));
        assert!(
            err.contains("png") || err.contains("table"),
            "expected incompatible save error: {}",
            err
        );
    }

    #[test]
    fn toml_read_json_save_conversion() {
        let dir = TempDir::new().expect("tempdir");
        let toml_in = dir.path().join("in.toml");
        fs::write(&toml_in, "name = \"converted\"\n").unwrap();
        let json_out = escape_path(&dir.path().join("out.json"));
        let src = format!(
            r#"
            cfg = read("{}")
            save(cfg, "{}")
            typeof(read("{}"))
        "#,
            escape_path(&toml_in),
            json_out,
            json_out
        );
        assert_string(&run_ok(&src), "object");
    }

    #[test]
    fn save_with_filename_kwarg() {
        let dir = TempDir::new().expect("tempdir");
        let out = escape_path(&dir.path().join("kw.csv"));
        let src = format!(
            r#"
            t = table([[1]], ["x"])
            save(t, filename="{}")
        "#,
            out
        );
        run_ok(&src);
        assert!(dir.path().join("kw.csv").exists());
    }

    #[test]
    fn list_files_read_save_loop() {
        let dir = TempDir::new().expect("tempdir");
        let sub = dir.path().join("in");
        let out = dir.path().join("out");
        fs::create_dir_all(&sub).unwrap();
        fs::create_dir_all(&out).unwrap();
        fs::write(sub.join("a.txt"), "line one").unwrap();
        fs::write(sub.join("b.csv"), "id,name\n1,Ann\n").unwrap();

        let sub_esc = escape_path(&sub);
        let out_esc = escape_path(&out);
        let src = format!(
            r#"
            count = 0
            for file in list_files("{}") {{
                data = read(file)
                if isinstance(data, table) {{
                    save(data, "{}/" + file.name)
                }} else {{
                    save(data, "{}/" + file.name)
                }}
                count = count + 1
            }}
            count
        "#,
            sub_esc, out_esc, out_esc
        );
        assert_number(&run_ok(&src), 2.0);
        assert!(out.join("a.txt").exists());
        assert!(out.join("b.csv").exists());
    }

    #[test]
    fn read_relative_path_resolves_against_base_path() {
        let dir = TempDir::new().expect("tempdir");
        let csv_path = dir.path().join("sample.csv");
        fs::write(&csv_path, "a,b\n1,2\n").unwrap();
        let src = r#"
            t = read(path("./sample.csv"))
            typeof(t)
        "#;
        assert_string(
            &run_with_base_path(src, dir.path()).expect("read relative to script dir"),
            "table",
        );
    }

    fn assert_string(v: &Value, expected: &str) {
        match v {
            Value::String(s) => assert_eq!(s.as_str(), expected),
            other => panic!("expected string '{}', got {:?}", expected, other),
        }
    }

    fn assert_number(v: &Value, expected: f64) {
        let n = match v {
            Value::Number(n) => *n,
            Value::Int(IntValue::Finite(i)) => *i as f64,
            other => panic!("expected number {}, got {:?}", expected, other),
        };
        assert!(
            (n - expected).abs() < 1e-9,
            "expected {}, got {}",
            expected,
            n
        );
    }
}

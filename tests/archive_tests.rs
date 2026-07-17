//! Integration tests for Archive API.

#[cfg(test)]
mod tests {
    use data_code::{run, Value};
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::Path;
    use tempfile::TempDir;
    use zip::write::SimpleFileOptions;
    use zip::ZipWriter;

    fn run_ok(source: &str) -> Value {
        run(source).unwrap_or_else(|e| panic!("execution failed: {:?}", e))
    }

    fn run_err(source: &str) -> String {
        match run(source) {
            Ok(v) => panic!("expected error, got {:?}", v),
            Err(e) => format!("{:?}", e),
        }
    }

    fn escape_path(p: &Path) -> String {
        p.to_string_lossy().replace('\\', "\\\\")
    }

    fn create_test_zip(path: &Path) {
        let file = File::create(path).expect("create zip");
        let mut zip = ZipWriter::new(file);
        let opts = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
        zip.start_file("README.md", opts).unwrap();
        zip.write_all(b"# Hello Archive\n").unwrap();
        zip.start_file("config/settings.json", opts).unwrap();
        zip.write_all(b"{\"enabled\": true, \"name\": \"demo\"}").unwrap();
        zip.start_file("data/sample.csv", opts).unwrap();
        zip.write_all(b"id,name\n1,alpha\n2,beta\n").unwrap();
        zip.finish().unwrap();
    }

    fn create_test_7z(path: &Path) {
        let dir = path.parent().unwrap().join("7z_src");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("note.txt"), "sevenz content").unwrap();
        sevenz_rust2::compress_to_path(dir, path).expect("create 7z");
    }

    fn create_fake_rar_magic(path: &Path) {
        let mut data = b"Rar!\x1a\x07\x00".to_vec();
        data.extend(b"not a real rar");
        fs::write(path, data).unwrap();
    }

    #[test]
    fn archive_open_zip_properties() {
        let dir = TempDir::new().expect("tempdir");
        let zip_path = dir.path().join("backup.zip");
        create_test_zip(&zip_path);
        let p = escape_path(&zip_path);

        assert_eq!(
            run_ok(&format!(r#"archive("{}").format"#, p)),
            Value::String("zip".into())
        );
        assert_eq!(
            run_ok(&format!(r#"archive("{}").count"#, p)),
            Value::Number(3.0)
        );
        assert_eq!(
            run_ok(&format!(r#"archive("{}").size > 0"#, p)),
            Value::Bool(true)
        );
        assert_eq!(
            run_ok(&format!(r#"typeof(archive("{}").files)"#, p)),
            Value::String("array".into())
        );
    }

    #[test]
    fn archive_read_json_and_csv() {
        let dir = TempDir::new().expect("tempdir");
        let zip_path = dir.path().join("data.zip");
        create_test_zip(&zip_path);
        let p = escape_path(&zip_path);

        let json_src = format!(
            r#"typeof(archive("{}").read("config/settings.json"))"#,
            p
        );
        assert_eq!(run_ok(&json_src), Value::String("object".into()));

        let csv_src = format!(
            r#"typeof(archive("{}").read("data/sample.csv"))"#,
            p
        );
        assert_eq!(run_ok(&csv_src), Value::String("table".into()));
    }

    #[test]
    fn archive_read_text() {
        let dir = TempDir::new().expect("tempdir");
        let zip_path = dir.path().join("docs.zip");
        create_test_zip(&zip_path);
        let p = escape_path(&zip_path);

        let src = format!(
            r#"
            z = archive("{}")
            t = z.read_text("README.md")
            contains(t, "Hello Archive")
        "#,
            p
        );
        assert_eq!(run_ok(&src), Value::Bool(true));
    }

    #[test]
    fn archive_extract_all() {
        let dir = TempDir::new().expect("tempdir");
        let zip_path = dir.path().join("bundle.zip");
        create_test_zip(&zip_path);
        let out = dir.path().join("out");
        fs::create_dir_all(&out).unwrap();

        let src = format!(
            r#"
            z = archive("{}")
            z.extract("{}")
            z.close()
            path_exists(path("{}/config/settings.json"))
        "#,
            escape_path(&zip_path),
            escape_path(&out),
            escape_path(&out),
        );
        assert_eq!(run_ok(&src), Value::Bool(true));
    }

    #[test]
    fn archive_missing_entry_error() {
        let dir = TempDir::new().expect("tempdir");
        let zip_path = dir.path().join("small.zip");
        create_test_zip(&zip_path);
        let p = escape_path(&zip_path);

        let src = format!(
            r#"archive("{}").read("missing.txt")"#,
            p
        );
        let err = run_err(&src);
        assert!(err.contains("not found") || err.contains("missing.txt"));
    }

    #[test]
    fn archive_missing_file_error() {
        let dir = TempDir::new().expect("tempdir");
        let missing = dir.path().join("nope.zip");
        let src = format!(r#"archive("{}")"#, escape_path(&missing));
        let err = run_err(&src);
        assert!(err.contains("not found") || err.contains("Archive"));
    }

    #[test]
    fn archive_unsupported_format_error() {
        let dir = TempDir::new().expect("tempdir");
        let bad = dir.path().join("bad.bin");
        fs::write(&bad, b"NOT_AN_ARCHIVE").unwrap();
        let src = format!(r#"archive("{}")"#, escape_path(&bad));
        let err = run_err(&src);
        assert!(err.contains("Unsupported") || err.contains("unsupported"));
    }

    #[test]
    fn archive_typeof_isinstance() {
        let dir = TempDir::new().expect("tempdir");
        let zip_path = dir.path().join("t.zip");
        create_test_zip(&zip_path);
        let p = escape_path(&zip_path);

        assert_eq!(
            run_ok(&format!(r#"typeof(archive("{}"))"#, p)),
            Value::String("archive".into())
        );
        assert_eq!(
            run_ok(&format!(r#"isinstance(archive("{}"), "archive")"#, p)),
            Value::Bool(true)
        );
    }

    #[test]
    fn archive_open_7z() {
        let dir = TempDir::new().expect("tempdir");
        let archive_path = dir.path().join("backup.7z");
        create_test_7z(&archive_path);
        let p = escape_path(&archive_path);

        assert_eq!(
            run_ok(&format!(r#"archive("{}").format"#, p)),
            Value::String("7z".into())
        );
        assert_eq!(
            run_ok(&format!(r#"archive("{}").count >= 1"#, p)),
            Value::Bool(true)
        );
    }

    #[test]
    fn archive_rar_without_feature_gives_clear_error() {
        #[cfg(feature = "archive-rar")]
        {
            return;
        }
        let dir = TempDir::new().expect("tempdir");
        let rar_path = dir.path().join("fake.rar");
        create_fake_rar_magic(&rar_path);
        let src = format!(r#"archive("{}")"#, escape_path(&rar_path));
        let err = run_err(&src);
        assert!(
            err.contains("archive-rar") || err.contains("RAR"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn archive_close_prevents_read() {
        let dir = TempDir::new().expect("tempdir");
        let zip_path = dir.path().join("closed.zip");
        create_test_zip(&zip_path);
        let p = escape_path(&zip_path);

        let src = format!(
            r#"
            z = archive("{}")
            z.close()
            z.read("README.md")
        "#,
            p
        );
        let err = run_err(&src);
        assert!(err.contains("closed") || err.contains("Closed"));
    }

    #[test]
    fn archive_lru_cache_repeat_read() {
        let dir = TempDir::new().expect("tempdir");
        let zip_path = dir.path().join("cache.zip");
        create_test_zip(&zip_path);
        let p = escape_path(&zip_path);

        let src = format!(
            r#"
            z = archive("{}")
            a = z.read("config/settings.json")
            b = z.read("config/settings.json")
            a == b
        "#,
            p
        );
        assert_eq!(run_ok(&src), Value::Bool(true));
    }
}

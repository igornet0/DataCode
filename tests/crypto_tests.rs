//! VM crypto builtins and built-in `crypto` module (Argon2, bcrypt).

#[cfg(test)]
mod tests {
    use data_code::{run, Value};

    fn run_plain(source: &str) -> Result<Value, data_code::LangError> {
        run(source)
    }

    fn assert_bool_result(source: &str, expected: bool) {
        let result = run_plain(source);
        match result {
            Ok(Value::Bool(b)) => assert_eq!(b, expected, "expected {}, got {}", expected, b),
            Ok(v) => panic!("expected Bool({}), got {:?}", expected, v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    #[test]
    fn sha256_empty_length_32_bytes() {
        let source = r#"len(sha256("")) == 32"#;
        assert_bool_result(source, true);
    }

    #[test]
    fn hmac_sha256_with_bytes() {
        // Single-byte keys: build minimal byte buffers via read is messy; use random_bytes for key/data.
        let source = r#"
            let key = random_bytes(16)
            let data = random_bytes(8)
            len(hmac_sha256(key, data)) == 32
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn random_bytes_over_limit_raises() {
        let result = run_plain(r#"random_bytes(2000000)"#);
        assert!(
            result.is_err(),
            "expected runtime error for oversized random_bytes, got {:?}",
            result
        );
    }

    #[test]
    fn random_int_inclusive_range() {
        let source = r#"
            let ok = true
            for i in range(0, 50) {
                let x = random_int(3, 3)
                if x != 3 { ok = false }
            }
            ok
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn crypto_argon2_hash_and_verify() {
        let source = r#"
            from crypto import Argon2
            let h = Argon2.hash("mypassword")
            Argon2.verify("mypassword", h) and !Argon2.verify("other", h)
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn crypto_bcrypt_hash_and_verify() {
        let source = r#"
            from crypto import bcrypt
            let h = bcrypt.hash("mypassword")
            bcrypt.verify("mypassword", h) and !bcrypt.verify("other", h)
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn crypto_secure_compare_strings() {
        let source = r#"
            from crypto import secure_compare
            secure_compare("a", "a") and !secure_compare("a", "b")
        "#;
        assert_bool_result(source, true);
    }
}

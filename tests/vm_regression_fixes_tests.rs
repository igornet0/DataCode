//! Regression tests for rolling hash int overflow, array(set/view), slice assign, heapq class wrapper.

#[cfg(test)]
mod tests {
    use data_code::{run, Value};

    fn assert_bool(source: &str, expected: bool) {
        let result = run(source);
        match result {
            Ok(Value::Bool(b)) => assert_eq!(b, expected, "expr:\n{}", source),
            Ok(v) => panic!("expected Bool, got {:?}\n{}", v, source),
            Err(e) => panic!("error {:?}\n{}", e, source),
        }
    }

    fn assert_number(source: &str, expected: f64) {
        let result = run(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, expected, "expr:\n{}", source),
            Ok(v) => panic!("expected Number({}), got {:?}\n{}", expected, v, source),
            Err(e) => panic!("error {:?}\n{}", e, source),
        }
    }

    #[test]
    fn array_from_set_materializes() {
        let source = r#"
s = set([3, 1, 2])
a = array(s)
len(a) == 3 and 1 in a and 2 in a and 3 in a
"#;
        assert_bool(source, true);
    }

    #[test]
    fn array_from_slice_view_materializes() {
        let source = r#"
arr = [10, 20, 30]
copy = array(arr[:2])
len(copy) == 2 and copy[0] == 10 and copy[1] == 20
"#;
        assert_bool(source, true);
    }

    #[test]
    fn slice_assign_to_local_is_owned_copy() {
        let source = r#"
keys = [1, 2, 3, 4, 5]
left = keys[:2]
right = keys[2:]
keys = keys[:2]
keys.push(99)
len(left) == 2 and len(right) == 3 and len(keys) == 3 and keys[2] == 99
"#;
        assert_bool(source, true);
    }

    #[test]
    fn btree_split_style_slice_assign() {
        let source = r#"
full_keys = [10, 20, 5, 6, 12]
new_keys = full_keys[2:]
full_keys = full_keys[:2]
full_keys.push(7)
new_keys.push(17)
len(full_keys) == 3 and len(new_keys) == 4 and full_keys[0] == 10 and new_keys[0] == 5
"#;
        assert_bool(source, true);
    }

    #[test]
    fn int_mul_large_no_i32_overflow() {
        let source = r#"
base = 911382323
x = 588671872
(x * base) % 1000000007 == 432582718
"#;
        assert_bool(source, true);
    }
}

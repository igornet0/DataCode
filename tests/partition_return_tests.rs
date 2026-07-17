//! QuickSort partition return value regression.

use data_code::{run, Value};

fn assert_number(source: &str, expected: f64) {
    match run(source) {
        Ok(v) if v.as_ieee_f64() == Some(expected) => {}
        Ok(v) => panic!("expected numeric({}), got {:?}", expected, v),
        Err(e) => panic!("error: {:?}", e),
    }
}

#[test]
fn partition_returns_index_not_null() {
    let source = r#"
        fn partition(arr, lo, hi) {
            pivot = arr[hi]
            i = lo
            for j in range(lo, hi) {
                if arr[j] <= pivot {
                    arr[i], arr[j] = arr[j], arr[i]
                    i = i + 1
                }
            }
            arr[i], arr[hi] = arr[hi], arr[i]
            return i
        }
        partition([10, 7, 8, 9, 1, 5], 0, 5)
    "#;
    assert_number(source, 1.0);
}

#[test]
fn typed_partition_returns_int() {
    let source = r#"
        fn partition(arr: list[int], lo: int, hi: int) -> int {
            pivot = arr[hi]
            i = lo
            for j in range(lo, hi) {
                if arr[j] <= pivot {
                    arr[i], arr[j] = arr[j], arr[i]
                    i = i + 1
                }
            }
            arr[i], arr[hi] = arr[hi], arr[i]
            return i
        }
        partition([10, 7, 8, 9, 1, 5], 0, 5)
    "#;
    assert_number(source, 1.0);
}

#[test]
fn full_quick_sort_one_array() {
    let source = r#"
        fn partition(arr: list[int], lo: int, hi: int) -> int {
            pivot = arr[hi]
            i = lo
            for j in range(lo, hi) {
                if arr[j] <= pivot {
                    arr[i], arr[j] = arr[j], arr[i]
                    i = i + 1
                }
            }
            arr[i], arr[hi] = arr[hi], arr[i]
            return i
        }
        fn quick_sort_inplace(arr: list[int], lo: int, hi: int) {
            if lo >= hi: return
            p = partition(arr, lo, hi)
            quick_sort_inplace(arr, lo, p - 1)
            quick_sort_inplace(arr, p + 1, hi)
        }
        fn quick_sort(arr: list[int]) -> list[int] {
            a = arr.clone()
            if len(a) > 0: quick_sort_inplace(a, 0, len(a) - 1)
            return a
        }
        data = [10, 7, 8, 9, 1, 5]
        got = quick_sort(data)
        expected = sort(data.clone())
        got == expected
    "#;
    match run(source) {
        Ok(Value::Bool(true)) => {}
        Ok(v) => panic!("expected quick_sort to match sort(), got {:?}", v),
        Err(e) => panic!("error: {:?}", e),
    }
}

#[test]
fn quick_sort_with_debug_prints_like_example() {
    let source = r#"
        fn partition(arr: list[int], lo: int, hi: int) -> int {
            pivot = arr[hi]
            i = lo
            print(arr)
            for j in range(lo, hi) {
                if arr[j] <= pivot {
                    arr[i], arr[j] = arr[j], arr[i]
                    print(arr)
                    i = i + 1
                    print()
                }
            }
            arr[i], arr[hi] = arr[hi], arr[i]
            return i
        }
        fn quick_sort_inplace(arr: list[int], lo: int, hi: int) {
            if lo >= hi: return
            p = partition(arr, lo, hi)
            print("${p=}")
            quick_sort_inplace(arr, lo, p - 1)
            quick_sort_inplace(arr, p + 1, hi)
        }
        fn quick_sort(arr: list[int]) -> list[int] {
            a = arr.clone()
            if len(a) > 0: quick_sort_inplace(a, 0, len(a) - 1)
            return a
        }
        data = [10, 7, 8, 9, 1, 5]
        got = quick_sort(data)
        expected = sort(data.clone())
        got == expected
    "#;
    match run(source) {
        Ok(Value::Bool(true)) => {}
        Ok(v) => panic!("expected quick_sort to match sort(), got {:?}", v),
        Err(e) => panic!("error: {:?}", e),
    }
}

#[test]
fn quick_sort_inplace_early_return_returns_partition() {
    let source = r#"
        fn partition(arr: list[int], lo: int, hi: int) -> int {
            pivot = arr[hi]
            i = lo
            print(arr)
            for j in range(lo, hi) {
                if arr[j] <= pivot {
                    arr[i], arr[j] = arr[j], arr[i]
                    print(arr)
                    i = i + 1
                    print()
                }
            }
            arr[i], arr[hi] = arr[hi], arr[i]
            return i
        }
        fn quick_sort_inplace(arr: list[int], lo: int, hi: int) {
            if lo >= hi: return
            p = partition(arr, lo, hi)
            return p
        }
        quick_sort_inplace([10, 7, 8, 9, 1, 5], 0, 5)
    "#;
    assert_number(source, 1.0);
}

#[test]
fn partition_called_from_void_without_early_return() {
    let source = r#"
        fn partition(arr: list[int], lo: int, hi: int) -> int {
            return 1
        }
        fn quick_sort_inplace(arr: list[int], lo: int, hi: int) {
            p = partition(arr, lo, hi)
            return p
        }
        quick_sort_inplace([10, 7, 8, 9, 1, 5], 0, 5)
    "#;
    assert_number(source, 1.0);
}

#[test]
fn disasm_quick_sort_inplace_with_early_return() {
    use data_code::compile;
    let source = r#"
        fn partition(arr: list[int], lo: int, hi: int) -> int { return 1 }
        fn quick_sort_inplace(arr: list[int], lo: int, hi: int) {
            if lo >= hi: return
            p = partition(arr, lo, hi)
            return p
        }
        quick_sort_inplace([1,2], 0, 1)
    "#;
    let (_chunk, funcs) = compile(source).unwrap();
    for f in &funcs {
        if f.name.contains("quick_sort") {
            eprintln!("{}", f.chunk.disassemble(&f.name));
        }
    }
}

// Тесты импорта переменных модуля (`from poizen_name import NAMES, SUFFIXES`).
// Фикстуры: tests/import_fixtures/poizen.
//
// Баг: функция модуля читала глобальную переменную по имени из таблицы глобалов
// вызывающего скрипта. Если в главном скрипте была переменная с тем же именем
// (NAMES, SUFFIXES, ...), функция модуля видела её вместо своей. Запись `global X = ...`
// внутри функции модуля так же перезаписывала одноимённую переменную главного скрипта.
//
// Баг 2: переменная модуля при каждом чтении из функции модуля копировалась из пространства
// имён модуля в новую ячейку хранилища. `push(ITEMS, x)`, `ITEMS[i] = x`, `DICT[k] = v`
// изменяли временную копию и терялись; каждое чтение стоило O(len).

#[cfg(test)]
mod tests {
    use data_code::common::numeric::IntValue;
    use data_code::{run_with_base_path, Value};
    use std::path::PathBuf;

    fn fixtures_dir() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tests");
        path.push("import_fixtures");
        path.push("poizen");
        path
    }

    fn run(source: &str) -> Value {
        match run_with_base_path(source, fixtures_dir().as_path()) {
            Ok(v) => v,
            Err(e) => panic!("error: {:?}", e),
        }
    }

    fn assert_number(source: &str, expected: f64) {
        match run(source) {
            Value::Int(IntValue::Finite(i)) => assert_eq!(i as f64, expected),
            Value::Float(f) => assert_eq!(f.as_raw_f64(), expected),
            Value::Number(n) => assert!(
                (n - expected).abs() < 1e-10,
                "expected {}, got {}",
                expected,
                n
            ),
            v => panic!("expected Number({}), got {:?}", expected, v),
        }
    }

    fn assert_string(source: &str, expected: &str) {
        match run(source) {
            Value::String(s) => assert_eq!(s, expected),
            v => panic!("expected String('{}'), got {:?}", expected, v),
        }
    }

    // ========== Прямой импорт списков ==========

    #[test]
    fn test_from_import_lists_top_level() {
        assert_number(
            r#"from poizen_name import NAMES, SUFFIXES
len(NAMES) * 10 + len(SUFFIXES)"#,
            32.0,
        );
    }

    #[test]
    fn test_from_import_lists_used_in_function() {
        assert_number(
            r#"from poizen_name import NAMES, SUFFIXES
fn create_poizen() {
    return len(NAMES) * 10 + len(SUFFIXES)
}
create_poizen()"#,
            32.0,
        );
    }

    #[test]
    fn test_from_import_list_element() {
        assert_string(
            r#"from poizen_name import NAMES
NAMES[1]"#,
            "Белладонна",
        );
    }

    #[test]
    fn test_import_module_attribute_lists() {
        assert_number(
            r#"import poizen_name
len(poizen_name.NAMES) * 10 + len(poizen_name.SUFFIXES)"#,
            32.0,
        );
    }

    // ========== Функция модуля, использующая импортированные списки ==========

    #[test]
    fn test_module_function_sees_imported_lists() {
        assert_number(
            r#"from poizen import create_poizen
create_poizen()"#,
            32.0,
        );
    }

    #[test]
    fn test_module_function_sees_imported_lists_via_namespace() {
        assert_number(
            r#"import poizen
poizen.create_poizen()"#,
            32.0,
        );
    }

    #[test]
    fn test_module_and_caller_import_same_lists() {
        assert_number(
            r#"from poizen_name import NAMES, SUFFIXES
from poizen import create_poizen
create_poizen() + len(NAMES) * 100"#,
            332.0,
        );
    }

    // ========== Изоляция: одноимённые переменные в главном скрипте ==========

    #[test]
    fn test_module_function_ignores_caller_vars_with_same_name() {
        assert_number(
            r#"NAMES = [1]
SUFFIXES = [1, 2, 3, 4, 5, 6, 7, 8]
from poizen import create_poizen
create_poizen()"#,
            32.0,
        );
    }

    #[test]
    fn test_module_function_ignores_caller_vars_defined_after_import() {
        assert_number(
            r#"from poizen import create_poizen
NAMES = 5
SUFFIXES = "abc"
create_poizen()"#,
            32.0,
        );
    }

    #[test]
    fn test_module_function_string_from_list_with_caller_shadow() {
        assert_string(
            r#"NAMES = ["host"]
from poizen import first_name
first_name()"#,
            "Аконит",
        );
    }

    #[test]
    fn test_caller_vars_not_overwritten_by_import() {
        assert_number(
            r#"NAMES = [1]
from poizen import create_poizen
create_poizen()
len(NAMES)"#,
            1.0,
        );
    }

    // ========== Изоляция: запись глобала внутри функции модуля ==========

    #[test]
    fn test_module_global_write_visible_in_module() {
        assert_number(
            r#"from counter import bump, get_count
bump()
bump()
get_count()"#,
            2.0,
        );
    }

    #[test]
    fn test_module_global_write_does_not_leak_into_caller() {
        assert_number(
            r#"COUNT = 100
from counter import bump, get_count
bump()
bump()
COUNT * 1000 + get_count()"#,
            100002.0,
        );
    }

    // ========== Изменение данных модуля на месте (inventory.dc) ==========

    #[test]
    fn test_module_push_persists() {
        assert_number(
            r#"from inventory import add_item, items_count
add_item("a")
add_item("b")
items_count()"#,
            3.0,
        );
    }

    #[test]
    fn test_module_push_returns_growing_len() {
        assert_number(
            r#"from inventory import add_item
add_item("a") * 10 + add_item("b")"#,
            23.0,
        );
    }

    #[test]
    fn test_module_index_assign_persists() {
        assert_string(
            r#"from inventory import rename_first, first_item
rename_first("эликсир")
first_item()"#,
            "эликсир",
        );
    }

    #[test]
    fn test_module_nested_index_assign_persists() {
        assert_number(
            r#"from inventory import bump_cell
bump_cell()
bump_cell()"#,
            203.0,
        );
    }

    #[test]
    fn test_module_dict_assign_persists() {
        assert_number(
            r#"from inventory import set_price, price_of
set_price("яд", 25)
set_price("антидот", 40)
price_of("яд") + price_of("зелье")"#,
            35.0,
        );
    }

    #[test]
    fn test_module_rebind_then_push() {
        assert_number(
            r#"from inventory import add_item, reset_items, items_count
add_item("a")
reset_items()
add_item("b")
items_count()"#,
            2.0,
        );
    }

    #[test]
    fn test_module_local_alias_push_persists() {
        assert_number(
            r#"from inventory import alias_push, items_count
alias_push("a")
alias_push("b")
items_count()"#,
            3.0,
        );
    }

    #[test]
    fn test_module_push_via_namespace_import() {
        assert_number(
            r#"import inventory
from inventory import add_item
add_item("a")
add_item("b")
inventory.items_count()"#,
            3.0,
        );
    }

    #[test]
    fn test_module_push_does_not_touch_caller_var() {
        assert_number(
            r#"ITEMS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
from inventory import add_item, items_count
add_item("a")
add_item("b")
len(ITEMS) * 100 + items_count()"#,
            903.0,
        );
    }

    #[test]
    fn test_module_large_fill_and_iterate() {
        // 0 + 1 + ... + 4999 = 12497500; без стабильной ячейки чтение было O(len) и push терялся.
        assert_number(
            r#"from inventory import fill, sum_numbers
fill(5000)
sum_numbers()"#,
            12497500.0,
        );
    }
}

// push внутри функции: переменная цикла `range` живёт в call-arena; без продвижения
// при записи в глобальный массив после возврата из функции элемент читался как null.
#[cfg(test)]
mod push_arena_tests {
    use data_code::common::numeric::IntValue;
    use data_code::{run, Value};

    #[test]
    fn test_push_range_var_into_global_array_from_function() {
        let result = run(r#"A = ["s"]
fn fill(n) {
    for i in range(n) {
        push(A, i)
    }
}
fill(3)
A[1] + A[2] + A[3]"#);
        match result {
            Ok(Value::Int(IntValue::Finite(n))) => assert_eq!(n, 3),
            Ok(Value::Float(f)) => assert_eq!(f.as_raw_f64(), 3.0),
            Ok(Value::Number(n)) => assert_eq!(n, 3.0),
            other => panic!("expected 3, got {:?}", other),
        }
    }
}

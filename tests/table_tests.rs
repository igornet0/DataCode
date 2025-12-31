// Тесты для работы с таблицами в DataCode
// Тестируем: создание таблиц, доступ к данным, загрузка из файлов, операции с таблицами

#[cfg(test)]
mod tests {
    use data_code::{run, Value};
    use std::path::PathBuf;

    // Вспомогательная функция для проверки результата выполнения
    fn run_and_get_result(source: &str) -> Result<Value, data_code::LangError> {
        run(source)
    }

    // Вспомогательная функция для проверки числового результата
    fn assert_number_result(source: &str, expected: f64) {
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => {
                assert_eq!(n, expected, "Expected {}, got {}", expected, n);
            }
            Ok(v) => panic!("Expected Number({}), got {:?}", expected, v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    // Вспомогательная функция для проверки строкового результата
    fn assert_string_result(source: &str, expected: &str) {
        let result = run_and_get_result(source);
        match result {
            Ok(Value::String(s)) => {
                assert_eq!(s, expected, "Expected '{}', got '{}'", expected, s);
            }
            Ok(v) => panic!("Expected String('{}'), got {:?}", expected, v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    // Вспомогательная функция для получения пути к тестовым данным
    fn get_test_data_path(filename: &str) -> String {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tests");
        path.push("test_data");
        path.push(filename);
        path.to_string_lossy().to_string()
    }

    // ========== 1. Тесты создания таблиц ==========

    #[test]
    fn test_table_creation_basic() {
        // Создание таблицы из двумерного массива без заголовков
        let source = r#"
            let data = [[1, 25], [2, 30], [3, 35]]
            let my_table = table(data)
            len(my_table["Column_0"])
        "#;
        // Проверяем, что таблица создана и имеет 3 строки
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_table_creation_with_headers() {
        // Создание таблицы с заголовками
        let source = r#"
            let data = [[1, "Alice", 28], [2, "Bob", 35], [3, "Charlie", 42]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            len(my_table["id"])
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_table_creation_numeric_data() {
        // Таблица с числовыми данными
        let source = r#"
            let data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            let headers = ["a", "b", "c"]
            let my_table = table(data, headers)
            len(my_table["a"])
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_table_creation_string_data() {
        // Таблица со строковыми данными
        let source = r#"
            let data = [["Alice", "Engineer"], ["Bob", "Manager"], ["Charlie", "Designer"]]
            let headers = ["name", "role"]
            let my_table = table(data, headers)
            len(my_table["name"])
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_table_creation_mixed_data() {
        // Таблица со смешанными типами данных
        let source = r#"
            let data = [[1, "Active", true], [2, "Inactive", false], [3, "Pending", true]]
            let headers = ["id", "status", "enabled"]
            let my_table = table(data, headers)
            len(my_table["id"])
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_table_creation_with_nulls() {
        // Таблица с null значениями
        let source = r#"
            let data = [[1, "Alice", 28], [2, null, 35], [3, "Charlie", null]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            len(my_table["id"])
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_table_creation_single_row() {
        // Таблица с одной строкой
        let source = r#"
            let data = [[1, "Alice", 28]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            len(my_table["id"])
        "#;
        assert_number_result(source, 1.0);
    }

    #[test]
    fn test_table_creation_single_column() {
        // Таблица с одной колонкой
        let source = r#"
            let data = [[1], [2], [3]]
            let headers = ["value"]
            let my_table = table(data, headers)
            len(my_table["value"])
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_table_creation_empty_data() {
        // Пустая таблица
        let source = r#"
            let data = []
            let headers = ["id", "name"]
            let my_table = table(data, headers)
            len(my_table["id"])
        "#;
        assert_number_result(source, 0.0);
    }

    // ========== 2. Тесты доступа к данным таблицы ==========

    #[test]
    fn test_table_column_access_by_name() {
        // Доступ к колонке по имени
        let source = r#"
            let data = [[1, "Alice", 28], [2, "Bob", 35], [3, "Charlie", 42]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            let names = my_table["name"]
            len(names)
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_table_column_length() {
        // Проверка длины колонки
        let source = r#"
            let data = [[1, "Alice"], [2, "Bob"], [3, "Charlie"], [4, "David"]]
            let headers = ["id", "name"]
            let my_table = table(data, headers)
            len(my_table["name"])
        "#;
        assert_number_result(source, 4.0);
    }

    #[test]
    fn test_table_multiple_columns() {
        // Доступ к нескольким колонкам
        let source = r#"
            let data = [[1, "Alice", 28], [2, "Bob", 35]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            let ids = my_table["id"]
            let ages = my_table["age"]
            len(ids) + len(ages)
        "#;
        assert_number_result(source, 4.0);
    }

    #[test]
    fn test_table_column_values() {
        // Проверка значений в колонке
        let source = r#"
            let data = [[1, "Alice"], [2, "Bob"], [3, "Charlie"]]
            let headers = ["id", "name"]
            let my_table = table(data, headers)
            let ids = my_table["id"]
            ids[0]
        "#;
        assert_number_result(source, 1.0);
    }

    // ========== 3. Тесты загрузки таблиц из файлов ==========

    #[test]
    fn test_load_table_from_csv() {
        // Загрузка CSV файла
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let csv_table = read_file("{}")
            len(csv_table["Name"])
            "#,
            csv_path
        );
        // sample.csv содержит 5 строк данных (плюс заголовок)
        assert_number_result(&source, 5.0);
    }

    #[test]
    fn test_load_table_from_csv_headers() {
        // Проверка заголовков загруженной таблицы
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let csv_table = read_file("{}")
            let names = csv_table["Name"]
            names[0]
            "#,
            csv_path
        );
        assert_string_result(&source, "John Doe");
    }

    #[test]
    fn test_load_table_from_csv_column_access() {
        // Доступ к колонкам загруженной таблицы
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let csv_table = read_file("{}")
            let ages = csv_table["Age"]
            ages[0]
            "#,
            csv_path
        );
        assert_number_result(&source, 30.0);
    }

    #[test]
    fn test_load_table_from_csv_multiple_columns() {
        // Доступ к нескольким колонкам загруженной таблицы
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let csv_table = read_file("{}")
            let names = csv_table["Name"]
            let cities = csv_table["City"]
            len(names) + len(cities)
            "#,
            csv_path
        );
        assert_number_result(&source, 10.0);
    }

    #[test]
    fn test_load_table_from_xlsx() {
        // Загрузка XLSX файла
        let xlsx_path = get_test_data_path("sample.xlsx");
        let source = format!(
            r#"
            let xlsx_table = read_file("{}")
            len(xlsx_table)
            "#,
            xlsx_path
        );
        // Проверяем, что файл загружается (может вернуть таблицу или массив)
        let result = run_and_get_result(&source);
        assert!(result.is_ok(), "Failed to load XLSX file");
    }

    #[test]
    fn test_load_table_nonexistent_file() {
        // Обработка несуществующего файла
        let source = r#"
            let table = read_file("nonexistent_file.csv")
        "#;
        let result = run_and_get_result(source);
        // Должна быть ошибка или null
        assert!(result.is_err() || matches!(result, Ok(Value::Null)));
    }

    // ========== 4. Тесты операций с таблицами ==========

    #[test]
    fn test_table_info() {
        // Получение информации о таблице
        let source = r#"
            let data = [[1, "Alice", 28], [2, "Bob", 35], [3, "Charlie", 42]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            table_info(my_table)
        "#;
        // table_info может вернуть null или строку с информацией
        let result = run_and_get_result(source);
        assert!(result.is_ok(), "table_info should execute without error");
    }

    #[test]
    fn test_table_head() {
        // Получение первых n строк
        let source = r#"
            let data = [[1, "Alice"], [2, "Bob"], [3, "Charlie"], [4, "David"], [5, "Eve"]]
            let headers = ["id", "name"]
            let my_table = table(data, headers)
            let head_table = table_head(my_table, 2)
            len(head_table["id"])
        "#;
        assert_number_result(source, 2.0);
    }

    #[test]
    fn test_table_tail() {
        // Получение последних n строк
        let source = r#"
            let data = [[1, "Alice"], [2, "Bob"], [3, "Charlie"], [4, "David"], [5, "Eve"]]
            let headers = ["id", "name"]
            let my_table = table(data, headers)
            let tail_table = table_tail(my_table, 2)
            len(tail_table["id"])
        "#;
        assert_number_result(source, 2.0);
    }

    #[test]
    fn test_table_select() {
        // Выбор определенных колонок
        let source = r#"
            let data = [[1, "Alice", 28], [2, "Bob", 35], [3, "Charlie", 42]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            let selected = table_select(my_table, ["name", "age"])
            len(selected["name"])
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_table_sort() {
        // Сортировка таблицы
        let source = r#"
            let data = [[3, "Charlie", 42], [1, "Alice", 28], [2, "Bob", 35]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            let sorted = table_sort(my_table, "age")
            sorted["age"][0]
        "#;
        // После сортировки по возрасту, первый должен быть 28
        assert_number_result(source, 28.0);
    }

    #[test]
    fn test_table_sort_descending() {
        // Сортировка по убыванию
        let source = r#"
            let data = [[1, "Alice", 28], [2, "Bob", 35], [3, "Charlie", 42]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            let sorted = table_sort(my_table, "age", false)
            sorted["age"][0]
        "#;
        // После сортировки по убыванию, первый должен быть 42
        assert_number_result(source, 42.0);
    }

    #[test]
    fn test_table_where() {
        // Фильтрация таблицы
        let source = r#"
            let data = [[1, "Alice", 28], [2, "Bob", 35], [3, "Charlie", 42]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            let filtered = table_where(my_table, "age", ">", 30)
            len(filtered["age"])
        "#;
        // Должно быть 2 строки с возрастом > 30
        assert_number_result(source, 2.0);
    }

    #[test]
    fn test_table_where_equals() {
        // Фильтрация по равенству
        let source = r#"
            let data = [[1, "Alice", 28], [2, "Bob", 35], [3, "Charlie", 28]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            let filtered = table_where(my_table, "age", "==", 28)
            len(filtered["age"])
        "#;
        // Должно быть 2 строки с возрастом == 28
        assert_number_result(source, 2.0);
    }

    // ========== 5. Интеграционные тесты ==========

    #[test]
    fn test_table_creation_filter_sort_select() {
        // Комплексный сценарий: создание → фильтрация → сортировка → выборка
        let source = r#"
            let data = [[1, "Alice", 28, 50000], [2, "Bob", 35, 60000], [3, "Charlie", 42, 70000], [4, "David", 25, 45000]]
            let headers = ["id", "name", "age", "salary"]
            let my_table = table(data, headers)
            let filtered = table_where(my_table, "age", ">", 30)
            let sorted = table_sort(filtered, "salary")
            let selected = table_select(sorted, ["name", "salary"])
            len(selected["name"])
        "#;
        // После фильтрации (age > 30) должно быть 2 строки
        assert_number_result(source, 2.0);
    }

    #[test]
    fn test_table_load_process_create() {
        // Загрузка из файла → обработка → создание новой таблицы
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let csv_table = read_file("{}")
            let names = csv_table["Name"]
            let ages = csv_table["Age"]
            let processed_data = []
            let i = 0
            while i < len(names) {{
                let row = [names[i], ages[i]]
                push(processed_data, row)
                i = i + 1
            }}
            let new_table = table(processed_data, ["name", "age"])
            len(new_table["name"])
            "#,
            csv_path
        );
        assert_number_result(&source, 5.0);
    }

    #[test]
    fn test_table_multiple_tables() {
        // Работа с несколькими таблицами одновременно
        let source = r#"
            let data1 = [[1, "Alice"], [2, "Bob"]]
            let data2 = [[10, "Charlie"], [20, "David"]]
            let headers = ["id", "name"]
            let table1 = table(data1, headers)
            let table2 = table(data2, headers)
            len(table1["id"]) + len(table2["id"])
        "#;
        assert_number_result(source, 4.0);
    }

    #[test]
    fn test_table_large_table() {
        // Создание большой таблицы (100+ строк)
        let source = r#"
            let data = []
            let i = 0
            while i < 100 {
                let row = [i, "Item_" + i, i * 2]
                push(data, row)
                i = i + 1
            }
            let headers = ["id", "name", "value"]
            let large_table = table(data, headers)
            len(large_table["id"])
        "#;
        assert_number_result(source, 100.0);
    }

    #[test]
    fn test_table_operations_on_large_table() {
        // Операции с большой таблицей
        let source = r#"
            let data = []
            let i = 0
            while i < 100 {
                let row = [i, "Item_" + i, i * 2]
                push(data, row)
                i = i + 1
            }
            let headers = ["id", "name", "value"]
            let large_table = table(data, headers)
            let filtered = table_where(large_table, "value", ">", 50)
            len(filtered["id"])
        "#;
        // Должно быть строк с value > 50 (начиная с id=26, value=52)
        assert_number_result(source, 74.0);
    }

    #[test]
    fn test_table_column_operations() {
        // Операции с колонками таблицы
        let source = r#"
            let data = [[1, "Alice", 28], [2, "Bob", 35], [3, "Charlie", 42]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            let ages = my_table["age"]
            let sum = 0
            let i = 0
            while i < len(ages) {
                sum = sum + ages[i]
                i = i + 1
            }
            sum
        "#;
        // Сумма возрастов: 28 + 35 + 42 = 105
        assert_number_result(source, 105.0);
    }

    #[test]
    fn test_table_nested_operations() {
        // Вложенные операции с таблицами
        let source = r#"
            let data = [[1, "Alice", 28], [2, "Bob", 35], [3, "Charlie", 42], [4, "David", 25]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            let filtered = table_where(my_table, "age", ">", 30)
            let sorted = table_sort(filtered, "age")
            let head = table_head(sorted, 1)
            head["age"][0]
        "#;
        // После фильтрации (age > 30) и сортировки, первая строка должна быть 35
        assert_number_result(source, 35.0);
    }

    #[test]
    fn test_table_create_from_performance_example() {
        // Тест на основе примера из simple_performance_test.dc
        let source = r#"
            let data = []
            let i = 0
            while i < 100 {
                let row = [i, "Item_" + i, i * 2]
                push(data, row)
                i = i + 1
            }
            let headers = ["id", "name", "value"]
            let test_table = table(data, headers)
            len(test_table["id"])
        "#;
        assert_number_result(source, 100.0);
    }

    #[test]
    fn test_table_column_access_performance_example() {
        // Тест доступа к колонкам на основе примера
        let source = r#"
            let data = []
            let i = 0
            while i < 100 {
                let row = [i, "Item_" + i, i * 2]
                push(data, row)
                i = i + 1
            }
            let headers = ["id", "name", "value"]
            let test_table = table(data, headers)
            let names = test_table["name"]
            let values = test_table["value"]
            len(names) + len(values)
        "#;
        assert_number_result(source, 200.0);
    }

    #[test]
    fn test_table_get_row() {
        let source = r#"
            let data = [[1, "Alice", 28], [2, "Bob", 35], [3, "Charlie", 42]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            let row = my_table.idx[1]
            row["name"]
        "#;

        assert_string_result(source, "Bob");
    }

    // ========== 6. Тесты свойств table.rows и table.columns ==========

    #[test]
    fn test_table_rows_property() {
        // Проверка что table.rows возвращает правильное количество строк
        let source = r#"
            let data = [[1, "Alice", 28], [2, "Bob", 35], [3, "Charlie", 42]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            let rows = my_table.rows
            len(rows)
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_table_rows_content() {
        // Проверка содержимого строк
        let source = r#"
            let data = [[1, "Alice", 28], [2, "Bob", 35]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            let rows = my_table.rows
            let first_row = rows[0]
            first_row[0]
        "#;
        assert_number_result(source, 1.0);
    }

    #[test]
    fn test_table_rows_multiple_elements() {
        // Проверка доступа к нескольким элементам строк
        let source = r#"
            let data = [[1, "Alice", 28], [2, "Bob", 35]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            let rows = my_table.rows
            let first_row = rows[0]
            let second_row = rows[1]
            first_row[1] + second_row[1]
        "#;
        // Проверяем что можем получить доступ к элементам (хотя конкатенация строк может не работать)
        // Проверяем что первая строка доступна
        let result = run_and_get_result(source);
        assert!(result.is_ok(), "Should access rows and elements");
    }

    #[test]
    fn test_table_columns_property() {
        // Проверка что table.columns возвращает правильные имена колонок
        let source = r#"
            let data = [[1, "Alice", 28], [2, "Bob", 35]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            let columns = my_table.columns
            columns[0]
        "#;
        assert_string_result(source, "id");
    }

    #[test]
    fn test_table_columns_count() {
        // Проверка количества колонок
        let source = r#"
            let data = [[1, "Alice", 28], [2, "Bob", 35]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            let columns = my_table.columns
            len(columns)
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_table_columns_all_names() {
        // Проверка всех имен колонок
        let source = r#"
            let data = [[1, "Alice", 28], [2, "Bob", 35]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            let columns = my_table.columns
            columns[1]
        "#;
        assert_string_result(source, "name");
    }

    #[test]
    fn test_table_rows_and_columns_together() {
        // Комплексный тест использования обоих свойств
        let source = r#"
            let data = [[1, "Alice", 28], [2, "Bob", 35], [3, "Charlie", 42]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            let rows = my_table.rows
            let columns = my_table.columns
            len(rows) + len(columns)
        "#;
        // 3 строки + 3 колонки = 6
        assert_number_result(source, 6.0);
    }

    #[test]
    fn test_table_rows_empty_table() {
        // Проверка пустой таблицы
        let source = r#"
            let data = []
            let headers = ["id", "name"]
            let my_table = table(data, headers)
            let rows = my_table.rows
            len(rows)
        "#;
        assert_number_result(source, 0.0);
    }

    #[test]
    fn test_table_columns_empty_table() {
        // Проверка колонок пустой таблицы
        let source = r#"
            let data = []
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            let columns = my_table.columns
            len(columns)
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_table_rows_single_row() {
        // Проверка таблицы с одной строкой
        let source = r#"
            let data = [[1, "Alice", 28]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            let rows = my_table.rows
            let row = rows[0]
            row[2]
        "#;
        assert_number_result(source, 28.0);
    }

    // ========== JOIN Operations Tests ==========

    #[test]
    fn test_inner_join_basic() {
        let source = r#"
            let users = table([[1, "Alice"], [2, "Bob"], [3, "Charlie"]], ["id", "name"])
            let orders = table([[1, 100], [1, 200], [3, 300]], ["user_id", "amount"])
            let result = inner_join(users, orders, "id", "user_id")
            len(result)
        "#;
        // Должно быть 3 строки: Alice с двумя заказами, Charlie с одним
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 3.0, "Expected 3 rows in inner join"),
            Ok(v) => panic!("Expected Number(3), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_left_join_basic() {
        let source = r#"
            let users = table([[1, "Alice"], [2, "Bob"], [3, "Charlie"]], ["id", "name"])
            let orders = table([[1, 100], [3, 300]], ["user_id", "amount"])
            let result = left_join(users, orders, "id", "user_id")
            len(result)
        "#;
        // Должно быть 3 строки: все пользователи, Bob без заказов
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 3.0, "Expected 3 rows in left join"),
            Ok(v) => panic!("Expected Number(3), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_cross_join_basic() {
        let source = r#"
            let table1 = table([[1], [2]], ["col1"])
            let table2 = table([["a"], ["b"]], ["col2"])
            let result = cross_join(table1, table2)
            len(result)
        "#;
        // Декартово произведение: 2 * 2 = 4 строки
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 4.0, "Expected 4 rows in cross join"),
            Ok(v) => panic!("Expected Number(4), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_semi_join_basic() {
        let source = r#"
            let users = table([[1, "Alice"], [2, "Bob"], [3, "Charlie"]], ["id", "name"])
            let orders = table([[1, 100], [3, 300]], ["user_id", "amount"])
            let result = semi_join(users, orders, "id", "user_id")
            len(result)
        "#;
        // Только пользователи с заказами: Alice и Charlie (2 строки)
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 2.0, "Expected 2 rows in semi join"),
            Ok(v) => panic!("Expected Number(2), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_anti_join_basic() {
        let source = r#"
            let users = table([[1, "Alice"], [2, "Bob"], [3, "Charlie"]], ["id", "name"])
            let orders = table([[1, 100], [3, 300]], ["user_id", "amount"])
            let result = anti_join(users, orders, "id", "user_id")
            len(result)
        "#;
        // Только пользователи без заказов: Bob (1 строка)
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 1.0, "Expected 1 row in anti join"),
            Ok(v) => panic!("Expected Number(1), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_zip_join_basic() {
        let source = r#"
            let table1 = table([[1], [2], [3]], ["col1"])
            let table2 = table([["a"], ["b"], ["c"]], ["col2"])
            let result = zip_join(table1, table2)
            len(result)
        "#;
        // Позиционное соединение: минимум из длин таблиц = 3
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 3.0, "Expected 3 rows in zip join"),
            Ok(v) => panic!("Expected Number(3), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_join_with_multiple_keys() {
        let source = r#"
            let table1 = table([[1, "A"], [2, "B"]], ["id", "region"])
            let table2 = table([[1, "A"], [2, "C"]], ["id", "region"])
            let result = inner_join(table1, table2, [["id", "id"], ["region", "region"]])
            len(result)
        "#;
        // Только первая строка совпадает по обоим ключам
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 1.0, "Expected 1 row with multiple keys"),
            Ok(v) => panic!("Expected Number(1), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    // ========== JOIN с пустыми таблицами ==========

    #[test]
    fn test_inner_join_empty_right() {
        let source = r#"
            let users = table([[1, "Alice"]], ["id", "name"])
            let orders = table([], ["user_id", "amount"])
            let result = inner_join(users, orders, "id", "user_id")
            len(result)
        "#;

        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 0.0, "Expected 0 rows in inner join with empty right table"),
            Ok(v) => panic!("Expected Number(0), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_left_join_empty_right() {
        let source = r#"
            let users = table([[1, "Alice"], [2, "Bob"]], ["id", "name"])
            let orders = table([], ["user_id", "amount"])
            let result = left_join(users, orders, "id", "user_id")
            len(result)
        "#;

        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 2.0, "Expected 2 rows in left join with empty right table"),
            Ok(v) => panic!("Expected Number(2), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_right_join_empty_left() {
        let source = r#"
            let users = table([], ["id", "name"])
            let orders = table([[1, 100]], ["user_id", "amount"])
            let result = right_join(users, orders, "id", "user_id")
            len(result)
        "#;

        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 1.0, "Expected 1 row in right join with empty left table"),
            Ok(v) => panic!("Expected Number(1), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    // ========== JOIN без совпадений ==========

    #[test]
    fn test_inner_join_no_matches() {
        let source = r#"
            let t1 = table([[1], [2]], ["id"])
            let t2 = table([[3], [4]], ["id"])
            let result = inner_join(t1, t2, "id", "id")
            len(result)
        "#;

        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 0.0, "Expected 0 rows in inner join with no matches"),
            Ok(v) => panic!("Expected Number(0), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_anti_join_all() {
        let source = r#"
            let t1 = table([[1], [2]], ["id"])
            let t2 = table([[3]], ["id"])
            let result = anti_join(t1, t2, "id", "id")
            len(result)
        "#;

        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 2.0, "Expected 2 rows in anti join when no matches"),
            Ok(v) => panic!("Expected Number(2), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    // ========== Поведение при дубликатах ==========

    #[test]
    fn test_inner_join_duplicates() {
        let source = r#"
            let t1 = table([[1], [1]], ["id"])
            let t2 = table([[1], [1], [1]], ["id"])
            let result = inner_join(t1, t2, "id", "id")
            len(result)
        "#;

        // 2 * 3 = 6
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 6.0, "Expected 6 rows in inner join with duplicates (2*3)"),
            Ok(v) => panic!("Expected Number(6), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_semi_join_no_duplicates() {
        let source = r#"
            let t1 = table([[1], [1], [2]], ["id"])
            let t2 = table([[1]], ["id"])
            let result = semi_join(t1, t2, "id", "id")
            len(result)
        "#;

        // id=1 есть → берём ВСЕ строки t1 с id=1
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 2.0, "Expected 2 rows in semi join (all rows with matching id)"),
            Ok(v) => panic!("Expected Number(2), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    // ========== NULL-значения ==========

    #[test]
    fn test_inner_join_with_null() {
        let source = r#"
            let t1 = table([[1], [null]], ["id"])
            let t2 = table([[1], [null]], ["id"])
            let result = inner_join(t1, t2, "id", "id")
            len(result)
        "#;

        // NULL != NULL
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 1.0, "Expected 1 row in inner join with null (NULL != NULL)"),
            Ok(v) => panic!("Expected Number(1), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    // ========== Конфликт имён колонок ==========

    #[test]
    fn test_join_column_name_conflict() {
        let source = r#"
            let t1 = table([[1]], ["id"])
            let t2 = table([[1]], ["id"])
            let result = inner_join(t1, t2, "id", "id")
            let columns = result.columns
            len(columns)
        "#;

        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => {
                // Должно быть как минимум 2 колонки (возможно с префиксами)
                assert!(n >= 2.0, "Expected at least 2 columns in join result");
            }
            Ok(v) => panic!("Expected Number, got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    // ========== Ошибки API ==========

    #[test]
    fn test_join_invalid_column() {
        let source = r#"
            let t1 = table([[1]], ["id"])
            let t2 = table([[1]], ["id"])
            inner_join(t1, t2, "foo", "id")
        "#;

        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for invalid column name");
    }

    // ========== ZIP JOIN edge cases ==========

    #[test]
    fn test_zip_join_different_lengths() {
        let source = r#"
            let t1 = table([[1], [2], [3]], ["id"])
            let t2 = table([[10]], ["val"])
            let result = zip_join(t1, t2)
            len(result)
        "#;

        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 1.0, "Expected 1 row in zip join with different lengths (min)"),
            Ok(v) => panic!("Expected Number(1), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    // ========== APPLY JOIN Tests ==========

    #[test]
    fn test_apply_join_basic() {
        // Базовый тест с функцией, возвращающей таблицу
        let source = r#"
            fn expand_row(row) {
                let id = row[0]
                let name = row[1]
                return table([[id * 10, name + "_1"], [id * 20, name + "_2"]], ["mult_id", "mult_name"])
            }
            let left = table([[1, "Alice"], [2, "Bob"]], ["id", "name"])
            let result = apply_join(left, expand_row)
            len(result)
        "#;
        // Должно быть 4 строки: каждая строка left умножается на 2 строки из функции
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 4.0, "Expected 4 rows in apply_join"),
            Ok(v) => panic!("Expected Number(4), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_apply_join_inner() {
        // Inner join семантика (по умолчанию) - строки с Null не включаются
        let source = r#"
            fn filter_row(row) {
                let id = row[0]
                if id == 1 {
                    return table([[id * 10]], ["mult_id"])
                }
            }
            let left = table([[1, "Alice"], [2, "Bob"]], ["id", "name"])
            let result = apply_join(left, filter_row, "inner")
            len(result)
        "#;
        // Должна быть только 1 строка (для id=1), строка с id=2 должна быть исключена
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 1.0, "Expected 1 row in inner apply_join"),
            Ok(v) => panic!("Expected Number(1), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_apply_join_left() {
        // Left join семантика - строки с Null включаются с NULL значениями
        let source = r#"
            fn filter_row(row) {
                let id = row[0]
                if id == 1 {
                    return table([[id * 10]], ["mult_id"])
                }
            }
            let left = table([[1, "Alice"], [2, "Bob"]], ["id", "name"])
            let result = apply_join(left, filter_row, "left")
            len(result)
        "#;
        // Должно быть 2 строки: одна с результатом функции, одна с NULLs
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 2.0, "Expected 2 rows in left apply_join"),
            Ok(v) => panic!("Expected Number(2), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_apply_join_empty_result() {
        // Функция возвращает пустую таблицу
        let source = r#"
            fn empty_result(row) {
                return table([], ["col"])
            }
            let left = table([[1, "Alice"]], ["id", "name"])
            let result = apply_join(left, empty_result)
            len(result)
        "#;
        // Должно быть 0 строк
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 0.0, "Expected 0 rows when function returns empty table"),
            Ok(v) => panic!("Expected Number(0), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_apply_join_multiple_rows() {
        // Функция возвращает несколько строк для одной входной строки
        let source = r#"
            fn expand(row) {
                let count = row[0]
                let data = []
                let i = 0
                while i < count {
                    data = push(data, [i, "item_" + str(i)])
                    i = i + 1
                }
                return table(data, ["idx", "item"])
            }
            let left = table([[3]], ["count"])
            let result = apply_join(left, expand)
            len(result)
        "#;
        // Должно быть 3 строки (функция возвращает 3 строки для count=3)
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 3.0, "Expected 3 rows from expand function"),
            Ok(v) => panic!("Expected Number(3), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    // ========== ASOF JOIN with BY Tests ==========

    #[test]
    fn test_asof_join_with_by_single_column() {
        // Группировка по одной колонке
        let source = r#"
            let left = table([[1, 10.0, "A"], [1, 20.0, "A"], [2, 15.0, "B"]], ["group", "time", "val"])
            let right = table([[1, 12.0, "X"], [1, 18.0, "Y"], [2, 14.0, "Z"]], ["group", "time", "data"])
            let result = asof_join(left, right, "time", "group")
            len(result)
        "#;
        // Должно быть 3 строки (по одной для каждой строки left)
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 3.0, "Expected 3 rows in asof_join with by"),
            Ok(v) => panic!("Expected Number(3), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_asof_join_with_by_multiple_columns() {
        // Группировка по нескольким колонкам
        let source = r#"
            let left = table([[1, "A", 10.0], [1, "B", 20.0]], ["id", "region", "time"])
            let right = table([[1, "A", 12.0], [1, "B", 18.0]], ["id", "region", "time"])
            let result = asof_join(left, right, "time", ["id", "region"])
            len(result)
        "#;
        // Должно быть 2 строки
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 2.0, "Expected 2 rows in asof_join with multiple by columns"),
            Ok(v) => panic!("Expected Number(2), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_asof_join_with_by_no_matches() {
        // Нет совпадений в группе
        let source = r#"
            let left = table([[1, 10.0], [2, 20.0]], ["group", "time"])
            let right = table([[1, 5.0]], ["group", "time"])
            let result = asof_join(left, right, "time", "group")
            len(result)
        "#;
        // Должно быть 2 строки (обе из left, вторая с NULLs справа)
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 2.0, "Expected 2 rows (one matched, one with NULLs)"),
            Ok(v) => panic!("Expected Number(2), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    // ========== SUFFIXES Tests ==========

    #[test]
    fn test_suffixes_basic() {
        let source = r#"
            let users = table([[1, "Alice"], [2, "Bob"]], ["id", "name"])
            let orders = table([[1, 100], [2, 200]], ["id", "amount"])
            let joined = left_join(orders, users, "id", "id")
            let result = joined.suffixes("_o", "_u")
            let columns = result.columns
            len(columns)
        "#;

        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert!(n >= 3.0, "Expected at least 3 columns after suffixes"),
            Ok(v) => panic!("Expected Number, got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_suffixes_column_names() {
        let source = r#"
            let users = table([[1, "Alice"]], ["id", "name"])
            let orders = table([[1, 100]], ["id", "amount"])
            let joined = left_join(orders, users, "id", "id")
            let result = joined.suffixes("_o", "_u")
            let columns = result.columns
            columns[0]
        "#;

        // Проверяем, что первая колонка имеет суффикс _o
        let result = run_and_get_result(source);
        match result {
            Ok(Value::String(s)) => {
                assert!(s.contains("_o") || s == "id_o", "Expected column name with _o suffix, got {}", s);
            }
            Ok(v) => panic!("Expected String, got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_suffixes_left_join() {
        let source = r#"
            let users = table([[1, "Alice"], [2, "Bob"]], ["id", "name"])
            let orders = table([[1, 100]], ["id", "amount"])
            let joined = left_join(orders, users, "id", "id")
            let result = joined.suffixes("_order", "_user")
            len(result)
        "#;

        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 1.0, "Expected 1 row after left join with suffixes"),
            Ok(v) => panic!("Expected Number(1), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_suffixes_inner_join() {
        let source = r#"
            let users = table([[1, "Alice"], [2, "Bob"]], ["id", "name"])
            let orders = table([[1, 100], [2, 200]], ["id", "amount"])
            let joined = inner_join(users, orders, "id", "id")
            let result = joined.suffixes("_u", "_o")
            len(result)
        "#;

        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 2.0, "Expected 2 rows after inner join with suffixes"),
            Ok(v) => panic!("Expected Number(2), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_suffixes_no_conflicts() {
        let source = r#"
            let users = table([[1, "Alice"]], ["id", "name"])
            let orders = table([[1, 100]], ["order_id", "amount"])
            let joined = left_join(orders, users, "order_id", "id")
            let result = joined.suffixes("_o", "_u")
            let columns = result.columns
            len(columns)
        "#;

        // Если нет конфликтов, колонки без префиксов должны остаться без изменений
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert!(n >= 3.0, "Expected at least 3 columns"),
            Ok(v) => panic!("Expected Number, got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_suffixes_chain_with_select() {
        let source = r#"
            let users = table([[1, "Alice"], [2, "Bob"]], ["id", "name"])
            let orders = table([[1, 100], [2, 200]], ["id", "amount"])
            let joined = left_join(orders, users, "id", "id")
            let with_suffixes = joined.suffixes("_o", "_u")
            let selected = table_select(with_suffixes, ["id_o", "name"])
            len(selected)
        "#;

        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 2.0, "Expected 2 rows after chain operations"),
            Ok(v) => panic!("Expected Number(2), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_suffixes_empty_strings() {
        let source = r#"
            let users = table([[1, "Alice"]], ["id", "name"])
            let orders = table([[1, 100]], ["id", "amount"])
            let joined = left_join(orders, users, "id", "id")
            let result = joined.suffixes("", "")
            len(result)
        "#;

        // Пустые суффиксы должны работать (просто убрать префиксы)
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 1.0, "Expected 1 row with empty suffixes"),
            Ok(v) => panic!("Expected Number(1), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_suffixes_multiple_conflicts() {
        let source = r#"
            let t1 = table([[1, "A", 10]], ["id", "name", "value"])
            let t2 = table([[1, "B", 20]], ["id", "name", "value"])
            let joined = inner_join(t1, t2, "id", "id")
            let result = joined.suffixes("_left", "_right")
            let columns = result.columns
            len(columns)
        "#;

        // Должно быть 6 колонок: id_left, name_left, value_left, id_right, name_right, value_right
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 6.0, "Expected 6 columns with multiple conflicts"),
            Ok(v) => panic!("Expected Number(6), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_suffixes_right_join() {
        let source = r#"
            let users = table([[1, "Alice"]], ["id", "name"])
            let orders = table([[1, 100], [2, 200]], ["id", "amount"])
            let joined = right_join(users, orders, "id", "id")
            let result = joined.suffixes("_u", "_o")
            len(result)
        "#;

        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 2.0, "Expected 2 rows after right join with suffixes"),
            Ok(v) => panic!("Expected Number(2), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_suffixes_full_join() {
        let source = r#"
            let users = table([[1, "Alice"], [2, "Bob"]], ["id", "name"])
            let orders = table([[1, 100], [3, 300]], ["id", "amount"])
            let joined = full_join(users, orders, "id", "id")
            let result = joined.suffixes("_user", "_order")
            len(result)
        "#;

        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 3.0, "Expected 3 rows after full join with suffixes"),
            Ok(v) => panic!("Expected Number(3), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_suffixes_preserves_data() {
        let source = r#"
            let users = table([[1, "Alice"]], ["id", "name"])
            let orders = table([[1, 100]], ["id", "amount"])
            let joined = left_join(orders, users, "id", "id")
            let result = joined.suffixes("_o", "_u")
            let id_o_col = result["id_o"]
            id_o_col[0]
        "#;

        // Проверяем, что данные сохраняются после применения suffixes
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 1.0, "Expected id_o[0] to be 1"),
            Ok(v) => panic!("Expected Number(1), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_suffixes_long_suffixes() {
        let source = r#"
            let users = table([[1, "Alice"]], ["id", "name"])
            let orders = table([[1, 100]], ["id", "amount"])
            let joined = left_join(orders, users, "id", "id")
            let result = joined.suffixes("_orders_table", "_users_table")
            len(result)
        "#;

        // Проверяем работу с длинными суффиксами
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 1.0, "Expected 1 row with long suffixes"),
            Ok(v) => panic!("Expected Number(1), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_suffixes_special_characters() {
        let source = r#"
            let users = table([[1, "Alice"]], ["id", "name"])
            let orders = table([[1, 100]], ["id", "amount"])
            let joined = left_join(orders, users, "id", "id")
            let result = joined.suffixes("_1", "_2")
            len(result)
        "#;

        // Проверяем работу с суффиксами, содержащими цифры
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 1.0, "Expected 1 row with numeric suffixes"),
            Ok(v) => panic!("Expected Number(1), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }
}


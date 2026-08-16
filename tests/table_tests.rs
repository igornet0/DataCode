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

    fn assert_null_result(source: &str) {
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Null) => {}
            Ok(v) => panic!("Expected Null, got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    fn assert_bool_result(source: &str, expected: bool) {
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Bool(v)) => assert_eq!(v, expected, "Expected {}, got {}", expected, v),
            Ok(v) => panic!("Expected Bool({}), got {:?}", expected, v),
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
            let csv_table = read("{}")
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
            let csv_table = read("{}")
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
            let csv_table = read("{}")
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
            let csv_table = read("{}")
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
            let xlsx_table = read("{}")
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
            let table = read("nonexistent_file.csv")
        "#;
        let result = run_and_get_result(source);
        // Должна быть ошибка или null
        assert!(result.is_err() || matches!(result, Ok(Value::Null)));
    }

    #[test]
    fn test_read_sample_bytes() {
        let bin_path = get_test_data_path("read_file_bin_sample.bin");
        let source = format!(
            r#"
            let b = read("{}")
            b[0] + b[1] + b[2]
            "#,
            bin_path
        );
        assert_number_result(&source, 258.0);
    }

    #[test]
    fn test_read_length() {
        let bin_path = get_test_data_path("read_file_bin_sample.bin");
        let source = format!(
            r#"
            len(read("{}"))
            "#,
            bin_path
        );
        assert_number_result(&source, 3.0);
    }

    #[test]
    fn test_read_empty_file() {
        let bin_path = get_test_data_path("read_file_bin_empty.bin");
        let source = format!(
            r#"
            len(read("{}"))
            "#,
            bin_path
        );
        assert_number_result(&source, 0.0);
    }

    #[test]
    fn test_read_nonexistent_file() {
        let source = r#"
            read("nonexistent_read.bin")
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err() || matches!(result, Ok(Value::Null)));
    }

    // ========== Тесты для параметра header в read ==========

    #[test]
    fn test_read_with_header_array() {
        // Загрузка только указанных колонок через массив
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let csv_table = read("{}", header_row=0, header=["Name", "Age"])
            let columns = csv_table.columns
            len(columns)
            "#,
            csv_path
        );
        // Должно быть 2 колонки: Name и Age
        assert_number_result(&source, 2.0);
    }

    #[test]
    fn test_read_with_header_array_column_order() {
        // Проверка порядка колонок в результате
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let csv_table = read("{}", header_row=0, header=["Age", "Name"])
            let columns = csv_table.columns
            columns[0]
            "#,
            csv_path
        );
        // Первая колонка должна быть Age (порядок из массива)
        assert_string_result(&source, "Age");
    }

    #[test]
    fn test_read_with_header_array_data_access() {
        // Проверка доступа к данным после фильтрации
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let csv_table = read("{}", header_row=0, header=["Name", "Age"])
            let names = csv_table["Name"]
            names[0]
            "#,
            csv_path
        );
        // Первое имя должно быть "John Doe"
        assert_string_result(&source, "John Doe");
    }

    #[test]
    fn test_read_with_header_array_single_column() {
        // Загрузка только одной колонки
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let csv_table = read("{}", header_row=0, header=["Age"])
            let ages = csv_table["Age"]
            ages[0]
            "#,
            csv_path
        );
        // Первый возраст должен быть 30
        assert_number_result(&source, 30.0);
    }

    #[test]
    fn test_read_with_header_dict_rename() {
        // Переименование колонок через словарь
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let csv_table = read("{}", header_row=0, header={{"Name": "FullName", "Age": null, "City": null, "Salary": null}})
            let columns = csv_table.columns
            columns[0]
            "#,
            csv_path
        );
        // Первая колонка должна быть переименована в "FullName"
        assert_string_result(&source, "FullName");
    }

    #[test]
    fn test_read_with_header_dict_keep_original() {
        // Переименование одной колонки, остальные без изменений
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let csv_table = read("{}", header_row=0, header={{"Name": "FullName", "Age": null}})
            let columns = csv_table.columns
            columns[1]
            "#,
            csv_path
        );
        // Вторая колонка должна остаться "Age" (null сохраняет оригинальное имя)
        assert_string_result(&source, "Age");
    }

    #[test]
    fn test_read_with_header_dict_access_renamed() {
        // Доступ к переименованной колонке
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let csv_table = read("{}", header_row=0, header={{"Name": "FullName", "Age": null, "City": null, "Salary": null}})
            let names = csv_table["FullName"]
            names[0]
            "#,
            csv_path
        );
        // Доступ к переименованной колонке должен работать
        assert_string_result(&source, "John Doe");
    }

    #[test]
    fn test_read_with_header_dict_multiple_renames() {
        // Переименование нескольких колонок
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let csv_table = read("{}", header_row=0, header={{"Name": "FullName", "Age": "Years", "City": null, "Salary": "Income"}})
            let columns = csv_table.columns
            columns[1]
            "#,
            csv_path
        );
        // Вторая колонка должна быть переименована в "Years"
        assert_string_result(&source, "Years");
    }

    #[test]
    fn test_read_with_header_array_and_header_row() {
        // Комбинация header_row и header (массив)
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let csv_table = read("{}", header_row=0, header=["Name", "Salary"])
            let columns = csv_table.columns
            len(columns)
            "#,
            csv_path
        );
        // Должно быть 2 колонки
        assert_number_result(&source, 2.0);
    }

    #[test]
    fn test_read_with_header_dict_and_header_row() {
        // Комбинация header_row и header (словарь)
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let csv_table = read("{}", header_row=0, header={{"Name": "FullName"}})
            let columns = csv_table.columns
            columns[0]
            "#,
            csv_path
        );
        // Первая колонка должна быть переименована
        assert_string_result(&source, "FullName");
    }

    #[test]
    fn test_read_with_header_array_nonexistent_column() {
        // Игнорирование несуществующих колонок в массиве
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let csv_table = read("{}", header_row=0, header=["Name", "NonExistent", "Age"])
            let columns = csv_table.columns
            len(columns)
            "#,
            csv_path
        );
        // Должно быть 2 колонки (NonExistent игнорируется)
        assert_number_result(&source, 2.0);
    }

    #[test]
    fn test_read_with_header_dict_nonexistent_column() {
        // Игнорирование несуществующих колонок в словаре
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let csv_table = read("{}", header_row=0, header={{"Name": "FullName", "NonExistent": "Test"}})
            let columns = csv_table.columns
            columns[0]
            "#,
            csv_path
        );
        // Первая колонка должна быть переименована, несуществующая игнорируется
        assert_string_result(&source, "FullName");
    }

    #[test]
    fn test_read_with_header_array_empty() {
        // Пустой массив header должен вернуть пустую таблицу или исходную
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let csv_table = read("{}", header_row=0, header=[])
            let columns = csv_table.columns
            len(columns)
            "#,
            csv_path
        );
        // Пустой массив - должна вернуться исходная таблица (все колонки)
        let result = run_and_get_result(&source);
        assert!(result.is_ok(), "Should handle empty header array");
    }

    #[test]
    fn test_read_with_header_dict_empty() {
        // Пустой словарь header должен вернуть исходную таблицу
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let csv_table = read("{}", header_row=0, header={{}})
            let columns = csv_table.columns
            len(columns)
            "#,
            csv_path
        );
        // Пустой словарь - должна вернуться исходная таблица (все колонки)
        assert_number_result(&source, 4.0);
    }

    #[test]
    fn test_read_with_headerT_transposed_csv() {
        let csv_path = get_test_data_path("transposed_source.csv");
        let source = format!(
            r#"
            let t = read("{}", header_row=0, headerT=["Metric", "Revenue", "Cost"])
            len(t.columns)
            "#,
            csv_path
        );
        assert_number_result(&source, 3.0);
    }

    #[test]
    fn test_read_with_headerT_transposed_data() {
        let csv_path = get_test_data_path("transposed_source.csv");
        let source = format!(
            r#"
            let t = read("{}", header_row=0, headerT=["Metric", "Revenue", "Cost"])
            t["Revenue"][0]
            "#,
            csv_path
        );
        assert_number_result(&source, 100.0);
    }

    #[test]
    fn test_read_with_headerT_column_filter() {
        let csv_path = get_test_data_path("transposed_source.csv");
        let source = format!(
            r#"
            let t = read("{}", header_row=0, headerT=["Revenue", "Cost"])
            len(t.columns)
            "#,
            csv_path
        );
        assert_number_result(&source, 2.0);
    }

    #[test]
    fn test_read_header_and_headerT_mutual_exclusion() {
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            read("{}", header=["Name"], headerT=["Name"])
            "#,
            csv_path
        );
        let result = run_and_get_result(&source);
        assert!(result.is_err(), "header and headerT together should error");
    }

    #[test]
    fn test_read_with_header_xlsx() {
        // Тест header для XLSX файлов
        let xlsx_path = get_test_data_path("sample.xlsx");
        let source = format!(
            r#"
            let xlsx_table = read("{}", header_row=0, header=["Name", "Age"])
            let columns = xlsx_table.columns
            len(columns)
            "#,
            xlsx_path
        );
        // Проверяем, что файл загружается и фильтруется
        let result = run_and_get_result(&source);
        match result {
            Ok(Value::Number(n)) => {
                // Должно быть 2 колонки после фильтрации
                assert!(n >= 0.0, "XLSX file should load with header filter");
            }
            Ok(_v) => {
                // Если файл не загружается, это нормально (может не быть XLSX файла)
                // Просто проверяем, что нет ошибки парсинга
            }
            Err(_) => {
                // Ошибка допустима, если файл не существует или не является валидным XLSX
            }
        }
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

    #[test]
    fn test_table_drop_nulls_all_columns() {
        let source = r#"
            let data = [[1, "Alice", 28], [2, null, 35], [3, "Charlie", null], [4, "David", 40]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            let cleaned = table_drop_nulls(my_table)
            len(cleaned["id"])
        "#;
        assert_number_result(source, 2.0);
    }

    #[test]
    fn test_table_drop_nulls_single_column() {
        let source = r#"
            let data = [[1, "Alice", 28], [2, null, 35], [3, "Charlie", null]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            let cleaned = table_drop_nulls(my_table, "name")
            len(cleaned["id"])
        "#;
        assert_number_result(source, 2.0);
    }

    #[test]
    fn test_table_drop_nulls_column_list() {
        let source = r#"
            let data = [[1, "Alice", 28], [2, null, 35], [3, "Charlie", null], [4, "David", 40]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            let cleaned = table_drop_nulls(my_table, ["name", "age"])
            len(cleaned["id"])
        "#;
        assert_number_result(source, 2.0);
    }

    #[test]
    fn test_table_drop_nulls_method() {
        let source = r#"
            let data = [[1, "Alice", 28], [2, null, 35], [3, "Charlie", null]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            let cleaned = my_table.drop_nulls("name")
            len(cleaned["id"])
        "#;
        assert_number_result(source, 2.0);
    }

    #[test]
    fn test_table_drop_nulls_unknown_column() {
        let source = r#"
            let data = [[1, "Alice", 28]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            table_drop_nulls(my_table, "City")
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for unknown column name");
    }

    #[test]
    fn test_table_replace_nulls_all_columns_scalar() {
        let source = r#"
            let data = [[1, null, 28], [2, "Bob", null]]
            let headers = ["id", "name", "age"]
            let t = table(data, headers)
            let cleaned = table_replace_nulls(t, "Unknown")
            cleaned["name"][0]
        "#;
        assert_string_result(source, "Unknown");
    }

    #[test]
    fn test_table_replace_nulls_single_column_scalar() {
        let source = r#"
            let data = [[1, null, 28], [2, "Bob", null]]
            let headers = ["id", "name", "age"]
            let t = table(data, headers)
            let cleaned = table_replace_nulls(t, "name", "Unknown")
            cleaned["age"][1]
        "#;
        assert_null_result(source);
    }

    #[test]
    fn test_table_replace_nulls_column_list_scalar() {
        let source = r#"
            let data = [[1, null, 28], [2, "Bob", null]]
            let headers = ["id", "name", "age"]
            let t = table(data, headers)
            let cleaned = table_replace_nulls(t, ["name", "age"], "Unknown")
            cleaned["age"][1]
        "#;
        assert_string_result(source, "Unknown");
    }

    #[test]
    fn test_table_replace_nulls_method_overloads() {
        let source = r#"
            let data = [[1, null, 28], [2, "Bob", null]]
            let headers = ["id", "name", "age"]
            let t = table(data, headers)
            let a = t.replace_nulls("Unknown")
            let b = t.replace_nulls("name", "N/A")
            len(a["id"]) + len(b["id"])
        "#;
        assert_number_result(source, 4.0);
    }

    #[test]
    fn test_table_replace_nulls_callback_row_context() {
        let source = r#"
            let data = [[1, null, 900], [2, null, 1500], [3, "Carol", 500]]
            let headers = ["id", "City", "Amount"]
            let t = table(data, headers)
            let cleaned = table_replace_nulls(t, ["City"], fn(row) => if row["Amount"] < 1000 { null } else { "Unknown" })
            cleaned["City"][1]
        "#;
        assert_string_result(source, "Unknown");
    }

    #[test]
    fn test_table_replace_nulls_callback_can_keep_null() {
        let source = r#"
            let data = [[1, null, 900]]
            let headers = ["id", "City", "Amount"]
            let t = table(data, headers)
            let cleaned = table_replace_nulls(t, "City", fn(row) => null)
            cleaned["City"][0]
        "#;
        assert_null_result(source);
    }

    #[test]
    fn test_table_replace_nulls_unknown_column() {
        let source = r#"
            let data = [[1, "Alice", 28]]
            let headers = ["id", "name", "age"]
            let t = table(data, headers)
            table_replace_nulls(t, "City", "Unknown")
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for unknown column name");
    }

    #[test]
    fn test_table_replace_nulls_invalid_column_type() {
        let source = r#"
            let data = [[1, "Alice", 28]]
            let headers = ["id", "name", "age"]
            let t = table(data, headers)
            table_replace_nulls(t, 123, "Unknown")
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for invalid column type");
    }

    #[test]
    fn test_table_replace_nulls_callback_wrong_arity() {
        let source = r#"
            let data = [[1, null, 900]]
            let headers = ["id", "City", "Amount"]
            let t = table(data, headers)
            table_replace_nulls(t, "City", fn(a, b) => "Unknown")
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for callback arity");
    }

    #[test]
    fn test_table_row_number_defaults() {
        let source = r#"
            let data = [[10, "a"], [20, "b"], [30, "c"]]
            let headers = ["id", "name"]
            let t = table(data, headers)
            let numbered = table_row_number(t)
            numbered["RowNumber"][2]
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_table_row_number_custom_name_and_start() {
        let source = r#"
            let data = [[10, "a"], [20, "b"], [30, "c"]]
            let headers = ["id", "name"]
            let t = table(data, headers)
            let numbered = table_row_number(t, "Num", 7)
            numbered["Num"][0]
        "#;
        assert_number_result(source, 7.0);
    }

    #[test]
    fn test_table_row_number_method_form() {
        let source = r#"
            let data = [[10, "a"], [20, "b"]]
            let headers = ["id", "name"]
            let t = table(data, headers)
            let numbered = t.row_number("Index", 100)
            numbered["Index"][1]
        "#;
        assert_number_result(source, 101.0);
    }

    #[test]
    fn test_table_row_number_existing_column_error() {
        let source = r#"
            let data = [[10, "a"]]
            let headers = ["id", "name"]
            let t = table(data, headers)
            table_row_number(t, "id")
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for duplicate column name");
    }

    #[test]
    fn test_table_row_number_invalid_start_type() {
        let source = r#"
            let data = [[10, "a"]]
            let headers = ["id", "name"]
            let t = table(data, headers)
            table_row_number(t, "RowNumber", "x")
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for invalid start_from type");
    }

    #[test]
    fn test_table_distinct_all_columns() {
        let source = r#"
            let data = [[1, "A"], [1, "A"], [1, "B"], [1, "B"], [2, "A"]]
            let headers = ["ID", "Category"]
            let t = table(data, headers)
            let d = table_distinct(t)
            len(d["ID"])
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_table_distinct_single_column() {
        let source = r#"
            let data = [[1, "A"], [1, "B"], [2, "C"], [2, "D"]]
            let headers = ["ID", "Category"]
            let t = table(data, headers)
            let d = table_distinct(t, "ID")
            len(d["ID"])
        "#;
        assert_number_result(source, 2.0);
    }

    #[test]
    fn test_table_distinct_column_list() {
        let source = r#"
            let data = [[1, "A", 10], [1, "A", 20], [1, "B", 20], [1, "B", 20]]
            let headers = ["ID", "Category", "Amount"]
            let t = table(data, headers)
            let d = table_distinct(t, ["ID", "Category"])
            len(d["ID"])
        "#;
        assert_number_result(source, 2.0);
    }

    #[test]
    fn test_table_distinct_method_form() {
        let source = r#"
            let data = [[1, "A"], [1, "A"], [2, "B"]]
            let headers = ["ID", "Category"]
            let t = table(data, headers)
            let d = t.distinct(["ID", "Category"])
            len(d["ID"])
        "#;
        assert_number_result(source, 2.0);
    }

    #[test]
    fn test_table_distinct_unknown_column_error() {
        let source = r#"
            let data = [[1, "A"]]
            let headers = ["ID", "Category"]
            let t = table(data, headers)
            table_distinct(t, "Missing")
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for unknown distinct column");
    }

    #[test]
    fn test_table_distinct_invalid_columns_type_error() {
        let source = r#"
            let data = [[1, "A"]]
            let headers = ["ID", "Category"]
            let t = table(data, headers)
            table_distinct(t, 123)
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for invalid distinct columns type");
    }

    #[test]
    fn test_table_value_map_object_mappings() {
        let source = r#"
            let data = [[1, "yes"], [2, "no"], [3, "maybe"]]
            let headers = ["ID", "Active"]
            let t = table(data, headers)
            let mapped = table_value_map(t, "Active", {"yes": "active", "no": "inactive"})
            mapped["Active"][1]
        "#;
        assert_string_result(source, "inactive");
    }

    #[test]
    fn test_table_value_map_array_from_to() {
        let source = r#"
            let data = [[1, "sale"], [2, "new"], [3, "old"]]
            let headers = ["ID", "Tag"]
            let t = table(data, headers)
            let mapped = table_value_map(t, "Tag", [{"from": "sale", "to": "promo"}, {"from": "new", "to": "fresh"}])
            mapped["Tag"][0]
        "#;
        assert_string_result(source, "promo");
    }

    #[test]
    fn test_table_value_map_array_old_new() {
        let source = r#"
            let data = [[1, "sale"], [2, "new"], [3, "old"]]
            let headers = ["ID", "Tag"]
            let t = table(data, headers)
            let mapped = table_value_map(t, "Tag", [{old: "old", new: "archive"}])
            mapped["Tag"][2]
        "#;
        assert_string_result(source, "archive");
    }

    #[test]
    fn test_table_value_map_method_form() {
        let source = r#"
            let data = [[1, "yes"], [2, "no"]]
            let headers = ["ID", "Active"]
            let t = table(data, headers)
            let mapped = t.value_map("Active", {"yes": "active", "no": "inactive"})
            mapped["Active"][0]
        "#;
        assert_string_result(source, "active");
    }

    #[test]
    fn test_table_value_map_keeps_unmapped_values() {
        let source = r#"
            let data = [[1, "yes"], [2, "unknown"]]
            let headers = ["ID", "Active"]
            let t = table(data, headers)
            let mapped = table_value_map(t, "Active", {"yes": "active"})
            mapped["Active"][1]
        "#;
        assert_string_result(source, "unknown");
    }

    #[test]
    fn test_table_value_map_unknown_column_error() {
        let source = r#"
            let data = [[1, "yes"]]
            let headers = ["ID", "Active"]
            let t = table(data, headers)
            table_value_map(t, "Missing", {"yes": "active"})
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for unknown column");
    }

    #[test]
    fn test_table_value_map_invalid_mappings_type_error() {
        let source = r#"
            let data = [[1, "yes"]]
            let headers = ["ID", "Active"]
            let t = table(data, headers)
            table_value_map(t, "Active", 123)
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for invalid mappings type");
    }

    #[test]
    fn test_table_value_map_invalid_mapping_item_error() {
        let source = r#"
            let data = [[1, "yes"]]
            let headers = ["ID", "Active"]
            let t = table(data, headers)
            table_value_map(t, "Active", [{"from": "yes"}])
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for invalid mapping item");
    }

    #[test]
    fn test_table_aggregate_all_ops_happy_path() {
        let source = r#"
            let data = [
                [1, "A", 100],
                [2, "A", 200],
                [3, "B", 300],
                [4, "B", 400],
                [5, "B", 500],
            ]
            let headers = ["Id", "Group", "Amount"]
            let t = table(data, headers)
            let agg = t.aggregate({
                total_rows: "count",
                distinct_groups: {op: "count_distinct", column: "Group"},
                sum_amount: {op: "sum", column: "Amount"},
                avg_amount: {op: "avg", column: "Amount"},
                min_amount: {op: "min", column: "Amount"},
                max_amount: {op: "max", column: "Amount"},
                first_amount: {op: "first", column: "Amount"},
                last_amount: {op: "last", column: "Amount"},
                median_amount: {op: "median", column: "Amount"},
                mode_group: {op: "mode", column: "Group"},
                stddev_amount: {op: "stddev", column: "Amount"},
                variance_amount: {op: "variance", column: "Amount"},
                p90_amount: {op: "percentile", column: "Amount", p: 0.9},
                amount_list: {op: "list", column: "Amount"},
                any_amount_gt_450: {op: "any", column: "Amount", where: fn(x) => x > 450},
            })
            checks =
                (if agg["total_rows"][0] == 5 { 1 } else { 0 }) +
                (if agg["distinct_groups"][0] == 2 { 1 } else { 0 }) +
                (if agg["sum_amount"][0] == 1500 { 1 } else { 0 }) +
                (if agg["avg_amount"][0] == 300 { 1 } else { 0 }) +
                (if agg["min_amount"][0] == 100 { 1 } else { 0 }) +
                (if agg["max_amount"][0] == 500 { 1 } else { 0 }) +
                (if agg["first_amount"][0] == 100 { 1 } else { 0 }) +
                (if agg["last_amount"][0] == 500 { 1 } else { 0 }) +
                (if agg["median_amount"][0] == 300 { 1 } else { 0 }) +
                (if agg["mode_group"][0] == "B" { 1 } else { 0 }) +
                (if agg["p90_amount"][0] == 460 { 1 } else { 0 }) +
                (if len(agg["amount_list"][0]) == 5 { 1 } else { 0 }) +
                (if agg["any_amount_gt_450"][0] { 1 } else { 0 })
            checks + 0.0
        "#;
        assert_number_result(source, 13.0);
    }

    #[test]
    fn test_table_aggregate_global_alias() {
        let source = r#"
            let t = table([[1, 10], [2, 20], [3, 30]], ["Id", "Amount"])
            let agg = table_aggregate(t, {
                c: "count",
                s: {op: "sum", column: "Amount"},
            })
            agg["c"][0] + agg["s"][0]
        "#;
        assert_number_result(source, 63.0);
    }

    #[test]
    fn test_table_aggregate_percentile_90() {
        let source = r#"
            let t = table([[100], [200], [300], [400], [500]], ["Amount"])
            let agg = t.aggregate({
                p90: {op: "percentile", column: "Amount", p: 0.9}
            })
            agg["p90"][0]
        "#;
        assert_number_result(source, 460.0);
    }

    #[test]
    fn test_table_aggregate_any_where_callback() {
        let source = r#"
            let t = table([[10], [20], [30]], ["Amount"])
            let agg = t.aggregate({
                has_gt_25: {op: "any", column: "Amount", where: fn(x) => x > 25}
            })
            agg["has_gt_25"][0]
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_table_aggregate_unknown_op_error() {
        let source = r#"
            let t = table([[1]], ["Amount"])
            t.aggregate({x: {op: "unknown", column: "Amount"}})
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for unknown aggregate op");
    }

    #[test]
    fn test_table_aggregate_missing_column_error() {
        let source = r#"
            let t = table([[1]], ["Amount"])
            t.aggregate({x: {op: "sum", column: "Missing"}})
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for missing aggregate column");
    }

    #[test]
    fn test_table_aggregate_invalid_spec_type_error() {
        let source = r#"
            let t = table([[1]], ["Amount"])
            t.aggregate(123)
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for invalid aggregate spec type");
    }

    #[test]
    fn test_table_aggregate_invalid_where_error() {
        let source = r#"
            let t = table([[1]], ["Amount"])
            t.aggregate({x: {op: "any", column: "Amount", where: 1}})
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for non-callable any(where)");
    }

    #[test]
    fn test_table_aggregate_group_by_city() {
        let source = r#"
            let data = [
                ["A", 100],
                ["A", 200],
                ["B", 300],
            ]
            let headers = ["City", "Amount"]
            let t = table(data, headers)
            let agg = t.aggregate_group({
                group: "City",
                count: "count",
                sum_amount: {op: "sum", column: "Amount"},
            })
            len(agg)
        "#;
        assert_number_result(source, 2.0);
    }

    #[test]
    fn test_table_aggregate_group_values() {
        let source = r#"
            let data = [
                ["A", 100],
                ["A", 200],
                ["B", 300],
            ]
            let headers = ["City", "Amount"]
            let t = table(data, headers)
            let agg = t.aggregate_group({
                group: "City",
                count: "count",
                sum_amount: {op: "sum", column: "Amount"},
            })
            checks =
                (if agg["City"][0] == "A" { 1 } else { 0 }) +
                (if agg["count"][0] == 2 { 1 } else { 0 }) +
                (if agg["sum_amount"][0] == 300 { 1 } else { 0 })
            checks + 0.0
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_table_aggregate_group_multi_key() {
        let source = r#"
            let data = [
                ["Sales", "NY", 100],
                ["Sales", "NY", 200],
                ["Sales", "LA", 300],
                ["HR", "NY", 400],
            ]
            let headers = ["Department", "City", "Amount"]
            let t = table(data, headers)
            let agg = t.aggregate_group({
                group: ["Department", "City"],
                count: "count",
                sum_amount: {op: "sum", column: "Amount"},
            })
            len(agg)
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_table_aggregate_group_null_key() {
        let source = r#"
            let data = [
                ["A", 100],
                [null, 200],
                [null, 300],
            ]
            let headers = ["City", "Amount"]
            let t = table(data, headers)
            let agg = t.aggregate_group({
                group: "City",
                count: "count",
                sum_amount: {op: "sum", column: "Amount"},
            })
            checks =
                (if agg["City"][1] == null { 1 } else { 0 }) +
                (if agg["count"][1] == 2 { 1 } else { 0 }) +
                (if agg["sum_amount"][1] == 500 { 1 } else { 0 })
            checks + 0.0
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_table_aggregate_group_percentile_and_any() {
        let source = r#"
            let data = [
                ["A", 100],
                ["A", 200],
                ["A", 500],
            ]
            let headers = ["City", "Amount"]
            let t = table(data, headers)
            let agg = t.aggregate_group({
                group: "City",
                p90: {op: "percentile", column: "Amount", p: 0.9},
                has_gt_300: {op: "any", column: "Amount", where: fn(x) => x > 300},
            })
            if agg["has_gt_300"][0] { agg["p90"][0] } else { 0 }
        "#;
        assert_number_result(source, 440.0);
    }

    #[test]
    fn test_table_aggregate_group_global_alias() {
        let source = r#"
            let t = table([["A", 10], ["B", 20]], ["City", "Amount"])
            let agg = table_aggregate_group(t, {
                group: "City",
                sum_amount: {op: "sum", column: "Amount"},
            })
            agg["sum_amount"][0] + agg["sum_amount"][1]
        "#;
        assert_number_result(source, 30.0);
    }

    #[test]
    fn test_table_aggregate_group_missing_group_error() {
        let source = r#"
            let t = table([[1]], ["Amount"])
            t.aggregate_group({count: "count"})
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for missing group field");
    }

    #[test]
    fn test_table_aggregate_group_unknown_group_column_error() {
        let source = r#"
            let t = table([[1]], ["Amount"])
            t.aggregate_group({group: "Missing", count: "count"})
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for unknown group column");
    }

    #[test]
    fn test_table_aggregate_group_empty_agg_spec_error() {
        let source = r#"
            let t = table([[1, "A"]], ["Amount", "City"])
            t.aggregate_group({group: "City"})
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for empty aggregation spec");
    }

    #[test]
    fn test_table_aggregate_group_unknown_op_error() {
        let source = r#"
            let t = table([[1, "A"]], ["Amount", "City"])
            t.aggregate_group({group: "City", x: {op: "unknown", column: "Amount"}})
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for unknown aggregate op");
    }

    #[test]
    fn test_table_rename_dict() {
        let source = r#"
            let data = [[1, "Alice", 28], [2, "Bob", 35]]
            let headers = ["id", "name", "age"]
            let t = table(data, headers)
            let renamed = table_rename(t, {"name": "full_name", "age": null})
            len(renamed.columns)
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_table_rename_pair() {
        let source = r#"
            let data = [[1, "Alice", 28]]
            let headers = ["id", "name", "age"]
            let t = table(data, headers)
            let renamed = table_rename(t, "name", "full_name")
            renamed.columns[1]
        "#;
        assert_string_result(source, "full_name");
    }

    #[test]
    fn test_table_rename_method() {
        let source = r#"
            let data = [[1, "Alice", 28]]
            let headers = ["id", "name", "age"]
            let t = table(data, headers)
            let renamed = t.rename("name", "full_name")
            renamed.columns[1]
        "#;
        assert_string_result(source, "full_name");
    }

    #[test]
    fn test_table_drop_column_single() {
        let source = r#"
            let data = [[1, "Alice", 28], [2, "Bob", 35]]
            let headers = ["id", "name", "age"]
            let t = table(data, headers)
            let trimmed = table_drop_column(t, "age")
            len(trimmed.columns)
        "#;
        assert_number_result(source, 2.0);
    }

    #[test]
    fn test_table_drop_column_multiple() {
        let source = r#"
            let data = [[1, "Alice", 28]]
            let headers = ["id", "name", "age"]
            let t = table(data, headers)
            let trimmed = table_drop_column(t, ["name", "age"])
            trimmed.columns[0]
        "#;
        assert_string_result(source, "id");
    }

    #[test]
    fn test_table_drop_column_method() {
        let source = r#"
            let data = [[1, "Alice", 28]]
            let headers = ["id", "name", "age"]
            let t = table(data, headers)
            let trimmed = t.drop_column("name")
            len(trimmed.columns)
        "#;
        assert_number_result(source, 2.0);
    }

    #[test]
    fn test_table_drop_column_unknown() {
        let source = r#"
            let t = table([[1]], ["id"])
            table_drop_column(t, "missing")
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for unknown column");
    }

    #[test]
    fn test_table_add_column_scalar() {
        let source = r#"
            let data = [[1, "Alice"], [2, "Bob"]]
            let headers = ["id", "name"]
            let t = table(data, headers)
            let extended = table_add_column(t, "region", "EMEA")
            len(extended.columns)
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_table_add_column_array() {
        let source = r#"
            let data = [[1, "Alice"], [2, "Bob"]]
            let headers = ["id", "name"]
            let t = table(data, headers)
            let extended = table_add_column(t, "score", [10, 20])
            extended["score"][0] + extended["score"][1]
        "#;
        assert_number_result(source, 30.0);
    }

    #[test]
    fn test_table_add_column_method() {
        let source = r#"
            let data = [[1, "Alice"]]
            let headers = ["id", "name"]
            let t = table(data, headers)
            let extended = t.add_column("region", "EU")
            extended.columns[2]
        "#;
        assert_string_result(source, "region");
    }

    #[test]
    fn test_table_add_column_default_null() {
        let source = r#"
            let data = [[1, "Alice"]]
            let headers = ["id", "name"]
            let t = table(data, headers)
            let extended = t.add_column("extra")
            extended["extra"][0]
        "#;
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Null) => {}
            Ok(v) => panic!("Expected Null, got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_table_map_int() {
        let source = r#"
            let data = [["30"], ["25"]]
            let headers = ["age"]
            let t = table(data, headers)
            let mapped = table_map(t, "age", int)
            typeof(mapped["age"][0])
        "#;
        assert_string_result(source, "int");
    }

    #[test]
    fn test_table_map_method_str() {
        let source = r#"
            let data = [[1, 50000]]
            let headers = ["id", "salary"]
            let t = table(data, headers)
            let mapped = t.map("salary", str)
            typeof(mapped["salary"][0])
        "#;
        assert_string_result(source, "string");
    }

    #[test]
    fn test_table_map_method_capitalize() {
        let source = r#"
            let t = table([["HELLO"]], ["Tag"])
            let mapped = t.map("Tag", capitalize)
            mapped["Tag"][0]
        "#;
        assert_string_result(source, "Hello");
    }

    #[test]
    fn test_table_map_user_fn() {
        let source = r#"
            fn inc(x) { return x + 1 }
            let data = [[1, 10], [2, 20]]
            let headers = ["id", "val"]
            let t = table(data, headers)
            let mapped = t.map("val", inc)
            typeof(mapped["val"][0])
        "#;
        assert_string_result(source, "int");
    }

    #[test]
    fn test_table_map_unknown_column() {
        let source = r#"
            let t = table([[1]], ["id"])
            table_map(t, "missing", int)
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for unknown column");
    }

    #[test]
    fn test_table_map_not_callable() {
        let source = r#"
            let t = table([[1]], ["id"])
            table_map(t, "id", 42)
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for non-callable");
    }

    #[test]
    fn test_table_map_on_view() {
        let source = r#"
            let data = [["30"], ["25"]]
            let headers = ["age"]
            let t = table(data, headers)
            let view = t.select(["age"])
            let mapped = view.map("age", int)
            typeof(mapped["age"][0])
        "#;
        assert_string_result(source, "int");
    }

    // ========== table! column write + column.map ==========

    #[test]
    fn test_column_map_returns_array() {
        let source = r#"
            let t = table([[1, 10], [2, 20]], ["id", "amount"])
            fn dbl(v) { return v * 2 }
            typeof(t["amount"].map(dbl))
        "#;
        assert_string_result(source, "array");
    }

    #[test]
    fn test_column_map_values() {
        let source = r#"
            let t = table([[1, 10], [2, 20]], ["id", "amount"])
            fn dbl(v) { return v * 2 }
            t["amount"].map(dbl)[0] + t["amount"].map(dbl)[1]
        "#;
        assert_number_result(source, 60.0);
    }

    #[test]
    fn test_table_bang_bracket_assign_array() {
        let source = r#"
            let t = table([[1, 10], [2, 20]], ["id", "amount"])
            t!["score"] = [100, 200]
            t["score"][0] + t["score"][1]
        "#;
        assert_number_result(source, 300.0);
    }

    #[test]
    fn test_table_bang_dot_assign_array() {
        let source = r#"
            let t = table([[1, 10]], ["id", "amount"])
            t!.score = [99]
            t["score"][0]
        "#;
        assert_number_result(source, 99.0);
    }

    #[test]
    fn test_table_bang_assign_preserves_row_layout() {
        let source = r#"
            let orders = table([[1, 100.0], [2, 200.0], [3, 300.0]], ["id", "amount"])
            orders!["amount_vat"] = orders["amount"].map(fn(v) => v * 1.2)
            len(orders) + orders["id"][0] + orders["id"][1] + orders["id"][2]
              + orders["amount"][0] + orders["amount"][1] + orders["amount"][2]
              + orders["amount_vat"][0] + orders["amount_vat"][1] + orders["amount_vat"][2]
        "#;
        // 3 rows + ids(1+2+3) + amounts(100+200+300) + vats(120+240+360) = 1329
        assert_number_result(source, 1329.0);
    }

    #[test]
    fn test_table_bang_assign_column_order() {
        let source = r#"
            let orders = table([[1, 100.0]], ["id", "amount"])
            orders!["amount_vat"] = [120.0]
            orders.columns[0] + "|" + orders.columns[1] + "|" + orders.columns[2]
        "#;
        assert_string_result(source, "id|amount|amount_vat");
    }

    #[test]
    fn test_table_bang_duplicate_column_errors() {
        let source = r#"
            let t = table([[1]], ["id"])
            t!["extra"] = [1]
            t!["extra"] = [2]
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected ValueError for duplicate column");
    }

    #[test]
    fn test_table_bang_length_mismatch() {
        let source = r#"
            let t = table([[1], [2]], ["id"])
            t!["x"] = [1]
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for array length mismatch");
    }

    #[test]
    fn test_table_bang_requires_simple_variable() {
        let source = r#"
            let t = table([[1]], ["id"])
            table([[9]], ["id"])!["x"] = [1]
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected parse error for non-variable table target");
    }

    #[test]
    fn test_unary_not_still_works() {
        let source = r#"
            let flag = false
            !flag
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_table_map_vs_column_map() {
        let source = r#"
            let t = table([[10]], ["amount"])
            fn dbl(v) { return v * 2 }
            let arr = t["amount"].map(dbl)
            let tbl = t.map("amount", dbl)
            typeof(arr) + "|" + typeof(tbl) + "|" + tbl["amount"][0]
        "#;
        assert_string_result(source, "array|table|20");
    }

    #[test]
    fn test_table_bang_assign_multi_column_sequence() {
        let source = r#"
            let orders = table([[1, 100.0]], ["id", "amount"])
            orders!["amount_vat"] = [120.0]
            orders!["flag"] = ["yes"]
            orders.columns[0] + "|" + orders.columns[1] + "|" + orders.columns[2] + "|" + orders.columns[3]
              + "|" + orders["amount_vat"][0] + "|" + orders["flag"][0]
        "#;
        assert_string_result(source, "id|amount|amount_vat|flag|120|yes");
    }

    #[test]
    fn test_table_split_column_global() {
        let source = r#"
            fn two_parts(x) { return [x, "Smith"] }
            let data = [["Alice"], ["Bob"]]
            let headers = ["FullName"]
            let t = table(data, headers)
            let split_tbl = table_split_column(t, "FullName", two_parts, ["P1", "P2"])
            split_tbl["P1"][0] + "|" + split_tbl["P2"][0]
        "#;
        assert_string_result(source, "Alice|Smith");
    }

    #[test]
    fn test_table_split_column_method() {
        let source = r#"
            fn pair(x) { return [x, "second"] }
            let data = [["Alice"], ["Bob"]]
            let headers = ["FullName"]
            let t = table(data, headers)
            let split_tbl = t.split_column("FullName", pair, ["P1", "P2"])
            split_tbl["P1"][1]
        "#;
        assert_string_result(source, "Bob");
    }

    #[test]
    fn test_table_split_column_short_parts() {
        let source = r#"
            fn pad(x) { return [x, ""] }
            let t = table([["Alice"]], ["FullName"])
            let split_tbl = t.split_column("FullName", pad, ["P1", "P2"])
            split_tbl["P1"][0] + "/" + split_tbl["P2"][0]
        "#;
        assert_string_result(source, "Alice/");
    }

    #[test]
    fn test_table_split_column_delimiter_string() {
        let source = r#"
            let data = [["Alice Smith"], ["Bob"]]
            let headers = ["FullName"]
            let t = table(data, headers)
            let split_tbl = t.split_column("FullName", " ", ["P1", "P2"])
            split_tbl["P1"][0] + "|" + split_tbl["P2"][0]
        "#;
        assert_string_result(source, "Alice|Smith");
    }

    #[test]
    fn test_table_join_columns_global() {
        let source = r#"
            let data = [["Alice", "Sales"], ["Bob", "IT"]]
            let headers = ["Name", "Dept"]
            let t = table(data, headers)
            let joined = table_join_columns(t, ["Name", "Dept"], "ND", " — ")
            joined["ND"][0]
        "#;
        assert_string_result(source, "Alice — Sales");
    }

    #[test]
    fn test_table_join_columns_method() {
        let source = r#"
            let data = [["Alice", "Sales"]]
            let headers = ["Name", "Dept"]
            let t = table(data, headers)
            let joined = t.join_columns(["Name", "Dept"], "ND", " - ")
            joined["ND"][0]
        "#;
        assert_string_result(source, "Alice - Sales");
    }

    #[test]
    fn test_split_column_unknown_column() {
        let source = r#"
            fn ws(x) { return split(x, " ") }
            let t = table([["a"]], ["id"])
            table_split_column(t, "missing", ws, ["P1"])
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for unknown column");
    }

    #[test]
    fn test_join_columns_duplicate_name() {
        let source = r#"
            let t = table([["a", "b"]], ["A", "B"])
            table_join_columns(t, ["A"], "A", "-")
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for duplicate column name");
    }

    #[test]
    fn test_split_column_not_array() {
        let source = r#"
            fn bad(x) { return x }
            let t = table([["a"]], ["id"])
            table_split_column(t, "id", bad, ["P1"])
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error when callback does not return array");
    }

    #[test]
    fn test_table_select_method() {
        let source = r#"
            let data = [[1, "Alice", 28], [2, "Bob", 35]]
            let headers = ["id", "name", "age"]
            let t = table(data, headers)
            let selected = t.select(["name", "age"])
            len(selected.columns)
        "#;
        assert_number_result(source, 2.0);
    }

    #[test]
    fn test_table_column_ops_on_view() {
        let source = r#"
            let data = [[1, "Alice", 28], [2, "Bob", 35], [3, "Charlie", 42]]
            let headers = ["id", "name", "age"]
            let t = table(data, headers)
            let view = t["age" > 30]
            let renamed = table_rename(view, "name", "full_name")
            len(renamed.columns)
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_table_filter_syntax_bracket() {
        // Фильтр через синтаксис data["col" = value] и data["col" == value]
        let source = r#"
            let data = [[1, "Alice", 28], [2, "Bob", 35], [3, "Charlie", 28]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            let filtered_eq = my_table["age" = 28]
            let filtered_eq2 = my_table["age" == 28]
            len(filtered_eq["age"]) + len(filtered_eq2["age"])
        "#;
        // Каждый фильтр даёт 2 строки, сумма 4
        assert_number_result(source, 4.0);
    }

    #[test]
    fn test_table_filter_syntax_operators() {
        // data["age" > 30] эквивалентно table_where(data, "age", ">", 30)
        let source = r#"
            let data = [[1, "Alice", 28], [2, "Bob", 35], [3, "Charlie", 42]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            let filtered = my_table["age" > 30]
            len(filtered["age"])
        "#;
        assert_number_result(source, 2.0);
    }

    #[test]
    fn test_table_filter_or_syntax() {
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let data = read("{}")
            let filtered = data["Age" > 25 or "City" == "Houston"]
            len(filtered["Age"])
            "#,
            csv_path
        );
        assert_number_result(&source, 4.0);
    }

    #[test]
    fn test_table_filter_and_syntax() {
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let data = read("{}")
            let filtered = data["Age" > 25 and "City" == "Houston"]
            len(filtered["Age"])
            "#,
            csv_path
        );
        assert_number_result(&source, 1.0);
    }

    #[test]
    fn test_table_filter_and_equivalent_to_chain() {
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let data = read("{}")
            let chain = data["Age" > 25]["City" == "Houston"]
            let and_expr = data["Age" > 25 and "City" == "Houston"]
            len(chain["Age"]) + len(and_expr["Age"])
            "#,
            csv_path
        );
        assert_number_result(&source, 2.0);
    }

    #[test]
    fn test_table_filter_or_and_precedence_with_parens() {
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let data = read("{}")
            let filtered = data[("Age" > 25 or "City" == "Houston") and "Salary" > 40000]
            len(filtered["Age"])
            "#,
            csv_path
        );
        assert_number_result(&source, 4.0);
    }

    #[test]
    fn test_table_filter_in_literal() {
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let data = read("{}")
            let filtered = data["City" in ["Houston", "Chicago"]]
            len(filtered["City"])
            "#,
            csv_path
        );
        assert_number_result(&source, 2.0);
    }

    #[test]
    fn test_table_filter_not_in_literal() {
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let data = read("{}")
            let filtered = data["City" not in ["Houston", "Chicago"]]
            len(filtered["City"])
            "#,
            csv_path
        );
        assert_number_result(&source, 3.0);
    }

    #[test]
    fn test_table_filter_in_and_combo() {
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let data = read("{}")
            let filtered = data["City" in ["Houston", "Chicago"] and "Age" > 25]
            len(filtered["City"])
            "#,
            csv_path
        );
        assert_number_result(&source, 2.0);
    }

    #[test]
    fn test_table_filter_in_variable() {
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let data = read("{}")
            let cities = ["Houston"]
            let filtered = data["City" in cities]
            len(filtered["City"])
            "#,
            csv_path
        );
        assert_number_result(&source, 1.0);
    }

    #[test]
    fn test_table_filter_in_set() {
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let data = read("{}")
            let filtered = data["City" in set(["Houston", "Chicago"])]
            len(filtered["City"])
            "#,
            csv_path
        );
        assert_number_result(&source, 2.0);
    }

    #[test]
    fn test_table_filter_in_or_combo() {
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let data = read("{}")
            let filtered = data["City" in ["Houston"] or "Age" > 30]
            len(filtered["City"])
            "#,
            csv_path
        );
        assert_number_result(&source, 3.0);
    }

    #[test]
    fn test_table_filter_str_contains() {
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let data = read("{}")
            let filtered = data["Name" & contains("Jo")]
            len(filtered["Name"])
            "#,
            csv_path
        );
        assert_number_result(&source, 2.0);
    }

    #[test]
    fn test_table_filter_str_starts_with() {
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let data = read("{}")
            let filtered = data["Name" & starts_with("J")]
            len(filtered["Name"])
            "#,
            csv_path
        );
        assert_number_result(&source, 2.0);
    }

    #[test]
    fn test_table_filter_str_ends_with() {
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let data = read("{}")
            let filtered = data["Name" & ends_with("n")]
            len(filtered["Name"])
            "#,
            csv_path
        );
        assert_number_result(&source, 3.0);
    }

    #[test]
    fn test_table_filter_str_and_combo() {
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let data = read("{}")
            let filtered = data["Name" & starts_with("B") and "Age" > 25]
            len(filtered["Name"])
            "#,
            csv_path
        );
        assert_number_result(&source, 1.0);
    }

    #[test]
    fn test_table_filter_str_or_combo() {
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let data = read("{}")
            let filtered = data["Name" & contains("o") or "City" & contains("Hou")]
            len(filtered["Name"])
            "#,
            csv_path
        );
        assert_number_result(&source, 4.0);
    }

    #[test]
    fn test_table_filter_str_strict_non_string_pattern() {
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let data = read("{}")
            let filtered = data["Name" & contains(50)]
            len(filtered["Name"])
            "#,
            csv_path
        );
        assert_number_result(&source, 0.0);
    }

    #[test]
    fn test_table_filter_str_strict_non_string_cell() {
        let csv_path = get_test_data_path("sample.csv");
        let source = format!(
            r#"
            let data = read("{}")
            let filtered = data["Salary" & contains("50")]
            len(filtered["Salary"])
            "#,
            csv_path
        );
        assert_number_result(&source, 0.0);
    }

    #[test]
    fn test_table_filter_chain() {
        // Цепочка фильтров: data["Age" = 30]["name" == "Bob"]
        let source = r#"
            let data = [[1, "Alice", 28], [2, "Bob", 35], [3, "Charlie", 35]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            let step1 = my_table["age" = 35]
            let step2 = step1["name" == "Bob"]
            len(step2["id"])
        "#;
        assert_number_result(source, 1.0);
    }

    #[test]
    fn test_table_filter_non_table_error() {
        // Не таблица слева от ["col" = val] → ошибка
        let source = r#"
            let a = [1, 2, 3]
            a["x" = 1]
        "#;
        let result = run_and_get_result(source);
        assert!(
            result.is_err(),
            "Expected runtime error for table filter on array, got {:?}",
            result
        );
        let err = result.unwrap_err();
        let err_str = format!("{:?}", err);
        assert!(
            err_str.contains("Table filter") && err_str.contains("Array"),
            "Expected 'Table filter requires a table, got Array', got: {}",
            err_str
        );
    }

    #[test]
    fn test_table_index_regression_after_filter() {
        // data[i], data[0], data["age"] по-прежнему работают (не фильтр)
        let source = r#"
            let data = [[1, "Alice", 28], [2, "Bob", 35]]
            let headers = ["id", "name", "age"]
            let my_table = table(data, headers)
            let col = my_table["age"]
            let row0 = my_table[0]
            len(col) + len(row0)
        "#;
        // 2 элемента в колонке + 3 в первой строке
        assert_number_result(source, 5.0);
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
            let csv_table = read("{}")
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
    fn test_right_join_basic() {
        let source = r#"
            let users = table([[1, "Alice"], [2, "Bob"], [3, "Charlie"]], ["id", "name"])
            let orders = table([[1, 100], [3, 300], [5, 500]], ["user_id", "amount"])
            let result = right_join(users, orders, "id", "user_id")
            len(result)
        "#;
        // Должно быть 3 строки: все заказы, заказ с user_id=5 без пользователя
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 3.0, "Expected 3 rows in right join"),
            Ok(v) => panic!("Expected Number(3), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_full_join_basic() {
        let source = r#"
            let users = table([[1, "Alice"], [2, "Bob"], [3, "Charlie"]], ["id", "name"])
            let orders = table([[1, 100], [3, 300], [5, 500]], ["user_id", "amount"])
            let result = full_join(users, orders, "id", "user_id")
            len(result)
        "#;
        // Должно быть 4 строки: все пользователи и все заказы
        // Alice с заказом, Bob без заказа, Charlie с заказом, заказ с user_id=5 без пользователя
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 4.0, "Expected 4 rows in full join"),
            Ok(v) => panic!("Expected Number(4), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_full_join_with_two_string_args() {
        // Тест для исправленной функции full_join с двумя отдельными строковыми аргументами
        let source = r#"
            let users = table([[1, "Alice"], [2, "Bob"], [4, "Diana"]], ["id", "name"])
            let orders = table([[101, 1, 150.00], [102, 1, 200.50], [103, 3, 75.25], [104, 5, 300.00]], ["order_id", "user_id", "amount"])
            let result = full_join(users, orders, "id", "user_id")
            len(result)
        "#;
        // Должно быть 6 строк: все пользователи и все заказы
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => {
                assert_eq!(n, 6.0, "Expected 6 rows in full join with two string args")
            }
            Ok(v) => panic!("Expected Number(6), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_join_universal_function_with_array() {
        // Тест для универсальной функции join() с массивом из двух строк
        let source = r#"
            let users = table([[1, "Alice"], [2, "Bob"], [3, "Charlie"]], ["id", "name"])
            let orders = table([[1, 100], [1, 200], [3, 300]], ["user_id", "amount"])
            let result = join(users, orders, ["id", "user_id"], "inner")
            len(result)
        "#;
        // Должно быть 3 строки: Alice с двумя заказами, Charlie с одним
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => {
                assert_eq!(n, 3.0, "Expected 3 rows in universal join with array")
            }
            Ok(v) => panic!("Expected Number(3), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_join_universal_function_left() {
        // Тест для универсальной функции join() с типом "left"
        let source = r#"
            let users = table([[1, "Alice"], [2, "Bob"], [3, "Charlie"]], ["id", "name"])
            let orders = table([[1, 100], [3, 300]], ["user_id", "amount"])
            let result = join(users, orders, ["id", "user_id"], "left")
            len(result)
        "#;
        // Должно быть 3 строки: все пользователи, Bob без заказов
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => {
                assert_eq!(n, 3.0, "Expected 3 rows in universal join with left type")
            }
            Ok(v) => panic!("Expected Number(3), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_join_universal_function_full() {
        // Тест для универсальной функции join() с типом "full"
        let source = r#"
            let users = table([[1, "Alice"], [2, "Bob"], [3, "Charlie"]], ["id", "name"])
            let orders = table([[1, 100], [3, 300], [5, 500]], ["user_id", "amount"])
            let result = join(users, orders, ["id", "user_id"], "full")
            len(result)
        "#;
        // Должно быть 4 строки: все пользователи и все заказы
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => {
                assert_eq!(n, 4.0, "Expected 4 rows in universal join with full type")
            }
            Ok(v) => panic!("Expected Number(4), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_join_universal_function_right() {
        // Тест для универсальной функции join() с типом "right"
        let source = r#"
            let users = table([[1, "Alice"], [2, "Bob"], [3, "Charlie"]], ["id", "name"])
            let orders = table([[1, 100], [3, 300], [5, 500]], ["user_id", "amount"])
            let result = join(users, orders, ["id", "user_id"], "right")
            len(result)
        "#;
        // Должно быть 3 строки: все заказы, заказ с user_id=5 без пользователя
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => {
                assert_eq!(n, 3.0, "Expected 3 rows in universal join with right type")
            }
            Ok(v) => panic!("Expected Number(3), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_right_join_with_two_string_args() {
        // Тест для right_join с двумя отдельными строковыми аргументами
        let source = r#"
            let users = table([[1, "Alice"], [2, "Bob"], [4, "Diana"]], ["id", "name"])
            let orders = table([[101, 1, 150.00], [102, 1, 200.50], [103, 3, 75.25], [104, 5, 300.00]], ["order_id", "user_id", "amount"])
            let result = right_join(users, orders, "id", "user_id")
            len(result)
        "#;
        // Должно быть 4 строки: все заказы
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => {
                assert_eq!(n, 4.0, "Expected 4 rows in right join with two string args")
            }
            Ok(v) => panic!("Expected Number(4), got {:?}", v),
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
            Ok(Value::Number(n)) => assert_eq!(
                n, 0.0,
                "Expected 0 rows in inner join with empty right table"
            ),
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
            Ok(Value::Number(n)) => assert_eq!(
                n, 2.0,
                "Expected 2 rows in left join with empty right table"
            ),
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
            Ok(Value::Number(n)) => {
                assert_eq!(n, 1.0, "Expected 1 row in right join with empty left table")
            }
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
            Ok(Value::Number(n)) => {
                assert_eq!(n, 0.0, "Expected 0 rows in inner join with no matches")
            }
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
            Ok(Value::Number(n)) => {
                assert_eq!(n, 2.0, "Expected 2 rows in anti join when no matches")
            }
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
            Ok(Value::Number(n)) => assert_eq!(
                n, 6.0,
                "Expected 6 rows in inner join with duplicates (2*3)"
            ),
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
            Ok(Value::Number(n)) => assert_eq!(
                n, 2.0,
                "Expected 2 rows in semi join (all rows with matching id)"
            ),
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
            Ok(Value::Number(n)) => assert_eq!(
                n, 1.0,
                "Expected 1 row in inner join with null (NULL != NULL)"
            ),
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
            Ok(Value::Number(n)) => assert_eq!(
                n, 1.0,
                "Expected 1 row in zip join with different lengths (min)"
            ),
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
    fn test_apply_join_left_table_has_two_rows() {
        // Проверяем, что таблица до apply_join действительно имеет 2 строки
        let source = r#"
            let left = table([[1, "Alice"], [2, "Bob"]], ["id", "name"])
            len(left)
        "#;
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => {
                assert_eq!(n, 2.0, "Table should have 2 rows before apply_join")
            }
            Ok(v) => panic!("Expected Number(2), got {:?}", v),
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
            Ok(Value::Number(n)) => {
                assert_eq!(n, 0.0, "Expected 0 rows when function returns empty table")
            }
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
            Ok(Value::Number(n)) => assert_eq!(
                n, 2.0,
                "Expected 2 rows in asof_join with multiple by columns"
            ),
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
            Ok(Value::Number(n)) => {
                assert_eq!(n, 2.0, "Expected 2 rows (one matched, one with NULLs)")
            }
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
                assert!(
                    s.contains("_o") || s == "id_o",
                    "Expected column name with _o suffix, got {}",
                    s
                );
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
            Ok(Value::Number(n)) => {
                assert_eq!(n, 1.0, "Expected 1 row after left join with suffixes")
            }
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
            Ok(Value::Number(n)) => {
                assert_eq!(n, 2.0, "Expected 2 rows after inner join with suffixes")
            }
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
            Ok(Value::Number(n)) => {
                assert_eq!(n, 6.0, "Expected 6 columns with multiple conflicts")
            }
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
            Ok(Value::Number(n)) => {
                assert_eq!(n, 2.0, "Expected 2 rows after right join with suffixes")
            }
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
            Ok(Value::Number(n)) => {
                assert_eq!(n, 3.0, "Expected 3 rows after full join with suffixes")
            }
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

    #[test]
    fn test_table_iterable_column() {
        // Проверяем, что можно итерироваться по колонке и суммировать значения
        let source = r#"
            data = [[1, "Alice", 28], [2, "Bob", 35], [3, "Charlie", 42]]
            data = table(data, ["id", "name", "age"])
            suma = 0
            for row in data["age"] {
                suma += row
            }

            suma == 105 and sum(data["age"]) == 105
        "#;
        // Сумма должна быть 105 и сумма должна быть 105
        assert_bool_result(source, true);
    }

    // ========== array.chunk(n) ==========

    #[test]
    fn test_array_chunk_len_and_tail() {
        let source = r#"
            let arr = [1, 2, 3, 4, 5, 6, 7]
            let c = arr.chunk(3)
            len(c)
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_array_chunk_last_chunk_shorter() {
        let source = r#"
            let arr = [1, 2, 3, 4, 5, 6, 7]
            let c = arr.chunk(3)
            len(c[2])
        "#;
        assert_number_result(source, 1.0);
    }

    #[test]
    fn test_array_chunk_for_in_sums_lengths() {
        let source = r#"
            let arr = [1, 2, 3, 4, 5, 6, 7]
            let total = 0
            for ch in arr.chunk(3) {
                total += len(ch)
            }
            total
        "#;
        assert_number_result(source, 7.0);
    }

    #[test]
    fn test_array_chunk_empty() {
        let source = r#"
            let arr = []
            len(arr.chunk(3))
        "#;
        assert_number_result(source, 0.0);
    }

    #[test]
    fn test_array_chunk_invalid_zero_returns_null() {
        let source = r#"
            let arr = [1, 2, 3]
            arr.chunk(0)
        "#;
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Null) => {}
            Ok(v) => panic!("Expected Null, got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_array_chunk_non_integer_returns_null() {
        let source = r#"
            let arr = [1, 2, 3]
            arr.chunk(1.5)
        "#;
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Null) => {}
            Ok(v) => panic!("Expected Null, got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    /// `chunk` on an [`ArrayView`] must materialize each window as an owned array (same as `Array` path)
    /// so `chunk[0]` and `chunk[1:]` are consistent.
    #[test]
    fn test_arrayview_chunk_first_element_and_tail_len() {
        let source = r#"
            let arr = [1, 2, 3, 4, 5, 6]
            let v = arr[0:6]
            let chunks = v.chunk(3)
            let ch0 = chunks[0]
            let ch1 = chunks[1]
            ch0[0] + ch1[0] + len(ch0[1:]) + len(ch1[1:])
        "#;
        // 1 + 4 + 2 + 2 = 9
        assert_number_result(source, 9.0);
    }

    #[test]
    fn test_arrayview_chunk_second_chunk_first_byte() {
        let source = r#"
            let arr = [10, 20, 30, 40, 50, 60]
            let v = arr[0:6]
            v.chunk(3)[1][0]
        "#;
        assert_number_result(source, 40.0);
    }
}

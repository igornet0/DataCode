/**
 * Canonical DataCode language catalog for web docs and the editor.
 * Source of truth: src/vm/globals.rs (117 globals) + docs/ru.
 */
(function (root) {
    'use strict';

    function p(name, type, description, optional) {
        return { name: name, type: type, description: description, optional: !!optional };
    }

    function fn(name, signature, description, parameters, returnType, category, example) {
        return {
            name: name,
            signature: signature,
            description: description,
            parameters: parameters || [],
            returnType: returnType,
            category: category,
            example: example || ''
        };
    }

    function method(name, signature, description, parameters, returnType, kind) {
        return {
            name: name,
            signature: signature,
            description: description,
            parameters: parameters || [],
            returnType: returnType,
            kind: kind || 'method'
        };
    }

    var builtins = [
        fn('print', 'print(...values)', 'Выводит значения в консоль через пробел.', [p('values', 'any', 'Значения для вывода')], 'null', 'utility', 'print("Hello", 42, true)'),
        fn('len', 'len(value)', 'Длина строки (байты UTF-8), массива, таблицы, объекта или множества.', [p('value', 'any', 'Значение')], 'int', 'utility', 'len("Hello")\nlen([1, 2, 3])'),
        fn('copy', 'copy(value)', 'Глубокая копия контейнера. Скаляры возвращаются без изменений.', [p('value', 'any', 'Значение')], 'any', 'utility', 'b = copy(a)'),
        fn('range', 'range(end) / range(start, end, step?)', 'Ленивый диапазон целых чисел.', [p('start', 'int', 'Начало или конец, если один аргумент'), p('end', 'int', 'Конец (не включая)', true), p('step', 'int', 'Шаг', true)], 'array', 'utility', 'for i in range(5) {\n    print(i)\n}'),

        fn('int', 'int(value)', 'Преобразует значение в целое число.', [p('value', 'any', 'Значение')], 'int', 'type', 'int("42")  # 42\nint(3.14)  # 3'),
        fn('float', 'float(value)', 'Преобразует значение в число с плавающей точкой.', [p('value', 'any', 'Значение')], 'float', 'type', 'float("3.14")  # 3.14'),
        fn('bool', 'bool(value)', 'Преобразует значение в логическое. Ложны: null, false, 0, "", [], пустой объект.', [p('value', 'any', 'Значение')], 'bool', 'type', 'bool(1)     # true\nbool("")    # false'),
        fn('str', 'str(value)', 'Преобразует значение в строку.', [p('value', 'any', 'Значение')], 'string', 'type', 'str(42)  # "42"'),
        fn('array', 'array(...)', 'Создаёт массив из аргументов.', [p('values', 'any', 'Элементы')], 'array', 'type', 'array(1, 2, 3)'),
        fn('typeof', 'typeof(value)', 'Возвращает имя типа: int, float, string, bool, array, table, path, date, …', [p('value', 'any', 'Значение')], 'string', 'type', 'typeof(42)         # "int"\ntypeof(now())      # "date"'),
        fn('isinstance', 'isinstance(value, type)', 'Проверяет тип по имени, конструктору или классу.', [p('value', 'any', 'Значение'), p('type', 'string | function | class', 'Ожидаемый тип')], 'bool', 'type', 'isinstance(42, "int")\nisinstance(now(), date)'),
        fn('date', 'date(value)', 'Создаёт значение date из строки, Unix-секунд или другой даты.', [p('value', 'any', 'Строка, число или date')], 'date', 'type', 'date("2024-03-15")'),
        fn('money', 'money(amount, format?)', 'Форматирует сумму как денежную строку.', [p('amount', 'number', 'Сумма'), p('format', 'string', 'Формат', true)], 'string', 'type', 'money(100.5)'),

        fn('now', 'now()', 'Текущий момент UTC, тип date.', [], 'date', 'datetime', 'd = now()\nprint(d.year, d.month)'),
        fn('date_to_unix', 'date_to_unix(value)', 'Unix-время в секундах (дробная часть — наносекунды).', [p('value', 'date | string | number', 'Дата, ISO-строка или число')], 'float', 'datetime', 'date_to_unix(now())'),
        fn('parse_date', 'parse_date(string, format)', 'Разбор даты по шаблону chrono strftime. При ошибке — null.', [p('string', 'string', 'Текст даты'), p('format', 'string', 'Шаблон, например %d.%m.%Y')], 'date', 'datetime', 'parse_date("15.03.2024", "%d.%m.%Y")'),
        fn('format_date', 'format_date(date, format)', 'Форматирует date в строку. Эквивалент d.format(format).', [p('date', 'date', 'Дата'), p('format', 'string', 'Шаблон')], 'string', 'datetime', 'format_date(now(), "%Y-%m-%d")'),
        fn('duration', 'duration(seconds=, minutes=, hours=, days=, milliseconds=)', 'Интервал времени. Все именованные аргументы необязательны.', [p('seconds', 'number', 'Секунды', true), p('minutes', 'number', 'Минуты', true), p('hours', 'number', 'Часы', true), p('days', 'number', 'Дни', true), p('milliseconds', 'number', 'Миллисекунды', true)], 'duration', 'datetime', 'h = duration(hours=1)\nprint(now() + h)'),

        fn('path', 'path(string)', 'Создаёт объект пути.', [p('string', 'string', 'Путь')], 'path', 'path', 'p = path("data.csv")\nprint(p["exists"])'),
        fn('path_name', 'path_name(path)', 'Имя файла или последний сегмент пути.', [p('path', 'path', 'Путь')], 'string', 'path', 'path_name(path("data.csv"))  # "data.csv"'),
        fn('path_parent', 'path_parent(path)', 'Родительский каталог.', [p('path', 'path', 'Путь')], 'path', 'path', 'path_parent(path("/tmp/a.txt"))'),
        fn('path_exists', 'path_exists(path)', 'Существует ли путь.', [p('path', 'path', 'Путь')], 'bool', 'path', 'path_exists(path("README.md"))'),
        fn('path_is_file', 'path_is_file(path)', 'Является ли путь обычным файлом.', [p('path', 'path', 'Путь')], 'bool', 'path', 'path_is_file(path("README.md"))'),
        fn('path_is_dir', 'path_is_dir(path)', 'Является ли путь каталогом.', [p('path', 'path', 'Путь')], 'bool', 'path', 'path_is_dir(getcwd())'),
        fn('path_extension', 'path_extension(path)', 'Расширение файла или null.', [p('path', 'path', 'Путь')], 'string', 'path', 'path_extension(path("a.csv"))  # "csv"'),
        fn('path_stem', 'path_stem(path)', 'Имя файла без расширения.', [p('path', 'path', 'Путь')], 'string', 'path', 'path_stem(path("data.csv"))  # "data"'),
        fn('path_len', 'path_len(path)', 'Длина пути. Для path не используйте len().', [p('path', 'path', 'Путь')], 'int', 'path', 'path_len(path("/tmp"))'),
        fn('getcwd', 'getcwd()', 'Текущая рабочая директория как path.', [], 'path', 'path', 'print(getcwd())'),
        fn('list_files', 'list_files(path, regex?)', 'Список файлов в каталоге. Опциональный фильтр — регулярное выражение.', [p('path', 'path | string', 'Каталог'), p('regex', 'string', 'Фильтр имён', true)], 'array', 'path', 'list_files(getcwd(), ".*\\\\.csv")'),
        fn('archive', 'archive(path)', 'Открывает ZIP/7Z/RAR архив.', [p('path', 'path | string', 'Файл архива')], 'archive', 'path', 'zip = archive("./backup.zip")\nprint(zip.count)'),

        fn('abs', 'abs(n)', 'Абсолютное значение числа.', [p('n', 'number', 'Число')], 'number', 'math', 'abs(-5)  # 5'),
        fn('sqrt', 'sqrt(n)', 'Квадратный корень. Для отрицательных — null.', [p('n', 'number', 'Неотрицательное число')], 'float', 'math', 'sqrt(16)  # 4.0'),
        fn('pow', 'pow(base, exp)', 'Возведение в степень.', [p('base', 'number', 'Основание'), p('exp', 'number', 'Показатель')], 'float', 'math', 'pow(2, 3)  # 8.0'),
        fn('min', 'min(...)', 'Минимум из чисел или из одного массива чисел.', [p('values', 'number | array', 'Числа или массив')], 'number', 'math', 'min(5, 2, 8)\nmin([3, 1, 2])'),
        fn('max', 'max(...)', 'Максимум из чисел или из одного массива чисел.', [p('values', 'number | array', 'Числа или массив')], 'number', 'math', 'max(5, 2, 8)'),
        fn('round', 'round(n)', 'Округление к ближайшему целому.', [p('n', 'number', 'Число')], 'number', 'math', 'round(3.6)  # 4'),
        fn('ceil', 'ceil(n)', 'Округление вверх.', [p('n', 'number', 'Число')], 'number', 'math', 'ceil(3.1)  # 4'),
        fn('floor', 'floor(n)', 'Округление вниз.', [p('n', 'number', 'Число')], 'number', 'math', 'floor(3.9)  # 3'),
        fn('divmod', 'divmod(a, b)', 'Частное и остаток как кортеж (q, r).', [p('a', 'number', 'Делимое'), p('b', 'number', 'Делитель')], 'tuple', 'math', 'q, r = divmod(17, 5)  # 3, 2'),
        fn('isinf', 'isinf(x)', 'Проверяет, является ли число бесконечностью.', [p('x', 'number', 'Число')], 'bool', 'math', 'isinf(1 / 0)'),

        fn('upper', 'upper(s)', 'Строка в верхнем регистре. Эквивалент s.upper().', [p('s', 'string', 'Строка')], 'string', 'string', 'upper("hello")  # "HELLO"'),
        fn('lower', 'lower(s)', 'Строка в нижнем регистре. Эквивалент s.lower().', [p('s', 'string', 'Строка')], 'string', 'string', 'lower("HELLO")  # "hello"'),
        fn('trim', 'trim(s)', 'Убирает пробелы по краям. Эквивалент s.trim().', [p('s', 'string', 'Строка')], 'string', 'string', 'trim("  hi  ")  # "hi"'),
        fn('split', 'split(s, delim)', 'Делит строку по разделителю. Эквивалент s.split(delim).', [p('s', 'string', 'Строка'), p('delim', 'string', 'Разделитель')], 'array', 'string', 'split("a,b,c", ",")  # ["a", "b", "c"]'),
        fn('join', 'join(array, delim)', 'Склеивает массив строк. Если первый аргумент — таблица, это JOIN таблиц.', [p('array', 'array | table', 'Массив строк или левая таблица'), p('delim', 'string | table', 'Разделитель или правая таблица')], 'string | table', 'string', 'join(["a", "b"], "-")  # "a-b"'),
        fn('contains', 'contains(s, sub)', 'Проверяет вхождение подстроки. Эквивалент s.contains(sub).', [p('s', 'string', 'Строка'), p('sub', 'string', 'Подстрока')], 'bool', 'string', 'contains("hello", "ell")  # true'),
        fn('starts_with', 'starts_with(s, prefix)', 'Начинается ли строка с префикса.', [p('s', 'string', 'Строка'), p('prefix', 'string', 'Префикс')], 'bool', 'string', 'starts_with("hello", "he")'),
        fn('ends_with', 'ends_with(s, suffix)', 'Заканчивается ли строка суффиксом.', [p('s', 'string', 'Строка'), p('suffix', 'string', 'Суффикс')], 'bool', 'string', 'ends_with("hello", "lo")'),
        fn('isupper', 'isupper(s)', 'Все буквы в верхнем регистре. Эквивалент s.isupper().', [p('s', 'string', 'Строка')], 'bool', 'string', 'isupper("ABC")  # true'),
        fn('islower', 'islower(s)', 'Все буквы в нижнем регистре. Эквивалент s.islower().', [p('s', 'string', 'Строка')], 'bool', 'string', 'islower("abc")  # true'),
        fn('replace', 'replace(s, find, repl)', 'Заменяет все вхождения. Эквивалент s.replace(find, repl).', [p('s', 'string', 'Строка'), p('find', 'string', 'Что искать'), p('repl', 'string', 'На что заменить')], 'string', 'string', 'replace("a-b-a", "-", "_")  # "a_b_a"'),
        fn('capitalize', 'capitalize(s)', 'Первая буква заглавная, остальные строчные. Эквивалент s.capitalize().', [p('s', 'string', 'Строка')], 'string', 'string', 'capitalize("hELLO")  # "Hello"'),
        fn('ord', 'ord(ch)', 'Код Unicode первого символа строки.', [p('ch', 'string', 'Символ или строка')], 'int', 'string', 'ord("A")  # 65'),

        fn('push', 'push(array, item)', 'Добавляет элемент в конец массива (in-place). Эквивалент arr.push(item).', [p('array', 'array', 'Массив'), p('item', 'any', 'Элемент')], 'array', 'array', 'x = [1]\npush(x, 2)\nx.push(3)'),
        fn('pop', 'pop(array)', 'Снимает и возвращает последний элемент. Эквивалент arr.pop().', [p('array', 'array', 'Массив')], 'any', 'array', 'pop([1, 2, 3])  # 3'),
        fn('unique', 'unique(array)', 'Уникальные элементы с сохранением порядка. Эквивалент arr.unique().', [p('array', 'array', 'Массив')], 'array', 'array', 'unique([1, 1, 2])  # [1, 2]'),
        fn('reverse', 'reverse(array)', 'Переворачивает массив (in-place). Эквивалент arr.reverse().', [p('array', 'array', 'Массив')], 'array', 'array', 'reverse([1, 2, 3])'),
        fn('sort', 'sort(array)', 'Сортирует массив (in-place). Для таблицы — table_sort.', [p('array', 'array', 'Массив')], 'array', 'array', 'sort([3, 1, 2])'),
        fn('sum', 'sum(array)', 'Сумма чисел массива или колонки. Эквивалент arr.sum().', [p('array', 'array | column', 'Массив или колонка')], 'number', 'array', 'sum([1, 2, 3])  # 6'),
        fn('average', 'average(array)', 'Среднее арифметическое. Эквивалент arr.average().', [p('array', 'array | column', 'Массив или колонка')], 'float', 'array', 'average([1, 2, 3])  # 2.0'),
        fn('count', 'count(array)', 'Число элементов. Эквивалент arr.count().', [p('array', 'array | column', 'Массив или колонка')], 'int', 'array', 'count([1, 2, 3])  # 3'),
        fn('any', 'any(array)', 'true, если есть истинный элемент. Эквивалент arr.any().', [p('array', 'array', 'Массив')], 'bool', 'array', 'any([false, true])  # true'),
        fn('all', 'all(array)', 'true, если все элементы истинны. Эквивалент arr.all().', [p('array', 'array', 'Массив')], 'bool', 'array', 'all([true, true])  # true'),
        fn('enum', 'enum(iterable)', 'Обёртка (index, value) для for. typeof → "enumerate".', [p('iterable', 'array | tuple | string | table', 'Источник')], 'enumerate', 'array', 'for i, item in enum(["a", "b"]) {\n    print(i, item)\n}'),
        fn('array_with_capacity', 'array_with_capacity(n)', 'Пустой массив с запасом памяти.', [p('n', 'int', 'Ёмкость')], 'array', 'array', 'array_with_capacity(100)'),
        fn('set', 'set() / set(iterable)', 'Множество уникальных hashable-элементов.', [p('iterable', 'array', 'Источник', true)], 'set', 'array', 's = set([1, 2, 2])\ns.add(3)'),
        fn('map', 'map(col, fn)', 'Применяет функцию к каждому элементу массива или колонки.', [p('col', 'array | column', 'Источник'), p('fn', 'function', 'fn(item) или fn(item, index)')], 'array', 'array', 'map([1, 2, 3], fn(x) => x * 2)'),
        fn('filter', 'filter(col, pred)', 'Оставляет элементы, для которых предикат истинен.', [p('col', 'array', 'Источник'), p('pred', 'function', 'Предикат')], 'array', 'array', 'filter([1, 2, 3, 4], fn(x) => x % 2 == 0)'),
        fn('reduce', 'reduce(col, fn, initial)', 'Свёртка массива.', [p('col', 'array', 'Источник'), p('fn', 'function', 'fn(acc, item)'), p('initial', 'any', 'Начальное значение')], 'any', 'array', 'reduce([1, 2, 3], fn(a, x) => a + x, 0)'),

        fn('table', 'table(data, headers?)', 'Создаёт таблицу из строк и необязательных заголовков.', [p('data', 'array', 'Массив строк (массивов)'), p('headers', 'array', 'Имена колонок', true)], 'table', 'table', 'table([[1, "Ann"], [2, "Bob"]], ["id", "name"])'),
        fn('read_file', 'read_file(path, header_row?, sheet_name?, header?, headerT?)', 'Читает CSV/XLSX как таблицу, TXT как строку. Алиас: read().', [p('path', 'path | string', 'Файл'), p('header_row', 'int', 'Строка заголовков', true), p('sheet_name', 'string', 'Лист XLSX', true), p('header', 'array | object', 'Фильтр/переименование колонок', true), p('headerT', 'array | object', 'То же после транспонирования', true)], 'table | string', 'table', 'employees = read_file("data.csv")\nread("notes.txt")'),
        fn('read_file_bin', 'read_file_bin(path)', 'Читает файл как bytes. Алиас: read_bin().', [p('path', 'path | string', 'Файл')], 'bytes', 'table', 'raw = read_file_bin("image.bin")'),
        fn('table_info', 'table_info(table)', 'Печатает сводку по таблице (колонки, типы, размер).', [p('table', 'table', 'Таблица')], 'null', 'table', 'table_info(employees)'),
        fn('table_head', 'table_head(table, n=5)', 'Первые n строк.', [p('table', 'table', 'Таблица'), p('n', 'int', 'Число строк', true)], 'table', 'table', 'table_head(employees, 5)'),
        fn('table_tail', 'table_tail(table, n=5)', 'Последние n строк.', [p('table', 'table', 'Таблица'), p('n', 'int', 'Число строк', true)], 'table', 'table', 'table_tail(employees, 5)'),
        fn('table_select', 'table_select(table, columns)', 'Оставляет указанные колонки. Эквивалент t.select(columns).', [p('table', 'table', 'Таблица'), p('columns', 'array', 'Имена колонок')], 'table', 'table', 'table_select(employees, ["name", "age"])'),
        fn('table_sort', 'table_sort(table, column, ascending=true)', 'Сортирует по колонке.', [p('table', 'table', 'Таблица'), p('column', 'string', 'Колонка'), p('ascending', 'bool', 'По возрастанию', true)], 'table', 'table', 'table_sort(employees, "salary", false)'),
        fn('table_where', 'table_where(table, column, op, value)', 'Фильтр по сравнению колонки. Также: t["Age" > 25].', [p('table', 'table', 'Таблица'), p('column', 'string', 'Колонка'), p('op', 'string', 'Оператор: >, <, ==, !=, …'), p('value', 'any', 'Значение')], 'table', 'table', 'table_where(employees, "age", ">", 25)'),
        fn('table_drop_nulls', 'table_drop_nulls(table, column?)', 'Удаляет строки с null. Эквивалент t.drop_nulls().', [p('table', 'table', 'Таблица'), p('column', 'string', 'Колонка', true)], 'table', 'table', 'table_drop_nulls(data, "age")'),
        fn('table_replace_nulls', 'table_replace_nulls(table, ...)', 'Заполняет null. Эквивалент t.replace_nulls(...).', [p('table', 'table', 'Таблица')], 'table', 'table', 'table_replace_nulls(data)'),
        fn('table_rename', 'table_rename(table, mapping)', 'Переименовывает колонки. Эквивалент t.rename(...).', [p('table', 'table', 'Таблица'), p('mapping', 'object | string', 'Словарь old→new или старое имя'), p('new', 'string', 'Новое имя', true)], 'table', 'table', 'table_rename(t, {"old": "new"})'),
        fn('table_drop_column', 'table_drop_column(table, name)', 'Удаляет колонку(и). Эквивалент t.drop_column(name).', [p('table', 'table', 'Таблица'), p('name', 'string | array', 'Имя или список')], 'table', 'table', 'table_drop_column(t, "tmp")'),
        fn('table_add_column', 'table_add_column(table, name, value?)', 'Добавляет колонку. Эквивалент t.add_column(name, value).', [p('table', 'table', 'Таблица'), p('name', 'string', 'Имя'), p('value', 'any', 'Значение или массив', true)], 'table', 'table', 'table_add_column(t, "vat", 0.2)'),
        fn('table_map', 'table_map(table, column, fn)', 'Преобразует одну колонку → новая таблица. Эквивалент t.map(column, fn).', [p('table', 'table', 'Таблица'), p('column', 'string', 'Колонка'), p('fn', 'function', 'Преобразование')], 'table', 'table', 't.map("age", fn(x) => x + 1)'),
        fn('table_split_column', 'table_split_column(table, column, delim, new_cols)', 'Делит колонку на несколько. Эквивалент t.split_column(...).', [p('table', 'table', 'Таблица'), p('column', 'string', 'Колонка'), p('delim', 'string | function', 'Разделитель'), p('new_cols', 'array', 'Новые имена')], 'table', 'table', 't.split_column("full", " ", ["first", "last"])'),
        fn('table_join_columns', 'table_join_columns(table, cols, new, delim)', 'Склеивает колонки. Эквивалент t.join_columns(...).', [p('table', 'table', 'Таблица'), p('cols', 'array', 'Колонки'), p('new', 'string', 'Новое имя'), p('delim', 'string', 'Разделитель')], 'table', 'table', 't.join_columns(["a", "b"], "ab", " ")'),
        fn('show_table', 'show_table(table)', 'Печатает таблицу в консоль.', [p('table', 'table', 'Таблица')], 'null', 'table', 'show_table(employees)'),
        fn('merge_tables', 'merge_tables(tables, mode?)', 'Объединяет список таблиц по вертикали или горизонтали.', [p('tables', 'array', 'Массив таблиц'), p('mode', 'string', 'Режим', true)], 'table', 'table', 'merge_tables([t1, t2])'),
        fn('table_row_number', 'table_row_number(table, name?, start?)', 'Добавляет номер строки. Эквивалент t.row_number().', [p('table', 'table', 'Таблица'), p('name', 'string', 'Имя колонки', true), p('start', 'int', 'Старт', true)], 'table', 'table', 't.row_number("n", 1)'),
        fn('table_distinct', 'table_distinct(table, cols?)', 'Уникальные строки. Эквивалент t.distinct().', [p('table', 'table', 'Таблица'), p('cols', 'array', 'Колонки', true)], 'table', 'table', 't.distinct(["city"])'),
        fn('table_value_map', 'table_value_map(table, column, mappings)', 'Перекодирует значения колонки. Эквивалент t.value_map(...).', [p('table', 'table', 'Таблица'), p('column', 'string', 'Колонка'), p('mappings', 'object', 'Словарь замен')], 'table', 'table', 't.value_map("status", {"ok": 1})'),
        fn('table_aggregate', 'table_aggregate(table, spec)', 'Агрегаты в одну строку. Эквивалент t.aggregate(spec).', [p('table', 'table', 'Таблица'), p('spec', 'object', 'Спецификация агрегатов')], 'table', 'table', 't.aggregate({sum: "amount"})'),
        fn('table_aggregate_group', 'table_aggregate_group(table, spec)', 'Group-by агрегаты. Эквивалент t.aggregate_group(spec).', [p('table', 'table', 'Таблица'), p('spec', 'object', 'group + агрегаты')], 'table', 'table', 't.aggregate_group({group: "city", sum: "amount"})'),
        fn('Table', 'Table(path?)', 'Конструктор/класс табличной модели. Для наследования cls Child(Table).', [p('path', 'path | string', 'Файл', true)], 'table', 'table', 'cls Sales(Table) {\n    # ...\n}'),
        fn('relate', 'relate(pk_col, fk_col, ...)', 'Объявляет связи таблиц: первая колонка — PK, остальные — FK.', [p('pk_col', 'column', 'Первичный ключ'), p('fk_cols', 'column', 'Внешние ключи')], 'null', 'table', 'relate(users["id"], orders["user_id"])'),
        fn('primary_key', 'primary_key(col)', 'Помечает колонку как первичный ключ.', [p('col', 'column | string', 'Колонка')], 'null', 'table', 'primary_key("id")'),
        fn('datasource', 'datasource({ type, url, ... })', 'Коннектор HTTP / файл / SQL / MongoDB.', [p('config', 'object', 'Конфигурация: type, url, path, …')], 'datasource', 'table', 'ds = datasource({ type: "http", url: "https://api.example.com" })'),
        fn('save', 'save(table, path)', 'Сохраняет таблицу (extended builtin).', [p('table', 'table', 'Таблица'), p('path', 'path | string', 'Файл')], 'null', 'table', 'save(employees, "out.csv")'),
        fn('save_tables_sqlite', 'save_tables_sqlite(tables, filename="db", **kwargs)', 'Экспорт таблиц в SQLite (extended builtin).', [p('tables', 'object | array', 'Таблицы'), p('filename', 'string', 'Файл БД', true)], 'null', 'table', 'save_tables_sqlite({employees: t}, "model.db")'),

        fn('inner_join', 'inner_join(left, right, on, ...)', 'Строки, у которых есть совпадение в обеих таблицах.', [p('left', 'table', 'Левая таблица'), p('right', 'table', 'Правая таблица'), p('on', 'string | array', 'Ключ(и)')], 'table', 'join', 'inner_join(users, orders, "id", "user_id")'),
        fn('left_join', 'left_join(left, right, on, ...)', 'Все строки left; пропуски right заполняются null.', [p('left', 'table', 'Левая'), p('right', 'table', 'Правая'), p('on', 'string | array', 'Ключ(и)')], 'table', 'join', 'left_join(users, orders, "id", "user_id")'),
        fn('right_join', 'right_join(left, right, on, ...)', 'Все строки right; пропуски left заполняются null.', [p('left', 'table', 'Левая'), p('right', 'table', 'Правая'), p('on', 'string | array', 'Ключ(и)')], 'table', 'join', 'right_join(users, orders, "id", "user_id")'),
        fn('full_join', 'full_join(left, right, on, ...)', 'Все строки обеих таблиц.', [p('left', 'table', 'Левая'), p('right', 'table', 'Правая'), p('on', 'string | array', 'Ключ(и)')], 'table', 'join', 'full_join(users, orders, "id", "user_id")'),
        fn('cross_join', 'cross_join(left, right)', 'Декартово произведение.', [p('left', 'table', 'Левая'), p('right', 'table', 'Правая')], 'table', 'join', 'cross_join(colors, sizes)'),
        fn('semi_join', 'semi_join(left, right, on, ...)', 'Строки left, для которых есть match в right, без колонок right.', [p('left', 'table', 'Левая'), p('right', 'table', 'Правая'), p('on', 'string | array', 'Ключ(и)')], 'table', 'join', 'semi_join(users, orders, "id", "user_id")'),
        fn('anti_join', 'anti_join(left, right, on, ...)', 'Строки left без совпадений в right.', [p('left', 'table', 'Левая'), p('right', 'table', 'Правая'), p('on', 'string | array', 'Ключ(и)')], 'table', 'join', 'anti_join(users, orders, "id", "user_id")'),
        fn('zip_join', 'zip_join(left, right, ...)', 'Соединение по позиции строк.', [p('left', 'table', 'Левая'), p('right', 'table', 'Правая')], 'table', 'join', 'zip_join(a, b)'),
        fn('asof_join', 'asof_join(left, right, on, ..., direction?)', 'Соединение по ближайшему времени.', [p('left', 'table', 'Левая'), p('right', 'table', 'Правая'), p('on', 'string', 'Ключ времени'), p('direction', 'string', 'backward | forward', true)], 'table', 'join', 'asof_join(trades, prices, "date", "date", "backward")'),
        fn('apply_join', 'apply_join(left, right, on, ...)', 'JOIN с пользовательской логикой.', [p('left', 'table', 'Левая'), p('right', 'table', 'Правая'), p('on', 'any', 'Условие')], 'table', 'join', 'apply_join(left, right, "id")'),
        fn('join_on', 'join_on(left, right, on, ...)', 'Универсальный JOIN по ключам или выражению.', [p('left', 'table', 'Левая'), p('right', 'table', 'Правая'), p('on', 'any', 'Ключи или выражение')], 'table', 'join', 'join_on(users, orders, "id")'),
        fn('table_suffixes', 'table_suffixes(left, right, left_suffix, right_suffix)', 'Суффиксы колонок при конфликте имён JOIN.', [p('left', 'table', 'Левая'), p('right', 'table', 'Правая'), p('left_suffix', 'string', 'Суффикс left'), p('right_suffix', 'string', 'Суффикс right')], 'table', 'join', 'table_suffixes(a, b, "_l", "_r")'),

        fn('sha256', 'sha256(data)', 'SHA-256 от строки или bytes → 32 байта. str(digest) — hex.', [p('data', 'string | bytes', 'Данные')], 'bytes', 'crypto', 'str(sha256("hello"))'),
        fn('sha512', 'sha512(data)', 'SHA-512 → 64 байта. str(digest) — hex.', [p('data', 'string | bytes', 'Данные')], 'bytes', 'crypto', 'len(sha512("hello"))  # 64'),
        fn('hmac_sha256', 'hmac_sha256(key, data)', 'HMAC-SHA256. Оба аргумента — bytes.', [p('key', 'bytes', 'Ключ'), p('data', 'bytes', 'Данные')], 'bytes', 'crypto', 'hmac_sha256(random_bytes(16), random_bytes(8))'),
        fn('hmac_sha512', 'hmac_sha512(key, data)', 'HMAC-SHA512. Оба аргумента — bytes.', [p('key', 'bytes', 'Ключ'), p('data', 'bytes', 'Данные')], 'bytes', 'crypto', 'hmac_sha512(random_bytes(16), random_bytes(8))'),
        fn('random_bytes', 'random_bytes(size)', 'Криптостойкие случайные байты.', [p('size', 'int', 'Длина')], 'bytes', 'crypto', 'random_bytes(16)'),
        fn('random', 'random()', 'Случайное float в [0, 1).', [], 'float', 'crypto', 'random()'),
        fn('random_int', 'random_int(min, max)', 'Случайное целое в диапазоне [min, max].', [p('min', 'int', 'Нижняя граница'), p('max', 'int', 'Верхняя граница')], 'int', 'crypto', 'random_int(1, 6)'),
        fn('random_seed', 'random_seed(seed)', 'Задаёт зерно генератора случайных чисел.', [p('seed', 'int', 'Зерно')], 'null', 'crypto', 'random_seed(42)')
    ];

    var aliases = {
        read: 'read_file',
        read_bin: 'read_file_bin'
    };

    var aliasDefs = [
        fn('read', 'read(path, ...)', 'Алиас read_file: читает CSV/XLSX как таблицу, TXT как строку.', [p('path', 'path | string', 'Файл')], 'table | string', 'table', 't = read("data.csv")'),
        fn('read_bin', 'read_bin(path)', 'Алиас read_file_bin: читает файл как bytes.', [p('path', 'path | string', 'Файл')], 'bytes', 'table', 'raw = read_bin("blob.bin")')
    ];

    var allBuiltins = builtins.concat(aliasDefs);

    var typeMethods = {
        string: [
            method('upper', 's.upper()', 'Верхний регистр', [], 'string'),
            method('lower', 's.lower()', 'Нижний регистр', [], 'string'),
            method('trim', 's.trim()', 'Убрать пробелы по краям', [], 'string'),
            method('split', 's.split(delim)', 'Разделить по разделителю', [p('delim', 'string', 'Разделитель')], 'array'),
            method('contains', 's.contains(sub)', 'Есть ли подстрока', [p('sub', 'string', 'Подстрока')], 'bool'),
            method('isupper', 's.isupper()', 'Все буквы заглавные', [], 'bool'),
            method('islower', 's.islower()', 'Все буквы строчные', [], 'bool'),
            method('replace', 's.replace(find, repl)', 'Заменить все вхождения', [p('find', 'string', 'Что искать'), p('repl', 'string', 'На что')], 'string'),
            method('capitalize', 's.capitalize()', 'Первая заглавная, остальные строчные', [], 'string')
        ],
        array: [
            method('push', 'arr.push(item)', 'Добавить элемент в конец', [p('item', 'any', 'Элемент')], 'array'),
            method('pop', 'arr.pop()', 'Снять последний элемент', [], 'any'),
            method('unique', 'arr.unique()', 'Уникальные элементы', [], 'array'),
            method('reverse', 'arr.reverse()', 'Перевернуть массив', [], 'array'),
            method('sort', 'arr.sort()', 'Сортировать in-place', [], 'array'),
            method('sum', 'arr.sum()', 'Сумма чисел', [], 'number'),
            method('average', 'arr.average()', 'Среднее', [], 'float'),
            method('count', 'arr.count()', 'Число элементов', [], 'int'),
            method('any', 'arr.any()', 'Есть ли истинный элемент', [], 'bool'),
            method('all', 'arr.all()', 'Все элементы истинны', [], 'bool'),
            method('chunk', 'arr.chunk(n)', 'Разбить на куски длины n', [p('n', 'int', 'Размер')], 'array'),
            method('clone', 'arr.clone()', 'Глубокая копия', [], 'array'),
            method('map', 'arr.map(fn)', 'Преобразовать элементы', [p('fn', 'function', 'fn(item)')], 'array')
        ],
        set: [
            method('add', 's.add(x)', 'Вставить элемент', [p('x', 'any', 'Элемент')], 'null'),
            method('remove', 's.remove(x)', 'Удалить элемент или ошибка', [p('x', 'any', 'Элемент')], 'null'),
            method('discard', 's.discard(x)', 'Удалить, если есть', [p('x', 'any', 'Элемент')], 'null'),
            method('pop', 's.pop()', 'Извлечь произвольный элемент', [], 'any'),
            method('clear', 's.clear()', 'Очистить множество', [], 'null'),
            method('copy', 's.copy()', 'Глубокая копия', [], 'set'),
            method('update', 's.update(iterable)', 'Добавить все элементы', [p('iterable', 'array', 'Источник')], 'null'),
            method('contains', 's.contains(x)', 'Есть ли элемент', [p('x', 'any', 'Элемент')], 'bool')
        ],
        table: [
            method('add_row', 't.add_row(row)', 'Добавить строку (массив ячеек)', [p('row', 'array', 'Строка')], 'table'),
            method('push', 't.push(row)', 'Алиас add_row', [p('row', 'array', 'Строка')], 'table'),
            method('select', 't.select(columns)', 'Оставить колонки', [p('columns', 'array', 'Имена')], 'table'),
            method('rename', 't.rename(mapping)', 'Переименовать колонки', [p('mapping', 'object', 'old → new')], 'table'),
            method('drop_column', 't.drop_column(name)', 'Удалить колонку(и)', [p('name', 'string | array', 'Имя')], 'table'),
            method('add_column', 't.add_column(name, value?)', 'Добавить колонку', [p('name', 'string', 'Имя'), p('value', 'any', 'Значение', true)], 'table'),
            method('map', 't.map(column, fn)', 'Преобразовать колонку', [p('column', 'string', 'Колонка'), p('fn', 'function', 'Преобразование')], 'table'),
            method('split_column', 't.split_column(col, delim, new_cols)', 'Разделить колонку', [p('col', 'string', 'Колонка'), p('delim', 'string', 'Разделитель'), p('new_cols', 'array', 'Новые имена')], 'table'),
            method('join_columns', 't.join_columns(cols, new, delim)', 'Склеить колонки', [p('cols', 'array', 'Колонки'), p('new', 'string', 'Новое имя'), p('delim', 'string', 'Разделитель')], 'table'),
            method('drop_nulls', 't.drop_nulls(col?)', 'Удалить строки с null', [p('col', 'string', 'Колонка', true)], 'table'),
            method('replace_nulls', 't.replace_nulls(...)', 'Заполнить null', [], 'table'),
            method('row_number', 't.row_number(name?, start?)', 'Добавить номер строки', [p('name', 'string', 'Имя', true), p('start', 'int', 'Старт', true)], 'table'),
            method('distinct', 't.distinct(cols?)', 'Уникальные строки', [p('cols', 'array', 'Колонки', true)], 'table'),
            method('value_map', 't.value_map(col, mappings)', 'Перекодировать значения', [p('col', 'string', 'Колонка'), p('mappings', 'object', 'Замены')], 'table'),
            method('aggregate', 't.aggregate(spec)', 'Агрегаты в одну строку', [p('spec', 'object', 'Спецификация')], 'table'),
            method('aggregate_group', 't.aggregate_group(spec)', 'Group-by агрегаты', [p('spec', 'object', 'group + агрегаты')], 'table'),
            method('sort', 't.sort(column, ascending?)', 'Сортировка по колонке', [p('column', 'string', 'Колонка'), p('ascending', 'bool', 'По возрастанию', true)], 'table')
        ],
        column: [
            method('map', 'col.map(fn)', 'Преобразовать ячейки → массив', [p('fn', 'function', 'fn(cell)')], 'array')
        ],
        date: [
            method('year', 'd.year', 'Год', [], 'int', 'property'),
            method('month', 'd.month', 'Месяц 1–12', [], 'int', 'property'),
            method('day', 'd.day', 'День месяца', [], 'int', 'property'),
            method('quarter', 'd.quarter', 'Квартал 1–4', [], 'int', 'property'),
            method('hour', 'd.hour', 'Час 0–23', [], 'int', 'property'),
            method('minute', 'd.minute', 'Минута', [], 'int', 'property'),
            method('second', 'd.second', 'Секунда', [], 'int', 'property'),
            method('weekday', 'd.weekday', 'День недели ISO (1=пн … 7=вс)', [], 'int', 'property'),
            method('utc', 'd.utc', 'Тот же момент в UTC', [], 'date', 'property'),
            method('to_utc', 'd.to_utc', 'Тот же момент в UTC', [], 'date', 'property'),
            method('format', 'd.format(fmt)', 'Формат chrono strftime', [p('fmt', 'string', 'Шаблон')], 'string')
        ],
        duration: [
            method('seconds', 'd.seconds', 'Полные секунды интервала', [], 'number', 'property')
        ],
        path: [
            method('name', 'p.name / p["name"]', 'Имя файла', [], 'string', 'property'),
            method('parent', 'p.parent / p["parent"]', 'Родительский путь', [], 'path', 'property'),
            method('exists', 'p.exists / p["exists"]', 'Существует ли', [], 'bool', 'property'),
            method('is_file', 'p.is_file / p["is_file"]', 'Обычный файл', [], 'bool', 'property'),
            method('is_dir', 'p.is_dir / p["is_dir"]', 'Каталог', [], 'bool', 'property'),
            method('extension', 'p.extension / p["extension"]', 'Расширение', [], 'string', 'property')
        ],
        archive: [
            method('path', 'a.path', 'Путь к архиву', [], 'path', 'property'),
            method('format', 'a.format', 'zip / 7z / rar', [], 'string', 'property'),
            method('files', 'a.files', 'Список файлов', [], 'array', 'property'),
            method('count', 'a.count', 'Число файлов', [], 'int', 'property'),
            method('size', 'a.size', 'Размер после распаковки', [], 'int', 'property'),
            method('compressed_size', 'a.compressed_size', 'Размер на диске', [], 'int', 'property'),
            method('read', 'a.read(path)', 'Прочитать файл (автотип)', [p('path', 'string', 'Путь внутри архива')], 'any'),
            method('read_text', 'a.read_text(path)', 'Прочитать как UTF-8', [p('path', 'string', 'Путь')], 'string'),
            method('extract', 'a.extract(dest)', 'Распаковать в каталог', [p('dest', 'path | string', 'Назначение')], 'null'),
            method('close', 'a.close()', 'Закрыть архив', [], 'null')
        ],
        datasource: [
            method('type', 'ds.type', 'Тип коннектора', [], 'string', 'property'),
            method('name', 'ds.name', 'Имя', [], 'string', 'property'),
            method('url', 'ds.url', 'URL / строка подключения', [], 'string', 'property'),
            method('enabled', 'ds.enabled', 'Включён ли', [], 'bool', 'property'),
            method('capabilities', 'ds.capabilities', 'Флаги возможностей', [], 'object', 'property'),
            method('request', 'ds.request(spec)', 'Сырой запрос → response', [p('spec', 'object', 'Параметры')], 'response'),
            method('get_table', 'ds.get_table(spec)', 'Получить таблицу', [p('spec', 'object', 'sql / filter / …')], 'table'),
            method('send_table', 'ds.send_table(spec)', 'Отправить таблицу', [p('spec', 'object', 'table, mode, …')], 'null'),
            method('connect', 'ds.connect()', 'Открыть соединение', [], 'null'),
            method('disconnect', 'ds.disconnect()', 'Закрыть соединение', [], 'null'),
            method('ping', 'ds.ping()', 'Проверка доступности', [], 'bool'),
            method('test', 'ds.test()', 'Диагностика { ok, message, … }', [], 'object'),
            method('clone', 'ds.clone()', 'Копия handle', [], 'datasource')
        ],
        response: [
            method('status', 'r.status', 'HTTP-статус', [], 'int', 'property'),
            method('success', 'r.success', 'Успех', [], 'bool', 'property'),
            method('url', 'r.url', 'URL', [], 'string', 'property'),
            method('headers', 'r.headers', 'Заголовки', [], 'object', 'property'),
            method('content_type', 'r.content_type', 'Content-Type', [], 'string', 'property'),
            method('content_length', 'r.content_length', 'Длина тела', [], 'int', 'property'),
            method('elapsed', 'r.elapsed', 'Время запроса', [], 'duration', 'property'),
            method('body', 'r.body', 'Тело', [], 'any', 'property'),
            method('text', 'r.text', 'Тело как строка', [], 'string', 'property'),
            method('bytes', 'r.bytes', 'Тело как bytes', [], 'bytes', 'property'),
            method('json', 'r.json()', 'Разобрать JSON', [], 'object'),
            method('table', 'r.table()', 'Разобрать как таблицу', [], 'table'),
            method('csv', 'r.csv()', 'Разобрать CSV', [], 'table'),
            method('save', 'r.save(path)', 'Сохранить тело', [p('path', 'path | string', 'Файл')], 'null'),
            method('save_text', 'r.save_text(path)', 'Сохранить как текст', [p('path', 'path | string', 'Файл')], 'null'),
            method('save_json', 'r.save_json(path)', 'Сохранить JSON', [p('path', 'path | string', 'Файл')], 'null')
        ],
        generator: [
            method('next', 'g.next()', 'Следующее значение потока', [], 'any'),
            method('send', 'g.send(value)', 'Отправить значение в ireturn', [p('value', 'any', 'Значение')], 'any'),
            method('final', 'g.final()', 'Значение ereturn после завершения', [], 'any'),
            method('live', 'g.live', 'Жив ли генератор', [], 'bool', 'property')
        ],
        uuid: [
            method('to_string', 'u.to_string() / str(u)', 'Строка UUID с дефисами', [], 'string'),
            method('to_bytes', 'u.to_bytes()', '16 байт UUID', [], 'bytes')
        ]
    };

    var returnTypes = {};
    allBuiltins.forEach(function (b) {
        returnTypes[b.name] = b.returnType;
    });
    returnTypes.read = 'table';
    returnTypes.read_file = 'table';
    returnTypes.read_bin = 'bytes';
    returnTypes.read_file_bin = 'bytes';

    var methodReturnTypes = {};
    Object.keys(typeMethods).forEach(function (typeName) {
        typeMethods[typeName].forEach(function (m) {
            methodReturnTypes[typeName + '.' + m.name] = m.returnType;
        });
    });

    var categoryNames = {
        utility: 'Утилиты',
        type: 'Типы',
        datetime: 'Дата и время',
        path: 'Пути и файлы',
        math: 'Математика',
        string: 'Строки',
        array: 'Массивы',
        table: 'Таблицы',
        join: 'JOIN',
        crypto: 'Крипто и RNG'
    };

    var keywords = [
        'global', 'let', 'fn', 'stream',
        'if', 'else',
        'for', 'in', 'while',
        'return', 'ireturn', 'ereturn',
        'try', 'catch', 'finally', 'throw',
        'break', 'continue',
        'import', 'from', 'as',
        'cls', 'new', 'super', 'this',
        'Abstract',
        'public', 'private', 'protected',
        'true', 'false', 'null',
        'and', 'or',
        'inf', 'nan'
    ];

    var operators = [
        '==', '!=', '<=', '>=', '<<', '>>',
        'and', 'or', 'in',
        '+', '-', '*', '/', '%',
        '<', '>', '!',
        '&', '|', '^', '~',
        '?', ':', '='
    ];

    var modules = [
        'plot', 'uuid', 'settings_env', 'system', 'database_engine',
        'heapq', 'pathfind', 'grid', 'crypto', 'web', 'websocket', 'ws', 'debug'
    ];

    var typeAliases = {
        str: 'string',
        string: 'string',
        int: 'int',
        integer: 'int',
        float: 'float',
        number: 'number',
        bool: 'bool',
        boolean: 'bool',
        array: 'array',
        table: 'table',
        path: 'path',
        date: 'date',
        duration: 'duration',
        set: 'set',
        bytes: 'bytes',
        object: 'object',
        tuple: 'tuple',
        column: 'column',
        enumerate: 'enumerate',
        archive: 'archive',
        datasource: 'datasource',
        response: 'response',
        uuid: 'uuid',
        function: 'function',
        generator: 'generator',
        null: 'null'
    };

    var catalog = {
        keywords: keywords,
        operators: operators,
        builtins: allBuiltins,
        builtinNames: allBuiltins.map(function (b) { return b.name; }),
        aliases: aliases,
        typeMethods: typeMethods,
        returnTypes: returnTypes,
        methodReturnTypes: methodReturnTypes,
        categoryNames: categoryNames,
        modules: modules,
        typeAliases: typeAliases,
        getBuiltin: function (name) {
            if (!name) return null;
            var key = String(name);
            var aliased = aliases[key] || aliases[key.toLowerCase()];
            var target = aliased || key;
            for (var i = 0; i < allBuiltins.length; i++) {
                if (allBuiltins[i].name === target || allBuiltins[i].name.toLowerCase() === target.toLowerCase()) {
                    return allBuiltins[i];
                }
            }
            return null;
        },
        getTypeMethods: function (typeName) {
            if (!typeName) return [];
            var mapped = typeAliases[typeName] || typeAliases[String(typeName).toLowerCase()] || typeName;
            return typeMethods[mapped] || typeMethods[String(mapped).toLowerCase()] || [];
        },
        normalizeType: function (typeName) {
            if (!typeName) return null;
            return typeAliases[typeName] || typeAliases[String(typeName).toLowerCase()] || typeName;
        }
    };

    root.DATACODE_LANG = catalog;
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = catalog;
    }
})(typeof window !== 'undefined' ? window : globalThis);

# 🧠 ДатаКод - Интерактивный язык программирования

<img width="935" height="451" alt="DataCode image" src="https://github.com/user-attachments/assets/5f2d3745-dcf2-47cf-9caa-881c7d10ae71" />

**ДатаКод** — простой интерактивный язык программирования, разработанный для быстрой обработки данных и легкого обучения. Он отличается интуитивным синтаксисом, мощной поддержкой массивов, встроенными функциями и пользовательскими функциями с локальной областью видимости.

## 🚀 Возможности

- **Интерактивный REPL** с поддержкой многострочного ввода и историей команд
- **Выполнение файлов** - пишите программы в файлах `.dc`
- **Литералы массивов** - `[1, 2, 3]`, `['a', 'b']`, поддерживаются смешанные типы
- **Индексирование массивов** - `arr[0]`, `nested[0][1]` с полной поддержкой вложенности
- **Пользовательские функции** с локальной областью видимости, параметрами и рекурсией
- **Условные операторы** - if/else/endif с поддержкой вложенности
- **Циклы for** - итерация по массивам с `for...in`
- **Циклы while** - условные циклы с `while`
- **Арифметические и логические операции** с правильным приоритетом
- **Работа со строками** и конкатенация
- **Операции с таблицами** - работа с CSV/Excel файлами, автоматическая типизация
- **50 встроенных функций** - математические, строковые, массивные, файловые и табличные операции
- **Манипуляции с путями** для операций с файловой системой
- **Гибкие типы данных** - числа, строки, булевы значения, массивы, объекты, таблицы, пути
- **Обработка ошибок** - try/catch/throw для обработки исключений
- **Улучшенные сообщения об ошибках** с номерами строк и контекстом
- **Поддержка комментариев** с `#`
- **Импорт модулей** — `import ml`, `from ml import ...`, разрешение через файлы и DPM-пакеты
- **ООП** — классы (`cls`), наследование, конструкторы, методы, видимость (private/protected/public)
- **Аннотации типов для функций** — типы параметров и возвращаемого значения (int, float, str и др.)

## 📦 Установка

### Вариант 1: Глобальная установка (Рекомендуется)
Установите DataCode как глобальную команду:

```bash
# Клонируйте и установите
git clone https://github.com/igornet0/DataCode.git
cd DataCode

# Установите глобально
make install
# или
./install.sh

# Теперь вы можете использовать datacode откуда угодно!
datacode --help
```

### Вариант 2: Режим разработки
Запустите напрямую с помощью Cargo:

```bash
git clone https://github.com/igornet0/DataCode.git
cd DataCode
cargo run
```

### 🛠️ Поддержка редакторов

#### Visual Studio Code
Установите расширение для поддержки синтаксиса DataCode в VS Code:

**[📦 DataCode Syntax Extension](https://marketplace.visualstudio.com/items?itemName=datacode.datacode-syntax&ssr=false#version-history)**

Расширение предоставляет:
- Подсветку синтаксиса для файлов `.dc`
- Автодополнение встроенных функций
- Проверку ошибок
- Поддержку навигации по коду

### 🚀 Поддержка GPU (Опционально)

DataCode поддерживает ускорение вычислений на GPU для машинного обучения через опциональные функции компиляции:

#### Metal (macOS)
Для использования GPU на macOS (Apple Silicon или Intel с Metal-совместимой видеокартой):

```bash
# Режим разработки с Metal
cargo run --features metal -- filename.dc

# Или используйте Makefile
make run-metal FILE=filename.dc
```

#### CUDA (Linux/Windows)
Для использования CUDA на Linux или Windows с NVIDIA GPU:

```bash
# Режим разработки с CUDA
cargo run --features cuda -- filename.dc

# Или используйте Makefile
make run-cuda FILE=filename.dc
```

#### Использование в коде
```datacode
import ml

# Автоматическое определение GPU (если доступно)
ml.set_device("auto")

# Или явно указать устройство
ml.set_device("metal")  # macOS
ml.set_device("cuda")   # Linux/Windows с NVIDIA GPU
ml.set_device("cpu")    # Принудительно использовать CPU

# Для нейронных сетей
model = ml.neural_network(...)
model.device("metal")  # или "cuda", "cpu"
```

**Примечание:** Если GPU недоступен или проект скомпилирован без GPU-поддержки, код автоматически переключится на CPU с предупреждением. Для полной поддержки GPU необходимо перекомпилировать проект с соответствующим feature flag.

## 🎯 Использование

### После глобальной установки
```bash
datacode                   # Запустить интерактивный REPL (по умолчанию)
datacode filename.dc       # Выполнить файл DataCode
datacode filename.dc --debug  # Выполнить с отладочной информацией
datacode filename.dc --build_model  # Выполнить и экспортировать таблицы в SQLite
datacode filename.dc --build_model output.db  # Экспортировать в указанный файл
datacode --websocket       # Запустить WebSocket сервер (ws://127.0.0.1:8080)
datacode --websocket --host 0.0.0.0 --port 8899  # Кастомный хост/порт
datacode --websocket --use-ve  # Режим виртуальной среды (изоляция сессий)
datacode --help            # Показать справку
```

### Режим разработки
```bash
cargo run                  # Запустить интерактивный REPL
cargo run filename.dc      # Выполнить файл DataCode
cargo run -- --help       # Показать справку

# С поддержкой GPU
cargo run --features metal -- filename.dc  # macOS с Metal
cargo run --features cuda -- filename.dc   # Linux/Windows с CUDA

# Или используйте Makefile
make run                   # Запустить REPL
make run-metal FILE=filename.dc  # Запустить с Metal (macOS)
make run-cuda FILE=filename.dc  # Запустить с CUDA (Linux/Windows)
make examples              # Запустить все примеры
make test                  # Запустить тесты
```

### Быстрые примеры
```bash
# Создайте простой файл DataCode
echo 'print("Hello, DataCode!")' > hello.dc

# Создайте пример с массивом
echo 'global arr = [1, 2, 3]
print("Array:", arr)
print("First element:", arr[0])' > arrays.dc

# Выполните файлы
datacode hello.dc          # (после глобальной установки)
datacode arrays.dc
# или
cargo run hello.dc         # (режим разработки)
cargo run arrays.dc
```

### Программное использование
```rust
use data_code::run;

fn main() {
    run("global basePath = getcwd()").unwrap();
    run("global files = list_files(basePath / 'data')").unwrap();
}
```
---

## 📄 Синтаксис языка

### 🔹 Переменные
```DataCode
global path = getcwd()
let subdir = 'data'
```
• `global` — сохраняет переменную глобально
• `let` — ограничена текущим контекстом (например, циклом)

### 🔹 Арифметические операции
```DataCode
global x = 10
global y = 20
global sum = x + y          # Сложение
global diff = x - y         # Вычитание
global prod = x * y         # Умножение
global quot = x / y         # Деление
global complex = (x + y) * 2 - 5  # Сложные выражения
```

### 🔹 Массивы
```DataCode
# Создание массивов любых типов
global numbers = [1, 2, 3, 4, 5]
global strings = ['hello', 'world', 'datacode']
global booleans = [true, false, true]
global mixed = [1, 'hello', true, 3.14]
global empty = []

# Вложенные массивы
global matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
global nested_mixed = [[1, 'a'], [true, 3.14]]

# Доступ к элементам (индексирование с 0)
print(numbers[0])        # 1
print(strings[1])        # world
print(mixed[2])          # true
print(matrix[0][1])      # 2
print(nested_mixed[1][0]) # true

# Trailing comma поддерживается
global trailing = [1, 2, 3,]

# Использование в циклах
for item in [1, 2, 3] {
    print('Item:', item)
}
```

### 🔹 Операторы сравнения
```DataCode
global eq = x == y          # Равенство
global ne = x != y          # Неравенство
global gt = x > y           # Больше
global lt = x < y           # Меньше
global ge = x >= y          # Больше или равно
global le = x <= y          # Меньше или равно
```

### 🔹 Логические операции
```DataCode
global flag1 = true
global flag2 = false
global and_result = flag1 and flag2    # Логическое И
global or_result = flag1 or flag2      # Логическое ИЛИ
global not_result = not flag1          # Логическое НЕ
global complex_logic = (x > 5) and (y < 30) or flag1
```

### 🔹 Конкатенация путей
```DataCode
global dir = basePath / 'data' / 'images'
```
• `/` используется для Path + String (контекстно определяется)

### 🔹 Сложение строк
```DataCode
global name = 'image' + '001.jpg'
global greeting = 'Hello, ' + name + '!'
```
• `+` объединяет строки

### 🔹 Обработка ошибок
```DataCode
# Базовый try/catch
try {
    global result = 10 / 0
} catch e {
    print('Ошибка перехвачена:', e)
}

# try/catch/finally
try {
    global value = process_data()
} catch e {
    print('Ошибка:', e)
} finally {
    print('Очистка выполнена')
}

# Использование throw
fn validate(age) {
    if age < 0 {
        throw 'Возраст не может быть отрицательным'
    }
    return true
}
```
• `try { ... } catch e { ... }` - перехват ошибок
• `finally { ... }` - блок, который всегда выполняется
• `throw message` - генерация исключения

---

## 🔁 Циклы

### Циклы for
```DataCode
# Цикл по массиву переменных
for file in files {
    let path = basePath / 'data' / file
    let text = read_file(path)
    print('>>', file, 'length:', text)
}

# Цикл по литералу массива
for number in [1, 2, 3, 4, 5] {
    print('Number:', number, 'Squared:', number * number)
}

# Цикл по смешанному массиву
for item in ['hello', 42, true] {
    print('Item:', item)
}

# Цикл по вложенному массиву
for row in [[1, 2], [3, 4], [5, 6]] {
    print('Row:', row, 'Sum:', sum(row))
}
```
- `for x in array { ... }` - итерация по массиву
- `x` — переменная, доступная внутри тела цикла
- Поддерживаются как переменные-массивы, так и литералы массивов

### Циклы while
```DataCode
# Простой цикл while
let counter = 0
while counter < 5 {
    print('Counter:', counter)
    counter = counter + 1
}

# Цикл с условием
let x = 10
while x > 0 {
    print('x =', x)
    x = x - 1
}
```
- `while condition { ... }` - выполнение пока условие истинно
- Поддерживаются вложенные циклы и комбинации for/while

---

## 🔧 Встроенные функции (50)

### 📁 Файловые операции
| Функция | Описание |
|---------|----------|
| `getcwd()` | Текущая директория |
| `path(string)` | Создание пути из строки |
| `read_file(path)` | Чтение файлов (.txt, .csv, .xlsx) |
| `read_file(path, sheet_name="sheet_name")` | Чтение XLSX с выбором листа по имени |
| `read_file(path, header_row)` | Чтение CSV/XLSX с выбором строки заголовка (0-based) |
| `read_file(path, header_row, sheet_name)` | Чтение XLSX с выбором строки заголовка и листа по имени |
| `read_file(path, header_row, sheet_name, header)` | Чтение с фильтрацией/переименованием колонок |

**Опциональные параметры `read_file()`:**
- `header_row` (число) - номер строки с заголовками, начиная с 0 (по умолчанию 0)
- `sheet_name` (строка) - имя листа для XLSX файлов (по умолчанию первый лист)
- `header` (массив | словарь) - фильтр колонок или переименование:
  - Массив: список имен колонок для загрузки (загружаются только указанные)
  - Словарь: соответствие оригинальных имен новым (используйте `null` для сохранения оригинального имени)

**Примеры:**
```datacode
# Базовое чтение
data = read_file(path("data.csv"))

# Чтение конкретного листа Excel
data = read_file(path("report.xlsx"), sheet_name="Sales")

# Чтение с заголовком в строке 2
data = read_file(path("data.csv"), 2)

# Комбинация: лист + строка заголовка
data = read_file(path("report.xlsx"), 1, "DataSheet")

# Загрузка только указанных колонок
sample_table = read_file(path("sample.csv"), header_row=0, header=["Name", "Age", "City", "Salary"])

# Переименование колонок при загрузке
sample_table = read_file(path("sample.csv"), header_row=0, header={"Name": "Name_A", "Age": null, "City": null, "Salary": null})
```

### 🧮 Математические функции
| Функция | Описание |
|---------|----------|
| `abs(n)` | Абсолютное значение |
| `sqrt(n)` | Квадратный корень |
| `pow(base, exp)` | Возведение в степень |
| `min(...)` | Минимальное значение |
| `max(...)` | Максимальное значение |
| `round(n)` | Округление |

### 📝 Строковые функции
| Функция | Описание |
|---------|----------|
| `len(str)` | Длина строки |
| `upper(str)` | В верхний регистр |
| `lower(str)` | В нижний регистр |
| `trim(str)` | Удаление пробелов |
| `split(str, delim)` | Разделение строки |
| `join(array, delim)` | Объединение массива |
| `contains(str, substr)` | Проверка вхождения |

### 📊 Функции массивов
| Функция | Описание |
|---------|----------|
| `push(array, item)` | Добавить элемент |
| `pop(array)` | Удалить последний |
| `unique(array)` | Уникальные элементы |
| `reverse(array)` | Обратный порядок |
| `sort(array)` | Сортировка |
| `sum(array)` | Сумма чисел |
| `average(array)` | Среднее значение |
| `count(array)` | Количество элементов |

### 📋 Табличные функции
| Функция | Описание |
|---------|----------|
| `table(data, headers)` | Создание таблицы |
| `show_table(table)` | Вывод таблицы |
| `table_info(table)` | Информация о таблице |
| `table_head(table, n)` | Первые n строк |
| `table_tail(table, n)` | Последние n строк |
| `table_select(table, cols)` | Выбор колонок |
| `table_sort(table, col, asc)` | Сортировка таблицы |

### 🔧 Утилиты
| Функция | Описание |
|---------|----------|
| `print(...)` | Вывод значений |
| `now()` | Текущее время |

---

## 🗄️ Экспорт в SQLite (--build_model)

DataCode поддерживает автоматический экспорт всех таблиц из глобальных переменных в базу данных SQLite с автоматическим определением зависимостей между таблицами.

### Основные возможности

- ✅ **Автоматический экспорт таблиц** - все таблицы из глобальных переменных экспортируются в отдельные таблицы SQLite
- ✅ **Метаданные переменных** - создается таблица `_datacode_variables` с информацией о всех глобальных переменных
- ✅ **Автоматическое определение зависимостей** - система автоматически находит связи между таблицами по ID-колонкам
- ✅ **Создание индексов** - автоматическое создание индексов для ID-колонок и внешних ключей
- ✅ **Преобразование типов** - автоматическое преобразование типов DataCode в типы SQLite

### Использование

```bash
# Экспорт с именем по умолчанию (script_name.db)
datacode load_model_data.dc --build_model

# Экспорт с указанием выходного файла
datacode load_model_data.dc --build_model output.db

# Использование переменной окружения
DATACODE_SQLITE_OUTPUT=model.db datacode load_model_data.dc --build_model
```

### Пример скрипта

```datacode
# Загрузка данных
global sales = read_file("sales.csv")
global products = read_file("products.csv")
global customers = read_file("customers.csv")

# Обработка данных
global sales_table = table(sales)
global products_table = table(products)
global customers_table = table(customers)

# Фильтрация и преобразование
global filtered_sales = table_where(sales_table, "amount > 100")
```

Выполнение:
```bash
datacode load_model_data.dc --build_model
```

### Результат экспорта

После выполнения создается SQLite база данных со следующими таблицами:

1. **Таблицы данных** - каждая глобальная переменная типа `Table` экспортируется в отдельную таблицу:
   - `sales_table` - все данные из sales
   - `products_table` - все данные из products
   - `customers_table` - все данные из customers
   - `filtered_sales` - отфильтрованные данные

2. **Таблица метаданных `_datacode_variables`** - содержит информацию о всех глобальных переменных:
   ```sql
   CREATE TABLE _datacode_variables (
       variable_name TEXT PRIMARY KEY,
       variable_type TEXT NOT NULL,      -- Table, Array, Object, Number, String, etc.
       table_name TEXT,                   -- Имя SQLite таблицы (для таблиц)
       row_count INTEGER,                 -- Количество строк (для таблиц)
       column_count INTEGER,              -- Количество колонок (для таблиц)
       created_at TEXT,                   -- Временная метка экспорта
       description TEXT,                  -- Описание (опционально)
       value TEXT                         -- Строковое представление значения
   );
   ```

3. **Автоматические зависимости** - если в таблице есть колонки с ID-подобными именами (`*_id`, `id`), система автоматически определяет связи:
   - Если в `sales_table` есть `product_id` и в `products_table` есть `id`, создается связь
   - Если в `sales_table` есть `customer_id` и в `customers_table` есть `id`, создается связь

### Алгоритм определения зависимостей

Система автоматически определяет первичные ключи и внешние ключи:

**Первичные ключи определяются по:**
- Колонкам с именем `id` типа Integer
- Колонкам с именами `*_id` типа Integer
- Колонкам с префиксом `pk_` или `key_`
- Колонкам, где все значения уникальны

**Внешние ключи определяются по:**
- Колонкам с ID-подобными именами: `*_id`, `id`, `*Id`, `*ID`
- Совпадению типов данных (Integer)
- Наличию соответствующего первичного ключа в другой таблице

### Преобразование типов

| DataCode тип | SQLite тип |
|--------------|------------|
| `Integer` | `INTEGER` |
| `Float` | `REAL` |
| `String` | `TEXT` |
| `Bool` | `INTEGER` (0/1) |
| `Date` | `TEXT` (ISO format) |
| `Currency` | `REAL` |
| `Null` | `NULL` |
| `Mixed` | `TEXT` |

### Явные связи через relate()

Функция `relate()` позволяет явно создать связи между колонками таблиц:

```datacode
global products = table(product_data, ["product_id", "name", "price"])
global sales = table(sales_data, ["sale_id", "product_id", "amount"])

# Создание явной связи
relate(products["product_id"], sales["product_id"])
```

При экспорте в SQLite:
- Явные связи, созданные через `relate()`, имеют приоритет над автоматическим определением
- Создаются полноценные FOREIGN KEY constraints (не только индексы)
- Таблицы пересоздаются с FOREIGN KEY constraints для обеспечения целостности данных

### Ограничения

- Экспортируются только **глобальные переменные** (локальные переменные не экспортируются)
- Экспортируются только переменные типа `Table` (другие типы сохраняются только в метаданных)

---

## 🧪 Пример программы
```DataCode
# Пользовательская функция для анализа массивов
fn analyze_array(arr) {
    let size = count(arr)
    let sum_val = sum(arr)
    let avg_val = average(arr)

    print('📊 Анализ массива:', arr)
    print('  Размер:', size)
    print('  Сумма:', sum_val)
    print('  Среднее:', avg_val)

    return [size, sum_val, avg_val]
}

# Работа с массивами и файлами
basePath = getcwd()
dataPath = basePath / 'examples'

# Создаем массивы данных
global numbers = [10, 20, 30, 40, 50]
global mixed_data = [1, 'test', true, 3.14]
global matrix = [[1, 2], [3, 4], [5, 6]]

print('🧮 Анализ числовых данных')
global stats = analyze_array(numbers)

print('')
print('📋 Работа с файлами')
global files = ['sample.csv', 'data.txt']

for file in files {
    let fullPath = dataPath / file
    print('📄 Обрабатываем:', file)

    # Если это CSV файл, показываем таблицу
    if contains(file, '.csv') {
        let table = read_file(fullPath)
        print('📊 Содержимое таблицы:')
        table_head(table, 3)
    }
}

print('')
print('🔢 Работа с вложенными массивами')
for row in matrix {
    let row_sum = sum(row)
    print('Строка:', row, 'Сумма:', row_sum)
}

print('✅ Анализ завершен!')
```

---

## 📦 Поддерживаемые типы

| Тип | Пример | Описание |
|-----|--------|----------|
| String | `'abc'`, `'hello.txt'` | Всегда в одинарных кавычках |
| Number | `42`, `3.14` | Целые и дробные числа |
| Bool | `true`, `false` | Логические значения |
| Array | `[1, 'hello', true]` | Массивы любых типов данных |
| Object | `{key: 'value', num: 42}` | Объекты (словари) с ключ-значение |
| Path | `base / 'file.csv'` | Строится через `/` |
| Table | `table(data, headers)` | Табличные данные |
| Null | — | Возвращается `print(...)` |


---

## ⚠️ Ошибки

Типичные сообщения об ошибках:
- Unknown variable: foo
- Invalid / expression
- Unsupported expression
- read_file() expects 1-3 arguments (path, [header_row], [sheet_name])

---

## 📚 Расширение

Проект легко расширяется:
- Добавить функции в builtins.rs
- Добавить типы в value.rs
- Добавить синтаксис в interpreter.rs

---

## 🧪 Тестирование

Выполните:
```bash
cargo test
```
Тесты проверяют:
- Объявление переменных
- Конкатенацию путей
- Вызов встроенных функций
- Исполнение for-циклов

---

## 🛠 Пример вызова из CLI
```bash
cargo run
```

---

## 🎯 Интерактивный REPL

### Запуск
```bash
cargo run
```

### Специальные команды REPL
- `help` — показать справку
- `exit` или `quit` — выйти из интерпретатора
- `clear` — очистить экран
- `vars` — показать все переменные
- `reset` — сбросить интерпретатор

### Пример сессии
```
🧠 DataCode Interactive Interpreter
>>> x = 10
✓ x = Number(10.0)
>>> y = 20
✓ y = Number(20.0)
>>> result = (x + y) * 2
✓ result = Number(60.0)
>>> print('Result is:', result)
Result is: 60
>>> vars
📊 Current Variables:
  x = Number(10.0)
  y = Number(20.0)
  result = Number(60.0)
>>> exit
Goodbye! �
```

### Многострочные конструкции
REPL поддерживает многострочный ввод для циклов и массивов:
```
>>> arr = [1, 2, 3, 4, 5]
✓ arr = Array([Number(1.0), Number(2.0), Number(3.0), Number(4.0), Number(5.0)])
>>> print(arr[0])
1
>>> nested = [[1, 2], [3, 4]]
✓ nested = Array([Array([Number(1.0), Number(2.0)]), Array([Number(3.0), Number(4.0)])])
>>> print(nested[0][1])
2
>>> for i in [1, 2, 3] {
...     print('Number:', i)
...     doubled = i * 2
...     print('Doubled:', doubled)
... }
Number: 1
Doubled: 2
Number: 2
Doubled: 4
Number: 3
Doubled: 6
```

---

## 📚 Примеры и обучение

DataCode поставляется с профессионально организованной коллекцией примеров, которые помогут вам изучить язык от основ до продвинутых техник.

### 🎯 Быстрый старт с примерами

```bash
# Простейший пример - начните здесь!
cargo run examples/ru/01-основы/hello.dc

# Работа с переменными
cargo run examples/ru/01-основы/variables.dc

# Работа с функциями
cargo run examples/ru/05-функции/simple_functions.dc

# Обработка данных из CSV
cargo run examples/ru/09-создание\ модели\ данных/05-table-joins.dc
```

### 📁 Организация примеров

Примеры организованы в тематические разделы для систематического изучения:

#### 🚀 [examples/ru/01-основы/](examples/ru/01-основы/) - Основы языка
Начните изучение DataCode с этих примеров:
- `hello.dc` - расширенный Hello World
- `variables.dc` - работа с переменными
- `arithmetic.dc` - арифметические операции
- `strings.dc` - работа со строками

#### 🔧 [examples/ru/02-синтаксис/](examples/ru/02-синтаксис/) - Синтаксис
Изучите основные конструкции языка:
- `conditionals.dc` - условные конструкции (if/else)
- `expressions.dc` - сложные выражения
- `booleans.dc` - булевы значения и логические операции

#### 📦 [examples/ru/03-типы данных/](examples/ru/03-типы%20данных/) - Система типов
Изучите типы данных и их проверку:
- `type_conversion_functions.dc` - функции преобразования типов
- `type_conversion_guide.dc` - руководство по преобразованию типов
- `type_date.dc` - работа с датами
- `objects.dc` - работа с объектами (словарями)

#### 🎯 [examples/ru/04-продвинутые/](examples/ru/04-продвинутые/) - Продвинутые техники
Рекурсия, обработка ошибок и функциональное программирование:
- `complex.dc` - комплексный пример
- `scope_demo.dc` - область видимости переменных
- `error_handling.dc` - обработка ошибок с try/catch/throw

#### 🔧 [examples/ru/05-функции/](examples/ru/05-функции/) - Функции
Примеры создания и использования функций:
- `simple_functions.dc` - простые функции
- `recursion.dc` - рекурсивные функции
- `nested_functions.dc` - вложенные вызовы функций

#### 🎪 [examples/ru/06-демонстрации/](examples/ru/06-демонстрации/) - Полные демонстрации
- `showcase.dc` - комплексная демонстрация всех возможностей

#### 🔄 [examples/ru/07-циклы/](examples/ru/07-циклы/) - Циклы
Примеры использования циклов:
- `while_loops.dc` - циклы while
- `for_loops.dc` - циклы for
- `nested_loops.dc` - вложенные циклы

#### 📊 [examples/ru/09-создание модели данных/](examples/ru/09-создание%20модели%20данных/) - Работа с данными
Мощные возможности для обработки табличных данных:
- `01-file-operations.dc` - работа с файлами и директориями
- `02-merge-tables.dc` - объединение нескольких таблиц
- `03-create-relations.dc` - создание связей между таблицами
- `05-table-joins.dc` - операции объединения таблиц

### 📖 Подробная документация

Каждый раздел содержит подробную документацию на русском языке:
- **[examples/README.md](examples/README.md)** - главная страница примеров
- Индивидуальные README.md в каждом разделе с пошаговыми объяснениями
- Рекомендуемый порядок изучения
- Практические советы и лучшие практики

### 🎓 Рекомендуемый путь обучения

1. **Основы** → `examples/ru/01-основы/hello.dc` и `variables.dc`
2. **Синтаксис** → `examples/ru/02-синтаксис/conditionals.dc` и `expressions.dc`
3. **Типы данных** → `examples/ru/03-типы данных/type_conversion_guide.dc` и `objects.dc`
4. **Функции** → `examples/ru/05-функции/simple_functions.dc` и `recursion.dc`
5. **Циклы** → `examples/ru/07-циклы/while_loops.dc` и `for_loops.dc`
6. **Продвинутые возможности** → `examples/ru/04-продвинутые/error_handling.dc`
7. **Работа с данными** → `examples/ru/09-создание модели данных/05-table-joins.dc`
8. **Полная демонстрация** → `examples/ru/06-демонстрации/showcase.dc`

---

## 📋 Техническая документация

### Документация разработчика
- **[docs/ru/builtin_functions.md](docs/ru/builtin_functions.md)** - Полное описание всех 50 встроенных функций
- **[docs/ru/data_types.md](docs/ru/data_types.md)** - Подробное описание типов данных
- **[docs/ru/table_create_function.md](docs/ru/table_create_function.md)** - Работа с таблицами
- **[docs/ru/websocket_server.md](docs/ru/websocket_server.md)** - WebSocket сервер для удаленного выполнения

---

### 📋 Планируется в будущем
- 📋 Публичная документация и стабилизация DPM (менеджер пакетов)
- 📋 Отладчик (пошаговая отладка, брейкпоинты)
- 📋 Языковой сервер (LSP) для редакторов
- 📋 Асинхронность (async/await или аналог)

---

## 🧑‍💻 Автор

Создано Igornet0.

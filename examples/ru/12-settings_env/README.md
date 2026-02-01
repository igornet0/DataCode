# settings_env в DataCode

Этот раздел демонстрирует работу с встроенным модулем `settings_env`: загрузка переменных окружения из .env файлов, приведение типов, фильтрация по префиксу, конфигурация и классы-наследники Settings с полями Field.

## Содержание

### 1. `01-базовое_использование.dc` — Базовое использование
**Описание**: Импорт модуля, загрузка .env через load_env и Settings, чтение ключей, проверка на null при отсутствии файла.

**Что изучается**:
- Импорт: `from settings_env import load_env, Settings`
- Загрузка файла: `load_env("simple.env")`, `Settings("simple.env")`
- Доступ к ключам: `env["app_name"]`, `env["debug"]` (приведение к bool)
- Отсутствующий файл: load_env возвращает `null` без падения

**Запуск**:
```bash
datacode --bin datacode examples/ru/12-settings_env/01-базовое_использование.dc
```

### 2. `02-типы_и_приведение.dc` — Типы и приведение значений
**Описание**: Загрузка .env с разными типами значений; автоматическое приведение к bool, number, string.

**Что изучается**:
- Приведение: "true"/"false" → Bool, числа → Number, остальное → String
- Ключи: bool_true, bool_false, int_num, float_num, string_val, quoted

**Запуск**:
```bash
datacode --bin datacode examples/ru/12-settings_env/02-типы_и_приведение.dc
```

### 3. `03-префикс_и_config.dc` — Префикс и config
**Описание**: Загрузка только ключей с заданным префиксом через Settings.config(env_prefix=...) и load_env(path, [], cfg).

**Что изучается**:
- `Settings.config(env_prefix="APP__")` — объект конфигурации
- `load_env("with_prefix.env", [], cfg)` — только ключи APP__* (без OTHER_KEY, DB__*)
- Отдельная загрузка с префиксом DB__

**Запуск**:
```bash
datacode --bin datacode examples/ru/12-settings_env/03-префикс_и_config.dc
```

### 4. `04-field.dc` — Дескрипторы полей Field
**Описание**: Создание дескрипторов полей в стиле Pydantic: default, обязательное поле (...), min_length, max_length, alias.

**Что изучается**:
- `Field("dev")`, `Field(default="prod")` — значение по умолчанию
- `Field(...)` — обязательное поле (required)
- `Field(min_length=1, max_length=10)`, `Field(alias="userId")` — валидация и алиас

**Запуск**:
```bash
datacode --bin datacode examples/ru/12-settings_env/04-field.dc
```

### 5. `05-класс_config.dc` — Класс-наследник Settings
**Описание**: Классы DatabaseConfig и AppConfig с model_config и полями через Field; создание экземпляра AppConfig("dev.env") и вложенный конфиг БД.

**Что изучается**:
- `cls DatabaseConfig(Settings)` с `model_config = Settings.config(env_prefix="DB__")`
- `cls AppConfig(Settings)` с полями env, debug и вложенным db: DatabaseConfig
- Вызов `AppConfig("dev.env")` и доступ к полям, в том числе `cfg["db"]["url"]`

**Запуск**:
```bash
datacode --bin datacode examples/ru/12-settings_env/05-класс_config.dc
```

### 6. `06-практический_пример.dc` — Практический пример
**Описание**: Сводный пример: загрузка конфига приложения из simple.env, переменных с префиксом APP__ и конфига БД с префиксом DB__ из dev.env.

**Что изучается**:
- Простая загрузка load_env("simple.env")
- Загрузка с префиксами APP__ и DB__ из одного файла dev.env
- Использование полей конфигурации в приложении

**Запуск**:
```bash
datacode --bin datacode examples/ru/12-settings_env/06-практический_пример.dc
```

## Изучаемые концепции

### Загрузка .env
- **load_env(path [, required_keys [, model_config]])**: Читает .env, возвращает объект с ключами (lowercase) и приведёнными значениями (bool, number, string).
- **Settings(path)**: То же, что load_env(path).

### Конфигурация
- **Settings.config(env_prefix?, extra?, env_file?, ...)**: Объект конфигурации для load_env (env_prefix, extra, case_sensitive и т.д.).
- При вызове load_env с третьим аргументом model_config загружаются только ключи с заданным префиксом; префикс в именах ключей результата отбрасывается.

### Поля Field
- **Field(default=...)**, **Field(...)**: Дескриптор поля для классов-наследников Settings (значение по умолчанию или обязательное поле).
- **Field(min_length=..., max_length=..., alias=...)**: Валидация и алиас в стиле Pydantic.

### Классы-наследники Settings
- Класс с `model_config = Settings.config(env_prefix="...")` и полями в `public:` при вызове `Config("path")` загружает .env и возвращает объект с заполненными полями (в том числе вложенными конфигами через default_factory).

## Навигация

### Предыдущий раздел
- **[11-mnist-mlp](../11-mnist-mlp/)** — Пример MNIST MLP

### Следующий раздел
- **[13-uuid](../13-uuid/)** — Модуль UUID

# Модуль settings_env — переменные окружения и конфигурация

Модуль `settings_env` предназначен для загрузки переменных окружения из .env файлов, приведения типов значений, фильтрации по префиксу, настройки конфигурации и создания классов-наследников Settings с полями через дескрипторы Field (в стиле Pydantic).

## Импорт модуля

Модуль импортируется и экспортируются функции и классы:

```datacode
from settings_env import load_env, Settings, Field
```

## load_env(path [, required_keys [, model_config]]) -> Object | null

Читает .env файл и возвращает объект с ключами (в нижнем регистре, если не задано `case_sensitive`) и приведёнными значениями (bool, number, string). При отсутствии файла возвращает `null` без выброса ошибки.

**Параметры:**
- `path` (string или Path): путь к .env файлу. Относительный путь разрешается относительно директории запускаемого скрипта (base path из file_import).
- `required_keys` (array, необязательно): массив строк — список обязательных ключей; если какой-то ключ отсутствует в файле, устанавливается нативная ошибка и возвращается `null`.
- `model_config` (Object, необязательно): объект конфигурации, возвращённый `Settings.config(...)`; применяются `env_prefix`, `env_file`, `extra`, `case_sensitive`, `env_nested_delimiter`. При заданном префиксе загружаются только ключи с этим префиксом, префикс в именах ключей результата отбрасывается.

**Возвращает:**
- `Object`: объект с ключами и приведёнными значениями.
- `null`: если файл не найден, путь некорректен или не хватает обязательных ключей.

**Пример:**
```datacode
let env = load_env("simple.env")
if env != null {
    print(env["app_name"])
    print(env["debug"])
}
```

**Примечание:** при относительном пути base path должен быть установлен (например, при запуске из файла скрипта); иначе будет установлена ошибка и возвращён `null`.

## Settings(path) -> Object | null

Эквивалент `load_env(path)`. Удобен для вызова вида `cfg = Settings("dev.env")`.

**Параметры:**
- `path` (string или Path): путь к .env файлу.

**Возвращает:** то же, что `load_env(path)`.

## Settings.config(env_prefix?, extra?, env_file?, env_file_encoding?, case_sensitive?, env_nested_delimiter?) -> Object

Возвращает объект конфигурации (SettingsConfigDict) для передачи третьим аргументом в `load_env`. Параметры задаются по порядку (именованные аргументы компилируются в позиционные).

**Параметры:**
- `env_prefix` (string, по умолчанию ""): префикс ключей в .env; загружаются только ключи с этим префиксом, префикс в результате отбрасывается.
- `extra` (string, по умолчанию "ignore"): поведение для лишних ключей.
- `env_file` (string или null): переопределение пути к файлу (если задано, используется вместо path в load_env).
- `env_file_encoding` (string, по умолчанию "utf-8"): кодировка файла.
- `case_sensitive` (bool, по умолчанию false): сохранять ли регистр имён ключей.
- `env_nested_delimiter` (string, по умолчанию "__"): разделитель для вложенных ключей.

**Возвращает:** Object с ключами `env_prefix`, `extra`, `env_file`, `env_file_encoding`, `case_sensitive`, `env_nested_delimiter`.

**Пример:**
```datacode
let cfg = Settings.config(env_prefix="APP__", extra="ignore")
let env = load_env("with_prefix.env", [], cfg)
# env содержит только ключи, бывшие APP__*, без префикса в именах
```

## Field(...) — дескриптор полей

Создаёт дескриптор поля для использования в классах-наследниках Settings (синтаксис в стиле Pydantic). Поддерживаются аргументы: `default`, `default_factory`, `alias`, `title`, `description`, `examples`, `exclude`, `include`, `const`, `gt`, `ge`, `lt`, `le`, `multiple_of`, `min_length`, `max_length`, `regex`, `deprecated`, `repr`, `json_schema_extra`, `validate_default`, `frozen`.

**Основные варианты:**
- `Field(default=...)` или `Field("значение")` — значение по умолчанию.
- `Field(...)` — обязательное поле (required).
- `Field(default_factory=SomeClass)` — фабрика по умолчанию (например, для вложенного конфига).
- `Field(min_length=n, max_length=m)` — валидация длины (строка, массив, кортеж).
- `Field(alias="имя")` — алиас имени ключа в .env.
- `Field(gt=..., ge=..., lt=..., le=..., multiple_of=...)` — ограничения для чисел.
- `Field(regex="шаблон")` — проверка строки по регулярному выражению.

**Пример:**
```datacode
env: str = Field(default="dev")
debug: bool = Field(default=true)
url: str = Field(...)
name: str = Field(min_length=1, max_length=10)
db: DatabaseConfig = Field(default_factory=DatabaseConfig)
```

## Классы-наследники Settings

Класс с базовым классом `Settings`, атрибутом класса `model_config = Settings.config(env_prefix="...")` и полями в `public:` при вызове как конструктор с путём к .env загружает файл и возвращает объект с заполненными полями. Вложенные конфиги задаются через `Field(default_factory=ДругойConfig)`.

**Пример:**
```datacode
cls DatabaseConfig(Settings) {
    model_config = Settings.config(env_prefix="DB__", extra="ignore")
    public:
        url: str = Field(...)
}

cls AppConfig(Settings) {
    model_config = Settings.config(env_prefix="APP__", extra="ignore")
    public:
        env: str = Field(default="dev")
        debug: bool = Field(default=true)
        db: DatabaseConfig = Field(default_factory=DatabaseConfig)
}

let cfg = AppConfig("dev.env")
print(cfg.env)
print(cfg.db.url)
```

## Приведение типов значений

При чтении .env строковые значения автоматически приводятся к типам DataCode:

- `"true"` / `"false"` (без учёта регистра) → Bool
- Числовая строка (целое или с плавающей точкой) → Number
- Остальное → String

Кавычки вокруг значения в .env снимаются при разборе.

## Примеры

Практические примеры расположены в каталогах:

- **Русский:** [examples/ru/12-settings_env/](../../../examples/ru/12-settings_env/)
- **English:** [examples/en/12-settings-env/](../../../examples/en/12-settings-env/)

### Содержание примеров (ru)

1. **01-базовое_использование.dc** — импорт, load_env, Settings, чтение ключей, обработка null при отсутствии файла.
2. **02-типы_и_приведение.dc** — приведение "true"/"false" и чисел к Bool/Number, остальное — String.
3. **03-префикс_и_config.dc** — Settings.config(env_prefix=...), load_env с config, загрузка только ключей с префиксом.
4. **04-field.dc** — Field(default=...), Field(...), min_length, max_length, alias.
5. **05-класс_config.dc** — классы DatabaseConfig и AppConfig с model_config и вложенным конфигом через default_factory.
6. **06-практический_пример.dc** — сводный пример: simple.env, префиксы APP__ и DB__ из dev.env.

### Содержание примеров (en)

1. **01-basic-usage.dc** — import, load_env, Settings, reading keys, null when file is missing.
2. **02-types-and-coercion.dc** — coercion to Bool, Number, String.
3. **03-prefix-and-config.dc** — Settings.config(env_prefix=...), load_env with config.
4. **04-field.dc** — Field descriptors: default, required, min_length, max_length, alias.
5. **05-settings-class.dc** — DatabaseConfig and AppConfig with model_config and nested config.
6. **06-practical-example.dc** — combined example with simple.env and dev.env (APP__, DB__).

**Запуск примера (из корня проекта):**
```bash
datacode --bin datacode examples/ru/12-settings_env/01-базовое_использование.dc
```

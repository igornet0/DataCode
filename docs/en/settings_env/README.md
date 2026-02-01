# settings_env Module — Environment Variables and Configuration

The `settings_env` module loads environment variables from .env files, coerces value types, filters by prefix, configures loading options, and supports Settings subclasses with Field descriptors (Pydantic-style).

## Module Import

Import the module and use the exported functions and classes:

```datacode
from settings_env import load_env, Settings, Field
```

## load_env(path [, required_keys [, model_config]]) -> Object | null

Reads a .env file and returns an object with keys (lowercase unless `case_sensitive` is set) and coerced values (bool, number, string). If the file is missing, returns `null` without throwing.

**Parameters:**
- `path` (string or Path): path to the .env file. Relative paths are resolved against the directory of the script being run (base path from file_import).
- `required_keys` (array, optional): list of required key names (strings); if any key is missing in the file, a native error is set and `null` is returned.
- `model_config` (Object, optional): configuration object returned by `Settings.config(...)`; applies `env_prefix`, `env_file`, `extra`, `case_sensitive`, `env_nested_delimiter`. When a prefix is set, only keys with that prefix are loaded, and the prefix is stripped from result key names.

**Returns:**
- `Object`: object with keys and coerced values.
- `null`: if the file is not found, path is invalid, or required keys are missing.

**Example:**
```datacode
let env = load_env("simple.env")
if env != null {
    print(env["app_name"])
    print(env["debug"])
}
```

**Note:** When using a relative path, the base path must be set (e.g. when running from a script file); otherwise an error is set and `null` is returned.

## Settings(path) -> Object | null

Same as `load_env(path)`. Convenient for calls like `cfg = Settings("dev.env")`.

**Parameters:**
- `path` (string or Path): path to the .env file.

**Returns:** same as `load_env(path)`.

## Settings.config(env_prefix?, extra?, env_file?, env_file_encoding?, case_sensitive?, env_nested_delimiter?) -> Object

Returns a configuration object (SettingsConfigDict) to pass as the third argument to `load_env`. Parameters are passed by position (named arguments are compiled as positional).

**Parameters:**
- `env_prefix` (string, default ""): prefix for keys in the .env file; only keys with this prefix are loaded, and the prefix is stripped from result key names.
- `extra` (string, default "ignore"): behavior for extra keys.
- `env_file` (string or null): override path to the file (if set, used instead of the path argument to load_env).
- `env_file_encoding` (string, default "utf-8"): file encoding.
- `case_sensitive` (bool, default false): whether to preserve key name case.
- `env_nested_delimiter` (string, default "__"): delimiter for nested key names.

**Returns:** Object with keys `env_prefix`, `extra`, `env_file`, `env_file_encoding`, `case_sensitive`, `env_nested_delimiter`.

**Example:**
```datacode
let cfg = Settings.config(env_prefix="APP__", extra="ignore")
let env = load_env("with_prefix.env", [], cfg)
# env contains only keys that were APP__*, with prefix stripped from names
```

## Field(...) — Field descriptors

Creates a field descriptor for use in Settings subclasses (Pydantic-style syntax). Supported arguments include: `default`, `default_factory`, `alias`, `title`, `description`, `examples`, `exclude`, `include`, `const`, `gt`, `ge`, `lt`, `le`, `multiple_of`, `min_length`, `max_length`, `regex`, `deprecated`, `repr`, `json_schema_extra`, `validate_default`, `frozen`.

**Common variants:**
- `Field(default=...)` or `Field("value")` — default value.
- `Field(...)` — required field.
- `Field(default_factory=SomeClass)` — default factory (e.g. for nested config).
- `Field(min_length=n, max_length=m)` — length validation (string, array, tuple).
- `Field(alias="name")` — alias for the key name in .env.
- `Field(gt=..., ge=..., lt=..., le=..., multiple_of=...)` — numeric constraints.
- `Field(regex="pattern")` — string validation by regular expression.

**Example:**
```datacode
env: str = Field(default="dev")
debug: bool = Field(default=true)
url: str = Field(...)
name: str = Field(min_length=1, max_length=10)
db: DatabaseConfig = Field(default_factory=DatabaseConfig)
```

## Settings subclasses

A class that extends `Settings`, has class attribute `model_config = Settings.config(env_prefix="...")`, and fields in `public:` will, when called as a constructor with a path to a .env file, load that file and return an object with filled fields. Nested configs are defined via `Field(default_factory=OtherConfig)`.

**Example:**
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

## Value type coercion

When reading .env, string values are coerced to DataCode types:

- `"true"` / `"false"` (case-insensitive) → Bool
- Numeric string (integer or float) → Number
- Everything else → String

Quotes around values in .env are stripped during parsing.

## Examples

Example scripts are located in:

- **English:** [examples/en/12-settings-env/](../../../examples/en/12-settings-env/)
- **Russian:** [examples/ru/12-settings_env/](../../../examples/ru/12-settings_env/)

### Example contents (en)

1. **01-basic-usage.dc** — import, load_env, Settings, reading keys, null when file is missing.
2. **02-types-and-coercion.dc** — coercion of "true"/"false" and numbers to Bool/Number, rest to String.
3. **03-prefix-and-config.dc** — Settings.config(env_prefix=...), load_env with config, loading only prefixed keys.
4. **04-field.dc** — Field(default=...), Field(...), min_length, max_length, alias.
5. **05-settings-class.dc** — DatabaseConfig and AppConfig with model_config and nested config via default_factory.
6. **06-practical-example.dc** — combined example: simple.env, APP__ and DB__ prefixes from dev.env.

### Example contents (ru)

1. **01-базовое_использование.dc** — импорт, load_env, Settings, чтение ключей, null при отсутствии файла.
2. **02-типы_и_приведение.dc** — приведение типов к Bool, Number, String.
3. **03-префикс_и_config.dc** — префикс и config.
4. **04-field.dc** — дескрипторы Field.
5. **05-класс_config.dc** — класс-наследник Settings.
6. **06-практический_пример.dc** — практический пример.

**Run an example (from project root):**
```bash
datacode --bin datacode examples/en/12-settings-env/01-basic-usage.dc
```

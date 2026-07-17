# settings_env in DataCode

This section demonstrates the built-in `settings_env` module: loading environment variables from .env files, type coercion, prefix filtering, configuration, and Settings subclasses with Field descriptors.

## Contents

### 1. `01-basic-usage.dc` — Basic usage
**Description**: Import the module, load .env via load_env and Settings, read keys, handle null when file is missing.

**What you'll learn**:
- Import: `from settings_env import load_env, Settings`
- Load file: `load_env("simple.env")`, `Settings("simple.env")`
- Access keys: `env["app_name"]`, `env["debug"]` (bool coercion)
- Missing file: load_env returns `null` without crashing

**Run**:
```bash
datacode --bin datacode examples/en/12-settings-env/01-basic-usage.dc
```

### 2. `02-types-and-coercion.dc` — Types and value coercion
**Description**: Load .env with different value types; automatic coercion to bool, number, string.

**What you'll learn**:
- Coercion: "true"/"false" → Bool, numbers → Number, else → String
- Keys: bool_true, bool_false, int_num, float_num, string_val, quoted

**Run**:
```bash
datacode --bin datacode examples/en/12-settings-env/02-types-and-coercion.dc
```

### 3. `03-prefix-and-config.dc` — Prefix and config
**Description**: Load only keys with a given prefix via Settings.config(env_prefix=...) and load_env(path, [], cfg).

**What you'll learn**:
- `Settings.config(env_prefix="APP__")` — config object
- `load_env("with_prefix.env", [], cfg)` — only APP__* keys (no OTHER_KEY, DB__*)
- Separate load with prefix DB__

**Run**:
```bash
datacode --bin datacode examples/en/12-settings-env/03-prefix-and-config.dc
```

### 4. `04-field.dc` — Field descriptors
**Description**: Pydantic-style field descriptors: default, required field (...), min_length, max_length, alias.

**What you'll learn**:
- `Field("dev")`, `Field(default="prod")` — default value
- `Field(...)` — required field
- `Field(min_length=1, max_length=10)`, `Field(alias="userId")` — validation and alias

**Run**:
```bash
datacode --bin datacode examples/en/12-settings-env/04-field.dc
```

### 5. `05-settings-class.dc` — Settings subclass
**Description**: DatabaseConfig and AppConfig classes with model_config and Field-based fields; creating instance AppConfig("dev.env") and nested DB config.

**What you'll learn**:
- `cls DatabaseConfig(Settings)` with `model_config = Settings.config(env_prefix="DB__")`
- `cls AppConfig(Settings)` with fields env, debug and nested db: DatabaseConfig
- Calling `AppConfig("dev.env")` and accessing fields, including `cfg.db.url`

**Run**:
```bash
datacode --bin datacode examples/en/12-settings-env/05-settings-class.dc
```

### 6. `06-practical-example.dc` — Practical example
**Description**: Summary example: load app config from simple.env, variables with prefix APP__ and DB config with prefix DB__ from dev.env.

**What you'll learn**:
- Simple load_env("simple.env")
- Loading with prefixes APP__ and DB__ from a single dev.env file
- Using config fields in an application

**Run**:
```bash
datacode --bin datacode examples/en/12-settings-env/06-practical-example.dc
```

## Concepts

### Loading .env
- **load_env(path [, required_keys [, model_config]])**: Reads .env, returns object with keys (lowercase) and coerced values (bool, number, string).
- **Settings(path)**: Same as load_env(path).

### Configuration
- **Settings.config(env_prefix?, extra?, env_file?, ...)**: Config object for load_env (env_prefix, extra, case_sensitive, etc.).
- When calling load_env with third argument model_config, only keys with the given prefix are loaded; the prefix is stripped from result key names.

### Field descriptors
- **Field(default=...)**, **Field(...)**: Field descriptor for Settings subclasses (default value or required field).
- **Field(min_length=..., max_length=..., alias=...)**: Pydantic-style validation and alias.

### Settings subclasses
- A class with `model_config = Settings.config(env_prefix="...")` and fields in `public:` when called as `Config("path")` loads .env and returns an object with filled fields (including nested configs via default_factory).

## Navigation

### Previous section
- **[11-mnist-mlp](../11-mnist-mlp/)** — MNIST MLP example

### Next section
- **[13-uuid](../13-uuid/)** — UUID module

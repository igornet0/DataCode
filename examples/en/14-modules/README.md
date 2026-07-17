# 📦 Modules and Packages

Example of creating modules and using them: package layout, imports, and entry point.

## Structure

```
15-modules/
├── main.dc                 # Entry point: import from core.config and call main
├── core/
│   ├── __lib__.dc          # Package core (required for importing core.config)
│   └── config/
│       ├── __lib__.dc      # Package core.config: get_settings, load_settings
│       ├── base.dc         # Base constants and configs (DevAppBaseConfig, ProdAppBaseConfig)
│       ├── config.dc       # Classes DatabaseConfig, ConfigApp
│       ├── dev_config.dc   # DevSettings(ConfigApp)
│       └── prod_config.dc  # ProdSettings(ConfigApp)
└── README.md
```

## What this demonstrates

- **Package** — a directory with `__lib__.dc`; it is imported as a single module.
- **Nested packages** — import by dotted name: `from core.config import ...`.
- **Cross-module imports** — in `config`, modules use `from base import ...`, `from config import ...`.
- **Global state** — in `core.config`’s `__lib__.dc`: `global settings`, and functions `load_settings(env)` and `get_settings()`.

## Run

From the project root:

```bash
cargo run --bin datacode examples/en/15-modules/main.dc
cargo run --bin datacode examples/en/15-modules/main.dc dev
cargo run --bin datacode examples/en/15-modules/main.dc prod
```

Default environment is `dev`. For a full example with .env loading see `sandbox/web_api` and the `settings_env` module.

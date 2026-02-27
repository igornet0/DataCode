# 📦 Модули и пакеты

Пример создания модулей и их использования: структура пакетов, импорты и точка входа.

## Структура

```
15-модули/
├── main.dc                 # Точка входа: импорт из core.config и вызов main
├── core/
│   ├── __lib__.dc          # Пакет core (нужен для импорта core.config)
│   └── config/
│       ├── __lib__.dc      # Пакет core.config: get_settings, load_settings
│       ├── base.dc         # Базовые константы и конфиги (DevAppBaseConfig, ProdAppBaseConfig)
│       ├── config.dc       # Классы DatabaseConfig, ConfigApp
│       ├── dev_config.dc   # DevSettings(ConfigApp)
│       └── prod_config.dc  # ProdSettings(ConfigApp)
└── README.md
```

## Что демонстрируется

- **Пакет** — папка с файлом `__lib__.dc`; импортируется как один модуль.
- **Вложенные пакеты** — импорт по составному имени: `from core.config import ...`.
- **Импорты между модулями** — в `config` используются `from base import ...`, `from config import ...`.
- **Глобальное состояние** — в `__lib__.dc` пакета `core.config`: `global settings`, функции `load_settings(env)` и `get_settings()`.

## Запуск

Из корня проекта:

```bash
cargo run --bin datacode examples/ru/15-модули/main.dc
cargo run --bin datacode examples/ru/15-модули/main.dc dev
cargo run --bin datacode examples/ru/15-модули/main.dc prod
```

По умолчанию используется среда `dev`. Для полного примера с загрузкой из .env см. `sandbox/web_api` и модуль `settings_env`.

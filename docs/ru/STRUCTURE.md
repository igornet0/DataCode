# Архитектура документации DataCode

Единая **продакшен**-структура. Весь контент — только в канонических путях ниже.

## Дерево

```
docs/
├── README.md
├── STRUCTURE.md
│
├── en/
│   ├── README.md             # User-facing EN documentation
│   ├── 0-syntax/             # Lessons 01–12
│   ├── 1-examples/           # Guide for examples/en/
│   ├── 2-language/           # Types, functions, modules
│   └── 200-developers/       # VM, ABI, native modules (EN)
│
└── ru/
    ├── README.md
    ├── 0-синтаксис/          # Уроки 01–12
    ├── 1-примеры/            # Гид examples/ru/
    ├── 2-язык/               # Типы, функции, модули
    └── 200-разработчикам/      # VM, ABI (RU)
```

## Канонические пути

| Аудитория | RU | EN |
|-----------|----|----|
| Синтаксис | `ru/0-синтаксис/NN-*.md` | `en/0-syntax/NN-*.md` |
| Практика | `ru/1-примеры/` + `examples/ru/` | `en/1-examples/` + `examples/en/` |
| Типы | `ru/2-язык/типы-данных/` | `en/2-language/data-types/` |
| Функции | `ru/2-язык/функции/` | `en/2-language/functions/` |
| Модули | `ru/2-язык/модули/<name>/` | `en/2-language/modules/<name>/` |
| Таблицы / JOIN | `ru/2-язык/таблицы/` | `en/2-language/tables/` |
| WebSocket | `ru/2-язык/сервисы/` | `en/2-language/services/` |
| VM / ABI | `ru/200-разработчикам/` | `en/200-developers/` |

## Нумерация

- **0, 1, 2, 200** — уровни: синтаксис → примеры → язык → ядро.
- **01–12** в `0-синтаксис/` — порядок уроков.
- **01–18** в `examples/ru/` — совпадает с `1-примеры/`.

## Связь с кодом

| Документ | Исходники |
|----------|-----------|
| `ru/2-язык/функции/README.md` | `src/vm/globals.rs` |
| `ru/200-разработчикам/vm_architecture.md` | `src/vm/`, `src/bytecode/` |
| `en/200-developers/native_call_lifecycle.md` | `src/vm/native_loader.rs`, `datacode_abi/` |

## Правила для PR

1. Пользовательский материал RU → `0-` / `1-` / `2-`.
2. Один канонический файл на тему — без дубликатов в корне `ru/`.
3. Нумерация примеров = нумерация в `1-примеры/`.
4. Dev docs EN → `en/200-developers/`; RU → `ru/200-разработчикам/`.

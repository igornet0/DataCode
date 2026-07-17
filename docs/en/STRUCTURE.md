# DataCode Documentation Architecture

Unified **production** layout. All content lives only in the canonical paths below.

## Tree

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
    ├── 0-синтаксис/          # Lessons 01–12
    ├── 1-примеры/            # Guide for examples/ru/
    ├── 2-язык/               # Types, functions, modules
    └── 200-разработчикам/      # VM, ABI (RU)
```

## Canonical paths

| Audience | RU | EN |
|----------|----|----|
| Syntax | `ru/0-синтаксис/NN-*.md` | `en/0-syntax/NN-*.md` |
| Practice | `ru/1-примеры/` + `examples/ru/` | `en/1-examples/` + `examples/en/` |
| Types | `ru/2-язык/типы-данных/` | `en/2-language/data-types/` |
| Functions | `ru/2-язык/функции/` | `en/2-language/functions/` |
| Modules | `ru/2-язык/модули/<name>/` | `en/2-language/modules/<name>/` |
| Tables / JOIN | `ru/2-язык/таблицы/` | `en/2-language/tables/` |
| WebSocket | `ru/2-язык/сервисы/` | `en/2-language/services/` |
| VM / ABI | `ru/200-разработчикам/` | `en/200-developers/` |

## Numbering

- **0, 1, 2, 200** — levels: syntax → examples → language → core.
- **01–12** in `0-syntax/` — lesson order.
- **01–18** in `examples/en/` — matches `1-examples/`.

## Code mapping

| Document | Source code |
|----------|-------------|
| `en/2-language/functions/README.md` | `src/vm/globals.rs` |
| `en/200-developers/vm_architecture.md` | `src/vm/`, `src/bytecode/` |
| `en/200-developers/native_call_lifecycle.md` | `src/vm/native_loader.rs`, `datacode_abi/` |

## PR rules

1. User-facing EN material → `0-` / `1-` / `2-`.
2. One canonical file per topic — no duplicates in `docs/en/` root.
3. Example numbering = numbering in `1-examples/`.
4. Dev docs EN → `en/200-developers/`; RU → `ru/200-разработчикам/`.

# DataCode Documentation

**Russian production layout:** [STRUCTURE.md](./ru/STRUCTURE.md)  
**English production layout:** [STRUCTURE.md](./en/STRUCTURE.md)

## Languages

| Language | User docs | Developer docs |
|----------|-----------|----------------|
| **English** | [docs/en/](./en/README.md) — sections **0 · 1 · 2 · 200** | [en/200-developers/](./en/200-developers/README.md) |
| **Русский** | [docs/ru/](./ru/README.md) — sections **0 · 1 · 2 · 200** | [ru/200-разработчикам/](./ru/200-разработчикам/README.md) |

## Quick start

**English:**
1. [Lesson 01 — Basics](./en/0-syntax/01-basics.md)
2. Run [`examples/en/01-basics/hello.dc`](../examples/en/01-basics/hello.dc)
3. Follow [1 — Examples guide](./en/1-examples/README.md)

**Russian:**
1. [Урок 01 — Основы](./ru/0-синтаксис/01-основы.md)
2. Run [`examples/ru/01-основы/привет.dc`](../examples/ru/01-основы/привет.dc)
3. Follow [1 — Примеры](./ru/1-примеры/README.md)

## Developer docs (ABI / native modules)

| Topic | 🇬🇧 English | 🇷🇺 Russian |
|-------|---------|---------|
| `.dcmodule` format | [dcmodule_artifact.md](./en/200-developers/dcmodule_artifact.md) | [dcmodule-artifact.md](./ru/200-разработчикам/dcmodule-artifact.md) |
| Native call path | [native_call_lifecycle.md](./en/200-developers/native_call_lifecycle.md) | [native-call-lifecycle.md](./ru/200-разработчикам/native-call-lifecycle.md) |
| Custom operators | [operator_descriptor.md](./en/200-developers/operator_descriptor.md) | [operator-descriptor.md](./ru/200-разработчикам/operator-descriptor.md) |
| Mutating natives | [vm_mutating_natives_audit.md](./en/200-developers/vm_mutating_natives_audit.md) | [vm-mutating-natives-audit.md](./ru/200-разработчикам/vm-mutating-natives-audit.md) |

## Examples

- [examples/en/](../examples/en/README.md) — numbered course (matches `docs/en/1-examples/`)
- [examples/ru/](../examples/ru/README.md) — numbered course (matches `docs/ru/1-примеры/`)

## Project

- [Main README](../README.md)
- [INSTALL](../INSTALL.md)

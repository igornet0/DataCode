# DataCode Documentation (English)

Unified **production** documentation: numbered sections from your first script to VM internals.

**Structure map:** [docs/STRUCTURE.md](./STRUCTURE.md)

---

## Sections

| # | Section | Audience | Contents |
|---|---------|----------|----------|
| **0** | [Syntax](./0-syntax/README.md) | Beginners | Lessons **01–12**: how to write code |
| **1** | [Examples](./1-examples/README.md) | Beginners | Hands-on → [`examples/en/`](../../examples/en/) |
| **2** | [Language](./2-language/README.md) | Users | Types, 117 functions, tables, modules |
| **200** | [Developers](./200-developers/README.md) | Core | VM, ABI, `.dcmodule`, native lifecycle |

---

## Quick start

```bash
datacode examples/en/01-basics/hello.dc
```

| Step | Theory | Practice |
|------|--------|----------|
| 1 | [01-basics](./0-syntax/01-basics.md) | [`01-basics`](../../examples/en/01-basics/) |
| 2 | [04-if-else](./0-syntax/04-if-else.md) | [`02-syntax`](../../examples/en/02-syntax/) |
| 3 | [data-types](./2-language/data-types/README.md) | [`03-data-types`](../../examples/en/03-data-types/) |
| 4 | [06-functions](./0-syntax/06-functions.md) | [`04-functions`](../../examples/en/04-functions/) |
| 5 | [05-loops](./0-syntax/05-loops.md) | [`06-loops`](../../examples/en/06-loops/) |

Full path: [1 — Examples](./1-examples/README.md).  
Builtin coverage matrix: [COVERAGE.md](./COVERAGE.md) (`python3 scripts/generate_coverage.py`).

---

## Reference (section 2)

### Types and functions

- [Data types](./2-language/data-types/README.md)
- [Built-in functions — 117 fn](./2-language/functions/README.md)
- [Tables and JOIN](./2-language/tables/README.md)
- [Scoping and closures](./2-language/scoping-and-closures.md)
- [Special methods](./2-language/special-methods.md)

### Built-in modules

All modules live in **[2-language/modules/](./2-language/modules/README.md)**:

`plot` · `uuid` · `settings_env` · `system` · `database_engine` · `heapq` · `grid` · `pathfind` · `crypto` · `ml`

### Services

- [WebSocket server](./2-language/services/websocket-server.md)

---

## Examples by topic (section 1)

| # | Guide | Folder |
|---|-------|--------|
| 01–06 | [main course](./1-examples/README.md#recommended-order-main-course) | `examples/en/01` … `06`, `09` |
| 07–18 | [thematic](./1-examples/README.md#thematic-sections) | websocket, plot, database, … |

---

## Links

- [Main docs/](../README.md) (EN + RU)
- [Examples](../../examples/en/README.md)
- [Project README](../../README.md)

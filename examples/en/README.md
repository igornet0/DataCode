# 📚 DataCode Examples (English)

Practical `.dc` files organized **as a course**: folder numbering matches [`docs/en/1-examples/`](../docs/en/1-examples/README.md).

## Structure (recommended order)

### Main course

| # | Folder | Description | README |
|---|--------|-------------|--------|
| 01 | [01-basics](01-basics/) | `print`, variables, arithmetic, classes | [docs](../docs/en/1-examples/01-basics.md) |
| 02 | [02-syntax](02-syntax/) | `if`/`else`, expressions, boolean logic | [docs](../docs/en/1-examples/02-syntax.md) |
| 03 | [03-data-types](03-data-types/) | arrays, objects, files, tables | [docs](../docs/en/1-examples/03-data-types.md) |
| 04 | [04-functions](04-functions/) | `fn`, types, recursion, `stream fn` | [docs](../docs/en/1-examples/04-functions.md) |
| 05 | [05-demonstrations](05-demonstrations/) | full language demonstration | [docs](../docs/en/1-examples/05-demonstrations.md) |
| 06 | [06-loops](06-loops/) | `for`, `while`, nested loops | [docs](../docs/en/1-examples/06-loops.md) |
| 09 | [09-advanced](09-advanced/) | errors, scope, algorithms | [docs](../docs/en/1-examples/09-advanced.md) |

### Thematic sections

| # | Folder | Description |
|---|--------|-------------|
| 07 | [07-websocket](07-websocket/) | WebSocket server (`dc/`), clients (`python/`, `node/`, `bash/`, `html/`) |
| 08 | [08-data-model-creation](08-data-model-creation/) | CSV/XLSX, JOIN, SQLite model |
| 10 | [10-plot](10-plot/) | `plot` module |
| 11 | [11-settings-env](11-settings-env/) | `.env`, Settings, Field |
| 12 | [12-uuid](12-uuid/) | UUID v4/v7, v3/v5 |
| 13 | [13-database](13-database/) | `database_engine` |
| 14 | [14-modules](14-modules/) | packages, `__lib__.dc` |
| 15 | [15-datasource](15-datasource/) | HTTP / file / SQLite DataSource |
| 16 | [16-crypto](16-crypto/) | `sha256`, `random`, `crypto` module |
| 17 | [17-process-gpu](17-process-gpu/) | `system`: GPU, benchmarks |
| 18 | [18-debug](18-debug/) | `debug.operators()` |

## Quick start

```bash
# 1. First script
datacode examples/en/01-basics/hello.dc

# 2. Conditionals
datacode examples/en/02-syntax/conditionals.dc

# 3. Functions (numbered files inside the folder)
datacode examples/en/04-functions/simple_functions.dc

# 4. Loops
datacode examples/en/06-loops/for_loops.dc

# 5. Full demo
datacode examples/en/05-demonstrations/showcase.dc
```

## File numbering inside folders

In **04-functions**, **06-loops**, **10-plot**, **11-settings-env**, **12-uuid**, and **08-data-model-creation**, files are numbered `01-…`, `02-…` — follow them in order.

## Documentation links

| Task | Syntax | Function reference |
|------|--------|-------------------|
| How to write `if`, `for`, `fn` | [0-syntax](../docs/en/0-syntax/README.md) | — |
| What `table()`, `read()` do | — | [2-language/functions](../docs/en/2-language/functions/README.md) |
| Data types | — | [2-language/data-types](../docs/en/2-language/data-types/README.md) |

Full documentation: [docs/en/README.md](../docs/en/README.md)

---

**Happy learning DataCode!** 🧠✨

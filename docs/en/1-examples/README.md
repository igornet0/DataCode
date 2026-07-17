# 1 — DataCode Examples

Practical `.dc` files with explanations. Source files: [`examples/en/`](../../../examples/en/).

Numbering **matches** the folders in `examples/en/` — work through them in order, like lessons.

## Recommended order (core course)

| # | Section | Document | Folder |
|---|---------|----------|--------|
| 01 | Basics | [01-basics](./01-basics.md) | [`01-basics`](../../../examples/en/01-basics/) |
| 02 | Syntax | [02-syntax](./02-syntax.md) | [`02-syntax`](../../../examples/en/02-syntax/) |
| 03 | Data types | [03-data-types](./03-data-types.md) | [`03-data-types`](../../../examples/en/03-data-types/) |
| 04 | Functions | [04-functions](./04-functions.md) | [`04-functions`](../../../examples/en/04-functions/) |
| 05 | Demonstrations | [05-demonstrations](./05-demonstrations.md) | [`05-demonstrations`](../../../examples/en/05-demonstrations/) |
| 06 | Loops | [06-loops](./06-loops.md) | [`06-loops`](../../../examples/en/06-loops/) |
| 09 | Advanced | [09-advanced](./09-advanced.md) | [`09-advanced`](../../../examples/en/09-advanced/) |

> **Note:** loops (06) are best studied after functions (04), but before advanced topics (09). The demonstration (05) can be done at any point after 01–04.

## Thematic sections

| # | Section | Document | Folder |
|---|---------|----------|--------|
| 07 | WebSocket | [07-websocket](./07-websocket.md) | [`07-websocket`](../../../examples/en/07-websocket/) |
| 08 | Data models and tables | [08-data-model-creation](./08-data-model-creation.md) | [`08-data-model-creation`](../../../examples/en/08-data-model-creation/) |
| 10 | Charts (`plot`) | [10-plot](./10-plot.md) | [`10-plot`](../../../examples/en/10-plot/) |
| 11 | `settings_env` | [11-settings-env](./11-settings-env.md) | [`11-settings-env`](../../../examples/en/11-settings-env/) |
| 12 | `uuid` | [12-uuid](./12-uuid.md) | [`12-uuid`](../../../examples/en/12-uuid/) |
| 13 | `database_engine` | [13-database](./13-database.md) | [`13-database`](../../../examples/en/13-database/) |
| 14 | Modules and packages | [14-modules](./14-modules.md) | [`14-modules`](../../../examples/en/14-modules/) |
| 15 | DataSource | [15-datasource](./15-datasource.md) | [`15-datasource`](../../../examples/en/15-datasource/) |
| 16 | Cryptography | [16-crypto](./16-crypto.md) | [`16-crypto`](../../../examples/en/16-crypto/) |
| 17 | GPU / processes | [17-process-gpu](./17-process-gpu.md) | [`17-process-gpu`](../../../examples/en/17-process-gpu/) |
| 18 | VM debugging | [18-debug](./18-debug.md) | [`18-debug`](../../../examples/en/18-debug/) |

### ML (no examples folder)

| Section | Document |
|---------|----------|
| MNIST / ML | [11-mnist-mlp](./11-mnist-mlp.md) — theory and links to [2-language/modules/ml-module](../2-language/modules/ml-module.md) |

## How to run

```bash
datacode examples/en/01-basics/hello.dc
```

Or from the repository root:

```bash
cargo run --bin datacode examples/en/14-modules/main.dc
```

## Connection: syntax ↔ semantics ↔ practice

| What you study | Syntax (0) | Semantics (2) | Examples (1) |
|----------------|------------|---------------|--------------|
| Variables | [01-basics](../0-syntax/01-basics.md) | [data-types](../2-language/data-types/README.md) | [01-basics](./01-basics.md) |
| if/else | [04-if-else](../0-syntax/04-if-else.md) | [numbers-logic-null](../2-language/data-types/numbers-logic-null.md) | [02-syntax](./02-syntax.md) |
| Arrays | [03-arrays](../0-syntax/03-arrays.md) | [arrays](../2-language/data-types/arrays.md) | [03-data-types](./03-data-types.md) |
| Functions | [06-functions](../0-syntax/06-functions.md) | [functions](../2-language/functions/README.md) | [04-functions](./04-functions.md) |
| Tables | — | [tables](../2-language/tables/) | [08-data-model-creation](./08-data-model-creation.md) |
| Built-in functions | — | [functions/README](../2-language/functions/README.md) (117 fn) | all sections 01–17 |

Full index: [`examples/en/README.md`](../../../examples/en/README.md)

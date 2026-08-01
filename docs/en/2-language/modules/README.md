# Built-in Modules

Documentation for modules shipped with DataCode.

| Module | Description |
|--------|-------------|
| [plot](./plot/README.md) | Charts, images, windows |
| [uuid](./uuid/README.md) | UUID generation |
| [settings_env](./settings_env/README.md) | Environment variables from `.env` |
| [system](./system/README.md) | OS, runtime, sandbox |
| [database](./database/README.md) | `database_engine`, SQL, SQLite |
| [heapq](./heapq/README.md) | Min-heap (priority queue) |
| [pathfind](./pathfind/README.md) | Native grid A* (production SLA) |
| [grid](./grid/README.md) | Flat buffers + `grid.astar` for DC A* |
| [crypto](./crypto/README.md) | Argon2id, bcrypt, `secure_compare` |
| [web](./web/README.md) | HTTP client, Chromium browser automation, HTML→table |
| [debug](./debug/README.md) | `debug.operators()` — VM operator table |

### Installable packages (DPM)

| Module | Description |
|--------|-------------|
| [ml](./ml-module.md) | Machine learning — `dpm add ml`, [ML-Datacode-lib](https://github.com/igornet0/ML-Datacode-lib) |

Imports and package layout: [0-syntax/11-modules-and-imports](../../0-syntax/11-modules-and-imports.md).

Module loading implementation: [200-developers/module_import_system.md](../../200-developers/module_import_system.md).

# Window, image, plot, database, plugins

These types appear when you connect **graphics**, **databases**, or **native modules**. Each has its own operations; some are available through **`["name"]`**, others only through module API calls.

Type check by string: **`isinstance(x, "window")`**, **`"image"`**, … — full list in [typeof-and-isinstance.md](typeof-and-isinstance.md).

---

## Window (`window`)

UI window handle. Indexing **`window["something"]`** is generally **not** supported — see the **plot** module / your project's GUI documentation.

---

## Image (`image`)

Raster data. Direct access via **`img["field"]`** is limited; the value is passed between graphics module functions.

---

## Figure (`figure`)

Plot object. Supported property:

- **`figure["axes"]`** — array of **rows** of axes; each row is an array of **axis** values.

---

## Axis (`axis`)

Plot axis. Methods (string keys) when the **plot** module is loaded:

- **`imshow`**, **`set_title`**, **`axis`**

Without the plot module you get an error like "Plot module not found".

---

## Database engine (`database_engine`)

A string key returns a bound call:

- **`connect`**, **`execute`**, **`query`**, **`run`**

The exact workflow (creating the engine, connection string) is in the project's **`database`** / **`database_engine`** module documentation.

---

## Database cluster (`database_cluster`)

For keys **`add`**, **`get`**, **`names`**, both the cluster and the corresponding function are pushed onto the stack — call as described in the database module API.

---

## Plugin object (`plugin_opaque`)

A "black box" from a native module (for example, a tensor). **`typeof`** may return the type name from the plugin.

- If **`native_plugin_call`** is configured, **`opaque[key]`** may dispatch to plugin code.
- For some operations **`str(...)`**, **`sum`**, **`average`** may call the plugin (see implementation in `basic.rs` / `array.rs`).

Check: **`isinstance(x, "plugin_opaque")`** or the type name returned by the plugin.

---

## Quick `isinstance` table

| String for second argument | Value type |
|----------------------------|------------|
| `window` | window |
| `image` | image |
| `figure` | figure |
| `axis` | axis |
| `database_engine` | database engine |
| `database_cluster` | cluster |
| `plugin_opaque` | opaque plugin object |

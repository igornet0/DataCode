# Functions

A function value can be **stored in a variable**, **passed** to another function, or used in **`map` / `filter` / `reduce`** (where supported).

`typeof` for any function → **`"function"`**.

Check: `isinstance(f, "function")` — [typeof-and-isinstance.md](typeof-and-isinstance.md).

---

## Kinds of functions (inside the VM)

To the user they usually look the same — as a **call** `f(...)`:

- function declared in the current file or chunk;
- function **imported** from a module;
- **built-in** language function (`print`, `len`, …);
- function from a **native library** (loaded `.dylib` / `.so`).

---

## Indexing `f[...]`

Usually you **cannot** index a function like an array.

**Exception:** global **`str`** is also a callable value; **`str[N]`** with integer **`N ≥ 0`** creates a **string-length descriptor** object (see [string.md](string.md)).

---

## Where functions are used

```dc
fn double(x) {
    return x * 2
}
f = double
print(f(21))

a = [1, 2, 3]
print(map(a, double))

```

Built-in global function names are defined by **`BUILTIN_GLOBAL_NAMES`** in `src/vm/globals.rs`. Plugin natives may be added at runtime.

---

## Limitation

In **`map` / `filter` / `reduce`**, a callback **cannot** be a function from **another module** in some modes (error like "module functions as callback are not supported") — see `higher_order.rs`. Local functions and built-in natives work there.

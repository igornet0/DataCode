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

## Call arguments: `*args` and `**kwargs`

Introduction with examples: [lesson 06 — functions](../../0-syntax/06-functions.md#variable-number-of-arguments-args-and-kwargs). Full example: [`variadic_arguments.dc`](../../../../examples/en/04-functions/variadic_arguments.dc).

### Declaration

```dc
fn f(a, b = 2, *args, **kwargs) { ... }
```

| Rule | Error when violated |
|------|---------------------|
| Order: regular → with a default → `*args` → `**kwargs` | `Regular parameters cannot follow *args or **kwargs`, `Non-default argument follows default argument` |
| At most one `*args` and one `**kwargs` | `Only one *args parameter allowed`, `Only one **kwargs parameter allowed` |
| `*args` before `**kwargs`, `**kwargs` last | `*args must appear before **kwargs`, `**kwargs must be the last parameter` |
| `*args` / `**kwargs` have no default | `*args and **kwargs cannot have default values` |

Inside the function `args` is an **array** (`typeof` → `"array"`) and `kwargs` is an **object** (`typeof` → `"object"`). With no extra arguments they are `[]` and `{}`.

### How arguments bind to parameters

1. Named arguments (`name = value`) go to the parameter with that name.
2. Positional arguments fill the leading regular parameters in order. If such a parameter already got a value by name — `got multiple values for argument '<name>'`.
3. Positional arguments left over go to `*args`. Without `*args` — a "too many positional arguments" error.
4. Parameters still unset take their default. Without one — `missing required argument '<name>'`.
5. Named arguments that match no parameter go to `**kwargs`. Without `**kwargs` — `got an unexpected keyword argument '<name>'`.

```dc
fn f(a, b = 2, *args, **kwargs) {
    return [a, b, args, kwargs]
}

print(f(1))                  # [1, 2, [], {}]
print(f(1, 5, 6, 7))         # [1, 5, [6, 7], {}]
print(f(1, x = 9))           # [1, 2, [], {x: 9}]
print(f(b = 8, a = 0))       # [0, 8, [], {}]
```

### Unpacking at the call site

| Syntax | Value must be | Result |
|--------|---------------|--------|
| `f(*arr)` | an array | elements become positional arguments |
| `f(**obj)` | an object with string keys | pairs become named arguments |

- Several unpackings are allowed: `f(*a, *b)` passes the elements of `a`, then those of `b`.
- Order in a call: positional → `*array` → named and `**object` (these two can be mixed). A positional argument after `*` is `Positional argument follows * unpacking`; `*` after a named argument is `* unpacking must appear before named arguments`.
- A name passed twice (`f(a = 1, **{"a": 2})`) is `got multiple values for argument 'a'`.
- A non-array after `*` is `* unpacking requires an array`; a non-object after `**` is `** unpacking requires an object with string keys`.

### What the function receives

- `args` and `kwargs` are a **new** array and a **new** object. Changing them inside the function (`push(args, x)`, `kwargs["k"] = v`) does not affect the array or object passed with `*` / `**`.
- The key order of `kwargs` is **not guaranteed**. Iterate with `for key in kwargs.keys { ... }`, test with `"key" in kwargs`, read with a default via `kwargs.get("key", default)`.

### Where it works

- functions of the current file, nested functions, lambdas (`fn(*xs) => len(xs)`);
- functions held in a variable (`g = f; g(1, b = 2)`);
- functions from modules: `from m import f; f(1, b = 2)` and `m.f(1, *xs, b = 2)`.

For functions of the current file the compiler knows the signature, so binding errors are reported **at compile time**. For functions in variables and from modules arguments are bound **at call time**: errors happen at run time and can be caught with `try` / `catch`.

### Limitations

- **Class methods and constructors** do not accept `*args` / `**kwargs` in their declaration.
- A type annotation on `*args` / `**kwargs` (`*nums: int`) is accepted but **not checked** for the elements.
- Built-in functions and native plugin functions accept named arguments only where their documentation says so.

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

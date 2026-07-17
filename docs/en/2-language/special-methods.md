# Class Special Methods (@-methods)

Special methods are class methods named `@name`. They **cannot be called directly** from user code (`obj.@string()`, `@add(...)`) — the compiler and VM invoke them for operators, builtins, and iteration protocols.

**Example:** [`examples/en/02-syntax/special-methods.dc`](../../examples/en/02-syntax/special-methods.dc)

**See also:** [Classes (syntax)](../0-syntax/09-classes.md)

---

## Table of Contents

1. [Declaration](#declaration)
2. [Method Reference](#method-reference)
3. [Vector Example](#vector-example)
4. [Restrictions](#restrictions)
5. [@init and @drop](#init-and-drop)

---

## Declaration

Inside a `cls` body, declare a method with an `@` prefix after `fn`:

```datacode
cls Vector {
    x: float
    y: float
    new Vector(x: float, y: float) {
        this.x = x
        this.y = y
    }
    fn @add(other: Vector) -> Vector {
        return Vector(this.x + other.x, this.y + other.y)
    }
    fn @string() -> str {
        return "Vector(" + str(this.x) + ", " + str(this.y) + ")"
    }
    fn @len() -> int {
        return 2
    }
}
```

Operators and builtins then dispatch to these methods:

```datacode
v1 = Vector(1, 2)
v2 = Vector(3, 4)
v3 = v1 + v2          # Vector.@add
print(v3)             # Vector.@string
print(len(v3))        # Vector.@len  → 2
```

---

## Method Reference

| Method | Invoked by | Signature (excluding `this`) | Return |
|--------|------------|--------------------------------|--------|
| `@string` | `str(x)`, `print(x)` on instance | `()` | `str` |
| `@len` | `len(x)` | `()` | `int` |
| `@add` | `a + b` | `(other)` | any |
| `@sub` | `a - b` | `(other)` | any |
| `@mul` | `a * b` | `(other)` | any |
| `@div` | `a / b` | `(other)` | any |
| `@mod` | `a % b` | `(other)` | any |
| `@pow` | `a ^ b` | `(other)` | any |
| `@eq` | `a == b` | `(other)` | `bool` / `logic` |
| `@neq` | `a != b` | `(other)` | `logic` |
| `@lt` | `a < b` | `(other)` | `bool` / `logic` |
| `@lte` | `a <= b` | `(other)` | `bool` / `logic` |
| `@gt` | `a > b` | `(other)` | `bool` / `logic` |
| `@gte` | `a >= b` | `(other)` | `bool` / `logic` |
| `@get` | `obj[key]` (when no field with that name) | `(key)` | any |
| `@set` | `obj[key] = value` | `(key, value)` | `null` / void |
| `@contains` | `value in obj` | `(member)` | `bool` |
| `@call` | `obj(...)` | variadic | any |
| `@iter` | `for x in obj` | `()` | iterator / `self` |
| `@next` | `for-in` step | `()` | item; `null` = stop |
| `@clone` | `obj.clone()` | `()` | copy |
| `@hash` | `set` / `dict` key | `()` | `int` |
| `@init` | after constructor body | same as constructor | `null` / void |
| `@drop` | — | — | **reserved, not implemented** |

Invalid signatures are rejected at compile time, e.g. `` `@len` must take 0 arguments ``.

---

## Vector Example

See [`special-methods.dc`](../../examples/en/02-syntax/special-methods.dc) for a runnable script.

```datacode
v1 = Vector(1, 2)
v2 = Vector(3, 4)
print(v1 + v2)   # Vector(4.0, 6.0)
print(len(v1))   # 2
```

---

## Restrictions

- **No direct calls:** `v.@string()`, `@len(v)` → compile error (*Special methods cannot be called directly*).
- **Class body only:** top-level `fn @add` is not allowed.
- **Without a special method**, normal VM behavior applies (e.g. `a + b` without `@add` → runtime error such as *Operands must be numbers or strings*).
- **`@hash` and `@eq`:** for correct `set` behavior, implement both.
- **Iteration:** `for x in obj` requires **both** `@iter` and `@next`; `@next` returns `null` when done.

---

## @init and @drop

**`@init`** runs automatically at the end of every constructor (after field assignments), with the same parameters as the constructor (excluding `this`):

```datacode
cls Widget {
    ready: bool
    new Widget() { this.ready = false }
    fn @init() { this.ready = true }
}
w = Widget()
# w.ready == true
```

**`@drop`** is reserved for a future destructor; declaring `fn @drop()` currently errors with: *`@drop` is reserved but not implemented yet*.

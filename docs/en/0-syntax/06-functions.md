# Functions

Named blocks of code with parameters and a return value.

## Declaration and call

```datacode
fn greet(name) {
    return "Hello, " + name + "!"
}

print(greet("DataCode"))

```

## Default parameters

```datacode
fn power(base, exp = 2) {
    result = 1
    for _ in range(exp) {
        result = result * base
    }
    return result
}

```

A default value can be a **literal** or an **expression built from module compile-time constants** (names assigned at the top level before the function declaration):

```datacode
BASE = 10

fn mul(x, n = BASE) {
    return x * n
}

print(mul(5))   # 50
BASE = 99       # does not change the default of already compiled mul
print(mul(5))   # still 50
```

Defaults are fixed when the function definition is compiled (as in Python). You cannot refer to parameters in the same signature, to forward references (`fn f(x = RED)` before `RED = 0`), or to runtime values (`RED = read_config()`).

## Variable number of arguments: `*args` and `**kwargs`

A starred parameter collects the "extra" arguments of a call:

- **`*name`** — every positional argument left over after the regular parameters, as an **array**;
- **`**name`** — every named argument that matches no parameter, as an **object** (dict).

```datacode
fn total(*nums) {
    s = 0
    for n in nums {
        s = s + n
    }
    return s
}

print(total())         # 0
print(total(1, 2, 3))  # 6

fn connect(host, **opts) {
    port = opts.get("port", 5432)
    return host + ":" + str(port)
}

print(connect("db"))               # db:5432
print(connect("db", port = 6000))  # db:6000
```

Parameter order: regular → with a default → `*args` → `**kwargs`. Each of `*args` and `**kwargs` may appear at most once and has no default (with no extra arguments they are an empty array `[]` and an empty object `{}`):

```datacode
fn log(level, prefix = ">", *parts, **meta) {
    return prefix + " " + level + ": " + join(parts, " ")
}

print(log("INFO"))                          # > INFO:
print(log("INFO", "#", "server", "ready"))  # # INFO: server ready
```

### Unpacking at the call site

In a call, `*array` spreads its elements as positional arguments and `**object` spreads its key-value pairs as named arguments:

```datacode
fn point(x, y, z) {
    return [x, y, z]
}

coords = [1, 2]
print(point(0, *coords))                # [0, 1, 2]
print(point(*[1], *[2, 3]))             # [1, 2, 3]
print(point(1, **{"y": 20, "z": 30}))   # [1, 20, 30]

# Forward all arguments
fn wrapper(*args, **kwargs) {
    return point(*args, **kwargs)
}
```

Arguments in a call go in this order: positional → `*array` → named and `**object`.

Full binding rules, errors and limitations: [functions — call arguments](../2-language/data-types/functions.md#call-arguments-args-and-kwargs).

## return

```datacode
fn abs(x) {
    if x < 0 {
        return -x
    }
    return x
}

```

```datacode
fn abs(x) {
    return x if x > 0 else -x
}

```

Without `return`, a function returns `null`.

## Nested functions

```datacode
fn outer() {
    fn inner(x) {
        return x * 2
    }
    return inner(21)
}

```

Variable capture rules: [scoping and closures](../2-language/scoping-and-closures.md).

## Typed functions

Parameter and return type annotations:

```datacode
fn add(a: int, b: int) -> int {
    return a + b
}

```

Full guide: [typed-functions](./07-typed-functions.md).

## Stream functions

Generators via `stream fn`: [stream-functions](./08-stream-functions.md).

## Examples

| File | Topic |
|------|-------|
| [`simple_functions.dc`](../../../examples/en/04-functions/simple_functions.dc) | basic `fn` |
| [`recursion.dc`](../../../examples/en/04-functions/recursion.dc) | recursion |
| [`nested_functions.dc`](../../../examples/en/04-functions/nested_functions.dc) | nesting |
| [`typed_functions.dc`](../../../examples/en/04-functions/typed_functions.dc) | type annotations |
| [`stream_functions.dc`](../../../examples/en/04-functions/stream_functions.dc) | `stream fn` |
| [`variadic_arguments.dc`](../../../examples/en/04-functions/variadic_arguments.dc) | `*args`, `**kwargs`, unpacking |

More: [1 — Examples / 04-functions](../1-examples/04-functions.md)

## Next

- [function as a value type](../2-language/data-types/README.md)
- [built-in functions](../2-language/functions/README.md)

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

More: [1 — Examples / 04-functions](../1-examples/04-functions.md)

## Next

- [function as a value type](../2-language/data-types/README.md)
- [built-in functions](../2-language/functions/README.md)

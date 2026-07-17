# Error Handling

Exceptions: `try`, `catch`, `throw`, `finally`.

## try / catch

```datacode
try {
    x = 1 / 0
} catch (e) {
    print("Error:", e)
}

```

The variable in `catch (e)` is optional:

```datacode
try {
    risky()
} catch {
    print("something went wrong")
}

```

## throw

```datacode
fn validate(x) {
    if x < 0 {
        throw "negative value"
    }
    return x
}

```

## finally

```datacode
try {
    open_and_process()
} catch (e) {
    log(e)
} finally {
    cleanup()
}

```

The `finally` block always runs — on success and on error.

## Nested try

```datacode
try {
    try {
        work()
    } catch (inner) {
        throw "wrapped error: " + str(inner)
    }
} catch (outer) {
    print(outer)
}

```

## Examples

| File | Topic |
|------|-------|
| [`error_handling.dc`](../../../examples/en/09-advanced/error_handling.dc) | full overview |

More: [1 — Examples / 09-advanced](../1-examples/09-advanced.md)

## Next

- [scoping and closures](../2-language/scoping-and-closures.md)
- [execution model (internals)](../200-developers/execution_model.md)

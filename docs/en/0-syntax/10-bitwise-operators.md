# Bitwise Operators

Operators for working with integers at the bit level. Only operands of type **`int`** are supported (integer literals without a fractional part and `int` variables). For `float`, `string`, `bool`, arrays, and objects — a runtime error:

`Bitwise operator is supported only for int values`

## Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `a & b` | bitwise AND | `5 & 3` → `1` |
| `a \| b` | bitwise OR | `5 \| 3` → `7` |
| `a ^ b` | exclusive OR | `5 ^ 3` → `6` |
| `~a` | bitwise NOT | `~5` → `-6` |
| `a << b` | shift left | `1 << 5` → `32` |
| `a >> b` | arithmetic shift right | `32 >> 2` → `8` |

The literal `5.0` has type `float` and is **not** allowed in bitwise expressions, even if the fractional part is zero.

## Precedence

From high to low (fragment):

1. `~` (unary)
2. `<<`, `>>`
3. `&`
4. `^`
5. `|`
6. `+`, `-` (higher than shifts by associativity with arithmetic: `1 << 2 + 1` = `1 << (2 + 1)` = `8`)

Logical `and`, `or`, `!` are separate operators for conditions; for bit masks use `&`, `|`, `~`.

## Examples

```datacode
# Check a bit
if mask & (1 << i) {
    print("bit set")
}

# Set / clear / toggle
mask = mask | (1 << i)
mask = mask & ~(1 << i)
mask = mask ^ (1 << i)
```

**Examples in the repository:** [`examples/en/02-syntax/bitwise.dc`](../../../examples/en/02-syntax/bitwise.dc), TSP/bitmask: [`examples/en/09-advanced/dp/bitmask_dp.dc`](../../../examples/en/09-advanced/dp/bitmask_dp.dc).

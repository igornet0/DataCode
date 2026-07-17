# User-Defined Functions with Type Annotations

This document describes how to create user-defined functions with type annotations in DataCode, including parameter types, return types, and union types.

**📚 Usage examples:**
- Typed functions: [`examples/en/04-functions/typed_functions.dc`](../../../examples/en/04-functions/typed_functions.dc)
- Stream functions (generators): [`examples/en/04-functions/stream_functions.dc`](../../../examples/en/04-functions/stream_functions.dc) · [Guide](./08-stream-functions.md)
- Simple functions: [`examples/en/04-functions/simple_functions.dc`](../../../examples/en/04-functions/simple_functions.dc)
- Recursion: [`examples/en/04-functions/recursion.dc`](../../../examples/en/04-functions/recursion.dc)

---

## Contents

1. [Basic type annotations](#basic-type-annotations)
2. [Return type annotations](#return-type-annotations)
3. [Union types](#union-types)
4. [Type checking](#type-checking)
5. [Partial typing](#partial-typing)
6. [Default values with types](#default-values-with-types)
7. [Supported types](#supported-types)
8. [Error handling](#error-handling)

---

## Basic type annotations

Functions can have type annotations for their parameters. Syntax:

```datacode
fn function_name(parameter: type) {
    // function body
}

```

### Examples

```datacode
# Function with integer parameters
fn int_add(a: int, b: int) -> int {
    return a + b
}

# Function with float parameters
fn float_multiply(a: float, b: float) -> float {
    return a * b
}

# Function with a string parameter
fn greet(name: str) -> str {
    return "Hello, " + name + "!"
}

# Usage
print(int_add(5, 3))           # 8
print(float_multiply(2.5, 4.0)) # 10.0
print(greet("DataCode"))        # "Hello, DataCode!"

```

---

## Return type annotations

Functions can specify their return type using the `->` syntax:

```datacode
fn function_name(parameters) -> return_type {
    return value
}

```

### Examples

```datacode
fn get_string() -> str {
    return "This is a string"
}

fn get_number() -> int {
    return 42
}

fn get_boolean() -> bool {
    return true
}

print(get_string())   # "This is a string"
print(get_number())   # 42
print(get_boolean())  # true

```

---

## Union types

Union types let a parameter or return value accept multiple types. Use the `|` operator to separate types:

```datacode
fn function_name(parameter: type1 | type2 | type3) -> return_type {
    // function body
}

```

### Examples

```datacode
# Function accepting a string or integer
fn process_value(value: str | int) -> str {
    return "Value: " + str(value)
}

print(process_value("hello"))  # "Value: hello"
print(process_value(42))       # "Value: 42"

# Function with multiple union types
fn flexible_add(a: int | float, b: int | float) -> float {
    return a + b
}

print(flexible_add(5, 3))      # 8.0
print(flexible_add(2.5, 3.7))  # 6.2
print(flexible_add(5, 3.5))    # 8.5

# Union type with null
fn process_or_default(value: null | str | int = null) -> str {
    if value == null {
        return "default value"
    } else {
        return "received: " + str(value)
    }
}

print(process_or_default())        # "default value"
print(process_or_default("test"))  # "received: test"
print(process_or_default(123))     # "received: 123"

```

### Union return types

Functions can also return union types:

```datacode
fn get_value(flag: bool) -> str | int {
    if flag {
        return "string result"
    } else {
        return 100
    }
}

print(get_value(true))   # "string result"
print(get_value(false))  # 100

```

---

## Type checking

DataCode performs **runtime type checking** when functions are called. If an argument does not match the expected type, a `TypeError` is raised.

### Type checking behavior

- Type checking happens at function call time
- On type mismatch, a `TypeError` is raised
- Union types accept any of the listed types
- Type checking is optional — functions without annotations work as before

### Examples

```datacode
fn int_add(a: int, b: int) -> int {
    return a + b
}

# This works
print(int_add(5, 3))  # 8

# This raises TypeError
try {
    int_add(5, 3.5)  # TypeError: Argument 'b' expected type 'int', got 'float'
} catch TypeError {
    print("Type error caught!")
}

```

### Handling type errors

You can catch `TypeError` with try-catch blocks:

```datacode
fn greet(name: str) -> str {
    return "Hello, " + name + "!"
}

try {
    greet(123)  # TypeError
} catch TypeError {
    print("Invalid type provided")
}

# Or catch via RuntimeError (TypeError is a subtype)
try {
    greet(123)
} catch RuntimeError {
    print("Runtime error caught")
}

```

---

## Partial typing

You can mix typed and untyped parameters in one function:

```datacode
# First parameter typed, second not
fn mixed_types(a: int, b) -> str {
    return str(a) + " + " + str(b) + " = " + str(a + b)
}

print(mixed_types(5, 3))    # "5 + 3 = 8"
print(mixed_types(5, 3.5))  # "5 + 3.5 = 8.5"

# Second parameter typed, first not
fn first_untyped(a, b: str) -> str {
    return str(a) + " " + b
}

print(first_untyped(42, "hello"))  # "42 hello"

```

**Note:** Only typed parameters are checked. Untyped parameters accept any type.

---

## Default values with types

You can combine type annotations with default parameter values:

```datacode
fn process(value: null | str | int = null) -> str {
    if value == null {
        return "default value"
    } else {
        return "received: " + str(value)
    }
}

# Using the default
print(process())           # "default value"

# Overriding the default
print(process("test"))     # "received: test"
print(process(123))        # "received: 123"

# Type checking still applies
try {
    process(true)  # TypeError: Argument 'value' expected type 'null | str | int', got 'bool'
} catch TypeError {
    print("Type error caught")
}

```

A default may refer to **module constants** declared earlier in the file:

```datacode
RED = 0
BLACK = 1

fn tag(value: int, color: int = RED) -> int {
    return value * 10 + color
}

print(tag(5))        # 50
RED = 99             # tag default already fixed as 0
print(tag(5))        # 50
```

The value is computed once when the function definition is compiled; changing the variable later does not change the default.

---

## Supported types

The following types can be used in annotations:

### Basic types
- `int` — integers
- `float` — floating-point numbers (any number)
- `str` or `string` — strings
- `bool` or `boolean` — booleans (true/false)
- `null` — null value

### Collection types
- `array` — arrays
- `tuple` — tuples
- `object` — objects/dictionaries

### Special types
- `table` — tables
- `path` — file paths
- `tensor` — ML tensors
- `graph` — computation graphs
- `dataset` — ML datasets
- `neural_network` — neural networks
- `sequential` — sequential models
- `layer` — neural network layers

### Type compatibility

- `int` values can be passed to `float` parameters (`int` is a subset of `float`)
- `float` values cannot be passed to `int` parameters (unless they are whole numbers)
- Union types accept any of the listed types

### Examples

```datacode
# array type
fn sum_array(arr: array) -> float {
    sum = 0.0
    for item in arr {
        sum = sum + item
    }
    return sum
}

print(sum_array([1, 2, 3, 4, 5]))  # 15.0

# object type
fn get_name(obj: object) -> str {
    return obj["name"]
}

person = {"name": "Alice", "age": 30}
print(get_name(person))  # "Alice"

# tuple type
fn get_first(t: tuple) -> int {
    return t[0]
}

print(get_first((10, 20, 30)))  # 10

```

---

## Error handling

### TypeError

When a type mismatch occurs, a `TypeError` is raised. It is a subtype of `RuntimeError`, so you can catch it either way:

```datacode
fn int_add(a: int, b: int) -> int {
    return a + b
}

# Catch by specific type
try {
    int_add(1, 2.5)
} catch TypeError {
    print("Type error caught")
}

# Catch by parent type
try {
    int_add(1, 2.5)
} catch RuntimeError {
    print("Runtime error caught")
}

# Catch with a variable
try {
    int_add(1, 2.5)
} catch TypeError e {
    print("Error:", e)
}

```

### Error messages

TypeError messages include:
- The parameter name that failed the type check
- Expected type(s)
- Actual received type

Example error message:
```
TypeError: Argument 'b' expected type 'int', got 'float'
```

For union types:
```
TypeError: Argument 'value' expected type 'str | int', got 'bool'
```

---

## Recommendations

### 1. Use type annotations for clarity

Type annotations make code more readable and self-documenting:

```datacode
# Clear intent
fn calculate_area(width: float, height: float) -> float {
    return width * height
}

```

### 2. Use union types for flexibility

Union types provide flexibility while keeping type safety:

```datacode
# Safely accepts multiple types
fn format_output(value: null | str | int) -> str {
    if value == null {
        return "null"
    } else {
        return str(value)
    }
}

```

### 3. Combine with default values

Default values work well with union types that include `null`:

```datacode
fn process(value: null | str | int = null) -> str {
    if value == null {
        return "default"
    } else {
        return str(value)
    }
}

```

### 4. Handle type errors properly

Use try-catch blocks to handle type errors:

```datacode
fn safe_process(value: str | int) -> str {
    return str(value)
}

try {
    print(safe_process("hello"))
    print(safe_process(42))
    print(safe_process(true))  # TypeError
} catch TypeError {
    print("Invalid type provided")
}

```

---

## Typed recursive functions

Type annotations work with recursive functions:

```datacode
# Typed factorial
fn factorial(n: int) -> int {
    if n <= 1 {
        return 1
    } else {
        return n * factorial(n - 1)
    }
}

print(factorial(5))  # 120

# Typed Fibonacci
fn fibonacci(n: int) -> int {
    if n <= 1 {
        return n
    } else {
        return fibonacci(n - 1) + fibonacci(n - 2)
    }
}

print(fibonacci(7))  # 13

```

---

## Comparison with untyped functions

Functions without type annotations work exactly as before — they accept any types:

```datacode
# Untyped function (works as before)
fn add(a, b) {
    return a + b
}

print(add(1, 2))      # 3
print(add(1, 2.5))    # 3.5
print(add("a", "b"))  # "ab"

```

Type annotations are **optional** — use them when you want type safety, or omit them for flexibility.

---

## Summary

- **Type annotations** provide runtime type checking for function parameters and return values
- **Union types** (`str | int`) allow multiple types for flexibility
- **Type checking** happens at runtime and raises `TypeError` on mismatch
- **Partial typing** lets you mix typed and untyped parameters
- **Default values** work with typed parameters
- **All language types** are supported in annotations
- **Type annotations are optional** — untyped functions work as before

---

## Related documentation

- [Stream functions (generators)](./08-stream-functions.md) — `stream fn`, `.next()`, `.send()`, `ereturn`, `.final()`
- [Built-in Functions](../2-language/functions/README.md) — type conversion and checking
- [Data Types](../2-language/data-types/README.md) — complete type system documentation
- [Function examples](../../../examples/en/04-functions/) — practical examples

---

**📚 Examples:** [`examples/en/04-functions/typed_functions.dc`](../../../examples/en/04-functions/typed_functions.dc)

# User-Defined Functions with Type Annotations

This document describes how to create user-defined functions with type annotations in DataCode, including parameter types, return types, and union types.

**📚 Usage examples:**
- Typed functions: [`examples/en/05-functions/typed_functions.dc`](../../examples/en/05-functions/typed_functions.dc)
- Simple functions: [`examples/en/05-functions/simple_functions.dc`](../../examples/en/05-functions/simple_functions.dc)
- Recursion: [`examples/en/05-functions/recursion.dc`](../../examples/en/05-functions/recursion.dc)

---

## Table of Contents

1. [Basic Type Annotations](#basic-type-annotations)
2. [Return Type Annotations](#return-type-annotations)
3. [Union Types](#union-types)
4. [Type Checking](#type-checking)
5. [Partial Typing](#partial-typing)
6. [Default Values with Types](#default-values-with-types)
7. [Supported Types](#supported-types)
8. [Error Handling](#error-handling)

---

## Basic Type Annotations

Functions can have type annotations for their parameters. The syntax is:

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

# Function with string parameter
fn greet(name: str) -> str {
    return "Hello, " + name + "!"
}

# Usage
print(int_add(5, 3))           # 8
print(float_multiply(2.5, 4.0)) # 10.0
print(greet("DataCode"))        # "Hello, DataCode!"
```

---

## Return Type Annotations

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

## Union Types

Union types allow a parameter or return value to accept multiple types. Use the `|` operator to separate types:

```datacode
fn function_name(param: type1 | type2 | type3) -> return_type {
    // function body
}
```

### Examples

```datacode
# Function accepting string or integer
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

### Union Return Types

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

## Type Checking

DataCode performs **runtime type checking** when calling functions. If an argument doesn't match the expected type, a `TypeError` is raised.

### Type Checking Behavior

- Type checking happens at function call time
- If a type mismatch occurs, a `TypeError` is raised
- Union types allow any of the specified types
- Type checking is optional - functions without type annotations work as before

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

### Handling Type Errors

You can catch `TypeError` using try-catch blocks:

```datacode
fn greet(name: str) -> str {
    return "Hello, " + name + "!"
}

try {
    greet(123)  # TypeError
} catch TypeError {
    print("Invalid type provided")
}

# Or catch by RuntimeError (TypeError is a subtype)
try {
    greet(123)
} catch RuntimeError {
    print("Runtime error caught")
}
```

---

## Partial Typing

You can mix typed and untyped parameters in the same function:

```datacode
# First parameter typed, second untyped
fn mixed_types(a: int, b) -> str {
    return str(a) + " + " + str(b) + " = " + str(a + b)
}

print(mixed_types(5, 3))    # "5 + 3 = 8"
print(mixed_types(5, 3.5))  # "5 + 3.5 = 8.5"

# Second parameter typed, first untyped
fn first_untyped(a, b: str) -> str {
    return str(a) + " " + b
}

print(first_untyped(42, "hello"))  # "42 hello"
```

**Note:** Only typed parameters are checked. Untyped parameters accept any type.

---

## Default Values with Types

You can combine type annotations with default parameter values:

```datacode
fn process(value: null | str | int = null) -> str {
    if value == null {
        return "default value"
    } else {
        return "received: " + str(value)
    }
}

# Using default value
print(process())           # "default value"

# Overriding default
print(process("test"))     # "received: test"
print(process(123))        # "received: 123"

# Type checking still applies
try {
    process(true)  # TypeError: Argument 'value' expected type 'null | str | int', got 'bool'
} catch TypeError {
    print("Type error caught")
}
```

---

## Supported Types

The following types can be used in type annotations:

### Basic Types
- `int` - Integer numbers (whole numbers)
- `float` - Floating-point numbers (any number)
- `str` or `string` - Strings
- `bool` or `boolean` - Boolean values (true/false)
- `null` - Null value

### Collection Types
- `array` - Arrays
- `tuple` - Tuples
- `object` - Objects/dictionaries

### Special Types
- `table` - Tables
- `path` - File paths
- `tensor` - ML tensors
- `graph` - Computation graphs
- `dataset` - ML datasets
- `neural_network` - Neural networks
- `sequential` - Sequential models
- `layer` - Neural network layers

### Type Compatibility

- `int` values can be passed to `float` parameters (int is a subset of float)
- `float` values cannot be passed to `int` parameters (unless they are whole numbers)
- Union types accept any of the specified types

### Examples

```datacode
# Array type
fn sum_array(arr: array) -> float {
    let sum = 0.0
    for item in arr {
        sum = sum + item
    }
    return sum
}

print(sum_array([1, 2, 3, 4, 5]))  # 15.0

# Object type
fn get_name(obj: object) -> str {
    return obj["name"]
}

let person = {"name": "Alice", "age": 30}
print(get_name(person))  # "Alice"

# Tuple type
fn get_first(t: tuple) -> int {
    return t[0]
}

print(get_first((10, 20, 30)))  # 10
```

---

## Error Handling

### TypeError

When a type mismatch occurs, a `TypeError` is raised. This is a subtype of `RuntimeError`, so you can catch it either way:

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

# Catch with variable
try {
    int_add(1, 2.5)
} catch TypeError e {
    print("Error:", e)
}
```

### Error Messages

TypeError messages include:
- Parameter name that failed type check
- Expected type(s)
- Actual type received

Example error message:
```
TypeError: Argument 'b' expected type 'int', got 'float'
```

For union types:
```
TypeError: Argument 'value' expected type 'str | int', got 'bool'
```

---

## Best Practices

### 1. Use Type Annotations for Clarity

Type annotations make code more readable and self-documenting:

```datacode
# Clear intent
fn calculate_area(width: float, height: float) -> float {
    return width * height
}
```

### 2. Use Union Types for Flexibility

Union types provide flexibility while maintaining type safety:

```datacode
# Accepts multiple types safely
fn format_output(value: null | str | int) -> str {
    if value == null {
        return "null"
    } else {
        return str(value)
    }
}
```

### 3. Combine with Default Values

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

### 4. Handle Type Errors Gracefully

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

## Typed Recursive Functions

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

## Comparison with Untyped Functions

Functions without type annotations work exactly as before - they accept any types:

```datacode
# Untyped function (works as before)
fn add(a, b) {
    return a + b
}

print(add(1, 2))      # 3
print(add(1, 2.5))    # 3.5
print(add("a", "b"))  # "ab"
```

Type annotations are **optional** - you can use them when you want type safety, or omit them for flexibility.

---

## Summary

- **Type annotations** provide runtime type checking for function parameters and return values
- **Union types** (`str | int`) allow multiple types for flexibility
- **Type checking** happens at runtime and raises `TypeError` on mismatch
- **Partial typing** allows mixing typed and untyped parameters
- **Default values** work with typed parameters
- **All types** from the language are supported in annotations
- **Type annotations are optional** - untyped functions work as before

---

## Related Documentation

- [Built-in Functions](./builtin_functions.md) - Type conversion and checking functions
- [Data Types](./data_types.md) - Complete type system documentation
- [Function Examples](../../examples/en/05-functions/) - Practical examples

---

**📚 Examples:** [`examples/en/05-functions/typed_functions.dc`](../../examples/en/05-functions/typed_functions.dc)

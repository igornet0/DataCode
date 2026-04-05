# 📦 Data Types in DataCode

This section demonstrates working with different data types in DataCode, including type conversion, checking, and object manipulation.

## 📋 Contents

### 1. `type_conversion_functions.dc` - Type Conversion Functions
**Description**: Demonstrates all type conversion functions available in DataCode.

**What you'll learn**:
- Converting between types with `int()`, `float()`, `bool()`, `str()`
- Creating arrays with `array()`
- Working with dates using `date()`
- Formatting money with `money()`
- Determining types with `typeof()`
- Practical type conversion examples

**Run**:
```bash
datacode examples/en/03-data-types/type_conversion_functions.dc
```

### 2. `type_conversion_guide.dc` - Type Conversion Guide
**Description**: Comprehensive guide to working with types and conversions.

**What you'll learn**:
- Basic data types
- Type checking with `isinstance()`
- Type conversion strategies
- Working with different type combinations

**Run**:
```bash
datacode examples/en/03-data-types/type_conversion_guide.dc
```

### 3. `type_date.dc` - Working with Dates
**Description**: Demonstrates date handling and formatting.

**What you'll learn**:
- Creating dates
- Date formats
- Working with date strings

**Run**:
```bash
datacode examples/en/03-data-types/type_date.dc
```

### 4. `objects.dc` - Working with Objects
**Description**: Demonstrates creating and working with objects (dictionaries) in DataCode.

**What you'll learn**:
- Creating objects with key-value pairs
- Accessing properties with dot notation (`obj.property`)
- Accessing properties with bracket notation (`obj['key']`)
- Nested objects
- Objects with different value types
- Objects in arrays
- Using objects as dictionaries
- Practical configuration examples

**Run**:
```bash
datacode examples/en/03-data-types/objects.dc
```

### 5. `arrays.dc` - Arrays (globals and methods)
**Description**: Walkthrough of array creation, `push`/`pop`, `unique`, `reverse`, `sort`, numeric aggregates, `any`/`all`, and `arr.chunk(n)` (chunk is only available as a method).

**What you'll learn**:
- `array()` and `[...]` literals
- `array_with_capacity()` with repeated `push`
- Global calls vs method style (`push(arr, x)` and `arr.push(x)`)
- `unique`, `reverse`, `sort`
- `sum`, `average`, `count`, and `len`
- `min` and `max` with several numbers or a single numeric array (`min([])` is `null`)
- Truthiness checks with `any` and `all` (including empty arrays)
- `map`, `filter`, and `reduce` (with required initial value for `reduce`)
- Splitting with `arr.chunk(n)`

**Run**:
```bash
datacode examples/en/03-data-types/arrays.dc
```

## 🎯 Concepts Covered

### Type Conversion
- **int()**: Convert to integer
- **float()**: Convert to floating point number
- **bool()**: Convert to boolean
- **str()**: Convert to string
- **array()**: Create array from arguments
- **date()**: Create date from string
- **money()**: Format number as currency

### Type Checking
- **typeof()**: Get type name as string
- **isinstance()**: Check if value is instance of type

### Arrays
- **Creation**: `array(...)`, `[...]`, `array_with_capacity(n)`
- **Mutation**: `push`, `pop`, `reverse`, `sort`; also `arr.method(...)` form
- **Copies / views**: `unique`; split with `arr.chunk(n)` (method only)
- **Aggregates**: `sum`, `average`, `count`, `len`; `min`/`max` with several numbers or one numeric array; logic: `any`, `all`; higher-order: `map`, `filter`, `reduce(..., initial)`

### Objects (Dictionaries)
- **Creation**: `{key: value, key2: value2}`
- **Access**: `obj.key` or `obj['key']`
- **Nested**: Objects can contain other objects
- **Mixed types**: Values can be any DataCode type
- **String keys**: Use quotes for keys with spaces: `{'key name': value}`

## 🔗 Navigation

### Previous Steps
- **[02-syntax](../02-syntax/)** - Language syntax constructs

### Next Steps
- **[04-advanced](../04-advanced/)** - Advanced features including error handling
- **[05-functions](../05-functions/)** - Creating and using functions

---

**Keep learning!** 🚀


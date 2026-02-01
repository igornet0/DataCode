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


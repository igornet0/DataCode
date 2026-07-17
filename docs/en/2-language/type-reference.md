# Data Types in DataCode

This document describes all data types supported by the DataCode programming language.

**📚 Usage examples:**
- Working with types: [`examples/en/03-data-types/`](../../../examples/en/03-data-types/)
- Basic types: [`examples/en/01-basics/`](../../../examples/en/01-basics/)
- Working with tables: [`examples/en/08-data-model-creation/`](../../../examples/en/08-data-model-creation/)

## Contents

1. [Basic Data Types](#basic-data-types)
2. [Composite Data Types](#composite-data-types)
3. [Special Types for Working with Tables](#special-types-for-working-with-tables)
4. [Special Types for Working with Files](#special-types-for-working-with-files)
5. [Type Recognition and Conversion](#type-recognition-and-conversion)
6. [Type Compatibility](#type-compatibility)

---

## Basic Data Types

### Integer

- **Internal representation**: `f64` (floating point, but fractional part is 0)
- **Description**: Whole numbers (42, -10, 0)
- **Examples**:
  ```datacode
  global x = 42
  global y = -100
  global z = 0
  
```

**📚 Examples:** [`examples/en/01-basics/arithmetic.dc`](../../../examples/en/01-basics/arithmetic.dc)

### Float

- **Internal representation**: `f64`
- **Description**: Numbers with a decimal part
- **Examples**:
  ```datacode
  global pi = 3.14
  global price = 99.99
  global temperature = -5.5
  
```

**Note**: Integer and Float are compatible in operations and are considered numeric types.

**📚 Examples:** [`examples/en/01-basics/arithmetic.dc`](../../../examples/en/01-basics/arithmetic.dc)

### String

- **Internal representation**: `String`
- **Description**: Text data
- **Examples**:
  ```datacode
  global name = 'Hello'
  global message = 'DataCode is awesome'
  global empty = ''
  
```

Strings support **interpolation**: `${expr}`, name=value output (`${x=}`), and number formatting (`${n:.2f}`). See [String formatting (interpolation)](../0-syntax/02-strings-and-interpolation.md).

**📚 Examples:** [`examples/en/01-basics/strings.dc`](../../../examples/en/01-basics/strings.dc)

### Bool

- **Internal representation**: `bool`
- **Description**: Logical values `true` or `false`
- **Examples**:
  ```datacode
  global is_active = true
  global is_deleted = false
  
```

**📚 Examples:** [`examples/en/02-syntax/booleans.dc`](../../../examples/en/02-syntax/booleans.dc)

### Null

- **Internal representation**: special value `Value::Null`
- **Description**: Absence of value or undefined value
- **Examples**:
  ```datacode
  global value = null
  
```

**Features**:
- `Null` is compatible with any data type
- Used by default for uninitialized variables

---

## Composite Data Types

### Array

- **Internal representation**: `Vec<Value>`
- **Description**: Ordered collection of values of any type
- **Examples**:
  ```datacode
  global numbers = [1, 2, 3, 4, 5]
  global mixed = [1, 'hello', true, null]
  global nested = [[1, 2], [3, 4]]
  
```

**Features**:
- Arrays may contain elements of different types
- Nesting is supported (arrays of arrays)
- Index access: `arr[0]`

**📚 Examples:** [`examples/en/01-basics/`](../../../examples/en/01-basics/)

### Object

- **Internal representation**: `HashMap<String, Value>`
- **Description**: Unordered collection of key-value pairs
- **Examples**:
  ```datacode
  global person = {
    'name': 'Alice',
    'age': 30,
    'active': true
  }
  
```

**Features**:
- Keys are always strings
- Values may be any type
- Access via dot: `person.name` or `person['name']`

---

## Special Types for Working with Tables

### Table

- **Internal representation**: `Rc<RefCell<Table>>`
- **Description**: Structured data with columns and rows
- **Examples**:
  ```datacode
  global data = read('data.csv')  # read can load csv, excel; returns table
  
```
  ```datacode
  global table1 = table(data, ['name', 'age', 'salary'])
  
```

**Features**:
- Automatic column type detection
- Optimized storage for large data
- Filtering, sorting, selection operations

**Table methods**:
- `table.rows` - access to rows
- `table.columns` - access to column list
- `table.idx[i]` - row by index
- `table['column_name']` - column by name

**📚 Examples:** 
- [`examples/en/08-data-model-creation/`](../../../examples/en/08-data-model-creation/)
- See also: [Working with tables](./tables/creation-and-operations.md)

### TableColumn

- **Internal representation**: `(Rc<RefCell<Table>>, String)`
- **Description**: Reference to a table column with name
- **Examples**:
  ```datacode
  global column = table.name  # Get column 'name'
  
```

**Features**:
- Represents a collection of values for one column
- Preserves column data type information

### TableIndexer

- **Internal representation**: `Rc<RefCell<Table>>`
- **Description**: Special object for row access by index
- **Examples**:
  ```datacode
  global first_row = table.idx[0]
  
```

**Features**:
- Used for row access via `.idx[i]`
- Returns an array of row values

### LazyTable

- **Internal representation**: `LazyTable`
- **Description**: Table with deferred operation execution
- **Examples**:
  ```datacode
  global filtered = table.filter(x > 10)
  
```

**Features**:
- Operations are not executed immediately; they accumulate
- Execution happens only on materialization
- Supported operations: `filter`, `where`, `select`, `head`, `tail`, `sort`

---

## Special Types for Working with Files

### Path

- **Internal representation**: `PathBuf`
- **Description**: Path to a file or directory
- **Examples**:
  ```datacode
  global file_path = path('/path/to/file.csv')
  global files = list_files('/path/to/dir')  # Returns array of Path objects
  
```
  
**Features**:
- `list_files()` returns an array of `Path` objects with full paths
- Path objects can be used directly with `read()`, `path()`, and others
- Recursively walks subdirectories by default
- Optional `regex` parameter to filter by name (glob like `*.csv` or regular expressions)

**Path object methods**:

| Method | Return type | Description |
|--------|-------------|-------------|
| `name` | String | File/directory name (last path component) |
| `parent` | Path or Null | Parent directory |
| `exists` | Bool | Whether path exists |
| `is_file` | Bool | Whether path is a file |
| `is_dir` | Bool | Whether path is a directory |
| `extension` | String | File extension (no dot); empty if none |
| `len` | Number | Length of string representation of path |

**Method examples**:
```datacode
global file = path('/home/user/document.txt')

print(file.name)        # 'document.txt'
print(file.parent)      # Path('/home/user')
print(file.exists)      # true/false
print(file.is_file)     # true
print(file.is_dir)      # false
print(file.extension)   # 'txt'
print(file.len)         # 25

# With list_files()
global files = list_files('/path/to/dir')
for file in files {
    if file.is_file and file.extension == 'csv' {
        print('CSV file:', file.name)
        global data = read(file)  # Use Path object directly
    }
}

# With regex filter (glob)
global csv_files = list_files('/path/to/dir', regex='*.csv')
for file in csv_files {
    print('CSV file:', file.name)
}

# Multiple extensions
global data_files = list_files('/path/to/dir', regex='*.csv|*.xlsx')
for file in data_files {
    print('Data file:', file.name)
}

# Regular expression
global numbered_files = list_files('/path/to/dir', regex='^file\\d+\\.txt$')
for file in numbered_files {
    print('Numbered file:', file.name)
}

```

**📚 Examples:** 
- [`examples/en/01-basics/`](../../../examples/en/01-basics/)
- [`examples/en/08-data-model-creation/01-file-operations.dc`](../../../examples/en/08-data-model-creation/01-file-operations.dc)

### PathPattern

- **Internal representation**: `PathBuf`
- **Description**: Glob pattern for file search
- **Note**: You can also use the `regex` parameter in `list_files()` directly
- **Examples**:
  ```datacode
  global pattern = path('/path/*.csv')
  global files = list_files(pattern)
  
  # Or use regex directly
  global files = list_files('/path/to/dir', regex='*.csv')
  
```

---

## Type Recognition and Conversion

### Date

- **Detection**: Automatically detected from strings when loading data
- **Supported formats**:
  - ISO: `YYYY-MM-DD` (e.g. `2023-12-25`)
  - RFC3339: `YYYY-MM-DDTHH:MM:SSZ` (e.g. `2023-12-25T10:30:00Z`)
  - European: `DD.MM.YYYY` or `DD/MM/YYYY` (e.g. `25.12.2023`, `25/12/2023`)
  - American: `MM/DD/YYYY` (e.g. `12/25/2023`)
  - Two-digit year: `DD.MM.YY` or `MM/DD/YY` (e.g. `25.12.23`)

**Examples**:
```datacode
global date1 = '2023-12-25'      # ISO
global date2 = '25.12.2023'      # European
global date3 = '12/25/2023'      # American

```

**Features**:
- Stored as string with automatic format recognition
- When reading from files, Date type is detected automatically

### Currency

**Examples**:
```datacode
global price1 = money('100', '$0,0')      # 100 -> $100,0
global price2 = money('50.99', '€0.0')     # 50.99 -> €50.9
global price3 = money('100', '0,0 USD')    # 100 -> 100,0 USD
global price4 = money('50.99', 'EUR 0.0') # 50.99 -> EUR 50.9

```

**Features**:
- Stored as string preserving original format

**📚 Examples:** See [money()](./functions/type-conversion.md#moneyamount-format)

### Mixed

- **Description**: Used for table columns containing values of different types
- **Examples**:
  ```datacode
  # Column may contain both numbers and strings
  global mixed_column = [1, 'two', 3, 'four']
  
```

**Features**:
- Automatically detected for columns with heterogeneous data
- Warnings when mixed types are found

---

## Type Compatibility

DataCode supports automatic compatibility between some types:

### Compatible types:

1. **Integer ↔ Float**: Whole and fractional numbers are compatible and may be used together in arithmetic
   ```datacode
   global x = 42      # Integer
   global y = 3.14    # Float
   global result = x + y  # Works correctly
   
```

2. **Null ↔ any type**: `Null` is compatible with any type
   ```datacode
   global value = null
   global result = value + 10  # Null is compatible, but result may be null
   
```

3. **Same types**: Types are compatible with themselves

### Incompatible types:

- String and Number
- Bool and Number (except logical operations)
- String and Bool
- Array and Object

### Automatic type coercion:

DataCode performs some automatic conversions:

- **Numbers → Bool**: `0` → `false`, any non-zero → `true`
- **Strings → Bool**: empty string → `false`, non-empty → `true`
- **Null → Bool**: always `false`
- **Array → Bool**: [] → `false`, non-empty → `true`

**📚 Examples:** [`examples/en/02-syntax/booleans.dc`](../../../examples/en/02-syntax/booleans.dc)

---

## Table Column Typing

When working with tables, DataCode automatically infers column data types:

1. **Automatic type inference**: When adding data, the system analyzes values and determines the main column type
2. **Type statistics**: Each column tracks counts per type
3. **Warnings**: Mixed types generate heterogeneity warnings
4. **Optimization**: Integer and Float are compatible numeric types and do not trigger warnings

**Example**:
```datacode
global table = table([
    [42, 'Alice', '$100'],
    [3.14, 'Bob', '€50.99']
], ['numbers', 'names', 'prices'])

# Result:
# Column 'numbers': type Float (Integer and Float compatible)
# Column 'names': type String
# Column 'prices': type Currency

```

**📚 Examples:** [`examples/en/08-data-model-creation/`](../../../examples/en/08-data-model-creation/)

---

## Conclusion

DataCode supports a rich type system including:

- **8 basic types**: Integer, Float, String, Bool, Date, Currency, Null, Mixed
- **2 composite types**: Array, Object
- **4 special table types**: Table, TableColumn, TableIndexer, LazyTable
- **2 file types**: Path, PathPattern

All types are detected automatically when loading data and support flexible conversions for working with various formats.

---

**See also:**
- [Built-in functions](./functions/README.md) — `typeof()`, `isinstance()`, type conversion
- [Working with tables](./tables/creation-and-operations.md) - more on table types
- [Type usage examples](../../../examples/en/03-data-types/) - practical examples

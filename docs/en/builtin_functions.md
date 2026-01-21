# DataCode Built-in Functions

This document contains a complete description of all built-in functions of the DataCode language, their arguments, and usage examples.

**📚 Usage examples:**
- Basic functions: [`examples/en/01-basics/`](../../examples/en/01-basics/)
- Advanced functions: [`examples/en/05-functions/`](../../examples/en/05-functions/)
- Data work: [`examples/en/09-data-model-creation/`](../../examples/en/09-data-model-creation/)

## Contents

1. [Utilities](#utilities)
2. [Type Conversion Functions](#type-conversion-functions)
3. [Type Operations](#type-operations)
4. [Path Operations](#path-operations)
5. [Mathematical Functions](#mathematical-functions)
6. [String Functions](#string-functions)
7. [Array Functions](#array-functions)
8. [Table Functions](#table-functions)

---

## Utilities

**📚 Examples:** [`examples/en/01-basics/`](../../examples/en/01-basics/)

### `print(...)`

Outputs values to the console. Accepts any number of arguments.

**Arguments:**
- `...` - any number of values of any type

**Returns:** `null`

**Examples:**
```datacode
print("Hello, World!")
print("Number:", 42, "String:", "test")
print()  # Empty line
```

---

### `len(value)`

Returns the length of a string, array, table, or object.

**Arguments:**
- `value` (string | array | table | object) - value to get length of

**Returns:** `number` - length of value, or `null` if type is not supported

**Examples:**
```datacode
len("Hello")        # 5
len([1, 2, 3])      # 3
len([])             # 0
```

---

### `range(end)` / `range(start, end)` / `range(start, end, step)`

Creates an array of numbers in the specified range.

**Arguments:**
- `end` (number) - end value (not included)
- `start` (number, optional) - start value (default 0)
- `step` (number, optional) - step (default 1)

**Returns:** `array` - array of numbers

**Examples:**
```datacode
range(5)              # [0, 1, 2, 3, 4]
range(1, 5)           # [1, 2, 3, 4]
range(1, 10, 2)       # [1, 3, 5, 7, 9]
range(10, 0, -1)      # [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
```

---

## Type Conversion Functions

**📚 Examples:** [`examples/en/03-data-types/`](../../examples/en/03-data-types/)

### `int(value)`

Converts a value to an integer.

**Arguments:**
- `value` (any) - value to convert

**Returns:** `number` - integer

**Examples:**
```datacode
int(42.7)      # 42
int("123")     # 123
int(true)      # 1
int(false)     # 0
int(null)      # 0
```

---

### `float(value)`

Converts a value to a floating point number.

**Arguments:**
- `value` (any) - value to convert

**Returns:** `number` - floating point number

**Examples:**
```datacode
float(42)        # 42.0
float("3.14")    # 3.14
float(true)      # 1.0
float(false)     # 0.0
```

---

### `bool(value)`

Converts a value to a boolean.

**Arguments:**
- `value` (any) - value to convert

**Returns:** `bool` - boolean value

**Examples:**
```datacode
bool(1)          # true
bool(0)          # false
bool(42)         # true
bool("")         # false
bool(null)       # false
bool([1, 2, 3])  # true
bool([])         # false
```

---

### `str(value)`

Converts a value to a string.

**Arguments:**
- `value` (any) - value to convert

**Returns:** `string` - string representation of value

**Examples:**
```datacode
str(42)          # "42"
str(3.14)        # "3.14"
str(true)        # "true"
str(false)       # "false"
str(null)        # "null"
str([1, 2, 3])   # "[1, 2, 3]"
```

---

### `array(...)`

Creates an array from passed arguments.

**Arguments:**
- `...` - any number of values of any type

**Returns:** `array` - array of values

**Examples:**
```datacode
array(1, 2, 3)                    # [1, 2, 3]
array("a", "b", "c")              # ["a", "b", "c"]
array(100.50, 11, "Hello")         # [100.5, 11, "Hello"]
array([1, 3], [1, 5, 6])          # [[1, 3], [1, 5, 6]]
```

---

### `date(value)`

Converts a value to a date (string in ISO format).

**Arguments:**
- `value` (string | number) - value to convert

**Returns:** `string` - string with date in ISO format

**Examples:**
```datacode
date("2024-01-15T10:30:00Z")  # "2024-01-15T10:30:00Z"
date("2023-12-25")            # "2023-12-25"
```

---

### `money(amount, format)`

Formats a number as a monetary amount.

**Arguments:**
- `amount` (number | string) - amount to format
- `format` (string, optional) - format string (e.g., "$0.00", "0,0 $", "0 EUR")

**Returns:** `string` - formatted monetary amount

**Examples:**
```datacode
money(100.50, "$0.00")      # "$100.50"
money(250.75, "0,0 $")      # "250,75 $"
money(50, "0 EUR")          # "50 EUR"
money(100.50)               # "100.5" (without formatting)
```

---

## Type Operations

**📚 Examples:** [`examples/en/03-data-types/`](../../examples/en/03-data-types/)

### `typeof(value)`

Returns a string with the type of the value.

**Arguments:**
- `value` (any) - value to determine type of

**Returns:** `string` - type name ("int", "float", "string", "bool", "array", "table", "path", "null", "function", "date", "money", "object")

**Examples:**
```datacode
typeof(42)              # "int"
typeof(3.14)            # "float"
typeof("hello")         # "string"
typeof(true)            # "bool"
typeof([1, 2, 3])       # "array"
typeof(null)            # "null"
typeof(path("test.txt")) # "path"
```

---

### `isinstance(value, type_name)`

Checks if a value is an instance of the specified type.

**Arguments:**
- `value` (any) - value to check
- `type_name` (string) - type name to check

**Returns:** `bool` - `true` if value matches type, otherwise `false`

**Examples:**
```datacode
isinstance(42, "int")           # true
isinstance(3.14, "float")       # true
isinstance("hello", "string")   # true
isinstance([1, 2], "array")     # true
isinstance(42, "string")        # false
```

---

## Path Operations

**📚 Examples:** 
- [`examples/en/01-basics/`](../../examples/en/01-basics/)
- [`examples/en/09-data-model-creation/01-file-operations.dc`](../../examples/en/09-data-model-creation/01-file-operations.dc)

### `path(string)`

Creates a path object from a string.

**Arguments:**
- `string` (string) - string with path to file or directory

**Returns:** `path` - path object

**Examples:**
```datacode
path("data.csv")
path("/home/user/file.txt")
path("folder/subfolder")
```

---

### `path_name(path)`

Returns the file or directory name from a path.

**Arguments:**
- `path` (path) - path object

**Returns:** `string` - file or directory name

**Examples:**
```datacode
path_name(path("data.csv"))           # "data.csv"
path_name(path("/home/user/file.txt")) # "file.txt"
```

---

### `path_parent(path)`

Returns the parent path.

**Arguments:**
- `path` (path) - path object

**Returns:** `path` - parent path, or `null` if no parent

**Examples:**
```datacode
path_parent(path("folder/file.txt"))  # path("folder")
path_parent(path("/home/user"))       # path("/home")
```

---

### `path_exists(path)`

Checks if a file or directory exists.

**Arguments:**
- `path` (path) - path object

**Returns:** `bool` - `true` if path exists, otherwise `false`

**Examples:**
```datacode
path_exists(path("data.csv"))
path_exists(path("/nonexistent"))
```

---

### `path_is_file(path)`

Checks if a path is a file.

**Arguments:**
- `path` (path) - path object

**Returns:** `bool` - `true` if path points to a file, otherwise `false`

**Examples:**
```datacode
path_is_file(path("data.csv"))
path_is_file(path("folder"))  # false
```

---

### `path_is_dir(path)`

Checks if a path is a directory.

**Arguments:**
- `path` (path) - path object

**Returns:** `bool` - `true` if path points to a directory, otherwise `false`

**Examples:**
```datacode
path_is_dir(path("folder"))
path_is_dir(path("file.txt"))  # false
```

---

### `path_extension(path)`

Returns the file extension.

**Arguments:**
- `path` (path) - path object

**Returns:** `string` - file extension (without dot), or empty string

**Examples:**
```datacode
path_extension(path("data.csv"))    # "csv"
path_extension(path("file.txt"))    # "txt"
path_extension(path("folder"))     # ""
```

---

### `path_stem(path)`

Returns the file name without extension.

**Arguments:**
- `path` (path) - path object

**Returns:** `string` - file name without extension

**Examples:**
```datacode
path_stem(path("data.csv"))    # "data"
path_stem(path("file.txt"))    # "file"
```

---

### `path_len(path)`

Returns the length of the string representation of the path.

**Arguments:**
- `path` (path) - path object

**Returns:** `number` - path length in characters

**Examples:**
```datacode
path_len(path("data.csv"))     # 8
path_len(path("folder/file"))  # 12
```

---

## Mathematical Functions

**📚 Examples:** [`examples/en/01-basics/arithmetic.dc`](../../examples/en/01-basics/arithmetic.dc)

### `abs(n)`

Returns the absolute value of a number.

**Arguments:**
- `n` (number) - number

**Returns:** `number` - absolute value, or `null` if argument is not a number

**Examples:**
```datacode
abs(-5)      # 5
abs(5)       # 5
abs(-3.14)   # 3.14
```

---

### `sqrt(n)`

Returns the square root of a number.

**Arguments:**
- `n` (number) - number (must be non-negative)

**Returns:** `number` - square root, or `null` if number is negative or argument is not a number

**Examples:**
```datacode
sqrt(16)     # 4.0
sqrt(9)      # 3.0
sqrt(-1)     # null
```

---

### `pow(base, exp)`

Raises a number to a power.

**Arguments:**
- `base` (number) - base
- `exp` (number) - exponent

**Returns:** `number` - result of exponentiation, or `null` if arguments are not numbers

**Examples:**
```datacode
pow(2, 3)      # 8.0
pow(10, 2)     # 100.0
pow(2, 0.5)    # 1.4142135623730951 (square root of 2)
```

---

### `min(...)`

Returns the minimum value from the passed numbers.

**Arguments:**
- `...` - any number of numbers

**Returns:** `number` - minimum value, or `null` if arguments are not numbers or list is empty

**Examples:**
```datacode
min(1, 2, 3)        # 1
min(5, 2, 8, 1)     # 1
min(-5, -2, -10)    # -10
```

---

### `max(...)`

Returns the maximum value from the passed numbers.

**Arguments:**
- `...` - any number of numbers

**Returns:** `number` - maximum value, or `null` if arguments are not numbers or list is empty

**Examples:**
```datacode
max(1, 2, 3)        # 3
max(5, 2, 8, 1)     # 8
max(-5, -2, -10)    # -2
```

---

### `round(n)`

Rounds a number to the nearest integer.

**Arguments:**
- `n` (number) - number to round

**Returns:** `number` - rounded number, or `null` if argument is not a number

**Examples:**
```datacode
round(3.5)      # 4
round(3.4)      # 3
round(-3.5)     # -3
round(-3.6)     # -4
```

---

## String Functions

**📚 Examples:** [`examples/en/01-basics/strings.dc`](../../examples/en/01-basics/strings.dc)

### `upper(str)`

Converts a string to uppercase.

**Arguments:**
- `str` (string) - string to convert

**Returns:** `string` - string in uppercase, or `null` if argument is not a string

**Examples:**
```datacode
upper("hello")      # "HELLO"
upper("Hello World") # "HELLO WORLD"
```

---

### `lower(str)`

Converts a string to lowercase.

**Arguments:**
- `str` (string) - string to convert

**Returns:** `string` - string in lowercase, or `null` if argument is not a string

**Examples:**
```datacode
lower("HELLO")      # "hello"
lower("Hello World") # "hello world"
```

---

### `trim(str)`

Removes whitespace from the beginning and end of a string.

**Arguments:**
- `str` (string) - string to process

**Returns:** `string` - string without leading and trailing whitespace, or `null` if argument is not a string

**Examples:**
```datacode
trim("  hello  ")      # "hello"
trim("  test  world  ") # "test  world"
```

---

### `split(str, delim)`

Splits a string by the specified delimiter.

**Arguments:**
- `str` (string) - string to split
- `delim` (string) - delimiter

**Returns:** `array` - array of strings, or `null` if arguments are not strings

**Examples:**
```datacode
split("a,b,c", ",")           # ["a", "b", "c"]
split("one two three", " ")   # ["one", "two", "three"]
split("hello", "")            # ["h", "e", "l", "l", "o"]
```

---

### `join(array, delim)`

Joins array elements into a string with the specified delimiter.

**Arguments:**
- `array` (array) - array of values
- `delim` (string) - delimiter

**Returns:** `string` - joined string, or `null` if arguments are of wrong type

**Examples:**
```datacode
join(["a", "b", "c"], ",")        # "a,b,c"
join([1, 2, 3], " - ")            # "1 - 2 - 3"
join(["hello", "world"], " ")     # "hello world"
```

---

### `contains(str, substr)`

Checks if a string contains a substring.

**Arguments:**
- `str` (string) - string to search
- `substr` (string) - substring to search for

**Returns:** `bool` - `true` if string contains substring, otherwise `false`

**Examples:**
```datacode
contains("hello world", "world")  # true
contains("hello world", "test")   # false
contains("hello", "lo")           # true
```

---

## Array Functions

**📚 Examples:** [`examples/en/01-basics/`](../../examples/en/01-basics/), [`examples/en/07-loops/`](../../examples/en/07-loops/)

### `push(array, item)`

Adds an element to the end of an array (modifies the original array).

**Arguments:**
- `array` (array) - array to modify
- `item` (any) - element to add

**Returns:** `array` - the same array (for method chaining)

**Examples:**
```datacode
let arr = [1, 2, 3]
push(arr, 4)        # arr is now [1, 2, 3, 4]
push(arr, "hello")  # arr is now [1, 2, 3, 4, "hello"]
```

---

### `pop(array)`

Removes and returns the last element of an array.

**Arguments:**
- `array` (array) - array to modify

**Returns:** `any` - last element of array, or `null` if array is empty or argument is not an array

**Examples:**
```datacode
let arr = [1, 2, 3]
pop(arr)     # returns 3, arr is now [1, 2]
pop(arr)     # returns 2, arr is now [1]
pop([])      # null
```

---

### `unique(array)`

Returns a new array with unique elements (preserves order of first occurrence).

**Arguments:**
- `array` (array) - array to process

**Returns:** `array` - new array with unique elements, or `null` if argument is not an array

**Examples:**
```datacode
unique([1, 2, 2, 3, 1])        # [1, 2, 3]
unique(["a", "b", "a", "c"])   # ["a", "b", "c"]
```

---

### `reverse(array)`

Reverses the order of elements in an array (modifies the original array).

**Arguments:**
- `array` (array) - array to modify

**Returns:** `array` - the same array with reversed element order

**Examples:**
```datacode
let arr = [1, 2, 3]
reverse(arr)  # arr is now [3, 2, 1]
```

---

### `sort(array)`

Sorts array elements by string representation (modifies the original array).

**Arguments:**
- `array` (array) - array to sort

**Returns:** `array` - sorted array

**Examples:**
```datacode
let arr = [3, 1, 2]
sort(arr)  # arr is now [1, 2, 3]

let arr2 = ["c", "a", "b"]
sort(arr2)  # arr2 is now ["a", "b", "c"]
```

---

### `sum(array)`

Calculates the sum of all numbers in an array.

**Arguments:**
- `array` (array) - array of numbers

**Returns:** `number` - sum of numbers, or `0` if no numbers or argument is not an array

**Examples:**
```datacode
sum([1, 2, 3])        # 6
sum([10, 20, 30])     # 60
sum([1.5, 2.5, 3.0])  # 7.0
```

---

### `average(array)`

Calculates the arithmetic mean of numbers in an array.

**Arguments:**
- `array` (array) - array of numbers

**Returns:** `number` - average value, or `0` if no numbers or argument is not an array

**Examples:**
```datacode
average([1, 2, 3])        # 2.0
average([10, 20, 30])     # 20.0
average([1.5, 2.5, 3.0])  # 2.3333333333333335
```

---

### `count(array)`

Returns the number of elements in an array.

**Arguments:**
- `array` (array) - array

**Returns:** `number` - number of elements, or `0` if argument is not an array

**Examples:**
```datacode
count([1, 2, 3])      # 3
count([])             # 0
count(["a", "b"])     # 2
```

---

## Table Functions

**📚 Examples:** 
- [`examples/en/09-data-model-creation/`](../../examples/en/09-data-model-creation/)
- See also: [Working with Tables](./table_create_function.md)

### `table(data, headers)`

Creates a table from data and headers.

**Arguments:**
- `data` (array) - array of arrays, where each inner array represents a row
- `headers` (array, optional) - array of strings with column names

**Returns:** `table` - table object, or `null` if arguments are of wrong type

**Examples:**
```datacode
table([[1, "a"], [2, "b"]], ["ID", "Name"])
table([[1, 2, 3], [4, 5, 6]])  # Without headers
```

---

### `read_file(path)` / `read_file(path, header_row)` / `read_file(path, sheet_name="sheet_name")` / `read_file(path, header_row, sheet_name)` / `read_file(path, header_row, sheet_name, header)`

Reads a file and returns a table (for CSV/XLSX) or string (for TXT).

**Arguments:**
- `path` (path | string) - path to file
- `header_row` (number, optional) - row number with headers (0-based, default 0)
- `sheet_name` (string, optional) - sheet name for XLSX files (default first sheet)
- `header` (array | object, optional) - column filter or rename mapping:
  - If array: list of column names to load (only these columns will be included)
  - If object: dictionary mapping original column names to new names (use `null` to keep original name)

**Returns:** 
- `table` - for CSV and XLSX files
- `string` - for TXT files
- `null` - if file not found or error occurred

**Examples:**
```datacode
read_file("data.csv")
read_file("report.xlsx", "Sales")
read_file("data.csv", 2)
read_file(path("report.xlsx"), 1, "DataSheet")
read_file(path("notes.txt"))  # Returns string

# Load only specific columns
sample_table = read_file(path("sample.csv"), header_row=0, header=["Name", "Age", "City", "Salary"])

# Rename columns during load
sample_table = read_file(path("sample.csv"), header_row=0, header={"Name": "Name_A", "Age": null, "City": null, "Salary": null})

# Combine with sheet_name for XLSX
data = read_file(path("report.xlsx"), header_row=1, sheet_name="Data", header=["ID", "Value"])
```

**Notes:**
- For CSV files, data types are automatically detected
- For XLSX files, you can specify a specific sheet
- For XLSX files, you can specify the header row (if it's not the first)
- Argument order: `read_file(path, header_row, sheet_name, header)`
- When `header` is an array, only specified columns are loaded (non-existent columns are ignored)
- When `header` is an object, columns are renamed according to the mapping (columns not in the mapping keep their original names)

---

### `table_info(table)`

Returns information about a table (number of rows, columns, data types).

**Arguments:**
- `table` (table) - table

**Returns:** `string` - string with table information

**Examples:**
```datacode
let data = read_file(path("data.csv"))
print(table_info(data))
```

---

### `table_head(table, n)`

Returns the first n rows of a table.

**Arguments:**
- `table` (table) - table
- `n` (number, optional) - number of rows (default 5)

**Returns:** `table` - new table with first n rows, or `null` if argument is not a table

**Examples:**
```datacode
table_head(data)      # First 5 rows
table_head(data, 10)  # First 10 rows
```

---

### `table_tail(table, n)`

Returns the last n rows of a table.

**Arguments:**
- `table` (table) - table
- `n` (number, optional) - number of rows (default 5)

**Returns:** `table` - new table with last n rows, or `null` if argument is not a table

**Examples:**
```datacode
table_tail(data)      # Last 5 rows
table_tail(data, 10)  # Last 10 rows
```

---

### `table_select(table, columns)`

Selects specified columns from a table.

**Arguments:**
- `table` (table) - table
- `columns` (array) - array of strings with column names to select

**Returns:** `table` - new table with only selected columns, or `null` if column not found or arguments are of wrong type

**Examples:**
```datacode
table_select(data, ["Name", "Age"])
table_select(data, ["ID"])
```

---

### `table_sort(table, column, ascending)`

Sorts a table by the specified column.

**Arguments:**
- `table` (table) - table
- `column` (string) - column name to sort by
- `ascending` (bool, optional) - sort direction (default `true`)

**Returns:** `table` - new sorted table, or `null` if column not found or arguments are of wrong type

**Examples:**
```datacode
table_sort(data, "Age")              # Ascending
table_sort(data, "Name", true)       # Ascending
table_sort(data, "Salary", false)    # Descending
```

---

### `table_where(table, column, operator, value)`

Filters a table by condition.

**Arguments:**
- `table` (table) - table
- `column` (string) - column name to filter by
- `operator` (string) - comparison operator (">", "<", ">=", "<=", "==", "=", "!=", "<>")
- `value` (any) - value to compare

**Returns:** `table` - new table with filtered rows, or `null` if column not found or arguments are of wrong type

**Examples:**
```datacode
table_where(data, "Age", ">", 18)
table_where(data, "Name", "==", "John")
table_where(data, "Salary", ">=", 50000)
table_where(data, "Status", "!=", "inactive")
```

---

### `show_table(table)`

Outputs a table to the console in a nice format.

**Arguments:**
- `table` (table) - table to output

**Returns:** `null`

**Examples:**
```datacode
let data = read_file(path("data.csv"))
show_table(data)
show_table(table_head(data, 10))
```

**Notes:**
- Outputs maximum 20 rows for large tables
- Automatically adjusts column widths
- Uses Unicode characters for table borders

---

## ML Module

DataCode includes a powerful machine learning module `ml` that provides **68 functions** for working with tensors, creating and training neural networks, working with data, and much more.

### Main ML Module Features:

- **Tensor operations** (12 functions) - creating tensors, operations on them (addition, multiplication, matrix multiplication, etc.)
- **Graph operations** (9 functions) - working with computation graph for automatic differentiation
- **Linear Regression** (4 functions) - creating and training linear regression models
- **Optimizers** (5 functions) - optimizers (SGD, Adam, etc.)
- **Loss functions** (9 functions) - loss functions (MSE, Cross Entropy, MAE, etc.)
- **Dataset functions** (5 functions) - working with datasets, loading MNIST
- **Layer functions** (4 functions) - creating neural network layers (Linear, ReLU, Softmax, Flatten)
- **Neural Network functions** (20 functions) - creating, training and saving neural networks

### Using the ML Module:

```datacode
import ml

# Creating a tensor
t = ml.tensor([[1, 2], [3, 4]])

# Creating a neural network
layer1 = ml.layer.linear(784, 128)
layer2 = ml.layer.relu()
layer3 = ml.layer.linear(128, 10)
model = ml.neural_network(ml.sequential([layer1, layer2, layer3]))

# Training the model
loss_history = model.train(x_train, y_train, 10, 32, 0.001, "cross_entropy")
```

**📚 Full documentation:** [DataCode ML Module](./ml/ml_module.md)

---

## Summary

DataCode provides **50 built-in functions** and **68 ML module functions**, organized into the following categories:

### Built-in Functions (50):

- **Utilities**: 3 functions (print, len, range)
- **Type conversion**: 7 functions (int, float, bool, str, array, date, money)
- **Type operations**: 2 functions (typeof, isinstance)
- **Path operations**: 9 functions (path, path_name, path_parent, path_exists, path_is_file, path_is_dir, path_extension, path_stem, path_len)
- **Mathematical**: 6 functions (abs, sqrt, pow, min, max, round)
- **String**: 6 functions (upper, lower, trim, split, join, contains)
- **Arrays**: 8 functions (push, pop, unique, reverse, sort, sum, average, count)
- **Tables**: 9 functions (table, read_file, table_info, table_head, table_tail, table_select, table_sort, table_where, show_table)

### ML Module (68 functions):

- **Tensor operations**: 12 functions
- **Graph operations**: 9 functions
- **Linear Regression**: 4 functions
- **Optimizers**: 5 functions
- **Loss functions**: 9 functions
- **Dataset functions**: 5 functions
- **Layer functions**: 4 functions
- **Neural Network functions**: 20 functions

All functions are fully integrated into the language and can be used in expressions, conditions, and loops.

---

## Related Documents

- [Data Types](./data_types.md) - detailed description of DataCode data types
- [Working with Tables](./table_create_function.md) - creating and working with tables
- [ML Module](./ml/ml_module.md) - complete description of the machine learning module
- [Usage Examples](../../examples/en/) - practical examples of all functions

---

**Note:** This documentation is created based on analysis of DataCode source code and usage examples. All functions are tested and work in the current version of the language.


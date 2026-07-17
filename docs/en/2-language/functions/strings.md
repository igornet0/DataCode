# String Functions

← [Built-in Functions](./README.md)

**📚 Examples:** [`examples/en/01-basics/strings.dc`](../../../examples/en/01-basics/strings.dc)

### `upper(str)`

Converts a string to uppercase.

**Arguments:**
- `str` (string) — string to convert

**Returns:** `string` — uppercase string, or `null` if argument is not a string

**Examples:**
```datacode
upper("hello")      # "HELLO"
upper("Hello World") # "HELLO WORLD"

```

---

### `lower(str)`

Converts a string to lowercase.

**Arguments:**
- `str` (string) — string to convert

**Returns:** `string` — lowercase string, or `null` if argument is not a string

**Examples:**
```datacode
lower("HELLO")      # "hello"
lower("Hello World") # "hello world"

```

---

### `trim(str)`

Removes whitespace from the start and end of a string.

**Arguments:**
- `str` (string) — string to process

**Returns:** `string` — trimmed string, or `null` if argument is not a string

**Examples:**
```datacode
trim("  hello  ")      # "hello"
trim("  test  world  ") # "test  world"

```

---

### `split(str, delim)`

Splits a string by the specified delimiter.

**Arguments:**
- `str` (string) — string to split
- `delim` (string) — delimiter

**Returns:** `array` — array of strings, or `null` if arguments are not strings

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
- `array` (array) — array of values
- `delim` (string) — delimiter

**Returns:** `string` — joined string, or `null` if arguments have wrong types

**Examples:**
```datacode
join(["a", "b", "c"], ",")        # "a,b,c"
join([1, 2, 3], " - ")            # "1 - 2 - 3"
join(["hello", "world"], " ")     # "hello world"

```

---

### `contains(str, substr)`

Checks whether a string contains a substring.

**Arguments:**
- `str` (string) — string to search in
- `substr` (string) — substring to find

**Returns:** `bool` — `true` if string contains substring, otherwise `false`

**Examples:**
```datacode
contains("hello world", "world")  # true
contains("hello world", "test")   # false
contains("hello", "lo")           # true

```

---

### `starts_with(str, prefix)`

Checks whether a string starts with the specified prefix.

**Arguments:**
- `str` (string) — string to check
- `prefix` (string) — prefix

**Returns:** `bool`

**Examples:**
```datacode
starts_with("hello", "he")   # true
starts_with("hello", "lo")   # false
```

---

### `ends_with(str, suffix)`

Checks whether a string ends with the specified suffix.

**Arguments:**
- `str` (string) — string to check
- `suffix` (string) — suffix

**Returns:** `bool`

**Examples:**
```datacode
ends_with("hello", "lo")   # true
ends_with("hello", "he")   # false
```

---

### `isupper(str)` / `islower(str)`

Check whether a string consists only of uppercase / lowercase characters (as in Python).

**Arguments:**
- `str` (string)

**Returns:** `bool`, or `null` if argument is not a string

**Examples:**
```datacode
isupper("HELLO")   # true
islower("hello")   # true
isupper("Hello")   # false
```

---

### `replace(str, find, replacement)`

Replaces **all** occurrences of substring `find` with `replacement` (like Python `str.replace`).

**Arguments:**
- `str` (string) — source string
- `find` (string) — substring to replace
- `replacement` (string) — replacement

**Returns:** `string`, or `null` if arguments are not strings

**Examples:**
```datacode
replace("foo-old-bar-old", "old", "new")   # "foo-new-bar-new"
"tag-old".replace("old", "archive")        # "tag-archive"
```

---

### `capitalize(str)`

First character uppercase, rest lowercase (like Python `str.capitalize`).

**Arguments:**
- `str` (string)

**Returns:** `string`, or `null` if argument is not a string

**Examples:**
```datacode
capitalize("hello WORLD")   # "Hello world"
"tag".capitalize()          # "Tag"
```

---

### `ord(ch)`

Returns the numeric Unicode code point of a single character (like Python `ord()`).

**Arguments:**
- `ch` (string) — string containing exactly one Unicode character

**Returns:** `int` — code point (not UTF-8 byte)

**Examples:**
```datacode
ord("A")    # 65
ord("a")    # 97
ord("Я")    # 1071
ord("中")   # 20013
ord("😀")   # 128512
```

**Errors:**
- `TypeError` — wrong number of arguments, empty string, or more than one character
- `RuntimeError` — argument is not a string (`ord(123)`, `ord(true)`, …)

---

# Path Functions

← [Built-in Functions](./README.md)

**📚 Examples:**
- [`examples/en/01-basics/`](../../../examples/en/01-basics/)
- [`examples/en/08-data-model-creation/01-file-operations.dc`](../../../examples/en/08-data-model-creation/01-file-operations.dc)

### `path(string)`

Creates a path object from a string.

**Arguments:**
- `string` (string) — path to a file or directory

**Returns:** `path` — path object

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
- `path` (path) — path object

**Returns:** `string` — file or directory name

**Examples:**
```datacode
path_name(path("data.csv"))           # "data.csv"
path_name(path("/home/user/file.txt")) # "file.txt"

```

---

### `path_parent(path)`

Returns the parent path.

**Arguments:**
- `path` (path) — path object

**Returns:** `path` — parent path, or `null` if there is no parent

**Examples:**
```datacode
path_parent(path("folder/file.txt"))  # path("folder")
path_parent(path("/home/user"))       # path("/home")

```

---

### `path_exists(path)`

Checks whether a file or directory exists.

**Arguments:**
- `path` (path) — path object

**Returns:** `bool` — `true` if path exists, otherwise `false`

**Examples:**
```datacode
path_exists(path("data.csv"))
path_exists(path("/nonexistent"))

```

---

### `path_is_file(path)`

Checks whether the path points to a file.

**Arguments:**
- `path` (path) — path object

**Returns:** `bool` — `true` if path is a file, otherwise `false`

**Examples:**
```datacode
path_is_file(path("data.csv"))
path_is_file(path("folder"))  # false

```

---

### `path_is_dir(path)`

Checks whether the path points to a directory.

**Arguments:**
- `path` (path) — path object

**Returns:** `bool` — `true` if path is a directory, otherwise `false`

**Examples:**
```datacode
path_is_dir(path("folder"))
path_is_dir(path("file.txt"))  # false

```

---

### `path_extension(path)`

Returns the file extension.

**Arguments:**
- `path` (path) — path object

**Returns:** `string` — file extension (without dot), or empty string

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
- `path` (path) — path object

**Returns:** `string` — file name without extension

**Examples:**
```datacode
path_stem(path("data.csv"))    # "data"
path_stem(path("file.txt"))    # "file"

```

---

### `path_len(path)`

Returns the length of the string representation of the path.

**Arguments:**
- `path` (path) — path object

**Returns:** `number` — path length in characters

**Examples:**
```datacode
path_len(path("data.csv"))     # 8
path_len(path("folder/file"))  # 12

```

---

### `getcwd()`

Current working directory of the process.

**Returns:** `path` (in `--use-ve` mode may return an empty path)

---

### `list_files(path, regex?)`

Recursively walks a directory; returns an array of paths to files and subfolders.

**Arguments:**
- `path` (path | string)
- `regex` (string, optional) — glob (`*.csv`) or regex to filter names

**Returns:** `array` of `path`

**Examples:**
```datacode
list_files(path("./data"))
list_files(path("./data"), "*.csv")
```

Supports local paths and `lib://` (SMB).

---

### `archive(path)`

Opens an archive and returns an `archive` object. Format is detected from magic bytes (ZIP, 7Z, RAR).

**Arguments:**
- `path` (path | string) — path to archive file

**Returns:** `archive`

**Example:**
```datacode
zip = archive("./backup.zip")
print(zip.format)
print(zip.count)
data = zip.read("config/settings.json")
text = zip.read_text("README.md")
zip.extract("./output")
zip.close()
```

See also: [archive type](../data-types/archive.md)

---

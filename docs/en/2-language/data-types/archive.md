# Archive type

← [Data types](./README.md)

The `archive` object represents an open archive (ZIP, 7Z, RAR). Created by the [`archive()`](../functions/paths.md#archivepath) function.

## Properties

| Property | Type | Description |
|----------|-----|----------|
| `path` | `path` | Path to the archive file |
| `format` | `string` | Format: `zip`, `7z`, `rar` |
| `files` | `array` | List of file metadata objects |
| `count` | `number` | Number of files |
| `size` | `number` | Total uncompressed size |
| `compressed_size` | `number` | Archive file size on disk |

Each entry in `files` contains:

- `path`, `name`, `directory`, `extension`
- `size`, `compressed_size`

## Methods

### `read(path)`

Reads a file from the archive with automatic type detection (same rules as [`read()`](../functions/tables.md)):

- `.json` → `object`
- `.csv` → `table`
- `.txt`, `.md` → `string`
- `.png`, `.jpg` → `image`
- `.bin` → bytes

### `read_text(path)`

Forces the file to be read as a UTF-8 string.

### `extract(dest)`

Extracts the entire archive into directory `dest` (zip-slip safe).

### `close()`

Closes the archive and releases resources.

## Example

```datacode
zip = archive("./backup.zip")
print(zip.format)
print(zip.count)
for file in zip.files {
    print(file.path)
}
data = zip.read("config/settings.json")
text = zip.read_text("README.md")
zip.extract("./output")
zip.close()
```

## Supported formats

| Format | Magic bytes | Notes |
|--------|-------------|-------|
| ZIP | `PK\x03\x04` | Built-in |
| 7Z | `7z\xBC\xAF\x27\x1C` | Built-in |
| RAR | `Rar!\x1a\x07` | Requires build with `--features archive-rar` |

Format is detected from file contents, not the extension.

## `typeof` / `isinstance`

```datacode
typeof(zip)                   # "archive"
isinstance(zip, "archive")    # true
```

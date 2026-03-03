# uuid Module — Unique Identifier Generation

The `uuid` module generates unique identifiers (UUIDs): random (v4), time-ordered (v7), deterministic (v3, v5), string parsing, byte representation, and metadata extraction.

## Module Import

Import the module:

```datacode
import uuid
```

## UUID Generation

### uuid.v4() -> UUID
### uuid.random() -> UUID

Generates a random UUID version 4. The `random()` alias is equivalent to `v4()`.

**Returns:** UUID (random).

**Example:**
```datacode
let u = uuid.v4()
print(str(u))
# or
let r = uuid.random()
print(str(r))
```

### uuid.v7() -> UUID
### uuid.new() -> UUID

Generates a time-ordered UUID version 7. Useful for indexes and sorting by creation time. The `new()` alias is equivalent to `v7()`.

**Returns:** UUID (time-ordered).

**Example:**
```datacode
let u = uuid.v7()
print(str(u))
let n = uuid.new()
print(str(n))
```

## Parsing and Strings

### uuid.parse(s: string) -> UUID | null

Parses a string in hyphenated format (e.g., `550e8400-e29b-41d4-a716-446655440000`) and returns a UUID. Returns `null` for invalid format.

**Parameters:**
- `s` (string): string in standard UUID format.

**Returns:** UUID or `null` on parse error.

**Example:**
```datacode
let parsed = uuid.parse("550e8400-e29b-41d4-a716-446655440000")
if parsed != null {
    print(str(parsed))
} else {
    print("Parse error")
}
```

### uuid.to_string(u: UUID) -> string

Converts a UUID to a string in standard hyphenated format.

**Parameters:**
- `u` (UUID): UUID value.

**Returns:** string in format `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`.

**Example:**
```datacode
let u = uuid.v4()
print(uuid.to_string(u))
```

## Bytes

### uuid.to_bytes(u: UUID) -> array

Returns an array of 16 numbers (0–255) — byte representation of the UUID.

**Parameters:**
- `u` (UUID): UUID value.

**Returns:** array of 16 numbers.

**Example:**
```datacode
let u = uuid.v4()
let bytes = uuid.to_bytes(u)
print(len(bytes))  # 16
```

### uuid.from_bytes(arr: array) -> UUID | null

Builds a UUID from an array of 16 numbers (0–255). The array must contain exactly 16 integers in range 0–255. On invalid input, returns `null` and sets a native error.

**Parameters:**
- `arr` (array): array of 16 numbers 0–255.

**Returns:** UUID or `null` on error.

**Example:**
```datacode
let u = uuid.v4()
let bytes = uuid.to_bytes(u)
let restored = uuid.from_bytes(bytes)
print(str(u) == str(restored))  # true
```

## Deterministic UUIDs (v3, v5)

### uuid.v3(namespace: UUID, name: string) -> UUID

Generates a UUID version 3 (MD5) from namespace and name. Same namespace and name yield the same UUID.

**Parameters:**
- `namespace` (UUID): namespace (use uuid.DNS, uuid.URL, uuid.OID, or a custom UUID).
- `name` (string): name.

**Returns:** UUID.

**Example:**
```datacode
let u = uuid.v3(uuid.URL, "https://example.com/page")
print(str(u))
```

### uuid.v5(namespace: UUID, name: string) -> UUID

Generates a UUID version 5 (SHA-1) from namespace and name. Same namespace and name yield the same UUID. Preferred over v3.

**Parameters:**
- `namespace` (UUID): namespace.
- `name` (string): name.

**Returns:** UUID.

**Example:**
```datacode
let u = uuid.v5(uuid.DNS, "example.com")
print(str(u))
```

### Standard namespaces: uuid.DNS, uuid.URL, uuid.OID

UUID constants for use with v3 and v5:

- **uuid.DNS** — namespace for DNS names.
- **uuid.URL** — namespace for URLs.
- **uuid.OID** — namespace for OID (hierarchical identifiers).

**Example:**
```datacode
print(str(uuid.DNS))
print(str(uuid.URL))
let u5 = uuid.v5(uuid.OID, "1.2.3.4.5")
```

## Metadata

### uuid.version(u: UUID) -> number

Returns the UUID version number (1–7).

**Parameters:**
- `u` (UUID): UUID value.

**Returns:** number 1–7.

**Example:**
```datacode
let u4 = uuid.v4()
print(uuid.version(u4))  # 4
let u7 = uuid.v7()
print(uuid.version(u7))  # 7
```

### uuid.variant(u: UUID) -> number

Returns the UUID variant: 0=NCS, 1=RFC4122, 2=Microsoft, 3=Future. Typically 1 (RFC4122).

**Parameters:**
- `u` (UUID): UUID value.

**Returns:** number 0–3.

### uuid.timestamp(u: UUID) -> number | null

Returns Unix time (seconds with fractional part) for UUID versions 1 and 7. For v4, returns `null`.

**Parameters:**
- `u` (UUID): UUID value.

**Returns:** number (Unix time) or `null` for v4.

**Example:**
```datacode
let u7 = uuid.v7()
let ts = uuid.timestamp(u7)
if ts != null {
    print("Created at: " + ts)
}
```

## Examples

Example scripts are located in:

- **English:** [examples/en/13-uuid/](../../../examples/en/13-uuid/)
- **Russian:** [examples/ru/13-uuid/](../../../examples/ru/13-uuid/)

### Example contents (en)

1. **01-basic-usage.dc** — import, v4, v7, to_string, parse, typeof.
2. **02-deterministic-uuid.dc** — v3, v5 with namespace DNS, URL, OID.
3. **03-bytes.dc** — to_bytes, from_bytes, round-trip.
4. **04-metadata.dc** — version, variant, timestamp.
5. **05-practical-example.dc** — record ids, array, object.

### Example contents (ru)

1. **01-базовое_использование.dc** — импорт, v4/v7, to_string, parse, typeof.
2. **02-детерминированные_uuid.dc** — v3 и v5 с namespace.
3. **03-байты.dc** — to_bytes и from_bytes.
4. **04-метаданные.dc** — version, variant, timestamp.
5. **05-практический_пример.dc** — генерация id для записей.

**Run an example (from project root):**
```bash
datacode --bin datacode examples/en/13-uuid/01-basic-usage.dc
```

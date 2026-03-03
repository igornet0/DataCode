# UUID in DataCode

This section demonstrates the built-in `uuid` module: generating unique identifiers (v4, v7), parsing strings, working with bytes, deterministic UUIDs (v3, v5), and metadata.

## Contents

### 1. `01-basic-usage.dc` — Basic usage
**Description**: Import the module, generate UUID v4 and v7, output as string, parse string to UUID.

**What you'll learn**:
- Import module `uuid`
- Generate random UUID: `uuid.v4()`, `uuid.random()`
- Generate time-ordered UUID: `uuid.v7()`, `uuid.new()`
- Convert to string: `uuid.to_string(u)`
- Parse string: `uuid.parse(s)` and null check
- Type: `typeof(u)`

**Run**:
```bash
datacode --bin datacode examples/en/13-uuid/01-basic-usage.dc
```

### 2. `02-deterministic-uuid.dc` — Deterministic UUIDs (v3, v5)
**Description**: Generate reproducible UUIDs from namespace and name using v3 and v5.

**What you'll learn**:
- Functions `uuid.v3(namespace, name)` and `uuid.v5(namespace, name)`
- Standard namespaces: `uuid.DNS`, `uuid.URL`, `uuid.OID`
- Same namespace and name yield the same UUID

**Run**:
```bash
datacode --bin datacode examples/en/13-uuid/02-deterministic-uuid.dc
```

### 3. `03-bytes.dc` — Byte representation
**Description**: Convert UUID to byte array and back.

**What you'll learn**:
- `uuid.to_bytes(u)` — get array of 16 numbers (0–255)
- `uuid.from_bytes(arr)` — build UUID from array
- Round-trip: UUID → bytes → UUID → to_string

**Run**:
```bash
datacode --bin datacode examples/en/13-uuid/03-bytes.dc
```

### 4. `04-metadata.dc` — UUID metadata
**Description**: Get version, variant, and timestamp from UUID.

**What you'll learn**:
- `uuid.version(u)` — UUID version (1–7)
- `uuid.variant(u)` — variant (0=NCS, 1=RFC4122, 2=Microsoft, 3=Future)
- `uuid.timestamp(u)` — Unix time for v1/v7 or `null` for v4

**Run**:
```bash
datacode --bin datacode examples/en/13-uuid/04-metadata.dc
```

### 5. `05-practical-example.dc` — Practical example
**Description**: Generate identifiers for records, output and store in data structures.

**What you'll learn**:
- Typical use of `uuid.v7()` for unique record ids
- Output as strings and store in array/object

**Run**:
```bash
datacode --bin datacode examples/en/13-uuid/05-practical-example.dc
```

## Concepts

### UUID generation
- **uuid.v4()**, **uuid.random()**: Random UUID (version 4)
- **uuid.v7()**, **uuid.new()**: Time-ordered UUID (version 7), good for indexes

### Parsing and strings
- **uuid.parse(s)**: String in hyphenated format → UUID or `null` on error
- **uuid.to_string(u)**: UUID → string in standard format or str(u)

### Bytes
- **uuid.to_bytes(u)**: UUID → array of 16 numbers 0–255
- **uuid.from_bytes(arr)**: Array of 16 numbers → UUID

### Deterministic UUIDs
- **uuid.v3(namespace, name)**: UUID version 3 (MD5)
- **uuid.v5(namespace, name)**: UUID version 5 (SHA-1)
- **uuid.DNS**, **uuid.URL**, **uuid.OID**: Standard namespaces for v3/v5

### Metadata
- **uuid.version(u)**: Version number (1–7)
- **uuid.variant(u)**: Variant (0–3)
- **uuid.timestamp(u)**: Unix time for v1/v7, `null` for v4

## Navigation

### Previous section
- **[12-settings-env](../12-settings-env/)** — settings_env module

### Next (back to start)
- **[01-basics](../01-basics/)** — Language basics

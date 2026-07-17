# Cryptography and Randomness

← [Built-in Functions](./README.md)

Hashing (SHA-2, HMAC) and random data generation. **Not intended for password storage** — for passwords use the [`crypto`](../modules/crypto/README.md) module (`Argon2`, `bcrypt`).

**📚 Tests:** `tests/crypto_tests.rs`

### `sha256(data)`

Computes SHA-256 of a string or bytes.

**Arguments:**
- `data` (string | bytes) — input data

**Returns:** `bytes` of length 32, or `null` on type error

**Display:** `typeof` → `"bytes"`; `str(sha256(...))` → lowercase hex (64 characters). Data remains binary — index `d[i]`, `len(d)`, `hmac_*` work as before.

**Examples:**
```datacode
len(sha256(""))       # 32
str(sha256("hello"))  # 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
print(sha256("hello"))
```

---

### `sha512(data)`

Same as `sha256`, but SHA-512 (64-byte result). `str(...)` — 128-character hex.

---

### `hmac_sha256(key, data)`

HMAC-SHA256. Both arguments must be **bytes** (not strings).

**Arguments:**
- `key` (bytes)
- `data` (bytes)

**Returns:** `bytes` (32 bytes), or `null`. `str(...)` — hex digest.

**Examples:**
```datacode
key = random_bytes(16)
data = random_bytes(8)
hmac_sha256(key, data)
```

---

### `hmac_sha512(key, data)`

HMAC-SHA512; both arguments are `bytes`, result is 64 bytes. `str(...)` — hex digest.

---

### `random_bytes(size)`

Cryptographically strong random bytes from OS RNG.

**Arguments:**
- `size` (int) — non-negative integer; maximum 1 048 576 (1 MiB)

**Returns:** `bytes`, or error on limit exceeded / wrong type

**Display:** `typeof` → `"bytes"`; `str(...)` → `"<bytes len=N>"` (not hex).

**Examples:**
```datacode
random_bytes(16)
```

---

### `random()`

Random **floating-point** number in half-open interval **[0, 1)** — 0 inclusive, 1 exclusive (like Python `random.random()`). Uses the same PRNG stream as `random_int` and `random_seed`.

**Arguments:** none

**Returns:** `float` (`number`)

**Examples:**
```datacode
x = random()
random_seed(42)
a = random()
random_seed(42)
b = random()
# a == b
```

---

### `random_int(min, max)`

Random **integer** in range `[min, max]` inclusive (PRNG stream; see `random_seed`).

**Arguments:**
- `min` (int)
- `max` (int), `min <= max`

**Returns:** `number` (integer), or `null` on error

**Examples:**
```datacode
random_int(1, 6)          # "dice roll"
random_int(0, 100)
```

---

### `random_seed(seed)`

Fixes the PRNG seed **in the current thread** (for reproducible `random`, `random_int` sequences).

**Arguments:**
- `seed` (int) — non-negative integer

**Returns:** `null`

**Examples:**
```datacode
random_seed(42)
a = random_int(0, 999999)
random_seed(42)
b = random_int(0, 999999)
# a == b
```

---

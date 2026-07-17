# Bytes (`bytes`)

Raw binary data: results of `sha256`, `read_bin`, `random_bytes`, and similar.

**Type checks:** [typeof and isinstance](typeof-and-isinstance.md)

---

## `typeof` and display

| Source | `typeof` | `str()` / `print()` |
|--------|----------|---------------------|
| `sha256`, `sha512`, `hmac_*` | `"bytes"` | **lowercase hex** (64 or 128 characters) |
| `random_bytes` | `"bytes"` | `"<bytes len=N>"` |

```datacode
d = sha256("hello")
print(typeof(d))   # bytes
print(str(d))      # 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824

b = random_bytes(8)
print(str(b))      # <bytes len=8>
```

`isinstance(x, "bytes")` is the primary check. For compatibility, `isinstance(x, "array")` is also **true** for `bytes`.

---

## Operations

- **`len(b)`** — number of bytes
- **`b[i]`** — byte as a number `0..255` (zero-based index)
- **Not iterable:** `for x in b` is not supported; use `for i in range(len(b)) { b[i] }`
- **Comparing digests:** `from crypto import secure_compare` — without converting to string

---

## Function typing

```datacode
fn digest_len(data: bytes) -> int {
    return len(data)
}
digest_len(sha256("hello"))   # 32
```

---

## Related documents

- [Cryptography and randomness](../functions/cryptography-and-randomness.md)
- [Path and files](path.md)

# crypto Module — Password Hashing and Secure Comparison

The `crypto` module is for **password storage** (Argon2id, bcrypt) and **comparing secrets** without timing leaks (`secure_compare`).

> For SHA-2, HMAC, and random bytes use **global** functions: [cryptography-and-randomness](../../functions/cryptography-and-randomness.md) (`sha256`, `random_bytes`, …).

**📚 Example:** [`examples/en/16-crypto/`](../../../../examples/en/16-crypto/)

## Import

```datacode
from crypto import Argon2, bcrypt, secure_compare
```

## Argon2id

Default parameters: ~19 MiB RAM, `t=2`, `p=1` (Argon2id v0x13).

### Argon2.hash(password) -> string

Hashes a password; returns a PHC-format string (`$argon2id$…`).

```datacode
h = Argon2.hash("mypassword")
```

### Argon2.verify(password, hash) -> bool

Verifies a password against a stored hash.

```datacode
Argon2.verify("mypassword", h)   # true
Argon2.verify("wrong", h)      # false
```

## bcrypt

Minimum cost: **12**.

### bcrypt.hash(password) -> string

```datacode
h = bcrypt.hash("mypassword")
```

### bcrypt.verify(password, hash) -> bool

```datacode
bcrypt.verify("mypassword", h)
```

## secure_compare(a, b) -> bool

**Constant-time** comparison for strings or `bytes` (HMAC tags, tokens). Both arguments must be the same type (both `string` or both `bytes`).

```datacode
from crypto import secure_compare

tag = hmac_sha256(key, data)
secure_compare(tag, expected_tag)
```

See also: [bytes type](../../data-types/README.md) (monolith; split doc pending).

## Related sections

| Task | Documentation |
|------|---------------|
| SHA-2, HMAC, RNG | [cryptography-and-randomness](../../functions/cryptography-and-randomness.md) |
| Examples | [1-examples/16-crypto](../../../1-examples/16-crypto.md) |

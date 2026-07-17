# Байты (`bytes`)

Сырые двоичные данные: результат `sha256`, `read_bin`, `random_bytes` и т.п.

**Проверка типа:** [typeof и isinstance](typeof-и-isinstance.md)

---

## `typeof` и отображение

| Источник | `typeof` | `str()` / `print()` |
|----------|----------|---------------------|
| `sha256`, `sha512`, `hmac_*` | `"bytes"` | **lowercase hex** (64 или 128 символов) |
| `random_bytes` | `"bytes"` | `"<bytes len=N>"` |

```datacode
d = sha256("hello")
print(typeof(d))   # bytes
print(str(d))      # 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824

b = random_bytes(8)
print(str(b))      # <bytes len=8>
```

`isinstance(x, "bytes")` — основная проверка. Для совместимости `isinstance(x, "array")` для `bytes` тоже **true**.

---

## Операции

- **`len(b)`** — число байт
- **`b[i]`** — байт как число `0..255` (индекс с нуля)
- **Не iterable:** `for x in b` не поддерживается; используйте `for i in range(len(b)) { b[i] }`
- **Сравнение digest:** `from crypto import secure_compare` — без преобразования в строку

---

## Типизация функций

```datacode
fn digest_len(data: bytes) -> int {
    return len(data)
}
digest_len(sha256("hello"))   # 32
```

---

## Связанные документы

- [Криптография и случайность](../функции/криптография-и-случайность.md)
- [Путь и файлы](путь.md)

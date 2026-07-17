# Модуль crypto — хеширование паролей и безопасное сравнение

Модуль `crypto` предназначен для **хранения паролей** (Argon2id, bcrypt) и **сравнения секретов** без утечки по времени (`secure_compare`).

> Для SHA-2, HMAC и генерации случайных байт используйте **глобальные** функции: [`криптография-и-случайность`](../../функции/криптография-и-случайность.md) (`sha256`, `random_bytes`, …).

**📚 Пример:** [`examples/ru/16-крипто/`](../../../../examples/ru/16-крипто/)

## Импорт

```datacode
from crypto import Argon2, bcrypt, secure_compare
```

## Argon2id

Параметры по умолчанию: ~19 MiB RAM, `t=2`, `p=1` (Argon2id v0x13).

### Argon2.hash(password) -> string

Хеширует пароль; возвращает строку в формате PHC (`$argon2id$…`).

```datacode
h = Argon2.hash("mypassword")
```

### Argon2.verify(password, hash) -> bool

Проверяет пароль против сохранённого хеша.

```datacode
Argon2.verify("mypassword", h)   # true
Argon2.verify("wrong", h)      # false
```

## bcrypt

Минимальная стоимость (cost): **12**.

### bcrypt.hash(password) -> string

```datacode
h = bcrypt.hash("mypassword")
```

### bcrypt.verify(password, hash) -> bool

```datacode
bcrypt.verify("mypassword", h)
```

## secure_compare(a, b) -> bool

Сравнение **за постоянное время** для строк или `bytes` (теги HMAC, токены). Оба аргумента должны быть одного типа (оба `string` или оба `bytes`).

```datacode
from crypto import secure_compare

tag = hmac_sha256(key, data)
secure_compare(tag, expected_tag)
```

См. также: [тип bytes](../../типы-данных/bytes.md).

## Связанные разделы

| Задача | Документация |
|--------|--------------|
| SHA-2, HMAC, RNG | [криптография-и-случайность](../../функции/криптография-и-случайность.md) |
| Примеры | [1-примеры/16-крипто](../../../1-примеры/16-крипто.md) |

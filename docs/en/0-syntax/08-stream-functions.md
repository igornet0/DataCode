# Stream Functions (Generators)

**Stream functions** (`stream fn`) let you pause execution, yield a value to the caller, and later continue **from the same place**. They are useful for lazy sequences, pipelines, and **two-way** exchange: the caller can pass a value back into the function.

Below is a practical guide without extra terminology. Parameters, local variables, `if`, and loops work the same as in an ordinary `fn`.

**Run the ready-made example:**

```bash
datacode examples/en/04-functions/stream_functions.dc
```

---

## In one sentence

- Declaration: **`stream fn`**.  
- **`return expr`** — yield **`expr`** to the consumer (like *yield*); the generator **stays alive**.  
- **`ereturn expr`** — **finish** the generator; read the result via **`.final()`**.  
- Step by step: **`.next()`**, **`.send(value)`**, flag **`.live`**.

---

## How to get a generator

Calling a **`stream fn`** does **not** run the body to completion immediately. It returns a **generator object** that you drive with a `for` loop or with `.next()` / `.send()` / `.final()`.

```datacode
stream fn count_three() {
    return 1
    return 2
    return 3
}

gen = count_three()
# gen is a generator; body runs as values are requested

```

---

## Three ways to "yield a value" inside `stream fn`

| Keyword | What it does | Who receives the value |
|---------|--------------|------------------------|
| **`return expr`** | Pause and **yield** `expr`. Generator **alive**. | Loop **`for x in gen`** or **`.next()`** / **`.send()`** |
| **`ireturn expr`** | Same meaning, but convenient for **`.next()`** and **`.send()`** scenarios | Caller via **`.next()`** / **`.send()`** |
| **`ereturn expr`** | **End** of generator; **`.live`** becomes `false` | Only **`.final()`** — **not** the next **`.next()`** value |

**Tip:** **`return`** / **`ireturn`** — "here is the next value outward"; **`ereturn`** — "I'm done; here is the result for **`.final()`**".

---

## Simple path: `for` loop

If you need all values in order — use **`for`**:

```datacode
stream fn nums() {
    return 10
    return 20
}

sum = 0
for x in nums() {
    sum = sum + x
}
# sum equals 30

```

If the stream ended via **`ereturn`**, read the result from the **same** **`gen`** object used in `for` via **`.final()`** (see below).

---

## `ereturn` and `.final()`

What **`return`** yields is enumerated in **`for`**. The **`ereturn`** expression is **separate**: it is returned by **`.final()`** after the generator finishes.

```datacode
stream fn demo(short: bool) {
    if short {
        return 100
        ereturn 999
    }
    return 200
}

g = demo(true)
for x in g {
    print(x)   # 100
}
print(g.final())   # 999

```

If there was no **`ereturn`**, after completion **`.final()`** is usually **`null`**.

---

## Generator methods (cheat sheet)

| Member | Meaning |
|--------|---------|
| **`.next()`** | Resume execution. Next **`return`** / **`ireturn`** value, or **`null`** if the generator already finished. |
| **`.send(value)`** | Like **`.next()`**, but if inside the function is waiting on **`x = return expr`**, **`value`** is stored in **`x`**. |
| **`.final()`** | After completion — **`ereturn`** result; otherwise often **`null`** until the end or if there was no **`ereturn`**. |
| **`.live`** | **`true`** until the generator reaches **`ereturn`** (or finishes). Handy for **`while g.live`**. |

---

## Two-way exchange: `ireturn`, `return`, `.send()`

When you need to **pass a value inside** to a paused function (analog of Python's `send`):

1. **`x = ireturn 40`** — on first resume, **40** goes outward; variable **`x`** is filled with what is supplied when resuming the **next** wait on **`return`**.
2. **`ereturn x * n`** — use the already supplied **`x`**.

```datacode
stream fn scaled(n: int) {
    x = ireturn 40
    ereturn x * n
}

g = scaled(20)
first = g.send(10)   # outward sees 40; inside x becomes 10
result = g.final()   # 10 * 20 = 200

```

If you call **`.next()`** instead of **`.send()`**, the slot gets the value from the last yield (in the spirit of "without a separate send"). In the tutorial example this gives **`x == 40`** and **`final == 800`** with **`n == 20`**.

---

## Manual iteration: `while g.live` and `.next()`

Without **`for`**, pull values yourself:

```datacode
stream fn steps(n: int) {
    x = ireturn 40
    d = return 50
    ereturn (x + d) * n
}

g = steps(10)
while g.live {
    r = g.next()
    print(r)
}
print(g.final())   # (40 + 50) * 10 = 900

```

Typically: first **`.next()`** — **`ireturn`** value, second — **`return`** value, another **`.next()`** may yield **`null`** when the stream is done; **`.final()`** still holds the **`ereturn`** result.

---

## Limitations (good to know)

- A **`for x in gen`** loop combined with **`x = return expr`** assignment (yield-await mode) does **not** replace the **`.next()`** / **`.send()`** scenario — for two-way exchange, use the iterator API.
- **`.final()`** is the value of the **last `ereturn`**, not a sum of all yields.

---

## See also

- [User-defined functions with type annotations](./07-typed-functions.md) — `fn` and type syntax inside `stream fn`
- Example: [`examples/en/04-functions/stream_functions.dc`](../../../examples/en/04-functions/stream_functions.dc)
- Integration script: [`tests/stream_fn_generators.dc`](../../../tests/stream_fn_generators.dc)

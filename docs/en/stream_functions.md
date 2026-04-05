# Stream functions (generators)

**Stream functions** let you pause a function, give a value to the caller, and later continue from the same place. They are useful for pipelines, lazy sequences, and **two-way** communication (the caller can send a value back).

This page explains how to use them in everyday code. Syntax details match ordinary `fn` functions (parameters, locals, `if`, loops).

**Run the walkthrough:**

```bash
datacode examples/en/05-functions/stream_functions.dc
```

---

## In one sentence

- Declare with **`stream fn`**.  
- **`return expr`** hands **`expr`** to whoever is consuming the generator (like *yield*).  
- **`ereturn expr`** **finishes** the generator; read the result with **`.final()`**.  
- For step-by-step control, use **`.next()`**, **`.send(value)`**, and **`.live`**.

---

## Creating a generator

Calling a `stream fn` does **not** run the body to the end. It returns a **generator object** you drive with `for`, or with `.next()` / `.send()` / `.final()`.

```datacode
stream fn count_three() {
    return 1
    return 2
    return 3
}

gen = count_three()
# gen is a generator; body runs as you pull values
```

---

## Three kinds of “return” inside `stream fn`

| Keyword | What it does | Who receives the value |
|--------|----------------|-------------------------|
| **`return expr`** | Pauses and **yields** `expr`. Generator stays **alive**. | `for x in gen`, or **`.next()`** / **`.send()`** (see below). |
| **`ireturn expr`** | Same idea as yield, but oriented toward **`.next()`** / **`.send()`** workflows (see two-way example). | Caller’s **`.next()`** or **`.send()`** |
| **`ereturn expr`** | **Ends** the generator. **`.live`** becomes `false`. | **`.final()`** — *not* the next `.next()` value |

**Tip:** Think of **`return`** / **`ireturn`** as “here is a value for the consumer” and **`ereturn`** as “I’m done; here is the closing result.”

---

## Easy path: iterate with `for`

If you only need each yielded value in order, use **`for`**:

```datacode
stream fn nums() {
    return 10
    return 20
}

sum = 0
for x in nums() {
    sum = sum + x
}
# sum is 30
```

After the loop, if the stream ended with **`ereturn`**, use **`.final()`** on the **same** generator variable you used in `for` (see next section).

---

## `ereturn` and `.final()`

Values from **`return`** appear in the `for` loop. The **`ereturn`** expression is **separate**: it is what **`.final()`** returns after the generator has finished.

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

If there is **no** `ereturn`, **`.final()`** is **`null`** once the generator is finished.

---

## Generator API (cheat sheet)

| Member | Meaning |
|--------|---------|
| **`.next()`** | Resume the stream. Returns the **next** yielded value (`return` / `ireturn`), or **`null`** if the generator is already finished. |
| **`.send(value)`** | Like `.next()`, but if the stream is waiting on a **`x = return expr`** assignment, **`value`** is what gets stored in **`x`**. |
| **`.final()`** | After completion, returns the **`ereturn`** expression, or **`null`** if there was no `ereturn`. Before completion, usually **`null`**. |
| **`.live`** | **`true`** until the generator hits **`ereturn`** or exits without more yields. Handy for **`while g.live`**. |

---

## Two-way channel: `ireturn`, `return`, `.send()`

Sometimes the **consumer** should pass a value **into** the paused function (similar to Python’s `send`).

Pattern:

1. **`x = ireturn 40`** — first resume yields **40** to the caller; inside the stream, **`x`** gets the value supplied when resuming the **following** `return` wait.
2. **`ereturn x * n`** — multiply using whatever ended up in **`x`**.

```datacode
stream fn scaled(n: int) {
    x = ireturn 40
    ereturn x * n
}

g = scaled(20)
first = g.send(10)   # caller sees 40; inside stream, x becomes 10
result = g.final()   # 10 * 20 = 200
```

If you use **`.next()`** instead of **`.send()`** where a value is expected, the runtime uses the **last yielded value** for that slot (similar in spirit to sending “no extra value”). Example from the sample file: **`.next()`** after **`ireturn 40`** leads to **`x == 40`** and **`final == 800`** with **`n == 20`**.

---

## Manual stepping: `while g.live` and `.next()`

When you do not use `for`, pull values yourself:

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

Typical pattern: first **`.next()`** yields **`ireturn`** value, next **`.next()`** yields **`return`** value, one more **`.next()`** may yield **`null`** once the generator has finished; **`.final()`** still holds the **`ereturn`** result.

---

## What not to do (yet)

- **`for x in gen`** together with **`x = return expr`** (yield-await style) is **not** supported the same way as **`.next()`** / **`.send()`** — use the iterator API for two-way flows.
- **`.final()`** returns the value of the **last `ereturn`**, not a running sum of every yield.

---

## Related

- [User-defined functions with type annotations](./user_functions.md) — `fn` and type syntax used inside `stream fn`
- Examples: [`examples/en/05-functions/stream_functions.dc`](../../examples/en/05-functions/stream_functions.dc)
- Integration-style script: [`tests/stream_fn_generators.dc`](../../tests/stream_fn_generators.dc)

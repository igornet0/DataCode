# UUID

**UUID** is a 128-bit identifier (stored internally as two 64-bit numbers). Useful for unique keys and interchange with native libraries.

`typeof` → **`"uuid"`**.

Check: `isinstance(x, "uuid")` — [typeof-and-isinstance.md](typeof-and-isinstance.md).

---

## What you can do in the language

- Compare for equality with **`==`** when two UUIDs represent the same value.
- Pass to API functions that expect a UUID.

Access like **`uuid["field"]`** is **not** supported in the core — there is no list of named properties on a UUID value.

---

## Where UUIDs come from

There is **no** standalone global constructor `uuid(...)` for an arbitrary string (at the time of this documentation). UUIDs usually come from:

- external API calls;
- native modules (`.dylib` / `.so`);
- project libraries.

---

## `len`

For UUID, **`len`** is not defined (you can expect an "empty" result for an unsupported type).

# File or directory path (`path`)

A **path** value stores a filesystem path. Created by **`path(...)`**, then used with `path_*` functions and **properties** in square brackets.

`typeof` → **`"path"`**.

Check: `isinstance(p, "path")` — [typeof-and-isinstance.md](typeof-and-isinstance.md).

---

## Properties `p["name"]`

String key — property name (like a field):

| Key | What you get |
|-----|----------------|
| `is_file` | `true` / `false` — regular file? |
| `is_dir` | `true` / `false` — directory? |
| `exists` | does the path exist? |
| `extension` | extension or `null` |
| `name` | file name / last segment or `null` |
| `parent` | parent path or `null` |

Example:

```dc
p = path("./README.md")
print(p["exists"])
print(p["name"])

```

---

## Built-in functions

**Path:**

- `path`, `path_name`, `path_parent`, `path_exists`, `path_is_file`, `path_is_dir`, `path_extension`, `path_stem`, **`path_len`**

**Files and directories:**

- `read`, `getcwd`, `list_files`

### Note on `len`

For paths **`len(p)` is not defined** — use **`path_len(p)`** if you need path length in the implementation sense.

```dc
p = path("/tmp")
print(path_len(p))

```

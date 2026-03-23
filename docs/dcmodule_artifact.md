# Datacode module artifact (`.dcmodule`)

A **`.dcmodule`** file is a **ZIP archive** containing a root `manifest.json` and one or more native libraries (`.so` / `.dylib` / `.dll`) at paths declared in the manifest. The VM does not use Cargo; it only reads the manifest, checks ABI compatibility, and `dlopen`s the chosen library.

## Canonical ABI entry symbols

The dynamic library must export the same symbols as any other native module:

- **`datacode_module_entry`** (preferred) — returns a pointer to `AbiModuleDescriptor`.
- **`datacode_module`** (fallback) — legacy `DatacodeModule` / register.

Do not document a different symbol name (for example `datacode_module_init`) as the canonical entry point.

## `manifest.json`

Minimum fields:

| Field | Meaning |
|-------|--------|
| `schema_version` | `1` for this format. |
| `name` | Import name (`import foo` → `foo`). |
| `version` | Artifact version string. |
| `abi_version` | `{ "major": N, "minor": M }` — must satisfy `datacode_abi::abi_compatible` with the VM. |
| `library` | *or* `targets` — relative path(s) to the native library inside the zip. |

Either `library` (single path) or `targets` (per-target triple → path) must be present. If both exist, the host resolves the library by matching `targets` to the current target triple; if there is no match, `library` is used as fallback.

JSON Schema: [`schemas/dcmodule-manifest.schema.json`](../schemas/dcmodule-manifest.schema.json).

### Example

```json
{
  "schema_version": 1,
  "name": "mathlib",
  "version": "1.0.0",
  "abi_version": { "major": 1, "minor": 3 },
  "library": "lib/libmathlib.dylib"
}
```

## Layout

```
name.dcmodule (zip)
├── manifest.json
└── lib/
    └── libmathlib.dylib
```

## Packing

Use `dpm pack <directory>` to produce `name.dcmodule` from a build directory that already contains `manifest.json` and the referenced files.

## Runtime resolution

The VM searches for `name.dcmodule` in the same locations as the loose `lib<name>.{dylib,so,dll}` (script base path, DPM package roots, current directory). When the archive is found, it is unpacked into a user cache directory (content-addressed) and the manifest is validated before `dlopen`.

## Registry / DPM

`dpm install` resolves a package into `<env>/packages/<name>/`. Native modules can live there as **`name.dcmodule`** alongside the clone; see `data_code::dpm::expected_dcmodule_path` for the conventional path.

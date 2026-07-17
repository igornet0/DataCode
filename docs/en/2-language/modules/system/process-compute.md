# CPU/GPU Control via `system.process`

```datacode
from system import process
```

Extension of the built-in `system.process` namespace (alongside shell `exec`).

## Device constants

| Constant | Value |
|----------|-------|
| `process.cpu` | CPU |
| `process.gpu` | Best available GPU |
| `process.cuda` | NVIDIA CUDA (stub) |
| `process.metal` | Apple Metal |
| `process.auto` | Auto-select by data size |

## API

| Method | Description |
|--------|-------------|
| `get_device()` / `set_device(d)` | Current device |
| `get_gpu_min_size()` / `set_gpu_min_size(n)` | Auto threshold (default 20000) |
| `auto_device(fn)` | Callback `(data_len) -> device` |
| `has_gpu()` / `has_cuda()` / `has_metal()` | Backend availability |
| `info()` | `{backend, gpu_name, memory, cores}` |
| `vector_add(a, b)` | Pairwise sum of numeric arrays |
| `vector_mul(a, b)` | Pairwise multiply |
| `run("add"\|"mul"\|"sum", ...)` | Universal dispatch |

## Example

```datacode
from system import process

process.set_device(process.auto)
process.set_gpu_min_size(50000)

a = range(100000).map(fn(x) => x * 1.0)
b = range(100000).map(fn(x) => x * 2.0)
c = process.vector_add(a, b)
print(process.get_device())
print(process.info()["backend"])
```

## Metal (macOS)

Build with GPU:

```bash
cargo build --release --features metal
cargo test process_compute --release --features metal
```

Element-wise `*` on two numeric arrays of the same length also uses the compute runtime (when auto/GPU is enabled and array length ≥ `gpu_min_size`).

## v1 limitations

- GPU runs **built-in** numeric kernels (`vector_add`, `vector_mul`, `sum`, array `*`), not arbitrary DC closures in `map`.
- CUDA — probe only (`has_cuda()` → false); implementation later.

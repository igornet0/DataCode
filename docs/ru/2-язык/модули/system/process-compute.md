# Управление CPU/GPU через `system.process`

```datacode
from system import process
```

Расширение встроенного пространства `system.process` (наряду с `exec` для shell).

## Константы устройств

| Константа | Значение |
|-----------|----------|
| `process.cpu` | CPU |
| `process.gpu` | Лучший доступный GPU |
| `process.cuda` | NVIDIA CUDA (заглушка) |
| `process.metal` | Apple Metal |
| `process.auto` | Автовыбор по размеру данных |

## API

| Метод | Описание |
|-------|----------|
| `get_device()` / `set_device(d)` | Текущее устройство |
| `get_gpu_min_size()` / `set_gpu_min_size(n)` | Порог auto (по умолчанию 20000) |
| `auto_device(fn)` | Callback `(data_len) -> device` |
| `has_gpu()` / `has_cuda()` / `has_metal()` | Доступность backend |
| `info()` | `{backend, gpu_name, memory, cores}` |
| `vector_add(a, b)` | Попарное сложение numeric arrays |
| `vector_mul(a, b)` | Попарное умножение |
| `run("add"\|"mul"\|"sum", ...)` | Универсальный dispatch |

## Пример

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

Сборка с GPU:

```bash
cargo build --release --features metal
cargo test process_compute --release --features metal
```

Element-wise `*` на двух numeric arrays одинаковой длины также использует compute runtime (если включён auto/GPU и массив ≥ `gpu_min_size`).

## Ограничения v1

- GPU выполняет **встроенные** numeric kernels (`vector_add`, `vector_mul`, `sum`, array `*`), не произвольные DC-closures в `map`.
- CUDA — только probe (`has_cuda()` → false), реализация позже.

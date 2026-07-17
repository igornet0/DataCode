# Жизненный цикл native-вызова (байткод → VM → ABI → плагин)

Как вызов из DataCode доходит до нативного плагина и возвращается, через существующий C ABI (`datacode_abi`). На границе только **`AbiValue`** ([`datacode_abi/src/value.rs`](../../../datacode_abi/src/value.rs)), не Rust `enum` из `data-code`.

## 1. Загрузка нативного модуля (`import`)

1. VM разрешает `import <name>` / `from <name> import …` через [`import_handler`](../../../src/vm/module_system/import_handler.rs).
2. Если нет `.dc` / встроенного модуля, вызывается [`try_load_native_module`](../../../src/vm/native_loader.rs):
   - опционально: `name.dcmodule` (zip + `manifest.json`) → кэш → путь к dylib;
   - иначе: `lib<name>.dylib` / `.so` / `<name>.dll` в base path, DPM, cwd.
3. `dlopen` + `dlsym` на **`datacode_module_entry`** (предпочтительно) или **`datacode_module`** — см. [`datacode_abi::module`](../../../datacode_abi/src/module.rs).
4. Экспорты → `Value::NativeFunction(…)` на объекте модуля; модуль хранится как обычный import.

Формат `.dcmodule`: [dcmodule-artifact.md](dcmodule-artifact.md).

## 2. Вызов натива из байткода

```mermaid
flowchart LR
  OpCall[Opcode call]
  Dispatch[call_dispatch]
  NativeExec[execute_native_call]
  Bridge[call_abi_native]
  AbiFn[NativeAbiFn in .so]
  OpCall --> Dispatch
  Dispatch --> NativeExec
  NativeExec --> Bridge
  Bridge --> AbiFn
```

1. **Opcode** кладёт callee и аргументы; VM вызывает [`call_dispatch`](../../../src/vm/runtime/call_engine/call_dispatch.rs).
2. Для `Value::NativeFunction(native_index)` при `native_index >= builtin_count` — [`execute_native_call`](../../../src/vm/runtime/call_engine/native_call.rs) → **`call_abi_native`** в [`native_loader.rs`](../../../src/vm/native_loader.rs).
3. **`call_abi_native`**: `Value` → `AbiValue` через [`AbiBridgeContext`](../../../src/vm/abi_bridge.rs), строит [`VmContext`](../../../datacode_abi/src/vm_context.rs) (`alloc`, `throw_error`, `register_native`), вызывает:

```text
extern "C" fn(*mut VmContext, *const AbiValue, argc) -> AbiValue
```

4. Ответ `AbiValue` → `Value` в bridge; ошибки ABI через `throw_error` → runtime exception.

## 3. `VmContext` и время жизни указателей

| Поле | Роль |
|------|------|
| `alloc` | Выделение памяти аллокатором VM (C-compatible). |
| `throw_error` | `DatacodeError` + UTF-8 сообщение в VM. |
| `register_native` | Legacy `register` (не для descriptor entry). |

Указатели внутри `AbiValue` (строки и т.д.) валидны **на время одного** native-вызова, если не указано иное.

## 4. Авторство плагинов

`datacode_sdk` + `cdylib`; см. [datacode_sdk/docs/ru/modules.md](../../../datacode_sdk/docs/ru/modules.md). Предпочтительно **`datacode_module_entry`** через `define_module_entry!` для статических таблиц экспорта.

## См. также

- [SDK и ABI в README](README.md#sdk-и-abi-нативные-модули)
- [Пользовательские операторы](operator-descriptor.md)
- [Система импорта модулей](module_import_system.md)

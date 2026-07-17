# 200 — Разработчикам

Внутренняя архитектура VM, рантайма и **нативных модулей** DataCode. Для разработчиков ядра.

**Пользовательская документация:** [docs/ru/README.md](../README.md)  
**English:** [docs/en/200-developers/](../../en/200-developers/README.md)  
**Карта репозитория:** [docs/STRUCTURE.md](../../STRUCTURE.md)

---

## VM и рантайм

| Документ | Описание |
|----------|----------|
| [Архитектура VM](vm_architecture.md) | `Vm`, `CallFrame`, executor; байткод (OpCode, Chunk, DCB) |
| [Модель выполнения](execution_model.md) | Стек, фреймы, вызов/возврат, исключения |
| [Глобалы и namespace](globals_and_namespace.md) | `GlobalSlot`, `global_names`, remap |
| [Система импорта модулей](module_import_system.md) | Import/ImportFrom, `ModuleObject`, `file_import` |
| [Профилирование](profiling.md) | Фича `profile`, `ProfileStats` |
| [Отрисовка текста в plot](text-rendering.md) | Реализация text rendering в модуле plot |

## Нативные плагины (ABI)

| Документ | EN |
|----------|-----|
| [Артефакт `.dcmodule`](dcmodule-artifact.md) | [dcmodule_artifact.md](../../en/200-developers/dcmodule_artifact.md) |
| [Жизненный цикл native-вызовов](native-call-lifecycle.md) | [native_call_lifecycle.md](../../en/200-developers/native_call_lifecycle.md) |
| [Пользовательские операторы](operator-descriptor.md) | [operator_descriptor.md](../../en/200-developers/operator_descriptor.md) |
| [Аудит мутирующих natives](vm-mutating-natives-audit.md) | [vm_mutating_natives_audit.md](../../en/200-developers/vm_mutating_natives_audit.md) |

## SDK и ABI

| Ресурс | Описание |
|--------|----------|
| [Datacode-sdk](https://github.com/igornet0/Datacode-sdk) | `define_module!`, `dc_fn!`, примеры |
| [Datacode-abi](https://github.com/igornet0/Datacode-abi) | C ABI: `AbiValue`, `VmContext`, версия **1.7** |

## Ключевые файлы исходного кода

- **VM:** `src/vm/vm.rs`, `src/vm/executor.rs`, `src/vm/frame.rs`
- **Байткод:** `src/bytecode/opcode.rs`, `src/bytecode/chunk.rs`, `src/vm/dcb.rs`
- **Нативные модули:** `src/vm/native_loader.rs`, `src/vm/abi_bridge.rs`
- **Мутирующие natives:** `src/vm/runtime/call_engine/native_call.rs`

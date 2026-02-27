# DataCode Internals — документация для разработчиков ядра

В этом разделе описаны внутренняя архитектура и реализация VM и рантайма DataCode. Документация предназначена для **разработчиков ядра** (интерпретатор, компилятор, рантайм).

**См. также:** [Основная документация на русском](../README.md) (пользовательская документация по языку и библиотекам).

---

## Содержание

| Документ | Описание |
|----------|----------|
| [Архитектура VM](vm_architecture.md) | Обзор VM: `Vm`, `CallFrame`, executor; байткод (OpCode, Chunk, DCB); цикл выполнения. |
| [Модель выполнения](execution_model.md) | Внутренняя модель: стек операндов, фреймы (slots, ip, stack_start), константы, вызов/возврат, исключения. |
| [Глобалы и namespace](globals_and_namespace.md) | `GlobalSlot`, builtins и глобалы модулей, `global_names`, `merge_global_names`, `update_chunk_indices_from_names`. |
| [Система импорта модулей](module_import_system.md) | Компиляция Import/ImportFrom; разрешение в рантайме (file_import); ModuleObject; remap после merge. |
| [Профилирование](profiling.md) | Фича `profile`, ProfileStats, сборка и интерпретация вывода. |

---

## Ключевые файлы исходного кода

- **VM:** `src/vm/vm.rs`, `src/vm/executor.rs`, `src/vm/frame.rs`
- **Байткод:** `src/bytecode/opcode.rs`, `src/bytecode/chunk.rs`, `src/vm/dcb.rs`
- **Глобалы:** `src/vm/global_slot.rs`, `src/vm/globals.rs`
- **Модули:** `src/vm/module_object.rs`, `src/vm/file_import.rs`
- **Remap:** `src/lib.rs` (`remap_module_function_ids_in_exports`, `remap_native_indices_in_exports`)
- **Компилятор:** `src/compiler/stmt/import.rs`, `src/compiler/variable/resolver.rs`, `src/compiler/scope.rs`
- **Профиль:** `src/vm/profile.rs`

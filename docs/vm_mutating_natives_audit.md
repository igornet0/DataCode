# Аудит мутирующих нативов VM (граница `load_value` / `ValueCell`)

Цель: отсутствие **O(n)** на каждый вызов для крупного контейнера из‑за полного `load_value` + `store_value` + `update_cell_if_mutable` в [`src/vm/runtime/call_engine/native_call.rs`](../src/vm/runtime/call_engine/native_call.rs).

## Статус (кратко)

| Натив | Индекс | Поведение | Заметки |
|-------|--------|-----------|---------|
| `push` | 35 | **Fast path** (in-place `ValueCell::Array` + `TaggedValue`) | Был узким местом O(n²); см. [`tests/performance_tests/LARGE_DATASET_10K_LOG.md`](../tests/performance_tests/LARGE_DATASET_10K_LOG.md) |
| `pop` | 36 | Общий путь | В [`native_push`/`native_pop`](../src/vm/natives/array.rs) `pop` делает CoW при `Rc::strong_count > 1`; при вызове из VM всё равно идёт `load_value` аргументов на границе — потенциальный кандидат на in-place fast path для `ValueCell::Array` |
| `reverse` | 38 | Общий путь | Мутирует массив; тот же класс проблемы при больших массивах в цикле |
| `sort` | 39 | Общий путь | То же |
| `unique` | 37 | Общий путь | То же |
| `table` | 45 | Уже есть fast path в `native_call` | Строит `flat_cell_ids` без полного `load_value` всей таблицы, где возможно |

## Приоритизация дальнейших оптимизаций

1. **Профиль**: `cargo build --release --features profile` и [`scripts/run_performance_profile.sh`](../scripts/run_performance_profile.sh) — топ опкодов по `store_allocations` / `store_get`.
2. **Паттерн как у `push`**: для мутаторов, которые **только** меняют существующий `ValueCell::Array`, добавлять ранний путь: `get_mut` + операция над `Vec<TaggedValue>` без полного `load_value` контейнера.
3. **Тесты**: [`tests/vm_push_complexity_tests.rs`](../tests/vm_push_complexity_tests.rs) — масштабирование по времени; при добавлении fast path для других нативов — аналогичные ratio-тесты.

## Связанные файлы

- [`src/vm/memory/convert.rs`](../src/vm/memory/convert.rs) — `load_value`, `store_value`, `update_cell_if_mutable`
- [`src/vm/natives/array.rs`](../src/vm/natives/array.rs) — реализации `native_*` над `Value::Array(Rc<RefCell<Vec<Value>>>)`

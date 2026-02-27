# Профилирование

VM может собирать лёгкую статистику выполнения при сборке с фичей **profile**. В документе описаны метрики, места их записи и способ сборки и использования фичи.

**Исходники:** [src/vm/profile.rs](../../../src/vm/profile.rs), [src/vm/executor.rs](../../../src/vm/executor.rs), [src/common/value_store.rs](../../../src/common/value_store.rs), [src/vm/vm.rs](../../../src/vm/vm.rs).

---

## Фича и стоимость

- **Флаг фичи:** `profile`. Включение: `cargo build --features profile` или добавление `profile` в `[features]` в Cargo.toml и обычная сборка.
- **При отключении:** Все вызовы API профиля — заглушки (пустые функции). Нет thread_local и аллокаций; нулевые накладные расходы в рантайме.
- **При включении:** В thread-local хранятся **ProfileStats** и имя текущего опкода. Записывается каждая выполненная инструкция и каждое allocate/get в store (см. ниже).

---

## ProfileStats

**ProfileStats** ([src/vm/profile.rs](../../../src/vm/profile.rs)) — структура со статистикой одного run():

| Поле | Значение |
|------|----------|
| **opcodes_executed** | Общее число выполненных инструкций. |
| **store_allocations** | Число аллокаций в value_store (напр. при сохранении нового значения). |
| **store_get_count** | Число операций get/load в value_store. |
| **alloc_by_opcode** | Отображение имя варианта опкода → число аллокаций, произошедших при текущем этом опкоде. |
| **get_by_opcode** | Отображение имя варианта опкода → число обращений get к store при этом опкоде. |

Имена опкодов — **variant name** без параметров (напр. `MakeArray(8)` и `MakeArray(3)` оба учитываются как `"MakeArray"`), из **OpCode::variant_name()**.

---

## Где записываются метрики

- **record_opcode()** — Вызывается в начале **execute_instruction** ([src/vm/executor.rs](../../../src/vm/executor.rs)). Увеличивает **opcodes_executed**.
- **set_current_opcode(op)** — Вызывается сразу после record_opcode с текущей инструкцией. Устанавливает thread-local «текущий опкод», к которому привязываются следующие allocate/get.
- **record_allocate()** — Вызывается из **value_store** при аллокации ([src/common/value_store.rs](../../../src/common/value_store.rs), напр. в allocate или при сохранении новой ячейки). Увеличивает **store_allocations** и **alloc_by_opcode[current_opcode]**.
- **record_store_get()** — Вызывается из value_store при get/load. Увеличивает **store_get_count** и **get_by_opcode[current_opcode]**.

Так можно соотнести горячие пути (напр. Constant, LoadLocal, MakeArray) с нагрузкой на store: какие опкоды дают больше всего аллокаций или загрузок.

---

## Жизненный цикл в run()

1. **Начало run()** ([src/vm/vm.rs](../../../src/vm/vm.rs)): **profile::set()** инициализирует thread-local ProfileStats (пустые alloc_by_opcode и get_by_opcode).
2. **Во время выполнения:** Каждый шаг вызывает **execute_instruction**; он вызывает **record_opcode()** и **set_current_opcode(&instruction)**. Операции store вызывают **record_allocate()** / **record_store_get()**.
3. **Конец run():** При выходе из основного цикла (по **VMStatus::Return** или **VMStatus::FrameEnded**) VM вызывает **profile::take()**, чтобы забрать статистику из thread-local. Если вернулось **Some(stats)**, вызывается **profile::print_stats(&stats)**.

**take()** забирает статистику из thread-local; повторный take в том же run возвращает None. Таким образом, за один run VM выводится статистика только одного запуска.

---

## Сборка и запуск с профилем

```bash
cargo build --features profile
# или
cargo run --features profile -- path/to/script.dc
```

Запускайте скрипт или REPL как обычно. В конце выполнения в stderr выводится профиль, например:

```
[profile] opcodes_executed   = 1234567
[profile] store_allocations  = 50000
[profile] store_get_count    = 200000
[profile] top 15 by alloc: [("MakeArray", 10000), ("Constant", 8000), ...]
[profile] top 15 by get:  [("LoadGlobal", 50000), ("GetArrayElement", 30000), ...]
[profile] top 15 by alloc+get: [("LoadGlobal", 55000), ("MakeArray", 12000), ...]
```

---

## Интерпретация вывода

- **opcodes_executed** — Общее число инструкций; при известном размере скрипта даёт примерное представление о весе циклов и функций.
- **store_allocations** / **store_get_count** — Высокие значения указывают на интенсивное использование value_store (объекты, массивы, не-inline глобалы). **alloc_by_opcode** и **get_by_opcode** показывают, какие инструкции доминируют.
- **Top by alloc** — Опкоды с наибольшим числом аллокаций (напр. MakeObject, MakeArray, Constant для не-immediate констант).
- **Top by get** — Опкоды, во время которых происходит больше всего загрузок из store (напр. LoadGlobal, GetArrayElement, load_value в разных путях).
- **Top by alloc+get** — Суммарная нагрузка на store по опкодам; удобно для поиска самых «горячих» опкодов с точки зрения store при оптимизации.

Профиль ведётся на один run и на один поток (thread-local). Время выполнения (wall/CPU) не включается; для этого используйте внешние средства (напр. `perf`, `cargo flamegraph`).

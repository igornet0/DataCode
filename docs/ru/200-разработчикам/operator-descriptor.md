# Пользовательские инфиксные операторы (`operator_descriptor`)

Нативные модули (dylib) могут объявить дополнительные инфиксные операторы **до** разбора исходника, экспортируя **`operator_descriptor`**: native-функцию **без аргументов**, возвращающую **массив строк**, каждая строка:

`[symbol, name, precedence, associativity]`

- **symbol** (string): токен в исходнике, например `"@"`.
- **name** (string): логическое имя для `opaque_binop` / dispatch VM, например `"matmul"`.
- **precedence** (number): уровень связывания (выше — сильнее; ориентир: `+`/`-` ≈ 50, `*` ≈ 60).
- **associativity**: `"left"` / `"right"` или `0` / `1`.

Пример: зарегистрировать `@` → `matmul` с тем же уровнем, что у умножения.

## Порядок импорта

Хост **предзагружает** каждое имя из `import` / `from … import` в потоке токенов, подгружает matching **native** dylib и сливает `operator_descriptor` в один [`OperatorRegistry`](../../../src/vm/operator_registry.rs). Чистые `.dc`-пакеты (без dylib) пропускаются.

Символ оператора в выражении **без** предварительного import модуля, который его регистрирует, даст ошибку parse time (например незарегистрированный `@`).

## Конфликты

Повторная регистрация того же **symbol** (разные модули или дубли строк) — **ошибка**, без тихого переопределения.

## Рантайм

Арифметика / dispatch плагинов — единый ABI-хук **`opaque_binop(left, right, op_name)`** для `PluginOpaque`; VM не хардкодит имена модулей.

## Интроспекция

После обычного запуска: `import debug`, затем `debug.operators()` — таблица операторов (symbol, name, precedence, associativity, модуль-источник).

## Заметка про opcode

Legacy байткод может использовать `MatMul`; новые emit предпочитают **`BinaryOp`** с именем `"matmul"`. `MatMul` остаётся для совместимости `.dcb`.

## См. также

- [Жизненный цикл native-вызовов](native-call-lifecycle.md)
- [`src/vm/import_scan.rs`](../../../src/vm/import_scan.rs) — preload перед parse

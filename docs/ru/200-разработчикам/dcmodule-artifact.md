# Артефакт модуля DataCode (`.dcmodule`)

Файл **`.dcmodule`** — это **ZIP-архив** с корневым `manifest.json` и одной или несколькими нативными библиотеками (`.so` / `.dylib` / `.dll`) по путям из манифеста. VM не использует Cargo: только читает манифест, проверяет совместимость ABI и делает `dlopen` выбранной библиотеки.

## Канонические символы входа ABI

Динамическая библиотека должна экспортировать те же символы, что и любой другой нативный модуль:

- **`datacode_module_entry`** (предпочтительно) — указатель на `AbiModuleDescriptor`.
- **`datacode_module`** (fallback) — legacy `DatacodeModule` / `register`.

Не используйте другое имя (например `datacode_module_init`) как каноническую точку входа.

## `manifest.json`

Минимальные поля:

| Поле | Значение |
|------|----------|
| `schema_version` | `1` для этого формата |
| `name` | Имя импорта (`import foo` → `foo`) |
| `version` | Строка версии артефакта |
| `abi_version` | `{ "major": N, "minor": M }` — должна проходить `datacode_abi::abi_compatible` с VM |
| `library` | *или* `targets` — относительный путь к библиотеке внутри zip |

Должно быть либо `library` (один путь), либо `targets` (triple → путь). Если оба заданы, хост выбирает библиотеку по текущему target triple; при отсутствии совпадения используется `library`.

JSON Schema: [`schemas/dcmodule-manifest.schema.json`](../../../schemas/dcmodule-manifest.schema.json).

### Пример

```json
{
  "schema_version": 1,
  "name": "mathlib",
  "version": "1.0.0",
  "abi_version": { "major": 1, "minor": 3 },
  "library": "lib/libmathlib.dylib"
}
```

## Структура архива

```
name.dcmodule (zip)
├── manifest.json
└── lib/
    └── libmathlib.dylib
```

## Упаковка

`dpm pack <directory>` создаёт `name.dcmodule` из каталога сборки с `manifest.json` и указанными файлами.

## Разрешение в рантайме

VM ищет `name.dcmodule` там же, где loose `lib<name>.{dylib,so,dll}` (base path скрипта, корни пакетов DPM, текущая директория). Архив распаковывается в пользовательский кэш (по хешу содержимого), манифест проверяется до `dlopen`.

## Реестр / DPM

`dpm install` кладёт пакет в `<env>/packages/<name>/`. Нативный модуль может быть **`name.dcmodule`** рядом с клоном; см. `data_code::dpm::expected_dcmodule_path`.

## `setup.dcmodule` (DPM — не zip)

Другое использование того же расширения: **JSON**-дескриптор в **корне клонированного пакета** только для **DPM** после `git clone`. Это **не** zip и **не** загружается VM.

### Назначение

- Описание нативного модуля (`module_name`, `package_version`).
- Шаги **build** (shell) с фильтрами по ОС.
- **Копирование** артеfactов в корень пакета для loose libs:
  - `<env>/packages/<name>/lib<module>.dylib` / `.so` / `<module>.dll`
  - или `<module>.dcmodule` (zip) через `post_build`.

### Когда DPM запускает

- После **`dpm add`** и **`dpm init`**, если есть `setup.dcmodule`.
- Вручную: **`dpm setup <package_name>`**.
- Отключить: **`DPM_SETUP_AUTO=0`**.

### Минимальная схема (`schema_version`: 1)

| Поле | Значение |
|------|----------|
| `schema_version` | `1` |
| `module_name` | Имя импорта (например `ml`) |
| `package_version` | Опционально |
| `hooks` | `pre_build`, `post_build` — shell в корне пакета |
| `build` | `command`, опционально `cwd`, `when.os` |
| `install` | `from` → `to`, опционально `when` |

### Пример

```json
{
  "schema_version": 1,
  "module_name": "ml",
  "build": [
    { "when": { "os": ["macos"] }, "command": "cargo build --release --features metal" }
  ],
  "install": [
    { "when": { "os": ["macos"] }, "from": "target/release/libml.dylib", "to": "libml.dylib" }
  ]
}
```

## См. также

- [Жизненный цикл native-вызовов](native-call-lifecycle.md)
- [Datacode-abi](https://github.com/igornet0/Datacode-abi/tree/3753be85ad4d9dd73d0756f66754fbc402f573ad)

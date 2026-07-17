# Тип `archive`

← [Типы данных](./README.md)

Объект `archive` представляет открытый архив (ZIP, 7Z, RAR). Создаётся функцией [`archive()`](../функции/пути.md#archivepath).

## Свойства

| Свойство | Тип | Описание |
|----------|-----|----------|
| `path` | `path` | Путь к файлу архива |
| `format` | `string` | Формат: `zip`, `7z`, `rar` |
| `files` | `array` | Список объектов с метаданными файлов |
| `count` | `number` | Количество файлов |
| `size` | `number` | Суммарный размер после распаковки |
| `compressed_size` | `number` | Размер файла архива на диске |

Каждый элемент `files` содержит:

- `path`, `name`, `directory`, `extension`
- `size`, `compressed_size`

## Методы

### `read(path)`

Читает файл из архива с автоопределением типа (как [`read()`](../функции/таблицы.md)):

- `.json` → `object`
- `.csv` → `table`
- `.txt`, `.md` → `string`
- `.png`, `.jpg` → `image`
- `.bin` → bytes

### `read_text(path)`

Принудительно читает файл как UTF-8 строку.

### `extract(dest)`

Полностью распаковывает архив в директорию `dest` (с защитой от zip-slip).

### `close()`

Закрывает архив и освобождает ресурсы.

## Пример

```datacode
zip = archive("./backup.zip")
print(zip.format)
print(zip.count)
for file in zip.files {
    print(file.path)
}
data = zip.read("config/settings.json")
text = zip.read_text("README.md")
zip.extract("./output")
zip.close()
```

## Поддерживаемые форматы

| Формат | Magic bytes | Примечание |
|--------|-------------|------------|
| ZIP | `PK\x03\x04` | Встроенная поддержка |
| 7Z | `7z\xBC\xAF\x27\x1C` | Встроенная поддержка |
| RAR | `Rar!\x1a\x07` | Требует сборки с `--features archive-rar` |

Формат определяется по содержимому файла, а не по расширению.

## `typeof` / `isinstance`

```datacode
typeof(zip)              # "archive"
isinstance(zip, "archive")  # true
```

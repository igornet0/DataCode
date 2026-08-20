# DataCode Web Editor

Веб-редактор DataCode: подсветка, автодополнение с отслеживанием типов, вкладки, поиск.

## Запуск

Нужен сервер из `web_helper/`, чтобы подхватить общий каталог языка:

```bash
cd web_helper
python3 -m http.server 8000
```

Откройте `http://localhost:8000/datacode_web_editor/`.

## Возможности языка в редакторе

- Ключевые слова: `fn`, `cls`, `stream`, `ireturn`, `ereturn`, `import`, `try/catch`, `Abstract`, …
- 117 встроенных функций + алиасы `read` / `read_bin` из [`../datacode_lang.js`](../datacode_lang.js)
- Интерполяция `"${name}"` в подсветке
- Методы после точки: `let s = "hi"` → `s.upper()`, `s.split(`
- Таблицы: `t = table(...)` → `t.select()`, `t.map()`
- Hover по функции, методу и переменной (выведенный тип)
- Сниппеты `if` / `for` / `fn` / `cls` / `try` на `{ }`

## Горячие клавиши

- **Ctrl+F** — поиск
- **Ctrl+Z** / **Ctrl+Y** — undo / redo
- **Tab** — отступ или принятие автодополнения
- **Escape** — закрыть автодополнение

## Файлы

```
web_helper/
├── datacode_lang.js
└── datacode_web_editor/
    ├── index.html
    ├── styles.css
    ├── editor.js
    ├── syntax-highlighter.js
    └── README.md
```

Каталог функций и методов типов общий с веб-документацией. Не дублируйте списки в highlighter.

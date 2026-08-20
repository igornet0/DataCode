# DataCode Web Documentation

Интерактивная веб-документация языка DataCode. Контент соответствует каноническим `docs/ru/` и `examples/ru/`.

## Просмотр

```bash
cd web_helper
python3 -m http.server 8000
```

Откройте `http://localhost:8000/datacode_web_docs/`. Нужен сервер из `web_helper/`, потому что каталог языка лежит в `../datacode_lang.js`.

Либо откройте `index.html` из папки `web_helper/datacode_web_docs/` — тогда тоже должен быть доступен соседний `datacode_lang.js`.

## Структура

```
web_helper/
├── datacode_lang.js          # 117 builtins, методы типов, keywords
└── datacode_web_docs/
    ├── index.html
    ├── styles.css
    ├── script.js
    └── README.md
```

## Разделы

1. Обзор — типы, 117 функций, модули
2. Установка — `./install.sh` / `make install`
3. Быстрый старт
4. Синтаксис — уроки 01–12
5. Типы данных — методы `s.upper()`, `t.select()`, date, set, datasource
6. Функции — карточки из `datacode_lang.js`
7. Модули — встроенные + DPM `ml`
8. Таблицы и JOIN
9. Примеры из `examples/ru/`
10. WebSocket (DCP)
11. Справочник CLI

## Обновление каталога функций

Править [`web_helper/datacode_lang.js`](../datacode_lang.js). Тот же файл использует веб-редактор.

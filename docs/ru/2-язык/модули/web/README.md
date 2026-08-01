# Модуль `web`

Встроенный модуль для HTTP-запросов, автоматизации реального браузера и преобразования веб/HTML-данных в таблицы DataCode.

```datacode
from web import http, browser, data
```

## Пространства имён

| Namespace | Назначение |
|-----------|------------|
| `http` | HTTP-клиент (`get` / `post` / …) → `HttpResponse` |
| `browser` | Запуск Chromium и автоматизация страниц → `WebPage` / `WebElement` |
| `data` | HTML / JSON / страница → таблица или массив записей |

## HTTP

```datacode
from web import http

response = http.get(
    "https://api.example.com/users",
    { params: { page: 1 }, headers: { Authorization: "Bearer TOKEN" }, timeout: 30 }
)

if response.ok {
    print(response.status)
    print(response.json)
}

users = http.get_table("https://api.example.com/users")
```

Методы: `get`, `post`, `put`, `patch`, `delete`, `head`, `options`, `get_table`.

Второй аргумент — объект опций (`params`, `headers`, `body`, `json`, `timeout`) или те же значения позиционно после `url`.

### Свойства `HttpResponse`

| Свойство | Описание |
|----------|----------|
| `status` | Код ответа |
| `status_text` | Текст статуса |
| `ok` | `true`, если статус 2xx |
| `headers` | Объект заголовков |
| `body` | Тело как UTF-8 строка |
| `json` | Разобранный JSON (кэш) |
| `url` | Итоговый URL после редиректов |
| `size` | Размер тела в байтах |

Разрешены только `http://` и `https://`. В режиме `Restricted` запрещён `net.http`.

## Browser

Нужен установленный Chrome/Chromium (CDP, chromiumoxide).

```datacode
from web import browser

page = browser.open("https://example.com", false)
page.click("#login")
page.type("#username", "admin")
page.fill("#password", "secret")
page.wait_for(".dashboard")
print(page.text())
page.screenshot("./dashboard.png")
page.close()
```

`browser.open(url, headless?, user_agent?, headers?, stealth?, options?, profile?)` или объект опций.

### Stealth

Режим против детекта автоматизации (флаги Chromium + JS-патчи + опциональный profile). HTTP-клиент не затрагивается.

```datacode
from web import browser

page = browser.open("https://example.com", stealth=true)

page = browser.open("https://example.com", options={
    stealth: true,
    profile: "./profiles/example"
})
```

| Опция | Описание |
|--------|----------|
| `stealth` | Включить слой `StealthBrowserDriver` |
| `profile` | Каталог профиля Chromium (cookies, localStorage, prefs) |
| `locale` / `timezone` / `viewport` | Настройки окружения |
| `headless` | Без окна (`false` по умолчанию) |

API страницы (`goto`, `click`, `type`, …) не меняется.

Методы страницы: `goto`, `close`, `click`, `type`, `fill`, `clear`, `select`, `text`, `html`, `screenshot`, `wait`, `wait_for`, `wait_for_navigation`, `find`, `find_all`, `set_cookie`, `delete_cookie`, свойство `cookies`.

Элементы (`find` / `find_all`): `click`, `text`, `html`, `type`, `fill`, `clear`, `attr`, свойство `attributes`.

Открытые страницы закрываются при завершении программы. В Restricted запрещён `net.browser`.

## Data

```datacode
from web import browser, data

page = browser.open("https://example.com/products", true)
products = data.extract(page, ".product", {
    name: ".name",
    price: ".price",
    url: { selector: "a", attribute: "href" }
})
table products = data.table(products)
```

- `data.table(source, selector?)` — HTML-таблица, страница/элемент, JSON ответа или массив объектов
- `data.extract(source, schema)` / `data.extract(source, item_selector, schema)`

`typeof`: `"http_response"`, `"web_page"`, `"web_element"`.

## Примеры

- [examples/ru/16-web/](../../../../examples/ru/16-web/)
- EN: [examples/en/16-web/](../../../../examples/en/16-web/)

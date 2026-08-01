# Module `web`

Built-in module for HTTP requests, real browser automation, and converting web/HTML data into DataCode tables.

```datacode
from web import http, browser, data
```

## Namespaces

| Namespace | Role |
|-----------|------|
| `http` | HTTP client (`get` / `post` / …) → `HttpResponse` |
| `browser` | Launch Chromium and automate pages → `WebPage` / `WebElement` |
| `data` | HTML / JSON / page → table or record arrays |

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

# Shortcut: GET + JSON → Table
users = http.get_table("https://api.example.com/users")
```

Methods: `get`, `post`, `put`, `patch`, `delete`, `head`, `options`, `get_table`.

Optional second argument may be an options object with keys `params`, `headers`, `body`, `json`, `timeout`, or the same values as positional arguments after `url` (in that order).

### `HttpResponse` properties

| Property | Description |
|----------|-------------|
| `status` | HTTP status code |
| `status_text` | Reason phrase |
| `ok` | `true` if status is 2xx |
| `headers` | Object of header strings |
| `body` | Response body as UTF-8 string |
| `json` | Parsed JSON value (cached) |
| `url` | Final URL after redirects |
| `size` | Body size in bytes |

Only `http://` and `https://` URLs are allowed. Under `PermissionPolicy::Restricted`, `net.http` is denied.

## Browser

Requires a local Chrome/Chromium installation (CDP via chromiumoxide).

```datacode
from web import browser

page = browser.open("https://example.com", false)  # headless=false by default
page.click("#login")
page.type("#username", "admin")
page.fill("#password", "secret")
page.wait_for(".dashboard")
print(page.text())
page.screenshot("./dashboard.png")
page.close()
```

`browser.open(url, headless?, user_agent?, headers?, stealth?, options?, profile?)` or an options object.

### Stealth mode

Anti-bot / stealth automation (Chromium flags + JS patches + optional persistent profile). HTTP client is unaffected.

```datacode
from web import browser

page = browser.open("https://example.com", stealth=true)

# or
page = browser.open("https://example.com", options={
    stealth: true,
    profile: "./profiles/example",
    locale: "en-US",
    viewport: [1920, 1080]
})

page.wait_for("#content")
page.close()
```

| Option | Description |
|--------|-------------|
| `stealth` | Enable `StealthBrowserDriver` layer |
| `profile` | Chromium user-data dir (cookies, localStorage, prefs) |
| `locale` / `timezone` / `viewport` | Environment fingerprint helpers |
| `headless` | Headless Chromium (`false` by default) |

Stealth is a swappable driver layer (`Standard` → `StealthBrowserDriver` → Chromium). Page API (`goto`, `click`, `type`, …) does not change.

### `WebPage` methods / properties

`goto`, `close`, `click`, `type`, `fill`, `clear`, `select`, `text`, `html`, `screenshot`, `wait`, `wait_for`, `wait_for_navigation`, `find`, `find_all`, `set_cookie`, `delete_cookie`, property `cookies`.

### `WebElement`

Returned by `find` / `find_all`: `click`, `text`, `html`, `type`, `fill`, `clear`, `attr`, property `attributes`.

Open pages are closed automatically when the DataCode run finishes (and on WebSocket client cleanup).

Under Restricted policy, `net.browser` is denied.

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

- `data.table(source, selector?)` — HTML `<table>`, `WebPage`/`WebElement`, `HttpResponse` JSON, or array of objects → `Table`
- `data.extract(source, schema)` — one record (wrapped in an array)
- `data.extract(source, item_selector, schema)` — all matching items

`typeof`: `"http_response"`, `"web_page"`, `"web_element"`.

## Examples

- [examples/en/16-web/](../../../../examples/en/16-web/)
- RU: [examples/ru/16-web/](../../../../examples/ru/16-web/)

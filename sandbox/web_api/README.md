# DataCode Web API

Example and library for building HTTP APIs with DataCode and `datacode-server`.

## Quick start

1. From the DataCode repo root:
   ```bash
   cargo run --bin datacode-server -- sandbox/web_api/app.dc --port 8080
   ```
   Or with the main binary:
   ```bash
   cargo run --bin datacode -- --http sandbox/web_api/app.dc --port 8080
   ```

2. Try the routes:
   - `GET http://127.0.0.1:8080/` → "Hello from DataCode"
   - `GET http://127.0.0.1:8080/health` → 200 OK
   - `GET http://127.0.0.1:8080/api/hello` → "Hello API"
   - `GET http://127.0.0.1:8080/users/42` → "User 42"

## Writing handlers

Use the `@route("METHOD", "/path")` decorator before a function. The function receives one argument `req` (Request) with:

- `req.method` – HTTP method
- `req.path` – request path
- `req.query` – query string
- `req.headers` – object of header names to values
- `req.body` – request body as string
- `req.params` – path params for routes like `/users/{id}` (e.g. `req.params.id`)

Return a string for a plain 200 response, or an object:

- `{"status": 200, "body": "..."}` – status and body
- `{"status": 200, "body": "...", "headers": {"content-type": "application/json"}}` – with headers
- Any object without `status`/`body` is serialized as JSON with 200 OK

## Production with Nginx

Put `datacode-server` behind Nginx as a reverse proxy. Example config:

```nginx
server {
    listen 80;
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    location /static/ {
        root /var/www/app;
    }
}
```

Run `datacode-server app.dc --host 127.0.0.1 --port 8080` and reload Nginx.

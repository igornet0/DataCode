# Nginx reverse proxy for datacode-server

Use Nginx in front of `datacode-server` for production: TLS, gzip, static files.

1. Run datacode-server on a local port (e.g. 8080):
   ```bash
   datacode-server app.dc --host 127.0.0.1 --port 8080
   ```

2. Use `datacode.conf` as a site config (adjust paths as needed):
   ```bash
   sudo cp datacode.conf /etc/nginx/sites-available/datacode
   sudo ln -s /etc/nginx/sites-available/datacode /etc/nginx/sites-enabled/
   sudo nginx -t && sudo systemctl reload nginx
   ```

3. Nginx accepts external traffic and proxies to `http://127.0.0.1:8080`. Static files can be served from `/static/` (set `root` to your app directory).

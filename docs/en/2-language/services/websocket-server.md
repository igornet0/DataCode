# DataCode WebSocket Server

WebSocket server for remote execution of DataCode code.

**📚 Usage examples:**
- WebSocket clients: [`examples/en/07-websocket/`](../../../examples/en/07-websocket/)

## Starting the server

```bash
# Default address (127.0.0.1:8080)
datacode --websocket

# Custom host and port via flags
datacode --websocket --host 0.0.0.0 --port 8899

# Custom address via environment variable
DATACODE_WS_ADDRESS=0.0.0.0:3000 datacode --websocket

# Flags take priority over environment variable
DATACODE_WS_ADDRESS=127.0.0.1:8080 datacode --websocket --host 0.0.0.0 --port 8899
# Result: server starts on 0.0.0.0:8899
```

### Application script (`ws_app.dc`)

Optional setup script (analogous to HTTP `app.dc`). Runs once at startup before accepting connections.

```bash
datacode examples/en/07-websocket/dc/ws_app.dc --websocket --host 0.0.0.0 --port 8899 --use-ve --build_model
datacode --websocket examples/en/07-websocket/dc/ws_app.dc --port 8899
```

Without `ws_app.dc` the server works as before — built-in handlers only.

**Example** [`examples/en/07-websocket/dc/ws_app.dc`](../../../examples/en/07-websocket/dc/ws_app.dc):

```dc
from websocket import configure, disable_builtin

configure({"execute_policy": "restricted"})

@ws_route("ping")
fn ping(req) {
    return {"success": true, "message": "pong"}
}
```

**`websocket` module:**

| Function | Description |
|----------|-------------|
| `websocket.configure({...})` | `execute_policy`: `"allow_all"` (default) or `"restricted"` |
| `websocket.disable_builtin("type")` | Disable built-in message type |
| `websocket.enable_builtin("type")` | Re-enable |

**`@ws_route("type")`** — handler for JSON with `"type": "type"`. Receives request object, returns response object. Overrides built-in type with the same name.

CLI flags (`--use-ve`, `--build_model`, `--host`, `--port`) are preserved; `ws_app.dc` adds routes and policy.

## Protocol

### Connection

Connect to the WebSocket server at `ws://127.0.0.1:8080` (or the configured address).

### Request format

The WebSocket server supports several request types. All requests must include a `type` field specifying the operation.

#### Code execution

Send a JSON message with type `execute` and field `code`:

```json
{
  "type": "execute",
  "code": "print('Hello, World!')"
}
```

**Backward compatibility:** The old format without `type` is also supported:

```json
{
  "code": "print('Hello, World!')"
}
```

#### SMB share connection

To connect to an SMB (Samba/CIFS) share, use type `smb_connect`:

```json
{
  "type": "smb_connect",
  "ip": "192.168.1.100",
  "login": "username",
  "password": "password",
  "domain": "WORKGROUP",
  "share_name": "share_name"
}
```

**Parameters:**
- `ip` - IP address or hostname of SMB server
- `login` - username
- `password` - password
- `domain` - domain (usually `WORKGROUP` or domain name; may be empty string)
- `share_name` - SMB share name

**Response:**
```json
{
  "success": true,
  "message": "Successfully connected to SMB share 'share_name'",
  "error": null
}
```

### Response format

The server returns JSON with execution result:

**Success:**
```json
{
  "success": true,
  "output": "Hello, World!\n",
  "error": null
}
```

**Execution error:**
```json
{
  "success": false,
  "output": "",
  "error": "Error: variable 'x' is not defined"
}
```

## Usage examples

### JavaScript/Node.js

```javascript
const WebSocket = require('ws');

const ws = new WebSocket('ws://127.0.0.1:8080');

ws.on('open', function open() {
    const request = {
        type: "execute",
        code: "print('Hello from WebSocket!')"
    };
    ws.send(JSON.stringify(request));
});

ws.on('message', function message(data) {
    const response = JSON.parse(data);
    console.log('Output:', response.output);
    if (response.error) {
        console.error('Error:', response.error);
    }
});
```

### Python

```python
import asyncio
import websockets
import json

async def execute_code():
    uri = "ws://127.0.0.1:8080"
    async with websockets.connect(uri) as websocket:
        request = {
            "type": "execute",
            "code": "print('Hello from Python!')"
        }
        await websocket.send(json.dumps(request))
        response = json.loads(await websocket.recv())
        print("Output:", response["output"])
        if response["error"]:
            print("Error:", response["error"])

asyncio.run(execute_code())
```

### cURL (via wscat)

```bash
# Install wscat: npm install -g wscat
wscat -c ws://127.0.0.1:8080
# Then send:
{"type": "execute", "code": "print('Hello!')"}
```

## SMB share connection

The WebSocket server supports connecting to SMB (Samba/CIFS) shares for working with files on remote servers.

### Requirements

**Linux/Mac:**
```bash
brew install samba  # macOS
# or
sudo apt-get install samba-client  # Ubuntu/Debian
```

**Windows:** SMB client is built into the system.

### Using the `lib://` protocol

After a successful `smb_connect` request, you can use the special `lib://` protocol in DataCode scripts:

```
lib://share_name/path/to/file
```

Where `share_name` is the connected SMB share name and `path/to/file` is the path on the share.

### SMB example

```python
import asyncio
import websockets
import json

async def smb_example():
    async with websockets.connect("ws://localhost:8899") as websocket:
        # 1. Connect to SMB
        connect_request = {
            "type": "smb_connect",
            "ip": "192.168.1.100",
            "login": "user",
            "password": "pass",
            "domain": "WORKGROUP",
            "share_name": "data"
        }
        await websocket.send(json.dumps(connect_request))
        response = json.loads(await websocket.recv())
        print("SMB Connect:", response)
        
        # 2. Run DataCode script using SMB
        code = """
        files = list_files(path("lib://data/reports"))
        for file in files {
            print("File:", file)
        }
        """
        
        execute_request = {
            "type": "execute",
            "code": code
        }
        await websocket.send(json.dumps(execute_request))
        response = json.loads(await websocket.recv())
        print("Execute:", response)

asyncio.run(smb_example())

```

### Supported operations

After connecting to an SMB share, these DataCode operations are available:

- **list_files(path("lib://share_name/dir"))** - list files (recursively walks subdirectories)
- **list_files(path("lib://share_name/dir"), regex="*.csv")** - filtered list (glob like `*.csv` or regular expressions)
- **read(path("lib://share_name/file.csv"))** - read file (CSV, XLSX, TXT supported)

See [`examples/en/07-websocket/README.md`](../../../examples/en/07-websocket/README.md) for details.

## Features

1. **Session isolation**: Each client gets its own interpreter. Variables and functions defined by one client are not visible to others.

2. **Output capture**: All `print()` calls are captured and sent to the client in the `output` field.

3. **Error handling**: Execution errors are returned in the `error` field with `success` set to `false`.

4. **Multi-line code**: Multi-line code execution is supported:

```json
{
  "type": "execute",
  "code": "global x = 10\nglobal y = 20\nprint('Sum:', x + y)"
}
```

5. **SMB connections**: Each client has its own SMB connections, closed automatically on disconnect.

## Web client

Open `examples/en/07-websocket/html/websocket_client_example.html` in a browser for interactive WebSocket server testing.

## Security

⚠️ **Warning**: The current implementation does not include authentication or access restrictions. Do not use on public servers without additional protection!

## Limitations

- The interpreter is not thread-safe (`Send`), so each client is handled in a separate local task
- Variables and functions are not persisted between requests from one client (each request runs in the same interpreter, but state may change)

---

**See also:**
- [WebSocket examples](../../../examples/en/07-websocket/) - practical usage
- [Path functions](../functions/paths.md) — working with files and paths

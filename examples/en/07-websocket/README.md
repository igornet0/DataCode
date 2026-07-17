# WebSocket Examples for DataCode

This folder contains examples for using the DataCode WebSocket server.

## Files

- **test_websocket.js** - Node.js client for testing WebSocket server
- **test_websocket.py** - Python client for testing WebSocket server  
- **test_websocket.sh** - Bash script for testing (requires websocat)
- **websocket_client_example.html** - HTML client with web interface
- **websocket_requests.json** - JSON file with request examples
- **websocket_test_requests.md** - WebSocket testing documentation
- **test_smb_connection.py** - Python client for testing SMB connection via WebSocket
- **test_smb_load_data.dc** - Example DataCode script for working with files on SMB share
- **test_file_upload.py** - Python client for testing file uploads via WebSocket (requires --use-ve mode)
- **ws_app.dc** - WebSocket app script: `@ws_route` handlers and `websocket.configure()`

## Starting the Server

**Standard mode:**
```bash
datacode --websocket --host 0.0.0.0 --port 8899
```

**With app script (`ws_app.dc`):**
```bash
datacode ws_app.dc --websocket --host 0.0.0.0 --port 8899 --use-ve --build_model
```

Custom routes: send `{"type": "ping"}` → `{"success": true, "message": "pong"}`.

**Virtual environment mode (for file uploads):**
```bash
datacode --websocket --host 0.0.0.0 --port 8899 --use-ve
```

The `--use-ve` flag enables:
- Isolated session folders in `src/temp_sessions`
- `getcwd()` returns empty string
- Support for file uploads via WebSocket
- Automatic cleanup of session folders on disconnect

## Usage

### Node.js
```bash
cd examples/en/08-websocket
node test_websocket.js
```

### Python
```bash
cd examples/en/08-websocket
pip install websockets
python3 test_websocket.py
```

### File Upload (Python, requires --use-ve)
```bash
cd examples/en/08-websocket
pip install websockets
# Make sure server is running with --use-ve flag
python3 test_file_upload.py
```

### Bash
```bash
cd examples/en/08-websocket
cargo install websocat  # if not already installed
bash test_websocket.sh
```

### HTML Client
Open `websocket_client_example.html` in a browser.

## Request Format

All requests are sent in JSON format:

```json
{
  "code": "print('Hello, World!')"
}
```

## Response Format

```json
{
  "success": true,
  "output": "Hello, World!\n",
  "error": null
}
```

## Code Examples

### Simple Output
```json
{"code": "print('Hello, World!')"}
```

### Variables
```json
{"code": "global x = 10\nglobal y = 20\nprint('Sum:', x + y)"}
```

### Loop
```json
{"code": "for i in [1, 2, 3] {\n    print('Number:', i)\n}"}
```

### Function
```json
{"code": "fn greet(name) {\n    return 'Hello, ' + name + '!'\n}\nprint(greet('DataCode'))"}
```

## Connecting to SMB Share

The WebSocket server supports connecting to SMB (Samba/CIFS) shares for working with files on remote servers.

### Requirements

**For Linux/Mac:**
```bash
# Install smbclient (Samba)
brew install samba  # macOS
# or
sudo apt-get install samba-client  # Ubuntu/Debian
```

**For Windows:**
SMB client is built into the system, no additional installation required.

### Connecting to SMB Share

To connect to an SMB share, send a request with type `smb_connect`:

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
- `ip` - IP address or name of SMB server
- `login` - username for connection
- `password` - user password
- `domain` - domain (usually `WORKGROUP` for Windows or domain name, can be empty string)
- `share_name` - name of SMB share to connect to

**Server Response:**
```json
{
  "success": true,
  "message": "Successfully connected to SMB share 'share_name'",
  "error": null
}
```

On error:
```json
{
  "success": false,
  "message": "",
  "error": "SMB connection error: ..."
}
```

### Using lib:// Protocol in DataCode

After successfully connecting to an SMB share, you can use the special `lib://` protocol in DataCode scripts to access files on the share.

**Path Format:**
```
lib://share_name/path/to/file
```

Where:
- `share_name` - name of connected SMB share
- `path/to/file` - path to file or directory on the share

### Usage Example

#### 1. Connection and Script Execution (Python)

Use `test_smb_connection.py` to connect to SMB and execute a DataCode script:

```bash
cd examples/en/08-websocket
python3 test_smb_connection.py test_smb_load_data.dc
```

The script automatically:
1. Connects to WebSocket server
2. Sends request to connect to SMB share
3. Executes DataCode script from specified file

**Configuring Connection Parameters:**

Edit variables at the beginning of `test_smb_connection.py`:
```python
username = "your_username"
password = "your_password"
smb_server = "192.168.1.100"  # IP or server name
smb_share = "share_name"      # SMB share name
domain = "WORKGROUP"          # Domain (can be empty string)
```

#### 2. Example DataCode Script for SMB

Example from `test_smb_load_data.dc`:

```datacode
let path = path("lib://Stream/my_dir")

for path_dir in list_files(path) {
    for file in list_files(path / path_dir) {
        if contains(file, 'data') {
            print(file)
            global data = read(path / path_dir / file, 11, ' data')
            print(len(data))
            table_info(data)
        }
    }
}
```

**What the script does:**
1. Creates path to directory on SMB share: `lib://Stream/my_dir`
2. Iterates over all files in directory
3. For each subdirectory, iterates over files
4. If filename contains 'data', reads it using `read`
5. Outputs information about loaded table

### Supported SMB Operations

After connecting to SMB share via WebSocket, the following operations are available in DataCode scripts:

#### list_files
Get list of files in directory on SMB share:
```datacode
let files = list_files(path("lib://share_name/directory"))
```

#### read
Read file from SMB share:
```datacode
let data = read(path("lib://share_name/path/to/file.csv"))
```

Supported file formats:
- **CSV** - automatically parsed into table
- **XLSX** - automatically parsed into table (with sheet support)
- **TXT** - read as text string

**read parameters for CSV/XLSX:**
```datacode
# CSV with header row specification
read(path("lib://share/file.csv"), 0)

# XLSX with sheet specification
read(path("lib://share/file.xlsx"), 0, "Sheet1")
```

### Working with Paths

Paths on SMB share can be combined using the `/` operator:

```datacode
let base_path = path("lib://share_name")
let file_path = base_path / "subdirectory" / "file.csv"
let data = read(file_path)
```

### Important Notes

1. **Connection persists for session** - SMB connection is active until client disconnects from WebSocket server
2. **Separate connection for each client** - each WebSocket client has its own set of SMB connections
3. **Security** - passwords are transmitted in plain text in JSON requests, use secure connection (WSS) in production
4. **Performance** - SMB operations are slower than local file system, consider this when working with large files

### Complete SMB Example

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
        response = await websocket.recv()
        print("SMB Connect:", json.loads(response))
        
        # 2. Execute DataCode script
        code = """
        let files = list_files(path("lib://data/reports"))
        for file in files {
            print("File:", file)
        }
        """
        
        execute_request = {
            "type": "execute",
            "code": code
        }
        await websocket.send(json.dumps(execute_request))
        response = await websocket.recv()
        print("Execute:", json.loads(response))

asyncio.run(smb_example())
```

## File Upload via WebSocket

The WebSocket server supports file uploads when started with the `--use-ve` flag. Each client connection gets an isolated session folder in `src/temp_sessions`.

### Requirements

1. Start server with `--use-ve` flag:
```bash
datacode --websocket --host 0.0.0.0 --port 8899 --use-ve
```

2. Install Python dependencies:
```bash
pip install websockets
```

### Uploading Files

#### Text Files

```python
import asyncio
import websockets
import json

async def upload_text_file():
    async with websockets.connect("ws://localhost:8899") as websocket:
        upload_request = {
            "type": "upload_file",
            "filename": "test.txt",
            "content": "Hello, DataCode!\nThis is a test file."
        }
        await websocket.send(json.dumps(upload_request))
        response = await websocket.recv()
        result = json.loads(response)
        print(result)

asyncio.run(upload_text_file())
```

#### Binary Files (Base64)

```python
import asyncio
import websockets
import json
import base64

async def upload_binary_file():
    async with websockets.connect("ws://localhost:8899") as websocket:
        # Read binary file and encode to base64
        with open("image.png", "rb") as f:
            binary_data = f.read()
        base64_data = base64.b64encode(binary_data).decode('utf-8')
        
        upload_request = {
            "type": "upload_file",
            "filename": "image.png",
            "content": f"base64:{base64_data}"
        }
        await websocket.send(json.dumps(upload_request))
        response = await websocket.recv()
        result = json.loads(response)
        print(result)

asyncio.run(upload_binary_file())
```

#### Files in Subdirectories

```python
upload_request = {
    "type": "upload_file",
    "filename": "subdir/nested_file.txt",
    "content": "Nested file content"
}
```

### Server Response

**Success:**
```json
{
  "success": true,
  "message": "File test.txt successfully uploaded",
  "error": null
}
```

**Error:**
```json
{
  "success": false,
  "message": "",
  "error": "File write error: ..."
}
```

### Working with Uploaded Files

After uploading files, you can work with them in DataCode scripts:

```python
# Upload CSV file
upload_csv = {
    "type": "upload_file",
    "filename": "data.csv",
    "content": "name,age\nAlice,30\nBob,25"
}
await websocket.send(json.dumps(upload_csv))

# Read and process CSV
code = """
global data = read(path("data.csv"), 0)
print("Loaded rows:", len(data))
table_info(data)
"""
execute_request = {
    "type": "execute",
    "code": code
}
await websocket.send(json.dumps(execute_request))
```

### Important Notes

1. **Session isolation** - Each WebSocket connection gets its own session folder
2. **Automatic cleanup** - Session folders are automatically deleted on client disconnect
3. **getcwd() behavior** - In `--use-ve` mode `getcwd()` returns empty string
4. **File paths** - Use relative paths in DataCode scripts (e.g., `path("data.csv")`)
5. **Base64 encoding** - Binary files must have `base64:` prefix in content field

### Complete Example

See `test_file_upload.py` for a complete example with multiple test cases:
```bash
python3 test_file_upload.py
```


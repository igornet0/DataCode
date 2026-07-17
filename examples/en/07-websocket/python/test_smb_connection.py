import asyncio
import websockets
import json
import sys
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DC = SCRIPT_DIR.parent / "dc" / "test_smb_load_data.dc"

username = "username"
password = "password"

smb_server = "smb_server"
smb_share = "smb_share"
domain = ""  # Usually WORKGROUP for Windows or domain name

ws_url = os.environ.get("DATACODE_WS_URL", "ws://127.0.0.1:8899")

async def connect_and_execute(datacode):
    """Connect to WebSocket and run SMB connection plus DataCode script"""
    try:
        print(f"🔌 Connecting to WebSocket server: {ws_url}")
        async with websockets.connect(ws_url) as websocket:
            print("✅ Connected to WebSocket server")
            
            # 1. Connect to SMB share
            print(f"\n📡 Sending SMB share connection request '{smb_share}'...")
            smb_connect_request = {
                "type": "smb_connect",
                "ip": smb_server,
                "login": username,
                "password": password,
                "domain": domain,
                "share_name": smb_share
            }
            
            await websocket.send(json.dumps(smb_connect_request))
            print(f"📤 Sent request: {json.dumps(smb_connect_request, indent=2)}")
            
            response = await websocket.recv()
            smb_response = json.loads(response)
            print(f"\n📥 Server response:")
            print(json.dumps(smb_response, indent=2, ensure_ascii=False))
            
            if smb_response.get("success"):
                print(f"✅ Successfully connected to SMB share '{smb_share}'")
            else:
                error = smb_response.get("error", "Unknown error")
                print(f"❌ Connection error: {error}")
                return
            
            # 2. Execute DataCode script
            print(f"\n📡 Executing DataCode script...")
            print(f"📝 Code:\n{datacode}\n")
            
            execute_request = {
                "type": "execute",
                "code": datacode
            }
            
            await websocket.send(json.dumps(execute_request))
            print("📤 Sent code execution request")
            
            response = await websocket.recv()
            execute_response = json.loads(response)
            print(f"\n📥 Execution result:")
            print(json.dumps(execute_response, indent=2, ensure_ascii=False))
            
            if execute_response.get("success"):
                print(f"\n✅ Code executed successfully")
                if execute_response.get("output"):
                    print(f"\n📋 Output:\n{execute_response['output']}")
            else:
                error = execute_response.get("error", "Unknown error")
                print(f"\n❌ Execution error: {error}")
                if execute_response.get("output"):
                    print(f"📋 Output:\n{execute_response['output']}")
                    
    except ConnectionRefusedError:
        print(f"❌ Could not connect to {ws_url}")
        print("💡 Make sure the WebSocket server is running")
        print("💡 Start server: datacode --websocket --host 0.0.0.0 --port 8899")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        dc_file = str(DEFAULT_DC)
        print(f"💡 No .dc file specified, using default: {dc_file}")
    else:
        dc_file = sys.argv[1]
    
    if not dc_file.endswith('.dc'):
        print(f"❌ Error: file must have .dc extension")
        print(f"💡 Got file: {dc_file}")
        sys.exit(1)
    
    if not os.path.exists(dc_file):
        print(f"❌ Error: file not found: {dc_file}")
        sys.exit(1)
    
    try:
        with open(dc_file, 'r', encoding='utf-8') as f:
            datacode = f.read()
        print(f"📄 Loaded file: {dc_file}")
        print(f"📏 Code size: {len(datacode)} characters\n")
    except Exception as e:
        print(f"❌ Error reading file {dc_file}: {e}")
        sys.exit(1)
    
    print("🚀 Starting SMB connection test via WebSocket\n")
    asyncio.run(connect_and_execute(datacode))

#!/usr/bin/env python3
"""
Test script for uploading files via the DataCode WebSocket server
Requires: pip install websockets

Important: Server must be started with --use-ve:
    datacode --websocket --host 0.0.0.0 --port 8899 --use-ve
"""

import asyncio
import websockets
import json
import base64
import os
from pathlib import Path

async def test_file_upload():
    uri = "ws://127.0.0.1:8899"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to server")
            print("💡 Make sure the server is started with --use-ve")
            print()
            
            # Test 1: Check getcwd() - should return empty string for security
            print("📋 Test 1: Check getcwd() (should return empty string for security)")
            test1 = {
                "type": "execute",
                "code": "global cwd = getcwd()\nprint('Current directory:', cwd)\nprint('Type of cwd:', typeof(cwd))"
            }
            print(f"📤 Sending: {json.dumps(test1, ensure_ascii=False)}")
            await websocket.send(json.dumps(test1))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"📥 Response:")
            print(f"  Success: {result['success']}")
            print(f"  Output: {result['output']}")
            if result.get('error'):
                print(f"  Error: {result['error']}")
            print()
            
            # Test 2: Upload text file
            print("📋 Test 2: Upload text file")
            text_content = """Hello, DataCode!
This is a test file uploaded via WebSocket.
Line 3 of the file.
"""
            upload_text = {
                "type": "upload_file",
                "filename": "test.txt",
                "content": text_content
            }
            print(f"📤 Sending file: test.txt ({len(text_content)} bytes)")
            await websocket.send(json.dumps(upload_text))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"📥 Response:")
            print(f"  Success: {result['success']}")
            print(f"  Message: {result.get('message', '')}")
            if result.get('error'):
                print(f"  Error: {result['error']}")
            print()
            
            # Test 3: Upload CSV file
            print("📋 Test 3: Upload CSV file")
            csv_content = """name,age,city
Alice,30,New York
Bob,25,London
Charlie,35,Paris
"""
            upload_csv = {
                "type": "upload_file",
                "filename": "data.csv",
                "content": csv_content
            }
            print(f"📤 Sending file: data.csv ({len(csv_content)} bytes)")
            await websocket.send(json.dumps(upload_csv))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"📥 Response:")
            print(f"  Success: {result['success']}")
            print(f"  Message: {result.get('message', '')}")
            if result.get('error'):
                print(f"  Error: {result['error']}")
            print()
            
            # Test 4: Upload file in subdirectory
            print("📋 Test 4: Upload file in subdirectory")
            subdir_content = "This file is in a subdirectory\n"
            upload_subdir = {
                "type": "upload_file",
                "filename": "subdir/nested_file.txt",
                "content": subdir_content
            }
            print(f"📤 Sending file: subdir/nested_file.txt")
            await websocket.send(json.dumps(upload_subdir))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"📥 Response:")
            print(f"  Success: {result['success']}")
            print(f"  Message: {result.get('message', '')}")
            if result.get('error'):
                print(f"  Error: {result['error']}")
            print()
            
            # Test 5: Upload binary file (base64)
            print("📋 Test 5: Upload binary file (base64)")
            # Create a simple PNG file (1x1 pixel, transparent)
            png_data = base64.b64encode(
                bytes.fromhex('89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000a49444154789c6300010000000500010d0a2db40000000049454e44ae426082')
            ).decode('utf-8')
            
            upload_binary = {
                "type": "upload_file",
                "filename": "image.png",
                "content": f"base64:{png_data}"
            }
            print(f"📤 Sending file: image.png (base64, {len(png_data)} characters)")
            await websocket.send(json.dumps(upload_binary))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"📥 Response:")
            print(f"  Success: {result['success']}")
            print(f"  Message: {result.get('message', '')}")
            if result.get('error'):
                print(f"  Error: {result['error']}")
            print()
            
            # Test 6: Read uploaded CSV via DataCode
            print("📋 Test 6: Read uploaded CSV via DataCode")
            read_csv_code = """
# Since getcwd() returns empty string, use relative paths
# Files are uploaded to the user session folder

# Basic file read
global data = read(path("data.csv"), header_row=0)
print("Loaded rows:", len(data))
table_info(data)

# Read with column filter via header (array)
global data_filtered = read(path("data.csv"), header=["Name", "Age", "City"])
print("Loaded rows with filter:", len(data_filtered))
print("Columns:", data_filtered.columns)

# Read with column rename via header (dict)
global data_renamed = read(path("data.csv"), header_row=0, header={"Name": "FullName", "Age": null, "City": null, "Salary": null})
print("Loaded rows with rename:", len(data_renamed))
print("Columns:", data_renamed.columns)
"""
            read_csv = {
                "type": "execute",
                "code": read_csv_code
            }
            print(f"📤 Executing code to read CSV")
            await websocket.send(json.dumps(read_csv))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"📥 Response:")
            print(f"  Success: {result['success']}")
            print(f"  Output: {result['output']}")
            if result.get('error'):
                print(f"  Error: {result['error']}")
            print()
            
            # Test 7: Work with multiple files
            print("📋 Test 7: Work with multiple uploaded files")
            multi_file_code = """
# Read text file
global text = read(path("test.txt"))
print("Content of test.txt:")
print(text)

# Read CSV file
global csv_data = read(path("data.csv"))
print("Row count in CSV:", len(csv_data))
"""
            multi_file = {
                "type": "execute",
                "code": multi_file_code
            }
            print(f"📤 Executing code for multiple files")
            await websocket.send(json.dumps(multi_file))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"📥 Response:")
            print(f"  Success: {result['success']}")
            print(f"  Output: {result['output']}")
            if result.get('error'):
                print(f"  Error: {result['error']}")
            print()
            
            # Test 8: Upload folder with mixed data types and iterate via loop
            print("📋 Test 8: Upload folder with mixed data types")
            data_dir = "data_dir"
            source_data_dir = Path(__file__).resolve().parent.parent / "data"
            
            # Upload files from data folder
            print(f"📤 Uploading files from {source_data_dir} to folder {data_dir}/...")
            
            if not source_data_dir.exists():
                print(f"  ⚠️  Folder {source_data_dir} not found")
            else:
                # List all files in data folder
                files_to_upload = []
                for file_path in source_data_dir.iterdir():
                    if file_path.is_file():
                        # Keep relative path for server upload
                        target_filename = f"{data_dir}/{file_path.name}"
                        files_to_upload.append((target_filename, file_path))
                
                for target_filename, file_path in files_to_upload:
                    try:
                        # Use upload_file_from_disk to prepare request
                        upload_req = upload_file_from_disk(websocket, str(file_path), target_filename)
                        
                        await websocket.send(json.dumps(upload_req))
                        response = await websocket.recv()
                        result = json.loads(response)
                        if result.get('success'):
                            print(f"  ✅ {target_filename}")
                        else:
                            print(f"  ❌ {target_filename}: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        print(f"  ❌ {target_filename}: Upload error - {e}")
            
            print()
            
            # Test 9: Looping through files in a folder (with getcwd())
            print("📋 Test 9: Iterate folder files via list_files (with getcwd())")
            list_files_code = f"""
# In --use-ve mode getcwd() returns empty string for security
# Relative paths resolve relative to session folder
global current_dir = getcwd()
print("Current directory (getcwd()): '", current_dir, "'")

# Use relative path — resolves relative to session folder
global dir_path = path("{data_dir}")
print("Folder path (relative):", dir_path)

global files = list_files(dir_path)

print("\\nFiles in folder """ + data_dir + """:")
for file in files {
    print("  -", file, file.parent, file.parent.parent)
}

print("\\nTotal files:", len(files))
"""
            list_files_request = {
                "type": "execute",
                "code": list_files_code
            }
            print(f"📤 Executing code to list files")
            await websocket.send(json.dumps(list_files_request))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"📥 Response:")
            print(f"  Success: {result['success']}")
            print(f"  Output: {result['output']}")
            if result.get('error'):
                print(f"  Error: {result['error']}")
            print()
            
            # Test 10: Process mixed file types
            print("📋 Test 10: Process mixed file types from folder")
            process_files_code = f"""
# Use relative path — resolves relative to session folder
global dir_path = path("{data_dir}")
global files = list_files(dir_path)
""" + """
print("Processing files:")
for file in files {
    print("File:", file)

    if !file.is_file {
        continue
    }
    
    # Detect file type by extension
    if file.extension == "txt" {
        global content = read(file)
        print("  Type: Text file")
        print("  Content:", content)
    }
    
    if file.extension == "csv" {
        global csv_data = read(file)
        print("  Type: CSV file")
        print("  Rows:", len(csv_data))
        if len(csv_data) > 0 {
            print("  First row:", csv_data.idx[0])
        }
    }
    
    if file.extension == "xlsx" {
        print("  Type: Excel file ", file)
        global xlsx_data = read(file)
        print("  Rows:", len(xlsx_data))
        if len(xlsx_data) > 0 {
            print("  First row:", xlsx_data.idx[0])
        }
    }
    
    if file.extension == "zip" {
        print("  Type: ZIP archive")
        print("  (Binary files uploaded successfully)")
    }
}
"""
            process_files_request = {
                "type": "execute",
                "code": process_files_code
            }
            print(f"📤 Executing code to process mixed file types")
            await websocket.send(json.dumps(process_files_request))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"📥 Response:")
            print(f"  Success: {result['success']}")
            print(f"  Output: {result['output']}")
            if result.get('error'):
                print(f"  Error: {result['error']}")
            print()

            print("📋 Test 11: Check file list in folder")
            list_files_code = """
            print('Files in folder getcwd():')
            for file in list_files(getcwd()) {
                print("  -", file)
            }

            print()
            print('Files in folder ".":')

            for file in list_files(".") {
                print("  -", file)
            }

            try {
                print("Files in folder '..' (should error):")
                for file in list_files("..") {
                    print("  -", file)
                }
            } catch e {
                print("Error: ", e)
                print("Should error")
            }

            try {
                print("Files in folder '../' (should error):")
                for file in list_files("../") {
                    print("  -", file)
                }
            } catch e {
                print("Error: ", e)
                print("Should error")
            }

            try {

                print("Files in folder '../getcwd()' (should error):")
                for file in list_files(".." / getcwd()) {
                    print("  -", file)
                }

            } catch e {
                print("Error: ", e)
                print("Should error")
            }

            try {
                print("Files in folder '../..' (should error):")
                for file in list_files("../..") {
                    print("  -", file)
                }

            } catch e { 
                print("Error: ", e)
                print("Should error")
            }

            try {
                print("Read file with nonexistent path (should error):")
                data = read(path("nonexistent.txt"))
                print("  -", data)
                
            } catch e {
                print("Error: ", e)
                print("Should error")
            }

            """
            list_files_request = {
                "type": "execute",
                "code": list_files_code
            }
            print(f"📤 Executing code to check file list")
            await websocket.send(json.dumps(list_files_request))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"📥 Response:")
            print(f"  Success: {result['success']}")
            print(f"  Output: {result['output']}")
            if result.get('error'):
                print(f"  Error: {result['error']}")
            print()
            
            print("✅ All tests completed")
            print("💡 Session folder is deleted automatically on disconnect")
            
    except websockets.exceptions.ConnectionRefused:
        print("❌ Error: Could not connect to server")
        print("💡 Make sure the server is started with --use-ve:")
        print("   datacode --websocket --host 0.0.0.0 --port 8899 --use-ve")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def upload_file_from_disk(websocket, file_path, target_filename=None):
    """
    Helper to upload a file from disk
    
    Args:
        websocket: WebSocket connection
        file_path: Path to file on disk
        target_filename: Server filename (if None, uses source filename)
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    filename = target_filename or path.name
    
    # Detect whether file is text or binary
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Text file — send as-is
        upload_request = {
            "type": "upload_file",
            "filename": filename,
            "content": content
        }
    except UnicodeDecodeError:
        # Binary file — encode as base64
        with open(path, 'rb') as f:
            binary_data = f.read()
        base64_data = base64.b64encode(binary_data).decode('utf-8')
        upload_request = {
            "type": "upload_file",
            "filename": filename,
            "content": f"base64:{base64_data}"
        }
    
    return upload_request

async def upload_local_file_example():
    """
    Example of uploading a local file from disk
    """
    uri = "ws://127.0.0.1:8899"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to server")
            print()
            
            # Example: upload file from current directory
            # Replace with your file path
            local_file = "example.txt"
            
            if os.path.exists(local_file):
                print(f"📤 Uploading local file: {local_file}")
                upload_request = upload_file_from_disk(websocket, local_file)
                
                await websocket.send(json.dumps(upload_request))
                response = await websocket.recv()
                result = json.loads(response)
                
                print(f"📥 Response:")
                print(f"  Success: {result['success']}")
                print(f"  Message: {result.get('message', '')}")
                if result.get('error'):
                    print(f"  Error: {result['error']}")
            else:
                print(f"⚠️  File {local_file} not found")
                print("💡 Create example.txt for testing")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--upload-local":
        # Local file upload mode
        asyncio.run(upload_local_file_example())
    else:
        # Normal test mode
        asyncio.run(test_file_upload())


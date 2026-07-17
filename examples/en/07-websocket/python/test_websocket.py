#!/usr/bin/env python3
"""
Test script for the DataCode WebSocket server
Requires: pip install websockets
"""

import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://127.0.0.1:8899"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to server")
            
            # Test 1: Simple print
            test1 = {
                "type": "execute",
                "code": "print('Hello, World!')"
            }
            print(f"\n📤 Sending test 1: {json.dumps(test1)}")
            await websocket.send(json.dumps(test1))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"\n📥 Response:")
            print(f"  Success: {result['success']}")
            print(f"  Output: {result['output']}")
            if result.get('error'):
                print(f"  Error: {result['error']}")
            
            # Test 2: Variables
            test2 = {
                "type": "execute",
                "code": "global x = 10\nglobal y = 20\nprint('Sum:', x + y)"
            }
            print(f"\n📤 Sending test 2: {json.dumps(test2)}")
            await websocket.send(json.dumps(test2))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"\n📥 Response:")
            print(f"  Success: {result['success']}")
            print(f"  Output: {result['output']}")
            
            # Test 3: Loop
            test3 = {
                "type": "execute",
                "code": "for i in [1, 2, 3] {\n    print('Number:', i)\n}"
            }
            print(f"\n📤 Sending test 3: {json.dumps(test3)}")
            await websocket.send(json.dumps(test3))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"\n📥 Response:")
            print(f"  Success: {result['success']}")
            print(f"  Output: {result['output']}")
            
            # Test 4: Function
            test4 = {
                "type": "execute",
                "code": "fn greet(name) {\n    return 'Hello, ' + name + '!'\n}\nprint(greet('DataCode'))"
            }
            print(f"\n📤 Sending test 4: {json.dumps(test4)}")
            await websocket.send(json.dumps(test4))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"\n📥 Response:")
            print(f"  Success: {result['success']}")
            print(f"  Output: {result['output']}")
            
            # Test 5: Error (error handling check)
            test5 = {
                "type": "execute",
                "code": "print(undefined_variable)"
            }
            print(f"\n📤 Sending test 5 (expect error): {json.dumps(test5)}")
            await websocket.send(json.dumps(test5))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"\n📥 Response:")
            print(f"  Success: {result['success']}")
            print(f"  Output: {result['output']}")
            if result.get('error'):
                print(f"  Error: {result['error']}")
            
            print("\n✅ All tests completed")
            
    except websockets.exceptions.ConnectionRefused:
        print("❌ Error: Could not connect to server")
        print("💡 Make sure the server is running: datacode --websocket --host 0.0.0.0 --port 8899")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())

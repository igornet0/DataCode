#!/usr/bin/env python3
"""
Тестовый скрипт для WebSocket сервера DataCode
Требуется: pip install websockets
"""

import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://127.0.0.1:8899"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Подключено к серверу")
            
            # Тест 1: Простой вывод
            test1 = {
                "type": "execute",
                "code": "print('Hello, World!')"
            }
            print(f"\n📤 Отправка теста 1: {json.dumps(test1)}")
            await websocket.send(json.dumps(test1))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"\n📥 Получен ответ:")
            print(f"  Success: {result['success']}")
            print(f"  Output: {result['output']}")
            if result.get('error'):
                print(f"  Error: {result['error']}")
            
            # Тест 2: Переменные
            test2 = {
                "type": "execute",
                "code": "global x = 10\nglobal y = 20\nprint('Sum:', x + y)"
            }
            print(f"\n📤 Отправка теста 2: {json.dumps(test2)}")
            await websocket.send(json.dumps(test2))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"\n📥 Получен ответ:")
            print(f"  Success: {result['success']}")
            print(f"  Output: {result['output']}")
            
            # Тест 3: Цикл
            test3 = {
                "type": "execute",
                "code": "for i in [1, 2, 3] {\n    print('Number:', i)\n}"
            }
            print(f"\n📤 Отправка теста 3: {json.dumps(test3)}")
            await websocket.send(json.dumps(test3))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"\n📥 Получен ответ:")
            print(f"  Success: {result['success']}")
            print(f"  Output: {result['output']}")
            
            # Тест 4: Функция
            test4 = {
                "type": "execute",
                "code": "fn greet(name) {\n    return 'Hello, ' + name + '!'\n}\nprint(greet('DataCode'))"
            }
            print(f"\n📤 Отправка теста 4: {json.dumps(test4)}")
            await websocket.send(json.dumps(test4))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"\n📥 Получен ответ:")
            print(f"  Success: {result['success']}")
            print(f"  Output: {result['output']}")
            
            # Тест 5: Ошибка (для проверки обработки ошибок)
            test5 = {
                "type": "execute",
                "code": "print(undefined_variable)"
            }
            print(f"\n📤 Отправка теста 5 (ожидаем ошибку): {json.dumps(test5)}")
            await websocket.send(json.dumps(test5))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"\n📥 Получен ответ:")
            print(f"  Success: {result['success']}")
            print(f"  Output: {result['output']}")
            if result.get('error'):
                print(f"  Error: {result['error']}")
            
            print("\n✅ Все тесты завершены")
            
    except websockets.exceptions.ConnectionRefused:
        print("❌ Ошибка: Не удалось подключиться к серверу")
        print("💡 Убедитесь, что сервер запущен: datacode --websocket --host 0.0.0.0 --port 8899")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())


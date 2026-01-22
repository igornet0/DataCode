#!/usr/bin/env python3
"""
Тестовый скрипт для загрузки файлов через WebSocket сервер DataCode
Требуется: pip install websockets

Важно: Сервер должен быть запущен с флагом --use-ve:
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
            print("✅ Подключено к серверу")
            print("💡 Убедитесь, что сервер запущен с флагом --use-ve")
            print()
            
            # Тест 1: Проверка getcwd() - должен вернуть пустую строку для безопасности
            print("📋 Тест 1: Проверка getcwd() (должен вернуть пустую строку для безопасности)")
            test1 = {
                "type": "execute",
                "code": "global cwd = getcwd()\nprint('Current directory:', cwd)\nprint('Type of cwd:', typeof(cwd))"
            }
            print(f"📤 Отправка: {json.dumps(test1, ensure_ascii=False)}")
            await websocket.send(json.dumps(test1))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"📥 Получен ответ:")
            print(f"  Success: {result['success']}")
            print(f"  Output: {result['output']}")
            if result.get('error'):
                print(f"  Error: {result['error']}")
            print()
            
            # Тест 2: Загрузка текстового файла
            print("📋 Тест 2: Загрузка текстового файла")
            text_content = """Hello, DataCode!
This is a test file uploaded via WebSocket.
Line 3 of the file.
"""
            upload_text = {
                "type": "upload_file",
                "filename": "test.txt",
                "content": text_content
            }
            print(f"📤 Отправка файла: test.txt ({len(text_content)} байт)")
            await websocket.send(json.dumps(upload_text))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"📥 Получен ответ:")
            print(f"  Success: {result['success']}")
            print(f"  Message: {result.get('message', '')}")
            if result.get('error'):
                print(f"  Error: {result['error']}")
            print()
            
            # Тест 3: Загрузка CSV файла
            print("📋 Тест 3: Загрузка CSV файла")
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
            print(f"📤 Отправка файла: data.csv ({len(csv_content)} байт)")
            await websocket.send(json.dumps(upload_csv))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"📥 Получен ответ:")
            print(f"  Success: {result['success']}")
            print(f"  Message: {result.get('message', '')}")
            if result.get('error'):
                print(f"  Error: {result['error']}")
            print()
            
            # Тест 4: Загрузка файла в поддиректории
            print("📋 Тест 4: Загрузка файла в поддиректории")
            subdir_content = "This file is in a subdirectory\n"
            upload_subdir = {
                "type": "upload_file",
                "filename": "subdir/nested_file.txt",
                "content": subdir_content
            }
            print(f"📤 Отправка файла: subdir/nested_file.txt")
            await websocket.send(json.dumps(upload_subdir))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"📥 Получен ответ:")
            print(f"  Success: {result['success']}")
            print(f"  Message: {result.get('message', '')}")
            if result.get('error'):
                print(f"  Error: {result['error']}")
            print()
            
            # Тест 5: Загрузка бинарного файла (base64)
            print("📋 Тест 5: Загрузка бинарного файла (base64)")
            # Создаем простой PNG файл (1x1 пиксель, прозрачный)
            png_data = base64.b64encode(
                bytes.fromhex('89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000a49444154789c6300010000000500010d0a2db40000000049454e44ae426082')
            ).decode('utf-8')
            
            upload_binary = {
                "type": "upload_file",
                "filename": "image.png",
                "content": f"base64:{png_data}"
            }
            print(f"📤 Отправка файла: image.png (base64, {len(png_data)} символов)")
            await websocket.send(json.dumps(upload_binary))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"📥 Получен ответ:")
            print(f"  Success: {result['success']}")
            print(f"  Message: {result.get('message', '')}")
            if result.get('error'):
                print(f"  Error: {result['error']}")
            print()
            
            # Тест 6: Чтение загруженного CSV файла через DataCode
            print("📋 Тест 6: Чтение загруженного CSV файла через DataCode")
            read_csv_code = """
# Поскольку getcwd() возвращает пустую строку, используем относительные пути
# Файлы загружаются в папку сессии пользователя

# Базовое чтение файла
global data = read_file(path("data.csv"), header_row=0)
print("Загружено строк:", len(data))
table_info(data)

# Чтение с фильтрацией колонок через header (массив)
global data_filtered = read_file(path("data.csv"), header=["Name", "Age", "City"])
print("Загружено строк с фильтрацией:", len(data_filtered))
print("Колонки:", data_filtered.columns)

# Чтение с переименованием колонок через header (словарь)
global data_renamed = read_file(path("data.csv"), header_row=0, header={"Name": "FullName", "Age": null, "City": null, "Salary": null})
print("Загружено строк с переименованием:", len(data_renamed))
print("Колонки:", data_renamed.columns)
"""
            read_csv = {
                "type": "execute",
                "code": read_csv_code
            }
            print(f"📤 Выполнение кода для чтения CSV")
            await websocket.send(json.dumps(read_csv))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"📥 Получен ответ:")
            print(f"  Success: {result['success']}")
            print(f"  Output: {result['output']}")
            if result.get('error'):
                print(f"  Error: {result['error']}")
            print()
            
            # Тест 7: Работа с несколькими файлами
            print("📋 Тест 7: Работа с несколькими загруженными файлами")
            multi_file_code = """
# Читаем текстовый файл
global text = read_file(path("test.txt"))
print("Содержимое test.txt:")
print(text)

# Читаем CSV файл
global csv_data = read_file(path("data.csv"))
print("Количество строк в CSV:", len(csv_data))
"""
            multi_file = {
                "type": "execute",
                "code": multi_file_code
            }
            print(f"📤 Выполнение кода для работы с несколькими файлами")
            await websocket.send(json.dumps(multi_file))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"📥 Получен ответ:")
            print(f"  Success: {result['success']}")
            print(f"  Output: {result['output']}")
            if result.get('error'):
                print(f"  Error: {result['error']}")
            print()
            
            # Тест 8: Загрузка папки с данными разных типов и перебор через цикл
            print("📋 Тест 8: Загрузка папки с данными разных типов")
            data_dir = "data_dir"
            source_data_dir = Path(__file__).parent / "data"
            
            # Загружаем файлы из папки data
            print(f"📤 Загрузка файлов из {source_data_dir} в папку {data_dir}/...")
            
            if not source_data_dir.exists():
                print(f"  ⚠️  Папка {source_data_dir} не найдена")
            else:
                # Получаем список всех файлов в папке data
                files_to_upload = []
                for file_path in source_data_dir.iterdir():
                    if file_path.is_file():
                        # Сохраняем относительный путь для загрузки на сервер
                        target_filename = f"{data_dir}/{file_path.name}"
                        files_to_upload.append((target_filename, file_path))
                
                for target_filename, file_path in files_to_upload:
                    try:
                        # Используем функцию upload_file_from_disk для подготовки запроса
                        upload_req = upload_file_from_disk(websocket, str(file_path), target_filename)
                        
                        await websocket.send(json.dumps(upload_req))
                        response = await websocket.recv()
                        result = json.loads(response)
                        if result.get('success'):
                            print(f"  ✅ {target_filename}")
                        else:
                            print(f"  ❌ {target_filename}: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        print(f"  ❌ {target_filename}: Ошибка при загрузке - {e}")
            
            print()
            
            # Тест 9: Перебор файлов в папке через цикл (с getcwd())
            print("📋 Тест 9: Перебор файлов в папке через цикл list_files (с getcwd())")
            list_files_code = f"""
# В режиме --use-ve getcwd() возвращает пустую строку для безопасности
# Но относительные пути автоматически разрешаются относительно папки сессии
global current_dir = getcwd()
print("Текущая директория (getcwd()): '", current_dir, "'")

# Используем относительный путь - он автоматически разрешится относительно папки сессии
global dir_path = path("{data_dir}")
print("Путь к папке (относительный):", dir_path)

global files = list_files(dir_path)

print("\\nФайлы в папке """ + data_dir + """:")
for file in files {
    print("  -", file, file.parent, file.parent.parent)
}

print("\\nВсего файлов:", len(files))
"""
            list_files_request = {
                "type": "execute",
                "code": list_files_code
            }
            print(f"📤 Выполнение кода для перебора файлов")
            await websocket.send(json.dumps(list_files_request))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"📥 Получен ответ:")
            print(f"  Success: {result['success']}")
            print(f"  Output: {result['output']}")
            if result.get('error'):
                print(f"  Error: {result['error']}")
            print()
            
            # Тест 10: Обработка файлов разных типов
            print("📋 Тест 10: Обработка файлов разных типов из папки")
            process_files_code = f"""
# Используем относительный путь - он автоматически разрешится относительно папки сессии
global dir_path = path("{data_dir}")
global files = list_files(dir_path)
""" + """
print("Обработка файлов:")
for file in files {
    print("Файл:", file)

    if !file.is_file {
        continue
    }
    
    # Определяем тип файла по расширению
    if file.extension == "txt" {
        global content = read_file(file)
        print("  Тип: Текстовый файл")
        print("  Содержимое:", content)
    }
    
    if file.extension == "csv" {
        global csv_data = read_file(file)
        print("  Тип: CSV файл")
        print("  Строк:", len(csv_data))
        if len(csv_data) > 0 {
            print("  Первая строка:", csv_data.idx[0])
        }
    }
    
    if file.extension == "xlsx" {
        print("  Тип: Excel файл ", file)
        global xlsx_data = read_file(file)
        print("  Строк:", len(xlsx_data))
        if len(xlsx_data) > 0 {
            print("  Первая строка:", xlsx_data.idx[0])
        }
    }
    
    if file.extension == "zip" {
        print("  Тип: ZIP архив")
        print("  (Бинарные файлы загружены успешно)")
    }
}
"""
            process_files_request = {
                "type": "execute",
                "code": process_files_code
            }
            print(f"📤 Выполнение кода для обработки файлов разных типов")
            await websocket.send(json.dumps(process_files_request))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"📥 Получен ответ:")
            print(f"  Success: {result['success']}")
            print(f"  Output: {result['output']}")
            if result.get('error'):
                print(f"  Error: {result['error']}")
            print()

            print("📋 Тест 11: Проверка списка файлов в папке")
            list_files_code = """
            print('Файлы в папке getcwd():')
            for file in list_files(getcwd()) {
                print("  -", file)
            }

            print()
            print('Файлы в папке ".":')

            for file in list_files(".") {
                print("  -", file)
            }

            try {
                print("Файлы в папке '..' (должно быть ошибка):")
                for file in list_files("..") {
                    print("  -", file)
                }
            } catch e {
                print("Error: ", e)
                print("Должно быть ошибка")
            }

            try {
                print("Файлы в папке '../' (должно быть ошибка):")
                for file in list_files("../") {
                    print("  -", file)
                }
            } catch e {
                print("Error: ", e)
                print("Должно быть ошибка")
            }

            try {

                print("Файлы в папке '../getcwd()' (должно быть ошибка):")
                for file in list_files(".." / getcwd()) {
                    print("  -", file)
                }

            } catch e {
                print("Error: ", e)
                print("Должно быть ошибка")
            }

            try {
                print("Файлы в папке '../..' (должно быть ошибка):")
                for file in list_files("../..") {
                    print("  -", file)
                }

            } catch e { 
                print("Error: ", e)
                print("Должно быть ошибка")
            }

            try {
                print("Чтение файла с несуществующим путем (должно быть ошибка):")
                data = read_file(path("nonexistent.txt"))
                print("  -", data)
                
            } catch e {
                print("Error: ", e)
                print("Должно быть ошибка")
            }

            """
            list_files_request = {
                "type": "execute",
                "code": list_files_code
            }
            print(f"📤 Выполнение кода для проверки списка файлов")
            await websocket.send(json.dumps(list_files_request))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"📥 Получен ответ:")
            print(f"  Success: {result['success']}")
            print(f"  Output: {result['output']}")
            if result.get('error'):
                print(f"  Error: {result['error']}")
            print()
            
            print("✅ Все тесты завершены")
            print("💡 Папка сессии будет автоматически удалена при отключении")
            
    except websockets.exceptions.ConnectionRefused:
        print("❌ Ошибка: Не удалось подключиться к серверу")
        print("💡 Убедитесь, что сервер запущен с флагом --use-ve:")
        print("   datacode --websocket --host 0.0.0.0 --port 8899 --use-ve")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

def upload_file_from_disk(websocket, file_path, target_filename=None):
    """
    Вспомогательная функция для загрузки файла с диска
    
    Args:
        websocket: WebSocket соединение
        file_path: Путь к файлу на диске
        target_filename: Имя файла на сервере (если None, используется имя исходного файла)
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    filename = target_filename or path.name
    
    # Определяем, текстовый это файл или бинарный
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Текстовый файл - отправляем как есть
        upload_request = {
            "type": "upload_file",
            "filename": filename,
            "content": content
        }
    except UnicodeDecodeError:
        # Бинарный файл - кодируем в base64
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
    Пример загрузки локального файла с диска
    """
    uri = "ws://127.0.0.1:8899"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Подключено к серверу")
            print()
            
            # Пример: загружаем файл из текущей директории
            # Замените на путь к вашему файлу
            local_file = "example.txt"
            
            if os.path.exists(local_file):
                print(f"📤 Загрузка локального файла: {local_file}")
                upload_request = upload_file_from_disk(websocket, local_file)
                
                await websocket.send(json.dumps(upload_request))
                response = await websocket.recv()
                result = json.loads(response)
                
                print(f"📥 Получен ответ:")
                print(f"  Success: {result['success']}")
                print(f"  Message: {result.get('message', '')}")
                if result.get('error'):
                    print(f"  Error: {result['error']}")
            else:
                print(f"⚠️  Файл {local_file} не найден")
                print("💡 Создайте файл example.txt для тестирования")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--upload-local":
        # Режим загрузки локального файла
        asyncio.run(upload_local_file_example())
    else:
        # Обычный режим тестирования
        asyncio.run(test_file_upload())


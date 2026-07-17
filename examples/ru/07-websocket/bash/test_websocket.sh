#!/bin/bash

# Тестовый скрипт для WebSocket сервера DataCode
# Требуется: websocat (cargo install websocat)

SERVER="ws://127.0.0.1:8899"

echo "🧪 Тестирование WebSocket сервера DataCode"
echo "=========================================="
echo ""

# Проверяем наличие websocat
if ! command -v websocat &> /dev/null; then
    echo "❌ websocat не установлен"
    echo "💡 Установите: cargo install websocat"
    exit 1
fi

# Тест 1: Простой вывод
echo "📤 Тест 1: Простой вывод"
echo '{"code": "print(\"Hello, World!\")"}' | websocat $SERVER
echo ""

# Тест 2: Переменные
echo "📤 Тест 2: Переменные"
echo '{"code": "global x = 10\nglobal y = 20\nprint(\"Sum:\", x + y)"}' | websocat $SERVER
echo ""

# Тест 3: Цикл
echo "📤 Тест 3: Цикл"
echo '{"code": "for i in [1, 2, 3] {\n    print(\"Number:\", i)\n }"}' | websocat $SERVER
echo ""

# Тест 4: Функция
echo "📤 Тест 4: Функция"
echo '{"code": "fn greet(name) {\n    return \"Hello, \" + name + \"!\"\n}\nprint(greet(\"DataCode\"))"}' | websocat $SERVER
echo ""

echo "✅ Тестирование завершено"


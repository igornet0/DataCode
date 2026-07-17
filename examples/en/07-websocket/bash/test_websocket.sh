#!/usr/bin/env bash

# Test script for the DataCode WebSocket server
# Requires: websocat (cargo install websocat)

SERVER="ws://127.0.0.1:8899"

echo "🧪 Testing DataCode WebSocket server"
echo "=========================================="
echo ""

# Check websocat is installed
if ! command -v websocat &> /dev/null; then
    echo "❌ websocat is not installed"
    echo "💡 Install: cargo install websocat"
    exit 1
fi

# Test 1: Simple print
echo "📤 Test 1: Simple print"
echo '{"code": "print(\"Hello, World!\")"}' | websocat $SERVER
echo ""

# Test 2: Variables
echo "📤 Test 2: Variables"
echo '{"code": "global x = 10\nglobal y = 20\nprint(\"Sum:\", x + y)"}' | websocat $SERVER
echo ""

# Test 3: Loop
echo "📤 Test 3: Loop"
echo '{"code": "for i in [1, 2, 3] {\n    print(\"Number:\", i)\n }"}' | websocat $SERVER
echo ""

# Test 4: Function
echo "📤 Test 4: Function"
echo '{"code": "fn greet(name) {\n    return \"Hello, \" + name + \"!\"\n}\nprint(greet(\"DataCode\"))"}' | websocat $SERVER
echo ""

echo "✅ Testing completed"

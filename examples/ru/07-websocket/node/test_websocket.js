const WebSocket = require('ws');

const ws = new WebSocket('ws://127.0.0.1:8899');

ws.on('open', function open() {
    console.log('✅ Подключено к серверу');
    
    // Тест 1: Простой вывод
    const test1 = {
        code: "print('Hello, World!')"
    };
    console.log('\n📤 Отправка теста 1:', JSON.stringify(test1));
    ws.send(JSON.stringify(test1));
});

let testCount = 0;

ws.on('message', function message(data) {
    const response = JSON.parse(data);
    console.log('\n📥 Получен ответ:');
    console.log('  Success:', response.success);
    console.log('  Output:', response.output);
    if (response.error) {
        console.log('  Error:', response.error);
    }
    
    testCount++;
    
    if (testCount === 1) {
        // Тест 2: Переменные
        const test2 = {
            code: "global x = 10\nglobal y = 20\nprint('Sum:', x + y)"
        };
        console.log('\n📤 Отправка теста 2:', JSON.stringify(test2));
        ws.send(JSON.stringify(test2));
    } else if (testCount === 2) {
        // Тест 3: Цикл
        const test3 = {
            code: "for i in [1, 2, 3] {\n    print('Number:', i)\n}"
        };
        console.log('\n📤 Отправка теста 3:', JSON.stringify(test3));
        ws.send(JSON.stringify(test3));
    } else if (testCount === 3) {
        // Тест 4: Функция
        const test4 = {
            code: "fn greet(name) {\n    return 'Hello, ' + name + '!'\n}\nprint(greet('DataCode'))"
        };
        console.log('\n📤 Отправка теста 4:', JSON.stringify(test4));
        ws.send(JSON.stringify(test4));
    } else {
        ws.close();
    }
});

ws.on('error', function error(err) {
    console.error('❌ Ошибка:', err.message);
});

ws.on('close', function close() {
    console.log('\n🔌 Соединение закрыто');
    console.log('✅ Все тесты завершены');
});


const WebSocket = require('ws');

const ws = new WebSocket('ws://127.0.0.1:8899');

ws.on('open', function open() {
    console.log('✅ Connected to server');
    
    // Test 1: Simple output
    const test1 = {
        code: "print('Hello, World!')"
    };
    console.log('\n📤 Sending test 1:', JSON.stringify(test1));
    ws.send(JSON.stringify(test1));
});

let testCount = 0;

ws.on('message', function message(data) {
    const response = JSON.parse(data);
    console.log('\n📥 Response received:');
    console.log('  Success:', response.success);
    console.log('  Output:', response.output);
    if (response.error) {
        console.log('  Error:', response.error);
    }
    
    testCount++;
    
    if (testCount === 1) {
        // Test 2: Variables
        const test2 = {
            code: "global x = 10\nglobal y = 20\nprint('Sum:', x + y)"
        };
        console.log('\n📤 Submitting test 2:', JSON.stringify(test2));
        ws.send(JSON.stringify(test2));
    } else if (testCount === 2) {
        // Test 3: Loop
        const test3 = {
            code: "for i in [1, 2, 3] {\n    print('Number:', i)\n}"
        };
        console.log('\n📤 Submitting test 3:', JSON.stringify(test3));
        ws.send(JSON.stringify(test3));
    } else if (testCount === 3) {
        // Test 4: Function
        const test4 = {
            code: "fn greet(name) {\n    return 'Hello, ' + name + '!'\n}\nprint(greet('DataCode'))"
        };
        console.log('\n📤 Submitting test 4:', JSON.stringify(test4));
        ws.send(JSON.stringify(test4));
    } else {
        ws.close();
    }
});

ws.on('error', function error(err) {
    console.error('❌ Error:', err.message);
});

ws.on('close', function close() {
    console.log('\n🔌 Connection closed');
    console.log('✅ All tests completed');
});


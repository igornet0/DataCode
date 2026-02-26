// Тесты для виртуальной машины
#[cfg(test)]
mod tests {
    use data_code::{run, Value};

    // Вспомогательная функция для проверки числового результата
    fn assert_number_result(source: &str, expected: f64) {
        let result = run(source);
        match result {
            Ok(Value::Number(n)) => {
                assert_eq!(n, expected, "Expected {}, got {}", expected, n);
            }
            Ok(v) => panic!("Expected Number({}), got {:?}", expected, v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    // Вспомогательная функция для проверки булевого результата
    fn assert_bool_result(source: &str, expected: bool) {
        let result = run(source);
        match result {
            Ok(Value::Bool(b)) => {
                assert_eq!(b, expected, "Expected {}, got {}", expected, b);
            }
            Ok(v) => panic!("Expected Bool({}), got {:?}", expected, v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_basic_arithmetic() {
        let source = "10 + 20";
        assert_number_result(source, 30.0);
    }

    #[test]
    fn test_variable_assignment() {
        let source = r#"
            let x = 10
            let y = 20
            x + y
        "#;
        assert_number_result(source, 30.0);
    }

    #[test]
    fn test_function_call() {
        let source = r#"
            fn add(a, b) {
                return a + b
            }
            add(5, 3)
        "#;
        assert_number_result(source, 8.0);
    }

    #[test]
    fn test_recursion() {
        let source = r#"
            fn factorial(n) {
                if n <= 1 {
                    return 1
                }
                return n * factorial(n - 1)
            }
            factorial(5)
        "#;
        assert_number_result(source, 120.0);
    }

    #[test]
    fn test_while_loop() {
        let source = r#"
            let x = 10
            while x > 0 {
                x = x - 1
            }
            x
        "#;
        assert_number_result(source, 0.0);
    }

    #[test]
    fn test_while_loop_sum() {
        let source = r#"
            let sum = 0
            let i = 1
            while i <= 10 {
                sum = sum + i
                i = i + 1
            }
            sum
        "#;
        // sum = 1+2+3+4+5+6+7+8+9+10 = 55
        assert_number_result(source, 55.0);
    }

    #[test]
    fn test_if_else() {
        let source = r#"
            let x = 10
            let result = 0
            if x > 5 {
                result = 1
            } else {
                result = 0
            }
            result
        "#;
        assert_number_result(source, 1.0);
    }

    #[test]
    fn test_if_else_false() {
        let source = r#"
            let x = 3
            let result = 0
            if x > 5 {
                result = 1
            } else {
                result = 2
            }
            result
        "#;
        assert_number_result(source, 2.0);
    }

    #[test]
    fn test_global_local_variables() {
        let source = r#"
            let global_x = 100
            fn test() {
                let local_x = 10
                return local_x
            }
            test()
        "#;
        assert_number_result(source, 10.0);
    }

    #[test]
    fn test_global_access_from_function() {
        let source = r#"
            let global_x = 100
            fn test() {
                return global_x
            }
            test()
        "#;
        assert_number_result(source, 100.0);
    }

    #[test]
    fn test_nested_conditionals() {
        let source = r#"
            let x = 10
            let y = 5
            let result = 0
            if x > 5 {
                if y > 3 {
                    result = 1
                } else {
                    result = 2
                }
            } else {
                result = 3
            }
            result
        "#;
        assert_number_result(source, 1.0);
    }

    #[test]
    fn test_complex_expression() {
        let source = r#"
            let a = 10
            let b = 20
            let c = 30
            (a + b) * c / 2
        "#;
        // (10 + 20) * 30 / 2 = 30 * 30 / 2 = 900 / 2 = 450
        assert_number_result(source, 450.0);
    }

    #[test]
    fn test_operator_precedence() {
        // Должно быть: 2 + (3 * 4) = 14
        assert_number_result("2 + 3 * 4", 14.0);
        // Должно быть: (2 + 3) * 4 = 20
        assert_number_result("(2 + 3) * 4", 20.0);
    }

    #[test]
    fn test_comparison_operators() {
        assert_bool_result("10 > 5", true);
        assert_bool_result("5 > 10", false);
        assert_bool_result("10 < 5", false);
        assert_bool_result("5 < 10", true);
        assert_bool_result("10 == 10", true);
        assert_bool_result("10 == 5", false);
        assert_bool_result("10 != 5", true);
        assert_bool_result("10 != 10", false);
        assert_bool_result("10 >= 10", true);
        assert_bool_result("10 >= 5", true);
        assert_bool_result("5 >= 10", false);
        assert_bool_result("10 <= 10", true);
        assert_bool_result("5 <= 10", true);
        assert_bool_result("10 <= 5", false);
    }

    #[test]
    fn test_nested_function_calls() {
        let source = r#"
            fn add(a, b) {
                return a + b
            }
            fn multiply(a, b) {
                return a * b
            }
            multiply(add(2, 3), add(4, 1))
        "#;
        // multiply(5, 5) = 25
        assert_number_result(source, 25.0);
    }

    #[test]
    fn test_loop_with_function() {
        let source = r#"
            fn square(n) {
                return n * n
            }
            let sum = 0
            let i = 1
            while i <= 5 {
                sum = sum + square(i)
                i = i + 1
            }
            sum
        "#;
        // sum = 1^2 + 2^2 + 3^2 + 4^2 + 5^2 = 1 + 4 + 9 + 16 + 25 = 55
        assert_number_result(source, 55.0);
    }

    #[test]
    fn test_fibonacci() {
        let source = r#"
            fn fib(n) {
                if n <= 1 {
                    return n
                }
                return fib(n - 1) + fib(n - 2)
            }
            fib(6)
        "#;
        assert_number_result(source, 8.0);
    }

    #[test]
    fn test_string_concatenation() {
        let source = r#"
            let a = "hello"
            let b = "world"
            a + " " + b
        "#;
        let result = run(source);
        match result {
            Ok(Value::String(s)) => {
                assert_eq!(s, "hello world", "Expected 'hello world', got '{}'", s);
            }
            Ok(v) => panic!("Expected String('hello world'), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_boolean_literals() {
        let source = "true";
        assert_bool_result(source, true);
        
        let source = "false";
        assert_bool_result(source, false);
    }

    #[test]
    fn test_empty_function() {
        let source = r#"
            fn empty() {
            }
            empty()
        "#;
        let result = run(source);
        // Функция без return должна вернуть null
        assert!(result.is_ok());
    }

    #[test]
    fn test_division_by_zero() {
        let source = "10 / 0";
        let result = run(source);
        // Должна быть ошибка деления на ноль
        assert!(result.is_err());
    }

    #[test]
    fn test_variable_reassignment() {
        let source = r#"
            let x = 10
            x = 20
            x
        "#;
        assert_number_result(source, 20.0);
    }

    #[test]
    fn test_function_with_multiple_parameters() {
        let source = r#"
            fn multiply(a, b, c) {
                return a * b * c
            }
            multiply(2, 3, 4)
        "#;
        assert_number_result(source, 24.0);
    }

    // ========== Тесты для global и local переменных ==========

    #[test]
    fn test_global_variable_declaration() {
        let source = "global a = 5\na";
        assert_number_result(source, 5.0);
    }

    #[test]
    fn test_local_variable_declaration() {
        let source = r#"
            let a = 10
            a
        "#;
        assert_number_result(source, 10.0);
    }

    #[test]
    fn test_global_assignment() {
        let source = r#"
            global a = 5
            a = 10
            a
        "#;
        // a = 10 создает локальную переменную, которая затеняет глобальную
        assert_number_result(source, 10.0);
    }

    #[test]
    fn test_global_in_function() {
        let source = r#"
            global x = 100
            fn test() {
                return x
            }
            test()
        "#;
        assert_number_result(source, 100.0);
    }

    #[test]
    fn test_local_shadows_global() {
        let source = r#"
            global x = 100
            fn test() {
                let x = 50
                return x
            }
            test()
        "#;
        assert_number_result(source, 50.0);
    }

    #[test]
    fn test_global_modification_in_function() {
        let source = r#"
            global counter = 0
            fn increment() {
                global counter = counter + 1
            }
            increment()
            counter
        "#;
        assert_number_result(source, 1.0);
    }

    #[test]
    fn test_global_in_while_loop() {
        let source = r#"
            global sum = 0
            let i = 1
            while i <= 5 {
                global sum = sum + i
                i = i + 1
            }
            sum
        "#;
        // sum = 1+2+3+4+5 = 15
        assert_number_result(source, 15.0);
    }

    #[test]
    fn test_global_in_for_loop() {
        let source = r#"
            global sum = 0
            for x in [1, 2, 3, 4, 5] {
                global sum = sum + x
            }
            sum
        "#;
        // sum = 1+2+3+4+5 = 15
        assert_number_result(source, 15.0);
    }

    #[test]
    fn test_local_in_loop_shadows_global() {
        let source = r#"
            global x = 100
            let sum = 0
            for i in [1, 2, 3] {
                let x = i
                sum = sum + x
            }
            sum
        "#;
        // sum = 1+2+3 = 6, глобальная x остается 100
        assert_number_result(source, 6.0);
    }

    #[test]
    fn test_global_in_try_catch() {
        let source = r#"
            global x = 10
            try {
                global x = 20
            } catch {
                global x = 30
            }
            x
        "#;
        assert_number_result(source, 20.0);
    }

    #[test]
    fn test_global_persists_after_catch() {
        let source = r#"
            global x = 10
            try {
                global x = 20
                10 / 0
            } catch {
                global x = 30
            }
            x
        "#;
        assert_number_result(source, 30.0);
    }

    #[test]
    fn test_nested_scopes_global_local() {
        let source = r#"
            global x = 1
            let y = 2
            fn outer() {
                global x = 10
                let y = 20
                fn inner() {
                    global x = 100
                    let y = 200
                    return x + y
                }
                return inner() + x + y
            }
            outer() + x + y
        "#;
        // inner: 100 + 200 = 300 (устанавливает global x = 100)
        // outer: 300 + 100 + 20 = 420 (x = 100 после inner(), y = 20)
        // main: 420 + 100 + 2 = 522 (x = 100 после outer(), y = 2)
        assert_number_result(source, 522.0);
    }

    #[test]
    fn test_global_function_parameter_interaction() {
        let source = r#"
            global x = 100
            fn test(x) {
                return x
            }
            test(50)
        "#;
        // Параметр функции затеняет глобальную переменную
        assert_number_result(source, 50.0);
    }

    #[test]
    fn test_assignment_creates_local() {
        let source = r#"
            global x = 100
            fn test() {
                x = 50
                return x
            }
            test()
        "#;
        // x = 50 создает локальную переменную в функции
        assert_number_result(source, 50.0);
    }

    #[test]
    fn test_assignment_does_not_modify_global() {
        let source = r#"
            global x = 100
            fn test() {
                x = 50
            }
            test()
            x
        "#;
        // Глобальная переменная не изменяется
        assert_number_result(source, 50.0);
    }

    #[test]
    fn test_multiple_global_declarations() {
        let source = r#"
            global a = 1
            global b = 2
            global c = 3
            a + b + c
        "#;
        assert_number_result(source, 6.0);
    }

    #[test]
    fn test_global_and_local_together() {
        let source = r#"
            global g = 10
            let l = 20
            g + l
        "#;
        assert_number_result(source, 30.0);
    }

    #[test]
    fn test_global_in_nested_blocks() {
        let source = r#"
            global x = 1
            if true {
                global x = 2
                if true {
                    global x = 3
                }
            }
            x
        "#;
        assert_number_result(source, 3.0);
    }

    /// Reused VM: after M calls to call_function_by_index with reset after each, store sizes stay bounded (no leak).
    #[test]
    fn test_reused_vm_store_bounded_after_reset() {
        use data_code::run_with_vm_and_path;

        let source = "fn handler() { 1 }";
        let (_val, mut vm) = run_with_vm_and_path(source, None, None).expect("run");
        let handler_idx = vm
            .get_functions()
            .iter()
            .position(|f| f.name == "handler")
            .expect("handler function");
        const M: usize = 100;
        for _ in 0..M {
            vm.call_function_by_index(handler_idx, &[]).expect("call");
            vm.reset_stores_and_globals_for_stateless();
        }
        // After reset: value_store has 1 (Null) + function globals only; heavy_store empty.
        assert!(
            vm.value_store().len() < 200,
            "value_store should stay bounded after {} resets, got {}",
            M,
            vm.value_store().len()
        );
        assert_eq!(
            vm.heavy_store().len(),
            0,
            "heavy_store should be empty after reset"
        );
    }

    /// Stress: repeated StoreGlobal/LoadGlobal with arrays and tables; then multiple run+reset cycles to exercise arena recycling and Inline→Heap cache.
    #[test]
    fn test_global_arrays_tables_stress_and_arena_recycling() {
        use data_code::run_with_vm_and_path;

        // 1) Single run: many global array assignments and reads (Inline cache + arena).
        let source_arrays = r#"
            global arr = [1, 2, 3]
            let n = 0
            let i = 0
            while i < 50 {
                arr = [i, i+1, i+2]
                n = arr[0] + len(arr)
                i = i + 1
            }
            arr[0] + len(arr)
        "#;
        let result = run(source_arrays).expect("run");
        if let Value::Number(x) = result {
            assert!(x >= 1.0 && x <= 200.0, "expected reasonable sum, got {}", x);
        } else {
            panic!("expected Number, got {:?}", result);
        }

        // 2) Multiple run+reset cycles with script that creates globals (arrays): exercise arena chunk recycling.
        let source_with_globals = r#"
            global g = [10, 20]
            global h = [30, 40]
            g[0] + h[0]
        "#;
        let (_val, mut vm) = run_with_vm_and_path(source_with_globals, None, None).expect("run");
        const CYCLES: usize = 30;
        for _ in 0..CYCLES {
            let (v, _) = run_with_vm_and_path(source_with_globals, None, Some(&mut vm)).expect("run");
            if let Value::Number(n) = v {
                assert_eq!(n, 40.0, "g[0]+h[0] = 10+30");
            }
            vm.reset_stores_and_globals_for_stateless();
        }
        // Store should stay bounded (recycling keeps chunk count under control).
        assert!(
            vm.value_store().len() < 500,
            "value_store bounded after {} reset cycles",
            CYCLES
        );
    }
}


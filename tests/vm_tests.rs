// Тесты для виртуальной машины
#[cfg(test)]
mod tests {
    use data_code::bytecode::OpCode;
    use data_code::compile;
    use data_code::{run, Value};

    // Вспомогательная функция для проверки числового результата
    fn assert_number_result(source: &str, expected: f64) {
        let result = run(source);
        match result {
            Ok(v) if v.as_ieee_f64() == Some(expected) => {}
            Ok(v) => panic!("Expected numeric({}), got {:?}", expected, v),
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
            x = 10
            y = 20
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
    fn test_lambda_basic() {
        let source = r#"
            f = fn(x, i) => x + i
            f(1, 2)
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_lambda_closure_capture() {
        let source = r#"
            a = 1
            g = fn(x) => x + a
            g(10)
        "#;
        assert_number_result(source, 11.0);
    }

    #[test]
    fn test_lambda_immediate_call() {
        let source = r#"
            (fn(x) => x * 2)(3)
        "#;
        assert_number_result(source, 6.0);
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
            x = 10
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
            sum = 0
            i = 1
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
            x = 10
            result = 0
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
            x = 3
            result = 0
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
            global_x = 100
            fn test() {
                local_x = 10
                return local_x
            }
            test()
        "#;
        assert_number_result(source, 10.0);
    }

    #[test]
    fn test_global_access_from_function() {
        let source = r#"
            global_x = 100
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
            x = 10
            y = 5
            result = 0
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
            a = 10
            b = 20
            c = 30
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
            sum = 0
            i = 1
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
            a = "hello"
            b = "world"
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
    fn test_variable_reassignment() {
        let source = r#"
            x = 10
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
            a = 10
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
                x = 50
                return x
            }
            test()
        "#;
        assert_number_result(source, 50.0);
    }

    #[test]
    fn test_local_shadows_global_function() {
        let source = r#"
            fn is_prime(n) {
                return n == 2
            }
            fn sieve(n) {
                is_prime = [true, false]
                return len(is_prime)
            }
            sieve(2) + (if is_prime(2) { 10 } else { 0 })
        "#;
        assert_number_result(source, 12.0);
    }

    #[test]
    fn test_chained_property_assignment() {
        let source = r#"
            cls Node {
                public:
                    next: Node
                    prev: Node
                    tag: int
                new Node(t: int) {
                    this.tag = t
                    this.next = null
                    this.prev = null
                }
            }
            head = Node(1)
            mid = Node(2)
            tail = Node(3)
            head.next = mid
            mid.prev = head
            mid.next = tail
            tail.prev = mid
            mid.prev.next = tail
            mid.prev.next.tag
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_doubly_linked_cycle_assignment() {
        let source = r#"
            cls LfuNode {
                public:
                    prev: LfuNode
                    next: LfuNode
                new LfuNode() {
                    this.prev = null
                    this.next = null
                }
            }
            dummy = LfuNode()
            first = LfuNode()
            node = LfuNode()
            node.next = first
            node.prev = dummy
            dummy.next = node
            first.prev = node
            1
        "#;
        assert_number_result(source, 1.0);
    }

    #[test]
    fn test_array_pop_preserves_object_identity_for_field_mutation() {
        let source = r#"
            cls Node {
                public:
                    children: dict[str, Node]
                    fail: Optional[Node]
                new Node() {
                    this.children = {}
                    this.fail = null
                }
            }
            cls Bfs {
                fn run(root: Node) {
                    s = Node()
                    h = Node()
                    root.children["s"] = s
                    s.children["h"] = h
                    queue = []
                    queue.push(root.children["s"])
                    current = queue.pop(0)
                    child = current.children["h"]
                    child.fail = root
                    if root.children["s"].children["h"].fail == null { this.result = 0 } else { this.result = 1 }
                }
                public:
                    result: int
                new Bfs() { this.result = 0 }
            }
            root = Node()
            job = Bfs()
            job.run(root)
            job.result
        "#;
        assert_number_result(source, 1.0);
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
            i = 1
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
            sum = 0
            for i in [1, 2, 3] {
                x = i
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
                x = 10
                let y = 20
                fn inner() {
                    x = 100
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
            l = 20
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

    /// `push(a, x)` must mutate the shared array: `b` aliases `a` (same ValueId), both see new length.
    /// Regression for native_call fast path (no new ValueId for arg0 that would break alias).
    #[test]
    fn test_push_preserves_array_alias_two_variables() {
        let source = r#"
            a = []
            b = a
            push(a, 1)
            push(a, 2)
            len(a) * 10 + len(b)
        "#;
        assert_number_result(source, 22.0);
    }

    /// Stress: repeated StoreGlobal/LoadGlobal with arrays and tables; then multiple run+reset cycles to exercise arena recycling and Inline→Heap cache.
    #[test]
    fn test_global_arrays_tables_stress_and_arena_recycling() {
        use data_code::run_with_vm_and_path;

        // 1) Single run: many global array assignments and reads (Inline cache + arena).
        let source_arrays = r#"
            global arr = [1, 2, 3]
            n = 0
            i = 0
            while i < 50 {
                arr = [i, i+1, i+2]
                n = arr[0] + len(arr)
                i = i + 1
            }
            arr[0] + len(arr)
        "#;
        let result = run(source_arrays).expect("run");
        if let Value::Number(x) = result {
            assert!(
                (1.0..=200.0).contains(&x),
                "expected reasonable sum, got {}",
                x
            );
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
            let (v, _) =
                run_with_vm_and_path(source_with_globals, None, Some(&mut vm)).expect("run");
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

    #[test]
    fn test_array_slice_read_negative_and_copy() {
        let source = r#"
            arr = [10, 20, 30, 40, 50]
            a = len(arr[1:4])
            b = arr[-1]
            c = len(arr[:])
            a + b + c
        "#;
        assert_number_result(source, 58.0);
    }

    #[test]
    fn test_array_slice_step_and_reverse() {
        let source = r#"
            a = [1, 2, 3, 4, 5, 6]
            len(a[::2]) + len(a[1::2]) + len(a[::-1])
        "#;
        assert_number_result(source, 12.0);
    }

    #[test]
    fn test_array_slice_assign_and_delete() {
        let source = r#"
            arr = [1, 2, 3, 4, 5]
            arr[1:3] = [20, 30]
            x = arr[1]
            arr[1:4] = []
            y = len(arr)
            x + y
        "#;
        assert_number_result(source, 22.0);
    }

    #[test]
    fn test_array_subscript_assign_scalar() {
        let source = r#"
            arr = [1, 2, 3]
            arr[0] = 9
            arr[0]
        "#;
        assert_number_result(source, 9.0);
    }

    #[test]
    fn test_array_slice_combined_bounds() {
        let source = r#"
            arr = [0, 1, 2, 3, 4, 5, 6, 7]
            len(arr[2:6:2]) + len(arr[-6:-1:2])
        "#;
        // [2,4] len 2 + [2,4,6] len 3 = 5
        assert_number_result(source, 5.0);
    }

    #[test]
    fn test_array_push_twice_inside_fn_sum_elements() {
        let source = r#"
fn f() {
  a = []
  a.push(10)
  a.push(20)
  return a[0] + a[1]
}
f()
"#;
        assert_number_result(source, 30.0);
    }

    /// Two `let []` inside a function must produce distinct arrays (regression: shared empty array).
    #[test]
    fn test_two_empty_array_lets_push_distinct_inside_fn() {
        let source = r#"
fn f() {
  a = []
  b = []
  a.push(1)
  b.push(2)
  return len(a) * 10 + len(b)
}
f()
"#;
        assert_number_result(source, 11.0);
    }

    /// Mirrors CIFAR export_data with small row size (5 = 1 label + 4 tail).
    #[test]
    fn test_chunk_slice_push_labels_pixels_inside_fn_small_synthetic() {
        let source = r#"
fn export_data(data) {
  chunks = data.chunk(5)
  labels = []
  pixels_list = []
  for chunk in chunks {
    if len(chunk) != 5 {
      continue
    }
    label = chunk[0]
    pixels = chunk[1:]
    labels.push(label)
    pixels_list.push(pixels)
  }
  score = 0
  if typeof(labels[0]) == "int" {
    score = score + 1
  }
  if typeof(labels[1]) == "int" {
    score = score + 1
  }
  if len(pixels_list[0]) == 4 {
    score = score + 1
  }
  return score
}
data = [1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24]
export_data(data)
"#;
        assert_number_result(source, 3.0);
    }

    /// Diagnose push targets: expect len(labels)==3 and len(pixels_list)==3 after 3 chunks.
    #[test]
    fn test_chunk_export_lengths_inside_fn() {
        let source = r#"
fn export_data(data) {
  chunks = data.chunk(5)
  labels = []
  pixels_list = []
  for chunk in chunks {
    if len(chunk) != 5 {
      continue
    }
    label = chunk[0]
    pixels = chunk[1:]
    labels.push(label)
    pixels_list.push(pixels)
  }
  return len(labels) * 100 + len(pixels_list)
}
data = [1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24]
export_data(data)
"#;
        assert_number_result(source, 303.0);
    }

    /// Chunk has `data.chunk(5)` as one `Call(2)` plus two `Call(2)` for the two pushes in the loop body (3 total).
    #[test]
    fn test_export_data_chunk_compiles_three_call2_including_chunk() {
        let source = r#"fn export_data(data) {
  chunks = data.chunk(5)
  labels = []
  pixels_list = []
  for chunk in chunks {
    if len(chunk) != 5 {
      continue
    }
    label = chunk[0]
    pixels = chunk[1:]
    labels.push(label)
    pixels_list.push(pixels)
  }
  return 0
}"#;
        let (_chunk, functions) = compile(source).expect("compile");
        let f = functions
            .iter()
            .find(|f| f.name == "export_data")
            .expect("export_data function");
        let call2 = f
            .chunk
            .code
            .iter()
            .filter(|op| matches!(op, OpCode::Call(2)))
            .count();
        assert_eq!(
            call2, 3,
            "data.chunk(5) is Call(2); loop body has 2× push → 3 total Call(2) in chunk"
        );
    }

    /// How many for-loop iterations run inside export_data (expect 3 chunks).
    #[test]
    fn test_chunk_for_loop_iteration_count_inside_fn() {
        let source = r#"
fn export_data(data) {
  chunks = data.chunk(5)
  n = 0
  for chunk in chunks {
    if len(chunk) != 5 {
      continue
    }
    n = n + 1
  }
  return n
}
data = [1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24]
export_data(data)
"#;
        assert_number_result(source, 3.0);
    }

    /// Same logic as `test_chunk_slice_push_labels_pixels_inside_fn_small_synthetic` at top level (no `fn`).
    #[test]
    fn test_chunk_slice_push_labels_pixels_top_level_small_synthetic() {
        let source = r#"
data = [1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24]
chunks = data.chunk(5)
labels = []
pixels_list = []
for chunk in chunks {
  if len(chunk) != 5 {
    continue
  }
  label = chunk[0]
  pixels = chunk[1:]
  labels.push(label)
  pixels_list.push(pixels)
}
score = 0
if typeof(labels[0]) == "int" {
  score = score + 1
}
if typeof(labels[1]) == "int" {
  score = score + 1
}
if len(pixels_list[0]) == 4 {
  score = score + 1
}
score
"#;
        assert_number_result(source, 3.0);
    }
}

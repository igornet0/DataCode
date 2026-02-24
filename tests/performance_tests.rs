// Performance tests for DataCode interpreter
// These tests execute performance-intensive .dc files and verify successful completion

#[cfg(test)]
mod tests {
    use data_code::run;
    use std::time::Instant;

    // Helper function to execute a test and measure time
    fn run_performance_test(source: &str, test_name: &str) -> Result<(), data_code::LangError> {
        let start = Instant::now();
        run(source)?;
        let duration = start.elapsed();
        println!("{} completed in {:?}", test_name, duration);
        Ok(())
    }

    #[test]
    fn test_simple_performance() {
        let source = include_str!("performance_tests/simple_performance_test.dc");
        run_performance_test(source, "Simple performance test")
            .expect("Simple performance test should complete without errors");
    }

    #[test]
    fn test_recursive_performance() {
        let source = include_str!("performance_tests/recursive_performance_test.dc");
        run_performance_test(source, "Recursive performance test")
            .expect("Recursive performance test should complete without errors");
    }

    #[test]
    fn test_memory_intensive() {
        let source = include_str!("performance_tests/memory_intensive_test.dc");
        run_performance_test(source, "Memory-intensive test")
            .expect("Memory-intensive test should complete without errors");
    }

    /// Large dataset (10k rows): stress arena and store; verifies no OOM and completion.
    #[test]
    fn test_large_dataset_10k() {
        let source = include_str!("performance_tests/10_short_large_dataset_test.dc");
        run_performance_test(source, "Large dataset 10k rows")
            .expect("Large dataset 10k test should complete without errors");
    }

    /// Long series of run+reset with VM reuse: stress arena recycling and lazy shrink.
    #[test]
    fn test_reset_cycles_arena_stress() {
        use data_code::run_with_vm_and_path;
        let source = r#"
            global g = [1, 2, 3]
            global h = [4, 5, 6]
            g[0] + h[0]
        "#;
        let (_v, mut vm) = run_with_vm_and_path(source, None, None).expect("run");
        const CYCLES: usize = 50;
        for _ in 0..CYCLES {
            let (v, _) = run_with_vm_and_path(source, None, Some(&mut vm)).expect("run");
            if let data_code::Value::Number(n) = v {
                assert_eq!(n, 5.0);
            }
            vm.reset_stores_and_globals_for_stateless();
        }
        assert!(
            vm.value_store().len() < 600,
            "value_store bounded after {} reset cycles (arena recycling + lazy shrink)",
            CYCLES
        );
    }

    /// Stress: 100 run+reset cycles to verify arena recycling and lazy shrink under long series.
    #[test]
    fn test_reset_cycles_100() {
        use data_code::run_with_vm_and_path;
        let source = r#"
            global a = [1, 2]
            global b = [3, 4]
            a[0] + b[0]
        "#;
        let (_v, mut vm) = run_with_vm_and_path(source, None, None).expect("run");
        const CYCLES: usize = 100;
        for _ in 0..CYCLES {
            let (v, _) = run_with_vm_and_path(source, None, Some(&mut vm)).expect("run");
            if let data_code::Value::Number(n) = v {
                assert_eq!(n, 4.0);
            }
            vm.reset_stores_and_globals_for_stateless();
        }
        assert!(
            vm.value_store().len() < 800,
            "value_store bounded after {} reset cycles",
            CYCLES
        );
    }

    /// Stress: many small global arrays (200 globals) to stress store_alloc and arena.
    #[test]
    fn test_many_small_global_arrays() {
        let mut lines = Vec::with_capacity(210);
        for i in 0..200 {
            lines.push(format!("global g{} = [1, 2, 3]", i));
        }
        lines.push("g0[0] + g1[0] + g199[0]".to_string());
        let source = lines.join("\n");
        let result = run(&source).expect("run");
        if let data_code::Value::Number(n) = result {
            assert_eq!(n, 3.0, "g0[0]=1 + g1[0]=1 + g199[0]=1");
        } else {
            panic!("expected Number, got {:?}", result);
        }
    }
}


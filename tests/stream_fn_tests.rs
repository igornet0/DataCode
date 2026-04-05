//! Integration tests for `stream fn` / generators.

#[cfg(test)]
mod tests {
    use data_code::{run, Value};

    #[test]
    fn stream_fn_generators_runs() {
        let source = include_str!("stream_fn_generators.dc");
        run(source).expect("stream_fn_generators.dc should run without errors");
    }

    #[test]
    fn ereturn_stores_final_value_not_yielded() {
        let source = r#"
stream fn t() {
    return 1
    ereturn 42
}
let gen = t()
let sum = 0
for x in gen {
    sum = sum + x
}
sum
"#;
        assert_eq!(run(source).unwrap(), Value::Number(1.0));

        let source2 = r#"
stream fn t() {
    return 1
    ereturn 42
}
let gen = t()
for x in gen {}
gen.final()
"#;
        assert_eq!(run(source2).unwrap(), Value::Number(42.0));
    }

    #[test]
    fn generator_final_null_without_ereturn() {
        let source = r#"
stream fn t() {
    return 1
}
let gen = t()
for x in gen {}
gen.final()
"#;
        assert_eq!(run(source).unwrap(), Value::Null);
    }

    #[test]
    fn generator_final_before_completion_is_null() {
        let source = r#"
stream fn t() {
    return 1
    ereturn 42
}
let gen = t()
gen.final()
"#;
        assert_eq!(run(source).unwrap(), Value::Null);
    }

    #[test]
    fn generator_final_idempotent() {
        let source = r#"
stream fn t() {
    ereturn 99
}
let g = t()
for x in g {}
g.final() + g.final()
"#;
        assert_eq!(run(source).unwrap(), Value::Number(198.0));
    }

    #[test]
    fn generator_return_assign_next_send_then_next() {
        let source = r#"
stream fn t() {
    x = return 10
    return x * 2
}
gen = t()
res1 = gen.next()
gen.send(5)
res2 = gen.next()
res1 + res2
"#;
        assert_eq!(run(source).unwrap(), Value::Number(20.0));
    }

    #[test]
    fn generator_ireturn_expr_next_send_then_next() {
        let source = r#"
stream fn t() {
    x = ireturn 10
    return x * 2
}
gen = t()
res1 = gen.next()
gen.send(5)
res2 = gen.next()
res1 + res2
"#;
        assert_eq!(run(source).unwrap(), Value::Number(20.0));
    }

    #[test]
    fn generator_ireturn_value_next_send_then_next() {
        let source = r#"
stream fn t() {
    x = ireturn 4
    return x * 3
}
gen = t()
res1 = gen.next()
gen.send(4)
res2 = gen.next()
res1 + res2
"#;
        assert_eq!(run(source).unwrap(), Value::Number(16.0));
    }

    #[test]
    fn generator_live_reflects_finished() {
        let source = r#"
stream fn t() {
    return 1
}
gen = t()
gen.next()
a = gen.live
gen.next()
b = gen.live
a == true and b == false
"#;
        assert_eq!(run(source).unwrap(), Value::Bool(true));
    }

    #[test]
    fn ireturn_outside_stream_fn_is_error() {
        let source = r#"
fn t() {
    x = ireturn
    return x
}
t()
"#;
        assert!(run(source).is_err());
    }

    #[test]
    fn generator_send_on_plain_yield_errors() {
        let source = r#"
stream fn t() {
    return 1
}
gen = t()
gen.send(10)
"#;
        assert!(run(source).is_err());
    }

    #[test]
    fn generator_return_assign_second_next_resumes_with_rhs_value() {
        let source = r#"
stream fn t() {
    x = return 10
    return x
}
gen = t()
gen.next()
gen.next()
"#;
        // Второй .next() без send подставляет RHS yield (10) в x; `return x` даёт 10.
        assert_eq!(run(source).unwrap(), Value::Number(10.0));
    }

    #[test]
    fn generator_for_in_errors_on_yield_await() {
        let source = r#"
stream fn t() {
    x = return 10
    return x
}
gen = t()
sum = 0
for x in gen {
    sum = sum + x
}
sum
"#;
        assert!(run(source).is_err());
    }

    #[test]
    fn generator_ireturn_next_final_no_print() {
        let source = r#"
stream fn input_arg(n: int) {
    x = ireturn 40
    ereturn x * n
}
gen = input_arg(20)
t = gen.next()
res = gen.final()
t + res
"#;
        assert_eq!(run(source).unwrap(), Value::Number(840.0));
    }

    #[test]
    fn generator_ireturn_print_no_args_then_ereturn() {
        let source = r#"
stream fn input_arg(n: int) {
    x = ireturn 40
    print()
    ereturn x * n
}
gen = input_arg(20)
t = gen.next()
res = gen.final()
t + res
"#;
        assert_eq!(run(source).unwrap(), Value::Number(840.0));
    }

    #[test]
    fn generator_ireturn_str_then_ereturn() {
        let source = r#"
stream fn input_arg(n: int) {
    x = ireturn 40
    str(1)
    ereturn x * n
}
gen = input_arg(20)
t = gen.next()
res = gen.final()
t + res
"#;
        assert_eq!(run(source).unwrap(), Value::Number(840.0));
    }

    /// Regression: `print` with one arg must not use the int/float/str/typeof arity-1 fast path (native_call).
    #[test]
    fn generator_ireturn_print_one_arg_then_ereturn() {
        let source = r#"
stream fn input_arg(n: int) {
    x = ireturn 40
    print("ok")
    ereturn x * n
}
gen = input_arg(20)
t = gen.next()
res = gen.final()
t + res
"#;
        assert_eq!(run(source).unwrap(), Value::Number(840.0));
    }

    /// Regression: `ireturn`/`return` assign slots must receive yielded RHS on `.next()`, not null,
    /// so `ereturn (x + d) * n` sees correct locals after a `while gen.live` loop (fin = 900).
    #[test]
    fn next_gen_test_while_loop_ereturn_fin_900() {
        let source = r#"
stream fn next_gen_test(n: int) {
    x = ireturn 40
    d = return 50
    ereturn (x + d) * n
}
gen = next_gen_test(10)
while gen.live {
    gen.next()
}
gen.final()
"#;
        assert_eq!(run(source).unwrap(), Value::Number(900.0));
    }
}

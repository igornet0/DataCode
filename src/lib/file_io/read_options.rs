//! Read options extracted from `read()` native arguments.

use crate::common::value::Value;

#[derive(Debug, Clone, Default)]
pub struct ReadOptions {
    pub header_row: usize,
    pub sheet_name: Option<String>,
    pub header_filter: Option<Value>,
    pub header_t_filter: Option<Value>,
}

impl ReadOptions {
    pub fn uses_transpose(&self) -> bool {
        self.header_t_filter.is_some()
    }
}

fn is_header_value(v: &Value) -> bool {
    matches!(v, Value::Array(_) | Value::Object(_))
}

fn non_null_header_at(args: &[Value], index: usize) -> Option<&Value> {
    args.get(index).filter(|v| !matches!(v, Value::Null) && is_header_value(v))
}

/// Extract CSV/XLSX options from `read(path, ...)` args (positional + named resolved).
pub fn read_options_from_args(args: &[Value]) -> Result<ReadOptions, String> {
    let mut opts = ReadOptions::default();
    if args.len() <= 1 {
        return Ok(opts);
    }

    // Named-arg layout: path, header_row, sheet_name, header, headerT
    let header_named = non_null_header_at(args, 3);
    let header_t_named = non_null_header_at(args, 4);

    if header_named.is_some() && header_t_named.is_some() {
        return Err(
            "read(): only one of 'header' or 'headerT' may be specified, not both".to_string(),
        );
    }

    if let Some(v) = header_named {
        opts.header_filter = Some(v.clone());
    }
    if let Some(v) = header_t_named {
        opts.header_t_filter = Some(v.clone());
    }

    // Positional fallback: scan for Array/Object when not resolved via named slots.
    if opts.header_filter.is_none() && opts.header_t_filter.is_none() {
        if args.len() >= 5 {
            if let Some(v) = non_null_header_at(args, 4) {
                opts.header_t_filter = Some(v.clone());
            } else if let Some(v) = non_null_header_at(args, 3) {
                opts.header_filter = Some(v.clone());
            }
        } else {
            for arg in args.iter().skip(1) {
                if is_header_value(arg) {
                    opts.header_filter = Some(arg.clone());
                    break;
                }
            }
        }
    }

    if args.len() > 1 {
        let is_header_1 = is_header_value(&args[1]);
        if !is_header_1 {
            match &args[1] {
                Value::Number(n) => {
                    opts.header_row = *n as usize;
                    if args.len() > 2 {
                        let is_header_2 = is_header_value(&args[2]);
                        if !is_header_2 {
                            if let Value::String(s) = &args[2] {
                                if !matches!(args[2], Value::Null) {
                                    opts.sheet_name = Some(s.clone());
                                }
                            }
                        }
                    }
                }
                Value::String(s) => {
                    if !matches!(args[1], Value::Null) {
                        opts.sheet_name = Some(s.clone());
                    }
                }
                _ => {}
            }
        } else if args.len() > 2 {
            let is_header_2 = is_header_value(&args[2]);
            if !is_header_2 {
                if let Value::String(s) = &args[2] {
                    if !matches!(args[2], Value::Null) {
                        opts.sheet_name = Some(s.clone());
                    }
                }
            }
        }
    }

    Ok(opts)
}

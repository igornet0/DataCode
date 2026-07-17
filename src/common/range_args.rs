//! `range(start, end, step)` argument parsing and lazy range metadata.

use crate::common::numeric::integer_value_as_i64_if_whole;
use crate::common::value::{IterableInner, Value};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RangeSpec {
    pub start: i64,
    pub end: i64,
    pub step: i64,
}

impl RangeSpec {
    /// Number of elements the range would yield (same rules as Python `range`).
    pub fn len(&self) -> usize {
        range_len(self.start, self.end, self.step)
    }
}

/// Parse `range` arguments from evaluated [`Value`]s (1, 2, or 3 args).
pub fn range_spec_from_values(args: &[Value]) -> Result<RangeSpec, &'static str> {
    match args.len() {
        1 => {
            let end = integer_value_as_i64_if_whole(&args[0]).ok_or("integral")?;
            Ok(RangeSpec {
                start: 0,
                end,
                step: 1,
            })
        }
        2 => {
            let start = integer_value_as_i64_if_whole(&args[0]).ok_or("integral")?;
            let end = integer_value_as_i64_if_whole(&args[1]).ok_or("integral")?;
            Ok(RangeSpec {
                start,
                end,
                step: 1,
            })
        }
        3 => {
            let start = integer_value_as_i64_if_whole(&args[0]).ok_or("integral")?;
            let end = integer_value_as_i64_if_whole(&args[1]).ok_or("integral")?;
            let step = integer_value_as_i64_if_whole(&args[2]).ok_or("integral")?;
            if step == 0 {
                return Err("zero_step");
            }
            Ok(RangeSpec {
                start,
                end,
                step,
            })
        }
        _ => Err("arity"),
    }
}

pub fn range_len(start: i64, end: i64, step: i64) -> usize {
    if step == 0 {
        return 0;
    }
    if step > 0 {
        if start >= end {
            0
        } else {
            let n = (end - start + step - 1) / step;
            n.max(0) as usize
        }
    } else if start <= end {
        0
    } else {
        let n = (start - end - step - 1) / -step;
        n.max(0) as usize
    }
}

/// Lazy `range(...)` as [`Value::Iterable`].
pub fn value_from_range_spec(spec: RangeSpec) -> Value {
    Value::Iterable(Rc::new(RefCell::new(IterableInner::Range {
        current: spec.start,
        end: spec.end,
        step: spec.step,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::numeric::IntValue;

    #[test]
    fn range_spec_from_int_and_number() {
        let s = range_spec_from_values(&[Value::Int(IntValue::Finite(10))]).unwrap();
        assert_eq!(s, RangeSpec { start: 0, end: 10, step: 1 });
        assert_eq!(s.len(), 10);

        let s = range_spec_from_values(&[Value::Number(3.0), Value::Number(7.0)]).unwrap();
        assert_eq!(s.start, 3);
        assert_eq!(s.end, 7);
    }

    #[test]
    fn float_whole_accepted_fractional_rejected() {
        use crate::common::numeric::FloatValue;
        let s = range_spec_from_values(&[Value::Float(FloatValue::Finite(5.0))]).unwrap();
        assert_eq!(s.end, 5);
        assert!(range_spec_from_values(&[Value::Float(FloatValue::Finite(5.5))]).is_err());
    }

    #[test]
    fn rejects_zero_step() {
        let args = [
            Value::Number(1.0),
            Value::Number(10.0),
            Value::Number(0.0),
        ];
        assert_eq!(range_spec_from_values(&args), Err("zero_step"));
    }

    #[test]
    fn range_len_three_arg() {
        assert_eq!(range_len(1, 10, 2), 5);
        assert_eq!(range_len(10, 0, -1), 10);
    }
}

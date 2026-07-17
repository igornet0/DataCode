//! Distinguished integer infinity, float infinity/NaN, and finite numeric primitives.
//!
//! Integer infinities are **not** `i64::MAX` / naive IEEE bits on tagged int-domain values.
//! Non-finite float values align with IEEE on the tagged float stack (`f64` bits).

use std::cmp::Ordering;

use crate::common::type_model::hash_finish;
use crate::common::value::Value;
use crate::common::value_store::ValueCell;
use crate::common::TaggedValue;

/// Integer sentinel: finite `i64` or ±∞ (ordered; typed as `int` in surface).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntValue {
    Finite(i64),
    PosInfinity,
    NegInfinity,
}

/// Float numeric: IEEE-like equality for non-finite; [`FloatValue::NaN`] participates in `PartialEq`.
#[derive(Debug, Clone, Copy)]
pub enum FloatValue {
    Finite(f64),
    NaN,
    PosInfinity,
    NegInfinity,
}

impl PartialEq for FloatValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (FloatValue::Finite(a), FloatValue::Finite(b)) => a == b,
            (FloatValue::PosInfinity, FloatValue::PosInfinity) => true,
            (FloatValue::NegInfinity, FloatValue::NegInfinity) => true,
            // IEEE: NaN is never equal to anything, including NaN
            (_, _) => false,
        }
    }
}

impl FloatValue {
    #[inline]
    pub fn classify_f64(n: f64) -> Self {
        if n.is_nan() {
            FloatValue::NaN
        } else if n == f64::INFINITY {
            FloatValue::PosInfinity
        } else if n == f64::NEG_INFINITY {
            FloatValue::NegInfinity
        } else {
            FloatValue::Finite(n)
        }
    }

    #[inline]
    pub fn as_f64(self) -> Option<f64> {
        match self {
            FloatValue::Finite(v) => Some(v),
            _ => None,
        }
    }

    #[inline]
    pub fn as_raw_f64(self) -> f64 {
        f64::from_bits(self.to_f64_bits_for_stack())
    }

    #[inline]
    pub fn to_f64_bits_for_stack(self) -> u64 {
        match self {
            FloatValue::Finite(n) => n.to_bits(),
            FloatValue::NaN => f64::NAN.to_bits(),
            FloatValue::PosInfinity => f64::INFINITY.to_bits(),
            FloatValue::NegInfinity => f64::NEG_INFINITY.to_bits(),
        }
    }

    #[inline]
    pub fn from_f64_for_stack(n: f64) -> Self {
        Self::classify_f64(n)
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        matches!(self, FloatValue::Finite(v) if v.is_finite())
    }

    #[inline]
    pub fn is_infinity(self) -> bool {
        matches!(self, FloatValue::PosInfinity | FloatValue::NegInfinity)
    }

    #[inline]
    pub fn is_nan(self) -> bool {
        matches!(self, FloatValue::NaN)
    }

    #[inline]
    pub fn neg(self) -> Self {
        match self {
            FloatValue::Finite(n) => FloatValue::Finite(-n),
            FloatValue::NaN => FloatValue::NaN,
            FloatValue::PosInfinity => FloatValue::NegInfinity,
            FloatValue::NegInfinity => FloatValue::PosInfinity,
        }
    }

    pub fn to_display_string(self) -> String {
        match self {
            FloatValue::PosInfinity => "inf".into(),
            FloatValue::NegInfinity => "-inf".into(),
            FloatValue::NaN => "nan".into(),
            FloatValue::Finite(n) => {
                if n.fract() == 0.0 && n.abs() <= (i64::MAX as f64) && n.is_finite() {
                    format!("{}", n as i64)
                } else {
                    format!("{}", n)
                }
            }
        }
    }
}

/// Whole finite floats (`42.0`) classify as `int` for `typeof`; fractional or non-finite as `float`.
#[inline]
pub fn float_is_int_surface(f: FloatValue) -> bool {
    f.as_f64().is_some_and(|n| n.is_finite() && n.fract() == 0.0)
}

/// Legacy `number` literals: whole values classify as `int` for `typeof`.
#[inline]
pub fn number_is_int_surface(n: f64) -> bool {
    n.fract() == 0.0
}

impl IntValue {
    #[inline]
    pub fn neg(self) -> Self {
        match self {
            IntValue::Finite(n) => IntValue::Finite(-n),
            IntValue::PosInfinity => IntValue::NegInfinity,
            IntValue::NegInfinity => IntValue::PosInfinity,
        }
    }

    #[inline]
    pub fn is_infinity(self) -> bool {
        matches!(self, IntValue::PosInfinity | IntValue::NegInfinity)
    }

    /// Map integer sentinel to IEEE float ±∞ (`float(int(±∞))`).
    #[inline]
    pub fn widen_to_float(self) -> FloatValue {
        match self {
            IntValue::Finite(n) => FloatValue::Finite(n as f64),
            IntValue::PosInfinity => FloatValue::PosInfinity,
            IntValue::NegInfinity => FloatValue::NegInfinity,
        }
    }

    #[inline]
    pub fn to_display_string(self) -> String {
        match self {
            IntValue::Finite(n) => format!("{}", n),
            IntValue::PosInfinity => "inf".into(),
            IntValue::NegInfinity => "-inf".into(),
        }
    }
}

#[inline]
pub fn value_classifies_as_float_surface(fv: FloatValue) -> bool {
    !matches!(
        fv,
        FloatValue::Finite(n) if n.fract() == 0.0 && n.is_finite()
    ) || fv.is_nan()
        || fv.is_infinity()
}

/// Cross-type equality: `int(±∞) == float(±∞)` with same sign; finite compares i as f64.
pub fn numeric_eq_int_float(a: IntValue, b: FloatValue) -> bool {
    match (a, b) {
        (IntValue::PosInfinity, FloatValue::PosInfinity) => true,
        (IntValue::NegInfinity, FloatValue::NegInfinity) => true,
        (IntValue::Finite(i), FloatValue::Finite(f)) => (i as f64) == f,
        _ => false,
    }
}

/// Canonical hash for dict/set keys that are mathematically integral (`hash(1) == hash(1.0)` in Python).
#[inline]
pub fn hash_integral_key(n: i64) -> u64 {
    hash_finish(0x20, n as u64)
}

pub fn hash_int_value(iv: IntValue) -> u64 {
    match iv {
        IntValue::Finite(v) => hash_integral_key(v),
        IntValue::PosInfinity => hash_finish(0x20, 1),
        IntValue::NegInfinity => hash_finish(0x20, 2),
    }
}

/// Stable hash aligned with [`numeric_eq_int_float`] (±∞ share payloads with ints).
pub fn hash_float_value(fv: FloatValue) -> u64 {
    match fv {
        FloatValue::PosInfinity => hash_finish(0x20, 1),
        FloatValue::NegInfinity => hash_finish(0x20, 2),
        FloatValue::NaN => hash_finish(0x22, 0),
        FloatValue::Finite(n) => {
            if n.is_finite() && n.fract() == 0.0 {
                hash_integral_key(f64_trunc_to_i64_clamped(n))
            } else {
                hash_finish(0x21, n.to_bits())
            }
        }
    }
}

/// Object-key hash for numeric [`Value`]s: whole numbers share one bucket across `int` / `number` / whole `float`.
pub fn hash_numeric_key_value(v: &Value) -> Option<u64> {
    match v {
        Value::Int(i) => Some(hash_int_value(*i)),
        Value::Number(n) => Some(hash_float_value(FloatValue::from_f64_for_stack(*n))),
        Value::Float(f) => Some(hash_float_value(*f)),
        _ => None,
    }
}

/// Total sort order: `-inf < finite < +inf < NaN` (crate-internal; for sort helpers only).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum SortAtom {
    NegInf,
    Finite(f64),
    PosInf,
    Nan,
}

impl IntValue {
    #[inline]
    pub(crate) fn into_sort_atom(self) -> SortAtom {
        match self {
            IntValue::NegInfinity => SortAtom::NegInf,
            IntValue::PosInfinity => SortAtom::PosInf,
            IntValue::Finite(n) => SortAtom::Finite(n as f64),
        }
    }
}

impl FloatValue {
    #[inline]
    pub(crate) fn sort_atom(self) -> SortAtom {
        match self {
            FloatValue::NegInfinity => SortAtom::NegInf,
            FloatValue::PosInfinity => SortAtom::PosInf,
            FloatValue::NaN => SortAtom::Nan,
            FloatValue::Finite(n) => SortAtom::Finite(n),
        }
    }
}

#[inline]
fn rank(a: SortAtom) -> u8 {
    match a {
        SortAtom::NegInf => 0,
        SortAtom::Finite(_) => 1,
        SortAtom::PosInf => 2,
        SortAtom::Nan => 3,
    }
}

pub fn cmp_numeric_values(a: &Value, b: &Value) -> Option<Ordering> {
    let atom_a = match a {
        Value::Int(i) => i.into_sort_atom(),
        Value::Float(f) => f.sort_atom(),
        Value::Number(n) => FloatValue::from_f64_for_stack(*n).sort_atom(),
        _ => return None,
    };
    let atom_b = match b {
        Value::Int(i) => i.into_sort_atom(),
        Value::Float(f) => f.sort_atom(),
        Value::Number(n) => FloatValue::from_f64_for_stack(*n).sort_atom(),
        _ => return None,
    };
    Some(numeric_sort_class(atom_a, atom_b))
}

pub(crate) fn numeric_sort_class(a: SortAtom, b: SortAtom) -> Ordering {
    rank(a).cmp(&rank(b)).then_with(|| match (a, b) {
        (SortAtom::Finite(x), SortAtom::Finite(y)) => {
            x.partial_cmp(&y).unwrap_or(Ordering::Equal)
        }
        _ => Ordering::Equal,
    })
}

/// Truncate `f64` toward zero and clamp to `i64` (for `int()` / string parse).
#[inline]
pub fn f64_trunc_to_i64_clamped(n: f64) -> i64 {
    let t = n.trunc();
    if t < i64::MIN as f64 {
        i64::MIN
    } else if t > i64::MAX as f64 {
        i64::MAX
    } else {
        t as i64
    }
}

/// Python-style floor division on `i64` (truncation toward −∞), `b != 0`.
#[inline]
pub fn floor_div_i64(a: i64, b: i64) -> i64 {
    debug_assert_ne!(b, 0);
    let (mut q, r) = (a / b, a % b);
    if r != 0 && ((a < 0) != (b < 0)) {
        q -= 1;
    }
    q
}

/// Python-style `%` on `i64` (same remainder as [`divmod_i64`]); `b != 0`.
#[inline]
pub fn floor_mod_i64(a: i64, b: i64) -> i64 {
    debug_assert_ne!(b, 0);
    let q = floor_div_i64(a, b);
    a - q * b
}

/// `(a // b, a % b)` with Python floor-div semantics; `b != 0`.
#[inline]
pub fn divmod_i64(a: i64, b: i64) -> (i64, i64) {
    debug_assert_ne!(b, 0);
    let q = floor_div_i64(a, b);
    (q, a - q * b)
}

/// Python-style `divmod` on finite `f64`; `b != 0`, `b` finite.
#[inline]
pub fn divmod_f64(a: f64, b: f64) -> (f64, f64) {
    debug_assert_ne!(b, 0.0);
    let q = (a / b).floor();
    (q, a - q * b)
}

/// Canonical `i64` from an immediate numeric [`TaggedValue`] on the stack (whole `number` / `int` only).
#[inline]
pub fn tagged_integral_canonical_if_whole(tv: TaggedValue) -> Option<i64> {
    // Float-domain stack numbers (`TaggedValue::from_f64`, ForRange locals) must use `get_f64` first:
    // some whole `f64` bit patterns also satisfy `is_int()`, and `get_i32()` would read the wrong canonical.
    if tv.is_number() {
        let n = tv.get_f64();
        if n.is_finite() && n.fract() == 0.0 {
            return Some(f64_trunc_to_i64_clamped(n));
        }
        return None;
    }
    if tv.is_int() {
        return Some(tv.get_i32() as i64);
    }
    None
}

/// If `cell` is a finite mathematical integer (`Int` / whole `Number` / whole `Float`), return canonical `i64`.
#[inline]
pub fn integer_cell_as_i64_if_whole(cell: &ValueCell) -> Option<i64> {
    match cell {
        ValueCell::Int(IntValue::Finite(n)) => Some(*n),
        ValueCell::Int(IntValue::PosInfinity | IntValue::NegInfinity) => None,
        ValueCell::Number(x) => {
            if !x.is_finite() || x.fract() != 0.0 {
                None
            } else {
                Some(f64_trunc_to_i64_clamped(*x))
            }
        }
        ValueCell::Float(FloatValue::Finite(x)) => {
            if !x.is_finite() || x.fract() != 0.0 {
                None
            } else {
                Some(f64_trunc_to_i64_clamped(*x))
            }
        }
        _ => None,
    }
}

/// If `v` represents a finite mathematical integer (`int`, or whole `number` / `float`),
/// return it as `i64` for the `divmod` fast path (literals are often `number`).
pub fn integer_value_as_i64_if_whole(v: &Value) -> Option<i64> {
    match v {
        Value::Int(IntValue::Finite(n)) => Some(*n),
        Value::Int(IntValue::PosInfinity | IntValue::NegInfinity) => None,
        Value::Number(x) => {
            if !x.is_finite() || x.fract() != 0.0 {
                return None;
            }
            Some(f64_trunc_to_i64_clamped(*x))
        }
        Value::Float(FloatValue::Finite(x)) => {
            if !x.is_finite() || x.fract() != 0.0 {
                return None;
            }
            Some(f64_trunc_to_i64_clamped(*x))
        }
        _ => None,
    }
}

#[inline]
pub fn int_value_from_f64_lossy(n: f64) -> IntValue {
    if n.is_nan() {
        IntValue::Finite(0)
    } else if n == f64::INFINITY {
        IntValue::PosInfinity
    } else if n == f64::NEG_INFINITY {
        IntValue::NegInfinity
    } else {
        IntValue::Finite(f64_trunc_to_i64_clamped(n))
    }
}

/// True when `/` with a zero divisor should raise (integer-style), not produce IEEE ±inf.
///
/// Float-domain operands (`float` literals and values) use IEEE rules; `int` and legacy whole
/// `number` literals use integer-style division-by-zero errors (`10 / 0`).
pub fn divide_by_zero_raises(a: &Value, b: &Value) -> bool {
    if b.as_ieee_f64() != Some(0.0) {
        return false;
    }
    if matches!(a, Value::Float(_)) || matches!(b, Value::Float(_)) {
        return false;
    }
    match (a, b) {
        (Value::Int(_), _) | (_, Value::Int(_)) => true,
        (Value::Number(_), Value::Number(_)) => true,
        _ => {
            integer_value_as_i64_if_whole(a).is_some()
                && integer_value_as_i64_if_whole(b).is_some()
        }
    }
}

/// IEEE `/` quotient with float-domain result typing when either operand is `float`.
pub fn ieee_div_quotient_value(a: &Value, b: &Value, x: f64, y: f64) -> Value {
    let q = x / y;
    if matches!(a, Value::Float(_)) || matches!(b, Value::Float(_)) {
        Value::Float(FloatValue::classify_f64(q))
    } else {
        Value::Number(q)
    }
}

/// `int(...)` coercion: maps legacy `Number`, typed `Int`/`Float`, strings, bool, null; other → 0.
pub fn coerce_to_int_value(v: &Value) -> IntValue {
    match v {
        Value::Int(i) => *i,
        Value::Float(f) => match *f {
            FloatValue::PosInfinity => IntValue::PosInfinity,
            FloatValue::NegInfinity => IntValue::NegInfinity,
            FloatValue::NaN => IntValue::Finite(0),
            FloatValue::Finite(n) => IntValue::Finite(f64_trunc_to_i64_clamped(n)),
        },
        Value::Number(n) => int_value_from_f64_lossy(*n),
        Value::String(s) => parse_special_float_string(s)
            .map(|f| match f {
                FloatValue::PosInfinity => IntValue::PosInfinity,
                FloatValue::NegInfinity => IntValue::NegInfinity,
                FloatValue::NaN => IntValue::Finite(0),
                FloatValue::Finite(n) => IntValue::Finite(f64_trunc_to_i64_clamped(n)),
            })
            .or_else(|| {
                s.parse::<f64>()
                    .ok()
                    .map(int_value_from_f64_lossy)
            })
            .unwrap_or(IntValue::Finite(0)),
        Value::Bool(b) => IntValue::Finite(if *b { 1 } else { 0 }),
        Value::Null => IntValue::Finite(0),
        _ => IntValue::Finite(0),
    }
}

/// Parse `inf` / `-inf` / `nan` (case-insensitive) for string coercion.
pub fn parse_special_float_string(s: &str) -> Option<FloatValue> {
    match s.trim().to_ascii_lowercase().as_str() {
        "inf" | "+inf" => Some(FloatValue::PosInfinity),
        "-inf" => Some(FloatValue::NegInfinity),
        "nan" => Some(FloatValue::NaN),
        _ => None,
    }
}

/// Parse a numeric literal lexeme, allowing `_` separators between digits (not at edges).
pub fn parse_number_lexeme(lexeme: &str) -> Result<f64, ()> {
    if lexeme.is_empty() {
        return Err(());
    }
    let bytes = lexeme.as_bytes();
    let mut compact = String::with_capacity(lexeme.len());
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'_' => {
                if i == 0 || i + 1 >= bytes.len() {
                    return Err(());
                }
                if !bytes[i - 1].is_ascii_digit() || !bytes[i + 1].is_ascii_digit() {
                    return Err(());
                }
            }
            b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-' => compact.push(b as char),
            _ => return Err(()),
        }
    }
    compact.parse::<f64>().map_err(|_| ())
}

/// `float(...)` coercion: preserves NaN / ±∞ on `Number` and strings; `Int` widens via [`IntValue::widen_to_float`].
pub fn coerce_to_float_value(v: &Value) -> FloatValue {
    match v {
        Value::Float(f) => *f,
        Value::Int(i) => i.widen_to_float(),
        Value::Number(n) => FloatValue::classify_f64(*n),
        Value::String(s) => parse_special_float_string(s)
            .or_else(|| {
                s.parse::<f64>()
                    .ok()
                    .map(FloatValue::classify_f64)
            })
            .unwrap_or(FloatValue::Finite(0.0)),
        Value::Bool(b) => FloatValue::Finite(if *b { 1.0 } else { 0.0 }),
        Value::Null => FloatValue::Finite(0.0),
        _ => FloatValue::Finite(0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_number_lexeme_underscores() {
        assert_eq!(parse_number_lexeme("2_000_000_000").unwrap(), 2_000_000_000.0);
        assert_eq!(parse_number_lexeme("1_234.5_6").unwrap(), 1234.56);
        assert!(parse_number_lexeme("_1").is_err());
        assert!(parse_number_lexeme("1_").is_err());
        assert!(parse_number_lexeme("1__2").is_err());
    }

    #[test]
    fn parse_number_lexeme_scientific() {
        assert!((parse_number_lexeme("1e-9").unwrap() - 1e-9).abs() < 1e-20);
        assert!((parse_number_lexeme("1E9").unwrap() - 1e9).abs() < 1.0);
        assert!((parse_number_lexeme("2.5e+10").unwrap() - 2.5e10).abs() < 1.0);
        assert!((parse_number_lexeme("6.022e23").unwrap() - 6.022e23).abs() < 1e15);
    }

    #[test]
    fn ieee_nan_never_partial_eq_itself_via_pattern() {
        assert_ne!(FloatValue::NaN, FloatValue::NaN);
    }

    #[test]
    fn int_float_inf_hash_aligns_with_eq_contract() {
        assert_eq!(
            hash_int_value(IntValue::PosInfinity),
            hash_float_value(FloatValue::PosInfinity)
        );
        assert!(numeric_eq_int_float(
            IntValue::PosInfinity,
            FloatValue::PosInfinity
        ));
    }

    #[test]
    fn total_order_nan_last() {
        assert_eq!(
            numeric_sort_class(
                IntValue::NegInfinity.into_sort_atom(),
                SortAtom::Finite(0.0)
            ),
            Ordering::Less
        );
        assert_eq!(
            numeric_sort_class(SortAtom::Finite(0.0), IntValue::PosInfinity.into_sort_atom()),
            Ordering::Less
        );
        assert_eq!(
            numeric_sort_class(
                IntValue::PosInfinity.into_sort_atom(),
                FloatValue::NaN.sort_atom()
            ),
            Ordering::Less
        );
    }

    #[test]
    fn coerce_int_from_number_inf_and_nan() {
        assert_eq!(
            coerce_to_int_value(&Value::Number(f64::INFINITY)),
            IntValue::PosInfinity
        );
        assert_eq!(
            coerce_to_int_value(&Value::Number(f64::NEG_INFINITY)),
            IntValue::NegInfinity
        );
        assert_eq!(
            coerce_to_int_value(&Value::Number(f64::NAN)),
            IntValue::Finite(0)
        );
    }

    #[test]
    fn coerce_int_float_cross_domain_inf() {
        assert_eq!(
            coerce_to_int_value(&Value::Float(FloatValue::PosInfinity)),
            IntValue::PosInfinity
        );
        assert_eq!(
            coerce_to_float_value(&Value::Int(IntValue::NegInfinity)),
            FloatValue::NegInfinity
        );
    }

    #[test]
    fn coerce_float_nan_and_string_inf() {
        assert!(coerce_to_float_value(&Value::Number(f64::NAN)).is_nan());
        assert_eq!(
            coerce_to_int_value(&Value::String("inf".to_string())),
            IntValue::PosInfinity
        );
        assert_eq!(
            coerce_to_int_value(&Value::String("-inf".to_string())),
            IntValue::NegInfinity
        );
        assert_eq!(
            coerce_to_float_value(&Value::String("inf".to_string())),
            FloatValue::PosInfinity
        );
        assert_eq!(
            coerce_to_float_value(&Value::String("-inf".to_string())),
            FloatValue::NegInfinity
        );
        assert!(coerce_to_float_value(&Value::String("nan".to_string())).is_nan());
    }

    #[test]
    fn int_float_inf_cross_eq_after_coerce() {
        let i = coerce_to_int_value(&Value::Number(f64::INFINITY));
        let f = coerce_to_float_value(&Value::Number(f64::INFINITY));
        assert!(numeric_eq_int_float(i, f));
    }

    #[test]
    fn floor_mod_i64_matches_divmod_remainder() {
        for (a, b) in [
            (10, 3),
            (-10, 3),
            (10, -3),
            (-10, -3),
            (-1, 5),
            (1, -5),
            (-1, -5),
        ] {
            let (_, r) = divmod_i64(a, b);
            assert_eq!(floor_mod_i64(a, b), r, "floor_mod_i64({a}, {b})");
        }
    }

    #[test]
    fn divmod_i64_matches_python_floor_semantics() {
        assert_eq!(divmod_i64(10, 3), (3, 1));
        assert_eq!(divmod_i64(20, 5), (4, 0));
        assert_eq!(divmod_i64(9, 2), (4, 1));
        assert_eq!(divmod_i64(-10, 3), (-4, 2));
        assert_eq!(divmod_i64(10, -3), (-4, -2));
        assert_eq!(divmod_i64(-10, -3), (3, -1));
        assert_eq!(divmod_i64(-1, 5), (-1, 4));
        assert_eq!(divmod_i64(1, -5), (-1, -4));
        assert_eq!(divmod_i64(-1, -5), (0, -1));
    }

    #[test]
    fn divmod_f64_sample_pairs() {
        let (q, r) = divmod_f64(10.5, 3.0);
        assert!((q - 3.0).abs() < 1e-12);
        assert!((r - 1.5).abs() < 1e-12);
        let (q, r) = divmod_f64(7.25, 2.5);
        assert!((q - 2.0).abs() < 1e-12);
        assert!((r - 2.25).abs() < 1e-12);
    }
}

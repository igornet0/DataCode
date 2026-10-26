//! Python-like format specifiers for string interpolation: `"${x:9.2f}"`.
//!
//! Grammar (subset of Python's format-spec mini-language):
//! `[[fill]align][sign][#][0][width][grouping][.precision][type]`
//! - align: `<` `>` `^` `=`; sign: `+` `-` ` `; grouping: `,` `_`
//! - type: `f` `F` `e` `E` `d` `%` `s` or none
//!
//! Width is a minimum: values longer than `width` are never truncated.
//! Numbers align right by default, other values align left.

use crate::common::numeric::IntValue;
use crate::common::value::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Align {
    Left,
    Right,
    Center,
    /// Padding goes between the sign and the digits (`=`).
    AfterSign,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormatSpec {
    pub fill: char,
    pub align: Option<Align>,
    pub sign: char,
    pub alternate: bool,
    pub zero_pad: bool,
    pub width: usize,
    pub grouping: Option<char>,
    pub precision: Option<usize>,
    pub ty: Option<char>,
}

fn align_of(c: char) -> Option<Align> {
    match c {
        '<' => Some(Align::Left),
        '>' => Some(Align::Right),
        '^' => Some(Align::Center),
        '=' => Some(Align::AfterSign),
        _ => None,
    }
}

impl FormatSpec {
    /// Parse a spec such as `.2f`, `9.2f`, `>10`, `+08,.1f`. Returns `None` for unsupported/invalid specs.
    pub fn parse(spec: &str) -> Option<FormatSpec> {
        let chars: Vec<char> = spec.trim().chars().collect();
        let mut i = 0;
        let mut out = FormatSpec {
            fill: ' ',
            align: None,
            sign: '-',
            alternate: false,
            zero_pad: false,
            width: 0,
            grouping: None,
            precision: None,
            ty: None,
        };

        if chars.len() >= 2 {
            if let Some(a) = align_of(chars[1]) {
                out.fill = chars[0];
                out.align = Some(a);
                i = 2;
            }
        }
        if out.align.is_none() {
            if let Some(a) = chars.first().copied().and_then(align_of) {
                out.align = Some(a);
                i = 1;
            }
        }
        if let Some(&c) = chars.get(i) {
            if matches!(c, '+' | '-' | ' ') {
                out.sign = c;
                i += 1;
            }
        }
        if chars.get(i) == Some(&'#') {
            out.alternate = true;
            i += 1;
        }
        if chars.get(i) == Some(&'0') {
            out.zero_pad = true;
            i += 1;
        }
        let start = i;
        while chars.get(i).is_some_and(|c| c.is_ascii_digit()) {
            i += 1;
        }
        if i > start {
            out.width = chars[start..i].iter().collect::<String>().parse().ok()?;
        }
        if let Some(&c) = chars.get(i) {
            if c == ',' || c == '_' {
                out.grouping = Some(c);
                i += 1;
            }
        }
        if chars.get(i) == Some(&'.') {
            i += 1;
            let start = i;
            while chars.get(i).is_some_and(|c| c.is_ascii_digit()) {
                i += 1;
            }
            if i == start {
                return None;
            }
            out.precision = Some(chars[start..i].iter().collect::<String>().parse().ok()?);
        }
        if let Some(&c) = chars.get(i) {
            if !"fFeEd%s".contains(c) {
                return None;
            }
            out.ty = Some(c);
            i += 1;
        }
        if i != chars.len() {
            return None;
        }
        Some(out)
    }

    /// Format `value` per this spec. Values that do not fit the type (e.g. a string with `f`)
    /// fall back to their plain string form, still padded to `width`.
    pub fn format(&self, value: &Value) -> String {
        match self.format_number(value) {
            Some((negative, body)) => {
                let sign = if negative {
                    "-"
                } else {
                    match self.sign {
                        '+' => "+",
                        ' ' => " ",
                        _ => "",
                    }
                };
                self.pad(sign, &body, true)
            }
            None => {
                let mut s = value.to_string();
                if self.ty.is_none() || self.ty == Some('s') {
                    if let Some(p) = self.precision {
                        s = s.chars().take(p).collect();
                    }
                }
                let numeric = self.ty.is_none() && value.as_ieee_f64().is_some();
                self.pad("", &s, numeric)
            }
        }
    }

    /// Returns `(is_negative, unsigned_body)` for numeric types, `None` when not applicable.
    fn format_number(&self, value: &Value) -> Option<(bool, String)> {
        let ty = self.ty?;
        let (negative, body) = match ty {
            'd' => {
                let n: i128 = match value {
                    Value::Int(IntValue::Finite(n)) => *n as i128,
                    _ => {
                        let f = value.as_ieee_f64()?;
                        if !f.is_finite() || f.fract() != 0.0 {
                            return None;
                        }
                        f as i128
                    }
                };
                (n < 0, n.unsigned_abs().to_string())
            }
            'f' | 'F' | 'e' | 'E' | '%' => {
                let n = value.as_ieee_f64()?;
                let n = if ty == '%' { n * 100.0 } else { n };
                let prec = self.precision.unwrap_or(6);
                let abs = n.abs();
                let mut body = match ty {
                    'e' | 'E' => format_exp(abs, prec),
                    _ => format!("{:.*}", prec, abs),
                };
                if self.alternate && prec == 0 && abs.is_finite() && !body.contains('.') {
                    match body.find('e') {
                        Some(e) => body.insert(e, '.'),
                        None => body.push('.'),
                    }
                }
                if ty == 'F' || ty == 'E' {
                    body = body.to_uppercase();
                }
                if ty == '%' {
                    body.push('%');
                }
                (n.is_sign_negative() && !n.is_nan(), body)
            }
            _ => return None,
        };
        Some((negative, self.group(body)))
    }

    /// Insert thousands separators into the integer part of `body`.
    fn group(&self, body: String) -> String {
        let Some(sep) = self.grouping else {
            return body;
        };
        let int_len = body.find(|c: char| !c.is_ascii_digit()).unwrap_or(body.len());
        let (int_part, rest) = body.split_at(int_len);
        let mut grouped = String::with_capacity(body.len() + int_len / 3);
        for (k, c) in int_part.chars().enumerate() {
            if k > 0 && (int_len - k) % 3 == 0 {
                grouped.push(sep);
            }
            grouped.push(c);
        }
        grouped.push_str(rest);
        grouped
    }

    fn pad(&self, sign: &str, body: &str, numeric: bool) -> String {
        let len = sign.chars().count() + body.chars().count();
        if len >= self.width {
            return format!("{}{}", sign, body);
        }
        let (fill, default_align) = if self.zero_pad && self.align.is_none() {
            ('0', if numeric { Align::AfterSign } else { Align::Left })
        } else {
            (self.fill, if numeric { Align::Right } else { Align::Left })
        };
        let n = self.width - len;
        let fill_str = |k: usize| std::iter::repeat_n(fill, k).collect::<String>();
        match self.align.unwrap_or(default_align) {
            Align::Left => format!("{}{}{}", sign, body, fill_str(n)),
            Align::Right => format!("{}{}{}", fill_str(n), sign, body),
            Align::Center => format!("{}{}{}{}", fill_str(n / 2), sign, body, fill_str(n - n / 2)),
            Align::AfterSign => format!("{}{}{}", sign, fill_str(n), body),
        }
    }
}

/// Python-style exponent notation: `1.50e+02` (sign and at least two exponent digits).
fn format_exp(abs: f64, prec: usize) -> String {
    let s = format!("{:.*e}", prec, abs);
    match s.split_once('e') {
        Some((mantissa, exp)) => {
            let (exp_sign, digits) = match exp.strip_prefix('-') {
                Some(d) => ('-', d),
                None => ('+', exp),
            };
            format!("{}e{}{:0>2}", mantissa, exp_sign, digits)
        }
        None => s,
    }
}

/// Format a value for string interpolation (`"${x:9.2f}"`). Unsupported specs fall back to `to_string()`.
pub fn format_interpolated(value: &Value, spec: &str) -> String {
    match FormatSpec::parse(spec) {
        Some(fs) => fs.format(value),
        None => value.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(v: Value, spec: &str) -> String {
        format_interpolated(&v, spec)
    }

    #[test]
    fn parse_width_precision() {
        let s = FormatSpec::parse("9.2f").unwrap();
        assert_eq!((s.width, s.precision, s.ty), (9, Some(2), Some('f')));
        let s = FormatSpec::parse(".2f").unwrap();
        assert_eq!((s.width, s.precision, s.ty), (0, Some(2), Some('f')));
        assert!(FormatSpec::parse("9.f").is_none());
        assert!(FormatSpec::parse("9.2q").is_none());
        assert!(FormatSpec::parse("2f3").is_none());
    }

    #[test]
    fn fixed_width() {
        assert_eq!(f(Value::Number(123.456), "9.2f"), "   123.46");
        assert_eq!(f(Value::Number(3.14159), "7.2f"), "   3.14");
        assert_eq!(f(Value::Number(-3.14159), "9.2f"), "    -3.14");
        assert_eq!(f(Value::Number(12345678.9), "7.2f"), "12345678.90");
        assert_eq!(f(Value::Int(IntValue::Finite(42)), "9.2f"), "    42.00");
        assert_eq!(f(Value::Number(-3.14159), "09.2f"), "-00003.14");
    }

    #[test]
    fn other_types() {
        assert_eq!(f(Value::Number(1234567.891), ",.2f"), "1,234,567.89");
        assert_eq!(f(Value::Number(0.256), ".1%"), "25.6%");
        assert_eq!(f(Value::Number(150.0), ".2e"), "1.50e+02");
        assert_eq!(f(Value::Number(0.00015), "10.1E"), "   1.5E-04");
        assert_eq!(f(Value::Int(IntValue::Finite(-42)), "+6d"), "   -42");
        assert_eq!(f(Value::Int(IntValue::Finite(42)), "+6d"), "   +42");
        assert_eq!(f(Value::String("abc".into()), "^7"), "  abc  ");
        assert_eq!(f(Value::String("abcdef".into()), ".3"), "abc");
    }

    #[test]
    fn unsupported_falls_back() {
        assert_eq!(f(Value::Number(1.5), "q"), "1.5");
        assert_eq!(f(Value::String("x".into()), ".2f"), "x");
        assert_eq!(f(Value::Number(1.5), "d"), "1.5");
    }
}

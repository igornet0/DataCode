//! Cryptographic builtins: SHA-2, HMAC, OS-backed randomness (not for password storage).

use crate::common::numeric::integer_value_as_i64_if_whole;
use crate::common::value::ByteBuffer;
use crate::common::value::Value;
use crate::websocket::set_native_error;
use getrandom::getrandom;
use hmac::Hmac;
use hmac::Mac;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sha2::Digest;
use sha2::Sha256;
use sha2::Sha512;
use std::cell::RefCell;

type HmacSha256 = Hmac<Sha256>;
type HmacSha512 = Hmac<Sha512>;

thread_local! {
    static THREAD_RNG: RefCell<StdRng> = RefCell::new(StdRng::from_entropy());
}

/// Max output length for random_bytes to limit DoS.
const RANDOM_BYTES_MAX: usize = 1024 * 1024;

fn byte_buffer_bytes(b: &ByteBuffer) -> Vec<u8> {
    b.bytes[b.offset..b.offset + b.len].to_vec()
}

fn bytes_or_utf8_string(v: &Value) -> Option<Vec<u8>> {
    match v {
        Value::ByteBuffer(b) => Some(byte_buffer_bytes(b)),
        Value::String(s) => Some(s.as_bytes().to_vec()),
        Value::Path(p) => Some(p.as_os_str().as_encoded_bytes().to_vec()),
        _ => None,
    }
}

pub fn native_sha256(args: &[Value]) -> Value {
    let Some(data) = args.first().and_then(bytes_or_utf8_string) else {
        set_native_error("sha256: expected bytes or string".to_string());
        return Value::Null;
    };
    let mut hasher = Sha256::new();
    hasher.update(&data);
    Value::ByteBuffer(ByteBuffer::from_vec_hex(hasher.finalize().to_vec()))
}

pub fn native_sha512(args: &[Value]) -> Value {
    let Some(data) = args.first().and_then(bytes_or_utf8_string) else {
        set_native_error("sha512: expected bytes or string".to_string());
        return Value::Null;
    };
    let mut hasher = Sha512::new();
    hasher.update(&data);
    Value::ByteBuffer(ByteBuffer::from_vec_hex(hasher.finalize().to_vec()))
}

pub fn native_hmac_sha256(args: &[Value]) -> Value {
    if args.len() < 2 {
        set_native_error("hmac_sha256: expected (key: bytes, data: bytes)".to_string());
        return Value::Null;
    }
    let Some(key) = args.first().and_then(|v| match v {
        Value::ByteBuffer(b) => Some(byte_buffer_bytes(b)),
        _ => None,
    }) else {
        set_native_error("hmac_sha256: key must be bytes".to_string());
        return Value::Null;
    };
    let Some(data) = args.get(1).and_then(|v| match v {
        Value::ByteBuffer(b) => Some(byte_buffer_bytes(b)),
        _ => None,
    }) else {
        set_native_error("hmac_sha256: data must be bytes".to_string());
        return Value::Null;
    };
    let mut mac = match HmacSha256::new_from_slice(&key) {
        Ok(m) => m,
        Err(_) => {
            set_native_error("hmac_sha256: invalid key length".to_string());
            return Value::Null;
        }
    };
    mac.update(&data);
    Value::ByteBuffer(ByteBuffer::from_vec_hex(mac.finalize().into_bytes().to_vec()))
}

pub fn native_hmac_sha512(args: &[Value]) -> Value {
    if args.len() < 2 {
        set_native_error("hmac_sha512: expected (key: bytes, data: bytes)".to_string());
        return Value::Null;
    }
    let Some(key) = args.first().and_then(|v| match v {
        Value::ByteBuffer(b) => Some(byte_buffer_bytes(b)),
        _ => None,
    }) else {
        set_native_error("hmac_sha512: key must be bytes".to_string());
        return Value::Null;
    };
    let Some(data) = args.get(1).and_then(|v| match v {
        Value::ByteBuffer(b) => Some(byte_buffer_bytes(b)),
        _ => None,
    }) else {
        set_native_error("hmac_sha512: data must be bytes".to_string());
        return Value::Null;
    };
    let mut mac = match HmacSha512::new_from_slice(&key) {
        Ok(m) => m,
        Err(_) => {
            set_native_error("hmac_sha512: invalid key length".to_string());
            return Value::Null;
        }
    };
    mac.update(&data);
    Value::ByteBuffer(ByteBuffer::from_vec_hex(mac.finalize().into_bytes().to_vec()))
}

pub fn native_random_bytes(args: &[Value]) -> Value {
    let n = match args.first() {
        Some(Value::Number(x)) if x.is_finite() && *x >= 0.0 && x.fract() == 0.0 => *x as usize,
        _ => {
            set_native_error("random_bytes: expected non-negative integer size".to_string());
            return Value::Null;
        }
    };
    if n > RANDOM_BYTES_MAX {
        set_native_error(format!(
            "random_bytes: size exceeds maximum ({})",
            RANDOM_BYTES_MAX
        ));
        return Value::Null;
    }
    let mut buf = vec![0u8; n];
    if getrandom(&mut buf).is_err() {
        set_native_error("random_bytes: OS RNG failure".to_string());
        return Value::Null;
    }
    Value::ByteBuffer(ByteBuffer::from_vec(buf))
}

pub fn native_random_seed(args: &[Value]) -> Value {
    if args.is_empty() {
        set_native_error("random_seed: expected (seed: int)".to_string());
        return Value::Null;
    }
    let seed = match &args[0] {
        Value::Number(x) if x.is_finite() && x.fract() == 0.0 && *x >= 0.0 => *x as u64,
        _ => {
            set_native_error("random_seed: seed must be a non-negative integer".to_string());
            return Value::Null;
        }
    };
    THREAD_RNG.with(|rng| {
        *rng.borrow_mut() = StdRng::seed_from_u64(seed);
    });
    Value::Null
}

pub fn native_random(args: &[Value]) -> Value {
    if !args.is_empty() {
        set_native_error("random: expected no arguments".to_string());
        return Value::Null;
    }
    let v: f64 = THREAD_RNG.with(|rng| rng.borrow_mut().gen());
    Value::Number(v)
}

pub fn native_random_int(args: &[Value]) -> Value {
    if args.len() < 2 {
        set_native_error("random_int: expected (min: int, max: int)".to_string());
        return Value::Null;
    }
    let min = match integer_value_as_i64_if_whole(&args[0]) {
        Some(v) => v,
        _ => {
            set_native_error("random_int: min must be integer".to_string());
            return Value::Null;
        }
    };
    let max = match integer_value_as_i64_if_whole(&args[1]) {
        Some(v) => v,
        _ => {
            set_native_error("random_int: max must be integer".to_string());
            return Value::Null;
        }
    };
    if min > max {
        set_native_error("random_int: min must be <= max".to_string());
        return Value::Null;
    }
    let v: i64 = THREAD_RNG.with(|rng| rng.borrow_mut().gen_range(min..=max));
    Value::Number(v as f64)
}

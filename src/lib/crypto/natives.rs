//! Native functions for `import crypto` — Argon2id, bcrypt, secure_compare.

use crate::common::value::Value;
use crate::websocket::set_native_error;
use argon2::password_hash::{PasswordHasher, PasswordVerifier, SaltString};
use argon2::password_hash::rand_core::OsRng;
use argon2::{Algorithm, Argon2, Params, Version};
use bcrypt::BcryptError;
use password_hash::PasswordHash;
use subtle::ConstantTimeEq;

/// Default Argon2id: ~19 MiB RAM, t=2, p=1 (server-friendly baseline).
fn argon2_instance() -> Argon2<'static> {
    let params = Params::new(19456, 2, 1, None).unwrap_or_else(|_| Params::default());
    Argon2::new(Algorithm::Argon2id, Version::V0x13, params)
}

pub fn native_crypto_argon2_hash(args: &[Value]) -> Value {
    let Some(Value::String(password)) = args.first() else {
        set_native_error("Argon2.hash: expected password: string".to_string());
        return Value::Null;
    };
    let salt = SaltString::generate(&mut OsRng);
    let argon2 = argon2_instance();
    match argon2.hash_password(password.as_bytes(), &salt) {
        Ok(h) => Value::String(h.to_string()),
        Err(e) => {
            set_native_error(format!("Argon2.hash: {}", e));
            Value::Null
        }
    }
}

pub fn native_crypto_argon2_verify(args: &[Value]) -> Value {
    if args.len() < 2 {
        set_native_error("Argon2.verify: expected (password: string, hash: string)".to_string());
        return Value::Null;
    }
    let Some(Value::String(password)) = args.first() else {
        set_native_error("Argon2.verify: password must be string".to_string());
        return Value::Null;
    };
    let Some(Value::String(hash_str)) = args.get(1) else {
        set_native_error("Argon2.verify: hash must be string".to_string());
        return Value::Null;
    };
    let parsed = match PasswordHash::new(hash_str) {
        Ok(p) => p,
        Err(e) => {
            set_native_error(format!("Argon2.verify: invalid hash: {}", e));
            return Value::Bool(false);
        }
    };
    let argon2 = argon2_instance();
    Value::Bool(argon2.verify_password(password.as_bytes(), &parsed).is_ok())
}

const BCRYPT_MIN_COST: u32 = 12;

pub fn native_crypto_bcrypt_hash(args: &[Value]) -> Value {
    let Some(Value::String(password)) = args.first() else {
        set_native_error("bcrypt.hash: expected password: string".to_string());
        return Value::Null;
    };
    match bcrypt::hash(password, BCRYPT_MIN_COST as u32) {
        Ok(h) => Value::String(h),
        Err(e) => {
            set_native_error(format!("bcrypt.hash: {}", e));
            Value::Null
        }
    }
}

pub fn native_crypto_bcrypt_verify(args: &[Value]) -> Value {
    if args.len() < 2 {
        set_native_error("bcrypt.verify: expected (password: string, hash: string)".to_string());
        return Value::Null;
    }
    let Some(Value::String(password)) = args.first() else {
        set_native_error("bcrypt.verify: password must be string".to_string());
        return Value::Null;
    };
    let Some(Value::String(hash_str)) = args.get(1) else {
        set_native_error("bcrypt.verify: hash must be string".to_string());
        return Value::Null;
    };
    match bcrypt::verify(password, hash_str) {
        Ok(ok) => Value::Bool(ok),
        Err(BcryptError::InvalidCost(_)) | Err(BcryptError::InvalidPrefix(_)) => Value::Bool(false),
        Err(e) => {
            set_native_error(format!("bcrypt.verify: {}", e));
            Value::Bool(false)
        }
    }
}

/// Constant-time equality for secret strings or raw bytes (HMAC tags, etc.).
pub fn native_crypto_secure_compare(args: &[Value]) -> Value {
    if args.len() < 2 {
        set_native_error("secure_compare: expected two arguments (bytes or string)".to_string());
        return Value::Null;
    }
    let a = &args[0];
    let b = &args[1];
    match (a, b) {
        (Value::String(sa), Value::String(sb)) => {
            Value::Bool(sa.as_bytes().ct_eq(sb.as_bytes()).into())
        }
        (Value::ByteBuffer(ba), Value::ByteBuffer(bb)) => {
            let xa = &ba.bytes[ba.offset..ba.offset + ba.len];
            let xb = &bb.bytes[bb.offset..bb.offset + bb.len];
            Value::Bool(xa.ct_eq(xb).into())
        }
        (Value::String(sa), Value::ByteBuffer(bb)) => {
            let xb = &bb.bytes[bb.offset..bb.offset + bb.len];
            Value::Bool(sa.as_bytes().ct_eq(xb).into())
        }
        (Value::ByteBuffer(ba), Value::String(sb)) => {
            let xa = &ba.bytes[ba.offset..ba.offset + ba.len];
            Value::Bool(xa.ct_eq(sb.as_bytes()).into())
        }
        _ => {
            set_native_error("secure_compare: arguments must be both string or both bytes".to_string());
            Value::Null
        }
    }
}

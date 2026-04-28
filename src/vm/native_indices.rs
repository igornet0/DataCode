//! Builtin native function indices — **must** stay in sync with
//! [`crate::vm::native_registry::register_builtin_natives`].
//!
//! Use these constants instead of raw `usize` literals in VM hot paths so refactors
//! and new builtins are less error-prone.

/// Indices for builtins registered in order by `register_builtin_natives`.
#[allow(dead_code)] // Not every index is referenced yet; reserved for documentation parity
pub mod builtin {
    pub const PRINT: usize = 0;
    pub const LEN: usize = 1;
    pub const RANGE: usize = 2;
    pub const INT: usize = 3;
    pub const FLOAT: usize = 4;
    pub const BOOL: usize = 5;
    pub const STR: usize = 6;
    pub const ARRAY: usize = 7;
    pub const TYPEOF: usize = 8;
    pub const ISINSTANCE: usize = 9;
    pub const DATE: usize = 10;
    pub const MONEY: usize = 11;
    pub const PATH: usize = 12;
    pub const PATH_NAME: usize = 13;
    pub const PATH_PARENT: usize = 14;
    pub const PATH_EXISTS: usize = 15;
    pub const PATH_IS_FILE: usize = 16;
    pub const PATH_IS_DIR: usize = 17;
    pub const PATH_EXTENSION: usize = 18;
    pub const PATH_STEM: usize = 19;
    pub const PATH_LEN: usize = 20;
    pub const ABS: usize = 21;
    pub const SQRT: usize = 22;
    pub const POW: usize = 23;
    pub const MIN: usize = 24;
    pub const MAX: usize = 25;
    pub const ROUND: usize = 26;
    pub const UPPER: usize = 27;
    pub const LOWER: usize = 28;
    pub const TRIM: usize = 29;
    pub const SPLIT: usize = 30;
    pub const JOIN: usize = 31;
    pub const CONTAINS: usize = 32;
    pub const ISUPPER: usize = 33;
    pub const ISLOWER: usize = 34;
    pub const PUSH: usize = 35;
    pub const POP: usize = 36;
    pub const UNIQUE: usize = 37;
    pub const REVERSE: usize = 38;
    pub const SORT: usize = 39;
    pub const SUM: usize = 40;
    pub const AVERAGE: usize = 41;
    pub const COUNT: usize = 42;
    pub const ANY: usize = 43;
    pub const ALL: usize = 44;
    pub const TABLE: usize = 45;
    pub const READ_FILE: usize = 46;
    pub const READ_FILE_BIN: usize = 47;
    pub const TABLE_INFO: usize = 48;
    pub const TABLE_HEAD: usize = 49;
    pub const TABLE_TAIL: usize = 50;
    pub const TABLE_SELECT: usize = 51;
    pub const TABLE_SORT: usize = 52;
    pub const TABLE_WHERE: usize = 53;
    pub const SHOW_TABLE: usize = 54;
    pub const MERGE_TABLES: usize = 55;
    pub const NOW: usize = 56;
    pub const GETCWD: usize = 57;
    pub const LIST_FILES: usize = 58;
    pub const INNER_JOIN: usize = 59;
    pub const LEFT_JOIN: usize = 60;
    pub const RIGHT_JOIN: usize = 61;
    pub const FULL_JOIN: usize = 62;
    pub const CROSS_JOIN: usize = 63;
    pub const SEMI_JOIN: usize = 64;
    pub const ANTI_JOIN: usize = 65;
    pub const ZIP_JOIN: usize = 66;
    pub const ASOF_JOIN: usize = 67;
    pub const APPLY_JOIN: usize = 68;
    pub const JOIN_ON: usize = 69;
    pub const TABLE_SUFFIXES: usize = 70;
    pub const RELATE: usize = 71;
    pub const PRIMARY_KEY: usize = 72;
    pub const ENUM: usize = 73;
    pub const TABLE_CLASS: usize = 74;
    pub const ARRAY_WITH_CAPACITY: usize = 75;
    pub const MAP: usize = 76;
    pub const FILTER: usize = 77;
    pub const REDUCE: usize = 78;
    pub const SHA256: usize = 79;
    pub const SHA512: usize = 80;
    pub const HMAC_SHA256: usize = 81;
    pub const HMAC_SHA512: usize = 82;
    pub const RANDOM_BYTES: usize = 83;
    pub const RANDOM_INT: usize = 84;
    /// `ValueError::new_1` — must match [`crate::vm::vm::Vm::VALUE_ERROR_NATIVE_INDEX`].
    pub const VALUE_ERROR: usize = 85;
    pub const CHUNK: usize = 86;
    pub const GENERATOR_FINAL: usize = 87;
    pub const GENERATOR_NEXT: usize = 88;
    pub const GENERATOR_SEND: usize = 89;
}

/// First `table(data, headers)` fast path in `native_call/execute.rs` (uses index `43`; registry slot `43` is `any` — kept for behavioral parity).
pub const TABLE_DATA_HEADERS_FAST_PATH_LEGACY: usize = 43;

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
    pub const CEIL: usize = 27;
    pub const FLOOR: usize = 28;
    pub const UPPER: usize = 29;
    pub const LOWER: usize = 30;
    pub const TRIM: usize = 31;
    pub const SPLIT: usize = 32;
    pub const JOIN: usize = 33;
    pub const CONTAINS: usize = 34;
    pub const STARTS_WITH: usize = 35;
    pub const ENDS_WITH: usize = 36;
    pub const ISUPPER: usize = 37;
    pub const ISLOWER: usize = 38;
    pub const REPLACE: usize = 39;
    pub const CAPITALIZE: usize = 40;
    pub const PUSH: usize = 41;
    pub const POP: usize = 42;
    pub const UNIQUE: usize = 43;
    pub const REVERSE: usize = 44;
    pub const SORT: usize = 45;
    pub const SUM: usize = 46;
    pub const AVERAGE: usize = 47;
    pub const COUNT: usize = 48;
    pub const ANY: usize = 49;
    pub const ALL: usize = 50;
    pub const TABLE: usize = 51;
    pub const READ_FILE: usize = 52;
    pub const READ_FILE_BIN: usize = 53;
    pub const TABLE_INFO: usize = 54;
    pub const TABLE_HEAD: usize = 55;
    pub const TABLE_TAIL: usize = 56;
    pub const TABLE_SELECT: usize = 57;
    pub const TABLE_SORT: usize = 58;
    pub const TABLE_WHERE: usize = 59;
    pub const TABLE_DROP_NULLS: usize = 60;
    pub const TABLE_REPLACE_NULLS: usize = 61;
    pub const TABLE_RENAME: usize = 62;
    pub const TABLE_DROP_COLUMN: usize = 63;
    pub const TABLE_ADD_COLUMN: usize = 64;
    pub const TABLE_MAP: usize = 65;
    pub const TABLE_SPLIT_COLUMN: usize = 66;
    pub const TABLE_JOIN_COLUMNS: usize = 67;
    pub const SHOW_TABLE: usize = 68;
    pub const MERGE_TABLES: usize = 69;
    pub const NOW: usize = 70;
    pub const GETCWD: usize = 71;
    pub const LIST_FILES: usize = 72;
    pub const INNER_JOIN: usize = 73;
    pub const LEFT_JOIN: usize = 74;
    pub const RIGHT_JOIN: usize = 75;
    pub const FULL_JOIN: usize = 76;
    pub const CROSS_JOIN: usize = 77;
    pub const SEMI_JOIN: usize = 78;
    pub const ANTI_JOIN: usize = 79;
    pub const ZIP_JOIN: usize = 80;
    pub const ASOF_JOIN: usize = 81;
    pub const APPLY_JOIN: usize = 82;
    pub const JOIN_ON: usize = 83;
    pub const TABLE_SUFFIXES: usize = 84;
    pub const RELATE: usize = 85;
    pub const PRIMARY_KEY: usize = 86;
    pub const ENUM: usize = 87;
    pub const TABLE_CLASS: usize = 88;
    pub const ARRAY_WITH_CAPACITY: usize = 89;
    pub const MAP: usize = 90;
    pub const FILTER: usize = 91;
    pub const REDUCE: usize = 92;
    pub const SHA256: usize = 93;
    pub const SHA512: usize = 94;
    pub const HMAC_SHA256: usize = 95;
    pub const HMAC_SHA512: usize = 96;
    pub const RANDOM_BYTES: usize = 97;
    pub const RANDOM_INT: usize = 98;
    pub const RANDOM_SEED: usize = 99;
    pub const RANDOM: usize = 100;
    pub const DATE_TO_UNIX: usize = 101;
    pub const PARSE_DATE: usize = 102;
    pub const FORMAT_DATE: usize = 103;
    pub const DURATION: usize = 104;
    /// `set()` / `set([...])` — global builtin.
    pub const SET: usize = 105;
    /// `divmod(a, b)` — global builtin.
    pub const DIVMOD: usize = 106;
    /// `isinf(x)` — global builtin.
    pub const ISINF: usize = 107;
    /// `copy(value)` — global builtin deep copy.
    pub const COPY: usize = 108;
    /// `ord(ch)` — global builtin Unicode code point.
    pub const ORD: usize = 109;
    pub const TABLE_ROW_NUMBER: usize = 110;
    pub const TABLE_DISTINCT: usize = 111;
    pub const TABLE_VALUE_MAP: usize = 112;
    pub const TABLE_AGGREGATE: usize = 113;
    pub const TABLE_AGGREGATE_GROUP: usize = 114;
    /// `archive(path)` — open archive file.
    pub const ARCHIVE: usize = 115;
    /// `datasource(config)` — create DataSource from config object.
    pub const DATASOURCE: usize = 116;
    /// `ValueError::new_1` — must match [`crate::vm::vm::Vm::VALUE_ERROR_NATIVE_INDEX`].
    pub const VALUE_ERROR: usize = 117;
    pub const CHUNK: usize = 118;
    pub const GENERATOR_FINAL: usize = 119;
    pub const GENERATOR_NEXT: usize = 120;
    pub const GENERATOR_SEND: usize = 121;
    /// `d.year()` — not a global; only via `GetArrayElement` on `Value::Date`.
    pub const DATE_YEAR: usize = 122;
    pub const DATE_MONTH: usize = 123;
    pub const DATE_DAY: usize = 124;
    pub const DATE_HOUR: usize = 125;
    pub const DATE_MINUTE: usize = 126;
    pub const DATE_SECOND: usize = 127;
    pub const DATE_TO_UTC: usize = 128;
    /// `set.add(x)` — only via `GetArrayElement` on `Value::Set` / `ValueCell::Set`.
    pub const SET_ADD: usize = 129;
    pub const SET_REMOVE: usize = 130;
    pub const SET_DISCARD: usize = 131;
    pub const SET_POP: usize = 132;
    pub const SET_CLEAR: usize = 133;
    pub const SET_COPY: usize = 134;
    pub const SET_UPDATE: usize = 135;
    pub const SET_CONTAINS: usize = 136;
    /// Plain bucket dict `obj.get(key, default=null)` — via `GetArrayElement("get")`; VM fast path in `native_call/execute.rs`.
    pub const OBJECT_GET: usize = 137;
    /// `table.add_row(row)` — only via `GetArrayElement` on `Value::Table`.
    pub const TABLE_ADD_ROW: usize = 138;
    /// Plain dict `obj.clear()` — via `GetArrayElement("clear")`; VM fast path in `object_clear_fast.rs`.
    pub const OBJECT_CLEAR: usize = 139;
    /// `table.push(data)` — via `GetArrayElement` on `Value::Table`.
    pub const TABLE_PUSH: usize = 140;
    pub const TABLE_SAVE_CSV: usize = 141;
    pub const TABLE_SAVE_SQLITE: usize = 142;
    /// Global `save()` via file_io (additive).
    pub const SAVE: usize = 143;
    /// `save_tables_sqlite(tables, filename, **kwargs)` — multi-table SQLite export.
    pub const SAVE_TABLES_SQLITE: usize = 144;
    /// `archive.read(path)` — via `GetArrayElement` on `Value::Archive`.
    pub const ARCHIVE_READ: usize = 145;
    pub const ARCHIVE_READ_TEXT: usize = 146;
    pub const ARCHIVE_EXTRACT: usize = 147;
    pub const ARCHIVE_CLOSE: usize = 148;
    /// `datasource.request(spec)` — via `GetArrayElement` on `Value::DataSource`.
    pub const DATASOURCE_REQUEST: usize = 149;
    pub const DATASOURCE_GET_TABLE: usize = 150;
    pub const DATASOURCE_SEND_TABLE: usize = 151;
    pub const DATASOURCE_CONNECT: usize = 152;
    pub const DATASOURCE_DISCONNECT: usize = 153;
    pub const DATASOURCE_PING: usize = 154;
    pub const DATASOURCE_TEST: usize = 155;
    pub const DATASOURCE_CLONE: usize = 156;
    /// `response.json()` — via `GetArrayElement` on `Value::DataSourceResponse`.
    pub const RESPONSE_JSON: usize = 157;
    pub const RESPONSE_TABLE: usize = 158;
    pub const RESPONSE_CSV: usize = 159;
    pub const RESPONSE_SAVE: usize = 160;
    pub const RESPONSE_SAVE_TEXT: usize = 161;
    pub const RESPONSE_SAVE_JSON: usize = 162;
    /// `column.map(fn)` — via `GetArrayElement` on `Value::ColumnReference`.
    pub const COLUMN_MAP: usize = 163;
    /// `columns.map(fn)` — via `GetArrayElement` on `Value::ColumnsReference`.
    pub const COLUMNS_MAP: usize = 164;
}

/// First `table(data, headers)` fast path in `native_call/execute.rs` (uses index `45`; registry slot `45` is `any`).
pub const TABLE_DATA_HEADERS_FAST_PATH_LEGACY: usize = 45;

/// Human-readable builtin name for VM profiling (`native#21` → `"abs"`).
pub fn builtin_native_name(index: usize) -> &'static str {
    const NAMES: [&str; builtin::COLUMNS_MAP + 1] = [
        "print",
        "len",
        "range",
        "int",
        "float",
        "bool",
        "str",
        "array",
        "typeof",
        "isinstance",
        "date",
        "money",
        "path",
        "path_name",
        "path_parent",
        "path_exists",
        "path_is_file",
        "path_is_dir",
        "path_extension",
        "path_stem",
        "path_len",
        "abs",
        "sqrt",
        "pow",
        "min",
        "max",
        "round",
        "ceil",
        "floor",
        "upper",
        "lower",
        "trim",
        "split",
        "join",
        "contains",
        "starts_with",
        "ends_with",
        "isupper",
        "islower",
        "replace",
        "capitalize",
        "push",
        "pop",
        "unique",
        "reverse",
        "sort",
        "sum",
        "average",
        "count",
        "any",
        "all",
        "table",
        "read_file",
        "read_file_bin",
        "table_info",
        "table_head",
        "table_tail",
        "table_select",
        "table_sort",
        "table_where",
        "table_drop_nulls",
        "table_replace_nulls",
        "table_rename",
        "table_drop_column",
        "table_add_column",
        "table_map",
        "table_split_column",
        "table_join_columns",
        "show_table",
        "merge_tables",
        "now",
        "getcwd",
        "list_files",
        "inner_join",
        "left_join",
        "right_join",
        "full_join",
        "cross_join",
        "semi_join",
        "anti_join",
        "zip_join",
        "asof_join",
        "apply_join",
        "join_on",
        "table_suffixes",
        "relate",
        "primary_key",
        "enum",
        "Table",
        "array_with_capacity",
        "map",
        "filter",
        "reduce",
        "sha256",
        "sha512",
        "hmac_sha256",
        "hmac_sha512",
        "random_bytes",
        "random_int",
        "random_seed",
        "random",
        "date_to_unix",
        "parse_date",
        "format_date",
        "duration",
        "set",
        "divmod",
        "isinf",
        "copy",
        "ord",
        "table_row_number",
        "table_distinct",
        "table_value_map",
        "table_aggregate",
        "table_aggregate_group",
        "archive",
        "datasource",
        "ValueError::new_1",
        "chunk",
        "generator.final",
        "generator.next",
        "generator.send",
        "date.year",
        "date.month",
        "date.day",
        "date.hour",
        "date.minute",
        "date.second",
        "date.to_utc",
        "set.add",
        "set.remove",
        "set.discard",
        "set.pop",
        "set.clear",
        "set.copy",
        "set.update",
        "set.contains",
        "object.get",
        "table.add_row",
        "object.clear",
        "table.push",
        "table.save_csv",
        "table.save_sqlite",
        "save",
        "save_tables_sqlite",
        "archive.read",
        "archive.read_text",
        "archive.extract",
        "archive.close",
        "datasource.request",
        "datasource.get_table",
        "datasource.send_table",
        "datasource.connect",
        "datasource.disconnect",
        "datasource.ping",
        "datasource.test",
        "datasource.clone",
        "response.json",
        "response.table",
        "response.csv",
        "response.save",
        "response.save_text",
        "response.save_json",
        "column.map",
        "columns.map",
    ];
    NAMES.get(index).copied().unwrap_or("native")
}

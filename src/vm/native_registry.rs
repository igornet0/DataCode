//! Registration of builtin native functions. Extracted from vm.rs for Phase 7 (VM Facade).
//! Registers via Host layer (HostEntry::Builtin) so VM does not depend on native implementations.

use crate::vm::host::{FnWrapper, HostEntry};
use crate::vm::native_indices::builtin;
use crate::vm::natives;
use crate::vm::natives::basic::{ArrayHostFunction, NativeGeneratorNext, NativeGeneratorSend};
use crate::vm::natives::higher_order::{FilterHostFunction, MapHostFunction, ReduceHostFunction};
use std::sync::Arc;

/// Fills `natives` with builtin native functions in the order expected by globals and executor.
pub fn register_builtin_natives(natives: &mut Vec<HostEntry>) {
    // Порядок важен — индексы должны соответствовать register_native_globals
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_print,
    )))); // 0 - print(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_len)))); // 1 - len(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_range,
    )))); // 2 - range(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_int)))); // 3 - int(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_float,
    )))); // 4 - float(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_bool,
    )))); // 5 - bool(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_str)))); // 6 - str(...)
    natives.push(HostEntry::Builtin(Arc::new(ArrayHostFunction))); // 7 - array(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_typeof,
    )))); // 8 - typeof(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_isinstance,
    )))); // 9 - isinstance(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_date,
    )))); // 10 - date(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_money,
    )))); // 11 - money(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_path,
    )))); // 12 - path(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_path_name,
    )))); // 13 - path_name(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_path_parent,
    )))); // 14 - path_parent(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_path_exists,
    )))); // 15 - path_exists(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_path_is_file,
    )))); // 16 - path_is_file(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_path_is_dir,
    )))); // 17 - path_is_dir(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_path_extension,
    )))); // 18 - path_extension(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_path_stem,
    )))); // 19 - path_stem(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_path_len,
    )))); // 20 - path_len(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_abs)))); // 21 - abs(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_sqrt,
    )))); // 22 - sqrt(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_pow)))); // 23 - pow(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_min)))); // 24 - min(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_max)))); // 25 - max(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_round,
    )))); // 26 - round(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_ceil,
    )))); // 27 - ceil(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_floor,
    )))); // 28 - floor(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_upper,
    )))); // 29 - upper(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_lower,
    )))); // 28 - lower(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_trim,
    )))); // 29 - trim(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_split,
    )))); // 30 - split(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_join,
    )))); // 31 - join(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_contains,
    )))); // 32 - contains(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_starts_with,
    )))); // 33 - starts_with(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_ends_with,
    )))); // 34 - ends_with(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_isupper,
    )))); // 35 - isupper(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_islower,
    )))); // 36 - islower(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_replace,
    )))); // 37 - replace(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_capitalize,
    )))); // 38 - capitalize(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_push,
    )))); // 39 - push(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_pop)))); // 38 - pop(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_unique,
    )))); // 39 - unique(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_reverse,
    )))); // 40 - reverse(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_sort,
    )))); // 39 - sort(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_sum)))); // 40 - sum(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_average,
    )))); // 41 - average(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_count,
    )))); // 42 - count(...)
    natives.push(HostEntry::Builtin(Arc::new(natives::AnyHostFunction))); // 43 - any(...)
    natives.push(HostEntry::Builtin(Arc::new(natives::AllHostFunction))); // 44 - all(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_table,
    )))); // 45 - table(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_read_file,
    )))); // 46 - read_file(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_read_file_bin,
    )))); // 47 - read_file_bin(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_table_info,
    )))); // 48 - table_info(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_table_head,
    )))); // 49 - table_head(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_table_tail,
    )))); // 50 - table_tail(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_table_select,
    )))); // 51 - table_select(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_table_sort,
    )))); // 52 - table_sort(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_table_where,
    )))); // 53 - table_where(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_table_drop_nulls,
    )))); // 54 - table_drop_nulls(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_table_replace_nulls,
    )))); // 55 - table_replace_nulls(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_table_rename,
    )))); // 56 - table_rename(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_table_drop_column,
    )))); // 57 - table_drop_column(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_table_add_column,
    )))); // 57 - table_add_column(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_table_map,
    )))); // 58 - table_map(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_table_split_column,
    )))); // 59 - table_split_column(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_table_join_columns,
    )))); // 60 - table_join_columns(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_show_table,
    )))); // 61 - show_table(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_merge_tables,
    )))); // 56 - merge_tables(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_now)))); // 57 - now(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_getcwd,
    )))); // 57 - getcwd(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_list_files,
    )))); // 58 - list_files(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_inner_join,
    )))); // 59 - inner_join(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_left_join,
    )))); // 60 - left_join(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_right_join,
    )))); // 61 - right_join(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_full_join,
    )))); // 62 - full_join(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_cross_join,
    )))); // 63 - cross_join(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_semi_join,
    )))); // 64 - semi_join(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_anti_join,
    )))); // 65 - anti_join(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_zip_join,
    )))); // 66 - zip_join(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_asof_join,
    )))); // 67 - asof_join(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_apply_join,
    )))); // 68 - apply_join(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_join_on,
    )))); // 69 - join_on(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_table_suffixes,
    )))); // 70 - table_suffixes(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_relate,
    )))); // 71 - relate(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_primary_key,
    )))); // 72 - primary_key(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_enum,
    )))); // 73 - enum(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_table_class,
    )))); // 74 - Table (built-in class for inheritance)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_array_with_capacity,
    )))); // 75 - array_with_capacity(...)
    natives.push(HostEntry::Builtin(Arc::new(MapHostFunction))); // 76 - map(...)
    natives.push(HostEntry::Builtin(Arc::new(FilterHostFunction))); // 77 - filter(...)
    natives.push(HostEntry::Builtin(Arc::new(ReduceHostFunction))); // 78 - reduce(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_sha256,
    )))); // 79 - sha256(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_sha512,
    )))); // 80 - sha512(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_hmac_sha256,
    )))); // 81 - hmac_sha256(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_hmac_sha512,
    )))); // 82 - hmac_sha512(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_random_bytes,
    )))); // 83 - random_bytes(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_random_int,
    )))); // 84 - random_int(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_random_seed,
    )))); // 85 - random_seed(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_random,
    )))); // 86 - random()
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_date_to_unix,
    )))); // 87 - date_to_unix(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_parse_date,
    )))); // 88 - parse_date(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_format_date,
    )))); // 89 - format_date(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_duration,
    )))); // 90 - duration(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_set,
    )))); // 89 - set(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_divmod,
    )))); // 90 - divmod(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_isinf,
    )))); // 91 - isinf(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_copy,
    )))); // 92 - copy(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_ord,
    )))); // 93 - ord(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_table_row_number,
    )))); // 94 - table_row_number(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_table_distinct,
    )))); // 95 - table_distinct(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_table_value_map,
    )))); // 96 - table_value_map(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_table_aggregate,
    )))); // 97 - table_aggregate(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_table_aggregate_group,
    )))); // 98 - table_aggregate_group(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        crate::archive::natives::native_archive,
    )))); // archive(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        crate::datasource::natives::native_datasource,
    )))); // datasource(config)
    let value_error = Arc::new(FnWrapper(natives::native_value_error_new));
    while natives.len() < builtin::VALUE_ERROR {
        natives.push(HostEntry::Builtin(value_error.clone())); // placeholder so indices line up
    }
    natives.push(HostEntry::Builtin(value_error)); // 94 - ValueError::new_1 for raise ValueError("...")
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_chunk,
    )))); // 95 - array.chunk(n)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_generator_final,
    )))); // 96 - generator.final()
    natives.push(HostEntry::Builtin(Arc::new(NativeGeneratorNext))); // 97 - generator.next()
    natives.push(HostEntry::Builtin(Arc::new(NativeGeneratorSend))); // 98 - generator.send()
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_date_year,
    )))); // 99 - date.year()
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_date_month,
    )))); // 100 - date.month()
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_date_day,
    )))); // 101 - date.day()
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_date_hour,
    )))); // 102 - date.hour()
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_date_minute,
    )))); // 103 - date.minute()
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_date_second,
    )))); // 104 - date.second()
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_date_to_utc,
    )))); // 105 - date.to_utc()
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_set_add,
    )))); // 106 - set.add() binding
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_set_remove,
    )))); // 107 - set.remove()
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_set_discard,
    )))); // 108 - set.discard()
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_set_pop,
    )))); // 109 - set.pop()
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_set_clear,
    )))); // 110 - set.clear()
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_set_copy,
    )))); // 111 - set.copy()
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_set_update,
    )))); // 112 - set.update()
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_set_contains,
    )))); // 113 - set.contains()
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_object_get,
    )))); // 114 - object.get(key [, default])
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_table_add_row,
    )))); // 115 - table.add_row(row)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_object_clear,
    )))); // 116 - dict.clear()
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_table_push,
    )))); // 117 - table.push(data)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_table_save_csv,
    )))); // 118 - table.save_csv(path)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_table_save_sqlite,
    )))); // 119 - table.save_sqlite(path)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::file_io_compat::native_save,
    )))); // 120 - save(...) file_io
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_save_tables_sqlite,
    )))); // 121 - save_tables_sqlite(tables, ...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        crate::archive::natives::native_archive_read,
    )))); // archive.read(path)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        crate::archive::natives::native_archive_read_text,
    )))); // archive.read_text(path)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        crate::archive::natives::native_archive_extract,
    )))); // archive.extract(dest)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        crate::archive::natives::native_archive_close,
    )))); // archive.close()
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        crate::datasource::natives::native_datasource_request,
    )))); // datasource.request(spec)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        crate::datasource::natives::native_datasource_get_table,
    )))); // datasource.get_table(spec)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        crate::datasource::natives::native_datasource_send_table,
    )))); // datasource.send_table(spec)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        crate::datasource::natives::native_datasource_connect,
    )))); // datasource.connect()
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        crate::datasource::natives::native_datasource_disconnect,
    )))); // datasource.disconnect()
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        crate::datasource::natives::native_datasource_ping,
    )))); // datasource.ping()
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        crate::datasource::natives::native_datasource_test,
    )))); // datasource.test()
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        crate::datasource::natives::native_datasource_clone,
    )))); // datasource.clone()
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        crate::datasource::natives::native_response_json,
    )))); // response.json()
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        crate::datasource::natives::native_response_table,
    )))); // response.table()
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        crate::datasource::natives::native_response_csv,
    )))); // response.csv()
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        crate::datasource::natives::native_response_save,
    )))); // response.save(path)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        crate::datasource::natives::native_response_save_text,
    )))); // response.save_text(path)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        crate::datasource::natives::native_response_save_json,
    )))); // response.save_json(path)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_column_map,
    )))); // column.map(fn)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_columns_map,
    )))); // columns.map(fn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_register_len_matches_native_indices_constants() {
        let mut natives = Vec::new();
        register_builtin_natives(&mut natives);
        let expected = crate::vm::native_indices::builtin::COLUMNS_MAP + 1;
        assert_eq!(
            natives.len(),
            expected,
            "builtin native count must match native_indices::builtin (last index COLUMNS_MAP)"
        );
    }
}

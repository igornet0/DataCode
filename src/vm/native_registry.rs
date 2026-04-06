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
        natives::native_upper,
    )))); // 27 - upper(...)
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
        natives::native_isupper,
    )))); // 33 - isupper(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_islower,
    )))); // 34 - islower(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_push,
    )))); // 35 - push(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_pop)))); // 36 - pop(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_unique,
    )))); // 37 - unique(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_reverse,
    )))); // 38 - reverse(...)
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
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_any)))); // 43 - any(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_all)))); // 44 - all(...)
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
        natives::native_show_table,
    )))); // 54 - show_table(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_merge_tables,
    )))); // 55 - merge_tables(...)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_now)))); // 56 - now(...)
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
    let value_error = Arc::new(FnWrapper(natives::native_value_error_new));
    while natives.len() < builtin::VALUE_ERROR {
        natives.push(HostEntry::Builtin(value_error.clone())); // placeholder so indices line up
    }
    natives.push(HostEntry::Builtin(value_error)); // 79 - ValueError::new_1 for raise ValueError("...")
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_chunk,
    )))); // 80 - array.chunk(n)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(
        natives::native_generator_final,
    )))); // 81 - generator.final()
    natives.push(HostEntry::Builtin(Arc::new(NativeGeneratorNext))); // 82 - generator.next()
    natives.push(HostEntry::Builtin(Arc::new(NativeGeneratorSend))); // 83 - generator.send()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_register_len_matches_native_indices_constants() {
        let mut natives = Vec::new();
        register_builtin_natives(&mut natives);
        let expected = crate::vm::native_indices::builtin::GENERATOR_SEND + 1;
        assert_eq!(
            natives.len(),
            expected,
            "builtin native count must match native_indices::builtin (last index GENERATOR_SEND)"
        );
    }
}

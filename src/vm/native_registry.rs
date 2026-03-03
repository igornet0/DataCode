//! Registration of builtin native functions. Extracted from vm.rs for Phase 7 (VM Facade).
//! Registers via Host layer (HostEntry::Builtin) so VM does not depend on native implementations.

use std::sync::Arc;
use crate::vm::host::{HostEntry, FnWrapper};
use crate::vm::natives;

/// ValueError::new_1 native index (must match VM's VALUE_ERROR_NATIVE_INDEX).
const VALUE_ERROR_NATIVE_INDEX: usize = 75;

/// Fills `natives` with builtin native functions in the order expected by globals and executor.
pub fn register_builtin_natives(natives: &mut Vec<HostEntry>) {
    // Порядок важен — индексы должны соответствовать register_native_globals
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_print))));      // 0
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_len))));        // 1
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_range))));      // 2
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_int))));        // 3
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_float))));      // 4
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_bool))));       // 5
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_str))));        // 6
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_array))));      // 7
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_typeof))));     // 8
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_isinstance)))); // 9
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_date))));       // 10
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_money))));      // 11
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_path))));       // 12
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_path_name))));  // 13
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_path_parent))));   // 14
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_path_exists))));   // 15
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_path_is_file))));  // 16
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_path_is_dir))));   // 17
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_path_extension)))); // 18
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_path_stem))));     // 19
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_path_len))));      // 20
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_abs))));       // 21
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_sqrt))));       // 22
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_pow))));        // 23
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_min))));        // 24
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_max))));        // 25
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_round))));      // 26
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_upper))));     // 27
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_lower))));      // 28
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_trim))));       // 29
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_split))));      // 30
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_join))));       // 31
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_contains))));   // 32
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_isupper))));    // 33
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_islower))));   // 34
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_push))));      // 35
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_pop))));       // 36
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_unique))));    // 37
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_reverse))));    // 38
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_sort))));       // 39
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_sum))));        // 40
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_average))));   // 41
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_count))));      // 42
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_any))));        // 43
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_all))));        // 44
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_table))));      // 45
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_read_file))));  // 46
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_table_info)))); // 47
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_table_head)))); // 48
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_table_tail)))); // 49
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_table_select)))); // 50
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_table_sort))));   // 51
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_table_where))));  // 52
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_show_table))));  // 53
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_merge_tables)))); // 54
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_now))));       // 55
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_getcwd))));     // 56
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_list_files)))); // 57
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_inner_join))));  // 58
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_left_join))));   // 59
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_right_join))));  // 60
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_full_join))));   // 61
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_cross_join))));   // 62
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_semi_join))));    // 63
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_anti_join))));   // 64
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_zip_join))));     // 65
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_asof_join))));    // 66
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_apply_join))));  // 67
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_join_on))));     // 68
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_table_suffixes)))); // 69
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_relate))));     // 70
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_primary_key)))); // 71
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_enum))));       // 72
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_table_class)))); // 73 - Table (built-in class for inheritance)
    natives.push(HostEntry::Builtin(Arc::new(FnWrapper(natives::native_array_with_capacity)))); // 74
    let value_error = Arc::new(FnWrapper(natives::native_value_error_new));
    while natives.len() < VALUE_ERROR_NATIVE_INDEX {
        natives.push(HostEntry::Builtin(value_error.clone())); // placeholder so indices line up
    }
    natives.push(HostEntry::Builtin(value_error)); // 75 - ValueError::new_1 for raise ValueError("...")
}

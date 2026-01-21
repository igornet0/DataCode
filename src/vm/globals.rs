// Global variables operations for VM

use crate::common::value::Value;

/// Регистрирует нативные функции в глобальных переменных
/// Порядок должен соответствовать register_natives() в vm.rs
pub fn register_native_globals(
    globals: &mut Vec<Value>,
    global_names: &std::collections::HashMap<usize, String>,
) {
    // Ensure globals vector is large enough for native functions (70) and any globals from compiler
    let max_global_index = global_names.keys().max().copied().unwrap_or(69);
    let min_size = (70).max(max_global_index + 1);
    globals.resize(min_size, Value::Null);
    
    globals[0] = Value::NativeFunction(0);  // print
    globals[1] = Value::NativeFunction(1);  // len
    globals[2] = Value::NativeFunction(2);  // range
    globals[3] = Value::NativeFunction(3);  // int
    globals[4] = Value::NativeFunction(4);  // float
    globals[5] = Value::NativeFunction(5);  // bool
    globals[6] = Value::NativeFunction(6);  // str
    globals[7] = Value::NativeFunction(7);  // array
    globals[8] = Value::NativeFunction(8);  // typeof
    globals[9] = Value::NativeFunction(9);  // isinstance
    globals[10] = Value::NativeFunction(10);  // date
    globals[11] = Value::NativeFunction(11);  // money
    globals[12] = Value::NativeFunction(12);  // path
    globals[13] = Value::NativeFunction(13);  // path_name
    globals[14] = Value::NativeFunction(14);  // path_parent
    globals[15] = Value::NativeFunction(15);  // path_exists
    globals[16] = Value::NativeFunction(16);  // path_is_file
    globals[17] = Value::NativeFunction(17);  // path_is_dir
    globals[18] = Value::NativeFunction(18);  // path_extension
    globals[19] = Value::NativeFunction(19);  // path_stem
    globals[20] = Value::NativeFunction(20);  // path_len
    // Математические функции
    globals[21] = Value::NativeFunction(21);  // abs
    globals[22] = Value::NativeFunction(22);  // sqrt
    globals[23] = Value::NativeFunction(23);  // pow
    globals[24] = Value::NativeFunction(24);  // min
    globals[25] = Value::NativeFunction(25);  // max
    globals[26] = Value::NativeFunction(26);  // round
    // Строковые функции
    globals[27] = Value::NativeFunction(27);  // upper
    globals[28] = Value::NativeFunction(28);  // lower
    globals[29] = Value::NativeFunction(29);  // trim
    globals[30] = Value::NativeFunction(30);  // split
    globals[31] = Value::NativeFunction(31);  // join
    globals[32] = Value::NativeFunction(32);  // contains
    // Функции массивов
    globals[33] = Value::NativeFunction(33);  // push
    globals[34] = Value::NativeFunction(34);  // pop
    globals[35] = Value::NativeFunction(35);  // unique
    globals[36] = Value::NativeFunction(36);  // reverse
    globals[37] = Value::NativeFunction(37);  // sort
    globals[38] = Value::NativeFunction(38);  // sum
    globals[39] = Value::NativeFunction(39);  // average
    globals[40] = Value::NativeFunction(40);  // count
    globals[41] = Value::NativeFunction(41);  // any
    globals[42] = Value::NativeFunction(42);  // all
    // Функции для работы с таблицами
    globals[43] = Value::NativeFunction(43);  // table
    globals[44] = Value::NativeFunction(44);  // read_file
    globals[45] = Value::NativeFunction(45);  // table_info
    globals[46] = Value::NativeFunction(46);  // table_head
    globals[47] = Value::NativeFunction(47);  // table_tail
    globals[48] = Value::NativeFunction(48);  // table_select
    globals[49] = Value::NativeFunction(49);  // table_sort
    globals[50] = Value::NativeFunction(50);  // table_where
    globals[51] = Value::NativeFunction(51);  // show_table
    globals[52] = Value::NativeFunction(52);  // merge_tables
    globals[53] = Value::NativeFunction(53);  // now
    globals[54] = Value::NativeFunction(54);  // getcwd
    globals[55] = Value::NativeFunction(55);  // list_files
    // JOIN операции
    globals[56] = Value::NativeFunction(56);  // inner_join
    globals[57] = Value::NativeFunction(57);  // left_join
    globals[58] = Value::NativeFunction(58);  // right_join
    globals[59] = Value::NativeFunction(59);  // full_join
    globals[60] = Value::NativeFunction(60);  // cross_join
    globals[61] = Value::NativeFunction(61);  // semi_join
    globals[62] = Value::NativeFunction(62);  // anti_join
    globals[63] = Value::NativeFunction(63);  // zip_join
    globals[64] = Value::NativeFunction(64);  // asof_join
    globals[65] = Value::NativeFunction(65);  // apply_join
    globals[66] = Value::NativeFunction(66);  // join_on
    globals[67] = Value::NativeFunction(67);  // table_suffixes
    globals[68] = Value::NativeFunction(68);  // relate
    globals[69] = Value::NativeFunction(69);  // primary_key
}

/// Заполняет имена глобальных переменных из chunk
pub fn merge_global_names(
    global_names: &mut std::collections::HashMap<usize, String>,
    explicit_global_names: &mut std::collections::HashMap<usize, String>,
    chunk_global_names: &std::collections::HashMap<usize, String>,
    chunk_explicit_global_names: &std::collections::HashMap<usize, String>,
) {
    // Merge with existing global_names instead of overwriting to preserve modules registered during execution
    for (idx, name) in chunk_global_names {
        global_names.insert(*idx, name.clone());
    }
    for (idx, name) in chunk_explicit_global_names {
        explicit_global_names.insert(*idx, name.clone());
    }
}
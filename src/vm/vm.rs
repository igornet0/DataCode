// Виртуальная машина

use crate::abi::NativeAbiFn;
use crate::debug_println;
use crate::bytecode::Chunk;
use crate::common::{error::LangError, value::Value};
use crate::vm::frame::CallFrame;
use crate::vm::natives;
use crate::vm::types::{ExplicitRelation, ExplicitPrimaryKey, VMStatus};
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::globals;
use crate::vm::calls;
use crate::vm::executor;
use libloading::Library;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;

pub type NativeFn = fn(&[Value]) -> Value;

// Thread-local storage для хранения контекста VM во время вызова нативных функций
// Это позволяет нативным функциям вызывать пользовательские функции
thread_local! {
    pub(crate) static VM_CALL_CONTEXT: RefCell<Option<*mut Vm>> = RefCell::new(None);
}

pub struct Vm {
    stack: Vec<Value>,
    frames: Vec<CallFrame>,
    globals: Vec<Value>,
    functions: Vec<crate::bytecode::Function>,
    natives: Vec<NativeFn>,
    exception_handlers: Vec<ExceptionHandler>, // Стек обработчиков исключений
    error_type_table: Vec<String>, // Таблица типов ошибок для текущей функции
    global_names: std::collections::HashMap<usize, String>, // Маппинг индексов глобальных переменных на их имена
    explicit_global_names: std::collections::HashMap<usize, String>, // Маппинг индексов переменных, явно объявленных с ключевым словом 'global'
    explicit_relations: Vec<ExplicitRelation>, // Явные связи, созданные через relate()
    explicit_primary_keys: Vec<ExplicitPrimaryKey>, // Явные первичные ключи, созданные через primary_key()
    loaded_modules: std::collections::HashSet<String>, // Множество загруженных модулей
    abi_natives: Vec<NativeAbiFn>, // Нативы из загруженных ABI-модулей
    loaded_native_libraries: Vec<Library>, // Библиотеки .so/.dylib, чтобы символы оставались валидными
    /// Base path for resolving relative paths (e.g. in settings_env load_env). Set before run() so thread-local is reinforced.
    base_path: Option<PathBuf>,
}

impl Vm {
    pub fn new() -> Self {
        let mut vm = Self {
            stack: Vec::new(),
            frames: Vec::new(),
            globals: Vec::new(),
            functions: Vec::new(),
            natives: Vec::new(),
            exception_handlers: Vec::new(),
            error_type_table: Vec::new(),
            global_names: std::collections::HashMap::new(),
            explicit_global_names: std::collections::HashMap::new(),
            explicit_relations: Vec::new(),
            explicit_primary_keys: Vec::new(),
            loaded_modules: std::collections::HashSet::new(),
            abi_natives: Vec::new(),
            loaded_native_libraries: Vec::new(),
            base_path: None,
        };
        vm.register_natives();
        vm
    }

    /// Set base path for resolving relative paths (e.g. in settings_env). Used at start of run() to set thread-local.
    pub fn set_base_path(&mut self, path: Option<PathBuf>) {
        self.base_path = path;
    }

    /// Base path for resolving relative paths (imports, load_env). Used by executor when thread-local may not be set.
    pub fn get_base_path(&self) -> Option<PathBuf> {
        self.base_path.clone()
    }

    fn register_natives(&mut self) {
        // Регистрируем нативные функции
        // Порядок важен - индексы должны соответствовать register_native_globals
        self.natives.push(natives::native_print);      // 0
        self.natives.push(natives::native_len);        // 1
        self.natives.push(natives::native_range);      // 2
        self.natives.push(natives::native_int);        // 3
        self.natives.push(natives::native_float);        // 4
        self.natives.push(natives::native_bool);        // 5
        self.natives.push(natives::native_str);        // 6
        self.natives.push(natives::native_array);      // 7
        self.natives.push(natives::native_typeof);     // 8
        self.natives.push(natives::native_isinstance); // 9
        self.natives.push(natives::native_date);      // 10
        self.natives.push(natives::native_money);     // 11
        self.natives.push(natives::native_path);     // 12
        self.natives.push(natives::native_path_name);     // 13
        self.natives.push(natives::native_path_parent);   // 14
        self.natives.push(natives::native_path_exists);   // 15
        self.natives.push(natives::native_path_is_file); // 16
        self.natives.push(natives::native_path_is_dir);  // 17
        self.natives.push(natives::native_path_extension); // 18
        self.natives.push(natives::native_path_stem);    // 19
        self.natives.push(natives::native_path_len);     // 20
        // Математические функции
        self.natives.push(natives::native_abs);          // 21
        self.natives.push(natives::native_sqrt);         // 22
        self.natives.push(natives::native_pow);          // 23
        self.natives.push(natives::native_min);          // 24
        self.natives.push(natives::native_max);          // 25
        self.natives.push(natives::native_round);        // 26
        // Строковые функции
        self.natives.push(natives::native_upper);        // 27
        self.natives.push(natives::native_lower);        // 28
        self.natives.push(natives::native_trim);         // 29
        self.natives.push(natives::native_split);        // 30
        self.natives.push(natives::native_join);         // 31
        self.natives.push(natives::native_contains);     // 32
        // Функции массивов
        self.natives.push(natives::native_push);         // 33
        self.natives.push(natives::native_pop);          // 34
        self.natives.push(natives::native_unique);       // 35
        self.natives.push(natives::native_reverse);      // 36
        self.natives.push(natives::native_sort);        // 37
        self.natives.push(natives::native_sum);          // 38
        self.natives.push(natives::native_average);     // 39
        self.natives.push(natives::native_count);        // 40
        self.natives.push(natives::native_any);          // 41
        self.natives.push(natives::native_all);          // 42
        // Функции для работы с таблицами
        self.natives.push(natives::native_table);        // 43
        self.natives.push(natives::native_read_file);    // 44
        self.natives.push(natives::native_table_info);   // 45
        self.natives.push(natives::native_table_head);   // 46
        self.natives.push(natives::native_table_tail);   // 47
        self.natives.push(natives::native_table_select); // 48
        self.natives.push(natives::native_table_sort);   // 49
        self.natives.push(natives::native_table_where);  // 50
        self.natives.push(natives::native_show_table);   // 51
        self.natives.push(natives::native_merge_tables); // 52
        self.natives.push(natives::native_now);          // 53
        self.natives.push(natives::native_getcwd);       // 54
        self.natives.push(natives::native_list_files);   // 55
        // JOIN операции
        self.natives.push(natives::native_inner_join);   // 56
        self.natives.push(natives::native_left_join);    // 57
        self.natives.push(natives::native_right_join);  // 58
        self.natives.push(natives::native_full_join);    // 59
        self.natives.push(natives::native_cross_join);   // 60
        self.natives.push(natives::native_semi_join);   // 61
        self.natives.push(natives::native_anti_join);   // 62
        self.natives.push(natives::native_zip_join);    // 63
        self.natives.push(natives::native_asof_join);   // 64
        self.natives.push(natives::native_apply_join);   // 65
        self.natives.push(natives::native_join_on);     // 66
        self.natives.push(natives::native_table_suffixes); // 67
        self.natives.push(natives::native_relate);      // 68
        self.natives.push(natives::native_primary_key); // 69
    }

    /// Добавляет в VM слоты для всех имён из chunk.global_names, которых ещё нет в VM.
    /// Итерация по отсортированным индексам, чтобы порядок слотов был детерминированным.
    /// Нужно вызывать перед update_chunk_indices, чтобы главный chunk мог ссылаться на __main__.
    pub fn ensure_globals_from_chunk(&mut self, chunk: &crate::bytecode::Chunk) {
        let mut entries: Vec<_> = chunk.global_names.iter().collect();
        entries.sort_by_key(|(idx, _)| *idx);
        for (_old_idx, name) in entries {
            let exists = self.global_names.iter().any(|(_, n)| n == name);
            if !exists {
                let new_idx = self.globals.len();
                self.globals.push(Value::Null);
                self.global_names.insert(new_idx, name.clone());
                debug_println!("[DEBUG ensure_globals_from_chunk] Добавлен слот для '{}' в globals[{}]", name, new_idx);
                if name == "Config" || name == "DatabaseConfig" {
                    debug_println!("[DEBUG ensure_globals_from_chunk] Config/DatabaseConfig: '{}' -> слот {}", name, new_idx);
                }
            }
        }
    }

    /// Как ensure_globals_from_chunk, но сохраняет индексы из chunk: слоты создаются по chunk.global_names (idx, name),
    /// с расширением globals при необходимости. Нужно для модулей (file_import), чтобы main chunk без патча писал по тем же индексам.
    pub fn ensure_globals_from_chunk_preserve_indices(&mut self, chunk: &crate::bytecode::Chunk) {
        let mut entries: Vec<_> = chunk.global_names.iter().map(|(i, n)| (*i, n.clone())).collect();
        entries.sort_by_key(|(idx, _)| *idx);
        for (idx, name) in entries {
            let exists = self.global_names.iter().any(|(_, n)| n == &name);
            if !exists {
                if idx >= self.globals.len() {
                    self.globals.resize(idx + 1, Value::Null);
                }
                self.global_names.insert(idx, name.clone());
                debug_println!("[DEBUG ensure_globals_from_chunk_preserve_indices] Добавлен слот для '{}' в globals[{}]", name, idx);
                if name == "Config" || name == "DatabaseConfig" {
                    debug_println!("[DEBUG ensure_globals_from_chunk_preserve_indices] Config/DatabaseConfig: '{}' -> слот {}", name, idx);
                }
            }
        }
    }

    /// Обновляет индексы глобальных переменных в chunk на основе реальных индексов в VM
    pub fn update_chunk_indices(&self, chunk: &mut crate::bytecode::Chunk) {
        Self::update_chunk_indices_from_names(chunk, &self.global_names, Some(&self.globals));
    }

    /// Sentinel index for model_config class load in Settings subclass constructors (must match compiler).
    const MODEL_CONFIG_CLASS_LOAD_INDEX: usize = 0x0FFF_FFFF;

    /// Обновляет индексы глобалов в chunk по переданной карте имён (для использования без &self, чтобы избежать конфликта заимствований).
    /// Двухфазный алгоритм: сначала строим маппинг old_idx → real_idx, затем один проход по байткоду.
    /// Это устраняет перезапись уже исправленных инструкций при недетерминированном порядке итерации по global_names.
    /// Для model_config (sentinel 0x0FFF_FFFF) явно разрешаем по имени из chunk.global_names и берём минимальный индекс в main.
    /// Если передан globals_for_verify, после патча sentinel проверяем, что в слоте лежит класс с ожидаемым __class_name; при несовпадении — debug-предупреждение.
    pub fn update_chunk_indices_from_names(
        chunk: &mut crate::bytecode::Chunk,
        global_names: &std::collections::HashMap<usize, String>,
        globals_for_verify: Option<&[Value]>,
    ) {
        // Phase 1: Build old_idx → real_idx mapping (no bytecode changes).
        // Resolve real_idx by name deterministically (min index if multiple) for stability.
        // Iterate in deterministic order (by name then index) to avoid HashMap iteration order affecting outcome.
        let mut old_to_real: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        let mut name_index_pairs: Vec<_> = chunk.global_names.iter().map(|(i, n)| (*i, n.clone())).collect();
        name_index_pairs.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        for (old_idx, name) in &name_index_pairs {
            let matching_indices: Vec<usize> = global_names
                .iter()
                .filter(|(_, n)| *n == name)
                .map(|(idx, _)| *idx)
                .collect();
            if let Some(&real_idx) = matching_indices.iter().min() {
                if *old_idx != real_idx {
                    old_to_real.insert(*old_idx, real_idx);
                    debug_println!("[DEBUG update_chunk_indices] Маппинг '{}': {} -> {}", name, old_idx, real_idx);
                }
            }
        }
        // Всегда явно привязать model_config (sentinel) к имени класса: при наличии LoadGlobal(sentinel) разрешаем по имени из chunk.global_names и перезаписываем маппинг, чтобы не зависеть от порядка итерации HashMap.
        let needs_sentinel = chunk.code.iter().any(|op| {
            matches!(op, crate::bytecode::OpCode::LoadGlobal(i) if *i == Self::MODEL_CONFIG_CLASS_LOAD_INDEX)
        });
        if needs_sentinel {
            if let Some(name) = chunk.global_names.get(&Self::MODEL_CONFIG_CLASS_LOAD_INDEX) {
                let matching_indices: Vec<usize> = global_names
                    .iter()
                    .filter(|(_, n)| *n == name)
                    .map(|(idx, _)| *idx)
                    .collect();
                if let Some(&real_idx) = matching_indices.iter().min() {
                    old_to_real.insert(Self::MODEL_CONFIG_CLASS_LOAD_INDEX, real_idx);
                    debug_println!("[DEBUG update_chunk_indices] Маппинг model_config (sentinel) класс '{}': sentinel -> globals[{}]", name, real_idx);
                    // Опциональная проверка: слот должен содержать класс с ожидаемым именем
                    if let Some(globals) = globals_for_verify {
                        if real_idx < globals.len() {
                            let slot_type = match &globals[real_idx] {
                                Value::Object(_) => "Object",
                                Value::Function(_) => "Function",
                                Value::Null => "Null",
                                _ => "Other",
                            };
                            debug_println!("[DEBUG update_chunk_indices] sentinel '{}' -> globals[{}], значение в слоте: {}", name, real_idx, slot_type);
                            if let Value::Object(obj_rc) = &globals[real_idx] {
                                let obj = obj_rc.borrow();
                                if let Some(Value::String(actual_name)) = obj.get("__class_name") {
                                    debug_println!("[DEBUG update_chunk_indices] globals[{}].__class_name = '{}'", real_idx, actual_name);
                                    if actual_name != name {
                                        debug_println!(
                                            "[DEBUG update_chunk_indices] WARNING: ожидался класс '{}', в слоте {} — класс '{}'",
                                            name, real_idx, actual_name
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Phase 2: Single pass over bytecode; use original idx from instructions only (no overwriting of already-patched).
        let mut updated_count = 0usize;
        for opcode in &mut chunk.code {
            match opcode {
                crate::bytecode::OpCode::LoadGlobal(idx) => {
                    if let Some(&real) = old_to_real.get(idx) {
                        *opcode = crate::bytecode::OpCode::LoadGlobal(real);
                        updated_count += 1;
                    }
                }
                crate::bytecode::OpCode::StoreGlobal(idx) => {
                    if let Some(&real) = old_to_real.get(idx) {
                        *opcode = crate::bytecode::OpCode::StoreGlobal(real);
                        updated_count += 1;
                    }
                }
                _ => {}
            }
        }
        if !old_to_real.is_empty() {
            debug_println!("[DEBUG update_chunk_indices] Обновлено {} инструкций по {} маппингам", updated_count, old_to_real.len());
        }

        // Phase 3: Update chunk.global_names. Collect (real_idx, name) first so we don't overwrite
        // or remove entries that are still needed when mappings overlap (e.g. 71->70, 72->71, 73->72).
        let to_insert: Vec<_> = old_to_real
            .iter()
            .filter_map(|(old_idx, real_idx)| {
                chunk.global_names.get(old_idx).map(|name| (*real_idx, name.clone()))
            })
            .collect();
        for old_idx in old_to_real.keys() {
            chunk.global_names.remove(old_idx);
        }
        for (real_idx, name) in to_insert {
            chunk.global_names.insert(real_idx, name);
        }
    }

    /// Находит индекс функции в списке функций VM по имени функции
    fn find_function_index_by_name(&self, function_name: &str) -> Option<usize> {
        self.functions.iter()
            .position(|f| f.name == function_name)
    }

    /// Устанавливает функции в VM. Если передан main_chunk, маппинг глобалов строится по нему
    /// (для run() главный chunk не входит в functions; для run_with_vm_internal_with_args можно передать None).
    /// Используется маппинг по имени (name -> new_idx), чтобы один и тот же old_idx в разных чанках
    /// с разными именами (например 72 = "config" в main и 72 = "DatabaseConfig" в конструкторе) патчился в разные слоты.
    pub fn set_functions(&mut self, functions: Vec<crate::bytecode::Function>, main_chunk: Option<&mut crate::bytecode::Chunk>) {
        let mut explicit_global_names_to_add = std::collections::HashMap::new();

        // First pass: collect all (old_idx, name) from main chunk + all function chunks (no overwrite by index).
        let mut all_pairs: Vec<(usize, String)> = Vec::new();
        if let Some(mc) = main_chunk.as_ref() {
            for (idx, name) in &mc.global_names {
                all_pairs.push((*idx, name.clone()));
            }
        }
        for function in &functions {
            for (idx, name) in &function.chunk.global_names {
                all_pairs.push((*idx, name.clone()));
            }
        }
        // Unique names in deterministic order (sort by name).
        let mut unique_names: Vec<String> = all_pairs.iter().map(|(_, n)| n.clone()).collect();
        unique_names.sort();
        unique_names.dedup();

        // Build name -> new_idx: each distinct name gets one slot in the VM.
        // Встроенные имена (sum, count, ...) всегда мапятся на канонический индекс 0..69.
        let mut name_to_new_idx: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for name in &unique_names {
            let new_idx = if let Some(builtin_idx) = globals::builtin_global_index(name) {
                debug_println!("[DEBUG set_functions] Встроенное имя '{}' -> канонический индекс {}", name, builtin_idx);
                builtin_idx
            } else {
                let matching_indices: Vec<usize> = self.global_names.iter()
                    .filter(|(_, n)| *n == name)
                    .map(|(idx, _)| *idx)
                    .collect();
                if let Some(&real_idx) = matching_indices.iter().min() {
                    debug_println!("[DEBUG set_functions] Найдена переменная '{}' в VM с индексом {} (детерминированный min)", name, real_idx);
                    if name == "Config" || name == "DatabaseConfig" {
                        debug_println!("[DEBUG set_functions] Config/DatabaseConfig: '{}' -> индекс {} (найден в self.global_names)", name, real_idx);
                    }
                    real_idx
                } else {
                    let new_idx = self.globals.len();
                    self.globals.push(Value::Null);
                    self.global_names.insert(new_idx, name.clone());
                    debug_println!("[DEBUG set_functions] Переменная '{}' не найдена в VM, создаем новый индекс {}", name, new_idx);
                    if name == "Config" || name == "DatabaseConfig" {
                        debug_println!("[DEBUG set_functions] Config/DatabaseConfig: '{}' -> индекс {} (создан новый слот)", name, new_idx);
                    }
                    new_idx
                }
            };
            name_to_new_idx.insert(name.clone(), new_idx);
        }

        if main_chunk.is_none() {
            if let Some(main_function) = functions.first() {
                for (idx, name) in &main_function.chunk.explicit_global_names {
                    explicit_global_names_to_add.insert(*idx, name.clone());
                }
            }
        }

        // Patch main chunk: один проход по байткоду с маппингом old_idx -> new_idx, чтобы уже пропатченные индексы не перезаписывались.
        if let Some(mc) = main_chunk {
            let old_to_new: std::collections::HashMap<usize, usize> = mc
                .global_names
                .iter()
                .filter_map(|(old_idx, name)| {
                    name_to_new_idx.get(name).and_then(|&new_idx| {
                        if *old_idx != new_idx {
                            Some((*old_idx, new_idx))
                        } else {
                            None
                        }
                    })
                })
                .collect();
            for opcode in &mut mc.code {
                match opcode {
                    crate::bytecode::OpCode::LoadGlobal(idx) => {
                        if let Some(&new_idx) = old_to_new.get(idx) {
                            *opcode = crate::bytecode::OpCode::LoadGlobal(new_idx);
                        }
                    }
                    crate::bytecode::OpCode::StoreGlobal(idx) => {
                        if let Some(&new_idx) = old_to_new.get(idx) {
                            *opcode = crate::bytecode::OpCode::StoreGlobal(new_idx);
                        }
                    }
                    _ => {}
                }
            }
            for (old_idx, new_idx) in &old_to_new {
                if let Some(name) = mc.global_names.remove(old_idx) {
                    mc.global_names.insert(*new_idx, name);
                }
            }
        }

        // Patch each function chunk: один проход с маппингом old_idx -> new_idx.
        // Sentinel (model_config) must resolve from VM's global_names so it points to the actual class slot (e.g. 71 from ensure/merge), not name_to_new_idx (which can allocate a new slot 74 when "Config" wasn't found yet).
        let existing_functions_count = self.functions.len();
        let mut updated_functions = functions;
        for function in &mut updated_functions {
            let old_to_new: std::collections::HashMap<usize, usize> = function
                .chunk
                .global_names
                .iter()
                .filter_map(|(old_idx, name)| {
                    let new_idx = if *old_idx == Self::MODEL_CONFIG_CLASS_LOAD_INDEX {
                        // Resolve sentinel by name from VM's current global_names so we patch to the real class slot.
                        let matching: Vec<usize> = self.global_names.iter()
                            .filter(|(_, n)| *n == name)
                            .map(|(idx, _)| *idx)
                            .collect();
                        if let Some(&real_idx) = matching.iter().min() {
                            debug_println!(
                                "[DEBUG set_functions] sentinel функция '{}' класс '{}' -> globals[{}] (из VM global_names)",
                                function.name, name, real_idx
                            );
                            Some(real_idx)
                        } else {
                            let fallback = name_to_new_idx.get(name).copied();
                            debug_println!(
                                "[DEBUG set_functions] WARNING: sentinel функция '{}' класс '{}' не найден в self.global_names, fallback name_to_new_idx -> {:?}",
                                function.name, name, fallback
                            );
                            fallback
                        }
                    } else {
                        name_to_new_idx.get(name).copied()
                    };
                    new_idx.and_then(|new_idx| {
                        if *old_idx != new_idx {
                            Some((*old_idx, new_idx))
                        } else {
                            None
                        }
                    })
                })
                .collect();
            for opcode in &mut function.chunk.code {
                match opcode {
                    crate::bytecode::OpCode::LoadGlobal(idx) => {
                        if let Some(&new_idx) = old_to_new.get(idx) {
                            *opcode = crate::bytecode::OpCode::LoadGlobal(new_idx);
                        }
                    }
                    crate::bytecode::OpCode::StoreGlobal(idx) => {
                        if let Some(&new_idx) = old_to_new.get(idx) {
                            *opcode = crate::bytecode::OpCode::StoreGlobal(new_idx);
                        }
                    }
                    _ => {}
                }
            }
            for (old_idx, new_idx) in &old_to_new {
                if let Some(name) = function.chunk.global_names.remove(old_idx) {
                    function.chunk.global_names.insert(*new_idx, name);
                }
            }
        }
        
        // Добавляем новые функции к существующим, а не заменяем их
        // Это важно, потому что функции из __lib__.dc уже были добавлены через merge_globals_from
        debug_println!("[DEBUG set_functions] Добавляем {} новых функций к существующим {} функциям", updated_functions.len(), existing_functions_count);
        
        // Отладочный вывод: показываем все Call инструкции в функциях после обновления
        for function in &updated_functions {
            debug_println!("[DEBUG set_functions] Проверка Call инструкций в функции '{}':", function.name);
            for (ip, opcode) in function.chunk.code.iter().enumerate() {
                if let crate::bytecode::OpCode::Call(arity) = opcode {
                    debug_println!("[DEBUG set_functions]   IP {}: Call({})", ip, arity);
                }
            }
        }
        
        // ВАЖНО: После добавления функций из основного скрипта, нужно обновить индексы функций
        // в глобальных переменных, потому что функции из __lib__.dc были добавлены первыми,
        // и их индексы могли измениться.
        
        // Собираем уникальные имена из chunk'ов новых функций. Итерация в детерминированном порядке (сортировка по имени),
        // чтобы слот для каждого имени всегда определялся через name_to_new_idx одинаково.
        let mut names_from_chunks: std::collections::HashSet<String> = std::collections::HashSet::new();
        for function in &updated_functions {
            for (_, name) in &function.chunk.global_names {
                names_from_chunks.insert(name.clone());
            }
        }
        let mut names_sorted: Vec<String> = names_from_chunks.into_iter().collect();
        names_sorted.sort();
        let chunk_global_names: Vec<(usize, String)> = names_sorted
            .iter()
            .filter_map(|name| name_to_new_idx.get(name).map(|&idx| (idx, name.clone())))
            .collect();
        
        self.functions.extend(updated_functions);
        
        // Обновляем индексы функций в глобальных переменных по имени и name_to_new_idx (детерминированный слот).
        debug_println!("[DEBUG set_functions] Начинаем обновление индексов функций в globals. Всего глобальных переменных в VM: {}, имён из chunk: {}", self.global_names.len(), chunk_global_names.len());
        
        for (real_global_idx, name) in &chunk_global_names {
            debug_println!("[DEBUG set_functions] Обрабатываем '{}' из chunk -> слот {}", name, real_global_idx);
            if *real_global_idx < self.globals.len() {
                debug_println!("[DEBUG set_functions] Проверяем globals[{}] для '{}'", real_global_idx, name);
                if let Value::Function(old_fn_idx) = &self.globals[*real_global_idx] {
                    debug_println!("[DEBUG set_functions] Найдена функция '{}' в globals[{}] (из chunk) с индексом функции {}", name, real_global_idx, old_fn_idx);
                    
                    // Находим функцию по имени в списке функций VM
                    if let Some(new_fn_idx) = self.find_function_index_by_name(name) {
                        if *old_fn_idx != new_fn_idx {
                            debug_println!("[DEBUG set_functions] Обновляем индекс функции для '{}' в globals[{}]: {} -> {}", 
                                name, real_global_idx, old_fn_idx, new_fn_idx);
                            self.globals[*real_global_idx] = Value::Function(new_fn_idx);
                        } else {
                            debug_println!("[DEBUG set_functions] Индекс функции для '{}' уже правильный: {}", name, new_fn_idx);
                        }
                    } else {
                        debug_println!("[DEBUG set_functions] WARNING: Функция '{}' не найдена в списке функций VM", name);
                    }
                } else {
                    // Слот Null или другой тип — если в VM есть функция с таким именем (например, конструктор из chunk), заполняем слот
                    if let Some(fn_idx) = self.find_function_index_by_name(name) {
                        self.globals[*real_global_idx] = Value::Function(fn_idx);
                        debug_println!("[DEBUG set_functions] Установлена функция '{}' в globals[{}] = Value::Function({}) (слот был Null/другой)", name, real_global_idx, fn_idx);
                    } else {
                        debug_println!("[DEBUG set_functions] globals[{}] для '{}' не является функцией: {:?}", real_global_idx, name, self.globals[*real_global_idx]);
                    }
                }
            } else {
                debug_println!("[DEBUG set_functions] WARNING: Индекс {} выходит за границы globals (всего: {})", real_global_idx, self.globals.len());
            }
        }
        
        // Также обрабатываем все функции, которые были только что добавлены
        // Это нужно для функций, которые не были добавлены в global_names (например, __main__)
        // Итерируемся по функциям, которые были добавлены в этом вызове set_functions
        let start_fn_idx = existing_functions_count;
        for new_fn_idx in start_fn_idx..self.functions.len() {
            let function_name = &self.functions[new_fn_idx].name;
            debug_println!("[DEBUG set_functions] Проверяем функцию '{}' с индексом {} (только что добавлена, compiler_idx: {})", 
                function_name, new_fn_idx, new_fn_idx - existing_functions_count);
            
            // Старый индекс функции в компиляторе равен (new_fn_idx - existing_functions_count)
            let compiler_fn_idx = new_fn_idx - existing_functions_count;
            
            // Ищем эту функцию в globals
            // Сначала проверяем, есть ли она в chunk's global_names
            let mut found_in_chunk = false;
            for function in &self.functions[start_fn_idx..] {
                if let Some(global_idx) = function.chunk.global_names.iter()
                    .find(|(_, name)| *name == function_name)
                    .map(|(idx, _)| *idx) {
                    // Нашли в chunk's global_names, используем mapped index
                    let real_global_idx = global_idx;
                    if real_global_idx < self.globals.len() {
                        if let Value::Function(old_fn_idx) = &self.globals[real_global_idx] {
                            if *old_fn_idx != new_fn_idx {
                                debug_println!("[DEBUG set_functions] Найдена функция '{}' в globals[{}] через chunk global_names: {} -> {}", 
                                    function_name, real_global_idx, old_fn_idx, new_fn_idx);
                                self.globals[real_global_idx] = Value::Function(new_fn_idx);
                                found_in_chunk = true;
                                break;
                            }
                        }
                    }
                }
            }
            
            // Если не нашли в chunk, ищем во всех globals по старому индексу компилятора
            // ИЛИ по имени функции (если старая функция с индексом old_fn_idx имеет то же имя)
            if !found_in_chunk {
                debug_println!("[DEBUG set_functions] Ищем функцию '{}' во всех globals (compiler_idx: {})", function_name, compiler_fn_idx);
                for global_idx in 0..self.globals.len() {
                    if let Value::Function(old_fn_idx) = &self.globals[global_idx] {
                        // Проверяем, не обработали ли мы уже эту функцию через global_names
                        let already_processed = self.global_names.get(&global_idx)
                            .map(|name| name == function_name)
                            .unwrap_or(false);
                        
                        if !already_processed && *old_fn_idx == compiler_fn_idx {
                            debug_println!("[DEBUG set_functions] Найден кандидат для '{}' в globals[{}] с old_fn_idx={}, compiler_idx={}", 
                                function_name, global_idx, old_fn_idx, compiler_fn_idx);
                        }
                        
                        if !already_processed {
                            // Проверяем три случая:
                            // 1. Старый индекс соответствует индексу функции в компиляторе
                            //    И функция с индексом old_fn_idx имеет другое имя (была перезаписана)
                            // 2. Старая функция с индексом old_fn_idx имеет то же имя, что и наша функция
                            let should_update = if *old_fn_idx == compiler_fn_idx {
                                // Проверяем, не была ли функция перезаписана
                                if *old_fn_idx < self.functions.len() {
                                    // Если функция с индексом old_fn_idx имеет другое имя, значит она была перезаписана
                                    // Это означает, что наша функция (__main__) была сохранена с индексом 0,
                                    // но теперь функция с индексом 0 имеет другое имя (Data::method_deposit)
                                    let old_fn_name = &self.functions[*old_fn_idx].name;
                                    let was_overwritten = old_fn_name != function_name;
                                    debug_println!("[DEBUG set_functions] Проверка для '{}': old_fn_idx={}, old_fn_name='{}', compiler_idx={}, was_overwritten={}", 
                                        function_name, old_fn_idx, old_fn_name, compiler_fn_idx, was_overwritten);
                                    was_overwritten
                                } else {
                                    true
                                }
                            } else if *old_fn_idx < self.functions.len() {
                                // Проверяем, совпадает ли имя старой функции с именем нашей функции
                                self.functions[*old_fn_idx].name == *function_name && *old_fn_idx != new_fn_idx
                            } else {
                                false
                            };
                            
                            if should_update {
                                debug_println!("[DEBUG set_functions] Найдена функция '{}' в globals[{}] с индексом {} -> {} (compiler_idx: {}, old_fn_name: '{}')", 
                                    function_name, global_idx, old_fn_idx, new_fn_idx, compiler_fn_idx,
                                    if *old_fn_idx < self.functions.len() { &self.functions[*old_fn_idx].name } else { "OUT_OF_BOUNDS" });
                                self.globals[global_idx] = Value::Function(new_fn_idx);
                                break;
                            }
                        }
                    }
                }
            }
        }
        
        // Затем обрабатываем функции из существующих глобальных переменных
        for (global_idx, name) in self.global_names.clone().iter() {
            if *global_idx < self.globals.len() {
                if let Value::Function(old_fn_idx) = &self.globals[*global_idx] {
                    debug_println!("[DEBUG set_functions] Найдена функция '{}' в globals[{}] с индексом функции {}", name, global_idx, old_fn_idx);
                    
                    // Проверяем, что старый индекс функции соответствует имени функции
                    // Это важно для безопасности - мы обновляем только если имя совпадает
                    let old_fn_name_matches = if *old_fn_idx < self.functions.len() {
                        let matches = self.functions[*old_fn_idx].name == *name;
                        debug_println!("[DEBUG set_functions] Старый индекс функции {}: имя='{}', совпадает с глобальной переменной '{}': {}", 
                            old_fn_idx, self.functions[*old_fn_idx].name, name, matches);
                        matches
                    } else {
                        debug_println!("[DEBUG set_functions] Старый индекс функции {} выходит за границы (всего функций: {})", 
                            old_fn_idx, self.functions.len());
                        false
                    };
                    
                    // Находим функцию по имени в списке функций VM
                    if let Some(new_fn_idx) = self.find_function_index_by_name(name) {
                        debug_println!("[DEBUG set_functions] Найдена функция '{}' в VM с индексом {}", name, new_fn_idx);
                        // Проверяем, что имя функции в новом индексе совпадает с именем глобальной переменной
                        if new_fn_idx < self.functions.len() && self.functions[new_fn_idx].name == *name {
                            if *old_fn_idx != new_fn_idx {
                                debug_println!("[DEBUG set_functions] Обновляем индекс функции для '{}' в globals[{}]: {} -> {} (старое имя совпадает: {})", 
                                    name, global_idx, old_fn_idx, new_fn_idx, old_fn_name_matches);
                                self.globals[*global_idx] = Value::Function(new_fn_idx);
                            } else {
                                debug_println!("[DEBUG set_functions] Индекс функции для '{}' уже правильный: {}", name, new_fn_idx);
                            }
                        } else {
                            debug_println!("[DEBUG set_functions] WARNING: Имя функции в новом индексе {} не совпадает с именем глобальной переменной '{}' (имя функции: '{}')", 
                                new_fn_idx, name, 
                                if new_fn_idx < self.functions.len() { &self.functions[new_fn_idx].name } else { "OUT OF BOUNDS" });
                        }
                    } else {
                        // Функция не найдена - это может быть нормально для некоторых случаев
                        // (например, если функция еще не была добавлена или была удалена)
                        debug_println!("[DEBUG set_functions] WARNING: Функция '{}' не найдена в списке функций VM (старый индекс: {}, старое имя совпадает: {})", 
                            name, old_fn_idx, old_fn_name_matches);
                    }
                }
            }
        }
        debug_println!("[DEBUG set_functions] Завершено обновление индексов функций в globals");
        
        // В конце явно устанавливаем слот __main__ (точка входа из главного chunk), чтобы его не перезаписали другие циклы
        if let Some((&global_idx, _)) = self.global_names.iter().find(|(_, n)| *n == "__main__") {
            if let Some(fn_idx) = self.find_function_index_by_name("__main__") {
                if global_idx < self.globals.len() {
                    self.globals[global_idx] = Value::Function(fn_idx);
                    debug_println!("[DEBUG set_functions] Установлен слот __main__ в globals[{}] = Value::Function({})", global_idx, fn_idx);
                }
            }
        }
        
        // Проверяем, что конструкторы из __lib__.dc все еще имеют правильные индексы функций
        for (idx, name) in &self.global_names {
            if name.contains("::new_") {
                if let Some(value) = self.globals.get(*idx) {
                    if let Value::Function(fn_idx) = value {
                        if *fn_idx < self.functions.len() {
                            let func = &self.functions[*fn_idx];
                            debug_println!("[DEBUG set_functions] Проверка: конструктор '{}' в globals[{}] имеет индекс функции {}, имя функции: '{}', arity: {}", 
                                name, idx, fn_idx, func.name, func.arity);
                        } else {
                            debug_println!("[DEBUG set_functions] ОШИБКА: конструктор '{}' в globals[{}] имеет индекс функции {} (выходит за границы, всего функций: {})", 
                                name, idx, fn_idx, self.functions.len());
                        }
                    }
                }
            }
        }
        
        // Добавляем explicit_global_names
        for (idx, name) in explicit_global_names_to_add {
            self.explicit_global_names.insert(idx, name);
        }
    }

    /// Добавляет функции к существующим функциям в VM
    /// Возвращает начальный индекс добавленных функций
    pub fn add_functions(&mut self, functions: Vec<crate::bytecode::Function>) -> usize {
        let start_index = self.functions.len();
        self.functions.extend(functions);
        // Обновляем имена глобальных переменных из новых функций
        for function in self.functions.iter().skip(start_index) {
            for (idx, name) in &function.chunk.global_names {
                self.global_names.insert(*idx, name.clone());
            }
            for (idx, name) in &function.chunk.explicit_global_names {
                self.explicit_global_names.insert(*idx, name.clone());
            }
        }
        start_index
    }
    
    /// Получает количество функций в VM
    pub fn functions_count(&self) -> usize {
        self.functions.len()
    }
    
    /// Получает функции VM (для использования при импорте модулей)
    pub fn get_functions(&self) -> &Vec<crate::bytecode::Function> {
        &self.functions
    }

    /// Мутабельный доступ к функциям VM (для обновления чанков после merge модуля)
    pub fn get_functions_mut(&mut self) -> &mut Vec<crate::bytecode::Function> {
        &mut self.functions
    }

    pub fn register_native_globals(&mut self) {
        globals::register_native_globals(&mut self.globals, &mut self.global_names);
    }

    /// Register all built-in modules (ml, plot, settings_env) so native indices are consistent
    /// across all VMs (main and sub-VMs used for module loading).
    pub fn register_all_builtin_modules(&mut self) -> Result<(), LangError> {
        use crate::vm::modules;
        modules::register_module("ml", &mut self.natives, &mut self.globals, &mut self.global_names)?;
        modules::register_module("plot", &mut self.natives, &mut self.globals, &mut self.global_names)?;
        modules::register_module("settings_env", &mut self.natives, &mut self.globals, &mut self.global_names)?;
        modules::register_module("uuid", &mut self.natives, &mut self.globals, &mut self.global_names)?;
        Ok(())
    }

    // Exception handling methods moved to exceptions.rs

    pub fn run(&mut self, chunk: &Chunk) -> Result<Value, LangError> {
        // Reinforce thread-local base path from VM copy so load_env and imports resolve relative paths deterministically
        crate::vm::file_import::set_base_path(self.base_path.clone());

        // Заполняем имена глобальных переменных из chunk
        globals::merge_global_names(
            &mut self.global_names,
            &mut self.explicit_global_names,
            &chunk.global_names,
            &chunk.explicit_global_names,
        );
        
        // Создаем начальный frame
        let function = crate::bytecode::Function::new("<main>".to_string(), 0);
        let mut function = function;
        function.chunk = chunk.clone();
        let frame = CallFrame::new(function, 0);
        self.frames.push(frame);

        loop {
            match self.step()? {
                VMStatus::Continue => {}
                VMStatus::Return(v) => return Ok(v),
                VMStatus::FrameEnded => break,
            }
        }

        // После завершения выполнения возвращаем последнее значение на стеке
        if !self.stack.is_empty() {
            Ok(self.stack.pop().unwrap())
        } else {
            Ok(Value::Null)
        }
    }

    /// Выполнить один шаг VM - получить следующую инструкцию и выполнить её
    fn step(&mut self) -> Result<VMStatus, LangError> {
        // Get raw pointer to self before any mutable borrows
        let vm_ptr = self as *mut Vm;
        
        // Get instruction and line from executor::step (which already increments IP)
        let (instruction, line) = {
            let frames_ref = &mut self.frames;
            match executor::step(frames_ref)? {
                Some((inst, ln)) => (inst, ln),
            None => return Ok(VMStatus::FrameEnded),
            }
        };
        
        // Execute the instruction (frames_ref is dropped, so we can borrow again)
        executor::execute_instruction(
            instruction,
                                line,
            &mut self.stack,
            &mut self.frames,
            &mut self.globals,
            &mut self.global_names,
            &self.explicit_global_names,
            &self.functions,
            &mut self.natives,
            &mut self.exception_handlers,
            &mut self.error_type_table,
            &mut self.explicit_relations,
            &mut self.explicit_primary_keys,
            &mut self.loaded_modules,
            &mut self.abi_natives,
            &mut self.loaded_native_libraries,
            vm_ptr,
        )
    }

    // execute_instruction moved to executor.rs
    // Stack operations moved to stack.rs
    // Binary and unary operations moved to operations.rs

    // Module registration methods moved to modules.rs

    /// Получить доступ к глобальным переменным (для экспорта)
    pub fn get_globals(&self) -> &Vec<Value> {
        &self.globals
    }

    /// Получить мутабельный доступ к глобальным переменным
    pub fn get_globals_mut(&mut self) -> &mut Vec<Value> {
        &mut self.globals
    }

    /// Количество встроенных нативов (natives.len()). Индексы >= этого — ABI-нативы.
    pub fn builtin_natives_count(&self) -> usize {
        self.natives.len()
    }

    /// Срез нативных функций (для merge: копирование нативов модуля в main VM).
    pub fn get_natives(&self) -> &[NativeFn] {
        &self.natives[..]
    }

    /// ABI-нативы (из загруженных .so/.dylib). Индекс в Value::NativeFunction = builtin_natives_count + индекс в этом векторе.
    pub fn get_abi_natives(&self) -> &[NativeAbiFn] {
        &self.abi_natives
    }

    /// Зарезервировано для будущего API (другие крейты, тесты).
    #[allow(dead_code)]
    pub(crate) fn get_abi_natives_mut(&mut self) -> &mut Vec<NativeAbiFn> {
        &mut self.abi_natives
    }

    /// Зарезервировано для будущего API (другие крейты, тесты).
    #[allow(dead_code)]
    pub(crate) fn get_loaded_native_libraries_mut(&mut self) -> &mut Vec<Library> {
        &mut self.loaded_native_libraries
    }

    /// Получить доступ к именам глобальных переменных
    pub fn get_global_names(&self) -> &std::collections::HashMap<usize, String> {
        &self.global_names
    }

    /// Получить мутабельный доступ к именам глобальных переменных
    pub fn get_global_names_mut(&mut self) -> &mut std::collections::HashMap<usize, String> {
        &mut self.global_names
    }

    /// Получить доступ к именам переменных, явно объявленных с ключевым словом 'global'
    pub fn get_explicit_global_names(&self) -> &std::collections::HashMap<usize, String> {
        &self.explicit_global_names
    }

    /// Объединяет глобальные переменные из другого VM в этот VM
    /// Используется для передачи глобальных переменных из __lib__.dc в основной файл
    pub fn merge_globals_from(&mut self, other: &Vm) {
        const BUILTIN_COUNT: usize = 70;
        // Чтобы NativeFunction(i) из модуля (i >= 70) был валиден в self, добавляем нативы модуля.
        let other_natives = other.get_natives();
        if other_natives.len() > BUILTIN_COUNT {
            self.natives.extend_from_slice(&other_natives[BUILTIN_COUNT..]);
            debug_println!("[DEBUG merge_globals_from] Добавлено нативов из модуля: {}", other_natives.len() - BUILTIN_COUNT);
        }

        let other_globals = other.get_globals();
        let other_global_names = other.get_global_names();
        
        debug_println!("[DEBUG merge_globals_from] Объединяем глобальные переменные из другого VM");
        debug_println!("[DEBUG merge_globals_from] Функций в текущем VM: {}", self.functions.len());
        debug_println!("[DEBUG merge_globals_from] Функций в другом VM: {}", other.functions.len());
        debug_println!("[DEBUG merge_globals_from] Глобальных переменных в другом VM: {}", other_global_names.len());
        
        // Также объединяем функции из другого VM
        // ВАЖНО: Если функции из модулей уже были добавлены в VM во время выполнения,
        // start_function_index должен учитывать эти функции. Но merge происходит ДО выполнения кода,
        // поэтому функции из модулей еще не добавлены. Индексы функций будут обновлены позже при импорте модулей.
        let start_function_index = self.functions.len();
        self.functions.extend(other.functions.iter().cloned());
        debug_println!("[DEBUG merge_globals_from] Начальный индекс функций: {} (функций в текущем VM: {})", start_function_index, self.functions.len() - other.functions.len());
        debug_println!("[DEBUG merge_globals_from] Всего функций после объединения: {}", self.functions.len());
        
        // Итерируем в детерминированном порядке (по имени, затем по индексу). При дубликатах имени
        // предпочитаем значение-класс (Value::Object с __class_name), иначе — значение с большим индексом.
        let mut pairs: Vec<_> = other_global_names
            .iter()
            .map(|(i, n)| (*i, n.clone()))
            .collect();
        pairs.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        // Группируем по имени и выбираем одно значение на имя: предпочитаем Object (класс), иначе последнее по индексу.
        let mut by_name: std::collections::HashMap<String, (usize, &Value)> = std::collections::HashMap::new();
        for (index, name) in &pairs {
            if name == "argv" {
                continue;
            }
            if let Some(value) = other_globals.get(*index) {
                let prefer = match by_name.get(name) {
                    Some((prev_index, existing)) => {
                        // Предпочитаем Object (класс) перед Function; при равных типах — больший индекс.
                        let existing_is_obj = matches!(existing, Value::Object(_));
                        let value_is_obj = matches!(value, Value::Object(_));
                        if value_is_obj && !existing_is_obj {
                            true
                        } else if !value_is_obj && existing_is_obj {
                            false
                        } else {
                            *index >= *prev_index
                        }
                    }
                    None => true,
                };
                if prefer {
                    by_name.insert(name.clone(), (*index, value));
                }
            }
        }

        let mut names_sorted: Vec<_> = by_name.keys().cloned().collect();
        names_sorted.sort();
        for name in names_sorted {
            let (index, value) = *by_name.get(&name).unwrap();
                debug_println!("[DEBUG merge_globals_from] Объединяем '{}' (index: {})", name, index);
                // Если это функция, нужно обновить индекс функции
                let value_to_store = match value {
                    Value::Function(function_index) => {
                        // Обновляем индекс функции с учетом уже добавленных функций
                        let new_function_index = start_function_index + *function_index;
                        debug_println!("[DEBUG merge_globals_from] Обновляем индекс функции для '{}': {} -> {}", name, function_index, new_function_index);
                        Value::Function(new_function_index)
                    }
                    Value::Object(obj_rc) => {
                        // Для объектов (классов) нужно обновить индексы функций и нативов внутри объекта
                        use std::rc::Rc;
                        use std::cell::RefCell;
                        let obj = obj_rc.borrow();
                        let mut new_obj = HashMap::new();
                        let other_natives = other.get_natives();
                        for (key, val) in obj.iter() {
                            let updated_val = match val {
                                Value::Function(function_index) => {
                                    Value::Function(start_function_index + *function_index)
                                }
                                Value::NativeFunction(i) => {
                                    if *i >= other_natives.len() {
                                        val.clone()
                                    } else {
                                        let fn_ptr = other_natives[*i] as *const ();
                                        let remapped = if *i < self.natives.len()
                                            && std::ptr::eq(self.natives[*i] as *const (), fn_ptr)
                                        {
                                            *i
                                        } else {
                                            self.natives[BUILTIN_COUNT..]
                                                .iter()
                                                .position(|&f| std::ptr::eq(f as *const (), fn_ptr))
                                                .map(|pos| BUILTIN_COUNT + pos)
                                                .unwrap_or_else(|| {
                                                    self.natives.push(other_natives[*i]);
                                                    self.natives.len() - 1
                                                })
                                        };
                                        Value::NativeFunction(remapped)
                                    }
                                }
                                _ => val.clone()
                            };
                            new_obj.insert(key.clone(), updated_val);
                        }
                        Value::Object(Rc::new(RefCell::new(new_obj)))
                    }
                    Value::NativeFunction(i) if *i >= BUILTIN_COUNT => {
                        // Remap by identity: other VM may have different natives layout; store index j
                        // such that self.natives[j] == other.natives[i] (same function pointer).
                        let other_natives = other.get_natives();
                        if *i >= other_natives.len() {
                            value.clone()
                        } else {
                            let fn_ptr = other_natives[*i] as *const ();
                            let remapped = if *i < self.natives.len()
                                && std::ptr::eq(self.natives[*i] as *const (), fn_ptr)
                            {
                                *i
                            } else {
                                self.natives[BUILTIN_COUNT..]
                                    .iter()
                                    .position(|&f| std::ptr::eq(f as *const (), fn_ptr))
                                    .map(|pos| BUILTIN_COUNT + pos)
                                    .unwrap_or_else(|| {
                                        self.natives.push(other_natives[*i]);
                                        self.natives.len() - 1
                                    })
                            };
                            Value::NativeFunction(remapped)
                        }
                    }
                    _ => value.clone()
                };
                
                // Проверяем, существует ли уже такая переменная. Берём минимальный индекс при дубликатах,
                // чтобы всегда перезаписывать один и тот же слот (Config в 71, а не создавать второй в 78).
                let existing_indices: Vec<usize> = self.global_names.iter()
                    .filter(|(_, n)| n.as_str() == name.as_str())
                    .map(|(idx, _)| *idx)
                    .collect();
                if let Some(&existing_index) = existing_indices.iter().min() {
                    // Не перезаписывать слоты 0..70 (канонические нативы) значениями из модуля.
                    if existing_index < BUILTIN_COUNT {
                        debug_println!("[DEBUG merge_globals_from] Пропуск перезаписи '{}' (слот {} — канонический натив)", name, existing_index);
                        continue;
                    }
                    // Не перезаписывать слот main встроенным модулем из other (ml, plot, settings_env, uuid):
                    // иначе затрутся экспорты модуля (Config, DatabaseConfig и т.д.), занявшие те же индексы.
                    if crate::vm::modules::is_known_module(name.as_str()) {
                        debug_println!("[DEBUG merge_globals_from] Пропуск перезаписи '{}' (встроенный модуль)", name);
                        continue;
                    }
                    // Не перезаписывать корректное значение ошибочным: если value_to_store — встроенный натив (0..70),
                    // но имя переменной не совпадает с каноническим именем этого натива, пропускаем перезапись.
                    if let Value::NativeFunction(i) = value_to_store {
                        if i < BUILTIN_COUNT {
                            if let Some(canonical) = globals::builtin_global_name(i) {
                                if canonical != name.as_str() {
                                    debug_println!("[DEBUG merge_globals_from] Пропуск перезаписи '{}' на встроенный {:?} (индекс {})", name, canonical, i);
                                    continue;
                                }
                            }
                        }
                    }
                    // Перезаписываем существующую переменную
                    let val_type = match &value_to_store {
                        Value::Object(_) => "Object",
                        Value::Function(_) => "Function",
                        _ => "Other",
                    };
                    debug_println!("[DEBUG merge_globals_from] '{}' уже существует, перезаписываем globals[{}] ({})", name, existing_index, val_type);
                    if name == "Config" || name == "DatabaseConfig" {
                        debug_println!("[DEBUG merge_globals_from] Config/DatabaseConfig: '{}' -> слот {} ({})", name, existing_index, val_type);
                    }
                    if existing_index < self.globals.len() {
                        self.globals[existing_index] = value_to_store;
                    } else {
                        // Индекс выходит за границы, расширяем массив
                        self.globals.resize(existing_index + 1, Value::Null);
                        self.globals[existing_index] = value_to_store;
                    }
                } else {
                    // Не создавать глобальную с ошибочным встроенным: если value — встроенный натив (0..70),
                    // но имя не совпадает с каноническим, пропускаем (дубликат из модуля вроде (0, "Settings")).
                    if let Value::NativeFunction(i) = value_to_store {
                        if i < BUILTIN_COUNT {
                            if let Some(canonical) = globals::builtin_global_name(i) {
                                if canonical != name.as_str() {
                                    debug_println!("[DEBUG merge_globals_from] Пропуск создания глобальной '{}' с встроенным {:?} (индекс {})", name, canonical, i);
                                    continue;
                                }
                            }
                        }
                    }
                    // Создаем новую глобальную переменную
                    let new_index = self.globals.len();
                    let val_type = match &value_to_store {
                        Value::Object(_) => "Object",
                        Value::Function(_) => "Function",
                        _ => "Other",
                    };
                    self.globals.push(value_to_store);
                    self.global_names.insert(new_index, name.clone());
                    debug_println!("[DEBUG merge_globals_from] Создана новая глобальная переменная '{}' в globals[{}] ({})", name, new_index, val_type);
                    if name == "Config" || name == "DatabaseConfig" {
                        debug_println!("[DEBUG merge_globals_from] Config/DatabaseConfig: '{}' -> новый слот {} ({})", name, new_index, val_type);
                    }
                }
        }
        debug_println!("[DEBUG merge_globals_from] Объединение завершено. Всего глобальных переменных: {}", self.global_names.len());
    }

    /// Добавить явную связь между колонками таблиц
    pub fn add_explicit_relation(&mut self, relation: ExplicitRelation) {
        self.explicit_relations.push(relation);
    }

    /// Получить все явные связи
    pub fn get_explicit_relations(&self) -> &Vec<ExplicitRelation> {
        &self.explicit_relations
    }

    /// Добавить явный первичный ключ таблицы
    pub fn add_explicit_primary_key(&mut self, primary_key: ExplicitPrimaryKey) {
        self.explicit_primary_keys.push(primary_key);
    }

    /// Получить явные первичные ключи таблиц
    pub fn get_explicit_primary_keys(&self) -> &Vec<ExplicitPrimaryKey> {
        &self.explicit_primary_keys
    }

    /// Вызвать пользовательскую функцию по индексу с заданными аргументами
    /// Используется нативными функциями для вызова пользовательских функций
    pub fn call_function_by_index(&mut self, function_index: usize, args: &[Value]) -> Result<Value, LangError> {
        // Setup function call (creates frame, handles cache, sets up captured variables)
        if let Some(cached_result) = calls::setup_function_call(
            function_index,
            args,
            &self.functions,
            &mut self.stack,
            &mut self.frames,
            &mut self.error_type_table,
        )? {
            return Ok(cached_result);
        }

        // Execute the function using step()
        let initial_stack_size = self.stack.len();
        let initial_frames_count = self.frames.len();
        
        loop {
            if self.frames.len() < initial_frames_count {
                break;
            }

            match self.step()? {
                VMStatus::Continue => {}
                VMStatus::Return(v) => {
                    return Ok(v);
                }
                VMStatus::FrameEnded => {
                    break;
                }
            }
        }

        // Return value from stack if any
        // Проверяем относительно текущего frame's stack_start (caller's frame)
        // после того как функция вернулась и её frame был удалён
        // Используем безопасное извлечение без вызова stack::pop, чтобы избежать
        // ошибки stack underflow, которая может быть неправильно обработана
        if let Some(frame) = self.frames.last() {
            if self.stack.len() > frame.stack_start {
                // Безопасно извлекаем значение напрямую, так как мы уже проверили
                // что стек не пуст относительно stack_start
                Ok(self.stack.pop().unwrap_or(Value::Null))
            } else {
                Ok(Value::Null)
            }
        } else {
            // Нет frame - проверяем относительно initial_stack_size
            if self.stack.len() > initial_stack_size {
                // Безопасно извлекаем значение напрямую
                Ok(self.stack.pop().unwrap_or(Value::Null))
            } else {
                Ok(Value::Null)
            }
        }
    }
}


/// Контекст компиляции для передачи между модулями
use crate::bytecode::{Chunk, Function};
use crate::compiler::labels::LabelManager;
use crate::compiler::scope::ScopeManager;

// Структура для отслеживания обработчиков исключений
#[derive(Clone)]
pub struct ExceptionHandler {
    pub catch_ips: Vec<usize>,               // IP начала каждого catch блока
    pub error_types: Vec<Option<usize>>,     // Типы ошибок для каждого catch (None для catch всех)
    pub error_var_slots: Vec<Option<usize>>, // Слоты для переменных ошибок
    pub else_ip: Option<usize>,              // IP начала else блока
    pub finally_ip: Option<usize>,           // IP начала finally блока
    pub stack_height: usize,                 // Высота стека при входе в try
}

// Структура для отслеживания контекста циклов
pub struct LoopContext {
    pub continue_label: usize, // Метка для continue (начало следующей итерации или инкремент)
    pub break_label: usize,    // Метка для break (конец цикла)
    /// true для for i in range(...); при break нужно снять состояние с for_range_stack
    pub is_for_range: bool,
}

/// Контекст компиляции для передачи между модулями
pub struct CompilationContext<'a> {
    pub chunk: &'a mut Chunk,
    pub scope: &'a mut ScopeManager,
    pub labels: &'a mut LabelManager,
    pub functions: &'a mut Vec<Function>,
    pub function_names: &'a mut Vec<String>,
    pub current_function: Option<usize>,
    pub current_line: &'a mut usize,
    pub exception_handlers: &'a mut Vec<ExceptionHandler>,
    pub error_type_table: &'a mut Vec<String>,
    pub loop_contexts: &'a mut Vec<LoopContext>,
    /// Symbols imported via `from module import X`: name -> module. Used to treat uppercase names as class constructor calls only for file modules (not builtins).
    pub imported_symbols: &'a mut std::collections::HashMap<String, String>,
    /// True after any `from M import *` in this compilation unit. Allows calls to unknown names to allocate global slots (filled at runtime by ImportFrom).
    pub has_star_import: &'a mut bool,
    /// Class name -> list of private field names (filled at end of each class; used to merge in subclass constructors).
    pub class_private_fields: &'a mut std::collections::HashMap<String, Vec<String>>,
    /// Class name -> list of protected field names (for inheritance: merge in subclass constructors).
    pub class_protected_fields: &'a mut std::collections::HashMap<String, Vec<String>>,
    /// Class name -> list of private method names (for inheritance: merge in subclass constructors).
    pub class_private_methods: &'a mut std::collections::HashMap<String, Vec<String>>,
    /// Class name -> list of protected method names (for inheritance: merge in subclass constructors).
    pub class_protected_methods: &'a mut std::collections::HashMap<String, Vec<String>>,
    /// Subclass name -> superclass name, set when implicit constructor was skipped (superclass has no matching constructor).
    pub class_superclass: &'a mut std::collections::HashMap<String, String>,
    /// Class name -> true if the class extends Table (directly or indirectly). Used for isinstance(x, Table).
    pub class_extends_table: &'a mut std::collections::HashMap<String, bool>,
    /// Class name -> true if the class extends SQLEnum (directly or indirectly).
    pub class_extends_sqenum: &'a mut std::collections::HashMap<String, bool>,
    /// Class name -> (constructor_name, function_index). Used to resolve named-arg constructor calls (e.g. User(name="Alice")) for extends_table classes.
    pub class_constructor: &'a mut std::collections::HashMap<String, (String, usize)>,
    /// Class names marked with @Abstract; calls to these must load the class object (not constructor) so VM can check __abstract.
    pub abstract_classes: &'a mut std::collections::HashSet<String>,
    /// Class name -> env_prefix from model_config (for Settings subclasses). Used to build nested_specs when calling load_env.
    pub class_settings_env_prefix: &'a mut std::collections::HashMap<String, String>,
    /// Class name -> nested_specs Value (array) for Settings subclasses. Used so subclasses (e.g. DevSettings) pass parent's nested_specs when calling super.
    pub class_nested_specs_value:
        &'a mut std::collections::HashMap<String, crate::common::value::Value>,
    /// Class name -> default required_keys array for Settings subclasses. Used at call site for Config(path) to pass 3 args.
    pub class_required_keys_value:
        &'a mut std::collections::HashMap<String, crate::common::value::Value>,
    /// Current class being compiled (for super.method() resolution).
    pub current_class: Option<String>,
    /// Superclass of current class (for super() and super.method() resolution).
    pub current_superclass: Option<String>,
    /// Whether we are currently compiling a constructor body.
    pub in_constructor: bool,
    /// When in constructor body, the slot index for "this" (arity). Used so body always uses correct slot regardless of scope.
    pub constructor_this_slot: Option<usize>,
    /// Source file path for error messages (propagated to chunk.source_name).
    pub source_name: Option<&'a str>,
    /// Parse-time native export param names (from `native_call_descriptor` preload), e.g. `native_dataset_split`.
    pub native_call_param_registry:
        Option<&'a crate::vm::native_call_registry::NativeCallParamRegistry>,
    /// Best-effort compile-time knowledge: variables assigned from `set()` in the current compilation unit.
    /// Used to safely emit `SetAddIntegral` / `SetDiscardIntegral` without miscompiling class methods named `add`.
    pub known_set_vars: &'a mut std::collections::HashSet<String>,
    /// Best-effort tracking: variables assigned from constructor-like calls (UpperCamelCase(...)).
    /// Used to avoid miscompiling `obj.add(x)` into set opcodes when `obj` is likely a class instance.
    pub known_class_instance_vars: &'a mut std::collections::HashSet<String>,
    /// Variables assigned from a compile-time constant tuple of 2-tuples, e.g. `neighbors_delta = ((-1,0), ...)`.
    pub known_const_pair_tuples: &'a mut std::collections::HashMap<String, Vec<(i64, i64)>>,
    /// Names assigned or declared in the current compilation unit (locals, params, for-vars, script globals).
    /// Used for scope-aware object literal keys: bound identifier → computed key, unbound → string key.
    pub known_bound_names: &'a mut std::collections::HashSet<String>,
    /// Nesting depth of active `for` pattern scopes (for declaring body locals in function scope).
    pub for_loop_scope_depth: &'a mut usize,
    /// Module-level names known to hold compile-time constants (for default parameter values).
    pub compile_time_bindings: &'a mut std::collections::HashMap<String, crate::common::value::Value>,
    /// Per-compile pass counter for duplicate function names (nested fns with the same name).
    pub function_compile_pass: &'a mut std::collections::HashMap<String, usize>,
    /// Local slot index -> user function index (nested fn bindings and self-recursion).
    pub local_fn_by_slot: &'a mut std::collections::HashMap<usize, usize>,
}

impl<'a> CompilationContext<'a> {
    #[inline]
    pub fn record_bound_name(&mut self, name: &str) {
        self.known_bound_names.insert(name.to_string());
    }

    /// Declare a user binding; inside `for` loop body, new names go to function scope, not pattern scope.
    pub fn declare_local_for_binding(&mut self, name: &str) -> usize {
        if *self.for_loop_scope_depth > 0 {
            self.scope
                .declare_local_outside_for_loops(*self.for_loop_scope_depth, name)
        } else {
            self.scope.declare_local(name)
        }
    }

    /// Local slot for `this` when emitting member access (`this.field`, `'this'` expr).
    /// In constructors, parameters occupy `0 .. params.len()-1` and `this` is at
    /// [`Self::constructor_this_slot`] (= `params.len()`). In methods, `this` is the first declared
    /// local (usually slot `0`).
    ///
    /// [`Self::constructor_this_slot`] must only apply while [`Self::in_constructor`] is true:
    /// it can otherwise leak into later method compilations for the same class and wrongly resolve
    /// `this` to the constructor's arity slot inside method bodies (e.g. `this.field` reads the last
    /// constructor parameter instead of the receiver).
    #[inline]
    pub fn this_local_slot_with_fallback_for_member_access(&self) -> usize {
        self.this_local_slot_for_member_access().unwrap_or(0)
    }

    #[inline]
    pub fn this_local_slot_for_member_access(&self) -> Option<usize> {
        if self.in_constructor {
            if let Some(s) = self.constructor_this_slot {
                return Some(s);
            }
            return self.scope.resolve_local("this");
        }
        self.scope.resolve_local("this")
    }

    pub fn get_error_type_index(&mut self, error_type_name: &str) -> usize {
        // Ищем в существующей таблице
        if let Some(index) = self
            .error_type_table
            .iter()
            .position(|s| s == error_type_name)
        {
            return index;
        }
        // Добавляем новый тип
        let index = self.error_type_table.len();
        self.error_type_table.push(error_type_name.to_string());
        index
    }
}

/// Управление областями видимости и переменными

pub struct ScopeManager {
    pub globals: std::collections::HashMap<String, usize>,
    /// Names declared with the `global` keyword at module/script top level.
    pub explicit_globals: std::collections::HashSet<String>,
    pub locals: Vec<std::collections::HashMap<String, usize>>,
    pub local_count: usize,
    /// Nesting depth of active method-call receiver temps (per-depth slot reuse).
    method_object_depth: usize,
    /// Local slot index per nesting depth (`__method_object_0`, `__method_object_1`, …).
    method_object_slots: Vec<usize>,
}

/// Saved method-object temp state while compiling a nested function/lambda.
#[derive(Clone)]
pub struct MethodObjectTempState {
    depth: usize,
    slots: Vec<usize>,
}

impl ScopeManager {
    pub fn new() -> Self {
        Self {
            globals: std::collections::HashMap::new(),
            explicit_globals: std::collections::HashSet::new(),
            locals: Vec::new(),
            local_count: 0,
            method_object_depth: 0,
            method_object_slots: Vec::new(),
        }
    }

    pub fn begin_scope(&mut self) {
        self.locals.push(std::collections::HashMap::new());
    }

    pub fn end_scope(&mut self) {
        self.locals.pop();
        // Slot indices are not reused within a function — body bindings declared during
        // a block (e.g. `for` pattern scope) keep their indices after the block ends.
    }

    pub fn declare_local(&mut self, name: &str) -> usize {
        let index = self.local_count;
        if let Some(scope) = self.locals.last_mut() {
            scope.insert(name.to_string(), index);
        }
        self.local_count += 1;
        index
    }

    /// Declare in the scope outside active `for` pattern scopes (function scope for loop body bindings).
    pub fn declare_local_outside_for_loops(&mut self, for_loop_depth: usize, name: &str) -> usize {
        let index = self.local_count;
        let target_idx = self
            .locals
            .len()
            .saturating_sub(1 + for_loop_depth);
        if target_idx < self.locals.len() {
            self.locals[target_idx].insert(name.to_string(), index);
        } else if let Some(scope) = self.locals.last_mut() {
            scope.insert(name.to_string(), index);
        }
        self.local_count += 1;
        index
    }

    pub fn resolve_local(&self, name: &str) -> Option<usize> {
        // Ищем переменную в текущих областях видимости (от последней к первой)
        for scope in self.locals.iter().rev() {
            if let Some(&index) = scope.get(name) {
                return Some(index);
            }
        }
        None
    }

    /// Reset per-function method-object temp state (call when `local_count` is reset).
    pub fn reset_method_object_temps(&mut self) {
        self.method_object_depth = 0;
        self.method_object_slots.clear();
    }

    pub fn snapshot_method_object_temps(&self) -> MethodObjectTempState {
        MethodObjectTempState {
            depth: self.method_object_depth,
            slots: self.method_object_slots.clone(),
        }
    }

    pub fn restore_method_object_temps(&mut self, state: MethodObjectTempState) {
        self.method_object_depth = state.depth;
        self.method_object_slots = state.slots;
    }

    /// Acquire a temp local for a method-call receiver.
    ///
    /// Default (`depth_stack`): one slot per nesting depth — safe for nested method calls in args.
    /// `method_object_per_call`: legacy — new slot per call site (large frames).
    /// `method_object_shared`: broken — single reused slot (incorrect for nested calls).
    pub fn begin_method_object_temp(&mut self) -> usize {
        #[cfg(feature = "method_object_per_call")]
        {
            return self.declare_local("__method_object");
        }

        #[cfg(feature = "method_object_shared")]
        {
            if let Some(index) = self.resolve_local("__method_object") {
                return index;
            }
            return self.declare_local("__method_object");
        }

        #[cfg(not(any(
            feature = "method_object_per_call",
            feature = "method_object_shared"
        )))]
        {
            let depth = self.method_object_depth;
            self.method_object_depth += 1;
            if depth < self.method_object_slots.len() {
                return self.method_object_slots[depth];
            }
            let slot = self.declare_local(&format!("__method_object_{}", depth));
            self.method_object_slots.push(slot);
            slot
        }
    }

    pub fn release_method_object_temp(&mut self) {
        #[cfg(not(any(
            feature = "method_object_per_call",
            feature = "method_object_shared"
        )))]
        {
            debug_assert!(
                self.method_object_depth > 0,
                "release_method_object_temp without begin"
            );
            if self.method_object_depth > 0 {
                self.method_object_depth -= 1;
            }
        }
    }

    /// Max method-call nesting depth observed while compiling the current function.
    pub fn method_object_max_depth(&self) -> usize {
        self.method_object_slots.len()
    }
}

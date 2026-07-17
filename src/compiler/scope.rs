/// Управление областями видимости и переменными

pub struct ScopeManager {
    pub globals: std::collections::HashMap<String, usize>,
    /// Names declared with the `global` keyword at module/script top level.
    pub explicit_globals: std::collections::HashSet<String>,
    pub locals: Vec<std::collections::HashMap<String, usize>>,
    pub local_count: usize,
}

impl ScopeManager {
    pub fn new() -> Self {
        Self {
            globals: std::collections::HashMap::new(),
            explicit_globals: std::collections::HashSet::new(),
            locals: Vec::new(),
            local_count: 0,
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
}

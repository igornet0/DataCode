/// Управление областями видимости и переменными

pub struct ScopeManager {
    pub globals: std::collections::HashMap<String, usize>,
    pub locals: Vec<std::collections::HashMap<String, usize>>,
    pub local_count: usize,
}

impl ScopeManager {
    pub fn new() -> Self {
        Self {
            globals: std::collections::HashMap::new(),
            locals: Vec::new(),
            local_count: 0,
        }
    }

    pub fn begin_scope(&mut self) {
        self.locals.push(std::collections::HashMap::new());
    }

    pub fn end_scope(&mut self) {
        if let Some(scope) = self.locals.pop() {
            // Уменьшаем счетчик локальных переменных на количество переменных в этой области
            self.local_count -= scope.len();
        }
    }

    pub fn declare_local(&mut self, name: &str) -> usize {
        let index = self.local_count;
        if let Some(scope) = self.locals.last_mut() {
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



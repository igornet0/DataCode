// Профилирование VM: счётчики opcodes, аллокаций, обращений к store.
// Включается флагом cargo build --features profile. Нулевая стоимость при отключении.

#[cfg(feature = "profile")]
use std::cell::RefCell;
#[cfg(feature = "profile")]
use std::collections::HashMap;

#[cfg(feature = "profile")]
thread_local! {
    static PROFILE: RefCell<Option<ProfileStats>> = RefCell::new(None);
}

#[cfg(feature = "profile")]
thread_local! {
    static CURRENT_OPCODE: RefCell<Option<String>> = RefCell::new(None);
}

/// Статистика выполнения для одного run().
#[cfg(feature = "profile")]
#[derive(Default, Debug, Clone)]
pub struct ProfileStats {
    pub opcodes_executed: u64,
    pub store_allocations: u64,
    pub store_get_count: u64,
    /// Аллокации по имени опкода (без параметров).
    pub alloc_by_opcode: HashMap<String, u64>,
    /// Обращения get к store по имени опкода.
    pub get_by_opcode: HashMap<String, u64>,
}

#[cfg(feature = "profile")]
pub fn set() {
    PROFILE.with(|p| {
        *p.borrow_mut() = Some(ProfileStats {
            alloc_by_opcode: HashMap::new(),
            get_by_opcode: HashMap::new(),
            ..ProfileStats::default()
        });
    });
}

#[cfg(feature = "profile")]
pub fn take() -> Option<ProfileStats> {
    PROFILE.with(|p| p.borrow_mut().take())
}

#[cfg(feature = "profile")]
#[inline(always)]
pub fn record_opcode() {
    PROFILE.with(|p| {
        if let Some(ref mut s) = *p.borrow_mut() {
            s.opcodes_executed += 1;
        }
    });
}

#[cfg(feature = "profile")]
#[inline(always)]
pub fn record_allocate() {
    PROFILE.with(|p| {
        if let Some(ref mut s) = *p.borrow_mut() {
            s.store_allocations += 1;
            let name = CURRENT_OPCODE.with(|c| c.borrow().clone()).unwrap_or_else(|| "?".to_string());
            *s.alloc_by_opcode.entry(name).or_insert(0) += 1;
        }
    });
}

#[cfg(feature = "profile")]
#[inline(always)]
pub fn record_store_get() {
    PROFILE.with(|p| {
        if let Some(ref mut s) = *p.borrow_mut() {
            s.store_get_count += 1;
            let name = CURRENT_OPCODE.with(|c| c.borrow().clone()).unwrap_or_else(|| "?".to_string());
            *s.get_by_opcode.entry(name).or_insert(0) += 1;
        }
    });
}

#[cfg(feature = "profile")]
pub fn set_current_opcode(op: &crate::bytecode::OpCode) {
    CURRENT_OPCODE.with(|c| *c.borrow_mut() = Some(op.variant_name().to_string()));
}

#[cfg(feature = "profile")]
const TOP_N: usize = 15;

#[cfg(feature = "profile")]
pub fn print_stats(stats: &ProfileStats) {
    eprintln!("[profile] opcodes_executed   = {}", stats.opcodes_executed);
    eprintln!("[profile] store_allocations  = {}", stats.store_allocations);
    eprintln!("[profile] store_get_count    = {}", stats.store_get_count);

    let mut by_alloc: Vec<_> = stats.alloc_by_opcode.iter().map(|(k, v)| (k.clone(), *v)).collect();
    by_alloc.sort_by(|a, b| b.1.cmp(&a.1));
    eprintln!("[profile] top {} by alloc: {:?}", TOP_N, &by_alloc[..TOP_N.min(by_alloc.len())]);

    let mut by_get: Vec<_> = stats.get_by_opcode.iter().map(|(k, v)| (k.clone(), *v)).collect();
    by_get.sort_by(|a, b| b.1.cmp(&a.1));
    eprintln!("[profile] top {} by get:  {:?}", TOP_N, &by_get[..TOP_N.min(by_get.len())]);

    let mut combined: HashMap<String, u64> = HashMap::new();
    for (k, v) in &stats.alloc_by_opcode {
        *combined.entry(k.clone()).or_insert(0) += *v;
    }
    for (k, v) in &stats.get_by_opcode {
        *combined.entry(k.clone()).or_insert(0) += *v;
    }
    let mut by_combined: Vec<_> = combined.into_iter().collect();
    by_combined.sort_by(|a, b| b.1.cmp(&a.1));
    eprintln!("[profile] top {} by alloc+get: {:?}", TOP_N, &by_combined[..TOP_N.min(by_combined.len())]);
}

// Stubs when feature is off: no cost, no thread_local.
#[cfg(not(feature = "profile"))]
pub fn set() {}

#[cfg(not(feature = "profile"))]
pub fn take() -> Option<()> {
    None
}

#[cfg(not(feature = "profile"))]
#[inline(always)]
pub fn record_opcode() {}

#[cfg(not(feature = "profile"))]
#[inline(always)]
pub fn record_allocate() {}

#[cfg(not(feature = "profile"))]
#[inline(always)]
pub fn record_store_get() {}

#[cfg(not(feature = "profile"))]
#[inline(always)]
pub fn set_current_opcode(_op: &crate::bytecode::OpCode) {}


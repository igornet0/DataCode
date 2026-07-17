// Профилирование VM: счётчики opcodes, аллокаций, обращений к store.
// Включается флагом cargo build --features profile. Нулевая стоимость при отключении.
//
// Hot path: только инкременты в `[u64; N]` и `u8` current opcode — без HashMap/String на каждый op/get.

#[cfg(feature = "profile")]
use std::cell::RefCell;

#[cfg(feature = "profile")]
use crate::bytecode::opcode::{OpCode, PROFILE_OPCODE_SLOTS, PROFILE_OPCODE_UNKNOWN};

#[cfg(feature = "profile")]
const PROFILE_HIST_BUCKETS: usize = PROFILE_OPCODE_SLOTS + 1;

#[cfg(feature = "profile")]
thread_local! {
    static PROFILE: RefCell<Option<ProfileStats>> = RefCell::new(None);
}

#[cfg(feature = "profile")]
thread_local! {
    /// Текущий опкод в `execute_instruction` (для атрибуции alloc/get).
    static CURRENT_OPCODE: RefCell<u8> = RefCell::new(PROFILE_OPCODE_UNKNOWN);
}

/// Статистика выполнения для одного run().
#[cfg(feature = "profile")]
#[derive(Debug, Clone)]
pub struct ProfileStats {
    pub opcodes_executed: u64,
    pub store_allocations: u64,
    pub store_get_count: u64,
    pub opcode_executed: [u64; PROFILE_HIST_BUCKETS],
    pub alloc_by_opcode: [u64; PROFILE_HIST_BUCKETS],
    pub get_by_opcode: [u64; PROFILE_HIST_BUCKETS],
    pub native_calls: Vec<(String, u64)>,
}

#[cfg(feature = "profile")]
impl Default for ProfileStats {
    fn default() -> Self {
        Self {
            opcodes_executed: 0,
            store_allocations: 0,
            store_get_count: 0,
            opcode_executed: [0; PROFILE_HIST_BUCKETS],
            alloc_by_opcode: [0; PROFILE_HIST_BUCKETS],
            get_by_opcode: [0; PROFILE_HIST_BUCKETS],
            native_calls: Vec::new(),
        }
    }
}

#[cfg(feature = "profile")]
pub fn set() {
    PROFILE.with(|p| {
        *p.borrow_mut() = Some(ProfileStats::default());
    });
    CURRENT_OPCODE.with(|c| *c.borrow_mut() = PROFILE_OPCODE_UNKNOWN);
}

#[cfg(feature = "profile")]
pub fn take() -> Option<ProfileStats> {
    PROFILE.with(|p| p.borrow_mut().take())
}

#[cfg(feature = "profile")]
#[inline(always)]
fn bump_slot(slots: &mut [u64; PROFILE_HIST_BUCKETS], index: u8) {
    let i = index as usize;
    if i < PROFILE_HIST_BUCKETS {
        slots[i] += 1;
    }
}

#[cfg(feature = "profile")]
#[inline(always)]
pub fn record_native_call(name: &str) {
    PROFILE.with(|p| {
        if let Some(ref mut s) = *p.borrow_mut() {
            if let Some((_, c)) = s.native_calls.iter_mut().find(|(k, _)| k == name) {
                *c += 1;
            } else {
                s.native_calls.push((name.to_string(), 1));
            }
        }
    });
}

#[cfg(feature = "profile")]
#[inline(always)]
pub fn record_allocate() {
    PROFILE.with(|p| {
        if let Some(ref mut s) = *p.borrow_mut() {
            s.store_allocations += 1;
            let idx = CURRENT_OPCODE.with(|c| *c.borrow());
            bump_slot(&mut s.alloc_by_opcode, idx);
        }
    });
}

#[cfg(feature = "profile")]
#[inline(always)]
pub fn record_store_get() {
    PROFILE.with(|p| {
        if let Some(ref mut s) = *p.borrow_mut() {
            s.store_get_count += 1;
            let idx = CURRENT_OPCODE.with(|c| *c.borrow());
            bump_slot(&mut s.get_by_opcode, idx);
        }
    });
}

#[cfg(feature = "profile")]
pub fn set_current_opcode(op: &OpCode) {
    let idx = op.profile_index();
    CURRENT_OPCODE.with(|c| *c.borrow_mut() = idx);
    PROFILE.with(|p| {
        if let Some(ref mut s) = *p.borrow_mut() {
            s.opcodes_executed += 1;
            bump_slot(&mut s.opcode_executed, idx);
        }
    });
}

#[cfg(feature = "profile")]
const TOP_N: usize = 15;

#[cfg(feature = "profile")]
fn top_n_from_slots(slots: &[u64; PROFILE_HIST_BUCKETS]) -> Vec<(String, u64)> {
    let mut v: Vec<(String, u64)> = slots
        .iter()
        .enumerate()
        .filter(|(_, &count)| count > 0)
        .map(|(i, &count)| {
            let name = if i < PROFILE_OPCODE_SLOTS {
                OpCode::profile_name(i as u8).to_string()
            } else {
                "?".to_string()
            };
            (name, count)
        })
        .collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    v
}

#[cfg(feature = "profile")]
pub fn print_stats(stats: &ProfileStats) {
    eprintln!("[profile] opcodes_executed   = {}", stats.opcodes_executed);
    eprintln!("[profile] store_allocations  = {}", stats.store_allocations);
    eprintln!("[profile] store_get_count    = {}", stats.store_get_count);

    let by_opcode = top_n_from_slots(&stats.opcode_executed);
    eprintln!(
        "[profile] top {} opcodes executed: {:?}",
        TOP_N,
        &by_opcode[..TOP_N.min(by_opcode.len())]
    );

    let mut by_native = stats.native_calls.clone();
    by_native.sort_by(|a, b| b.1.cmp(&a.1));
    eprintln!(
        "[profile] top {} native calls: {:?}",
        TOP_N,
        &by_native[..TOP_N.min(by_native.len())]
    );

    let by_alloc = top_n_from_slots(&stats.alloc_by_opcode);
    eprintln!(
        "[profile] top {} by alloc: {:?}",
        TOP_N,
        &by_alloc[..TOP_N.min(by_alloc.len())]
    );

    let by_get = top_n_from_slots(&stats.get_by_opcode);
    eprintln!(
        "[profile] top {} by get:  {:?}",
        TOP_N,
        &by_get[..TOP_N.min(by_get.len())]
    );

    let mut combined: Vec<(String, u64)> = by_alloc
        .iter()
        .cloned()
        .chain(by_get.iter().cloned())
        .fold(Vec::new(), |mut acc, (k, v)| {
            if let Some((_, c)) = acc.iter_mut().find(|(name, _)| name == &k) {
                *c += v;
            } else {
                acc.push((k, v));
            }
            acc
        });
    combined.sort_by(|a, b| b.1.cmp(&a.1));
    eprintln!(
        "[profile] top {} by alloc+get: {:?}",
        TOP_N,
        &combined[..TOP_N.min(combined.len())]
    );
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
pub fn record_opcode_executed(_variant: &str) {}

#[cfg(not(feature = "profile"))]
#[inline(always)]
pub fn record_native_call(_name: &str) {}

#[cfg(not(feature = "profile"))]
#[inline(always)]
pub fn record_allocate() {}

#[cfg(not(feature = "profile"))]
#[inline(always)]
pub fn record_store_get() {}

#[cfg(not(feature = "profile"))]
#[inline(always)]
pub fn set_current_opcode(_op: &crate::bytecode::OpCode) {}

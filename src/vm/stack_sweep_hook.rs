//! Safe-point hook: sweep dead stack slots before compaction (feature `deferred_stack_gc`).

use crate::common::TaggedValue;
use crate::vm::vm::Vm;

#[inline]
pub fn maybe_sweep_dead_stack(vm: &mut Vm, stack: &[TaggedValue], old_sp: usize, new_sp: usize) {
    if old_sp > new_sp {
        vm.sweep_dead_stack(stack, old_sp, new_sp);
    }
}

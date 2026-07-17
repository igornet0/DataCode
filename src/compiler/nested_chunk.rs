//! Label isolation when compiling bytecode into a nested `Chunk` (class methods / constructors).

use crate::bytecode::Chunk;
use crate::common::error::LangError;

use crate::compiler::labels::LabelManager;

/// Snapshot of `LabelManager` for restoring after compiling a nested function chunk.
pub struct LabelsNestedCheckpoint {
    label_counter: usize,
    labels: std::collections::HashMap<usize, usize>,
    pending_jumps: Vec<(usize, usize, bool)>,
    pending_for_range: Vec<(usize, usize)>,
}

#[inline]
pub fn checkpoint_labels(labels: &LabelManager) -> LabelsNestedCheckpoint {
    LabelsNestedCheckpoint {
        label_counter: labels.label_counter,
        labels: labels.labels.clone(),
        pending_jumps: labels.pending_jumps.clone(),
        pending_for_range: labels.pending_for_range.clone(),
    }
}

#[inline]
pub fn enter_nested_label_scope(labels: &mut LabelManager) {
    labels.label_counter = 0;
    labels.labels.clear();
    labels.pending_jumps.clear();
    labels.pending_for_range.clear();
}

#[inline]
pub fn restore_labels(labels: &mut LabelManager, checkpoint: LabelsNestedCheckpoint) {
    labels.label_counter = checkpoint.label_counter;
    labels.labels = checkpoint.labels;
    labels.pending_jumps = checkpoint.pending_jumps;
    labels.pending_for_range = checkpoint.pending_for_range;
}

/// Run `stabilize_layout` + `finalize_jumps` on `chunk`, then restore outer label state.
pub fn finalize_nested_chunk(
    labels: &mut LabelManager,
    checkpoint: LabelsNestedCheckpoint,
    chunk: &mut Chunk,
    line: usize,
) -> Result<(), LangError> {
    labels.stabilize_layout(chunk, line)?;
    labels.finalize_jumps(chunk, line)?;
    // Only restore the global label id counter. `labels` / `pending_*` from the checkpoint
    // belong to the outer chunk; restoring them after patching a nested method chunk would
    // apply wrong jump indices to the next nested compile (class methods regression).
    labels.label_counter = checkpoint.label_counter;
    labels.labels.clear();
    labels.pending_jumps.clear();
    labels.pending_for_range.clear();
    labels.pending_local_heap_empty_jumps.clear();
    Ok(())
}

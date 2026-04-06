//! Memory layer: ValueStore + HeavyStore and Value ↔ ValueId conversion.
//! Phase 6: implementations in store.rs and convert.rs.

mod convert;
mod store;

pub use convert::{
    load_value, slot_to_value, store_value, store_value_arena, tagged_to_value_id,
    tagged_to_value_id_arena, update_cell_if_mutable, value_cell_to_tagged,
};
pub use store::HeavyStore;

// Re-export from memory layer. Implementation lives in vm/memory/convert.rs.
pub use crate::vm::memory::{ 
    load_value, 
    store_value, 
    store_value_arena, 
    slot_to_value, 
    tagged_to_value_id, 
    tagged_to_value_id_arena, 
    update_cell_if_mutable, 
    value_cell_to_tagged 
};

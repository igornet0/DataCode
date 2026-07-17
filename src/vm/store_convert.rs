// Re-export from memory layer. Implementation lives in vm/memory/convert.rs.
pub use crate::vm::memory::{
    integral_slot_to_value_id, load_value, object_cell_try_lookup_by_key_id,
    object_map_find_integral_slot_by_key_id, object_map_lookup_by_key_id, object_map_lookup_value,
    object_map_try_lookup_by_key_id, object_map_upsert, object_map_upsert_in_place,
    push_integral_slot, push_stack_value_id, prepare_value_for_container_store,
    promote_value_id_if_escapes, slot_to_value, store_value, store_value_arena,
    tagged_to_mutable_value_id, tagged_to_value_id, tagged_to_value_id_arena, try_write_cell_from_tagged,
    update_cell_if_mutable, value_cell_to_tagged, value_id_is_truthy, value_id_is_truthy_mut,
    value_id_for_object_field_update,
};

//! Grid A* with Manhattan heuristic on flat buffers (low RAM vs dict/set VM path).

use std::collections::HashSet;

pub use crate::common::astar_grid_core::{
    astar_grid_blocked, bitmap_blocked, bitmap_bytes_for_cells, bitmap_set,
};

/// A* on a 4-connected grid. Returns path `(row, col)` from start to goal, or `None`.
pub fn astar_grid(
    rows: u32,
    cols: u32,
    start: (u32, u32),
    goal: (u32, u32),
    blocked: &HashSet<u32>,
) -> Option<Vec<(u32, u32)>> {
    astar_grid_blocked(rows, cols, start, goal, None, Some(blocked))
}

/// Same as [`astar_grid`] but blocked cells come from a compact bitmap.
pub fn astar_grid_bitmap(
    rows: u32,
    cols: u32,
    start: (u32, u32),
    goal: (u32, u32),
    blocked_bits: &[u8],
) -> Option<Vec<(u32, u32)>> {
    astar_grid_blocked(rows, cols, start, goal, Some(blocked_bits), None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_grid_small_path() {
        let blocked = HashSet::new();
        let path = astar_grid(5, 5, (0, 0), (2, 2), &blocked).expect("path");
        assert_eq!(path.first(), Some(&(0, 0)));
        assert_eq!(path.last(), Some(&(2, 2)));
        assert_eq!(path.len(), 5);
    }

    #[test]
    fn bitmap_matches_hashset() {
        let mut blocked = HashSet::new();
        blocked.insert(1);
        let mut bits = vec![0u8; bitmap_bytes_for_cells(3)];
        bitmap_set(&mut bits, 1);
        assert!(astar_grid(1, 3, (0, 0), (0, 2), &blocked).is_none());
        assert!(astar_grid_bitmap(1, 3, (0, 0), (0, 2), &bits).is_none());
    }
}

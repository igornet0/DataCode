//! Shared grid A* on flat buffers (used by `pathfind` and `grid` modules).

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

pub const UNVISITED_U32: u32 = u32::MAX;
pub const NO_PARENT_U32: u32 = u32::MAX;
pub const STATE_UNSEEN: u8 = 0;
pub const STATE_OPEN: u8 = 1;
pub const STATE_CLOSED: u8 = 2;

struct AstarScratch {
    g_score: Vec<u32>,
    f_score: Vec<u32>,
    state: Vec<u8>,
    came_from: Vec<u32>,
}

impl AstarScratch {
    fn ensure(&mut self, n_cells: usize) {
        if self.g_score.len() < n_cells {
            self.g_score.resize(n_cells, UNVISITED_U32);
            self.f_score.resize(n_cells, 0);
            self.state.resize(n_cells, STATE_UNSEEN);
            self.came_from.resize(n_cells, NO_PARENT_U32);
        }
    }
}

thread_local! {
    static ASTAR_SCRATCH: RefCell<AstarScratch> = RefCell::new(AstarScratch {
        g_score: Vec::new(),
        f_score: Vec::new(),
        state: Vec::new(),
        came_from: Vec::new(),
    });
}

/// Drop retained capacity in the thread-local A* scratch pool (diagnostics / stress loops).
pub fn purge_astar_scratch_pool() {
    ASTAR_SCRATCH.with(|cell| {
        let mut s = cell.borrow_mut();
        s.g_score.shrink_to_fit();
        s.f_score.shrink_to_fit();
        s.state.shrink_to_fit();
        s.came_from.shrink_to_fit();
    });
}

#[inline]
pub fn manhattan(r1: u32, c1: u32, r2: u32, c2: u32) -> u32 {
    r1.abs_diff(r2) + c1.abs_diff(c2)
}

#[inline]
pub fn bitmap_blocked(blocked_bits: &[u8], cell_id: u32) -> bool {
    let i = cell_id as usize;
    let byte = i / 8;
    let bit = i % 8;
    blocked_bits.get(byte).is_some_and(|b| (b >> bit) & 1 != 0)
}

#[inline]
pub fn bitmap_set(blocked_bits: &mut [u8], cell_id: u32) {
    let i = cell_id as usize;
    let byte = i / 8;
    let bit = i % 8;
    if let Some(b) = blocked_bits.get_mut(byte) {
        *b |= 1 << bit;
    }
}

pub fn bitmap_bytes_for_cells(n_cells: usize) -> usize {
    n_cells.div_ceil(8)
}

fn is_blocked_hash(blocked: &HashSet<u32>, id: u32) -> bool {
    blocked.contains(&id)
}

fn is_blocked_bits(blocked_bits: Option<&[u8]>, id: u32) -> bool {
    blocked_bits.is_some_and(|bits| bitmap_blocked(bits, id))
}

/// A* on a 4-connected grid with optional bitmap or hash-set blocked cells.
pub fn astar_grid_blocked(
    rows: u32,
    cols: u32,
    start: (u32, u32),
    goal: (u32, u32),
    blocked_bits: Option<&[u8]>,
    blocked_set: Option<&HashSet<u32>>,
) -> Option<Vec<(u32, u32)>> {
    let n_cells = rows as usize * cols as usize;
    if n_cells == 0 || rows == 0 || cols == 0 {
        return None;
    }

    let blocked = |id: u32| {
        is_blocked_bits(blocked_bits, id)
            || blocked_set.is_some_and(|s| is_blocked_hash(s, id))
    };

    let start_id = start.0 * cols + start.1;
    let goal_id = goal.0 * cols + goal.1;
    let goal_r = goal.0;
    let goal_c = goal.1;

    if blocked(start_id) || blocked(goal_id) {
        return None;
    }

    ASTAR_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        scratch.ensure(n_cells);

        // SAFETY: disjoint fields in `AstarScratch`.
        let (g_score, f_score, state, came_from) = unsafe {
            let g_ptr = scratch.g_score.as_mut_ptr();
            let f_ptr = scratch.f_score.as_mut_ptr();
            let s_ptr = scratch.state.as_mut_ptr();
            let p_ptr = scratch.came_from.as_mut_ptr();
            (
                std::slice::from_raw_parts_mut(g_ptr, n_cells),
                std::slice::from_raw_parts_mut(f_ptr, n_cells),
                std::slice::from_raw_parts_mut(s_ptr, n_cells),
                std::slice::from_raw_parts_mut(p_ptr, n_cells),
            )
        };

        g_score.fill(UNVISITED_U32);
        f_score.fill(0);
        state.fill(STATE_UNSEEN);
        came_from.fill(NO_PARENT_U32);

        let h0 = manhattan(start.0, start.1, goal_r, goal_c);
        g_score[start_id as usize] = 0;
        f_score[start_id as usize] = h0;
        state[start_id as usize] = STATE_OPEN;

        let mut open_heap = BinaryHeap::new();
        open_heap.push(Reverse((h0, start_id)));

        let neighbors_delta: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

        while let Some(Reverse((current_f, current))) = open_heap.pop() {
            if current_f != f_score[current as usize] {
                continue;
            }
            if state[current as usize] == STATE_CLOSED {
                continue;
            }
            if current == goal_id {
                return Some(reconstruct_path(came_from, current, cols));
            }
            state[current as usize] = STATE_CLOSED;

            let r = current / cols;
            let c = current % cols;

            for (dr, dc) in neighbors_delta {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if nr < 0 || nc < 0 || nr >= rows as i32 || nc >= cols as i32 {
                    continue;
                }
                let neighbor = (nr as u32) * cols + (nc as u32);
                if blocked(neighbor) || state[neighbor as usize] == STATE_CLOSED {
                    continue;
                }
                let tentative_g = g_score[current as usize].saturating_add(1);
                let g_nei = g_score[neighbor as usize];
                if g_nei == UNVISITED_U32 || tentative_g < g_nei {
                    came_from[neighbor as usize] = current;
                    g_score[neighbor as usize] = tentative_g;
                    let h = manhattan(nr as u32, nc as u32, goal_r, goal_c);
                    let f = tentative_g.saturating_add(h);
                    f_score[neighbor as usize] = f;
                    if state[neighbor as usize] != STATE_OPEN {
                        state[neighbor as usize] = STATE_OPEN;
                        open_heap.push(Reverse((f, neighbor)));
                    }
                }
            }
        }
        None
    })
}

/// Run A* reusing caller-owned scratch buffers (same layout as internal vecs).
pub fn astar_grid_on_buffers(
    rows: u32,
    cols: u32,
    start: (u32, u32),
    goal: (u32, u32),
    blocked_bits: Option<&[u8]>,
    g_score: &mut [i32],
    f_score: &mut [i32],
    came_from: &mut [i32],
    closed: &mut [u8],
    open_heap: &mut BinaryHeap<Reverse<(u32, u32)>>,
) -> Option<Vec<(u32, u32)>> {
    let n_cells = rows as usize * cols as usize;
    if n_cells == 0 || g_score.len() < n_cells {
        return None;
    }

    let blocked = |id: u32| blocked_bits.is_some_and(|bits| bitmap_blocked(bits, id));

    let start_id = start.0 * cols + start.1;
    let goal_id = goal.0 * cols + goal.1;
    let goal_r = goal.0;
    let goal_c = goal.1;

    if blocked(start_id) || blocked(goal_id) {
        return None;
    }

    const INF: i32 = i32::MAX / 2;
    g_score.fill(INF);
    f_score.fill(0);
    came_from.fill(-1);
    closed.fill(0);
    open_heap.clear();

    let h0 = manhattan(start.0, start.1, goal_r, goal_c) as i32;
    g_score[start_id as usize] = 0;
    f_score[start_id as usize] = h0;
    open_heap.push(Reverse((h0 as u32, start_id)));

    let neighbors_delta: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    while let Some(Reverse((current_f, current))) = open_heap.pop() {
        if current_f as i32 != f_score[current as usize] {
            continue;
        }
        if closed[current as usize] != 0 {
            continue;
        }
        if current == goal_id {
            return Some(reconstruct_path_i32(came_from, current, cols));
        }
        closed[current as usize] = 1;

        let r = current / cols;
        let c = current % cols;

        for (dr, dc) in neighbors_delta {
            let nr = r as i32 + dr;
            let nc = c as i32 + dc;
            if nr < 0 || nc < 0 || nr >= rows as i32 || nc >= cols as i32 {
                continue;
            }
            let neighbor = (nr as u32) * cols + (nc as u32);
            if blocked(neighbor) || closed[neighbor as usize] != 0 {
                continue;
            }
            let tentative_g = g_score[current as usize].saturating_add(1);
            if tentative_g < g_score[neighbor as usize] {
                came_from[neighbor as usize] = current as i32;
                g_score[neighbor as usize] = tentative_g;
                let h = manhattan(nr as u32, nc as u32, goal_r, goal_c) as i32;
                let f = tentative_g.saturating_add(h);
                f_score[neighbor as usize] = f;
                open_heap.push(Reverse((f as u32, neighbor)));
            }
        }
    }
    None
}

fn reconstruct_path(came_from: &[u32], mut current: u32, cols: u32) -> Vec<(u32, u32)> {
    let mut path = Vec::new();
    loop {
        let r = current / cols;
        let c = current % cols;
        path.push((r, c));
        let parent = came_from[current as usize];
        if parent == NO_PARENT_U32 {
            break;
        }
        current = parent;
    }
    path.reverse();
    path
}

fn reconstruct_path_i32(came_from: &[i32], mut current: u32, cols: u32) -> Vec<(u32, u32)> {
    let mut path = Vec::new();
    loop {
        let r = current / cols;
        let c = current % cols;
        path.push((r, c));
        let parent = came_from[current as usize];
        if parent < 0 {
            break;
        }
        current = parent as u32;
    }
    path.reverse();
    path
}

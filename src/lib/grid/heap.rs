//! Min-heap for grid A* — stores `(f_at_push, node_id)` for lazy deletion.

/// Push `(f[node], node)`; order by stored `f` key (min-heap).
pub fn heap_push(entries: &mut Vec<(i32, u32)>, node: u32, f: &[i32]) {
    let key = f.get(node as usize).copied().unwrap_or(i32::MAX);
    heap_push_key(entries, node, key);
}

/// Push with explicit priority key (for native paths that already read `f[node]`).
pub fn heap_push_key(entries: &mut Vec<(i32, u32)>, node: u32, key: i32) {
    entries.push((key, node));
    let mut i = entries.len() - 1;
    while i > 0 {
        let parent = (i - 1) / 2;
        if entries[i].0 < entries[parent].0 {
            entries.swap(i, parent);
            i = parent;
        } else {
            break;
        }
    }
}

/// Pop min entry; returns `(f_at_push, node)` or `None` if empty.
pub fn heap_pop(entries: &mut Vec<(i32, u32)>) -> Option<(i32, u32)> {
    if entries.is_empty() {
        return None;
    }
    let last = entries.pop().unwrap();
    if entries.is_empty() {
        return Some(last);
    }
    let root = entries[0];
    entries[0] = last;
    let mut i = 0;
    loop {
        let left = 2 * i + 1;
        let right = left + 1;
        let mut smallest = i;
        if left < entries.len() && entries[left].0 < entries[smallest].0 {
            smallest = left;
        }
        if right < entries.len() && entries[right].0 < entries[smallest].0 {
            smallest = right;
        }
        if smallest == i {
            break;
        }
        entries.swap(i, smallest);
        i = smallest;
    }
    Some(root)
}

pub fn heap_len(entries: &[(i32, u32)]) -> usize {
    entries.len()
}

pub fn heap_clear(entries: &mut Vec<(i32, u32)>) {
    entries.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_pop_order() {
        let mut f = vec![10, 5, 8, 3, 2];
        let mut h = Vec::new();
        for n in 0..5u32 {
            heap_push(&mut h, n, &f);
        }
        let mut out = Vec::new();
        while let Some((_, n)) = heap_pop(&mut h) {
            out.push(n);
        }
        assert_eq!(out, vec![4, 3, 1, 2, 0]);
    }
}

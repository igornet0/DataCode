//! Min-heap on [`Vec<TaggedValue>`] (VM array cells) — shared by inline fast path and tests.

use std::cmp::Ordering;

use crate::common::TaggedValue;

/// Compare two heap slots (caller supplies ordering, typically via [`value_partial_cmp`] on materialized values).
pub type SlotCompareFn<'a> = dyn FnMut(TaggedValue, TaggedValue) -> Result<Ordering, String> + 'a;

#[inline]
fn swap(heap: &mut [TaggedValue], i: usize, j: usize) {
    heap.swap(i, j);
}

pub fn sift_up<F>(heap: &mut [TaggedValue], mut index: usize, cmp: &mut F) -> Result<(), String>
where
    F: FnMut(TaggedValue, TaggedValue) -> Result<Ordering, String>,
{
    while index > 0 {
        let parent = (index - 1) / 2;
        match cmp(heap[parent], heap[index])? {
            Ordering::Greater => {
                swap(heap, parent, index);
                index = parent;
            }
            _ => break,
        }
    }
    Ok(())
}

pub fn sift_down<F>(heap: &mut [TaggedValue], mut index: usize, cmp: &mut F) -> Result<(), String>
where
    F: FnMut(TaggedValue, TaggedValue) -> Result<Ordering, String>,
{
    let n = heap.len();
    loop {
        let left = 2 * index + 1;
        let right = 2 * index + 2;
        let mut smallest = index;

        if left < n {
            match cmp(heap[left], heap[smallest])? {
                Ordering::Less => smallest = left,
                _ => {}
            }
        }
        if right < n {
            match cmp(heap[right], heap[smallest])? {
                Ordering::Less => smallest = right,
                _ => {}
            }
        }

        if smallest == index {
            break;
        }
        swap(heap, index, smallest);
        index = smallest;
    }
    Ok(())
}

/// Push `item` onto the heap (sift-up). Returns error message on incomparable elements.
pub fn heappush<F>(heap: &mut Vec<TaggedValue>, item: TaggedValue, cmp: &mut F) -> Result<(), String>
where
    F: FnMut(TaggedValue, TaggedValue) -> Result<Ordering, String>,
{
    heap.push(item);
    let idx = heap.len() - 1;
    sift_up(heap, idx, cmp)
}

/// Pop the smallest element. Returns `Err` message for empty heap or compare failure.
pub fn heappop<F>(heap: &mut Vec<TaggedValue>, cmp: &mut F) -> Result<TaggedValue, String>
where
    F: FnMut(TaggedValue, TaggedValue) -> Result<Ordering, String>,
{
    let n = heap.len();
    if n == 0 {
        return Err("IndexError: heappop from empty heap".to_string());
    }
    if n == 1 {
        return Ok(heap.pop().unwrap_or(TaggedValue::null()));
    }
    let root = heap[0];
    heap[0] = heap.pop().unwrap();
    sift_down(heap, 0, cmp)?;
    Ok(root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::TaggedValue;

    #[test]
    fn slot_heap_numeric_order() {
        let mut heap = Vec::new();
        let mut cmp = |a: TaggedValue, b: TaggedValue| -> Result<Ordering, String> {
            Ok(a.get_f64()
                .partial_cmp(&b.get_f64())
                .unwrap_or(Ordering::Equal))
        };
        heappush(&mut heap, TaggedValue::from_f64(5.0), &mut cmp).unwrap();
        heappush(&mut heap, TaggedValue::from_f64(1.0), &mut cmp).unwrap();
        heappush(&mut heap, TaggedValue::from_f64(3.0), &mut cmp).unwrap();
        assert_eq!(heappop(&mut heap, &mut cmp).unwrap().get_f64(), 1.0);
        assert_eq!(heappop(&mut heap, &mut cmp).unwrap().get_f64(), 3.0);
        assert_eq!(heappop(&mut heap, &mut cmp).unwrap().get_f64(), 5.0);
    }
}

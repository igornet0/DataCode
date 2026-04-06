//! Python-style slice index normalization for 1D sequences.

/// `step` must be non-zero. Returns indices in traversal order (copy order).
pub fn slice_indices(
    len: usize,
    start: Option<i64>,
    stop: Option<i64>,
    step: i64,
) -> Result<Vec<usize>, &'static str> {
    if step == 0 {
        return Err("slice step cannot be zero");
    }
    let n = len as i64;
    if n == 0 {
        return Ok(Vec::new());
    }
    if step > 0 {
        let mut a = start.unwrap_or(0);
        let mut b = stop.unwrap_or(n);
        if a < 0 {
            a += n;
        }
        if b < 0 {
            b += n;
        }
        if a < 0 {
            a = 0;
        }
        if b > n {
            b = n;
        }
        if a > n {
            a = n;
        }
        if b < 0 {
            b = 0;
        }
        let mut out = Vec::new();
        let mut i = a;
        while i < b {
            out.push(i as usize);
            i += step;
        }
        Ok(out)
    } else {
        let mut a = match start {
            None => n - 1,
            Some(mut x) => {
                if x < 0 {
                    x += n;
                }
                x
            }
        };
        let mut b = match stop {
            None => -1,
            Some(mut x) => {
                if x < 0 {
                    x += n;
                }
                x
            }
        };
        if a < -1 {
            a = -1;
        }
        if a >= n {
            a = n - 1;
        }
        if b < -1 {
            b = -1;
        }
        if b >= n {
            b = n - 1;
        }
        let mut out = Vec::new();
        let mut i = a;
        while i > b {
            if i >= 0 && i < n {
                out.push(i as usize);
            }
            i += step;
        }
        Ok(out)
    }
}

/// Границы полуинтервала `[a, b)` для `step == 1` (присваивание срезу / splice).
pub fn contiguous_positive_slice_bounds(
    len: usize,
    start: Option<i64>,
    stop: Option<i64>,
) -> (usize, usize) {
    let n = len as i64;
    let mut a = start.unwrap_or(0);
    let mut b = stop.unwrap_or(n);
    if a < 0 {
        a += n;
    }
    if b < 0 {
        b += n;
    }
    if a < 0 {
        a = 0;
    }
    if b > n {
        b = n;
    }
    if a > n {
        a = n;
    }
    if b < 0 {
        b = 0;
    }
    (a as usize, b as usize)
}

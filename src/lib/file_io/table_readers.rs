//! Tabular file readers (CSV, XLSX).

use crate::common::table::Table;
use crate::common::typed_column::{ColumnBuilder, TypedColumn};
use crate::common::value::Value;
use crate::vm::natives::table::try_parse_date;
use std::io;
use std::rc::Rc;

pub fn read_csv_bytes(content: &[u8], trim: bool) -> Result<Table, io::Error> {
    read_csv_bytes_inner(content, true, trim)
}

/// Read CSV with every row as data (no separate header row). Used before transpose.
pub fn read_csv_bytes_raw(content: &[u8], trim: bool) -> Result<Table, io::Error> {
    read_csv_bytes_inner(content, false, trim)
}

fn read_csv_bytes_inner(content: &[u8], has_headers: bool, trim: bool) -> Result<Table, io::Error> {
    use csv::ReaderBuilder;
    let mut reader = ReaderBuilder::new()
        .has_headers(has_headers)
        // Поля обрезаются в build_csv_table: `str::trim` даёт то же, что Trim::All
        // (ASCII-обрезка байтов + Unicode-обрезка строки), но без копии записи.
        .trim(if trim { csv::Trim::Headers } else { csv::Trim::None })
        .from_reader(content);
    if content.len() >= PAR_MIN_BYTES {
        let threads = csv_threads();
        if threads > 1 {
            if let Some(t) = build_csv_unquoted_parallel(content, has_headers, trim, threads, PAR_MIN_CHUNK) {
                return Ok(t);
            }
        }
    }
    let cap = estimate_rows(content);
    if let Some(t) = build_csv_unquoted(content, has_headers, trim, cap) {
        return Ok(t);
    }
    build_csv_table(&mut reader, has_headers, trim, cap)
}

/// Same table as `build_csv_table` for the common shape csv parses trivially: no quote
/// bytes, no BOM, every record as wide as the first. Records end at `\r`, `\n` or `\r\n`
/// and empty lines are skipped, as with csv's default terminator. Anything else (a quote,
/// a BOM, ragged rows, invalid UTF-8) returns None and the caller reparses with csv, so
/// results and error messages do not change.
fn build_csv_unquoted(content: &[u8], has_headers: bool, trim: bool, cap: usize) -> Option<Table> {
    if content.starts_with(b"\xEF\xBB\xBF") {
        return None;
    }
    let n = content.len();
    let mut pos = 0usize;
    let mut headers: Option<Vec<String>> = None;
    let mut cols: Vec<ColumnBuilder> = Vec::new();
    let mut width = usize::MAX;
    let mut rows = 0usize;
    loop {
        while pos < n && (content[pos] == b'\n' || content[pos] == b'\r') {
            pos += 1;
        }
        if pos == n {
            break;
        }
        let header_row = has_headers && headers.is_none();
        let mut names: Vec<String> = Vec::new();
        let mut col = 0usize;
        let mut start = pos;
        let mut hi = 0u8;
        loop {
            let c = if pos < n { content[pos] } else { b'\n' };
            match c {
                b',' | b'\n' | b'\r' => {
                    let f = &content[start..pos];
                    let field = if hi < 0x80 {
                        // SAFETY: every byte of `f` is ASCII.
                        unsafe { std::str::from_utf8_unchecked(f) }
                    } else {
                        std::str::from_utf8(f).ok()?
                    };
                    if header_row {
                        names.push(if trim { field.trim() } else { field }.to_string());
                    } else {
                        if col >= width {
                            return None;
                        }
                        if col == cols.len() {
                            cols.push(ColumnBuilder::with_capacity(cap));
                        }
                        let b = &mut cols[col];
                        let field = if trim { trim_field(field) } else { field };
                        if let Some(x) = fast_f64(field) {
                            b.push_number(x);
                        } else if !b.push_known_str(field) {
                            b.push(parse_csv_field(field));
                        }
                    }
                    col += 1;
                    if c != b',' {
                        break;
                    }
                    pos += 1;
                    start = pos;
                    hi = 0;
                }
                b'"' => return None,
                _ => {
                    hi |= c;
                    pos += 1;
                }
            }
        }
        if header_row {
            width = names.len();
            cols.resize_with(width, || ColumnBuilder::with_capacity(cap));
            headers = Some(names);
            continue;
        }
        if width == usize::MAX {
            width = col;
        } else if col != width {
            return None;
        }
        rows += 1;
    }
    let headers = match headers {
        Some(h) => h,
        None if has_headers => return None,
        None => (0..cols.len()).map(|i| format!("Column_{}", i)).collect(),
    };
    if rows == 0 {
        return Some(Table::from_data(Vec::new(), Some(headers)));
    }
    Some(Table::from_typed_columns(
        headers,
        cols.into_iter().map(ColumnBuilder::finish).collect(),
    ))
}

/// Inputs below this size parse faster on one thread than they split and merge.
const PAR_MIN_BYTES: usize = 2 << 20;
/// Smallest block handed to a worker, so per-block setup stays negligible.
const PAR_MIN_CHUNK: usize = 512 << 10;
/// Most threads the column merge uses; see `merge_columns`.
const MERGE_MAX_THREADS: usize = 4;
// Dictionary entries per string-copy job in the merge.
const DICT_STR_JOB: usize = 1 << 14;

/// Worker count for the parallel CSV path: `DATACODE_CSV_THREADS` if set, else the
/// machine's parallelism capped at 10 (parsing stops gaining past that; merge caps lower).
fn csv_threads() -> usize {
    if let Some(n) = std::env::var("DATACODE_CSV_THREADS").ok().and_then(|s| s.trim().parse::<usize>().ok()) {
        return n.max(1);
    }
    std::thread::available_parallelism().map_or(1, |n| n.get()).min(10)
}

/// Fx-style hash for the per-block string maps (keys are borrowed CSV fields).
#[derive(Default)]
struct FieldHasher(u64);

impl std::hash::Hasher for FieldHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        const K: u64 = 0x517c_c1b7_2722_0a95;
        let mut h = self.0 ^ bytes.len() as u64;
        let mut words = bytes.chunks_exact(8);
        for w in &mut words {
            h = (h.rotate_left(5) ^ u64::from_le_bytes(w.try_into().unwrap())).wrapping_mul(K);
        }
        let mut tail = [0u8; 8];
        let rest = words.remainder();
        tail[..rest.len()].copy_from_slice(rest);
        self.0 = (h.rotate_left(5) ^ u64::from_le_bytes(tail)).wrapping_mul(K);
    }

    #[inline]
    fn write_u8(&mut self, _: u8) {}

    #[inline]
    fn finish(&self) -> u64 {
        self.0 ^ (self.0 >> 29)
    }
}

type FieldMap<'a> = std::collections::HashMap<&'a str, u32, std::hash::BuildHasherDefault<FieldHasher>>;

/// Block-map key that carries its hash: a field is hashed once per row, and table
/// growth or probe mismatches never reread the text.
#[derive(Clone, Copy)]
struct HKey<'a> {
    h: u64,
    w: [u64; 2],
    s: &'a str,
}

/// The first 16 bytes of a field, zero-padded.
#[inline]
fn head_words(b: &[u8]) -> [u64; 2] {
    let mut t = [0u8; 16];
    let n = b.len().min(16);
    t[..n].copy_from_slice(&b[..n]);
    let mut w = [0u64; 2];
    for (i, x) in w.iter_mut().enumerate() {
        *x = u64::from_le_bytes(t[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

impl<'a> HKey<'a> {
    #[inline]
    fn new(s: &'a str) -> Self {
        let b = s.as_bytes();
        let w = head_words(b);
        let h = if b.len() <= 16 {
            // Same value as `FieldHasher` over `b`, fed from the padded head words.
            const K: u64 = 0x517c_c1b7_2722_0a95;
            let mix = |h: u64, x: u64| (h.rotate_left(5) ^ x).wrapping_mul(K);
            let h0 = b.len() as u64;
            let h = match b.len() {
                0..=7 => mix(h0, w[0]),
                8..=15 => mix(mix(h0, w[0]), w[1]),
                _ => mix(mix(mix(h0, w[0]), w[1]), 0),
            };
            h ^ (h >> 29)
        } else {
            let mut st = FieldHasher::default();
            std::hash::Hasher::write(&mut st, b);
            std::hash::Hasher::finish(&st)
        };
        HKey { h, w, s }
    }

    /// A key for `s` whose hash `h` is already known.
    #[inline]
    fn with_hash(h: u64, s: &'a str) -> Self {
        HKey { h, w: head_words(s.as_bytes()), s }
    }
}

impl PartialEq for HKey<'_> {
    #[inline]
    fn eq(&self, o: &Self) -> bool {
        self.h == o.h && self.w == o.w && self.s.len() == o.s.len() && (self.s.len() <= 16 || self.s == o.s)
    }
}

impl Eq for HKey<'_> {}

impl std::hash::Hash for HKey<'_> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(self.h);
    }
}

/// Passes an `HKey`'s stored hash through unchanged.
#[derive(Default)]
struct PassHasher(u64);

impl std::hash::Hasher for PassHasher {
    #[inline]
    fn write(&mut self, _: &[u8]) {
        unreachable!("HKey hashes via write_u64")
    }

    #[inline]
    fn write_u64(&mut self, h: u64) {
        self.0 = h;
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

type BlockMap<'a> = std::collections::HashMap<HKey<'a>, u32, std::hash::BuildHasherDefault<PassHasher>>;

/// One column of one block, in the shape `ColumnBuilder` would reach on those rows alone.
/// Workers never build `Value`s (they hold `Rc`s); strings stay borrowed from the input.
enum Part<'a> {
    Null(usize),
    Num {
        vals: Vec<f64>,
        nulls: Option<Vec<bool>>,
    },
    Str {
        codes: Vec<u32>,
        /// Local code -> text, None for the null entry.
        keys: Vec<Option<&'a str>>,
        /// Block row where each local code first occurs.
        first: Vec<usize>,
        map: BlockMap<'a>,
        null: Option<u32>,
    },
}

/// A field `parse_csv_field` turns into `Value::String`.
fn plain_str(f: &str) -> bool {
    !f.is_empty()
        && !matches!(f, "true" | "True" | "false" | "False")
        && f.parse::<f64>().is_err()
        && try_parse_date(f).is_none()
}

impl<'a> Part<'a> {
    fn str_part(lead: usize, cap: usize, hint: usize) -> Self {
        let mut codes = Vec::with_capacity(cap);
        let (keys, first, null) = if lead > 0 {
            codes.resize(lead, 0);
            (vec![None], vec![0], Some(0))
        } else {
            (Vec::new(), Vec::new(), None)
        };
        let map = BlockMap::with_capacity_and_hasher(hint, Default::default());
        Part::Str { codes, keys, first, map, null }
    }

    /// Mirrors `ColumnBuilder` on one field. False when the column would turn Mixed
    /// (or anything the merge does not handle): the caller falls back to one thread.
    #[inline]
    fn push(&mut self, f: &'a str, row: usize, cap: usize, hint: usize) -> bool {
        match self {
            Part::Num { vals, nulls } => {
                let x = match fast_f64(f) {
                    Some(x) => Some(x),
                    None if f.is_empty() => None,
                    None => Some(match f.parse::<f64>() {
                        Ok(x) => x,
                        Err(_) => return false,
                    }),
                };
                push_typed_f64(vals, nulls, x);
            }
            Part::Str { codes, keys, first, map, null } => {
                let k = HKey::new(f);
                if let Some(&c) = map.get(&k) {
                    codes.push(c);
                } else if f.is_empty() {
                    let c = *null.get_or_insert_with(|| {
                        keys.push(None);
                        first.push(row);
                        (keys.len() - 1) as u32
                    });
                    codes.push(c);
                } else if fast_f64(f).is_none() && plain_str(f) {
                    let c = keys.len() as u32;
                    keys.push(Some(f));
                    first.push(row);
                    map.insert(k, c);
                    codes.push(c);
                } else {
                    return false;
                }
            }
            Part::Null(n) => {
                let n = *n;
                if f.is_empty() {
                    *self = Part::Null(n + 1);
                } else if let Some(x) = fast_f64(f).or_else(|| f.parse::<f64>().ok()) {
                    let mut vals = Vec::with_capacity(cap);
                    vals.resize(n, 0.0);
                    let mut nulls = (n > 0).then(|| vec![true; n]);
                    push_typed_f64(&mut vals, &mut nulls, Some(x));
                    *self = Part::Num { vals, nulls };
                } else if plain_str(f) {
                    *self = Part::str_part(n, cap, hint);
                    return self.push(f, row, cap, hint);
                } else {
                    return false;
                }
            }
        }
        true
    }
}

#[inline]
fn push_typed_f64(data: &mut Vec<f64>, nulls: &mut Option<Vec<bool>>, x: Option<f64>) {
    match x {
        Some(x) => {
            data.push(x);
            if let Some(n) = nulls {
                n.push(false);
            }
        }
        None => {
            nulls.get_or_insert_with(|| vec![false; data.len()]).push(true);
            data.push(0.0);
        }
    }
}

/// Splits one unquoted record starting at `pos` into fields; returns them with the
/// position after the record's terminator byte. None on a quote or invalid UTF-8.
fn split_record(content: &[u8], mut pos: usize) -> Option<(Vec<&str>, usize)> {
    let n = content.len();
    let mut out = Vec::new();
    let mut start = pos;
    let mut hi = 0u8;
    loop {
        let c = if pos < n { content[pos] } else { b'\n' };
        match c {
            b',' | b'\n' | b'\r' => {
                let f = &content[start..pos];
                out.push(if hi < 0x80 {
                    // SAFETY: every byte of `f` is ASCII.
                    unsafe { std::str::from_utf8_unchecked(f) }
                } else {
                    std::str::from_utf8(f).ok()?
                });
                pos += 1;
                if c != b',' {
                    return Some((out, pos.min(n)));
                }
                start = pos;
                hi = 0;
            }
            b'"' => return None,
            _ => {
                hi |= c;
                pos += 1;
            }
        }
    }
}

/// Parses the data records of one newline-aligned block into per-column parts.
fn parse_block<'a>(
    buf: &'a [u8],
    width: usize,
    trim: bool,
    cap: usize,
    abort: &std::sync::atomic::AtomicBool,
    hints: &mut [usize],
) -> Option<(usize, Vec<Part<'a>>)> {
    use std::sync::atomic::Ordering;
    let n = buf.len();
    let mut parts: Vec<Part<'a>> = (0..width).map(|_| Part::Null(0)).collect();
    let mut pos = 0usize;
    let mut rows = 0usize;
    loop {
        while pos < n && (buf[pos] == b'\n' || buf[pos] == b'\r') {
            pos += 1;
        }
        if pos == n {
            break;
        }
        if rows & 0xFFF == 0 && abort.load(Ordering::Relaxed) {
            return None;
        }
        let mut col = 0usize;
        let mut start = pos;
        let mut hi = 0u8;
        loop {
            let c = if pos < n { buf[pos] } else { b'\n' };
            match c {
                b',' | b'\n' | b'\r' => {
                    let f = &buf[start..pos];
                    let field = if hi < 0x80 {
                        // SAFETY: every byte of `f` is ASCII.
                        unsafe { std::str::from_utf8_unchecked(f) }
                    } else {
                        std::str::from_utf8(f).ok()?
                    };
                    if col >= width {
                        return None;
                    }
                    let field = if trim { trim_field(field) } else { field };
                    if !parts[col].push(field, rows, cap, hints[col]) {
                        return None;
                    }
                    col += 1;
                    if c != b',' {
                        break;
                    }
                    pos += 1;
                    start = pos;
                    hi = 0;
                }
                b'"' => return None,
                _ => {
                    hi |= c;
                    pos += 1;
                }
            }
        }
        if col != width {
            return None;
        }
        rows += 1;
    }
    // The block map is only needed while pushing: free it now rather than
    // keeping every block's map alive until the merge.
    for (h, p) in hints.iter_mut().zip(&mut parts) {
        if let Part::Str { map, .. } = p {
            *h = map.len();
            *map = BlockMap::default();
        }
    }
    Some((rows, parts))
}


/// `build_csv_unquoted` split across threads: the body is cut into newline-aligned
/// blocks, each parsed into typed parts on its own thread, then the parts are joined
/// in row order on this thread (string dictionaries are merged and codes remapped).
/// Returns None whenever the one-thread path could differ (a column that would turn
/// Mixed mid-way, a quote, a ragged row, ...); the caller then runs that path.
fn build_csv_unquoted_parallel(
    content: &[u8],
    has_headers: bool,
    trim: bool,
    threads: usize,
    min_chunk: usize,
) -> Option<Table> {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    if content.starts_with(b"\xEF\xBB\xBF") {
        return None;
    }
    let n = content.len();
    let mut pos = 0usize;
    while pos < n && (content[pos] == b'\n' || content[pos] == b'\r') {
        pos += 1;
    }
    if pos == n {
        return None;
    }
    let (fields, next) = split_record(content, pos)?;
    let headers: Option<Vec<String>> = has_headers
        .then(|| fields.iter().map(|f| if trim { f.trim() } else { f }.to_string()).collect());
    let width = fields.len();
    if has_headers {
        pos = next;
    }
    let body = &content[pos..];
    if body.len() < min_chunk * 2 {
        return None;
    }

    // Newline-aligned blocks, several per thread so uneven blocks balance out.
    let target = (body.len() / (threads * 4)).max(min_chunk);
    let mut bounds = vec![0usize];
    let mut at = 0usize;
    while body.len() - at > target + target / 2 {
        let probe = at + target;
        match body[probe..].iter().position(|&b| b == b'\n') {
            Some(i) => at = probe + i + 1,
            None => break,
        }
        bounds.push(at);
    }
    bounds.push(body.len());
    let blocks = bounds.len() - 1;
    if blocks < 2 {
        return None;
    }
    let total_est = estimate_rows(body);

    let next_block = AtomicUsize::new(0);
    let abort = AtomicBool::new(false);
    let mut results: Vec<Option<(usize, Vec<Part>)>> = (0..blocks).map(|_| None).collect();
    let workers = threads.min(blocks);
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                s.spawn(|| {
                    let mut done = Vec::new();
                    // Distinct-key count of each string column in this worker's last block:
                    // sizes the next block's map so it skips most growth rehashes.
                    let mut hints = vec![0usize; width];
                    loop {
                        let i = next_block.fetch_add(1, Ordering::Relaxed);
                        if i >= blocks || abort.load(Ordering::Relaxed) {
                            break;
                        }
                        let buf = &body[bounds[i]..bounds[i + 1]];
                        let cap = total_est / blocks + total_est / blocks / 8 + 16;
                        match parse_block(buf, width, trim, cap, &abort, &mut hints) {
                            Some(r) => done.push((i, r)),
                            None => {
                                abort.store(true, Ordering::Relaxed);
                                break;
                            }
                        }
                    }
                    done
                })
            })
            .collect();
        for h in handles {
            match h.join() {
                Ok(done) => {
                    for (i, r) in done {
                        results[i] = Some(r);
                    }
                }
                Err(_) => abort.store(true, Ordering::Relaxed),
            }
        }
    });
    if abort.load(Ordering::Relaxed) {
        return None;
    }
    let mut row_counts = Vec::with_capacity(blocks);
    let mut per_block: Vec<Vec<Part>> = Vec::with_capacity(blocks);
    for r in results {
        let (rows, parts) = r?;
        row_counts.push(rows);
        per_block.push(parts);
    }
    let rows: usize = row_counts.iter().sum();
    let headers =
        headers.unwrap_or_else(|| (0..width).map(|i| format!("Column_{}", i)).collect());
    if rows == 0 {
        return Some(Table::from_data(Vec::new(), Some(headers)));
    }
    // Column-major: take column c out of every block.
    let mut by_col: Vec<Vec<Part>> = (0..width).map(|_| Vec::with_capacity(blocks)).collect();
    for parts in per_block {
        for (c, p) in parts.into_iter().enumerate() {
            by_col[c].push(p);
        }
    }
    let cols = merge_columns(by_col, &row_counts, rows, threads)?;
    let tb = Table::from_typed_columns(headers, cols);
    Some(tb)
}

/// One column once every block's part is seen: either finished on the spot, or a
/// zeroed destination that the copy jobs fill block by block.
enum Plan<'a> {
    Done(TypedColumn),
    Num {
        data: Rc<[f64]>,
        mask: Option<Rc<[bool]>>,
    },
    Dict {
        codes: Rc<[u32]>,
        gdict: Vec<Option<&'a str>>,
        gnull: Option<u32>,
        remaps: Vec<Vec<u32>>,
    },
}

/// One block of one column: its rows and the slice of the final buffer they land in.
enum Job<'s> {
    Num { vals: Vec<f64>, nulls: Option<Vec<bool>>, data: &'s mut [f64], mask: Option<&'s mut [bool]> },
    NullNum { mask: Option<&'s mut [bool]> },
    Codes { local: Vec<u32>, remap: &'s [u32], dst: &'s mut [u32] },
    NullCodes { code: u32, dst: &'s mut [u32] },
    Strs { keys: &'s [Option<&'s str>], dst: &'s mut [String] },
}

impl Job<'_> {
    fn run(self) {
        match self {
            Job::Num { vals, nulls, data, mask } => {
                data.copy_from_slice(&vals);
                if let (Some(m), Some(nl)) = (mask, nulls) {
                    m.copy_from_slice(&nl);
                }
            }
            // The data slice is already 0.0, as `ColumnBuilder` leaves null slots.
            Job::NullNum { mask } => {
                if let Some(m) = mask {
                    m.fill(true);
                }
            }
            Job::Codes { local, remap, dst } => {
                for (d, &c) in dst.iter_mut().zip(&local) {
                    *d = remap[c as usize];
                }
            }
            Job::NullCodes { code, dst } => dst.fill(code),
            Job::Strs { keys, dst } => {
                for (d, k) in dst.iter_mut().zip(keys) {
                    if let Some(s) = k {
                        *d = (*s).to_string();
                    }
                }
            }
        }
    }
}

/// Joins every column's block parts into the columns `ColumnBuilder::finish` would give
/// for all rows in order. None if the blocks disagree on a column's type. Types and
/// dictionaries are settled here; the bulk copy (and freeing the block buffers) runs as
/// column x block jobs on up to `threads` threads, each writing its own final slice.
fn merge_columns(
    by_col: Vec<Vec<Part>>,
    row_counts: &[usize],
    rows: usize,
    threads: usize,
) -> Option<Vec<TypedColumn>> {
    use crate::common::typed_column::F64Kind;
    use std::sync::Mutex;
    let mut plans = Vec::with_capacity(by_col.len());
    let mut sources = Vec::with_capacity(by_col.len());
    for parts in by_col {
        let (plan, src) = plan_column(parts, row_counts, rows, threads.min(MERGE_MAX_THREADS))?;
        plans.push(plan);
        sources.push(src);
    }
    // Dictionary strings are owned in jobs too, so the copies of a large dictionary
    // overlap the code remaps instead of following them on one thread.
    let mut strs: Vec<Vec<String>> = plans
        .iter()
        .map(|p| match p {
            Plan::Dict { gdict, .. } => vec![String::new(); gdict.len()],
            _ => Vec::new(),
        })
        .collect();
    let mut jobs: Vec<Job> = Vec::new();
    for ((plan, parts), strs) in plans.iter_mut().zip(sources).zip(strs.iter_mut()) {
        match plan {
            Plan::Done(_) => {}
            Plan::Num { data, mask } => {
                let mut data = Rc::get_mut(data).expect("fresh buffer");
                let mut mask = mask.as_mut().map(|m| Rc::get_mut(m).expect("fresh buffer"));
                for (p, &k) in parts.into_iter().zip(row_counts) {
                    let (d, rest) = std::mem::take(&mut data).split_at_mut(k);
                    data = rest;
                    let m = mask.take().map(|m| {
                        let (a, b) = m.split_at_mut(k);
                        mask = Some(b);
                        a
                    });
                    jobs.push(match p {
                        Part::Num { vals, nulls } => Job::Num { vals, nulls, data: d, mask: m },
                        _ => Job::NullNum { mask: m },
                    });
                }
            }
            Plan::Dict { codes, gdict, gnull, remaps } => {
                for (keys, dst) in gdict.chunks(DICT_STR_JOB).zip(strs.chunks_mut(DICT_STR_JOB)) {
                    jobs.push(Job::Strs { keys, dst });
                }
                let mut dst = Rc::get_mut(codes).expect("fresh buffer");
                for ((p, &k), remap) in parts.into_iter().zip(row_counts).zip(remaps.iter()) {
                    let (d, rest) = std::mem::take(&mut dst).split_at_mut(k);
                    dst = rest;
                    jobs.push(match p {
                        Part::Str { codes, .. } => Job::Codes { local: codes, remap, dst: d },
                        _ => Job::NullCodes { code: gnull.unwrap_or(0), dst: d },
                    });
                }
            }
        }
    }
    // The copies mostly fault in fresh zero pages, which stops scaling past ~4 threads
    // (M4, 3M rows: 1 → 54 ms load, 4 → 43 ms, 10 → 48 ms).
    let workers = threads.min(MERGE_MAX_THREADS).min(jobs.len());
    if workers > 1 {
        let queue = Mutex::new(jobs);
        std::thread::scope(|s| {
            for _ in 0..workers {
                s.spawn(|| loop {
                    let job = queue.lock().unwrap_or_else(|e| e.into_inner()).pop();
                    match job {
                        Some(j) => j.run(),
                        None => break,
                    }
                });
            }
        });
    } else {
        jobs.into_iter().for_each(Job::run);
    }
    Some(
        plans
            .into_iter()
            .zip(strs)
            .map(|(plan, strs)| match plan {
                Plan::Done(c) => c,
                Plan::Num { data, mask } => {
                    TypedColumn::f64_of(F64Kind::Number, data, mask)
                }
                Plan::Dict { codes, gdict, gnull, .. } => {
                    let dict: Rc<[Value]> = gdict
                        .into_iter()
                        .zip(strs)
                        .map(|(k, s)| match k {
                            Some(_) => Value::String(s),
                            None => Value::Null,
                        })
                        .collect();
                    TypedColumn::dict(codes, dict, gnull)
                }
            })
            .collect(),
    )
}

/// Settles one column's type (and string dictionary) from its block parts. Columns the
/// copy jobs fill come back with the parts they read; the rest are finished here.
fn plan_column<'a>(
    parts: Vec<Part<'a>>,
    row_counts: &[usize],
    rows: usize,
    workers: usize,
) -> Option<(Plan<'a>, Vec<Part<'a>>)> {
    use crate::common::typed_column::DICT_SPARSE;
    let has_num = parts.iter().any(|p| matches!(p, Part::Num { .. }));
    let has_str = parts.iter().any(|p| matches!(p, Part::Str { .. }));
    if has_num && has_str {
        return None;
    }
    if !has_num && !has_str {
        return Some((Plan::Done(TypedColumn::mixed(Rc::new(vec![Value::Null; rows]))), Vec::new()));
    }
    if has_num {
        let any_null = parts.iter().any(|p| match p {
            Part::Null(k) => *k > 0,
            Part::Num { nulls, .. } => nulls.is_some(),
            Part::Str { .. } => false,
        });
        // Zeroed buffers are mapped lazily, so the pages are first touched by the jobs.
        let plan = Plan::Num { data: zeroed(rows), mask: any_null.then(|| zeroed(rows)) };
        return Some((plan, parts));
    }

    // Strings: the builder's dictionary lists entries (null included) in order of first
    // occurrence, so walking blocks in order and each block's local codes in order
    // rebuilds it; the id-like probe is checked at each new string as the builder does.
    let keys: usize = parts.iter().map(|p| if let Part::Str { keys, .. } = p { keys.len() } else { 0 }).sum();
    let plan = if workers > 1 && keys >= DICT_PAR_KEYS && !probe_trips_early(&parts) {
        plan_dict_par(&parts, row_counts, workers)
    } else {
        plan_dict_seq(&parts, row_counts)
    };
    let mixed = plan.is_none();
    let (gdict, gnull, remaps) = plan.unwrap_or_default();
    let strings = gdict.len() - gnull.is_some() as usize;
    if mixed || (strings > DICT_SPARSE && strings * 2 > rows) {
        let mut out = Vec::with_capacity(rows);
        for p in &parts {
            match p {
                Part::Null(k) => out.resize(out.len() + *k, Value::Null),
                Part::Str { codes, keys, .. } => out.extend(codes.iter().map(|&c| match keys[c as usize] {
                    Some(s) => Value::String(s.to_string()),
                    None => Value::Null,
                })),
                Part::Num { .. } => unreachable!(),
            }
        }
        return Some((Plan::Done(TypedColumn::mixed(Rc::new(out))), Vec::new()));
    }
    Some((Plan::Dict { codes: zeroed(rows), gdict, gnull, remaps }, parts))
}

/// A string column's global dictionary (`None` is the null entry), its null code, and
/// each block's local-to-global code remap; `None` when the id-like probe trips.
type DictPlan<'a> = Option<(Vec<Option<&'a str>>, Option<u32>, Vec<Vec<u32>>)>;

/// Local keys from which a string dictionary is planned by hash partitions in parallel.
const DICT_PAR_KEYS: usize = 1 << 18;

/// The builder's id-like probe: the `n`-th distinct entry first seen at row `r`.
fn probe_trips(n: usize, r: usize) -> bool {
    use crate::common::typed_column::DICT_PROBE;
    r >= DICT_PROBE && n * 20 >= r * 19
}

/// Whether the probe already trips within the first block, where global and local
/// codes coincide: such id-like columns stop early on the sequential path.
fn probe_trips_early(parts: &[Part]) -> bool {
    match parts.first() {
        Some(Part::Str { keys, first, .. }) => {
            keys.iter().zip(first).enumerate().any(|(i, (k, &r))| k.is_some() && probe_trips(i, r))
        }
        _ => false,
    }
}

fn plan_dict_seq<'a>(parts: &[Part<'a>], row_counts: &[usize]) -> DictPlan<'a> {
    let mut gmap: FieldMap = FieldMap::default();
    let mut gdict: Vec<Option<&str>> = Vec::new();
    let mut gnull: Option<u32> = None;
    let mut remaps: Vec<Vec<u32>> = Vec::with_capacity(parts.len());
    let mut offset = 0usize;
    for (p, &k) in parts.iter().zip(row_counts) {
        match p {
            Part::Null(n) => {
                if *n > 0 {
                    gnull.get_or_insert_with(|| {
                        gdict.push(None);
                        (gdict.len() - 1) as u32
                    });
                }
                remaps.push(Vec::new());
            }
            Part::Str { keys, first, .. } => {
                let mut remap = Vec::with_capacity(keys.len());
                for (key, &row) in keys.iter().zip(first) {
                    let g = match key {
                        None => *gnull.get_or_insert_with(|| {
                            gdict.push(None);
                            (gdict.len() - 1) as u32
                        }),
                        Some(s) => match gmap.get(s) {
                            Some(&c) => c,
                            None => {
                                if probe_trips(gdict.len(), offset + row) {
                                    return None;
                                }
                                let c = gdict.len() as u32;
                                gdict.push(Some(s));
                                gmap.insert(s, c);
                                c
                            }
                        },
                    };
                    remap.push(g);
                }
                remaps.push(remap);
            }
            Part::Num { .. } => unreachable!(),
        }
        offset += k;
    }
    Some((gdict, gnull, remaps))
}

/// The same plan as `plan_dict_seq`, for columns with many distinct strings, where the
/// single global map is bound by cache misses. Keys are split by hash into one partition
/// per worker; each worker numbers its partition's strings in its own small map, walking
/// blocks in order. A final linear pass then hands out global codes in order of first
/// occurrence, so the dictionary, the codes and the probe come out exactly as before.
fn plan_dict_par<'a>(parts: &[Part<'a>], row_counts: &[usize], workers: usize) -> DictPlan<'a> {
    const NULL: u8 = u8::MAX;

    // Each key's partition, and each block's key indices and hashes per partition, blocks
    // shared out among the workers. Partitions take the hash's high bits: the maps index
    // buckets by the low ones.
    // Also each block's first null key, if any.
    type KeyParts = (Vec<u8>, Vec<Vec<u32>>, Vec<Vec<u64>>, Option<u32>);
    let mut kp: Vec<KeyParts> = vec![(Vec::new(), Vec::new(), Vec::new(), None); parts.len()];
    std::thread::scope(|sc| {
        let chunk = parts.len().div_ceil(workers);
        for (ps, out) in parts.chunks(chunk).zip(kp.chunks_mut(chunk)) {
            sc.spawn(move || {
                for (p, (kp, idx, hs, null_at)) in ps.iter().zip(out) {
                    if let Part::Str { keys, .. } = p {
                        let cap = keys.len() / workers + 16;
                        *idx = vec![Vec::with_capacity(cap); workers];
                        *hs = vec![Vec::with_capacity(cap); workers];
                        *kp = keys
                            .iter()
                            .enumerate()
                            .map(|(i, k)| match k {
                                Some(s) => {
                                    let h = HKey::new(s).h;
                                    let q = ((h >> 32) % workers as u64) as usize;
                                    idx[q].push(i as u32);
                                    hs[q].push(h);
                                    q as u8
                                }
                                None => {
                                    null_at.get_or_insert(i as u32);
                                    NULL
                                }
                            })
                            .collect();
                    }
                }
            });
        }
    });

    // Per partition: each block's keys of that partition, as partition-local ids, and the
    // partition's distinct keys in order of first appearance, as (block, key index): a
    // key's local id is its place in that list. The hashes from above are reused; `HKey`
    // still compares the text on equal hashes.
    let mut ids: Vec<(Vec<Vec<u32>>, Vec<(u32, u32)>)> = vec![(Vec::new(), Vec::new()); workers];
    std::thread::scope(|sc| {
        for (q, out) in ids.iter_mut().enumerate() {
            let kp = &kp;
            sc.spawn(move || {
                let mut map = BlockMap::default();
                out.0 = parts
                    .iter()
                    .zip(kp)
                    .enumerate()
                    .map(|(b, (p, (_, idx, hs, _)))| match p {
                        Part::Str { keys, .. } => idx[q]
                            .iter()
                            .zip(&hs[q])
                            .map(|(&i, &h)| {
                                let n = map.len() as u32;
                                let id = *map.entry(HKey::with_hash(h, keys[i as usize].unwrap())).or_insert(n);
                                if id == n {
                                    out.1.push((b as u32, i));
                                }
                                id
                            })
                            .collect(),
                        _ => Vec::new(),
                    })
                    .collect();
            });
        }
    });
    // Global codes in order of first appearance: only the new keys are walked, block by
    // block, merging the partitions' lists by key index, the first null among them.
    let mut glob: Vec<Vec<u32>> = ids.iter().map(|(_, f)| vec![u32::MAX; f.len()]).collect();
    let mut pos = vec![0usize; workers];
    let mut gdict: Vec<Option<&str>> = Vec::new();
    let mut gnull: Option<u32> = None;
    let mut offset = 0usize;
    for (b, (p, &k)) in parts.iter().zip(row_counts).enumerate() {
        match p {
            Part::Null(n) => {
                if *n > 0 {
                    gnull.get_or_insert_with(|| {
                        gdict.push(None);
                        (gdict.len() - 1) as u32
                    });
                }
            }
            Part::Str { keys, first, .. } => {
                let mut null_at = if gnull.is_none() { kp[b].3 } else { None };
                loop {
                    let mut next: Option<(u32, usize)> = None;
                    for (q, (_, fresh)) in ids.iter().enumerate() {
                        if let Some(&(fb, i)) = fresh.get(pos[q]) {
                            if fb as usize == b && next.is_none_or(|(j, _)| i < j) {
                                next = Some((i, q));
                            }
                        }
                    }
                    if null_at.is_some_and(|z| next.is_none_or(|(i, _)| z < i)) {
                        gnull = Some(gdict.len() as u32);
                        gdict.push(None);
                        null_at = None;
                        continue;
                    }
                    let Some((i, q)) = next else { break };
                    if probe_trips(gdict.len(), offset + first[i as usize]) {
                        return None;
                    }
                    glob[q][pos[q]] = gdict.len() as u32;
                    gdict.push(keys[i as usize]);
                    pos[q] += 1;
                }
            }
            Part::Num { .. } => unreachable!(),
        }
        offset += k;
    }

    // Each block's remap, blocks shared out among the workers.
    let mut remaps: Vec<Vec<u32>> = vec![Vec::new(); parts.len()];
    std::thread::scope(|sc| {
        let chunk = parts.len().div_ceil(workers).max(1);
        for (c, out) in remaps.chunks_mut(chunk).enumerate() {
            let (kp, ids, glob) = (&kp, &ids, &glob);
            sc.spawn(move || {
                let mut cur = vec![0usize; workers];
                for (j, remap) in out.iter_mut().enumerate() {
                    let b = c * chunk + j;
                    if !matches!(parts[b], Part::Str { .. }) {
                        continue;
                    }
                    cur.fill(0);
                    *remap = kp[b]
                        .0
                        .iter()
                        .map(|&q| match q {
                            NULL => gnull.unwrap(),
                            q => {
                                let q = q as usize;
                                let g = glob[q][ids[q].0[b][cur[q]] as usize];
                                cur[q] += 1;
                                g
                            }
                        })
                        .collect();
                }
            });
        }
    });
    Some((gdict, gnull, remaps))
}

/// Element types whose all-zero bit pattern is a valid value (0.0, false, 0).
trait ZeroValid: Copy {}
impl ZeroValid for f64 {}
impl ZeroValid for bool {}
impl ZeroValid for u32 {}

/// A final column buffer of `n` zeros, allocated zeroed so its pages are first touched
/// by the job that writes them, and filled in place: `Rc::from(Vec)` would copy it.
fn zeroed<T: ZeroValid>(n: usize) -> Rc<[T]> {
    // SAFETY: `ZeroValid` types are valid when all-zero, which `new_zeroed_slice` gives.
    // The safe `repeat_n(..).collect()` writes every page on the calling thread first,
    // which costs the merge 2-20 ms at 3M rows.
    unsafe { Rc::<[T]>::new_zeroed_slice(n).assume_init() }
}

/// Row count guess from the newline density of the first 64 KiB, so column buffers
/// are sized once instead of doubling (a reserve only maps pages; it does not touch them).
fn estimate_rows(content: &[u8]) -> usize {
    let sample = &content[..content.len().min(1 << 16)];
    let lines = sample.iter().filter(|&&b| b == b'\n').count().max(1);
    let est = (content.len() as u128 * lines as u128 / sample.len().max(1) as u128) as usize;
    (est + est / 16 + 16).max(1024)
}

fn build_csv_table<R: std::io::Read>(
    reader: &mut csv::Reader<R>,
    has_headers: bool,
    trim: bool,
    cap: usize,
) -> Result<Table, io::Error> {
    let headers: Option<Vec<String>> = if has_headers {
        Some(reader.headers()?.iter().map(|s| s.to_string()).collect())
    } else {
        None
    };
    // Каждая колонка собирается своим ColumnBuilder: числа ложатся в f64-буфер без Value,
    // смешанные колонки откатываются в Mixed. Короткие строки дополняются Null; строка
    // шире текущей (редкость) добавляет колонки, заполненные Null для прочитанных строк.
    let mut cols: Vec<ColumnBuilder> = Vec::new();
    let width0 = headers.as_ref().map_or(0, |h| h.len());
    cols.resize_with(width0, || ColumnBuilder::with_capacity(cap));
    let mut rows = 0usize;
    let mut record = csv::StringRecord::new();
    while reader.read_record(&mut record)? {
        while cols.len() < record.len() {
            let mut b = ColumnBuilder::with_capacity(rows.max(cap));
            for _ in 0..rows {
                b.push(Value::Null);
            }
            cols.push(b);
        }
        for (b, field) in cols.iter_mut().zip(record.iter()) {
            let field = if trim { trim_field(field) } else { field };
            if let Some(x) = fast_f64(field) {
                b.push_number(x);
            // Текст, уже лежащий в словаре колонки, всегда разбирается в ту же строку.
            } else if !b.push_known_str(field) {
                b.push(parse_csv_field(field));
            }
        }
        for b in cols.iter_mut().skip(record.len()) {
            b.push(Value::Null);
        }
        rows += 1;
    }
    let width = cols.len();
    let headers =
        headers.unwrap_or_else(|| (0..width).map(|i| format!("Column_{}", i)).collect());
    if rows == 0 {
        return Ok(Table::from_data(Vec::new(), Some(headers)));
    }
    if headers.len() != width {
        // Строки шире заголовка: прежний row-major путь сохраняет поведение как было.
        let cols: Vec<TypedColumn> = cols.into_iter().map(ColumnBuilder::finish).collect();
        let mut flat = Vec::with_capacity(rows * width);
        for r in 0..rows {
            flat.extend(cols.iter().map(|c| c.cell(r)));
        }
        return Ok(Table::from_flat_owned(flat, width, headers));
    }
    Ok(Table::from_typed_columns(
        headers,
        cols.into_iter().map(ColumnBuilder::finish).collect(),
    ))
}

fn pad_rows(rows: &mut [Vec<Value>], width: usize) {
    for row in rows.iter_mut() {
        while row.len() < width {
            row.push(Value::Null);
        }
    }
}

/// `str::trim` with a shortcut: a field whose first and last bytes are visible ASCII
/// has nothing to trim (every Unicode whitespace char starts with a byte <= 0x20 or >= 0x80).
#[inline]
fn trim_field(s: &str) -> &str {
    let visible = |b: Option<&u8>| b.map_or(true, |c| (0x21..0x80).contains(c));
    let b = s.as_bytes();
    if visible(b.first()) && visible(b.last()) {
        s
    } else {
        s.trim()
    }
}

const POW10: [f64; 16] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
];

/// Exact parse of `-?digits(.digits)?` with at most 15 digits, bit-identical to
/// `str::parse::<f64>`: the mantissa and the power of ten are exact f64s, so one
/// IEEE division rounds the true quotient correctly. Anything else returns None.
#[inline]
fn fast_f64(s: &str) -> Option<f64> {
    let b = s.as_bytes();
    let (neg, d) = match b.first()? {
        b'-' => (true, &b[1..]),
        _ => (false, b),
    };
    if d.is_empty() || d.len() > 16 {
        return None;
    }
    let mut m = 0u64;
    let mut digits = 0;
    let mut frac = None;
    for (i, &c) in d.iter().enumerate() {
        match c {
            b'0'..=b'9' => {
                m = m * 10 + (c - b'0') as u64;
                digits += 1;
            }
            b'.' if frac.is_none() && i > 0 => frac = Some(i + 1),
            _ => return None,
        }
    }
    let k = match frac {
        Some(at) if at == d.len() => return None,
        Some(at) => d.len() - at,
        None => 0,
    };
    if digits > 15 {
        return None;
    }
    let v = m as f64 / POW10[k];
    Some(if neg { -v } else { v })
}

fn parse_csv_field(field: &str) -> Value {
    if let Ok(num) = field.parse::<f64>() {
        Value::Number(num)
    } else if field == "true" || field == "True" {
        Value::Bool(true)
    } else if field == "false" || field == "False" {
        Value::Bool(false)
    } else if field.is_empty() {
        Value::Null
    } else if let Some(d) = try_parse_date(field) {
        Value::Date(d)
    } else {
        Value::String(field.to_string())
    }
}

pub fn read_xlsx_bytes(
    content: &[u8],
    header_row: usize,
    sheet_name: Option<&str>,
) -> Result<Table, Box<dyn std::error::Error>> {
    read_xlsx_bytes_inner(content, header_row, sheet_name, false)
}

/// Read XLSX with all rows as data (no header row extraction). Used before transpose.
pub fn read_xlsx_bytes_raw(
    content: &[u8],
    sheet_name: Option<&str>,
) -> Result<Table, Box<dyn std::error::Error>> {
    read_xlsx_bytes_inner(content, 0, sheet_name, true)
}

fn read_xlsx_bytes_inner(
    content: &[u8],
    header_row: usize,
    sheet_name: Option<&str>,
    raw: bool,
) -> Result<Table, Box<dyn std::error::Error>> {
    use calamine::{open_workbook_from_rs, Reader, Xlsx};
    use std::io::Cursor;
    let cursor = Cursor::new(content);
    let mut workbook: Xlsx<_> = open_workbook_from_rs(cursor)?;
    let sheet = if let Some(name) = sheet_name {
        workbook.worksheet_range(name)?
    } else {
        let sheet_names = workbook.sheet_names();
        if sheet_names.is_empty() {
            return Err("No sheets found".into());
        }
        workbook.worksheet_range(&sheet_names[0])?
    };
    if raw {
        read_xlsx_sheet_raw(sheet)
    } else {
        read_xlsx_sheet(sheet, header_row)
    }
}

fn parse_xlsx_row(row: &[calamine::Data]) -> Vec<Value> {
    row.iter()
        .map(|cell| match cell {
            calamine::Data::Int(n) => Value::Number(*n as f64),
            calamine::Data::Float(n) => Value::Number(*n),
            calamine::Data::String(s) => Value::String(s.clone()),
            calamine::Data::Bool(b) => Value::Bool(*b),
            calamine::Data::DateTime(dt) => try_parse_date(&dt.to_string())
                .map(Value::Date)
                .unwrap_or_else(|| Value::String(dt.to_string())),
            calamine::Data::DateTimeIso(s) => try_parse_date(s)
                .map(Value::Date)
                .unwrap_or_else(|| Value::String(s.clone())),
            calamine::Data::DurationIso(s) => try_parse_date(s)
                .map(Value::Date)
                .unwrap_or_else(|| Value::String(s.clone())),
            calamine::Data::Error(_) => Value::Null,
            calamine::Data::Empty => Value::Null,
        })
        .collect()
}

fn read_xlsx_sheet_raw(
    sheet: calamine::Range<calamine::Data>,
) -> Result<Table, Box<dyn std::error::Error>> {
    let mut rows = Vec::new();
    let mut max_cols = 0usize;
    for row in sheet.rows() {
        let values = parse_xlsx_row(row);
        max_cols = max_cols.max(values.len());
        rows.push(values);
    }
    pad_rows(&mut rows, max_cols);
    let headers: Vec<String> = (0..max_cols).map(|i| format!("Column_{}", i)).collect();
    Ok(Table::from_data(rows, Some(headers)))
}

fn read_xlsx_sheet(
    sheet: calamine::Range<calamine::Data>,
    header_row: usize,
) -> Result<Table, Box<dyn std::error::Error>> {
    let mut rows = Vec::new();
    let mut headers = Vec::new();
    for (row_idx, row) in sheet.rows().enumerate() {
        let values = parse_xlsx_row(row);
        if row_idx == header_row {
            headers = values
                .iter()
                .map(|v| match v {
                    Value::String(s) => s.clone(),
                    _ => v.to_string(),
                })
                .collect();
        } else {
            rows.push(values);
        }
    }
    if headers.is_empty() && !rows.is_empty() {
        let num_cols = rows[0].len();
        headers = (0..num_cols).map(|i| format!("Column_{}", i)).collect();
    }
    Ok(Table::from_data(rows, Some(headers)))
}

/// Transpose table: rows become columns and columns become rows.
pub fn transpose_table(table: Table) -> Table {
    let rows = match table.rows_ref() {
        Some(rr) => rr.to_vec(),
        None => return table,
    };
    let num_cols = table.headers().len();
    let num_rows = rows.len();
    if num_cols == 0 || num_rows == 0 {
        return table;
    }

    let mut new_rows = Vec::with_capacity(num_cols);
    for col_idx in 0..num_cols {
        let mut new_row = Vec::with_capacity(num_rows);
        for row in &rows {
            new_row.push(row.get(col_idx).cloned().unwrap_or(Value::Null));
        }
        new_rows.push(new_row);
    }
    let new_headers: Vec<String> = (0..num_rows).map(|i| format!("Column_{}", i)).collect();
    Table::from_data(new_rows, Some(new_headers))
}

/// Pick `header_row` from table rows as column headers; remaining rows become data.
pub fn apply_header_row(table: Table, header_row: usize) -> Table {
    let rows = match table.rows_ref() {
        Some(rr) => rr.to_vec(),
        None => return table,
    };
    if rows.is_empty() {
        return table;
    }
    let header_row = header_row.min(rows.len().saturating_sub(1));
    let headers: Vec<String> = rows[header_row]
        .iter()
        .map(|v| match v {
            Value::String(s) => s.clone(),
            _ => v.to_string(),
        })
        .collect();
    let mut data_rows = Vec::new();
    for (idx, row) in rows.iter().enumerate() {
        if idx != header_row {
            data_rows.push(row.clone());
        }
    }
    if headers.is_empty() && !data_rows.is_empty() {
        let num_cols = data_rows[0].len();
        let generated: Vec<String> = (0..num_cols).map(|i| format!("Column_{}", i)).collect();
        return Table::from_data(data_rows, Some(generated));
    }
    Table::from_data(data_rows, Some(headers))
}

/// Raw read → transpose → header_row on transposed table → headerT filter.
pub fn finish_transposed_read(table: Table, header_row: usize, header_t: Option<&Value>) -> Table {
    let transposed = transpose_table(table);
    let with_headers = apply_header_row(transposed, header_row);
    apply_header_filter(with_headers, header_t)
}

/// Apply column filter/rename from `read(..., header=...)`.
pub fn apply_header_filter(table: Table, header_arg: Option<&Value>) -> Table {
    let header_arg = match header_arg {
        Some(v) => v,
        None => return table,
    };
    match header_arg {
        Value::Array(cols_arr) => {
            let cols_arr_ref = cols_arr.borrow();
            let mut selected_cols = Vec::new();
            for col_val in cols_arr_ref.iter() {
                if let Value::String(s) = col_val {
                    selected_cols.push(s.clone());
                }
            }
            if selected_cols.is_empty() {
                return table;
            }
            let mut col_indices = Vec::new();
            let mut new_headers = Vec::new();
            for col_name in &selected_cols {
                if let Some(idx) = table.headers().iter().position(|h| h == col_name) {
                    col_indices.push(idx);
                    new_headers.push(col_name.clone());
                }
            }
            if col_indices.is_empty() {
                return table;
            }
            let mut new_rows = Vec::new();
            let rr = table.rows_ref().unwrap();
            for row in rr.iter() {
                let mut new_row = Vec::new();
                for &idx in &col_indices {
                    new_row.push(row.get(idx).cloned().unwrap_or(Value::Null));
                }
                new_rows.push(new_row);
            }
            Table::from_data(new_rows, Some(new_headers))
        }
        Value::Object(rename_map_rc) => {
            let rename_map = rename_map_rc.borrow();
            let mut new_headers = Vec::new();
            for old_header in table.headers() {
                if let Some(new_name_val) = rename_map.str_key_get(old_header) {
                    match new_name_val {
                        Value::String(new_name) => new_headers.push(new_name.clone()),
                        Value::Null => new_headers.push(old_header.clone()),
                        _ => new_headers.push(old_header.clone()),
                    }
                } else {
                    new_headers.push(old_header.clone());
                }
            }
            Table::from_data(table.rows_ref().unwrap().to_vec(), Some(new_headers))
        }
        _ => table,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Прежняя сборка: Vec<Vec<Value>> + дополнение + Table::from_data.
    fn old_way(content: &[u8], has_headers: bool) -> Table {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(has_headers)
            .flexible(true)
            .from_reader(content);
        let headers: Option<Vec<String>> =
            has_headers.then(|| reader.headers().unwrap().iter().map(|s| s.to_string()).collect());
        let mut max_cols = headers.as_ref().map_or(0, |h| h.len());
        let mut rows: Vec<Vec<Value>> = Vec::new();
        for r in reader.records() {
            let row: Vec<Value> = r.unwrap().iter().map(parse_csv_field).collect();
            max_cols = max_cols.max(row.len());
            rows.push(row);
        }
        pad_rows(&mut rows, max_cols);
        let headers =
            headers.unwrap_or_else(|| (0..max_cols).map(|i| format!("Column_{}", i)).collect());
        Table::from_data(rows, Some(headers))
    }

    fn new_way(content: &[u8], has_headers: bool) -> Table {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(has_headers)
            .flexible(true)
            .from_reader(content);
        build_csv_table(&mut reader, has_headers, false, 1024).unwrap()
    }

    #[test]
    fn flat_build_matches_row_build() {
        let cases: [&[u8]; 5] = [
            b"a,b,c\n1,x,\n2.5,true,2024-01-02\n",
            b"a,b\n",
            b"a\n1\n2,3\n4,5,6\n",
            b"a,b,c\n1\n2,3\n",
            b"",
        ];
        for c in cases {
            for h in [true, false] {
                let (o, n) = (old_way(c, h), new_way(c, h));
                assert_eq!(o, n, "{:?} headers={}", String::from_utf8_lossy(c), h);
                assert_eq!(o.headers(), n.headers());
            }
        }
    }

    fn via_csv(content: &[u8], has_headers: bool, trim: bool) -> Result<Table, String> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(has_headers)
            .trim(if trim { csv::Trim::Headers } else { csv::Trim::None })
            .from_reader(content);
        build_csv_table(&mut reader, has_headers, trim, 16).map_err(|e| e.to_string())
    }

    #[test]
    fn unquoted_fast_path_matches_csv() {
        let taken: [&[u8]; 14] = [
            b"a,b,c\n1,x,\n2.5,true,2024-01-02\n",
            b"a,b\r\n1,2\r\n3,4\r\n",
            b"a,b\r1,2\r3,4",
            b"a,b\n\n1,2\n\r\n\n3,4\n\n",
            b"a,b,\n1,2,\n3,4,5\n",
            b"a,b\n1,2",
            b" a , b \n 1 , x \n2,\xC2\xA0y\xC2\xA0\n",
            b"\xD0\xB8\xD0\xBC\xD1\x8F,n\n\xD0\xB0,1\n\xD0\xB1,2\n",
            b"c\nx\ny\nx\n\nz\n",
            b"n\n1\n2\n9223372036854775808\n",
            b"x,y\n,\n,\n",
            b"a",
            b"a,b\n1,2\n",
            b"a,b\n",
        ];
        for c in taken {
            for h in [true, false] {
                for t in [true, false] {
                    let fast = build_csv_unquoted(c, h, t, 16)
                        .unwrap_or_else(|| panic!("fallback on {:?}", String::from_utf8_lossy(c)));
                    let slow = via_csv(c, h, t).unwrap();
                    let ctx = format!("{:?} headers={} trim={}", String::from_utf8_lossy(c), h, t);
                    assert_eq!(fast, slow, "{ctx}");
                    assert_eq!(fast.headers(), slow.headers(), "{ctx}");
                }
            }
        }
        // Quotes, BOM, ragged rows and bad UTF-8 go to csv.
        let fallback: [&[u8]; 5] = [
            b"a,b\n\"1,2\",3\n",
            b"\xEF\xBB\xBFa,b\n1,2\n",
            b"a,b\n1,2,3\n",
            b"a,b\n1\n",
            b"a,b\n\xFF,1\n",
        ];
        for c in fallback {
            assert!(build_csv_unquoted(c, true, true, 16).is_none(), "{:?}", String::from_utf8_lossy(c));
        }
        // Errors and results through the public entry stay those of csv.
        for c in fallback {
            assert_eq!(
                read_csv_bytes(c, true).map_err(|e| e.to_string()),
                via_csv(c, true, true),
                "{:?}",
                String::from_utf8_lossy(c)
            );
        }
    }

    #[test]
    fn parallel_fast_path_matches_sequential() {
        fn gen(rows: usize, f: impl Fn(usize) -> String, crlf: bool) -> Vec<u8> {
            let nl = if crlf { "\r\n" } else { "\n" };
            let mut s = String::from("a,b,c");
            s.push_str(nl);
            for i in 0..rows {
                s.push_str(&f(i));
                s.push_str(nl);
            }
            s.into_bytes()
        }
        // Clean inputs: the parallel path must take them and match.
        let clean: Vec<Vec<u8>> = vec![
            gen(3000, |i| format!("{},k{},{}", i, i % 7, i as f64 * 0.5), false),
            gen(3000, |i| format!("{},k{},{}", i, i % 7, i), true),
            // Leading nulls that span several blocks, then numbers / strings.
            gen(3000, |i| if i < 1500 { ",,".into() } else { format!("{},s{},", i, i % 3) }, false),
            // Nulls scattered inside a dict column and a numeric column.
            gen(3000, |i| if i % 5 == 0 { ",,x".into() } else { format!("{},k{},x", i, i % 4) }, false),
            // Unicode and trailing spaces.
            gen(3000, |i| format!(" {} ,имя{} , ü{} ", i, i % 9, i % 2), false),
        ];
        // Inputs the parallel path may reject; when it does not, it must match.
        let mut any: Vec<Vec<u8>> = vec![
            gen(3000, |i| if i == 2500 { "x,k,1".into() } else { format!("{},k,1", i) }, false),
            gen(3000, |i| if i == 2500 { "1,true,1".into() } else { format!("{},k,1", i) }, false),
            gen(3000, |i| if i == 2500 { "1,2024-01-02,1".into() } else { format!("{},k,1", i) }, false),
            gen(3000, |i| if i == 2500 { "1,2".into() } else { format!("{},k,1", i) }, false),
            gen(3000, |i| if i == 2500 { "1,\"q\",1".into() } else { format!("{},k,1", i) }, false),
            // Sparse rule: mostly unique strings with numbers mixed in.
            gen(3000, |i| format!("{},{},1", i, if i % 3 == 0 { String::new() } else { format!("u{i}") }), false),
        ];
        // Dict probe: more than DICT_PROBE unique ids.
        any.push(gen(70_000, |i| format!("{},id{},1", i, i), false));
        let mut no_nl = gen(3000, |i| format!("{},k{},{}", i, i % 7, i), false);
        no_nl.pop();
        any.push(no_nl);
        let mut blank = gen(3000, |i| format!("{},k,{}", i, i), false);
        blank.extend_from_slice(b"\n\r\n\n");
        any.push(blank);

        let check = |c: &[u8], must: bool| {
            for h in [true, false] {
                for t in [true, false] {
                    let seq = build_csv_unquoted(c, h, t, 16).or_else(|| via_csv(c, h, t).ok());
                    for threads in [2, 3, 8] {
                        let par = build_csv_unquoted_parallel(c, h, t, threads, 1024);
                        let ctx = format!("len={} headers={} trim={} threads={}", c.len(), h, t, threads);
                        match par {
                            Some(p) => {
                                let s = seq.as_ref().expect(&ctx);
                                assert_eq!(&p, s, "{ctx}");
                                assert_eq!(p.headers(), s.headers(), "{ctx}");
                            }
                            None => assert!(!(must && h), "parallel fallback on clean input {ctx}"),
                        }
                    }
                }
            }
        };
        for c in &clean {
            check(c, true);
        }
        for c in &any {
            check(c, false);
        }
    }

    #[test]
    fn fast_f64_matches_std_parse() {
        let fixed = [
            "0", "-0", "1", "007", "1.", ".5", "-.5", "+5", "-", ".", "1.2.3", "1e5", "inf", "",
            "123456789012345", "1234567890123456", "9007199254740993", "0.000000000000001",
            "99999999999999.9", "-0.0", "12a", "1_0", " 1",
        ];
        let mut cases: Vec<String> = fixed.iter().map(|s| s.to_string()).collect();
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        for _ in 0..200_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let digits = (x % 17) as usize + 1;
            let s: String = (0..digits).map(|i| (b'0' + ((x >> (i * 3)) % 10) as u8) as char).collect();
            let dot = (x >> 52) as usize % (digits + 2);
            let mut s = if dot > 0 && dot < digits { format!("{}.{}", &s[..dot], &s[dot..]) } else { s };
            if x >> 63 == 1 {
                s.insert(0, '-');
            }
            cases.push(s);
        }
        for s in &cases {
            if let Some(v) = fast_f64(s) {
                let want: f64 = s.parse().unwrap();
                assert_eq!(v.to_bits(), want.to_bits(), "{:?}", s);
            }
        }
        assert!(fast_f64("+5").is_none() && fast_f64("1.").is_none() && fast_f64(".5").is_none());
    }

    // Блок строкового столбца, как его собирает разбор: локальные коды по первому появлению.
    fn str_part<'a>(rows: &[Option<&'a str>]) -> Part<'a> {
        let (mut codes, mut keys, mut first) = (Vec::new(), Vec::new(), Vec::new());
        for (r, v) in rows.iter().enumerate() {
            let c = keys.iter().position(|k| k == v).unwrap_or_else(|| {
                keys.push(*v);
                first.push(r);
                keys.len() - 1
            });
            codes.push(c as u32);
        }
        let null = keys.iter().position(|k| k.is_none()).map(|c| c as u32);
        Part::Str { codes, keys, first, map: BlockMap::default(), null }
    }

    fn plans_agree(blocks: &[Vec<Option<&str>>], nulls: &[(usize, usize)]) {
        let mut parts: Vec<Part> = blocks.iter().map(|b| str_part(b)).collect();
        let mut counts: Vec<usize> = blocks.iter().map(|b| b.len()).collect();
        for &(at, n) in nulls {
            parts.insert(at, Part::Null(n));
            counts.insert(at, n);
        }
        let seq = plan_dict_seq(&parts, &counts);
        for w in [2, 3, 4, 7] {
            assert_eq!(plan_dict_par(&parts, &counts, w), seq, "workers {w}");
        }
    }

    #[test]
    fn dict_par_matches_seq_on_mismatched_blocks() {
        let s = |v: &[&'static str]| -> Vec<Option<&'static str>> {
            v.iter().map(|x| if x.is_empty() { None } else { Some(*x) }).collect()
        };
        // Тот же набор, другой порядок; частичное пересечение; повторы; null.
        plans_agree(&[s(&["a", "b", "c"]), s(&["c", "b", "a"]), s(&["a", "a", "a"])], &[]);
        plans_agree(&[s(&["a", "b", "", "a"]), s(&["d", "", "b", "e"]), s(&["e", "f", "a", "f"])], &[]);
        plans_agree(&[s(&["x", "y"]), s(&["y", "z"])], &[(0, 3), (2, 0), (3, 5)]);
        plans_agree(&[s(&["", ""]), s(&["q", ""])], &[(1, 2)]);
        // Много различных значений вперемешку с повторами в разном порядке по блокам.
        let vals: Vec<String> = (0..3000).map(|i| format!("v{}", (i * 7919) % 1200)).collect();
        let blocks: Vec<Vec<Option<&str>>> = vals
            .chunks(400)
            .enumerate()
            .map(|(b, c)| c.iter().rev().skip(b % 3).map(|v| Some(v.as_str())).collect())
            .collect();
        plans_agree(&blocks, &[(2, 10)]);
    }

    #[test]
    fn hkey_hash_matches_field_hasher() {
        let text: String = (0..40u8).map(|i| (b'a' + i % 26) as char).collect();
        for n in 0..=40 {
            let s = &text[..n];
            let mut st = FieldHasher::default();
            std::hash::Hasher::write(&mut st, s.as_bytes());
            assert_eq!(HKey::new(s).h, std::hash::Hasher::finish(&st), "len {n}");
        }
    }

    // Все ключи с одним и тем же хешем: карта блока различает их только по тексту.
    #[test]
    fn block_map_survives_forced_hash_collisions() {
        let base = "abcdefghijklmnop"; // 16 байт: граница встроенного префикса
        let vals: Vec<String> = vec![
            "".into(), "a".into(), "a\0".into(), "abcdefgh".into(), base.into(),
            format!("{base}q"), format!("{base}r"), format!("{base}qq"), "abcdefghijklmnoq".into(),
            "b".into(), "a".into(), base.into(), format!("{base}r"), "".into(), format!("{base}qq"),
        ];
        let mut map = BlockMap::default();
        let mut want: Vec<&str> = Vec::new();
        for v in &vals {
            let n = map.len() as u32;
            let id = *map.entry(HKey::with_hash(7, v)).or_insert(n);
            let pos = want.iter().position(|w| w == v).unwrap_or_else(|| {
                want.push(v);
                want.len() - 1
            });
            assert_eq!(id as usize, pos, "{v:?}");
        }
        assert_eq!(map.len(), want.len());
        // Те же значения через параллельный план: границы префикса, общий префикс, повторы.
        let rows: Vec<Option<&str>> = vals.iter().map(|v| Some(v.as_str())).collect();
        let blocks: Vec<Vec<Option<&str>>> =
            vec![rows.clone(), rows.iter().rev().copied().collect(), rows[5..].to_vec()];
        plans_agree(&blocks, &[]);
    }

    // NULL получает код по месту первого появления: до, между и после новых ключей,
    // в блоке без новых ключей и в Part::Null раньше любого строкового NULL.
    #[test]
    fn dict_par_places_null_by_first_appearance() {
        let s = |v: &[&'static str]| -> Vec<Option<&'static str>> {
            v.iter().map(|x| if x.is_empty() { None } else { Some(*x) }).collect()
        };
        plans_agree(&[s(&["", "a", "b"]), s(&["b", "c"])], &[]);
        plans_agree(&[s(&["a", "", "b"]), s(&["c", ""])], &[]);
        plans_agree(&[s(&["a", "b", ""]), s(&["", "c"])], &[]);
        plans_agree(&[s(&["a", "b"]), s(&["b", "", "a"]), s(&["c"])], &[]);
        plans_agree(&[s(&["a"]), s(&["a", ""])], &[(1, 4)]);
        plans_agree(&[s(&["", "", ""]), s(&["", ""])], &[]);
        let parts: Vec<Part> = [s(&["a", "b"]), s(&["b", "", "c"])].iter().map(|b| str_part(b)).collect();
        let (dict, null, _) = plan_dict_par(&parts, &[2, 3], 3).unwrap();
        assert_eq!(dict, vec![Some("a"), Some("b"), None, Some("c")]);
        assert_eq!(null, Some(2));
    }

    // Пустая строка, 16/17 байт и длинные значения в разных блоках и порядках.
    #[test]
    fn dict_par_matches_seq_on_prefix_lengths() {
        let long = "x".repeat(300);
        let base = "abcdefghijklmnop";
        let v17 = format!("{base}z");
        let vals = ["", base, &v17, &long, "abcdefghijklmnoq", &long[..17], &long[..16]];
        let fwd: Vec<Option<&str>> = vals.iter().map(|v| Some(*v)).collect();
        let rev: Vec<Option<&str>> = fwd.iter().rev().copied().collect();
        let mix: Vec<Option<&str>> = vec![fwd[3], None, fwd[0], fwd[3], fwd[6]];
        plans_agree(&[fwd.clone(), rev.clone(), mix.clone()], &[]);
        plans_agree(&[mix, rev, fwd], &[(1, 2)]);
    }

    // Проверка «похоже на id» срабатывает в той же строке, что и в последовательном плане.
    #[test]
    fn dict_par_probe_trips_like_seq_near_threshold() {
        use crate::common::typed_column::DICT_PROBE;
        let n = DICT_PROBE + DICT_PROBE / 8;
        for every in [12, 19, 20, 21, 40] {
            let vals: Vec<String> =
                (0..n).map(|i| if i % every == 0 { "dup".to_string() } else { format!("k{i}") }).collect();
            let blocks: Vec<Vec<Option<&str>>> = vals
                .chunks(DICT_PROBE / 3 + 7)
                .map(|c| c.iter().map(|v| Some(v.as_str())).collect())
                .collect();
            let parts: Vec<Part> = blocks.iter().map(|b| str_part(b)).collect();
            let counts: Vec<usize> = blocks.iter().map(|b| b.len()).collect();
            let seq = plan_dict_seq(&parts, &counts);
            for w in [2, 4] {
                assert_eq!(plan_dict_par(&parts, &counts, w), seq, "every {every} workers {w}");
            }
        }
    }

    #[test]
    fn dict_par_falls_back_like_seq_when_id_like() {
        use crate::common::typed_column::DICT_PROBE;
        let vals: Vec<String> = (0..DICT_PROBE * 3).map(|i| format!("id{i}")).collect();
        let blocks: Vec<Vec<Option<&str>>> =
            vals.chunks(DICT_PROBE / 2 + 1).map(|c| c.iter().map(|v| Some(v.as_str())).collect()).collect();
        let parts: Vec<Part> = blocks.iter().map(|b| str_part(b)).collect();
        let counts: Vec<usize> = blocks.iter().map(|b| b.len()).collect();
        assert_eq!(plan_dict_seq(&parts, &counts), None);
        assert_eq!(plan_dict_par(&parts, &counts, 4), None);
    }
}

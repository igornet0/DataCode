//! CIFAR bin diagnostics: read → chunk/slice → lists (see tests/diagnostics/cifar_bin_diagnostics.dc).
//! Interpretation and architecture decisions: `tests/diagnostics/CIFAR_CONCLUSIONS_AND_ARCHITECTURE.md`.
//!
//! **Where copies happen (for interpreting results):**
//! - `array.chunk(n)` on `read` output (`ByteBuffer`): each window is a zero-copy byte slice (`materialize_chunk_at` in `src/vm/iterable.rs`); chunk over a heap `Array` still clones elements.
//! - `arr[a:b]` (slice): `src/vm/interpreter/element/array_ops.rs` (`get_array_slice_from_slots`) for step 1 yields an [`ArrayView`] (zero-copy over the backing).
//! - `dataset.concat` / `dataset.push_data` (ML plugin): both merge via `Tensor::concat_axis0` in
//!   `datacode_lib/ML-Datacode-lib/src/nn/dataset.rs` and `src/core/tensor.rs` (new contiguous buffer).

use data_code::{run, run_with_base_path, Value};
use std::path::PathBuf;
use std::time::Instant;

fn assert_number_near(v: Value, expected: f64, msg: &str) {
    let n = v
        .as_ieee_f64()
        .unwrap_or_else(|| panic!("{msg}: expected numeric, got {v:?}"));
    assert!(
        (n - expected).abs() < 1e-9,
        "{msg}: expected {expected}, got {n}"
    );
}

fn assert_number_in_range(v: Value, range: std::ops::RangeInclusive<f64>, msg: &str) {
    let n = v
        .as_ieee_f64()
        .unwrap_or_else(|| panic!("{msg}: expected numeric, got {v:?}"));
    assert!(range.contains(&n), "{msg}: {n} not in {range:?}");
}

fn cifar_batch_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/test_data/image-dataset/cifar-10-batches-bin/data_batch_1.bin")
}

fn diagnostics_source(bin_path: &str) -> String {
    let tpl = include_str!("diagnostics/cifar_bin_diagnostics.dc");
    tpl.replace("@@CIFAR_BIN@@", bin_path)
}

fn diagnostics_smoke_source(bin_path: &str) -> String {
    let tpl = include_str!("diagnostics/cifar_bin_diagnostics_smoke.dc");
    tpl.replace("@@CIFAR_BIN@@", bin_path)
}

#[test]
fn system_time_monotonic_ms_is_numeric_delta() {
    let source = r#"
import system
let a = system.time.monotonic_ms()
let b = system.time.monotonic_ms()
b - a
"#;
    let v = run(source).expect("run");
    assert_number_in_range(v, 0.0..=1_000_000.0, "monotonic_ms delta");
}

/// Fast sanity check: tiny bin file, full diagnostic script.
#[test]
fn cifar_diagnostics_script_runs_on_sample_bin() {
    let sample =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test_data/read_file_bin_sample.bin");
    let p = sample.to_string_lossy();
    let src = diagnostics_smoke_source(&p);
    run_with_base_path(&src, PathBuf::from(env!("CARGO_MANIFEST_DIR")).as_path())
        .expect("diagnostics on sample bin");
}

/// After `read`, `chunk(n)[0]` must be a scalar Number (CIFAR label byte), not a pixel slice.
#[test]
fn cifar_chunk_index0_is_number_on_sample_bin() {
    let sample =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test_data/read_file_bin_sample.bin");
    let p = sample.to_string_lossy();
    let src = format!(
        r#"let data = read("{}")
let ch0 = data.chunk(3073)[0]
ch0[0]"#,
        p
    );
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let v = run_with_base_path(&src, base.as_path()).expect("run");
    assert_number_in_range(v, 0.0..=255.0, "chunk[0] label byte");
}

/// `read` → slice (`ArrayView`) → `chunk(n)`: each row must be an owned array; first byte + tail length.
/// Sample file is 3 bytes `01 02 ff`: one full chunk of 3, `ch[0]+len(ch[1:])` = 1 + 2 = 3.
#[test]
fn cifar_read_slice_then_chunk_label_and_tail_len() {
    let sample =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test_data/read_file_bin_sample.bin");
    let p = sample.to_string_lossy();
    let src = format!(
        r#"let data = read("{}")
let v = data[0:3]
let ch = v.chunk(3)[0]
ch[0] + len(ch[1:])"#,
        p
    );
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let v = run_with_base_path(&src, base.as_path()).expect("run");
    assert_number_near(v, 3.0, "ch[0] + len(ch[1:])");
}

/// Repro for ML example bug: `labels` must stay scalars after `push`; `chunk[1:]` must have length 3072.
/// Uses first 3 CIFAR records only (prefix slice) for speed.
#[test]
fn cifar_labels_pixels_lists_after_chunk_and_slice_match_expectations() {
    let path = cifar_batch_path();
    if !path.is_file() {
        eprintln!("skip: missing {}", path.display());
        return;
    }
    let p = path.to_string_lossy();
    // 3 * 3073 bytes: three full rows; avoids scanning 30M numbers in CI.
    let src = format!(
        r#"
let data = read("{}")
let prefix = data[0:9219]
let chunks = prefix.chunk(3073)
let labels = []
let pixels_list = []
for chunk in chunks {{
  if len(chunk) != 3073 {{
    continue
  }}
  let label = chunk[0]
  let pixels = chunk[1:]
  labels.push(label)
  pixels_list.push(pixels)
}}
let score = 0
if typeof(labels[0]) == "int" {{
  score = score + 1
}}
if typeof(labels[1]) == "int" {{
  score = score + 1
}}
if len(pixels_list[0]) == 3072 {{
  score = score + 1
}}
score
"#,
        p
    );
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let v = run_with_base_path(&src, base.as_path()).expect("run");
    assert_number_near(v, 3.0, "labels/pixels score");
}

/// Same as `cifar_labels_pixels_lists_after_chunk_and_slice_match_expectations` but inside a function
/// (mirrors `export_data` in ML example) to catch scope / closure issues.
#[test]
fn cifar_labels_pixels_same_checks_inside_export_data_fn() {
    let path = cifar_batch_path();
    if !path.is_file() {
        eprintln!("skip: missing {}", path.display());
        return;
    }
    let p = path.to_string_lossy();
    let src = format!(
        r#"
fn export_data(data) {{
  let prefix = data[0:9219]
  let chunks = prefix.chunk(3073)
  let labels = []
  let pixels_list = []
  for chunk in chunks {{
    if len(chunk) != 3073 {{
      continue
    }}
    let label = chunk[0]
    let pixels = chunk[1:]
    labels.push(label)
    pixels_list.push(pixels)
  }}
  let score = 0
  if typeof(labels[0]) == "int" {{
    score = score + 1
  }}
  if typeof(labels[1]) == "int" {{
    score = score + 1
  }}
  if len(pixels_list[0]) == 3072 {{
    score = score + 1
  }}
  return score
}}
let data = read("{}")
export_data(data)
"#,
        p
    );
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let v = run_with_base_path(&src, base.as_path()).expect("run");
    assert_number_near(v, 3.0, "export_data score");
}

/// Full CIFAR batch (~30M bytes as VM numbers): long-running; run with
/// `cargo test cifar_diagnostics_full_batch_repro -- --ignored --nocapture`.
#[test]
#[ignore]
fn cifar_diagnostics_full_batch_repro() {
    let path = cifar_batch_path();
    assert!(
        path.is_file(),
        "missing {}; add tests/test_data/image-dataset",
        path.display()
    );
    let p = path.to_string_lossy();
    let src = diagnostics_source(&p);
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let start = Instant::now();
    run_with_base_path(&src, base.as_path()).expect("full diagnostics");
    eprintln!("full cifar diagnostics wall: {:?}", start.elapsed());
}

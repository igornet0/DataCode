//! Micro-benchmark: integral-only vs sync insert (hypothesis A for A* regression).

use data_code::common::set_map::SetMap;
use data_code::common::value_store::ValueId;
use std::time::Instant;

const N: i64 = 200_000;

#[test]
fn integral_only_vs_sync_insert_counts() {
    let mut only = SetMap::new();
    let t0 = Instant::now();
    for i in 0..N {
        let key_id = (i + 1) as ValueId;
        only.insert_integral_only(i, key_id);
    }
    let only_ms = t0.elapsed().as_millis();

    let mut sync = SetMap::new();
    let t1 = Instant::now();
    for i in 0..N {
        let key_id = (i + 1) as ValueId;
        sync.insert_integral_sync(i, key_id);
    }
    let sync_ms = t1.elapsed().as_millis();

    eprintln!(
        "set insert {} keys: integral_only={}ms sync={}ms ratio={:.1}x",
        N,
        only_ms,
        sync_ms,
        sync_ms as f64 / only_ms.max(1) as f64
    );
    assert!(only.len() == N as usize);
    assert!(sync.len() == N as usize);
}

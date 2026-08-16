//! Sync bridge for async chromiumoxide / MongoDB / MSSQL calls.

use std::future::Future;
use std::sync::OnceLock;
use tokio::runtime::Runtime;

fn dedicated_runtime() -> &'static Runtime {
    static RT: OnceLock<Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("datacode-async-bridge")
            .build()
            .expect("failed to create datacode async bridge tokio runtime")
    })
}

/// Drive an async future from sync code.
///
/// Requires a multi-thread Tokio runtime when already inside async context
/// (`block_in_place`). WebSocket client tasks are therefore spawned on their own
/// multi-thread runtimes (not `LocalSet`).
pub fn block_on<F: Future>(fut: F) -> F::Output {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(|| handle.block_on(fut)),
        Err(_) => dedicated_runtime().block_on(fut),
    }
}

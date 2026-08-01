//! Sync bridge for async chromiumoxide calls.

use std::future::Future;
use std::sync::OnceLock;
use tokio::runtime::Runtime;

fn dedicated_runtime() -> &'static Runtime {
    static RT: OnceLock<Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("datacode-web-browser")
            .build()
            .expect("failed to create web browser tokio runtime")
    })
}

pub fn block_on<F: Future>(fut: F) -> F::Output {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(|| handle.block_on(fut)),
        Err(_) => dedicated_runtime().block_on(fut),
    }
}

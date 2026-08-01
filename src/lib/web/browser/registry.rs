//! Per-thread registry of open browser pages for process cleanup.

use crate::web::values::WebPage;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

thread_local! {
    static PAGES: RefCell<HashMap<u64, Rc<RefCell<WebPage>>>> = RefCell::new(HashMap::new());
}

pub fn register(page: Rc<RefCell<WebPage>>) {
    let id = page.borrow().id;
    PAGES.with(|m| {
        m.borrow_mut().insert(id, page);
    });
}

pub fn unregister(id: u64) {
    PAGES.with(|m| {
        m.borrow_mut().remove(&id);
    });
}

pub fn get(id: u64) -> Option<Rc<RefCell<WebPage>>> {
    PAGES.with(|m| m.borrow().get(&id).cloned())
}

/// Close all tracked browser pages (call on VM / WS client teardown).
pub fn cleanup_all() {
    let pages: Vec<Rc<RefCell<WebPage>>> =
        PAGES.with(|m| m.borrow().values().cloned().collect());
    for page in pages {
        let mut p = page.borrow_mut();
        if !p.closed {
            let _ = p.driver.lock().map(|mut d| d.close());
            p.closed = true;
        }
    }
    PAGES.with(|m| m.borrow_mut().clear());
}

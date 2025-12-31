use std::cell::RefCell;
use std::io::Write;

thread_local! {
    static OUTPUT_BUFFER: RefCell<Option<Vec<u8>>> = RefCell::new(None);
}

/// Структура для перехвата вывода print()
pub struct OutputCapture;

impl OutputCapture {
    pub fn new() -> Self {
        Self
    }

    /// Включить перехват вывода
    pub fn set_capture(&self, enabled: bool) {
        OUTPUT_BUFFER.with(|buf| {
            if enabled {
                *buf.borrow_mut() = Some(Vec::new());
            } else {
                *buf.borrow_mut() = None;
            }
        });
    }

    /// Получить перехваченный вывод
    pub fn get_output(&self) -> String {
        OUTPUT_BUFFER.with(|buf| {
            buf.borrow()
                .as_ref()
                .map(|v| String::from_utf8_lossy(v).to_string())
                .unwrap_or_default()
        })
    }

    /// Записать данные в буфер (используется модифицированной функцией print)
    pub fn write_output(data: &str) {
        OUTPUT_BUFFER.with(|buf| {
            if let Some(ref mut buffer) = *buf.borrow_mut() {
                buffer.write_all(data.as_bytes()).ok();
                buffer.write_all(b"\n").ok();
            } else {
                // Если перехват не активен, выводим в stdout
                println!("{}", data);
            }
        });
    }

    /// Проверить, активен ли перехват вывода
    pub fn is_capturing() -> bool {
        OUTPUT_BUFFER.with(|buf| {
            buf.borrow().is_some()
        })
    }
}


//! Stealth browser layer: wraps a standard [`BrowserDriver`] with anti-automation hardening.
//!
//! Architecture:
//! ```text
//! DataCode Browser API
//!        ↓
//!   BrowserDriver (trait)
//!        ↓
//!  StealthBrowserDriver  ← this module (swappable implementation)
//!        ↓
//!   ChromiumDriver / Custom
//! ```
//!
//! The public DataCode API (`page.click`, `page.goto`, …) is unchanged; stealth is selected
//! only at `browser.open(..., stealth=true)`.

use crate::common::value::Value;
use crate::web::args::BrowserOptions;
use crate::web::browser::driver::{BrowserDriver, ElementHandle};
use crate::web::error::WebError;
use std::path::PathBuf;
use std::time::Duration;

/// Default desktop Chrome user-agent used when stealth is on and caller did not set one.
pub const STEALTH_USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) \
AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";

/// Chromium launch flags that reduce common automation fingerprints.
pub fn stealth_launch_args() -> Vec<&'static str> {
    vec![
        "--disable-blink-features=AutomationControlled",
        "--disable-infobars",
        "--disable-dev-shm-usage",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "--disable-component-update",
        "--disable-default-apps",
        "--disable-popup-blocking",
        "--password-store=basic",
        "--use-mock-keychain",
        "--lang=en-US",
    ]
}

/// Init script injected before page JS runs (via CDP `Page.addScriptToEvaluateOnNewDocument`
/// when available, otherwise evaluated after navigation).
pub fn stealth_init_script(opts: &BrowserOptions) -> String {
    let lang = opts
        .locale
        .clone()
        .unwrap_or_else(|| "en-US".to_string())
        .replace('\'', "\\'");
    let languages = format!("['{}', 'en']", lang);
    let hw = opts.hardware_concurrency.unwrap_or(8);
    let mem = opts.device_memory_gb.unwrap_or(8);
    format!(
        r#"(function() {{
  try {{
    Object.defineProperty(navigator, 'webdriver', {{ get: () => undefined }});
  }} catch (e) {{}}
  try {{
    Object.defineProperty(navigator, 'languages', {{ get: () => {languages} }});
    Object.defineProperty(navigator, 'language', {{ get: () => '{lang}' }});
  }} catch (e) {{}}
  try {{
    Object.defineProperty(navigator, 'plugins', {{
      get: () => [1, 2, 3, 4, 5]
    }});
  }} catch (e) {{}}
  try {{
    Object.defineProperty(navigator, 'hardwareConcurrency', {{ get: () => {hw} }});
  }} catch (e) {{}}
  try {{
    Object.defineProperty(navigator, 'deviceMemory', {{ get: () => {mem} }});
  }} catch (e) {{}}
  try {{
    window.chrome = window.chrome || {{ runtime: {{}} }};
  }} catch (e) {{}}
  try {{
    const originalQuery = window.navigator.permissions && window.navigator.permissions.query;
    if (originalQuery) {{
      window.navigator.permissions.query = (parameters) => (
        parameters && parameters.name === 'notifications'
          ? Promise.resolve({{ state: Notification.permission }})
          : originalQuery(parameters)
      );
    }}
  }} catch (e) {{}}
}})();"#
    )
}

/// Extra Chromium args derived from stealth options (locale, window size, profile already separate).
pub fn stealth_extra_args(opts: &BrowserOptions) -> Vec<String> {
    let mut args = Vec::new();
    if let Some(locale) = &opts.locale {
        args.push(format!("--lang={}", locale));
    }
    if let Some((w, h)) = opts.viewport {
        args.push(format!("--window-size={},{}", w, h));
    } else {
        args.push("--window-size=1920,1080".to_string());
    }
    if let Some(tz) = &opts.timezone {
        // Chromium picks TZ from environment; pass via arg where supported.
        let _ = tz;
    }
    args
}

/// Ensure a profile directory exists for persistent cookies / storage.
pub fn ensure_profile_dir(profile: &str) -> Result<PathBuf, WebError> {
    let path = PathBuf::from(profile);
    if path.as_os_str().is_empty() {
        return Err(WebError::value("profile path must not be empty"));
    }
    std::fs::create_dir_all(&path)
        .map_err(|e| WebError::io(format!("failed to create browser profile '{}': {}", path.display(), e)))?;
    Ok(path)
}

/// Stealth wrapper around any [`BrowserDriver`].
///
/// Replace the inner driver or this type to swap stealth backends without changing DataCode API.
pub struct StealthBrowserDriver {
    inner: Box<dyn BrowserDriver>,
    /// Small randomized delays on navigation / interaction for more natural timing.
    humanize: bool,
}

impl StealthBrowserDriver {
    /// Wrap an already-launched driver and apply post-launch stealth patches.
    pub fn wrap(inner: Box<dyn BrowserDriver>, humanize: bool) -> Self {
        Self { inner, humanize }
    }

    /// Launch Chromium with stealth flags / profile, wrap as StealthBrowserDriver.
    pub fn launch(url: &str, opts: &BrowserOptions) -> Result<Self, WebError> {
        let mut stealth_opts = opts.clone();
        stealth_opts.stealth = true;
        if stealth_opts.user_agent.is_none() {
            stealth_opts.user_agent = Some(STEALTH_USER_AGENT.to_string());
        }
        if stealth_opts.locale.is_none() {
            stealth_opts.locale = Some("en-US".to_string());
        }
        if stealth_opts.viewport.is_none() {
            stealth_opts.viewport = Some((1920, 1080));
        }
        if let Some(tz) = &stealth_opts.timezone {
            std::env::set_var("TZ", tz);
        }
        let inner = crate::web::browser::chromium::ChromiumDriver::launch_with_stealth(url, &stealth_opts)?;
        let mut driver = Self::wrap(Box::new(inner), true);
        driver.apply_runtime_stealth(&stealth_opts)?;
        Ok(driver)
    }

    fn apply_runtime_stealth(&mut self, opts: &BrowserOptions) -> Result<(), WebError> {
        let script = stealth_init_script(opts);
        self.inner.evaluate_on_new_document(&script)?;
        // Also run once on the current document.
        let _ = self.inner.evaluate(&script);
        Ok(())
    }

    fn human_pause(&self) {
        if !self.humanize {
            return;
        }
        // 20–80 ms jitter
        let n = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0)
            % 60)
            + 20;
        std::thread::sleep(Duration::from_millis(n as u64));
    }
}

impl BrowserDriver for StealthBrowserDriver {
    fn goto(&mut self, url: &str) -> Result<(), WebError> {
        self.human_pause();
        self.inner.goto(url)?;
        self.human_pause();
        Ok(())
    }

    fn click(&mut self, selector: &str) -> Result<(), WebError> {
        self.human_pause();
        self.inner.click(selector)
    }

    fn type_text(&mut self, selector: &str, text: &str) -> Result<(), WebError> {
        self.human_pause();
        self.inner.type_text(selector, text)
    }

    fn fill(&mut self, selector: &str, text: &str) -> Result<(), WebError> {
        self.human_pause();
        self.inner.fill(selector, text)
    }

    fn clear(&mut self, selector: &str) -> Result<(), WebError> {
        self.inner.clear(selector)
    }

    fn select(&mut self, selector: &str, value: &str) -> Result<(), WebError> {
        self.human_pause();
        self.inner.select(selector, value)
    }

    fn text(&mut self, selector: Option<&str>) -> Result<String, WebError> {
        self.inner.text(selector)
    }

    fn html(&mut self, selector: Option<&str>) -> Result<String, WebError> {
        self.inner.html(selector)
    }

    fn screenshot(&mut self, path: &str, selector: Option<&str>) -> Result<(), WebError> {
        self.inner.screenshot(path, selector)
    }

    fn wait(&mut self, secs: f64) -> Result<(), WebError> {
        self.inner.wait(secs)
    }

    fn wait_for(&mut self, selector: &str, timeout: Option<f64>) -> Result<(), WebError> {
        self.inner.wait_for(selector, timeout)
    }

    fn wait_for_navigation(&mut self, timeout: Option<f64>) -> Result<(), WebError> {
        self.inner.wait_for_navigation(timeout)
    }

    fn find(&mut self, selector: &str) -> Result<ElementHandle, WebError> {
        self.inner.find(selector)
    }

    fn find_all(&mut self, selector: &str) -> Result<Vec<ElementHandle>, WebError> {
        self.inner.find_all(selector)
    }

    fn cookies(&mut self) -> Result<Value, WebError> {
        self.inner.cookies()
    }

    fn set_cookie(&mut self, name: &str, value: &str) -> Result<(), WebError> {
        self.inner.set_cookie(name, value)
    }

    fn delete_cookie(&mut self, name: &str) -> Result<(), WebError> {
        self.inner.delete_cookie(name)
    }

    fn attr(&mut self, selector: &str, name: &str) -> Result<String, WebError> {
        self.inner.attr(selector, name)
    }

    fn attributes(&mut self, selector: &str) -> Result<Value, WebError> {
        self.inner.attributes(selector)
    }

    fn close(&mut self) -> Result<(), WebError> {
        self.inner.close()
    }

    fn evaluate(&mut self, script: &str) -> Result<Value, WebError> {
        self.inner.evaluate(script)
    }

    fn evaluate_on_new_document(&mut self, script: &str) -> Result<(), WebError> {
        self.inner.evaluate_on_new_document(script)
    }
}

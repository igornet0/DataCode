//! chromiumoxide (CDP) backend.

use crate::common::value::Value;
use crate::web::args::BrowserOptions;
use crate::web::browser::driver::{BrowserDriver, ElementHandle};
use crate::web::browser::runtime::block_on;
use crate::web::error::WebError;
use crate::web::security::{validate_url, DEFAULT_WAIT_TIMEOUT_SECS};
use chromiumoxide::browser::{Browser, BrowserConfig};
use chromiumoxide::cdp::browser_protocol::network::CookieParam;
use chromiumoxide::cdp::browser_protocol::page::CaptureScreenshotFormat;
use chromiumoxide::page::{Page, ScreenshotParams};
use futures::StreamExt;
use std::collections::HashMap;
use std::time::Duration;

pub struct ChromiumDriver {
    browser: Option<Browser>,
    page: Page,
    closed: bool,
}

impl ChromiumDriver {
    pub fn launch(url: &str, opts: &BrowserOptions) -> Result<Self, WebError> {
        Self::launch_inner(url, opts, false)
    }

    /// Launch with stealth Chromium flags / user-data profile (used by [`super::stealth`]).
    pub fn launch_with_stealth(url: &str, opts: &BrowserOptions) -> Result<Self, WebError> {
        Self::launch_inner(url, opts, true)
    }

    fn launch_inner(url: &str, opts: &BrowserOptions, stealth: bool) -> Result<Self, WebError> {
        validate_url(url)?;
        let headless = opts.headless;
        let user_agent = opts.user_agent.clone();
        let profile = opts.profile.clone();
        let extra_stealth_args: Vec<String> = if stealth {
            let mut a: Vec<String> = crate::web::browser::stealth::stealth_launch_args()
                .into_iter()
                .map(|s| s.to_string())
                .collect();
            a.extend(crate::web::browser::stealth::stealth_extra_args(opts));
            a
        } else {
            Vec::new()
        };
        let init_script = if stealth {
            Some(crate::web::browser::stealth::stealth_init_script(opts))
        } else {
            None
        };
        let url = url.to_string();
        block_on(async move {
            let mut builder = BrowserConfig::builder();
            if headless {
                builder = builder.arg("--headless=new").arg("--disable-gpu");
            } else {
                builder = builder.with_head();
            }
            if let Some(ua) = user_agent {
                builder = builder.arg(format!("--user-agent={}", ua));
            }
            if let Some(profile_path) = profile {
                let dir = crate::web::browser::stealth::ensure_profile_dir(&profile_path)?;
                builder = builder.user_data_dir(dir);
            }
            for arg in &extra_stealth_args {
                builder = builder.arg(arg);
            }
            let config = builder
                .build()
                .map_err(|e| WebError::runtime(format!("browser config: {}", e)))?;

            let (browser, mut handler) = Browser::launch(config)
                .await
                .map_err(|e| WebError::runtime(format!("failed to launch browser: {}", e)))?;

            tokio::spawn(async move {
                while handler.next().await.is_some() {}
            });

            let page = if stealth {
                let page = browser
                    .new_page("about:blank")
                    .await
                    .map_err(|e| WebError::io(format!("failed to open page: {}", e)))?;
                if let Some(script) = &init_script {
                    let _ = page.evaluate_on_new_document(script.clone()).await;
                    let _ = page.evaluate(script.clone()).await;
                }
                page.goto(&url)
                    .await
                    .map_err(|e| WebError::io(format!("failed to navigate: {}", e)))?;
                page
            } else {
                browser
                    .new_page(&url)
                    .await
                    .map_err(|e| WebError::io(format!("failed to open page: {}", e)))?
            };

            Ok(Self {
                browser: Some(browser),
                page,
                closed: false,
            })
        })
    }

    fn ensure_open(&self) -> Result<(), WebError> {
        if self.closed {
            Err(WebError::runtime("browser page is closed"))
        } else {
            Ok(())
        }
    }

    fn css_or_xpath(selector: &str) -> String {
        let s = selector.trim();
        if s.starts_with("xpath=") {
            s.to_string()
        } else if s.starts_with("//") || s.starts_with("(//") {
            format!("xpath={}", s)
        } else {
            s.to_string()
        }
    }
}

impl BrowserDriver for ChromiumDriver {
    fn goto(&mut self, url: &str) -> Result<(), WebError> {
        self.ensure_open()?;
        validate_url(url)?;
        let page = self.page.clone();
        let url = url.to_string();
        block_on(async move {
            page.goto(url)
                .await
                .map(|_| ())
                .map_err(|e| WebError::io(format!("goto failed: {}", e)))
        })
    }

    fn click(&mut self, selector: &str) -> Result<(), WebError> {
        self.ensure_open()?;
        let page = self.page.clone();
        let sel = Self::css_or_xpath(selector);
        block_on(async move {
            let el = page
                .find_element(&sel)
                .await
                .map_err(|e| WebError::value(format!("element not found '{}': {}", sel, e)))?;
            el.click()
                .await
                .map(|_| ())
                .map_err(|e| WebError::io(format!("click failed: {}", e)))
        })
    }

    fn type_text(&mut self, selector: &str, text: &str) -> Result<(), WebError> {
        self.ensure_open()?;
        let page = self.page.clone();
        let sel = Self::css_or_xpath(selector);
        let text = text.to_string();
        block_on(async move {
            let el = page
                .find_element(&sel)
                .await
                .map_err(|e| WebError::value(format!("element not found '{}': {}", sel, e)))?;
            el.type_str(&text)
                .await
                .map(|_| ())
                .map_err(|e| WebError::io(format!("type failed: {}", e)))
        })
    }

    fn fill(&mut self, selector: &str, text: &str) -> Result<(), WebError> {
        self.clear(selector)?;
        self.type_text(selector, text)
    }

    fn clear(&mut self, selector: &str) -> Result<(), WebError> {
        self.ensure_open()?;
        let page = self.page.clone();
        let sel = Self::css_or_xpath(selector);
        block_on(async move {
            page.evaluate(format!(
                r#"(function() {{
                    const el = document.querySelector({:?});
                    if (!el) throw new Error('not found');
                    el.value = '';
                    el.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    return true;
                }})()"#,
                sel
            ))
            .await
            .map(|_| ())
            .map_err(|e| WebError::io(format!("clear failed: {}", e)))
        })
    }

    fn select(&mut self, selector: &str, value: &str) -> Result<(), WebError> {
        self.ensure_open()?;
        let page = self.page.clone();
        let sel = Self::css_or_xpath(selector);
        let value = value.to_string();
        block_on(async move {
            page.evaluate(format!(
                r#"(function() {{
                    const el = document.querySelector({:?});
                    if (!el) throw new Error('not found');
                    el.value = {:?};
                    el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    return true;
                }})()"#,
                sel, value
            ))
            .await
            .map(|_| ())
            .map_err(|e| WebError::io(format!("select failed: {}", e)))
        })
    }

    fn text(&mut self, selector: Option<&str>) -> Result<String, WebError> {
        self.ensure_open()?;
        let page = self.page.clone();
        let sel = selector.map(|s| Self::css_or_xpath(s));
        block_on(async move {
            if let Some(sel) = sel {
                let el = page
                    .find_element(&sel)
                    .await
                    .map_err(|e| WebError::value(format!("element not found '{}': {}", sel, e)))?;
                el.inner_text()
                    .await
                    .map(|o| o.unwrap_or_default())
                    .map_err(|e| WebError::io(format!("text failed: {}", e)))
            } else {
                page.find_element("body")
                    .await
                    .map_err(|e| WebError::io(format!("body not found: {}", e)))?
                    .inner_text()
                    .await
                    .map(|o| o.unwrap_or_default())
                    .map_err(|e| WebError::io(format!("text failed: {}", e)))
            }
        })
    }

    fn html(&mut self, selector: Option<&str>) -> Result<String, WebError> {
        self.ensure_open()?;
        let page = self.page.clone();
        let sel = selector.map(|s| Self::css_or_xpath(s));
        block_on(async move {
            if let Some(sel) = sel {
                let el = page
                    .find_element(&sel)
                    .await
                    .map_err(|e| WebError::value(format!("element not found '{}': {}", sel, e)))?;
                el.inner_html()
                    .await
                    .map(|o| o.unwrap_or_default())
                    .map_err(|e| WebError::io(format!("html failed: {}", e)))
            } else {
                page.content()
                    .await
                    .map_err(|e| WebError::io(format!("html failed: {}", e)))
            }
        })
    }

    fn screenshot(&mut self, path: &str, selector: Option<&str>) -> Result<(), WebError> {
        self.ensure_open()?;
        let page = self.page.clone();
        let path = path.to_string();
        let sel = selector.map(|s| Self::css_or_xpath(s));
        block_on(async move {
            let bytes = if let Some(sel) = sel {
                let el = page
                    .find_element(&sel)
                    .await
                    .map_err(|e| WebError::value(format!("element not found '{}': {}", sel, e)))?;
                el.screenshot(CaptureScreenshotFormat::Png)
                    .await
                    .map_err(|e| WebError::io(format!("screenshot failed: {}", e)))?
            } else {
                page.screenshot(
                    ScreenshotParams::builder()
                        .format(CaptureScreenshotFormat::Png)
                        .build(),
                )
                .await
                .map_err(|e| WebError::io(format!("screenshot failed: {}", e)))?
            };
            std::fs::write(&path, bytes)
                .map_err(|e| WebError::io(format!("failed to write screenshot '{}': {}", path, e)))
        })
    }

    fn wait(&mut self, secs: f64) -> Result<(), WebError> {
        self.ensure_open()?;
        let dur = Duration::from_secs_f64(secs.max(0.0));
        std::thread::sleep(dur);
        Ok(())
    }

    fn wait_for(&mut self, selector: &str, timeout: Option<f64>) -> Result<(), WebError> {
        self.ensure_open()?;
        let page = self.page.clone();
        let sel = Self::css_or_xpath(selector);
        let timeout = timeout.unwrap_or(DEFAULT_WAIT_TIMEOUT_SECS);
        block_on(async move {
            let deadline = std::time::Instant::now() + Duration::from_secs_f64(timeout);
            loop {
                if page.find_element(&sel).await.is_ok() {
                    return Ok(());
                }
                if std::time::Instant::now() >= deadline {
                    return Err(WebError::io(format!(
                        "wait_for timed out waiting for '{}'",
                        sel
                    )));
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        })
    }

    fn wait_for_navigation(&mut self, timeout: Option<f64>) -> Result<(), WebError> {
        self.ensure_open()?;
        let _timeout = timeout.unwrap_or(DEFAULT_WAIT_TIMEOUT_SECS);
        // Best-effort: wait for document ready
        let page = self.page.clone();
        block_on(async move {
            page.wait_for_navigation()
                .await
                .map(|_| ())
                .map_err(|e| WebError::io(format!("wait_for_navigation failed: {}", e)))
        })
    }

    fn find(&mut self, selector: &str) -> Result<ElementHandle, WebError> {
        self.ensure_open()?;
        let page = self.page.clone();
        let sel = Self::css_or_xpath(selector);
        block_on(async move {
            page.find_element(&sel)
                .await
                .map_err(|e| WebError::value(format!("element not found '{}': {}", sel, e)))?;
            Ok(ElementHandle {
                selector: sel,
                index: 0,
            })
        })
    }

    fn find_all(&mut self, selector: &str) -> Result<Vec<ElementHandle>, WebError> {
        self.ensure_open()?;
        let page = self.page.clone();
        let sel = Self::css_or_xpath(selector);
        block_on(async move {
            let els = page
                .find_elements(&sel)
                .await
                .map_err(|e| WebError::value(format!("find_all failed '{}': {}", sel, e)))?;
            Ok(els
                .into_iter()
                .enumerate()
                .map(|(i, _)| ElementHandle {
                    selector: sel.clone(),
                    index: i,
                })
                .collect())
        })
    }

    fn cookies(&mut self) -> Result<Value, WebError> {
        self.ensure_open()?;
        let page = self.page.clone();
        block_on(async move {
            let cookies = page
                .get_cookies()
                .await
                .map_err(|e| WebError::io(format!("cookies failed: {}", e)))?;
            let mut arr = Vec::new();
            for c in cookies {
                let mut m = HashMap::new();
                m.insert("name".to_string(), Value::String(c.name));
                m.insert("value".to_string(), Value::String(c.value));
                m.insert("domain".to_string(), Value::String(c.domain));
                m.insert("path".to_string(), Value::String(c.path));
                arr.push(Value::legacy_object(m));
            }
            Ok(Value::Array(std::rc::Rc::new(std::cell::RefCell::new(arr))))
        })
    }

    fn set_cookie(&mut self, name: &str, value: &str) -> Result<(), WebError> {
        self.ensure_open()?;
        let page = self.page.clone();
        let name = name.to_string();
        let value = value.to_string();
        block_on(async move {
            let cookie = CookieParam::new(name, value);
            page.set_cookie(cookie)
                .await
                .map(|_| ())
                .map_err(|e| WebError::io(format!("set_cookie failed: {}", e)))
        })
    }

    fn delete_cookie(&mut self, name: &str) -> Result<(), WebError> {
        self.ensure_open()?;
        let page = self.page.clone();
        let name = name.to_string();
        block_on(async move {
            page.evaluate(format!(
                r#"(function() {{
                    document.cookie = {:?}+'=; Max-Age=0; path=/';
                    return true;
                }})()"#,
                name
            ))
            .await
            .map(|_| ())
            .map_err(|e| WebError::io(format!("delete_cookie failed: {}", e)))
        })
    }

    fn attr(&mut self, selector: &str, name: &str) -> Result<String, WebError> {
        self.ensure_open()?;
        let page = self.page.clone();
        let sel = Self::css_or_xpath(selector);
        let name = name.to_string();
        block_on(async move {
            let el = page
                .find_element(&sel)
                .await
                .map_err(|e| WebError::value(format!("element not found '{}': {}", sel, e)))?;
            el.attribute(&name)
                .await
                .map(|o| o.unwrap_or_default())
                .map_err(|e| WebError::io(format!("attr failed: {}", e)))
        })
    }

    fn attributes(&mut self, selector: &str) -> Result<Value, WebError> {
        self.ensure_open()?;
        let page = self.page.clone();
        let sel = Self::css_or_xpath(selector);
        block_on(async move {
            let js = format!(
                r#"(function() {{
                    const el = document.querySelector({:?});
                    if (!el) return null;
                    const out = {{}};
                    for (const a of el.attributes) out[a.name] = a.value;
                    return JSON.stringify(out);
                }})()"#,
                sel
            );
            let result = page
                .evaluate(js)
                .await
                .map_err(|e| WebError::io(format!("attributes failed: {}", e)))?;
            let s = result
                .into_value::<Option<String>>()
                .map_err(|e| WebError::io(format!("attributes parse: {}", e)))?
                .unwrap_or_else(|| "{}".to_string());
            let map: HashMap<String, String> =
                serde_json::from_str(&s).unwrap_or_default();
            let mut obj = HashMap::new();
            for (k, v) in map {
                obj.insert(k, Value::String(v));
            }
            Ok(Value::legacy_object(obj))
        })
    }

    fn close(&mut self) -> Result<(), WebError> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        let page = self.page.clone();
        block_on(async move {
            let _ = page.close().await;
        });
        // Dropping Browser terminates the Chromium process.
        self.browser.take();
        Ok(())
    }

    fn evaluate(&mut self, script: &str) -> Result<Value, WebError> {
        self.ensure_open()?;
        let page = self.page.clone();
        let script = script.to_string();
        block_on(async move {
            page.evaluate(script)
                .await
                .map(|_| Value::Bool(true))
                .map_err(|e| WebError::io(format!("evaluate failed: {}", e)))
        })
    }

    fn evaluate_on_new_document(&mut self, script: &str) -> Result<(), WebError> {
        self.ensure_open()?;
        let page = self.page.clone();
        let script = script.to_string();
        block_on(async move {
            page.evaluate_on_new_document(script)
                .await
                .map(|_| ())
                .map_err(|e| WebError::io(format!("evaluate_on_new_document failed: {}", e)))
        })
    }
}

impl Drop for ChromiumDriver {
    fn drop(&mut self) {
        if !self.closed {
            let _ = self.close();
        }
    }
}

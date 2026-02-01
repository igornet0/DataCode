//! User config for DPM (virtualenvs.in-project, etc.)

use std::path::PathBuf;

const CONFIG_DIR_NAME: &str = "datacode";
const CONFIG_FILE_NAME: &str = "dpm.toml";
const ENV_IN_PROJECT: &str = "DPM_IN_PROJECT";

/// Returns path to user config file for DPM.
/// Linux/macOS: ~/.config/datacode/dpm.toml
/// Windows: %APPDATA%\datacode\dpm.toml
pub fn config_file_path() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    let base = dirs::config_dir();

    #[cfg(not(target_os = "windows"))]
    let base = dirs::config_dir();

    base.map(|p| p.join(CONFIG_DIR_NAME).join(CONFIG_FILE_NAME))
}

/// User config (virtualenvs section).
#[derive(Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct DpmConfig {
    #[serde(default)]
    pub virtualenvs: VirtualenvsConfig,
}

#[derive(Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct VirtualenvsConfig {
    /// If true, create venv inside project (<project_root>/.dpm/)
    #[serde(rename = "in-project")]
    pub in_project: Option<bool>,
}

/// Whether to use in-project venv (from config file or DPM_IN_PROJECT env).
pub fn virtualenvs_in_project() -> bool {
    if let Ok(v) = std::env::var(ENV_IN_PROJECT) {
        if v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes") {
            return true;
        }
    }
    if let Some(path) = config_file_path() {
        if path.exists() {
            if let Ok(s) = std::fs::read_to_string(&path) {
                if let Ok(cfg) = toml::from_str::<DpmConfig>(&s) {
                    if let Some(true) = cfg.virtualenvs.in_project {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Write virtualenvs.in-project to user config file.
pub fn set_virtualenvs_in_project(in_project: bool) -> Result<(), String> {
    let path = config_file_path().ok_or("Could not determine config directory")?;
    let dir = path.parent().ok_or("Invalid config path")?;
    std::fs::create_dir_all(dir).map_err(|e| format!("Create config dir: {}", e))?;
    let mut cfg = if path.exists() {
        let s = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
        toml::from_str::<DpmConfig>(&s).unwrap_or_default()
    } else {
        DpmConfig::default()
    };
    cfg.virtualenvs.in_project = Some(in_project);
    let s = toml::to_string_pretty(&cfg).map_err(|e| e.to_string())?;
    std::fs::write(&path, s).map_err(|e| e.to_string())?;
    Ok(())
}

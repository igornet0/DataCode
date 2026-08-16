//! SQLite foreign-key check mode carried in the DCP `__config__` section.

use crate::dcp::error::DcpError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FkCheckMode {
    #[default]
    Strict,
    Warn,
    Skip,
}

impl FkCheckMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "strict" => Ok(Self::Strict),
            "warn" => Ok(Self::Warn),
            "skip" => Ok(Self::Skip),
            other => Err(format!(
                "invalid fk_check {other:?}, expected strict|warn|skip"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Warn => "warn",
            Self::Skip => "skip",
        }
    }
}

/// Parse `__config__` JSON. Missing `fk_check` → `Strict`. Invalid value → error.
pub fn parse_config_section(payload: &[u8]) -> Result<FkCheckMode, DcpError> {
    let text = std::str::from_utf8(payload)
        .map_err(|e| DcpError::Utf8Error(format!("CONFIG section: {e}")))?;
    let value: serde_json::Value = serde_json::from_str(text).map_err(|e| {
        DcpError::InvalidSection(format!("CONFIG section is not valid JSON: {e}"))
    })?;
    let Some(obj) = value.as_object() else {
        return Err(DcpError::InvalidSection(
            "CONFIG section must be a JSON object".to_string(),
        ));
    };
    match obj.get("fk_check") {
        None => Ok(FkCheckMode::Strict),
        Some(serde_json::Value::String(s)) => {
            FkCheckMode::parse(s).map_err(DcpError::InvalidSection)
        }
        Some(_) => Err(DcpError::InvalidSection(
            "fk_check must be a string (strict|warn|skip)".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_warn() {
        assert_eq!(
            parse_config_section(br#"{"fk_check":"warn"}"#).unwrap(),
            FkCheckMode::Warn
        );
    }

    #[test]
    fn parse_skip() {
        assert_eq!(
            parse_config_section(br#"{"fk_check":"skip"}"#).unwrap(),
            FkCheckMode::Skip
        );
    }

    #[test]
    fn parse_absent_is_strict() {
        assert_eq!(parse_config_section(br#"{}"#).unwrap(), FkCheckMode::Strict);
        assert_eq!(
            parse_config_section(br#"{"other":"x"}"#).unwrap(),
            FkCheckMode::Strict
        );
    }

    #[test]
    fn parse_invalid_value_errors() {
        assert!(parse_config_section(br#"{"fk_check":"nope"}"#).is_err());
        assert!(parse_config_section(br#"{"fk_check":1}"#).is_err());
        assert!(parse_config_section(b"[]").is_err());
    }
}

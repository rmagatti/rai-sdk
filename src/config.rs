use serde::{Deserialize, Serialize};

use crate::error;
use crate::retry::RetryConfig;

/// Configuration for the AI SDK client.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    /// OpenAI API key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openai_api_key: Option<String>,

    /// OpenAI API base URL (for proxies or Azure OpenAI).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openai_base_url: Option<String>,

    /// Anthropic API key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anthropic_api_key: Option<String>,

    /// Anthropic API base URL (for proxies).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anthropic_base_url: Option<String>,

    /// OpenRouter API key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openrouter_api_key: Option<String>,

    /// OpenRouter API base URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openrouter_base_url: Option<String>,

    /// Optional app URL for OpenRouter attribution headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openrouter_http_referer: Option<String>,

    /// Optional app title for OpenRouter attribution headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openrouter_title: Option<String>,

    /// Optional app categories for OpenRouter attribution headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openrouter_categories: Option<Vec<String>>,

    /// OpenRouter App URL (sent in headers).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openrouter_app_url: Option<String>,

    /// OpenRouter App Title (sent in headers).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openrouter_app_title: Option<String>,

    /// Request timeout in seconds (default: 120).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_seconds: Option<u64>,

    /// Default max tokens for generation (can be overridden per request).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_max_tokens: Option<i32>,

    /// Retry configuration for transient errors.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_config: Option<RetryConfig>,
}

impl Config {
    /// Create a new empty configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create configuration from environment variables.
    ///
    /// Reads `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `ANTHROPIC_API_KEY`,
    /// `ANTHROPIC_BASE_URL`, and `AI_TIMEOUT_SECONDS`.
    pub fn from_env() -> Self {
        let mut config = Self::new();

        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            config.openai_api_key = Some(key);
        }
        if let Ok(url) = std::env::var("OPENAI_BASE_URL") {
            config.openai_base_url = Some(url);
        }
        if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            config.anthropic_api_key = Some(key);
        }
        if let Ok(url) = std::env::var("ANTHROPIC_BASE_URL") {
            config.anthropic_base_url = Some(url);
        }
        if let Ok(key) = std::env::var("OPENROUTER_API_KEY") {
            config.openrouter_api_key = Some(key);
        }
        if let Ok(url) = std::env::var("OPENROUTER_BASE_URL") {
            config.openrouter_base_url = Some(url);
        }
        if let Ok(referer) = std::env::var("OPENROUTER_HTTP_REFERER") {
            config.openrouter_http_referer = Some(referer);
        } else if let Ok(url) = std::env::var("OPENROUTER_APP_URL") {
            config.openrouter_app_url = Some(url.clone());
            config.openrouter_http_referer = Some(url);
        }
        if let Ok(title) = std::env::var("OPENROUTER_TITLE") {
            config.openrouter_title = Some(title);
        } else if let Ok(title) = std::env::var("OPENROUTER_APP_TITLE") {
            config.openrouter_app_title = Some(title.clone());
            config.openrouter_title = Some(title);
        }
        if let Ok(categories) = std::env::var("OPENROUTER_CATEGORIES") {
            let categories = parse_openrouter_categories(&categories);
            if !categories.is_empty() {
                config.openrouter_categories = Some(categories);
            }
        }
        if let Ok(timeout) = std::env::var("AI_TIMEOUT_SECONDS") {
            if let Ok(secs) = timeout.parse() {
                config.timeout_seconds = Some(secs);
            }
        }

        let mut retry = RetryConfig::default();
        let mut retry_customized = false;

        if let Ok(value) = std::env::var("AI_MAX_RETRIES") {
            if let Ok(max_retries) = value.parse() {
                retry.max_retries = max_retries;
                retry_customized = true;
            }
        }
        if let Ok(value) = std::env::var("AI_RETRY_INITIAL_DELAY_MS") {
            if let Ok(milliseconds) = value.parse() {
                retry.initial_delay = std::time::Duration::from_millis(milliseconds);
                retry_customized = true;
            }
        }
        if let Ok(value) = std::env::var("AI_RETRY_MAX_DELAY_MS") {
            if let Ok(milliseconds) = value.parse() {
                retry.max_delay = std::time::Duration::from_millis(milliseconds);
                retry_customized = true;
            }
        }
        if let Ok(value) = std::env::var("AI_RETRY_BACKOFF_MULTIPLIER") {
            if let Ok(multiplier) = value.parse() {
                retry.backoff_multiplier = multiplier;
                retry_customized = true;
            }
        }
        if let Ok(value) = std::env::var("AI_RETRY_JITTER") {
            if let Some(jitter) = parse_bool(&value) {
                retry.jitter = jitter;
                retry_customized = true;
            }
        }

        if retry_customized {
            config.retry_config = Some(retry);
        }

        config
    }

    // ── Builder methods ──

    pub fn with_openai_key(mut self, key: impl Into<String>) -> Self {
        self.openai_api_key = Some(key.into());
        self
    }

    pub fn with_openai_base_url(mut self, url: impl Into<String>) -> Self {
        self.openai_base_url = Some(url.into());
        self
    }

    pub fn with_anthropic_key(mut self, key: impl Into<String>) -> Self {
        self.anthropic_api_key = Some(key.into());
        self
    }

    pub fn with_anthropic_base_url(mut self, url: impl Into<String>) -> Self {
        self.anthropic_base_url = Some(url.into());
        self
    }

    pub fn with_openrouter_key(mut self, key: impl Into<String>) -> Self {
        self.openrouter_api_key = Some(key.into());
        self
    }

    pub fn with_openrouter_base_url(mut self, url: impl Into<String>) -> Self {
        self.openrouter_base_url = Some(url.into());
        self
    }

    pub fn with_openrouter_http_referer(mut self, referer: impl Into<String>) -> Self {
        self.openrouter_http_referer = Some(referer.into());
        self
    }

    pub fn with_openrouter_title(mut self, title: impl Into<String>) -> Self {
        self.openrouter_title = Some(title.into());
        self
    }

    pub fn with_openrouter_categories(mut self, categories: Vec<String>) -> Self {
        self.openrouter_categories = Some(categories);
        self
    }

    pub fn with_openrouter_app_url(mut self, url: impl Into<String>) -> Self {
        let url = url.into();
        self.openrouter_app_url = Some(url.clone());
        self.openrouter_http_referer = Some(url);
        self
    }

    pub fn with_openrouter_app_title(mut self, title: impl Into<String>) -> Self {
        let title = title.into();
        self.openrouter_app_title = Some(title.clone());
        self.openrouter_title = Some(title);
        self
    }

    pub fn with_timeout(mut self, seconds: u64) -> Self {
        self.timeout_seconds = Some(seconds);
        self
    }

    pub fn with_default_max_tokens(mut self, max_tokens: i32) -> Self {
        self.default_max_tokens = Some(max_tokens);
        self
    }

    pub fn with_retry_config(mut self, retry_config: RetryConfig) -> Self {
        self.retry_config = Some(retry_config);
        self
    }

    // ── Getters with env-var fallback ──

    pub fn openai_key(&self) -> Option<String> {
        self.openai_api_key
            .clone()
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
    }

    pub fn anthropic_key(&self) -> Option<String> {
        self.anthropic_api_key
            .clone()
            .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
    }

    pub fn openrouter_key(&self) -> Option<String> {
        self.openrouter_api_key
            .clone()
            .or_else(|| std::env::var("OPENROUTER_API_KEY").ok())
    }

    pub fn openrouter_base_url(&self) -> Option<String> {
        self.openrouter_base_url
            .clone()
            .or_else(|| std::env::var("OPENROUTER_BASE_URL").ok())
    }

    pub fn openrouter_http_referer(&self) -> Option<String> {
        self.openrouter_http_referer
            .clone()
            .or_else(|| self.openrouter_app_url.clone())
            .or_else(|| std::env::var("OPENROUTER_HTTP_REFERER").ok())
            .or_else(|| std::env::var("OPENROUTER_APP_URL").ok())
    }

    pub fn openrouter_title(&self) -> Option<String> {
        self.openrouter_title
            .clone()
            .or_else(|| self.openrouter_app_title.clone())
            .or_else(|| std::env::var("OPENROUTER_TITLE").ok())
            .or_else(|| std::env::var("OPENROUTER_APP_TITLE").ok())
    }

    pub fn openrouter_categories(&self) -> Option<Vec<String>> {
        self.openrouter_categories.clone().or_else(|| {
            std::env::var("OPENROUTER_CATEGORIES")
                .ok()
                .map(|categories| parse_openrouter_categories(&categories))
                .filter(|categories| !categories.is_empty())
        })
    }

    pub fn openrouter_app_url(&self) -> Option<String> {
        self.openrouter_http_referer()
    }

    pub fn openrouter_app_title(&self) -> Option<String> {
        self.openrouter_title()
    }

    pub fn retry_config(&self) -> RetryConfig {
        self.retry_config.clone().unwrap_or_default()
    }

    pub fn timeout(&self) -> u64 {
        self.timeout_seconds.unwrap_or(120)
    }

    // ── Validation ──

    #[cfg(feature = "openai")]
    pub fn validate_openai(&self) -> error::Result<()> {
        if self.openai_key().is_none() {
            return Err(error::Error::Config(
                "OpenAI API key not configured. Set OPENAI_API_KEY env var or provide via config."
                    .into(),
            ));
        }
        Ok(())
    }

    #[cfg(feature = "anthropic")]
    pub fn validate_anthropic(&self) -> error::Result<()> {
        if self.anthropic_key().is_none() {
            return Err(error::Error::Config(
                "Anthropic API key not configured. Set ANTHROPIC_API_KEY env var or provide via config."
                    .into(),
            ));
        }
        Ok(())
    }

    #[cfg(feature = "openrouter")]
    pub fn validate_openrouter(&self) -> error::Result<()> {
        if self.openrouter_key().is_none() {
            return Err(error::Error::Config(
                "OpenRouter API key not configured. Set OPENROUTER_API_KEY env var or provide via config."
                    .into(),
            ));
        }
        Ok(())
    }
}

fn parse_openrouter_categories(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|category| !category.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn openrouter_attribution_builders_set_canonical_fields() {
        let config = Config::new()
            .with_openrouter_base_url("https://proxy.example.com/api/v1")
            .with_openrouter_http_referer("https://app.example.com")
            .with_openrouter_title("Example App")
            .with_openrouter_categories(vec!["productivity".to_string(), "agents".to_string()]);

        assert_eq!(
            config.openrouter_base_url(),
            Some("https://proxy.example.com/api/v1".to_string())
        );
        assert_eq!(
            config.openrouter_http_referer(),
            Some("https://app.example.com".to_string())
        );
        assert_eq!(config.openrouter_title(), Some("Example App".to_string()));
        assert_eq!(
            config.openrouter_categories(),
            Some(vec!["productivity".to_string(), "agents".to_string()])
        );
    }

    #[test]
    fn retry_config_defaults_when_not_set() {
        assert_eq!(Config::new().retry_config().max_retries, 3);
    }

    #[test]
    fn retry_config_builder_overrides_defaults() {
        let retry = RetryConfig::new().with_initial_delay(Duration::from_millis(250));
        let config = Config::new().with_retry_config(retry);

        assert_eq!(
            config.retry_config().initial_delay,
            Duration::from_millis(250)
        );
    }

    #[test]
    fn openrouter_category_parser_trims_empty_values() {
        assert_eq!(
            parse_openrouter_categories(" agents, , productivity "),
            vec!["agents".to_string(), "productivity".to_string()]
        );
    }
}

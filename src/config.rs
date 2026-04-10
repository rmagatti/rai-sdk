use serde::{Deserialize, Serialize};

use crate::error;

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
        if let Ok(url) = std::env::var("OPENROUTER_APP_URL") {
            config.openrouter_app_url = Some(url);
        }
        if let Ok(title) = std::env::var("OPENROUTER_APP_TITLE") {
            config.openrouter_app_title = Some(title);
        }
        if let Ok(timeout) = std::env::var("AI_TIMEOUT_SECONDS") {
            if let Ok(secs) = timeout.parse() {
                config.timeout_seconds = Some(secs);
            }
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

    pub fn with_openrouter_app_url(mut self, url: impl Into<String>) -> Self {
        self.openrouter_app_url = Some(url.into());
        self
    }

    pub fn with_openrouter_app_title(mut self, title: impl Into<String>) -> Self {
        self.openrouter_app_title = Some(title.into());
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

    pub fn openrouter_app_url(&self) -> Option<String> {
        self.openrouter_app_url
            .clone()
            .or_else(|| std::env::var("OPENROUTER_APP_URL").ok())
    }

    pub fn openrouter_app_title(&self) -> Option<String> {
        self.openrouter_app_title
            .clone()
            .or_else(|| std::env::var("OPENROUTER_APP_TITLE").ok())
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

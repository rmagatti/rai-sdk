//! Client-level configuration: credentials, endpoints, timeouts, retries.
//!
//! [`Config`] holds everything that is scoped to a client rather than to a
//! single request. It can be built explicitly, loaded from the environment
//! with [`Config::from_env`], or mixed: the getters fall back to environment
//! variables when a field was not set programmatically, so an explicit value
//! always wins over the environment.
//!
//! # Environment variables
//!
//! * `OPENAI_API_KEY`, `OPENAI_BASE_URL`
//! * `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`
//! * `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`
//! * `OPENROUTER_HTTP_REFERER` (or the legacy `OPENROUTER_APP_URL`),
//!   `OPENROUTER_TITLE` (or the legacy `OPENROUTER_APP_TITLE`),
//!   `OPENROUTER_CATEGORIES` (comma-separated)
//! * `AI_TIMEOUT_SECONDS`
//! * `AI_MAX_RETRIES`, `AI_RETRY_INITIAL_DELAY_MS`, `AI_RETRY_MAX_DELAY_MS`,
//!   `AI_RETRY_BACKOFF_MULTIPLIER`, `AI_RETRY_JITTER`
//!
//! The OpenAI-compatible endpoint settings are the one exception: they have no
//! environment variables and are always per client. See
//! [`Config::openai_compatible_base_url`].
//!
//! # Examples
//!
//! ```no_run
//! use std::time::Duration;
//! use rai_sdk::{Config, RetryConfig};
//!
//! // Start from the environment, then override selected values.
//! let config = Config::from_env()
//!     .with_timeout(30)
//!     .with_default_max_tokens(2048)
//!     .with_retry_config(RetryConfig::new().with_initial_delay(Duration::from_millis(250)));
//!
//! assert_eq!(config.timeout(), 30);
//! ```

use serde::{Deserialize, Serialize};

#[cfg(any(feature = "openai", feature = "anthropic", feature = "openrouter"))]
use crate::error;
use crate::retry::RetryConfig;

/// The base URL Ollama serves its OpenAI-compatible API on by default.
///
/// Used by [`Config::with_ollama`] and
/// [`ClientBuilder::ollama`](crate::ClientBuilder::ollama).
pub const OLLAMA_BASE_URL: &str = "http://localhost:11434/v1";

/// What an OpenAI-compatible endpoint supports beyond plain chat completions.
///
/// Endpoints that speak OpenAI's wire format vary in what they implement: a
/// 3B model behind Ollama may have no tool support at all, and a runtime may
/// accept `response_format` and then ignore it. This type is how a caller
/// states what its endpoint can do. Nothing is probed — auto-detection would
/// mean an extra round trip on every client build and still be wrong for the
/// per-model cases.
///
/// The default assumes full compatibility, so a capable endpoint needs no
/// configuration. Turn a capability off to convert what would be an opaque
/// HTTP 400 partway through into an immediate, typed
/// [`Error::CapabilityUnsupported`](crate::Error::CapabilityUnsupported).
///
/// # Examples
///
/// ```
/// use rai_sdk::EndpointCapabilities;
///
/// // A small local model that cannot call tools but does honor JSON schemas.
/// let capabilities = EndpointCapabilities::default().with_tool_calling(false);
///
/// assert!(!capabilities.tool_calling);
/// assert!(capabilities.structured_output);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EndpointCapabilities {
    /// Whether the endpoint accepts `tools` and can return tool calls.
    pub tool_calling: bool,

    /// Whether the endpoint honors `response_format` — JSON mode or a JSON
    /// Schema.
    pub structured_output: bool,
}

impl Default for EndpointCapabilities {
    /// Assume the endpoint implements everything, which is what "OpenAI
    /// compatible" claims.
    fn default() -> Self {
        Self::all()
    }
}

impl EndpointCapabilities {
    /// Every capability is supported.
    pub fn all() -> Self {
        Self {
            tool_calling: true,
            structured_output: true,
        }
    }

    /// Chat completions only: no tool calling, no structured output.
    pub fn text_only() -> Self {
        Self {
            tool_calling: false,
            structured_output: false,
        }
    }

    /// Declare whether the endpoint supports tool calling.
    pub fn with_tool_calling(mut self, supported: bool) -> Self {
        self.tool_calling = supported;
        self
    }

    /// Declare whether the endpoint supports structured output.
    pub fn with_structured_output(mut self, supported: bool) -> Self {
        self.structured_output = supported;
        self
    }
}

/// Configuration for the AI SDK client.
///
/// Every field is optional. Unset credentials simply mean the corresponding
/// provider is unavailable rather than an error at construction time; the
/// failure surfaces as [`Error::ProviderNotConfigured`](crate::Error::ProviderNotConfigured)
/// when that provider is actually used.
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

    /// Base URL of an OpenAI-compatible endpoint, such as
    /// `http://localhost:11434/v1`.
    ///
    /// Setting this is what makes [`ProviderKind::OpenAICompatible`] available
    /// on a client; there is no default endpoint and no environment variable,
    /// because a process routinely talks to several at once. See
    /// [`Config::openai_compatible_base_url`].
    ///
    /// [`ProviderKind::OpenAICompatible`]: crate::ProviderKind::OpenAICompatible
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openai_compatible_base_url: Option<String>,

    /// Bearer token for the OpenAI-compatible endpoint.
    ///
    /// Optional: endpoints that need no credential — the common case for a
    /// local runtime — simply leave this unset, and no `Authorization` header
    /// is sent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openai_compatible_api_key: Option<String>,

    /// What the OpenAI-compatible endpoint supports beyond plain chat.
    ///
    /// Declared by the caller, never probed. Defaults to
    /// [`EndpointCapabilities::default`], which assumes full compatibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openai_compatible_capabilities: Option<EndpointCapabilities>,

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
    /// Reads every variable listed in the [module docs](self). Missing or
    /// unparseable values are ignored, leaving the corresponding field unset
    /// (and therefore at its default). The retry configuration is only
    /// populated when at least one `AI_RETRY_*`/`AI_MAX_RETRIES` variable was
    /// recognized.
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

    /// Set the OpenAI API key.
    pub fn with_openai_key(mut self, key: impl Into<String>) -> Self {
        self.openai_api_key = Some(key.into());
        self
    }

    /// Set the OpenAI base URL, for proxies or Azure OpenAI deployments.
    pub fn with_openai_base_url(mut self, url: impl Into<String>) -> Self {
        self.openai_base_url = Some(url.into());
        self
    }

    /// Set the Anthropic API key.
    pub fn with_anthropic_key(mut self, key: impl Into<String>) -> Self {
        self.anthropic_api_key = Some(key.into());
        self
    }

    /// Set the Anthropic base URL, for proxies.
    pub fn with_anthropic_base_url(mut self, url: impl Into<String>) -> Self {
        self.anthropic_base_url = Some(url.into());
        self
    }

    /// Set the OpenRouter API key.
    pub fn with_openrouter_key(mut self, key: impl Into<String>) -> Self {
        self.openrouter_api_key = Some(key.into());
        self
    }

    /// Set the OpenRouter base URL, for proxies.
    pub fn with_openrouter_base_url(mut self, url: impl Into<String>) -> Self {
        self.openrouter_base_url = Some(url.into());
        self
    }

    /// Point this client at an OpenAI-compatible endpoint.
    ///
    /// The URL is the API root that serves `POST /chat/completions`, so it
    /// usually ends in `/v1`.
    pub fn with_openai_compatible_base_url(mut self, url: impl Into<String>) -> Self {
        self.openai_compatible_base_url = Some(url.into());
        self
    }

    /// Set the bearer token for the OpenAI-compatible endpoint.
    ///
    /// Leave it unset for endpoints that need no credential; no
    /// `Authorization` header is sent then.
    pub fn with_openai_compatible_key(mut self, key: impl Into<String>) -> Self {
        self.openai_compatible_api_key = Some(key.into());
        self
    }

    /// Declare what the OpenAI-compatible endpoint supports.
    pub fn with_openai_compatible_capabilities(
        mut self,
        capabilities: EndpointCapabilities,
    ) -> Self {
        self.openai_compatible_capabilities = Some(capabilities);
        self
    }

    /// Point this client at a local Ollama server ([`OLLAMA_BASE_URL`]).
    ///
    /// Shorthand for [`Config::with_openai_compatible_base_url`] with Ollama's
    /// default address; pass the URL explicitly for any other host or port.
    pub fn with_ollama(self) -> Self {
        self.with_openai_compatible_base_url(OLLAMA_BASE_URL)
    }

    /// Set the OpenRouter `HTTP-Referer` attribution header.
    pub fn with_openrouter_http_referer(mut self, referer: impl Into<String>) -> Self {
        self.openrouter_http_referer = Some(referer.into());
        self
    }

    /// Set the OpenRouter app title attribution header.
    pub fn with_openrouter_title(mut self, title: impl Into<String>) -> Self {
        self.openrouter_title = Some(title.into());
        self
    }

    /// Set the OpenRouter app categories attribution header.
    pub fn with_openrouter_categories(mut self, categories: Vec<String>) -> Self {
        self.openrouter_categories = Some(categories);
        self
    }

    /// Set the legacy OpenRouter app URL, which also sets the canonical
    /// `HTTP-Referer` value.
    pub fn with_openrouter_app_url(mut self, url: impl Into<String>) -> Self {
        let url = url.into();
        self.openrouter_app_url = Some(url.clone());
        self.openrouter_http_referer = Some(url);
        self
    }

    /// Set the legacy OpenRouter app title, which also sets the canonical
    /// title value.
    pub fn with_openrouter_app_title(mut self, title: impl Into<String>) -> Self {
        let title = title.into();
        self.openrouter_app_title = Some(title.clone());
        self.openrouter_title = Some(title);
        self
    }

    /// Set the HTTP request timeout, in seconds.
    pub fn with_timeout(mut self, seconds: u64) -> Self {
        self.timeout_seconds = Some(seconds);
        self
    }

    /// Set the default `max_tokens` used when a request does not specify one.
    pub fn with_default_max_tokens(mut self, max_tokens: i32) -> Self {
        self.default_max_tokens = Some(max_tokens);
        self
    }

    /// Set the retry policy applied to transient errors.
    pub fn with_retry_config(mut self, retry_config: RetryConfig) -> Self {
        self.retry_config = Some(retry_config);
        self
    }

    // ── Getters with env-var fallback ──

    /// The OpenAI API key, falling back to `OPENAI_API_KEY`.
    pub fn openai_key(&self) -> Option<String> {
        self.openai_api_key
            .clone()
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
    }

    /// The Anthropic API key, falling back to `ANTHROPIC_API_KEY`.
    pub fn anthropic_key(&self) -> Option<String> {
        self.anthropic_api_key
            .clone()
            .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
    }

    /// The OpenRouter API key, falling back to `OPENROUTER_API_KEY`.
    pub fn openrouter_key(&self) -> Option<String> {
        self.openrouter_api_key
            .clone()
            .or_else(|| std::env::var("OPENROUTER_API_KEY").ok())
    }

    /// The OpenRouter base URL, falling back to `OPENROUTER_BASE_URL`.
    ///
    /// `None` means the provider uses its built-in default endpoint.
    pub fn openrouter_base_url(&self) -> Option<String> {
        self.openrouter_base_url
            .clone()
            .or_else(|| std::env::var("OPENROUTER_BASE_URL").ok())
    }

    /// The OpenAI-compatible endpoint's base URL, or `None` when this client
    /// has none configured.
    ///
    /// Unlike every other getter here this one has **no environment-variable
    /// fallback**, and that is deliberate. The other providers each name one
    /// well-known service, so a process-wide `*_BASE_URL` is a sensible
    /// override. "OpenAI-compatible" names no service at all: a single process
    /// may talk to a local Ollama, a shared vLLM deployment, and a staging
    /// gateway at the same time, each with its own credentials and
    /// capabilities. That is per-client configuration, so it is set per client.
    ///
    /// `OPENAI_BASE_URL` keeps its existing meaning and still applies only to
    /// the real OpenAI provider.
    pub fn openai_compatible_base_url(&self) -> Option<String> {
        self.openai_compatible_base_url.clone()
    }

    /// The OpenAI-compatible endpoint's bearer token, if one was set.
    ///
    /// No environment-variable fallback, for the reasons given on
    /// [`Config::openai_compatible_base_url`].
    pub fn openai_compatible_key(&self) -> Option<String> {
        self.openai_compatible_api_key.clone()
    }

    /// What the OpenAI-compatible endpoint was declared to support, defaulting
    /// to [`EndpointCapabilities::default`].
    pub fn openai_compatible_capabilities(&self) -> EndpointCapabilities {
        self.openai_compatible_capabilities.unwrap_or_default()
    }

    /// The OpenRouter `HTTP-Referer` value.
    ///
    /// Resolution order: the explicit referer, the legacy app URL,
    /// `OPENROUTER_HTTP_REFERER`, then `OPENROUTER_APP_URL`.
    pub fn openrouter_http_referer(&self) -> Option<String> {
        self.openrouter_http_referer
            .clone()
            .or_else(|| self.openrouter_app_url.clone())
            .or_else(|| std::env::var("OPENROUTER_HTTP_REFERER").ok())
            .or_else(|| std::env::var("OPENROUTER_APP_URL").ok())
    }

    /// The OpenRouter app title used for attribution headers.
    ///
    /// Resolution order: the explicit title, the legacy app title,
    /// `OPENROUTER_TITLE`, then `OPENROUTER_APP_TITLE`.
    pub fn openrouter_title(&self) -> Option<String> {
        self.openrouter_title
            .clone()
            .or_else(|| self.openrouter_app_title.clone())
            .or_else(|| std::env::var("OPENROUTER_TITLE").ok())
            .or_else(|| std::env::var("OPENROUTER_APP_TITLE").ok())
    }

    /// The OpenRouter app categories, falling back to the comma-separated
    /// `OPENROUTER_CATEGORIES` variable. Empty lists are treated as unset.
    pub fn openrouter_categories(&self) -> Option<Vec<String>> {
        self.openrouter_categories.clone().or_else(|| {
            std::env::var("OPENROUTER_CATEGORIES")
                .ok()
                .map(|categories| parse_openrouter_categories(&categories))
                .filter(|categories| !categories.is_empty())
        })
    }

    /// Deprecated alias for [`Config::openrouter_http_referer`], kept for
    /// callers written against the older attribution field names.
    pub fn openrouter_app_url(&self) -> Option<String> {
        self.openrouter_http_referer()
    }

    /// Deprecated alias for [`Config::openrouter_title`], kept for callers
    /// written against the older attribution field names.
    pub fn openrouter_app_title(&self) -> Option<String> {
        self.openrouter_title()
    }

    /// The effective retry policy, or [`RetryConfig::default`] when unset.
    pub fn retry_config(&self) -> RetryConfig {
        self.retry_config.clone().unwrap_or_default()
    }

    /// The effective HTTP timeout in seconds (defaults to 120).
    pub fn timeout(&self) -> u64 {
        self.timeout_seconds.unwrap_or(120)
    }

    // ── Validation ──

    /// Check that OpenAI is usable.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`](crate::Error::Config) if no OpenAI API key is
    /// set programmatically or in `OPENAI_API_KEY`.
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

    /// Check that Anthropic is usable.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`](crate::Error::Config) if no Anthropic API key
    /// is set programmatically or in `ANTHROPIC_API_KEY`.
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

    /// Check that OpenRouter is usable.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`](crate::Error::Config) if no OpenRouter API key
    /// is set programmatically or in `OPENROUTER_API_KEY`.
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

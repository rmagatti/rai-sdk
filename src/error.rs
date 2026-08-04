//! Error types shared by every provider.
//!
//! [`enum@Error`] is the single error type returned across the SDK, and [`Result`]
//! is the matching alias. Errors carry the [`ProviderKind`] they originated
//! from where that is meaningful, and expose classification helpers
//! ([`Error::is_retryable`], [`Error::is_auth_error`], [`Error::kind_str`]) so
//! callers can branch on categories instead of matching every variant.
//!
//! # Examples
//!
//! ```no_run
//! use rai_sdk::{Error, ProviderKind};
//!
//! fn describe(error: &Error) -> String {
//!     if error.is_auth_error() {
//!         return "check your API key".to_string();
//!     }
//!     if error.is_retryable() {
//!         return format!("transient {} failure", error.kind_str());
//!     }
//!     match error.provider() {
//!         Some(ProviderKind::OpenAI) => "OpenAI rejected the request".to_string(),
//!         Some(provider) => format!("{provider} rejected the request"),
//!         None => error.to_string(),
//!     }
//! }
//! ```

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Identifies which AI provider was used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    /// OpenAI's own API (`https://api.openai.com`).
    OpenAI,
    /// Anthropic's Messages API (`https://api.anthropic.com`).
    Anthropic,
    /// OpenRouter's aggregating API (`https://openrouter.ai`).
    OpenRouter,
}

impl std::fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderKind::OpenAI => write!(f, "openai"),
            ProviderKind::Anthropic => write!(f, "anthropic"),
            ProviderKind::OpenRouter => write!(f, "openrouter"),
        }
    }
}

/// Diagnostic info about a tool argument validation failure.
///
/// One issue is produced per JSON Schema violation found in the arguments a
/// model supplied for a tool call. Issues are serialized back to the model as
/// part of the tool error message so it can correct itself and retry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolArgumentIssue {
    /// JSON pointer to the offending value in the arguments (`$` for the root).
    pub path: String,
    /// JSON pointer to the schema keyword that rejected the value.
    pub schema_path: String,
    /// Human-readable description of the violation.
    pub message: String,
}

/// Errors that can occur when using the AI SDK.
///
/// Every fallible operation in this crate returns this type through
/// [`Result`]. Rather than matching each variant, prefer the classification
/// helpers when you only care about the category of failure.
///
/// # Examples
///
/// ```no_run
/// use rai_sdk::{Error, ProviderKind};
///
/// let error = Error::RateLimit {
///     provider: ProviderKind::OpenAI,
///     message: "slow down".to_string(),
/// };
///
/// assert!(error.is_rate_limit());
/// assert!(error.is_retryable());
/// assert_eq!(error.kind_str(), "rate_limit");
/// assert_eq!(error.provider(), Some(ProviderKind::OpenAI));
/// ```
#[derive(Error, Debug)]
pub enum Error {
    /// Authentication failed (invalid API key, expired token, etc.)
    #[error("authentication error for {provider}: {message}")]
    Auth {
        /// Provider that rejected the credentials.
        provider: ProviderKind,
        /// Message reported by the provider.
        message: String,
    },

    /// The API request failed.
    ///
    /// Used for provider errors that do not map to a more specific variant,
    /// including malformed provider responses.
    #[error("API request failed for {provider}: {message}")]
    Request {
        /// Provider that produced the failure.
        provider: ProviderKind,
        /// Message reported by the provider, or a description of what was wrong
        /// with its response.
        message: String,
    },

    /// Rate limit exceeded.
    ///
    /// Retryable: see [`Error::is_retryable`].
    #[error("rate limit exceeded for {provider}: {message}")]
    RateLimit {
        /// Provider that throttled the request.
        provider: ProviderKind,
        /// Message reported by the provider.
        message: String,
    },

    /// Invalid request (bad parameters, etc.)
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// The requested model is not available or not supported.
    #[error("model not available: {model} for provider {provider}")]
    ModelNotAvailable {
        /// Provider the model was requested from.
        provider: ProviderKind,
        /// Model identifier that was rejected.
        model: String,
    },

    /// Provider not configured (missing API key, etc.)
    #[error("provider {0} is not configured")]
    ProviderNotConfigured(ProviderKind),

    /// Provider feature not enabled.
    #[error("provider {0} feature is not enabled — enable the '{0}' feature in Cargo.toml")]
    ProviderNotEnabled(ProviderKind),

    /// Content was filtered/blocked by the provider.
    #[error("content filtered by {provider}: {reason}")]
    ContentFiltered {
        /// Provider that filtered the content.
        provider: ProviderKind,
        /// Reason the provider gave for filtering.
        reason: String,
    },

    /// Configuration error.
    #[error("configuration error: {0}")]
    Config(String),

    /// Serialization/deserialization error.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// HTTP client error.
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// Stream error.
    #[error("stream error: {0}")]
    Stream(String),

    /// Timeout.
    ///
    /// Retryable: see [`Error::is_retryable`].
    #[error("request timed out for {provider}")]
    Timeout {
        /// Provider whose request timed out.
        provider: ProviderKind,
    },

    /// Tool calling is not supported for the selected provider.
    #[error("tool calling is not supported for provider {provider}")]
    ToolProviderUnsupported {
        /// Provider that does not support tool calling.
        provider: ProviderKind,
    },

    /// Tool arguments failed validation.
    ///
    /// Surfaced to the model as a tool error message rather than aborting the
    /// tool loop, so it can retry with corrected arguments.
    #[error("invalid arguments for tool '{name}': {message}")]
    ToolArguments {
        /// Name of the tool whose arguments were rejected.
        name: String,
        /// Summary of the validation failures.
        message: String,
        /// Per-violation diagnostics, sorted and deduplicated.
        issues: Vec<ToolArgumentIssue>,
    },

    /// A requested tool is not registered.
    #[error("tool not found: {name}")]
    ToolNotFound {
        /// Tool name the model tried to call.
        name: String,
    },

    /// Tool execution exceeded the configured loop limit.
    ///
    /// The limit comes from
    /// [`GenerationConfig::with_max_tool_rounds`](crate::GenerationConfig::with_max_tool_rounds).
    #[error("tool execution exceeded the maximum number of rounds ({max_rounds})")]
    ToolLoopLimitExceeded {
        /// Round limit that was hit.
        max_rounds: usize,
    },

    /// Structured output could not be validated against the requested type.
    #[error("structured output validation failed for {provider} model {model}: {message}")]
    StructuredOutput {
        /// Provider that produced the output.
        provider: ProviderKind,
        /// Model that produced the output.
        model: String,
        /// Why the output was rejected (empty, invalid JSON, schema violation,
        /// or deserialization failure).
        message: String,
    },
}

impl Error {
    /// Returns `true` if this error is likely transient and the request can be retried.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Error::RateLimit { .. } | Error::Timeout { .. } | Error::Http(_)
        )
    }

    /// Returns `true` if this is an authentication error.
    pub fn is_auth_error(&self) -> bool {
        matches!(self, Error::Auth { .. })
    }

    /// Returns `true` if this is a rate limit error.
    pub fn is_rate_limit(&self) -> bool {
        matches!(self, Error::RateLimit { .. })
    }

    /// Short error category string for use as a metrics or logging label.
    pub fn kind_str(&self) -> &'static str {
        match self {
            Error::Auth { .. } => "auth",
            Error::Request { .. } => "request",
            Error::RateLimit { .. } => "rate_limit",
            Error::InvalidRequest(_) => "invalid_request",
            Error::ModelNotAvailable { .. } => "model_not_available",
            Error::ProviderNotConfigured(_) => "provider_not_configured",
            Error::ProviderNotEnabled(_) => "provider_not_enabled",
            Error::ContentFiltered { .. } => "content_filtered",
            Error::Config(_) => "config",
            Error::Serialization(_) => "serialization",
            Error::Http(_) => "http",
            Error::Stream(_) => "stream",
            Error::Timeout { .. } => "timeout",
            Error::ToolProviderUnsupported { .. } => "tool_provider_unsupported",
            Error::ToolArguments { .. } => "tool_arguments",
            Error::ToolNotFound { .. } => "tool_not_found",
            Error::ToolLoopLimitExceeded { .. } => "tool_loop_limit_exceeded",
            Error::StructuredOutput { .. } => "structured_output",
        }
    }

    /// Get the provider associated with this error, if any.
    pub fn provider(&self) -> Option<ProviderKind> {
        match self {
            Error::Auth { provider, .. }
            | Error::Request { provider, .. }
            | Error::RateLimit { provider, .. }
            | Error::ModelNotAvailable { provider, .. }
            | Error::ProviderNotConfigured(provider)
            | Error::ProviderNotEnabled(provider)
            | Error::ContentFiltered { provider, .. }
            | Error::Timeout { provider }
            | Error::ToolProviderUnsupported { provider }
            | Error::StructuredOutput { provider, .. } => Some(*provider),
            _ => None,
        }
    }
}

/// Result type alias for AI SDK operations.
pub type Result<T> = std::result::Result<T, Error>;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Identifies which AI provider was used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    OpenAI,
    Anthropic,
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolArgumentIssue {
    pub path: String,
    pub schema_path: String,
    pub message: String,
}

/// Errors that can occur when using the AI SDK.
#[derive(Error, Debug)]
pub enum Error {
    /// Authentication failed (invalid API key, expired token, etc.)
    #[error("authentication error for {provider}: {message}")]
    Auth {
        provider: ProviderKind,
        message: String,
    },

    /// The API request failed.
    #[error("API request failed for {provider}: {message}")]
    Request {
        provider: ProviderKind,
        message: String,
    },

    /// Rate limit exceeded.
    #[error("rate limit exceeded for {provider}: {message}")]
    RateLimit {
        provider: ProviderKind,
        message: String,
    },

    /// Invalid request (bad parameters, etc.)
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// The requested model is not available or not supported.
    #[error("model not available: {model} for provider {provider}")]
    ModelNotAvailable {
        provider: ProviderKind,
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
        provider: ProviderKind,
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
    #[error("request timed out for {provider}")]
    Timeout { provider: ProviderKind },

    /// Tool calling is not supported for the selected provider.
    #[error("tool calling is not supported for provider {provider}")]
    ToolProviderUnsupported { provider: ProviderKind },

    /// Tool arguments failed validation.
    #[error("invalid arguments for tool '{name}': {message}")]
    ToolArguments {
        name: String,
        message: String,
        issues: Vec<ToolArgumentIssue>,
    },

    /// A requested tool is not registered.
    #[error("tool not found: {name}")]
    ToolNotFound { name: String },

    /// Tool execution exceeded the configured loop limit.
    #[error("tool execution exceeded the maximum number of rounds ({max_rounds})")]
    ToolLoopLimitExceeded { max_rounds: usize },

    /// Structured output could not be validated against the requested type.
    #[error("structured output validation failed for {provider} model {model}: {message}")]
    StructuredOutput {
        provider: ProviderKind,
        model: String,
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
